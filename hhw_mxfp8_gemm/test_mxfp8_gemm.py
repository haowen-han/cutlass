"""
Test script for MXFP8 GEMM CUDA extension.

Generates MXFP8 (e4m3) data and e8m0 scale factors, applies F8_128x4 reordering
to the scale tensors (same layout as cuDNN/cutlass), then calls CUTLASS GEMM
and compares against a PyTorch bf16 reference.
"""

import math
import torch
import hhw_mxfp8_gemm as _C


# ---------------------------------------------------------------------------
# FP8_E8M0 scale helpers
# ---------------------------------------------------------------------------
def e8m0_byte_to_float(byte_val):
    """Convert an FP8_E8M0 byte to its float value: 2^(byte - 127)."""
    return 2.0 ** (byte_val - 127)


def e8m0_bytes_to_floats(byte_tensor):
    """Convert a uint8 tensor of FP8_E8M0 byte patterns to float32 values."""
    return 2.0 ** (byte_tensor.to(torch.float32) - 127.0)


# ---------------------------------------------------------------------------
# F8_128x4 reordering (same as cuDNN / CUTLASS block-scaled layout)
# ---------------------------------------------------------------------------
def _reorder_f8_128x4(tensor, s_m, s_k, *, s_n=None):
    """Apply F8_128x4 reordering to a scale tensor for CUTLASS/cuDNN.

    For scale_a (s_n=None):
        tensor shape is (b, s_m, s_k)
        F8_128x4 reordering on (s_m, s_k) inner 2D.

    For scale_b (s_n given):
        tensor shape is (b, s_k, s_n).
        Transposed to (b, s_n, s_k), F8_128x4 applied to (s_n, s_k) inner 2D.

    The F8_128x4 reordering: within each 128-row M-block, the row
    dimension is decomposed as (4, 32) and swapped with the k_block
    dimension to produce the physical order:
        b -> M-block -> k_block -> row%32 -> row//32 -> k_inner

    Args:
        tensor: logical-order scale tensor (uint8).
        s_m: block_scale_dim_m (128-aligned).
        s_k: block_scale_dim_k (multiple of 4).
        s_n: block_scale_dim_n for scale_b. Omit for scale_a.

    Returns:
        Contiguous uint8 tensor with F8_128x4 physical layout, flattened to 1D.
    """
    if s_n is not None:
        # scale_b path: transpose (b, s_k, s_n) -> (b, s_n, s_k)
        x = tensor.transpose(1, 2).contiguous()
        s_128 = s_n
    else:
        # scale_a path
        x = tensor
        s_128 = s_m

    b = x.shape[0]
    num_m_blocks = s_128 // 128
    n_k_blocks = s_k // 4

    # Decompose: s_128 -> (num_m_blocks, 4, 32),  s_k -> (n_k_blocks, 4)
    # Then permute to physical order: b, M-block, k_block, row%32, row//32, k_inner
    x = x.view(b, num_m_blocks, 4, 32, n_k_blocks, 4)
    x = x.permute(0, 1, 4, 3, 2, 5)
    return x.contiguous().view(-1)


def _python_mxfp8_quantize(in_bf16):
    """Reference MXFP8 quantization in pure Python.

    Args:
        in_bf16: [M, K] bfloat16 tensor

    Returns:
        out_fp8: [M, K] float8_e4m3fn tensor
        scale_logical: [M, K//32] uint8 tensor of e8m0 scale bytes (logical order)
    """
    M, K = in_bf16.shape
    block_size = 32
    num_blocks_k = K // block_size
    in_f32 = in_bf16.to(torch.float32)

    in_blocked = in_f32.view(M, num_blocks_k, block_size)
    block_amax = in_blocked.abs().amax(dim=-1)  # (M, num_blocks_k)

    # Compute e8m0 scale: sf_val = amax / 448, round up to power of 2
    sf_val = block_amax / 448.0
    # Convert to e8m0 byte: ceil(log2(sf_val)) + 127
    scale_bytes = torch.zeros_like(block_amax, dtype=torch.uint8)
    nonzero = block_amax > 0
    if nonzero.any():
        log2sf = torch.log2(sf_val[nonzero])
        scale_bytes[nonzero] = (torch.ceil(log2sf) + 127).to(torch.uint8)

    # Compute inv_scale (2^x > 0 for any finite x, so scale_float is never zero)
    scale_float = e8m0_bytes_to_floats(scale_bytes)  # (M, num_blocks_k)
    inv_scale = 1.0 / scale_float

    # Quantize: fp8_val = round(input * inv_scale), saturate to [-448, 448]
    in_scaled = in_blocked * inv_scale.unsqueeze(-1)  # (M, num_blocks_k, block_size)
    in_scaled = in_scaled.clamp(-448.0, 448.0)

    # Convert to e4m3 via PyTorch cast
    out_fp8 = in_scaled.reshape(M, K).to(torch.float8_e4m3fn)

    return out_fp8, scale_bytes


def test_mxfp8_vs_bf16():
    """Quantize bf16 A/B -> MXFP8 GEMM, compare against dequantized reference."""
    device = torch.device("cuda")

    M, N, K = 256, 512, 1024
    block_size = 32

    print(f"\nTesting MXFP8 vs dequantized reference: M={M}, N={N}, K={K}")

    # Generate random bf16 inputs (uniform in [-1, 1])
    A_bf16 = torch.rand(M, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0
    B_bf16 = torch.rand(N, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0

    # ---- Quantize using Python (ensures FP8 data & scale consistency) ----
    A_fp8, scale_a_logical = _python_mxfp8_quantize(A_bf16)
    B_fp8, scale_b_logical = _python_mxfp8_quantize(B_bf16)

    # B_fp8 is (N,K) row-major in PyTorch, which matches CUTLASS ColumnMajor (N,K)
    # with stride (K,1) — K is the contiguous dimension. No transpose needed.

    # ---- Build correct scale layouts via F8_128x4 reorder ----
    num_blocks_k = K // block_size
    s_m = math.ceil(M / 128) * 128
    s_n = math.ceil(N / 128) * 128
    s_k = math.ceil(num_blocks_k / 4) * 4

    # SFA: from A's logical scales
    scale_a_padded = torch.zeros(1, s_m, s_k, dtype=torch.uint8, device=device)
    scale_a_padded[0, :M, :num_blocks_k] = scale_a_logical
    SFA = _reorder_f8_128x4(scale_a_padded, s_m, s_k).view(torch.float8_e8m0fnu)

    # SFB: from B's logical scales (transpose for column-major B)
    scale_b_padded = torch.zeros(1, s_k, s_n, dtype=torch.uint8, device=device)
    scale_b_padded[0, :num_blocks_k, :N] = scale_b_logical.T
    SFB = _reorder_f8_128x4(scale_b_padded, s_m=s_n, s_k=s_k, s_n=s_n).view(torch.float8_e8m0fnu)

    # MXFP8 GEMM with correct SFA/SFB layouts
    D_mxfp8 = _C.mxfp8_gemm(A_fp8, SFA, B_fp8, SFB)

    # Reference: original BF16 matmul
    D_bf16 = A_bf16 @ B_bf16.T


    print(D_mxfp8)
    print(D_bf16)



def test_mxfp8_cuda_quantize_vs_bf16():
    """Use CUDA mxfp8_quantize for quantization, compare MXFP8 GEMM vs BF16 reference."""
    device = torch.device("cuda")

    M, N, K = 256, 512, 1024

    print(f"\nTesting MXFP8 (CUDA quantize) vs BF16: M={M}, N={N}, K={K}")

    # Generate random bf16 inputs (uniform in [-1, 1])
    A_bf16 = torch.rand(M, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0
    B_bf16 = torch.rand(N, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0

    # Quantize via CUDA kernel (produces FP8 data + swizzled scale in one call)
    A_fp8, SFA = _C.mxfp8_quantize(A_bf16)
    B_fp8, SFB = _C.mxfp8_quantize(B_bf16)

    # B_fp8 is (N,K) row-major, matches CUTLASS ColumnMajor (N,K) with stride (K,1)

    # MXFP8 GEMM
    D_mxfp8 = _C.mxfp8_gemm(A_fp8, SFA, B_fp8, SFB)

    # BF16 reference
    D_bf16 = A_bf16 @ B_bf16.T

    print(D_mxfp8)
    print(D_bf16)

def test_python_quant_and_cuda_quant():
    """Compare CUDA mxfp8_quantize vs Python _python_mxfp8_quantize: verify both produce identical GEMM results."""
    device = torch.device("cuda")

    M, N, K = 256, 512, 1024

    A_bf16 = torch.rand(M, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0
    B_bf16 = torch.rand(N, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0

    A_fp8_cuda, SFA_cuda = _C.mxfp8_quantize(A_bf16)
    B_fp8_cuda, SFB_cuda = _C.mxfp8_quantize(B_bf16)
    D_mxfp8_cuda = _C.mxfp8_gemm(A_fp8_cuda, SFA_cuda, B_fp8_cuda, SFB_cuda)

    A_fp8_ref, scale_a_logical = _python_mxfp8_quantize(A_bf16)
    B_fp8_ref, scale_b_logical = _python_mxfp8_quantize(B_bf16)
    num_blocks_k = K // 32
    s_m = math.ceil(M / 128) * 128
    s_n = math.ceil(N / 128) * 128
    s_k = math.ceil(num_blocks_k / 4) * 4

    # SFA: from A's logical scales
    scale_a_padded = torch.zeros(1, s_m, s_k, dtype=torch.uint8, device=device)
    scale_a_padded[0, :M, :num_blocks_k] = scale_a_logical
    SFA_ref = _reorder_f8_128x4(scale_a_padded, s_m, s_k).view(torch.float8_e8m0fnu)

    # SFB: from B's logical scales (transpose for column-major B)
    scale_b_padded = torch.zeros(1, s_k, s_n, dtype=torch.uint8, device=device)
    scale_b_padded[0, :num_blocks_k, :N] = scale_b_logical.T
    SFB_ref = _reorder_f8_128x4(scale_b_padded, s_m=s_n, s_k=s_k, s_n=s_n).view(torch.float8_e8m0fnu)

    D_mxfp8_ref = _C.mxfp8_gemm(A_fp8_ref, SFA_ref, B_fp8_ref, SFB_ref)

    b1 = D_mxfp8_cuda.view(torch.uint8)
    b2 = D_mxfp8_ref.view(torch.uint8)
    if not torch.equal(b1, b2):
        diff = (D_mxfp8_cuda.to(torch.float32) - D_mxfp8_ref.to(torch.float32)).abs()
        print(f"Mismatch: max_abs_diff={diff.max().item()}, "
              f"num_diff_bytes={(b1 != b2).sum().item()}/{b1.numel()}")
    assert torch.equal(b1, b2)


if __name__ == "__main__":
    test_mxfp8_vs_bf16()
    test_mxfp8_cuda_quantize_vs_bf16()
    test_python_quant_and_cuda_quant()
