"""
Test script for NVFP4 GEMM CUDA extension.

Tests the NVFP4 quantization + CUTLASS GEMM pipeline against a bf16 reference,
following the same pattern as SGLang's test_nvfp4_gemm.py.

NVFP4 quantization process:
    scale1 = amax_per_block / 6.0              (ideal bf16 scale)
    scale2 = global_scale * scale1              (FP8-safe scale, stored as e4m3)
    alpha  = 1 / (global_scale_A * global_scale_B)
    fp4_val = round(x * global_scale / scale2)  (quantized to e2m1)

    GEMM: D = alpha * SFA * A * SFB * B ≈ A_orig @ B_orig
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import hhw_nvfp4_gemm as _C


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# E2M1 unsigned values indexed by 3-bit
_K_E2M1_TO_FLOAT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _break_fp4_bytes(a: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 FP4 tensor to float32. Each byte -> 2 FP4 values."""
    assert a.dtype == torch.uint8
    m, n = a.shape
    a = a.flatten()
    low_half = (a & 0x0F)
    high_half = (a >> 4) & 0x0F

    # E2M1: 4-bit with sign (bit3) + 3-bit magnitude
    # Magnitude values: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    e2m1_magnitudes = torch.tensor(_K_E2M1_TO_FLOAT, dtype=torch.float32, device=a.device)

    # Extract magnitude index (bits 0-2) and sign (bit 3)
    low_mag_idx = (low_half & 0x7).long()
    high_mag_idx = (high_half & 0x7).long()

    f_l = e2m1_magnitudes[low_mag_idx]
    f_h = e2m1_magnitudes[high_mag_idx]

    # Apply sign: bit 3 = sign
    sign_l = ((low_half & 0x8) != 0).to(torch.float32) * -2.0 + 1.0
    sign_h = ((high_half & 0x8) != 0).to(torch.float32) * -2.0 + 1.0

    f_l = f_l * sign_l
    f_h = f_h * sign_h

    return torch.stack((f_l, f_h), dim=-1).reshape(m, n * 2)


def _convert_swizzled_to_linear(a_sf_swizzled, m, k, block_size=16):
    """Convert swizzled NVFP4 scale tensor back to logical linear order."""
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4  # 64 for NVFP4
    k_tiles = (k + f - 1) // f
    Kb = k // block_size

    sf_shape = a_sf_swizzled.shape[0]
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:Kb]


def _reorder_nvfp4_128x4(tensor, s_m, s_k):
    """Apply NVFP4 F4_128x4 swizzle reordering."""
    x = tensor
    b = x.shape[0]
    num_m_blocks = s_m // 128
    n_k_blocks = s_k // 4

    x = x.view(b, num_m_blocks, 4, 32, n_k_blocks, 4)
    x = x.permute(0, 1, 4, 3, 2, 5)
    return x.contiguous().view(-1)


def _python_nvfp4_quantize(in_bf16, global_scale):
    """Reference NVFP4 quantization in pure Python.

    Returns:
        out_fp4: [M, K//2] uint8 (packed FP4, low nibble first)
        scale_swizzled: 1D float8_e4m3fn (swizzled layout)
        scale2_logical: [M, K//16] float8_e4m3fn (logical order, for reference)
    """
    M, K = in_bf16.shape
    block_size = 16
    packed_K = K // 2
    device = in_bf16.device
    num_blocks_k = K // block_size

    in_f32 = in_bf16.to(torch.float32)
    in_blocked = in_f32.view(M, num_blocks_k, block_size)
    block_amax = in_blocked.abs().amax(dim=-1)  # (M, num_blocks_k)

    # scale1 = amax / 6.0
    scale1 = block_amax / FLOAT4_E2M1_MAX

    # scale2 = global_scale * scale1
    global_scale_val = float(global_scale) if not isinstance(global_scale, torch.Tensor) else global_scale.item()
    scale2 = global_scale_val * scale1

    # Store scale2 as FP8 E4M3
    scale2_clamped = scale2.clamp(0.0, FLOAT8_E4M3_MAX)
    scale2_fp8 = scale2_clamped.to(torch.float8_e4m3fn)

    # Quantize: fp4_val = round(x * global_scale / scale2_fp8)
    scale2_f32 = scale2_fp8.to(torch.float32)
    inv_scale = global_scale_val / scale2_f32
    inv_scale = torch.where(block_amax > 0, inv_scale, torch.zeros_like(inv_scale))

    in_scaled = in_blocked * inv_scale.unsqueeze(-1)
    in_scaled = in_scaled.clamp(-FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)

    # Convert to E2M1
    signs = (in_scaled < 0).to(torch.uint8)
    abs_vals = in_scaled.abs()

    e2m1_vals = torch.tensor(_K_E2M1_TO_FLOAT, dtype=torch.float32, device=device)
    abs_flat = abs_vals.reshape(-1)
    diffs = (abs_flat.unsqueeze(-1) - e2m1_vals.unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)
    sign_flat = signs.reshape(-1)
    fp4_signed = indices | (sign_flat << 3)

    # Pack 2 FP4 per byte (low nibble = even index, high nibble = odd index)
    fp4_reshaped = fp4_signed.reshape(M, K)
    low_nibbles = fp4_reshaped[:, 0::2]
    high_nibbles = fp4_reshaped[:, 1::2]
    out_fp4 = (high_nibbles << 4) | low_nibbles

    # Swizzle scale factors
    scale_bytes = scale2_fp8.view(torch.uint8)
    s_m = math.ceil(M / 128) * 128
    s_k = math.ceil(num_blocks_k / 4) * 4

    scale_padded = torch.zeros(1, s_m, s_k, dtype=torch.uint8, device=device)
    scale_padded[0, :M, :num_blocks_k] = scale_bytes

    scale_swizzled = _reorder_nvfp4_128x4(scale_padded, s_m, s_k)
    scale_out = scale_swizzled.view(torch.float8_e4m3fn)

    return out_fp4, scale_out, scale2_fp8


def _dequantize_nvfp4(tensor_fp4, tensor_sf, global_scale, m, k, block_size=16):
    """Dequantize NVFP4 tensor back to float32 for reference."""
    tensor_f32 = _break_fp4_bytes(tensor_fp4)  # (m, k)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)

    tensor_sf_linear = _convert_swizzled_to_linear(
        tensor_sf.view(torch.uint8), m, k, block_size
    )
    tensor_sf_f32 = tensor_sf_linear.to(torch.float32) / global_scale

    return (tensor_f32 * tensor_sf_f32.unsqueeze(-1)).reshape(m, k)


def test_nvfp4_gemm_python_quant():
    """Quantize with Python reference, run NVFP4 GEMM, compare against bf16 reference."""
    device = torch.device("cuda")

    # M, N must be multiple of 256 (for 72a MmaTileShape _256,_256,_256)
    # K must be multiple of 256
    M, N, K = 256, 256, 256
    block_size = 16

    print(f"\nTesting NVFP4 GEMM (Python quant) vs BF16: M={M}, N={N}, K={K}")

    A_bf16 = torch.rand(M, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0
    B_bf16 = torch.rand(N, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0

    # Compute global scales
    a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / A_bf16.abs().max()).to(torch.float32)
    b_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / B_bf16.abs().max()).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    # Quantize using Python reference
    A_fp4, SFA, _ = _python_nvfp4_quantize(A_bf16, a_global_scale)
    B_fp4, SFB, _ = _python_nvfp4_quantize(B_bf16, b_global_scale)

    # NVFP4 GEMM
    D_nvfp4 = _C.nvfp4_gemm(A_fp4, SFA, B_fp4, SFB, alpha=alpha)

    # BF16 reference
    D_bf16 = A_bf16 @ B_bf16.T

    # Compare
    diff = (D_nvfp4.to(torch.float32) - D_bf16.to(torch.float32)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (D_bf16.to(torch.float32).abs() + 1e-6)).mean().item()

    print(f"  Max abs diff: {max_diff:.4f}")
    print(f"  Mean abs diff: {mean_diff:.4f}")
    print(f"  Mean rel diff: {rel_diff:.4f}")
    print(f"  D_nvfp4 sample: {D_nvfp4[0, :10]}")
    print(f"  D_bf16   sample: {D_bf16[0, :10]}")

    # NVFP4 has limited precision (4-bit), so tolerance is generous
    assert max_diff < 50.0, f"Max diff {max_diff} too large"
    print("  PASSED")


def test_nvfp4_gemm_cuda_quant():
    """Quantize with Python nvfp4_quantize, run NVFP4 GEMM, compare against bf16 reference."""
    device = torch.device("cuda")

    M, N, K = 256, 256, 256

    print(f"\nTesting NVFP4 GEMM (nvfp4_quantize) vs BF16: M={M}, N={N}, K={K}")

    A_bf16 = torch.rand(M, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0
    B_bf16 = torch.rand(N, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0

    a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / A_bf16.abs().max()).to(torch.float32)
    b_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / B_bf16.abs().max()).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    # Quantize using Python API
    A_fp4, SFA = _C.nvfp4_quantize(A_bf16, a_global_scale)
    B_fp4, SFB = _C.nvfp4_quantize(B_bf16, b_global_scale)

    print(f"  A_fp4 shape: {A_fp4.shape}, dtype: {A_fp4.dtype}")
    print(f"  SFA shape: {SFA.shape}, dtype: {SFA.dtype}")
    print(f"  B_fp4 shape: {B_fp4.shape}, dtype: {B_fp4.dtype}")
    print(f"  SFB shape: {SFB.shape}, dtype: {SFB.dtype}")
    print(f"  alpha: {alpha}")

    # NVFP4 GEMM
    D_nvfp4 = _C.nvfp4_gemm(A_fp4, SFA, B_fp4, SFB, alpha=alpha)

    # BF16 reference
    D_bf16 = A_bf16 @ B_bf16.T

    diff = (D_nvfp4.to(torch.float32) - D_bf16.to(torch.float32)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max abs diff: {max_diff:.4f}")
    print(f"  Mean abs diff: {mean_diff:.4f}")
    print(f"  D_nvfp4 sample: {D_nvfp4[0, :10]}")
    print(f"  D_bf16   sample: {D_bf16[0, :10]}")

    assert max_diff < 50.0, f"Max diff {max_diff} too large"
    print("  PASSED")


def test_nvfp4_cuda_quant_dequant():
    """Test CUDA quantization by dequantizing and comparing with original bf16."""
    device = torch.device("cuda")

    M, K = 256, 256

    print(f"\nTesting NVFP4 CUDA quantize/dequantize: M={M}, K={K}")

    A_bf16 = torch.rand(M, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0
    a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / A_bf16.abs().max()).to(torch.float32)

    A_fp4, SFA = _C.nvfp4_quantize(A_bf16, a_global_scale)

    # Dequantize
    A_dequant = _dequantize_nvfp4(A_fp4, SFA, a_global_scale, M, K)

    diff = (A_dequant - A_bf16.to(torch.float32)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max abs diff: {max_diff:.4f}")
    print(f"  Mean abs diff: {mean_diff:.4f}")
    print(f"  A_bf16 sample:    {A_bf16[0, :8]}")
    print(f"  A_dequant sample: {A_dequant[0, :8]}")

    # NVFP4 has 4-bit precision, so differences are expected
    assert max_diff < 2.0, f"Max diff {max_diff} too large"
    print("  PASSED")


def test_nvfp4_python_vs_cuda_quant():
    """Compare Python and CUDA quantization produce identical results for GEMM."""
    device = torch.device("cuda")

    M, N, K = 256, 256, 256

    print(f"\nTesting Python vs CUDA quantize consistency: M={M}, N={N}, K={K}")

    A_bf16 = torch.rand(M, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0
    B_bf16 = torch.rand(N, K, dtype=torch.bfloat16, device=device) * 2.0 - 1.0

    a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / A_bf16.abs().max()).to(torch.float32)
    b_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / B_bf16.abs().max()).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    # Python quantize
    A_fp4_py, SFA_py, _ = _python_nvfp4_quantize(A_bf16, a_global_scale)
    B_fp4_py, SFB_py, _ = _python_nvfp4_quantize(B_bf16, b_global_scale)

    # CUDA quantize
    A_fp4_cu, SFA_cu = _C.nvfp4_quantize(A_bf16, a_global_scale)
    B_fp4_cu, SFB_cu = _C.nvfp4_quantize(B_bf16, b_global_scale)

    # GEMM results
    D_py = _C.nvfp4_gemm(A_fp4_py, SFA_py, B_fp4_py, SFB_py, alpha=alpha)
    D_cu = _C.nvfp4_gemm(A_fp4_cu, SFA_cu, B_fp4_cu, SFB_cu, alpha=alpha)

    diff = (D_py.to(torch.float32) - D_cu.to(torch.float32)).abs()
    max_diff = diff.max().item()
    print(f"  Max abs diff between Python and CUDA quantize GEMM results: {max_diff:.4f}")
    print(f"  D_py sample: {D_py[0, :10]}")
    print(f"  D_cu sample: {D_cu[0, :10]}")

    # They should produce very similar results (may differ slightly due to
    # Python's nearest-e2m1 vs PTX's cvt.rn.satfinite.e2m1x2 rounding)
    assert max_diff < 5.0, f"Max diff {max_diff} too large"
    print("  PASSED")


if __name__ == "__main__":
    test_nvfp4_gemm_python_quant()
    test_nvfp4_cuda_quant_dequant()
    test_nvfp4_gemm_cuda_quant()
    test_nvfp4_python_vs_cuda_quant()
