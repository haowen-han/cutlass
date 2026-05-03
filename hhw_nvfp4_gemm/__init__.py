import math
import torch
from hhw_nvfp4_gemm._C import nvfp4_gemm as _nvfp4_gemm_c
from hhw_nvfp4_gemm._C import nvfp4_quantize_cuda as _nvfp4_quantize_cuda_c
from hhw_nvfp4_gemm._C import nvfp4_scale_sizes as _nvfp4_scale_sizes_c


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def nvfp4_gemm(
    A: torch.Tensor,
    SFA: torch.Tensor,
    B: torch.Tensor,
    SFB: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compute D = alpha * SFA * A * SFB * B where A and B are NVFP4 (e2m1 data with e4m3 scale factors).

    Args:
        A: [M, packed_K] row-major, dtype uint8 (2 FP4 values per byte, packed_K = logical_K / 2)
        SFA: scale factor for A, dtype float8_e4m3fn, 1D (swizzled layout)
        B: [N, packed_K] row-major, dtype uint8 (CUTLASS interprets as ColumnMajor)
        SFB: scale factor for B, dtype float8_e4m3fn, 1D (swizzled layout)
        alpha: scalar multiplier for the product. Default 1.0.
            When using global_scale quantization, set alpha = 1 / (global_scale_A * global_scale_B).

    Returns:
        D: [M, N] row-major, dtype bfloat16
    """
    M = A.size(0)
    N = B.size(0)
    D = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    _nvfp4_gemm_c(A, SFA, B, SFB, alpha, D)
    return D


def nvfp4_quantize(
    in_bf16: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize bf16 tensor to NVFP4 (e2m1 data + e4m3 scale factor in swizzled layout).

    Uses CUDA kernel with PTX cvt.rn.satfinite.e2m1x2.f32 for hardware FP4 conversion,
    vectorized bf16x2 loads, and uint32 stores.

    NVFP4 quantization process:
        scale1 = amax_per_block / 6.0              (ideal bf16 scale)
        scale2 = global_scale * scale1              (FP8-safe scale, stored as e4m3)
        fp4_val = round(x * global_scale / scale2)  (quantized to e2m1 via PTX)

    The scale2 output uses CUTLASS swizzled layout (128x4 basic blocks).

    Args:
        in_bf16: [M, K] row-major, dtype bfloat16.
            K must be multiple of 16 (NVFP4 block size).
        global_scale: scalar tensor, dtype float32.
            global_scale = (6.0 * 448.0) / amax(tensor)

    Returns:
        out_fp4: [M, K // 2] row-major, dtype uint8 (packed FP4, 2 values per byte)
        out_scale: 1D, dtype float8_e4m3fn, scale factors in swizzled layout
    """
    M, K = in_bf16.shape
    packed_K = K // 2

    # Allocate output FP4 tensor
    out_fp4 = torch.empty(M, packed_K, dtype=torch.uint8, device=in_bf16.device)

    # NVFP4 swizzled layout requires M % 128 == 0 and K % 64 == 0
    assert M % 128 == 0, f"M must be multiple of 128, got {M}"
    assert K % 64 == 0, f"K must be multiple of 64, got {K}"

    # Allocate scale tensor: each int32 holds 4 FP8 scale bytes
    num_blocks_k = K // 16
    out_scale_int32 = torch.empty(M, num_blocks_k // 4, dtype=torch.int32, device=in_bf16.device)
    out_scale = out_scale_int32.view(torch.float8_e4m3fn).flatten()

    # Convert global_scale to float
    if isinstance(global_scale, torch.Tensor):
        global_scale_val = float(global_scale.item())
    else:
        global_scale_val = float(global_scale)

    # Call CUDA kernel
    _nvfp4_quantize_cuda_c(in_bf16, global_scale_val, out_fp4, out_scale)

    return out_fp4, out_scale


__all__ = ['nvfp4_gemm', 'nvfp4_quantize']
