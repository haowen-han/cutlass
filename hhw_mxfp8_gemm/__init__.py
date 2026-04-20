import math
import torch
from hhw_mxfp8_gemm._C import mxfp8_gemm as _mxfp8_gemm_c
from hhw_mxfp8_gemm._C import mxfp8_quantize as _mxfp8_quantize_c


def mxfp8_gemm(
    A: torch.Tensor,
    SFA: torch.Tensor,
    B: torch.Tensor,
    SFB: torch.Tensor,
) -> torch.Tensor:
    """Compute D = A @ B where A and B are MXFP8 (e4m3 data with e8m0 scale factors).

    Args:
        A: [M, K] row-major, dtype float8_e4m3fn
        SFA: scale factor for A, dtype float8_e8m0fnu, 1D
        B: [N, K] col-major layout, dtype float8_e4m3fn
        SFB: scale factor for B, dtype float8_e8m0fnu, 1D

    Returns:
        D: [M, N] row-major, dtype bfloat16
    """
    M = A.size(0)
    N = B.size(0)
    D = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    _mxfp8_gemm_c(A, SFA, B, SFB, D)
    return D


def mxfp8_quantize(
    in_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize bf16 tensor to MXFP8 (e4m3 data + e8m0 scale factor).

    The scale factor output uses CUTLASS F8_128x4 swizzled layout.

    Args:
        in_bf16: [M, K] row-major, dtype bfloat16.
            M must be multiple of 16, K must be multiple of 256.

    Returns:
        out_fp8: [M, K] row-major, dtype float8_e4m3fn
        out_scale: 1D, dtype float8_e8m0fnu, scale factors in swizzled layout
    """
    M, K = in_bf16.shape
    out_fp8 = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=in_bf16.device)

    # Compute swizzled scale tensor size
    num_blocks_k = K // 32
    s_m = math.ceil(M / 128) * 128
    s_k = math.ceil(num_blocks_k / 4) * 4
    scale_size = s_m * s_k  # F8_128x4 total size
    out_scale = torch.empty(scale_size, dtype=torch.float8_e8m0fnu, device=in_bf16.device)

    _mxfp8_quantize_c(in_bf16, out_fp8, out_scale)
    return out_fp8, out_scale


# Override the C extension submodules' names so that
# `from hhw_mxfp8_gemm import mxfp8_gemm` gives the Python wrapper,
# not the raw C function.
# Python's import system gives priority to submodule attributes (the .so),
# so we must explicitly override after the functions are defined.
__all__ = ['mxfp8_gemm', 'mxfp8_quantize']
