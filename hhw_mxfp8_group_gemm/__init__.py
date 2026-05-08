import torch
from hhw_mxfp8_group_gemm._C import mxfp8_group_gemm as _mxfp8_group_gemm_c


def mxfp8_group_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Blackwell SM100 MXFP8 Grouped GEMM for MoE expert dispatch.

    Computes, for each expert g:
        D[offs_m : offs_m + m_g, :] = A[offs_m : offs_m + m_g, :] @ B[g].T

    with per-32-element block-scaled MXFP8 (e4m3 data + ue8m0 scale factor).

    Args:
        a: [sum_tokens, K] float8_e4m3fn, row-major. Tokens from all experts
            concatenated along the row dimension.
        b: [E, K, N] float8_e4m3fn, strides[1] == 1 (K-major per group, i.e.,
            [N, K] column-major).
        sfa: [sum(align(m_g, 128)), K/32] uint8 (interpreted as ue8m0).
        sfb: [E, K/32, N] uint8 (interpreted as ue8m0).
        problem_sizes: [E, 3] int32, rows are (m_g, N, K).
        expert_offsets: [E] int32, prefix sum of m_g.
        blockscale_offsets: [E] int32, prefix sum of align(m_g, 128).
        out_dtype: torch.bfloat16 or torch.float16.

    Returns:
        d: [sum_tokens, N] out_dtype row-major.
    """
    assert a.dim() == 2 and b.dim() == 3, "a 2D, b 3D"
    sum_tokens = a.size(0)
    N = b.size(2)
    d = torch.empty((sum_tokens, N), dtype=out_dtype, device=a.device)
    _mxfp8_group_gemm_c(
        d, a, b, sfa, sfb, problem_sizes, expert_offsets, blockscale_offsets
    )
    return d


__all__ = ["mxfp8_group_gemm"]
