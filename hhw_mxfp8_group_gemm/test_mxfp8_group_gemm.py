"""Test for MXFP8 Grouped GEMM PyTorch extension (Blackwell SM100).

Interface patterned after sglang test_es_mxfp8_blockscaled_moe.py, but quantization
is done in pure Python since this extension does not ship a quantize kernel.
"""

import math
import random
import sys

import pytest
import torch

from hhw_mxfp8_group_gemm import mxfp8_group_gemm


random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def align(val: int, alignment: int = 128) -> int:
    return int((val + alignment - 1) // alignment * alignment)


# Copy from DeepGEMM utils (same as sglang test)
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def is_sm100_supported(device=None) -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(device)[0] == 10
        and torch.version.cuda >= "12.8"
    )


# ----------------------------------------------------------------------------
# Python reference MXFP8 per-32-element block quantization
# ----------------------------------------------------------------------------
# Scale factor is ue8m0: 2^(byte - 127). Quantize per 32-element row block.
def swizzle_e8m0_scale(scale_rowmajor: torch.Tensor) -> torch.Tensor:
    """Convert row-major [M, K/32] e8m0 scale to CUTLASS F8_128x4 swizzled 1D.

    Used for both MXFP8 SFA and MXFP8 SFB (for SFB, pass the [N, K/32] view).
    Requires M % 128 == 0 and (K/32) % 4 == 0 (i.e. K % 128 == 0).

    Swizzle address formula:
        addr(m, kb) = tile_index * 512
                    + (m % 128) %  32 * 16
                    + (m % 128) // 32 *  4
                    + (kb % 4)
        tile_index = (m // 128) * (NK / 4) + (kb // 4)
    """
    M, NK = scale_rowmajor.shape
    assert M % 128 == 0 and NK % 4 == 0
    device = scale_rowmajor.device

    # Enumerate every (m, kb) as a flat 1D index i = m*NK + kb.
    i = torch.arange(M * NK, device=device)
    m  = i // NK
    kb = i %  NK

    tile_index = (m // 128) * (NK // 4) + (kb // 4)
    addr = (tile_index * 512
            + (m % 128) %  32 * 16
            + (m % 128) // 32 *  4
            + (kb % 4))

    out = torch.empty(M * NK, dtype=scale_rowmajor.dtype, device=device)
    out[addr] = scale_rowmajor.reshape(-1)
    return out


def mxfp8_quantize_rowblock_ue8m0(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize last-dim 32-element blocks to (e4m3, ue8m0-scale).

    x:    [..., K]   (K % 32 == 0), float32/bf16/fp16
    Returns:
        fp8:  [..., K] float8_e4m3fn
        scale: [..., K/32] uint8  (ue8m0 byte: 2^(byte - 127))
    """
    assert x.size(-1) % 32 == 0
    orig_shape = x.shape
    K = x.size(-1)
    num_blocks = K // 32

    xf = x.float().view(*orig_shape[:-1], num_blocks, 32)
    amax = xf.abs().amax(dim=-1)  # [..., num_blocks]

    # Scale: smallest power-of-2 such that amax/scale <= 448 (FP8 E4M3 max).
    # scale = 2^(ceil(log2(amax/448)))
    # byte = (exponent + 127), clamped to [0, 255]
    # Handle amax == 0 -> scale byte = 0 (scale = 2^-127, but we output 0 values anyway).
    eps = 1e-30
    ratio = amax / 448.0
    log2r = torch.log2(torch.where(ratio > 0, ratio, torch.full_like(ratio, eps)))
    exponent = torch.ceil(log2r)
    exponent = torch.where(amax > 0, exponent, torch.full_like(exponent, -127.0))
    byte = (exponent + 127.0).clamp(0.0, 255.0).to(torch.uint8)

    scale_f = torch.pow(2.0, byte.float() - 127.0)
    inv_scale = torch.where(scale_f > 0, 1.0 / scale_f, torch.zeros_like(scale_f))
    q = xf * inv_scale.unsqueeze(-1)
    q = q.clamp(-448.0, 448.0)
    fp8 = q.view(*orig_shape).to(torch.float8_e4m3fn)
    scale = byte  # uint8
    return fp8, scale


@pytest.mark.skipif(not is_sm100_supported(), reason="Blackwell SM100 required")
@pytest.mark.parametrize("num_experts", [8, 16])
@pytest.mark.parametrize("out_dtype", [torch.float16])
def test_mxfp8_group_gemm(num_experts, out_dtype):
    device = "cuda"
    alignment = 128

    # Problem shapes.
    n_g = random.randint(1, 16) * alignment
    k_g = random.randint(1, 16) * alignment

    expert_offsets = []
    a_blockscale_offsets = []
    problem_sizes = []
    expert_offset = 0
    a_blockscale_offset = 0
    m_g_list = []
    for _ in range(num_experts):
        m_g = random.randint(1, 256)
        m_g_list.append(m_g)
        expert_offsets.append(expert_offset)
        expert_offset += m_g
        a_blockscale_offsets.append(a_blockscale_offset)
        a_blockscale_offset += align(m_g, 128)
        problem_sizes.append([m_g, n_g, k_g])

    sum_tokens = expert_offset

    # High-precision reference inputs.
    a_ref_list = []
    b_ref_list = []
    for g in range(num_experts):
        a_ref = torch.randn(m_g_list[g], k_g, device=device, dtype=out_dtype)
        b_ref = torch.randn(n_g, k_g, device=device, dtype=out_dtype)
        a_ref_list.append(a_ref)
        b_ref_list.append(b_ref)

    a_ref_cat = torch.concat(a_ref_list, dim=0)  # [sum_tokens, K]

    # ---- Quantize A: per-row-block, [sum(align(m_g,128)), K/32] padded sfa ----
    a_quant = torch.zeros(sum_tokens, k_g, device=device, dtype=torch.float8_e4m3fn)
    sfa_logical = torch.zeros(a_blockscale_offset, k_g // 32, device=device, dtype=torch.uint8)
    sfa = torch.zeros(a_blockscale_offset, k_g // 32, device=device, dtype=torch.uint8)
    for g in range(num_experts):
        off = expert_offsets[g]
        m_g = m_g_list[g]
        sfa_off = a_blockscale_offsets[g]
        m_pad = align(m_g, 128)
        fp8_g, sc_g = mxfp8_quantize_rowblock_ue8m0(a_ref_list[g])
        a_quant[off : off + m_g] = fp8_g
        sfa_logical[sfa_off : sfa_off + m_g] = sc_g
        # Swizzle this expert's SFA block into F8_128x4 physical layout.
        sfa_block = sfa_logical[sfa_off : sfa_off + m_pad]   # [m_pad, K/32] row-major
        sfa[sfa_off : sfa_off + m_pad] = swizzle_e8m0_scale(sfa_block).view(m_pad, k_g // 32)

    # Quantize B. Store as [E, N, K] contiguous, then transpose last two dims to
    # get the [E, K, N] view with K-major strides (stride[1] == 1) required by
    # the kernel (mirroring sglang's b_quant.view(E, N, K).transpose(1, 2)).
    b_quant_nk = torch.empty(num_experts, n_g, k_g, device=device, dtype=torch.float8_e4m3fn)
    sfb_nk_logical = torch.empty(num_experts, n_g, k_g // 32, device=device, dtype=torch.uint8)
    for g in range(num_experts):
        fp8_g, sc_g = mxfp8_quantize_rowblock_ue8m0(b_ref_list[g])
        b_quant_nk[g] = fp8_g
        sfb_nk_logical[g] = sc_g
    b_quant = b_quant_nk.transpose(1, 2)             # [E, K, N], stride(1) == 1
    sfb_logical = sfb_nk_logical.transpose(1, 2)     # [E, K/32, N], stride(1) == 1 (for ref)
    # Swizzle each expert's SFB (viewed as [N, K/32] row-major) into F8_128x4 layout.
    sfb_phys_flat = torch.empty(num_experts, n_g * (k_g // 32), device=device, dtype=torch.uint8)
    for g in range(num_experts):
        # sfb_nk_logical[g] is [N, K/32] row-major: exactly what swizzle_e8m0_scale expects,
        # with the 128-aligned axis being N.
        sfb_phys_flat[g] = swizzle_e8m0_scale(sfb_nk_logical[g])
    sfb = sfb_phys_flat  # kernel-visible buffer
    assert b_quant.stride(1) == 1, f"b_quant inner stride {b_quant.stride()}"

    # Host tensors -> device int32 for kernel-side metadata.
    _problem_sizes = torch.tensor(problem_sizes, device=device, dtype=torch.int32)
    _expert_offsets = torch.tensor(expert_offsets, device=device, dtype=torch.int32)
    _a_blockscale_offsets = torch.tensor(a_blockscale_offsets, device=device, dtype=torch.int32)

    # Reference: run the high-precision (bf16/fp16) matmul on the *original*
    # unquantized A/B. The MXFP8 grouped GEMM is compared against this baseline
    # rather than against a dequantized reproduction of the quantization.
    ref_d_list = []
    for g in range(num_experts):
        a_hp = a_ref_list[g].float()          # [m_g, K]
        b_hp = b_ref_list[g].float()          # [N, K]
        ref_d_list.append((a_hp @ b_hp.T).to(out_dtype))

    d = mxfp8_group_gemm(
        a_quant,
        b_quant,
        sfa,
        sfb,
        _problem_sizes,
        _expert_offsets,
        _a_blockscale_offsets,
        out_dtype=out_dtype,
    )

    for g in range(num_experts):
        baseline = ref_d_list[g]
        off = expert_offsets[g]
        m_g = m_g_list[g]
        actual = d[off : off + m_g]
        print("")
        print("actual:",actual)
        print("baseline:",baseline)
        diff = calc_diff(actual, baseline)
        assert diff < 0.03, f"expert {g} diff {diff}"
        print(
            f"expert={g} m_g={m_g} n_g={n_g} k_g={k_g} "
            f"num_experts={num_experts} out_dtype={out_dtype} diff={diff:.5f}: OK"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s"]))
