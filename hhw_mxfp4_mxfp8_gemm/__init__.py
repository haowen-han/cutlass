import torch

from hhw_mxfp4_mxfp8_gemm._C import mxfp4_mxfp8_gemm as _mxfp4_mxfp8_gemm_c
from hhw_mxfp4_mxfp8_gemm._C import mxfp8_quantize as _mxfp8_quantize_c


# -----------------------------------------------------------------------------
# Core GEMM wrapper
# -----------------------------------------------------------------------------

def mxfp4_mxfp8_gemm(
    A: torch.Tensor,
    SFA: torch.Tensor,
    B: torch.Tensor,
    SFB: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """D = alpha * (SFA * A) @ (SFB * B).T  (bf16 output)

    A:   [M, K] float8_e4m3fn (MXFP8)
    SFA: 1D float8_e8m0fnu, F8_128x4 swizzled
    B:   [N, K/2] uint8 (packed MXFP4, 2 FP4 per byte), PyTorch row-major -> CUTLASS ColumnMajor
    SFB: 1D float8_e8m0fnu, F8_128x4 swizzled
    """
    M = A.size(0)
    N = B.size(0)
    D = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    _mxfp4_mxfp8_gemm_c(A, SFA, B, SFB, alpha, D)
    return D


# -----------------------------------------------------------------------------
# MXFP8 quantization (CUDA kernel)
# -----------------------------------------------------------------------------

def mxfp8_quantize(x_bf16: torch.Tensor):
    """Quantize bf16 tensor to MXFP8 with swizzled e8m0 scales.

    Input:
        x_bf16: [M, K] bfloat16 (M % 16 == 0, K % 256 == 0)
    Returns:
        fp8: [M, K] float8_e4m3fn
        scale: 1D float8_e8m0fnu in F8_128x4 swizzled layout, length = M * (K//32)
    """
    assert x_bf16.dtype == torch.bfloat16 and x_bf16.dim() == 2
    M, K = x_bf16.shape
    fp8 = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x_bf16.device)
    scale = torch.empty(M * (K // 32), dtype=torch.float8_e8m0fnu, device=x_bf16.device)
    _mxfp8_quantize_c(x_bf16.contiguous(), fp8, scale)
    return fp8, scale


# -----------------------------------------------------------------------------
# MXFP4 quantization (pure Python, adapted from sglang mxfp4_tensor.py)
# -----------------------------------------------------------------------------

_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_MAX = 6.0


def mxfp4_quantize_rowmajor(x: torch.Tensor, block_size: int = 32):
    """Quantize x -> (packed uint8 fp4 data [.., K/2], row-major e8m0 scale [.., K/block]).

    The returned scale is **row-major** (not swizzled). Apply `swizzle_e8m0_scale`
    before feeding into the GEMM.
    """
    assert x.shape[-1] % block_size == 0
    orig_shape = x.shape
    x_flat = x.reshape(-1, block_size).to(torch.float32)

    amax = x_flat.abs().amax(dim=-1, keepdim=True)
    descale = amax / _E2M1_MAX
    min_value = torch.tensor(-127.0, device=x.device)
    e8m0_scale_f = torch.ceil(torch.maximum(torch.log2(descale.clamp_min(1e-30)), min_value))
    e8m0_scale = (e8m0_scale_f + 127).to(torch.uint8)  # biased e8m0

    x_scaled = x_flat / torch.exp2(e8m0_scale_f)  # normalize to [-6, 6]

    # cast to fp4 indices
    sign = torch.sign(x_scaled)
    sign_bit = ((2 - sign) // 2).to(torch.uint8)  # +->0, -->1
    bounds = _E2M1_BOUNDS.to(x_scaled.device)
    ord_ = ((x_scaled.abs().unsqueeze(-1) - bounds) > 0).sum(dim=-1).to(torch.uint8)
    fp4_val = (sign_bit * 0b1000 + ord_)  # 4-bit value in low nibble

    # reshape back and pack two fp4 into one uint8 (even in low bits, odd in high bits)
    fp4_val = fp4_val.reshape(orig_shape)
    left = fp4_val[..., 0::2]
    right = fp4_val[..., 1::2]
    packed = (right << 4) | left  # uint8

    scale = e8m0_scale.reshape(orig_shape[:-1] + (orig_shape[-1] // block_size,))
    # reinterpret scale as float8_e8m0fnu (same bits)
    scale_e8m0 = scale.view(torch.float8_e8m0fnu)
    return packed.contiguous(), scale_e8m0.contiguous()


def mxfp4_dequantize(packed: torch.Tensor, scale_e8m0: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Dequantize packed MXFP4 + row-major e8m0 scale to float32.

    packed:      [.., K/2] uint8
    scale_e8m0:  [.., K/block] float8_e8m0fnu (row-major, biased e8m0)
    Returns float32 tensor of shape [.., K].
    """
    # unpack
    left = packed & 0x0F
    right = (packed >> 4) & 0x0F
    unpacked = torch.stack([left, right], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)

    sign = 1.0 - 2.0 * ((unpacked & 0b1000) >> 3).to(torch.float32)
    mag = (unpacked & 0b0111).to(torch.long)
    values = _E2M1_VALUES.to(packed.device)
    x = values[mag.reshape(-1)].reshape(mag.shape) * sign

    scale_u8 = scale_e8m0.view(torch.uint8).to(torch.float32)
    scale_f = torch.exp2(scale_u8 - 127.0)

    x = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    x = x * scale_f.unsqueeze(-1)
    return x.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


# -----------------------------------------------------------------------------
# E8M0 scale swizzle: row-major [M, K/32] -> F8_128x4 swizzled 1D
# This swizzle is used for BOTH MXFP8 and MXFP4 scales on Blackwell.
# -----------------------------------------------------------------------------

def swizzle_e8m0_scale(scale_rowmajor: torch.Tensor) -> torch.Tensor:
    """Convert a row-major [M, K/32] e8m0 scale to CUTLASS F8_128x4 swizzled 1D layout.

    Requires M % 128 == 0, (K/32) % 4 == 0 (i.e. K % 128 == 0).

    Swizzle address formula, directly applied below:
        addr(m, kb) = tile_index * 512
                    + (m % 128) % 32 * 16
                    + (m % 128) // 32 * 4
                    + (kb % 4)
        tile_index  = (m // 128) * (NK / 4) + (kb // 4)
    """
    M, NK = scale_rowmajor.shape
    assert M % 128 == 0 and NK % 4 == 0
    device = scale_rowmajor.device

    # Enumerate every (m, kb) as a flat 1D index i = m*NK + kb
    i = torch.arange(M * NK, device=device)
    m  = i // NK
    kb = i %  NK

    tile_index = (m // 128) * (NK // 4) + (kb // 4)
    addr = (tile_index * 512
            + (m % 128) % 32 * 16
            + (m % 128) // 32 * 4
            + (kb % 4))

    out = torch.empty(M * NK, dtype=scale_rowmajor.dtype, device=device)
    out[addr] = scale_rowmajor.reshape(-1)
    return out


__all__ = [
    'mxfp4_mxfp8_gemm',
    'mxfp8_quantize',
    'mxfp4_quantize_rowmajor',
    'mxfp4_dequantize',
    'swizzle_e8m0_scale',
]
