"""Unit tests for MXFP8 x MXFP4 mixed-precision GEMM.

Compare:
  D_ref = A_bf16 @ B_bf16.T                              (bf16 baseline)
  D_q   = mxfp4_mxfp8_gemm(quantize_mxfp8(A), quantize_mxfp4(B))
"""

import torch
import hhw_mxfp4_mxfp8_gemm as fp


def _run_case(M, N, K, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5

    # ----- bf16 reference -----
    D_ref_bf16 = A_bf16.float() @ B_bf16.float().T

    # ----- MXFP8 quantize A (CUDA kernel, produces swizzled scale directly) -----
    A_fp8, SFA_swizzled = fp.mxfp8_quantize(A_bf16)

    # ----- MXFP4 quantize B (Python, row-major scale) then swizzle -----
    B_packed, SFB_rowmajor = fp.mxfp4_quantize_rowmajor(B_bf16, block_size=32)
    SFB_swizzled = fp.swizzle_e8m0_scale(SFB_rowmajor)

    # ----- Run the kernel -----
    D_q = fp.mxfp4_mxfp8_gemm(A_fp8, SFA_swizzled, B_packed, SFB_swizzled, alpha=1.0)

    print(f"\n=== Case M={M} N={N} K={K} ===")
    print(f"D_ref (bf16 A@B.T)    sample [0,:8]:\n  {D_ref_bf16[0, :8].cpu()}")
    print(f"D_q   (mxfp8 x mxfp4) sample [0,:8]:\n  {D_q[0, :8].float().cpu()}")

    diff = (D_q.float() - D_ref_bf16).abs()
    print(f"diff: max={diff.max().item():.4f}  "
          f"mean={diff.mean().item():.4f}  "
          f"rel={(diff.mean() / D_ref_bf16.abs().mean()).item():.4f}")
    print(f"D_ref (bf16 A@B.T)   :\n  {D_ref_bf16.cpu()}")
    print(f"D_q   (mxfp8 x mxfp4):\n  {D_q.float().cpu()}")

if __name__ == "__main__":
    _run_case(256, 256, 256)
    _run_case(256, 256, 512)
    _run_case(512, 512, 512)
    _run_case(512, 512, 1024)
    _run_case(512, 1024, 1024)
