#include <pybind11/pybind11.h>
#include <torch/extension.h>

void mxfp8_group_gemm(
    torch::Tensor& d,                         // [sum_tokens, N] bf16/fp16 row-major
    const torch::Tensor& a,                   // [sum_tokens, K] float8_e4m3fn row-major
    const torch::Tensor& b,                   // [E, K, N]       float8_e4m3fn K-major
    const torch::Tensor& sfa,                 // [sum(align(m_g,128)), K/32] uint8/e8m0
    const torch::Tensor& sfb,                 // [E, K/32, N]    uint8/e8m0
    const torch::Tensor& problem_sizes,       // [E, 3] int32 (M_g, N, K)
    const torch::Tensor& expert_offsets,      // [E]    int32
    const torch::Tensor& blockscale_offsets); // [E]    int32

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "mxfp8_group_gemm",
      &mxfp8_group_gemm,
      "MXFP8 Grouped GEMM for MoE (Blackwell SM100)",
      pybind11::arg("d"),
      pybind11::arg("a"),
      pybind11::arg("b"),
      pybind11::arg("sfa"),
      pybind11::arg("sfb"),
      pybind11::arg("problem_sizes"),
      pybind11::arg("expert_offsets"),
      pybind11::arg("blockscale_offsets"));
}
