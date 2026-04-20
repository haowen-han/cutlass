#include <pybind11/pybind11.h>
#include <torch/extension.h>

void mxfp8_gemm(
    torch::Tensor A,         // [M, K] row-major, float8_e4m3fn
    torch::Tensor SFA,       // scale factor for A, float8_e8m0fnu, 1D
    torch::Tensor B,         // [N, K] col-major, float8_e4m3fn
    torch::Tensor SFB,       // scale factor for B, float8_e8m0fnu, 1D
    torch::Tensor D          // [M, N] row-major output, bfloat16
);

void mxfp8_quantize(
    torch::Tensor in_bf16,    // [M, K] row-major, bfloat16
    torch::Tensor out_fp8,    // [M, K] row-major, float8_e4m3fn
    torch::Tensor out_scale   // 1D, float8_e8m0fnu, swizzled layout
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp8_gemm", &mxfp8_gemm,
        "MXFP8 GEMM: D = A @ B (A/B are mxfp8 e4m3 with e8m0 scales, D is bf16)",
        pybind11::arg("A"), pybind11::arg("SFA"),
        pybind11::arg("B"), pybind11::arg("SFB"),
        pybind11::arg("D"));
    m.def("mxfp8_quantize", &mxfp8_quantize,
        "MXFP8 quantize: bf16 -> e4m3 data + e8m0 scale (CUTLASS swizzled layout)",
        pybind11::arg("in_bf16"), pybind11::arg("out_fp8"),
        pybind11::arg("out_scale"));
}
