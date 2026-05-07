#include <pybind11/pybind11.h>
#include <torch/extension.h>

void mxfp4_mxfp8_gemm(
    torch::Tensor A,
    torch::Tensor SFA,
    torch::Tensor B,
    torch::Tensor SFB,
    double alpha,
    torch::Tensor D
);

void mxfp8_quantize(
    torch::Tensor in_bf16,
    torch::Tensor out_fp8,
    torch::Tensor out_scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_mxfp8_gemm", &mxfp4_mxfp8_gemm,
        "MXFP8 x MXFP4 Mixed Precision GEMM (D = alpha * SFA * A * SFB * B, bf16 output)",
        pybind11::arg("A"), pybind11::arg("SFA"),
        pybind11::arg("B"), pybind11::arg("SFB"),
        pybind11::arg("alpha"),
        pybind11::arg("D"));

    m.def("mxfp8_quantize", &mxfp8_quantize,
        "MXFP8 quantize: bf16 -> (e4m3 data, e8m0 scale in F8_128x4 swizzled layout). "
        "Requires M % 16 == 0, K % 256 == 0.",
        pybind11::arg("in_bf16"),
        pybind11::arg("out_fp8"),
        pybind11::arg("out_scale"));
}
