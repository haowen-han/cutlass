#include <pybind11/pybind11.h>
#include <torch/extension.h>

void nvfp4_gemm(
    torch::Tensor A,         // [M, packed_K] row-major, uint8 (packed FP4)
    torch::Tensor SFA,       // scale factor for A, float8_e4m3fn, 1D swizzled
    torch::Tensor B,         // [N, packed_K] uint8 (CUTLASS ColumnMajor)
    torch::Tensor SFB,       // scale factor for B, float8_e4m3fn, 1D swizzled
    double alpha,            // epilogue scalar alpha
    torch::Tensor D          // [M, N] row-major output, bfloat16
);

void nvfp4_quantize_cuda(
    torch::Tensor in_bf16,     // [M, K] row-major, bfloat16
    double global_scale,       // scalar float32
    torch::Tensor out_fp4,     // [M, K//2] row-major, uint8 (packed FP4)
    torch::Tensor out_scale    // 1D, float8_e4m3fn, swizzled layout
);

std::tuple<int64_t, int64_t> nvfp4_scale_sizes(int M, int N, int K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemm", &nvfp4_gemm,
        "NVFP4 GEMM: D = alpha * SFA * A * SFB * B (A/B are packed FP4 with e4m3 scales, D is bf16)",
        pybind11::arg("A"), pybind11::arg("SFA"),
        pybind11::arg("B"), pybind11::arg("SFB"),
        pybind11::arg("alpha"),
        pybind11::arg("D"));
    m.def("nvfp4_quantize_cuda", &nvfp4_quantize_cuda,
        "NVFP4 quantize (CUDA): bf16 -> packed FP4 data + e4m3 scale (swizzled layout)",
        pybind11::arg("in_bf16"), pybind11::arg("global_scale"),
        pybind11::arg("out_fp4"), pybind11::arg("out_scale"));
    m.def("nvfp4_scale_sizes", &nvfp4_scale_sizes,
        "Compute swizzled scale factor tensor sizes for NVFP4 GEMM",
        pybind11::arg("M"), pybind11::arg("N"), pybind11::arg("K"));
}
