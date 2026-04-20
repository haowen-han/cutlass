import os
import subprocess
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

current_directory = os.path.abspath(os.path.dirname(__file__))
cutlass_dir = os.path.abspath(os.path.join(current_directory, ".."))

name = "hhw_mxfp8_gemm"

# Remove flags that disable half/bf16 operators
REMOVE_NVCC_FLAGS = [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_OPERATORS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT162_OPERATORS__",
    "-D__CUDA_NO_BFLOAT162_CONVERSIONS__",
]
for flag in REMOVE_NVCC_FLAGS:
    try:
        torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
        pass

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

CXX_FLAGS = [
    "-std=c++17",
    "-O3",
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
]

NVCC_FLAGS = [
    "-std=c++17",
    "-O3",
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--threads=8",
]

# Blackwell SM100 architecture
NVCC_FLAGS += [
    "-gencode=arch=compute_100a,code=sm_100a",
]

if CUDA_HOME is None:
    raise RuntimeError("Cannot find CUDA_HOME. CUDA must be available to build this extension.")

sources = [
    os.path.join(current_directory, "mxfp8_gemm.cu"),
    os.path.join(current_directory, "pybind.cpp"),
]

include_dirs = [
    os.path.join(cutlass_dir, "include"),
    os.path.join(cutlass_dir, "tools", "util", "include"),
]

setup(
    name=name,
    version="0.1.0",
    packages=["hhw_mxfp8_gemm"],
    package_dir={"hhw_mxfp8_gemm": "."},
    ext_modules=[
        CUDAExtension(
            "hhw_mxfp8_gemm._C",
            sources=sources,
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
            include_dirs=include_dirs,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
