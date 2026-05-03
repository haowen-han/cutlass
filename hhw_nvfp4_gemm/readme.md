# hhw_nvfp4_gemm 新手入门指南

本文基于 commit `5a225da204e7d5cda4bf0fbd4eb3d496978cadd4`（`support nvfp4 * nvfp4 gemm`）整理，目标是带你从零理解并跑通这个 PyTorch CUDA Extension：先知道它能做什么，再学会安装、测试、调用，最后逐步理解 NVFP4 量化和 CUTLASS GEMM 内部流程。

## 1. 这个目录是什么

`hhw_nvfp4_gemm` 是一个把 CUTLASS Blackwell SM100 Block-Scaled NVFP4 GEMM 包装成 PyTorch 接口的小模块。

它主要提供两个能力：

1. `nvfp4_quantize`：把 `bf16` 矩阵量化成 NVFP4 格式。
2. `nvfp4_gemm`：用 CUTLASS kernel 计算两个 NVFP4 矩阵乘法，输出 `bf16`。

可以把完整流程理解成：

```text
A_bf16 ----quantize----> A_fp4 + SFA --+
                                       +--> nvfp4_gemm --> D_bf16 ≈ A_bf16 @ B_bf16.T
B_bf16 ----quantize----> B_fp4 + SFB --+
```

## 2. 先建立直觉：什么是 NVFP4 GEMM

普通矩阵乘法是：

```text
D = A @ B.T
```

这里为了节省显存和提升吞吐，会先把 `A` 和 `B` 从 `bf16` 压缩成 NVFP4：

- 数据本体使用 4 bit `e2m1`，每个 `uint8` 存 2 个 FP4 数值。
- 每 16 个 FP4 元素共享一个 `float8_e4m3fn` scale factor。
- GEMM 时 CUTLASS 会把 FP4 数据和 scale factor 组合起来近似还原数值，再做矩阵乘。

因此实际计算形式是：

```text
D = alpha * SFA * A_fp4 * SFB * B_fp4
```

当输入来自本目录的量化接口时：

```text
alpha = 1 / (global_scale_A * global_scale_B)
```

这样输出 `D` 会近似等价于：

```text
A_bf16 @ B_bf16.T
```

## 3. 文件一览

```text
hhw_nvfp4_gemm/
├── __init__.py           # Python 友好接口：分配输出 tensor，并调用 C++/CUDA 扩展
├── nvfp4_gemm.cu         # 核心实现：CUTLASS NVFP4 GEMM + CUDA NVFP4 quantize kernel
├── pybind.cpp            # pybind11 绑定，把 C++ 函数暴露给 Python
├── setup.py              # PyTorch CUDAExtension 编译脚本
└── test_nvfp4_gemm.py    # 端到端测试和 Python 参考实现
```

## 4. 运行前提

这个模块面向 Blackwell SM100：

- 需要支持 `sm_100a` 的 NVIDIA GPU。
- 需要 CUDA / NVCC 支持 `compute_100a, sm_100a`。
- 需要 PyTorch 支持 `torch.float8_e4m3fn` 和 CUDA Extension。
- 需要在 CUTLASS 仓库根目录附近编译，因为 `setup.py` 会引用上级目录的 CUTLASS 头文件：
  - `../include`
  - `../tools/util/include`

如果你只是阅读代码，不需要满足这些硬件条件；如果要实际运行测试，则必须有对应 GPU 和 CUDA 环境。

## 5. Step 1：编译安装扩展

进入 CUTLASS 仓库根目录后执行：

```bash
cd cutlass/hhw_nvfp4_gemm
python setup.py install
```


`setup.py` 里最关键的编译配置是：

```text
-gencode=arch=compute_100a,code=sm_100a
```

这说明生成的是 Blackwell SM100a 目标代码。

## 6. Step 2：跑通内置测试

编译完成后运行：

```bash
cd cutlass/hhw_nvfp4_gemm
python hhw_nvfp4_gemm/test_nvfp4_gemm.py
```

测试脚本会依次验证：

1. Python 参考量化 + NVFP4 GEMM 对比 bf16 GEMM。
2. CUDA 量化后反量化，对比原始 bf16。
3. CUDA 量化 + NVFP4 GEMM 对比 bf16 GEMM。
4. Python 量化和 CUDA 量化的 GEMM 结果一致性。

因为 NVFP4 只有 4 bit 精度，测试里允许一定误差。
