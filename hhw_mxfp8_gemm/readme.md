# hhw_mxfp8_gemm

基于 CUTLASS Blackwell SM100 Block-Scaled GEMM 的 MXFP8 矩阵乘法 PyTorch 扩展。

计算 `D = A @ B`，其中 A、B 为 MXFP8（e4m3 数据 + e8m0 block scale），输出 D 为 bfloat16。

## 硬件要求

- NVIDIA Blackwell 架构 GPU（SM100, sm_100a）
- CUDA 12.x

## 依赖

- PyTorch（带 CUDA 支持）
- CUTLASS（位于上级目录 `../include`）

## 安装

```bash
# 在项目根目录下编译安装
cd /path/to/cutlass_hhw/hhw_mxfp8_gemm
python setup.py install
```

编译会自动链接上级 CUTLASS 的 include 目录，生成 `hhw_mxfp8_gemm._C` CUDA 扩展模块。

## 用法

### 1. CUDA 量化 + GEMM（推荐）

```python
import torch
import hhw_mxfp8_gemm as mxfp8

A_bf16 = torch.rand(256, 1024, dtype=torch.bfloat16, device="cuda") * 2.0 - 1.0
B_bf16 = torch.rand(512, 1024, dtype=torch.bfloat16, device="cuda") * 2.0 - 1.0

# 量化：bf16 -> (fp8 e4m3 数据 + e8m0 scale, F8_128x4 swizzled layout)
A_fp8, SFA = mxfp8.mxfp8_quantize(A_bf16)
B_fp8, SFB = mxfp8.mxfp8_quantize(B_bf16)

# GEMM
D = mxfp8.mxfp8_gemm(A_fp8, SFA, B_fp8, SFB)  # [256, 512], bfloat16
```

### 2. Python 量化 + GEMM

也可以使用 Python 端的 `_python_mxfp8_quantize` 进行量化（见 `test_mxfp8_gemm.py`），手动构造 F8_128x4 swizzled scale 后调用 `mxfp8_gemm`。此路径主要用于正确性验证。

## API

### `mxfp8_quantize(in_bf16) -> (out_fp8, out_scale)`

将 bf16 张量量化为 MXFP8。

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `in_bf16` | `[M, K]` | bfloat16 | 输入，M 须为 16 的倍数，K 须为 256 的倍数 |
| `out_fp8` | `[M, K]` | float8_e4m3fn | 量化后数据 |
| `out_scale` | 1D | float8_e8m0fnu | e8m0 scale，F8_128x4 swizzled layout |

### `mxfp8_gemm(A, SFA, B, SFB) -> D`

MXFP8 矩阵乘法 `D = A @ B`。

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `A` | `[M, K]` | float8_e4m3fn | 行优先 |
| `SFA` | 1D | float8_e8m0fnu | A 的 scale（swizzled layout） |
| `B` | `[N, K]` | float8_e4m3fn | C-contiguous（CUTLASS 按 ColumnMajor 解读 stride） |
| `SFB` | 1D | float8_e8m0fnu | B 的 scale（swizzled layout） |
| 返回 `D` | `[M, N]` | bfloat16 | 行优先输出 |

## 测试

```bash
python test_mxfp8_gemm.py
```

包含三个测试：

| 测试函数 | 说明 |
|----------|------|
| `test_mxfp8_vs_bf16` | Python 量化 → MXFP8 GEMM，与 bf16 参考结果对比 |
| `test_mxfp8_cuda_quantize_vs_bf16` | CUDA 量化 → MXFP8 GEMM，与 bf16 参考结果对比 |
| `test_python_quant_and_cuda_quant` | 验证 CUDA 量化与 Python 量化产生完全一致的 GEMM 结果（二进制级别） |

## 文件结构

```
hhw_mxfp8_gemm/
├── setup.py              # 编译配置（pip install）
├── mxfp8_gemm.cu         # CUDA kernel：GEMM + 量化
├── pybind.cpp            # PyTorch/Pybind11 绑定
├── __init__.py           # Python 封装（自动分配输出张量）
└── test_mxfp8_gemm.py    # 测试脚本
```

## MXFP8 Scale 说明

- **E8M0 格式**：1 符号位 + 8 指数位 + 0 尾数位，值为 `2^(byte - 127)`，只能是 2 的幂
- **Block 量化**：每 32 个元素共享一个 scale，`scale = ceil_pow2(amax / 448)`
- **F8_128x4 Swizzle**：CUTLASS Block-Scaled GEMM 要求 scale 按 F8_128x4 排布，CUDA 量化 kernel 在写出时直接完成 swizzle，避免额外 reorder
