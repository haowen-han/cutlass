# hhw_mxfp4_mxfp8_gemm

基于 CUTLASS Blackwell SM100 Block-Scaled Tensor Core 的 **MXFP8 × MXFP4 → BF16** 混合精度 GEMM 的 PyTorch 扩展。

- A 矩阵：MXFP8（`float8_e4m3fn` 数据 + `float8_e8m0fnu` scale，block_size=32）
- B 矩阵：MXFP4（`uint8` 打包，2 个 FP4 共 1 字节 + `float8_e8m0fnu` scale，block_size=32）
- D 矩阵：`bfloat16`
- 硬件原生 MMA：`tcgen05.mma.blockscaled`（不走软件反量化）

内核配置参考 `examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm.cu`。


## 安装方式

**不要用 `pip install .`**，它会拉取一堆 build 依赖、非常慢。

直接用 `setup.py build_ext --inplace` 就地编译，编出来的 `_C.cpython-*.so` 会直接放在本目录：

```bash
cd cutlass/hhw_mxfp4_mxfp8_gemm
python setup.py build_ext --inplace
```

编译选项说明（已配置好，无需改）：

- `-gencode=arch=compute_100a,code=sm_100a`：Blackwell 架构
- `--threads=8`：nvcc 并行编译
- 移除了 `-D__CUDA_NO_HALF/BFLOAT16*` 等限制 half/bf16 算子的标志
- include 路径自动从父目录取 `cutlass/include`、`cutlass/tools/util/include`

编译产物：
```
hhw_mxfp4_mxfp8_gemm/_C.cpython-<pyver>-x86_64-linux-gnu.so
```

## 测试方式

由于包名是 `hhw_mxfp4_mxfp8_gemm`（就是当前目录名），需要把 **父目录** 加到 `PYTHONPATH` 里运行测试：

```bash
cd cutlass
PYTHONPATH=. python hhw_mxfp4_mxfp8_gemm/test_mxfp4_mxfp8_gemm.py
```
