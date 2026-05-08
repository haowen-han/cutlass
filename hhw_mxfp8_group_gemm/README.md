# hhw_mxfp8_group_gemm

基于 CUTLASS 3.x 的 **Blackwell SM100 MXFP8 Grouped GEMM** PyTorch 扩展，用于 MoE 场景。

- C++ 侧直接改写自 [`examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu`](../examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu)
- Python 接口对齐 [sglang 的 `es_sm100_mxfp8_blockscaled_grouped_mm`](https://github.com/sgl-project/sglang)
- 数据类型：`mx_float8_t<float_e4m3_t>` 输入 + `ue8m0` 块标量，fp16 输出

---


## 安装（仅就地编译，不污染全局环境）

```bash
python setup.py build_ext --inplace
```
---

## 运行测试

```bash
cd cutlass/hhw_mxfp8_group_gemm
python -m pytest test_mxfp8_group_gemm.py -s -v
```