#!/bin/bash
set -e

# 自动调优：sed 改源码 → make → 跑 test_temp.sh → 收集 GFLOPS
# trap EXIT 保证源码还原

mkdir -p /ssd1/hanhaowen/autotune_results

SRC=/ssd1/hanhaowen/cutlass/examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm.cu
RESULT_FILE=/ssd1/hanhaowen/autotune_results/result_$(date +%Y%m%d_%H%M%S).log

CONFIGS=()
for mma_m in 256 512; do
for mma_n in 128 256; do
for mma_k in 128 256; do
for cluster_m in 2 4; do
for cluster_n in 1 2 4; do
for epi_m in 64 128; do
for epi_n in 64 128; do
  CONFIGS+=("$mma_m $mma_n $mma_k $cluster_m $cluster_n $epi_m $epi_n")
done; done; done; done; done; done; done

cp "$SRC" /tmp/72c_original_backup.cu
trap "cp /tmp/72c_original_backup.cu '$SRC' && echo 'Source restored.'" EXIT

for cfg in "${CONFIGS[@]}"; do
  read mma_m mma_n mma_k cluster_m cluster_n epi_m epi_n <<< "$cfg"
  tag="m${mma_m}_n${mma_n}_k${mma_k}_cm${cluster_m}_cn${cluster_n}_e${epi_m}x${epi_n}"

  echo "" >> $RESULT_FILE
  echo "========== Config: MmaTile=${mma_m}x${mma_n}x${mma_k}, Cluster=${cluster_m}x${cluster_n}, EpiTile=${epi_m}x${epi_n} ==========" >> $RESULT_FILE

  echo "[$tag] Patching source..."
  sed -i "119s|.*|using MmaTileShape        = Shape<_${mma_m},_${mma_n},_${mma_k}>;                          // MMA's tile size|" "$SRC"
  sed -i "120s|.*|using ClusterShape        = Shape<_${cluster_m},_${cluster_n},_1>;                                // Shape of the threadblocks in a cluster|" "$SRC"
  sed -i "121s|.*|using EpilogueTileShape   = Shape<_${epi_m}, _${epi_n}>;|" "$SRC"

  echo "[$tag] Building..."
  if ! (cd /ssd1/hanhaowen/cutlass/build_blackwell && make 72c_blackwell_mixed_mxfp8_bf16_gemm -j1 > /dev/null 2>&1); then
    echo "[$tag] Compile failed, skipping."
    echo "Compile failed." >> $RESULT_FILE
    continue
  fi

  echo "[$tag] Running test_temp.sh..."
  bash /ssd1/hanhaowen/test_temp.sh >> $RESULT_FILE 2>&1 || echo "[$tag] Test crashed." >> $RESULT_FILE

  echo "[$tag] Done."
done

echo ""
echo "All results: $RESULT_FILE"
