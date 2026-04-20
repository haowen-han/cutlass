/***************************************************************************************************
 * MXFP8 GEMM CUDA Extension for PyTorch
 *
 * Wraps the CUTLASS Blackwell SM100 Block-Scaled MXFP8 GEMM kernel into a PyTorch extension.
 * Computes: D = A @ B  where A and B are MXFP8 (e4m3) matrices with E8M0 scale factors.
 * Output D is bf16.
 *
 * Tensor dtypes (using PyTorch native FP8 types):
 *   A, B  : torch.float8_e4m3fn   (FP8 E4M3 data, 1 byte/element)
 *   SFA, SFB : torch.float8_e8m0fnu  (UE8M0 block scale factors, 1 byte/element)
 *   D     : torch.bfloat16        (BF16 output)
 *
 * Memory layout compatibility:
 *   CUTLASS float_e4m3_t::storage  (uint8_t) <-> c10::Float8_e4m3fn::x  (uint8_t) : identical bit layout
 *   CUTLASS float_ue8m0_t::storage (uint8_t) <-> c10::Float8_e8m0fnu::x (uint8_t) : identical bit layout
 **************************************************************************************************/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <optional>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"


#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// CUTLASS_CHECK macro (from examples/common/helper.h)
/////////////////////////////////////////////////////////////////////////////////////////////////

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper: reinterpret PyTorch FP8 tensor data as CUTLASS FP8 types
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CutlassType, typename TorchType>
CutlassType* torch_ptr_to_cutlass(torch::Tensor& tensor) {
    return reinterpret_cast<CutlassType*>(tensor.data_ptr<TorchType>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations (same as 72c example)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using ElementA         = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutATag       = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

// B matrix configuration
using ElementB         = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutBTag       = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16;

// C/D matrix configuration (bf16 output)
using ElementD         = cutlass::bfloat16_t;
using ElementC         = cutlass::bfloat16_t;
using LayoutCTag       = cutlass::layout::RowMajor;
using LayoutDTag       = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

// Kernel functional config
using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm100;
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;

// Kernel Perf config
using MmaTileShape        = Shape<_256, _128, _128>;
using ClusterShape        = Shape<_2, _1, _1>;
using EpilogueTileShape   = Shape<_128, _64>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    EpilogueTileShape,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::TmaWarpSpecialized2Sm
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmBlockScaledSm100
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Type aliases
/////////////////////////////////////////////////////////////////////////////////////////////////

using StrideA   = typename GemmKernel::StrideA;
using StrideB   = typename GemmKernel::StrideB;
using StrideC   = typename GemmKernel::StrideC;
using StrideD   = typename GemmKernel::StrideD;
using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;

using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute scale factor tensor sizes
/////////////////////////////////////////////////////////////////////////////////////////////////

static int64_t compute_sfa_size(int M, int N, int K) {
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    return size(filter_zeros(layout_SFA));
}

static int64_t compute_sfb_size(int M, int N, int K) {
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
    return size(filter_zeros(layout_SFB));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// MXFP8 GEMM: D = A @ B  (bf16 output)
///
/// A:   [M, K] row-major,       dtype float8_e4m3fn
/// SFA: scale factor for A,     dtype float8_e8m0fnu, 1D contiguous
/// B:   [N, K] col-major,       dtype float8_e4m3fn
/// SFB: scale factor for B,     dtype float8_e8m0fnu, 1D contiguous
/// D:   [M, N] row-major output, dtype bfloat16
/////////////////////////////////////////////////////////////////////////////////////////////////

void mxfp8_gemm(
    torch::Tensor A,         // [M, K] row-major, float8_e4m3fn
    torch::Tensor SFA,       // scale factor for A, float8_e8m0fnu, 1D
    torch::Tensor B,         // [N, K] float8_e4m3fn, C-contiguous in PyTorch (interpreted as ColumnMajor by CUTLASS)
    torch::Tensor SFB,       // scale factor for B, float8_e8m0fnu, 1D
    torch::Tensor D          // [M, N] row-major output, bfloat16
) {
    // Input validation - dtype
    TORCH_CHECK(A.dtype() == torch::kFloat8_e4m3fn, "A must be float8_e4m3fn");
    TORCH_CHECK(SFA.dtype() == torch::kFloat8_e8m0fnu, "SFA must be float8_e8m0fnu");
    TORCH_CHECK(B.dtype() == torch::kFloat8_e4m3fn, "B must be float8_e4m3fn");
    TORCH_CHECK(SFB.dtype() == torch::kFloat8_e8m0fnu, "SFB must be float8_e8m0fnu");
    TORCH_CHECK(D.dtype() == torch::kBFloat16, "D must be bfloat16");

    // Input validation - shape
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(D.dim() == 2, "D must be 2D");
    TORCH_CHECK(SFA.dim() == 1, "SFA must be 1D");
    TORCH_CHECK(SFB.dim() == 1, "SFB must be 1D");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    TORCH_CHECK(B.size(1) == K, "B inner dim must match K");
    TORCH_CHECK(D.size(0) == M, "D rows must match M");
    TORCH_CHECK(D.size(1) == N, "D cols must match N");

    // Validate scale factor sizes
    int64_t expected_sfa_size = compute_sfa_size(M, N, K);
    int64_t expected_sfb_size = compute_sfb_size(M, N, K);
    TORCH_CHECK(SFA.numel() >= expected_sfa_size,
        "SFA size mismatch: expected >= ", expected_sfa_size, ", got ", SFA.numel());
    TORCH_CHECK(SFB.numel() >= expected_sfb_size,
        "SFB size mismatch: expected >= ", expected_sfb_size, ", got ", SFB.numel());

    // All tensors must be on CUDA
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(SFA.is_cuda(), "SFA must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(SFB.is_cuda(), "SFB must be on CUDA");
    TORCH_CHECK(D.is_cuda(), "D must be on CUDA");

    // Contiguous check
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(SFA.is_contiguous(), "SFA must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(SFB.is_contiguous(), "SFB must be contiguous");
    TORCH_CHECK(D.is_contiguous(), "D must be contiguous");

    // Compute strides
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    // C stride: no bias, zero stride
    auto stride_C = cute::make_stride(int64_t(0), cute::C<1>{}, int64_t(0));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    // Compute scale factor layouts
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    // Build arguments
    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop arguments
            torch_ptr_to_cutlass<ElementA::DataType, c10::Float8_e4m3fn>(A), stride_A,
            torch_ptr_to_cutlass<ElementB::DataType, c10::Float8_e4m3fn>(B), stride_B,
            torch_ptr_to_cutlass<ElementA::ScaleFactorType, c10::Float8_e8m0fnu>(SFA), layout_SFA,
            torch_ptr_to_cutlass<ElementB::ScaleFactorType, c10::Float8_e8m0fnu>(SFB), layout_SFB
        },
        { // Epilogue arguments: D = alpha * acc + beta * C
          // No bias C, beta=0
          {1.f, 0.f},
          nullptr, stride_C,
          reinterpret_cast<ElementD*>(D.data_ptr()), stride_D
        }
    };

    arguments.scheduler.max_swizzle_size = 0;

    // Instantiate and run
    Gemm gemm;

    CUTLASS_CHECK(gemm.can_implement(arguments));

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));
    CUTLASS_CHECK(gemm.run(stream));
}


// --- 更新 1: 使用原生 intrinsic 计算 E8M0 Scale ---
__device__ __forceinline__ __nv_fp8_e8m0 compute_e8m0_scale(float block_max) {
    
    // 预计算的常数倒数。在现代 CUDA 编译器中，1.0f / 448.0f 会在编译期直接
    // 转换为一个常量乘法数。你也可以继续使用你代码库里的 reciprocal_approximate_ftz。
    constexpr float INV_MAX_FP8 = 1.0f / 448.0f;
    float sf_val = block_max * INV_MAX_FP8;
    
    // 调用底层原语进行转换，cudaRoundPosInf 完美实现向上取整到 2 的幂
    __nv_fp8_e8m0 tmp_sf_val;
    tmp_sf_val.__x = __nv_cvt_float_to_e8m0(sf_val, __NV_SATFINITE, cudaRoundPosInf);
    
    return tmp_sf_val;
}


// =========================================================
// 主量化 Kernel (1D Grid, 1D Block)
// 假设: M 是 16 的倍数, K 是 256 的倍数
// =========================================================
__global__ void mxfp8_quantize_kernel(
    const __nv_bfloat16* __restrict__ in_bf16, // 输入: M x K
    __nv_fp8_e4m3* __restrict__ out_fp8,       // 输出数据: M x K
    uint8_t* __restrict__ out_scale,           // 输出缩放系数 (Swizzle 排布)
    int M, 
    int K) 
{
    // 1. 获取线程和 Block ID
    int tid = threadIdx.x;   // 0 ~ 511
    int bid = blockIdx.x;    // 1D Grid ID

    // 2. 将 1D Grid 映射到二维 Tile (16 x 256)
    int K_tiles = K / 256;          // K 维度有多少个 Tile
    int tile_m = bid / K_tiles;     // 当前 Block 处理的 M 维度的块坐标
    int tile_k = bid % K_tiles;     // 当前 Block 处理的 K 维度的块坐标

    // 3. 将 1D Thread 映射到 Warp (Warp 级任务分配)
    int warp_id = tid / 32;         // 0 ~ 15 (对应当前 Block 内的 16 行)
    int lane_id = tid % 32;         // 0 ~ 31

    // 计算当前线程对应的全局行号 m
    int m = tile_m * 16 + warp_id;
    if (m >= M) return;             // 边界保护

    // 4. 向量化读取数据 (1 个线程读 8 个 bf16 = 16 Bytes = 1个 uint4)
    int k_start = tile_k * 256;
    int k = k_start + lane_id * 8;
    
    // 使用 uint4 强制 128-bit 对齐加载
    const uint4* in_vec_ptr = reinterpret_cast<const uint4*>(in_bf16 + m * K + k);
    uint4 vec_in = *in_vec_ptr;
    
    // 将 128-bit 数据强制转换为 bf16 数组
    const __nv_bfloat16* bf16_vals = reinterpret_cast<const __nv_bfloat16*>(&vec_in);

    // 5. 将 bf16 转为 float，并寻找该线程负责的 8 个元素的最大值
    float local_amax = 0.0f;
    float vals_f32[8];
    
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
        vals_f32[i] = __bfloat162float(bf16_vals[i]);
        local_amax = fmaxf(local_amax, fabsf(vals_f32[i]));
    }

    // 6. Sub-warp 归约 (每 4 个线程归约出 32 个元素的最大值)
    // 掩码 0xFFFFFFFF 表示 Warp 内所有线程均参与
    float max1 = fmaxf(local_amax, __shfl_down_sync(0xFFFFFFFF, local_amax, 1, 4));
    float max_group = fmaxf(max1, __shfl_down_sync(0xFFFFFFFF, max1, 2, 4));

    // 将归约结果广播回这 4 个线程
    int q_block_id = lane_id / 4;  // 当前线程属于 256 个元素中的第几个 32-element 量化块 (0~7)
    max_group = __shfl_sync(0xFFFFFFFF, max_group, q_block_id * 4, 32);

    // 7. 计算该 32 元素块的 Scale，并得到缩放因子
    __nv_fp8_e8m0 scale_e8m0 = compute_e8m0_scale(max_group);
    float scale_f32 = static_cast<float>(scale_e8m0);
    // 防止除 0
    float inv_scale = (max_group != 0.0f) ? (1.0f / scale_f32) : 0.0f;

    // 8. 向量化执行 FP8 量化并写出数据 (8 个 fp8 = 8 Bytes = 1 个 uint2)
    uint2 out_fp8_vec;
    uint8_t* fp8_vals = reinterpret_cast<uint8_t*>(&out_fp8_vec);
    
    #pragma unroll
    for(int i = 0; i < 8; ++i) {
        // 使用 CUDA 内置 intrinsics 进行量化 (带饱和截断)
        fp8_vals[i] = static_cast<uint8_t>(
            __nv_cvt_float_to_fp8(vals_f32[i] * inv_scale, __NV_SATFINITE, __NV_E4M3)
        );
    }

    // 强制 64-bit 对齐写出
    uint2* out_vec_ptr = reinterpret_cast<uint2*>(out_fp8 + m * K + k);
    *out_vec_ptr = out_fp8_vec;

    // 9. 极其关键：以 Swizzle 排布写出 Scale
    // 每 4 个线程 (即 32 个元素) 共享 1 个 scale，只需要 1 个线程去写内存即可
    int sub_lane_id = lane_id % 4;
    
    if (sub_lane_id == 0) {
        // 计算全局 inner 索引 (即它是总体的第几个 32-element 块)
        int global_k_block = k / 32; 
        
        // 宏观参数
        int sf_inner_dim = K / 32;      // inner 维度的总大小
        int sf_outer = m / 128;         // 这是外层的第几个宏观行 (Tile Row)
        
        // 步骤 1: 计算所在 Tile 的基址偏移
        // (sf_inner_dim / 4) 表示一行有多少个 Tile
        // (global_k_block / 4) 表示当前处于该行的第几个 Tile
        int tile_index = sf_outer * (sf_inner_dim / 4) + (global_k_block / 4);
        int offset_tile_base = tile_index * 512; // 每个 Tile 严格占 512 Bytes
        
        // 步骤 2: 计算 Tile 内部的微观 Swizzle 偏移
        int outer_local = m % 128;
        int inner_local = global_k_block % 4;
        int offset_local = (outer_local % 32) * 16 + (outer_local / 32) * 4 + inner_local;
        
        // 步骤 3: 绝对偏移量
        int final_scale_offset = offset_tile_base + offset_local;
        
        // 写出 Scale (access raw byte via __x)
        out_scale[final_scale_offset] = scale_e8m0.__x;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// MXFP8 Quantize: bf16 -> (e4m3 data + e8m0 scale in CUTLASS swizzled layout)
///
/// in_bf16:   [M, K] row-major, bfloat16 input
/// out_fp8:   [M, K] row-major, float8_e4m3fn output (same logical layout)
/// out_scale: 1D contiguous, float8_e8m0fnu, scale factors in F8_128x4 swizzled layout
///
/// Requirements: M must be multiple of 16, K must be multiple of 256
/////////////////////////////////////////////////////////////////////////////////////////////////

void mxfp8_quantize(
    torch::Tensor in_bf16,    // [M, K] row-major, bfloat16
    torch::Tensor out_fp8,    // [M, K] row-major, float8_e4m3fn
    torch::Tensor out_scale   // 1D, float8_e8m0fnu, swizzled layout
) {
    // Input validation - dtype
    TORCH_CHECK(in_bf16.dtype() == torch::kBFloat16, "in_bf16 must be bfloat16");
    TORCH_CHECK(out_fp8.dtype() == torch::kFloat8_e4m3fn, "out_fp8 must be float8_e4m3fn");
    TORCH_CHECK(out_scale.dtype() == torch::kFloat8_e8m0fnu, "out_scale must be float8_e8m0fnu");

    // Input validation - shape
    TORCH_CHECK(in_bf16.dim() == 2, "in_bf16 must be 2D");
    TORCH_CHECK(out_fp8.dim() == 2, "out_fp8 must be 2D");
    TORCH_CHECK(out_scale.dim() == 1, "out_scale must be 1D");

    const int M = in_bf16.size(0);
    const int K = in_bf16.size(1);

    TORCH_CHECK(out_fp8.size(0) == M, "out_fp8 rows must match M");
    TORCH_CHECK(out_fp8.size(1) == K, "out_fp8 cols must match K");

    // Alignment requirements from kernel
    TORCH_CHECK(M % 16 == 0, "M must be multiple of 16");
    TORCH_CHECK(K % 256 == 0, "K must be multiple of 256");

    // All tensors must be on CUDA and contiguous
    TORCH_CHECK(in_bf16.is_cuda(), "in_bf16 must be on CUDA");
    TORCH_CHECK(out_fp8.is_cuda(), "out_fp8 must be on CUDA");
    TORCH_CHECK(out_scale.is_cuda(), "out_scale must be on CUDA");
    TORCH_CHECK(in_bf16.is_contiguous(), "in_bf16 must be contiguous");
    TORCH_CHECK(out_fp8.is_contiguous(), "out_fp8 must be contiguous");
    TORCH_CHECK(out_scale.is_contiguous(), "out_scale must be contiguous");

    // Grid: each block handles a 16x256 tile
    int K_tiles = K / 256;
    int M_tiles = (M + 15) / 16;
    int grid_size = M_tiles * K_tiles;
    int block_size = 512;  // 16 warps (one per row in the 16-row tile)

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mxfp8_quantize_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in_bf16.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8.data_ptr()),
        reinterpret_cast<uint8_t*>(out_scale.data_ptr<c10::Float8_e8m0fnu>()),
        M, K
    );
}