/***************************************************************************************************
 * MXFP8 x MXFP4 Mixed Precision GEMM CUDA Extension for PyTorch
 *
 * Wraps the CUTLASS Blackwell SM100 Block-Scaled mixed precision GEMM kernel
 * (based on 72c_blackwell_mixed_mxfp8_bf16_gemm example) into a PyTorch extension.
 * Computes: D = alpha * SFA * A * SFB * B  (bf16 output)
 *
 * Data format:
 *   A     : MXFP8 (e4m3) data, stored as float8_e4m3fn in PyTorch
 *   SFA   : MX E8M0 block scale factors for A, stored as float8_e8m0fnu in PyTorch
 *   B     : MXFP4 (e2m1) packed data, stored as uint8 (2 FP4 values per byte)
 *   SFB   : MX E8M0 block scale factors for B, stored as float8_e8m0fnu in PyTorch
 *   D     : bfloat16 output
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
/// CUTLASS_CHECK macro
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
/// Helper: reinterpret PyTorch tensor data as CUTLASS types
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CutlassType, typename TorchType>
CutlassType* torch_ptr_to_cutlass(torch::Tensor& tensor) {
    return reinterpret_cast<CutlassType*>(tensor.data_ptr<TorchType>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations (based on 72c example: mxfp8 x mxfp4)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration (MXFP8)
using ElementA         = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutATag       = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;  // 16 FP8 elements = 16 bytes

// B matrix configuration (MXFP4)
using ElementB         = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag       = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128;  // 128 FP4 elements = 64 bytes

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

// Kernel Perf config (same as 72c)
using MmaTileShape        = Shape<_256, _256, _256>;
using ClusterShape        = Shape<_2, _4, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

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
/// MXFP8 x MXFP4 Mixed Precision GEMM: D = alpha * SFA * A * SFB * B  (bf16 output)
///
/// A:   [M, K] row-major,       dtype float8_e4m3fn (MXFP8 data)
/// SFA: scale factor for A,     dtype float8_e8m0fnu, 1D contiguous (swizzled layout)
/// B:   [N, packed_K] row-major, dtype uint8 (packed MXFP4, 2 values per byte)
///      PyTorch row-major [N, packed_K] with stride (packed_K, 1) = CUTLASS ColumnMajor
/// SFB: scale factor for B,     dtype float8_e8m0fnu, 1D contiguous (swizzled layout)
/// alpha: scalar multiplier for the product
/// D:   [M, N] row-major output, dtype bfloat16
/////////////////////////////////////////////////////////////////////////////////////////////////

void mxfp4_mxfp8_gemm(
    torch::Tensor A,         // [M, K] row-major, float8_e4m3fn (MXFP8)
    torch::Tensor SFA,       // scale factor for A, float8_e8m0fnu, 1D swizzled
    torch::Tensor B,         // [N, packed_K] uint8 (packed MXFP4), C-contiguous (CUTLASS ColumnMajor)
    torch::Tensor SFB,       // scale factor for B, float8_e8m0fnu, 1D swizzled
    double alpha,            // epilogue scalar alpha
    torch::Tensor D          // [M, N] row-major output, bfloat16
) {
    // Input validation - dtype
    TORCH_CHECK(A.dtype() == torch::kFloat8_e4m3fn, "A must be float8_e4m3fn");
    TORCH_CHECK(SFA.dtype() == torch::kFloat8_e8m0fnu, "SFA must be float8_e8m0fnu");
    TORCH_CHECK(B.dtype() == torch::kUInt8, "B must be uint8 (packed MXFP4)");
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
    const int packed_K = B.size(1);
    const int N = B.size(0);

    // MXFP4: 2 values per byte, so logical K = packed_K * 2
    TORCH_CHECK(packed_K * 2 == K, "B inner dim (packed_K) * 2 must match A inner dim (K)");
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
            torch_ptr_to_cutlass<ElementB::DataType, uint8_t>(B), stride_B,
            torch_ptr_to_cutlass<ElementA::ScaleFactorType, c10::Float8_e8m0fnu>(SFA), layout_SFA,
            torch_ptr_to_cutlass<ElementB::ScaleFactorType, c10::Float8_e8m0fnu>(SFB), layout_SFB
        },
        { // Epilogue arguments: D = alpha * acc + beta * C
          // No bias C, beta=0
          {static_cast<float>(alpha), 0.f},
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


/////////////////////////////////////////////////////////////////////////////////////////////////
/// MXFP8 Quantize Kernel (from hhw_mxfp8_gemm/mxfp8_gemm.cu)
/// Input:  bf16 [M, K]
/// Output: fp8_e4m3 [M, K] data, e8m0 scale in F8_128x4 swizzled layout
/// Requires: M % 16 == 0, K % 256 == 0
/////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ __nv_fp8_e8m0 compute_e8m0_scale(float block_max) {
    constexpr float INV_MAX_FP8 = 1.0f / 448.0f;
    float sf_val = block_max * INV_MAX_FP8;
    __nv_fp8_e8m0 tmp_sf_val;
    tmp_sf_val.__x = __nv_cvt_float_to_e8m0(sf_val, __NV_SATFINITE, cudaRoundPosInf);
    return tmp_sf_val;
}

__global__ void mxfp8_quantize_kernel(
    const __nv_bfloat16* __restrict__ in_bf16,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    uint8_t* __restrict__ out_scale,
    int M,
    int K)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int K_tiles = K / 256;
    int tile_m = bid / K_tiles;
    int tile_k = bid % K_tiles;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int m = tile_m * 16 + warp_id;
    if (m >= M) return;

    int k_start = tile_k * 256;
    int k = k_start + lane_id * 8;

    const uint4* in_vec_ptr = reinterpret_cast<const uint4*>(in_bf16 + m * K + k);
    uint4 vec_in = *in_vec_ptr;
    const __nv_bfloat16* bf16_vals = reinterpret_cast<const __nv_bfloat16*>(&vec_in);

    float local_amax = 0.0f;
    float vals_f32[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        vals_f32[i] = __bfloat162float(bf16_vals[i]);
        local_amax = fmaxf(local_amax, fabsf(vals_f32[i]));
    }

    float max1 = fmaxf(local_amax, __shfl_down_sync(0xFFFFFFFF, local_amax, 1, 4));
    float max_group = fmaxf(max1, __shfl_down_sync(0xFFFFFFFF, max1, 2, 4));

    int q_block_id = lane_id / 4;
    max_group = __shfl_sync(0xFFFFFFFF, max_group, q_block_id * 4, 32);

    __nv_fp8_e8m0 scale_e8m0 = compute_e8m0_scale(max_group);
    float scale_f32 = static_cast<float>(scale_e8m0);
    float inv_scale = (max_group != 0.0f) ? (1.0f / scale_f32) : 0.0f;

    uint2 out_fp8_vec;
    uint8_t* fp8_vals = reinterpret_cast<uint8_t*>(&out_fp8_vec);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        fp8_vals[i] = static_cast<uint8_t>(
            __nv_cvt_float_to_fp8(vals_f32[i] * inv_scale, __NV_SATFINITE, __NV_E4M3)
        );
    }
    uint2* out_vec_ptr = reinterpret_cast<uint2*>(out_fp8 + m * K + k);
    *out_vec_ptr = out_fp8_vec;

    int sub_lane_id = lane_id % 4;
    if (sub_lane_id == 0) {
        int global_k_block = k / 32;
        int sf_inner_dim = K / 32;
        int sf_outer = m / 128;
        int tile_index = sf_outer * (sf_inner_dim / 4) + (global_k_block / 4);
        int offset_tile_base = tile_index * 512;
        int outer_local = m % 128;
        int inner_local = global_k_block % 4;
        int offset_local = (outer_local % 32) * 16 + (outer_local / 32) * 4 + inner_local;
        int final_scale_offset = offset_tile_base + offset_local;
        out_scale[final_scale_offset] = scale_e8m0.__x;
    }
}

void mxfp8_quantize(
    torch::Tensor in_bf16,
    torch::Tensor out_fp8,
    torch::Tensor out_scale
) {
    TORCH_CHECK(in_bf16.dtype() == torch::kBFloat16, "in_bf16 must be bfloat16");
    TORCH_CHECK(out_fp8.dtype() == torch::kFloat8_e4m3fn, "out_fp8 must be float8_e4m3fn");
    TORCH_CHECK(out_scale.dtype() == torch::kFloat8_e8m0fnu, "out_scale must be float8_e8m0fnu");

    TORCH_CHECK(in_bf16.dim() == 2, "in_bf16 must be 2D");
    TORCH_CHECK(out_fp8.dim() == 2, "out_fp8 must be 2D");
    TORCH_CHECK(out_scale.dim() == 1, "out_scale must be 1D");

    const int M = in_bf16.size(0);
    const int K = in_bf16.size(1);

    TORCH_CHECK(out_fp8.size(0) == M && out_fp8.size(1) == K, "out_fp8 shape mismatch");
    TORCH_CHECK(M % 16 == 0, "M must be multiple of 16");
    TORCH_CHECK(K % 256 == 0, "K must be multiple of 256");

    TORCH_CHECK(in_bf16.is_cuda() && out_fp8.is_cuda() && out_scale.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(in_bf16.is_contiguous() && out_fp8.is_contiguous() && out_scale.is_contiguous(),
                "tensors must be contiguous");

    int K_tiles = K / 256;
    int M_tiles = (M + 15) / 16;
    int grid_size = M_tiles * K_tiles;
    int block_size = 512;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mxfp8_quantize_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in_bf16.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8.data_ptr()),
        reinterpret_cast<uint8_t*>(out_scale.data_ptr<c10::Float8_e8m0fnu>()),
        M, K
    );
}

