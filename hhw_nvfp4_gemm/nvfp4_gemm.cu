/***************************************************************************************************
 * NVFP4 GEMM CUDA Extension for PyTorch
 *
 * Wraps the CUTLASS Blackwell SM100 Block-Scaled NVFP4 GEMM kernel into a PyTorch extension.
 * Computes: D = alpha * SFA * A * SFB * B  (bf16 output)
 *
 * NVFP4 data format:
 *   A, B  : packed FP4 (e2m1) data, 2 values per byte, stored as uint8
 *   SFA, SFB : FP8 E4M3 block scale factors (float_ue4m3_t / float8_e4m3fn), block_size=16
 *   D     : bfloat16 output
 *
 * Memory layout compatibility:
 *   CUTLASS float_e2m1_t is 4-bit, packed 2-per-byte
 *   CUTLASS float_ue4m3_t::storage (uint8_t) <-> c10::Float8_e4m3fn::x (uint8_t) : same for positive values
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
/// GEMM kernel configurations (based on 72a example)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using ElementA         = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag       = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;  // 32 FP4 elements = 16 bytes = 128 bits

// B matrix configuration
using ElementB         = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag       = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

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

// Kernel Perf config (same as 72a)
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
/// NVFP4 GEMM: D = alpha * SFA * A * SFB * B  (bf16 output)
///
/// A:   [M, packed_K] row-major, dtype uint8 (2 FP4 values per byte, packed_K = logical_K / 2)
/// SFA: scale factor for A, dtype float8_e4m3fn, 1D contiguous (swizzled layout)
/// B:   [N, packed_K] col-major in CUTLASS view, dtype uint8
///      (PyTorch row-major [N, packed_K] with stride (packed_K, 1) = CUTLASS ColumnMajor)
/// SFB: scale factor for B, dtype float8_e4m3fn, 1D contiguous (swizzled layout)
/// alpha: scalar multiplier for the product (typically 1/(global_scale_A * global_scale_B))
/// D:   [M, N] row-major output, dtype bfloat16
/////////////////////////////////////////////////////////////////////////////////////////////////

void nvfp4_gemm(
    torch::Tensor A,         // [M, packed_K] row-major, uint8 (packed FP4)
    torch::Tensor SFA,       // scale factor for A, float8_e4m3fn, 1D swizzled
    torch::Tensor B,         // [N, packed_K] uint8, C-contiguous (CUTLASS ColumnMajor)
    torch::Tensor SFB,       // scale factor for B, float8_e4m3fn, 1D swizzled
    double alpha,            // epilogue scalar alpha
    torch::Tensor D          // [M, N] row-major output, bfloat16
) {
    // Input validation - dtype
    TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8 (packed FP4)");
    TORCH_CHECK(SFA.dtype() == torch::kFloat8_e4m3fn, "SFA must be float8_e4m3fn");
    TORCH_CHECK(B.dtype() == torch::kUInt8, "B must be uint8 (packed FP4)");
    TORCH_CHECK(SFB.dtype() == torch::kFloat8_e4m3fn, "SFB must be float8_e4m3fn");
    TORCH_CHECK(D.dtype() == torch::kBFloat16, "D must be bfloat16");

    // Input validation - shape
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(D.dim() == 2, "D must be 2D");
    TORCH_CHECK(SFA.dim() == 1, "SFA must be 1D");
    TORCH_CHECK(SFB.dim() == 1, "SFB must be 1D");

    const int M = A.size(0);
    const int packed_K = A.size(1);
    const int K = packed_K * 2;  // Logical K dimension (number of FP4 elements)
    const int N = B.size(0);

    TORCH_CHECK(B.size(1) == packed_K, "B inner dim must match A inner dim (packed_K)");
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

    // Compute strides using logical K dimension
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
            reinterpret_cast<ElementA::DataType*>(A.data_ptr<uint8_t>()), stride_A,
            reinterpret_cast<ElementB::DataType*>(B.data_ptr<uint8_t>()), stride_B,
            reinterpret_cast<ElementA::ScaleFactorType*>(SFA.data_ptr<c10::Float8_e4m3fn>()), layout_SFA,
            reinterpret_cast<ElementB::ScaleFactorType*>(SFB.data_ptr<c10::Float8_e4m3fn>()), layout_SFB
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
/// NVFP4 Quantize: CUDA kernel
///
/// Based on SGLang's nvfp4_quant_kernels.cuh, adapted for PyTorch extension.
/// Uses PTX cvt.rn.satfinite.e2m1x2.f32 for hardware FP4 conversion,
/// vectorized bf16x2 loads, and uint32 stores (8 FP4 values per store).
///
/// Quantization formula:
///   scale1 = amax(block) / 6.0
///   scale2 = global_scale * scale1      (stored as FP8 E4M3)
///   fp4_val = round(x / scale1)          (via PTX cvt instruction)
///
/// Scale factor swizzled layout: 128x4 basic blocks, same as CUTLASS Sm1xxBlockScaledBasicChunk
/////////////////////////////////////////////////////////////////////////////////////////////////

// Constants
constexpr int FP4_ELTS_PER_THREAD = 8;   // Each thread processes 8 FP4 elements
constexpr int FP4_SF_VEC_SIZE     = 16;   // 16 FP4 elements share one scale factor
constexpr int FP4_THREADS_PER_SF  = FP4_SF_VEC_SIZE / FP4_ELTS_PER_THREAD; // = 2

// Fast reciprocal approximation (flush-to-zero)
__device__ __forceinline__ float reciprocal_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

// Convert 4 float2 values (8 floats) into 8 e2m1 values packed into uint32_t
// Uses PTX cvt.rn.satfinite.e2m1x2.f32 (requires SM100+)
__device__ __forceinline__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0].x), "f"(array[0].y),
          "f"(array[1].x), "f"(array[1].y),
          "f"(array[2].x), "f"(array[2].y),
          "f"(array[3].x), "f"(array[3].y));
    return val;
}

// Compute the swizzled offset for a scale factor write
// Layout: [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]
// Same as CUTLASS Sm1xxBlockScaledBasicChunk<16> atom layout
__device__ __forceinline__ uint8_t* get_sf_out_offset(
    int rowIdx, int colIdx, int numCols, uint32_t* SFout) {

    if (threadIdx.x % FP4_THREADS_PER_SF != 0) return nullptr;

    // SF vector index (16 elements share one SF in the K dimension)
    int32_t kIdx = colIdx / FP4_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF layout: [numMTiles, numKTiles, 32 (outerM), 4 (innerM), 4 (innerK)]
    int32_t mTileIdx = mIdx / (32 * 4);  // 128 rows per M tile
    int factor = FP4_SF_VEC_SIZE * 4;     // 16 * 4 = 64 data elements per K tile
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int64_t mTileStride = numKTiles * 32 * 4 * 4;

    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;

    // M tile layout [32, 4] is column-major (swizzle)
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride
                     + outerMIdx * outerMStride + innerMIdx * innerMStride
                     + innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
}

// Packed bf16x2 vector type for vectorized loads
struct Bf16PackedVec {
    __nv_bfloat162 elts[4];  // 4 x bf16x2 = 8 bf16 values = 16 bytes
};

// Core per-thread quantization: bf16 -> FP4 + scale factor
__device__ __forceinline__ uint32_t cvt_bf16_to_fp4(
    Bf16PackedVec& vec, float globalScale, uint8_t* sfOut) {

    // Step 1: Find absolute max among 8 local values (4 bf16x2 pairs)
    auto localMax = __habs2(vec.elts[0]);
    #pragma unroll
    for (int i = 1; i < 4; i++) {
        localMax = __hmax2(localMax, __habs2(vec.elts[i]));
    }

    // Step 2: Cross-thread reduction to get max of all 16 values (2 threads share 1 SF)
    localMax = __hmax2(__shfl_xor_sync(0xFFFFFFFF, localMax, 1), localMax);
    float vecMax = float(__hmax(localMax.x, localMax.y));

    // Step 3: Compute scale2 = global_scale * (vecMax / 6.0)
    float SFValue = globalScale * (vecMax * reciprocal_approx_ftz(6.0f));

    // Step 4: Quantize SF to FP8 E4M3
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    uint8_t fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);

    // Step 5: Compute output scale = 1 / (SFValue / globalScale) = globalScale / SFValue
    float outputScale = SFValue != 0.0f
        ? reciprocal_approx_ftz(SFValue * reciprocal_approx_ftz(globalScale))
        : 0.0f;

    // Step 6: Write scale factor to global memory
    if (sfOut) { *sfOut = fp8SFVal; }

    // Step 7: Scale input values and convert to float2
    float2 fp2Vals[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
        fp2Vals[i].x *= outputScale;
        fp2Vals[i].y *= outputScale;
    }

    // Step 8: Convert 8 float values to 8 e2m1 values packed into uint32_t
    return fp32_vec_to_e2m1(fp2Vals);
}

// NVFP4 quantization kernel
// Each thread processes 8 bf16 values -> produces 8 FP4 values (1 uint32_t) + 1 scale factor
// Grid: blockIdx.x = row index, threadIdx.x = column tile index
__global__ void __launch_bounds__(512, 4)
nvfp4_quantize_kernel(
    int32_t numRows, int32_t numCols,
    const __nv_bfloat16* __restrict__ input,
    float globalScale,
    uint32_t* __restrict__ output,
    uint32_t* __restrict__ sfOutput)
{
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
        for (int colIdx = threadIdx.x; colIdx < numCols / FP4_ELTS_PER_THREAD; colIdx += blockDim.x) {
            // Vectorized load: 8 bf16 values = 16 bytes
            int64_t inOffset = rowIdx * (numCols / FP4_ELTS_PER_THREAD) + colIdx;
            Bf16PackedVec in_vec = reinterpret_cast<const Bf16PackedVec*>(input)[inOffset];

            // Get swizzled SF output pointer
            auto sfOut = get_sf_out_offset(rowIdx, colIdx, numCols, sfOutput);

            // Quantize and write
            output[inOffset] = cvt_bf16_to_fp4(in_vec, globalScale, sfOut);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// NVFP4 Quantize host function
/////////////////////////////////////////////////////////////////////////////////////////////////

void nvfp4_quantize_cuda(
    torch::Tensor in_bf16,     // [M, K] row-major, bfloat16
    double global_scale,       // scalar float32
    torch::Tensor out_fp4,     // [M, K//2] row-major, uint8 (packed FP4)
    torch::Tensor out_scale    // 1D, float8_e4m3fn, swizzled layout
) {
    // Input validation - dtype
    TORCH_CHECK(in_bf16.dtype() == torch::kBFloat16, "in_bf16 must be bfloat16");
    TORCH_CHECK(out_fp4.dtype() == torch::kUInt8, "out_fp4 must be uint8");
    TORCH_CHECK(out_scale.dtype() == torch::kFloat8_e4m3fn, "out_scale must be float8_e4m3fn");

    // Input validation - shape
    TORCH_CHECK(in_bf16.dim() == 2, "in_bf16 must be 2D");
    TORCH_CHECK(out_fp4.dim() == 2, "out_fp4 must be 2D");
    TORCH_CHECK(out_scale.dim() == 1, "out_scale must be 1D");

    const int M = in_bf16.size(0);
    const int K = in_bf16.size(1);

    TORCH_CHECK(K % 16 == 0, "K must be multiple of 16 (NVFP4 block size)");
    TORCH_CHECK(out_fp4.size(0) == M, "out_fp4 rows must match M");
    TORCH_CHECK(out_fp4.size(1) == K / 2, "out_fp4 cols must be K/2");

    // All tensors must be on CUDA and contiguous
    TORCH_CHECK(in_bf16.is_cuda(), "in_bf16 must be on CUDA");
    TORCH_CHECK(out_fp4.is_cuda(), "out_fp4 must be on CUDA");
    TORCH_CHECK(out_scale.is_cuda(), "out_scale must be on CUDA");
    TORCH_CHECK(in_bf16.is_contiguous(), "in_bf16 must be contiguous");
    TORCH_CHECK(out_fp4.is_contiguous(), "out_fp4 must be contiguous");
    TORCH_CHECK(out_scale.is_contiguous(), "out_scale must be contiguous");

    // Launch configuration
    int block_size = std::min(K / FP4_ELTS_PER_THREAD, 512);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, in_bf16.get_device());
    int num_blocks_per_sm = 2048 / block_size;
    int grid_size = std::min(M, sm_count * num_blocks_per_sm);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    nvfp4_quantize_kernel<<<grid_size, block_size, 0, stream>>>(
        M, K,
        reinterpret_cast<const __nv_bfloat16*>(in_bf16.data_ptr<at::BFloat16>()),
        static_cast<float>(global_scale),
        reinterpret_cast<uint32_t*>(out_fp4.data_ptr<uint8_t>()),
        reinterpret_cast<uint32_t*>(out_scale.data_ptr<c10::Float8_e4m3fn>())
    );
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// NVFP4 Quantize helpers
/////////////////////////////////////////////////////////////////////////////////////////////////

// Compute the swizzled scale factor tensor size for given dimensions
// Returns a tuple of (sfa_size, sfb_size)
std::tuple<int64_t, int64_t> nvfp4_scale_sizes(int M, int N, int K) {
    return {compute_sfa_size(M, N, K), compute_sfb_size(M, N, K)};
}
