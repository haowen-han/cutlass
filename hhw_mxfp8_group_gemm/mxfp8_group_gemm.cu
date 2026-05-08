/***************************************************************************************************
 * MXFP8 Grouped GEMM CUDA Extension for PyTorch (Blackwell SM100)
 *
 * Directly adapts the single-file structure of
 *   cutlass/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu
 * switching NVFP4 operands to MXFP8 (mx_float8_t<e4m3> data + ue8m0 block scale factors)
 * and the epilogue to a plain (non block-scaled) fp16 output.
 *
 * Python-layer interface still mirrors sglang's es_sm100_mxfp8_blockscaled_grouped_mm
 * (concatenated-token MoE convention). All per-expert pointer / stride / layout
 * metadata is assembled on the host (same as example 75) and uploaded in one shot,
 * rather than computed on-device.
 *
 * Tensor layout:
 *   A:   [sum_tokens, K]              float8_e4m3fn   row-major
 *   B:   [E, K, N]                    float8_e4m3fn   stride(1)==1 (K-major per expert)
 *   SFA: [sum(align(m_g,128)), K/32]  uint8  (F8_128x4 swizzled per-expert slice)
 *   SFB: [E, align(N,128)*K/32]       uint8  (F8_128x4 swizzled per expert)
 *   D:   [sum_tokens, N]              fp16             row-major
 *   problem_sizes:      [E, 3]  int32  (M_g, N, K)
 *   expert_offsets:     [E]     int32  prefix sum of m_g (A/D row offsets)
 *   blockscale_offsets: [E]     int32  prefix sum of align(m_g, 128) (SFA row offsets)
 **************************************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace hhw_mxfp8_group_gemm {

using namespace cute;

#define CUTLASS_CHECK(status)                                                         \
  {                                                                                   \
    cutlass::Status _err = (status);                                                  \
    TORCH_CHECK(                                                                      \
        _err == cutlass::Status::kSuccess,                                            \
        "cutlass error: ", cutlassGetStatusString(_err), " at line ", __LINE__);      \
  }

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// ---- Kernel configuration (mirrors example 75, retargeted to MXFP8 -> fp16) ----
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e4m3_t;
using ElementSF    = cutlass::float_ue8m0_t;
using ElementC     = cutlass::half_t;

using ElementA = cutlass::mx_float8_t<ElementInput>;
using LayoutA  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInput>::value;  // 16

using ElementB = cutlass::mx_float8_t<ElementInput>;
using LayoutB  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInput>::value;  // 16

using ElementD           = ElementC;
using LayoutC            = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;

using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape  = Shape<int32_t, int32_t, _1>;

struct MMA1SMConfig {
  using MmaTileShape     = Shape<_128, _128, _128>;
  using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    typename MMA1SMConfig::MmaTileShape, ClusterShape,
    Shape<_128, _64>,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC*, AlignmentC,
    ElementD, LayoutC*, AlignmentD,
    typename MMA1SMConfig::EpilogueSchedule>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA*, AlignmentA,
    ElementB, LayoutB*, AlignmentB,
    ElementAccumulator,
    typename MMA1SMConfig::MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    typename MMA1SMConfig::KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

// Helper to allocate a device tensor and async-copy host data into it.
static inline torch::Tensor upload_bytes(const void* host_ptr,
                                         size_t bytes,
                                         const torch::Device& device,
                                         cudaStream_t stream) {
  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(device);
  torch::Tensor t = torch::empty(static_cast<long>(bytes), opts);
  AT_CUDA_CHECK(cudaMemcpyAsync(t.data_ptr(), host_ptr, bytes,
                                cudaMemcpyHostToDevice, stream));
  return t;
}

static void run_mxfp8_group_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb,
    torch::Tensor& d,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets,
    cudaStream_t stream) {
  // ---------------------------------------------------------------------------
  // Step 0: 读取 expert 数量 E。
  // problem_sizes 形状是 [E, 3]（每行 (M_g, N, K)），所以 size(0) 就是 expert 数。
  // ---------------------------------------------------------------------------
  int const E = static_cast<int>(problem_sizes.size(0));

  // ---------------------------------------------------------------------------
  // Step 1: 把 device 上的元数据拉回 CPU。
  // CUTLASS Grouped GEMM 在 host 侧装参数时，需要知道每个 expert 的 (M, N, K)、
  // 前缀和偏移，才能计算每个 expert 对应的：
  //   - A/D 子块的行起点（expert_offsets 前缀和）
  //   - SFA 子块的行起点（blockscale_offsets，考虑 M 按 128 对齐）
  //   - 每个 expert 的 stride 和 ScaleFactor Layout
  // .to(torch::kCPU) 会自动在当前 stream 上同步一次，把 device 内存复制到 CPU。
  // 这是 "host 侧组装" 风格（example 75 的做法），和 sglang 的 "device kernel
  // 组装" 是两条路线——前者简单直观，后者避免了 D2H 同步。
  // ---------------------------------------------------------------------------
  auto ps_cpu = problem_sizes.to(torch::kCPU);
  auto eo_cpu = expert_offsets.to(torch::kCPU);
  auto bo_cpu = blockscale_offsets.to(torch::kCPU);
  const int32_t* ps = ps_cpu.data_ptr<int32_t>();
  const int32_t* eo = eo_cpu.data_ptr<int32_t>();
  const int32_t* bo = bo_cpu.data_ptr<int32_t>();

  // ---------------------------------------------------------------------------
  // Step 2: 为每个 expert 准备 host 侧的元数据数组。
  // CUTLASS 的 Grouped GEMM 需要一串 "per-expert" 的参数；我们先在 host 上算好
  // 每个 expert 的 ProblemShape / Stride / Layout，然后一次性传到 device。
  //
  //   problem_shapes_host : 每个 expert 的 (M, N, K)   —— 描述问题尺寸
  //   stride_A/B/D_host   : 每个 expert 的 cute stride —— 告诉 TMA 内存步长
  //   layout_SFA/SFB_host : 每个 expert 的 SF 张量布局 —— 描述 F8_128x4 swizzle
  // ---------------------------------------------------------------------------
  std::vector<UnderlyingProblemShape> problem_shapes_host(E);
  std::vector<StrideA>   stride_A_host(E);
  std::vector<StrideB>   stride_B_host(E);
  std::vector<StrideD>   stride_D_host(E);
  std::vector<LayoutSFA> layout_SFA_host(E);
  std::vector<LayoutSFB> layout_SFB_host(E);

  // 每个 expert 的数据指针也要单独算出来，CUTLASS 会按 (ptr_array[g], stride[g])
  // 的方式在 device 上迭代各 expert。
  std::vector<const ElementInput*> ptr_A_host(E);
  std::vector<const ElementInput*> ptr_B_host(E);
  std::vector<const ElementSF*>    ptr_SFA_host(E);
  std::vector<const ElementSF*>    ptr_SFB_host(E);
  std::vector<ElementD*>           ptr_D_host(E);

  // ---------------------------------------------------------------------------
  // Step 3: 拿到 5 个大张量的 base 指针（整块连续内存的起点）。
  // 后面只需要 base + 偏移 就能得到每个 expert 的起始地址。
  // 注意 PyTorch 的 float8_e4m3fn 底层就是一个 uint8，我们 reinterpret_cast 成
  // CUTLASS 的 float_e4m3_t*，字节布局一致，安全无拷贝。
  // ---------------------------------------------------------------------------
  const auto* a_base   = reinterpret_cast<const ElementInput*>(a.data_ptr());
  const auto* b_base   = reinterpret_cast<const ElementInput*>(b.data_ptr());
  const auto* sfa_base = reinterpret_cast<const ElementSF*>(sfa.data_ptr());
  const auto* sfb_base = reinterpret_cast<const ElementSF*>(sfb.data_ptr());
  auto*       d_base   = reinterpret_cast<ElementD*>(d.data_ptr());

  // ---------------------------------------------------------------------------
  // Step 4: 对每个 expert 计算它自己的一套 (shape, stride, layout, ptr)。
  // ---------------------------------------------------------------------------
  for (int i = 0; i < E; ++i) {
    // 4.1 取出当前 expert 的 (M_g, N, K)
    int M = ps[i * 3 + 0];
    int N = ps[i * 3 + 1];
    int K = ps[i * 3 + 2];
    problem_shapes_host[i] = {M, N, K};

    // 4.2 用 CUTLASS util 生成 "packed stride"：
    //   - A 是 RowMajor，形状 (M, K)，stride = (K, 1, 0)
    //   - B 是 ColumnMajor（每 expert 内 N×K 且 K-major），stride = (1, N, 0) 的 cute 版本
    //     (example 75 也是这样传 {N, K, 1})
    //   - D 是 RowMajor (M, N)，stride = (N, 1, 0)
    // 第三维的 "1" 是 batch stride 占位，Grouped GEMM 里每个 expert 自己就是一个 batch。
    stride_A_host[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    stride_B_host[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    stride_D_host[i] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    // 4.3 生成 Scale Factor 张量的 CUTLASS Layout。
    // MXFP8 的 scale 不是简单 [M, K/32] 行主序，而是 F8_128x4 swizzle：把 M 拆成 128
    // 的块，块内再把行按 (4, 32) 重排，并与 K/32 方向的 4-block 位置交换。
    // tile_atom_to_shape_SFA/SFB 会把 (M, N, K, 1) 翻译成这个 swizzled Layout，
    // Kernel 里 TMA 就按这个 Layout 去访存。
    // Python 侧的 _reorder_f8_128x4 helper 做的就是同一件事，二者配对才能对上。
    layout_SFA_host[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    layout_SFB_host[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    // 4.4 计算每个 expert 在 5 个大张量里的起始指针。
    // 关键区别：
    //   - A / D 沿 "tokens 维" 拼接，偏移用 expert_offsets (eo) —— 真实 token 行数前缀和
    //   - SFA 也沿 M 维拼接，但偏移用 blockscale_offsets (bo)  —— 按 align(M,128) 前缀和
    //     (因为 MXFP8 SF 需要 M 方向 128 对齐)
    //   - B 和 SFB 是 [E, ...] 的 3D 张量，每 expert 的 size 相同，按 i 乘以单 expert 体积
    ptr_A_host[i]   = a_base   + static_cast<size_t>(eo[i]) * K;
    ptr_B_host[i]   = b_base   + static_cast<size_t>(i)     * K * N;
    ptr_SFA_host[i] = sfa_base + static_cast<size_t>(bo[i]) * (K / 32);
    // SFB 单 expert 的物理字节数 = filter_zeros(LayoutSFB) 的 size。
    // filter_zeros 去掉 stride=0 的维度，只留真正占内存的那部分，所以 size(...) 就是
    // 这个 expert 自己这一片 Scale 张量占的 uint8 字节数。
    ptr_SFB_host[i] = sfb_base + static_cast<size_t>(i) *
                                    size(filter_zeros(layout_SFB_host[i]));
    ptr_D_host[i]   = d_base   + static_cast<size_t>(eo[i]) * N;
  }

  // ---------------------------------------------------------------------------
  // Step 5: 把所有 per-expert 的 host 数组一次性拷到 device。
  // CUTLASS kernel 在 device 上按 (ptr_array[g], stride_array[g]) 的方式取每个
  // expert 的参数，所以这些数组必须落在 device 内存里。
  // upload_bytes 是个简单包装：allocate 一个 uint8 tensor，cudaMemcpyAsync 拷贝过去。
  // 全部用异步拷贝，不会阻塞 host，且都排到同一个 stream 保序。
  // ---------------------------------------------------------------------------
  auto device = a.device();
  torch::Tensor problem_sizes_dev = upload_bytes(
      problem_shapes_host.data(), sizeof(UnderlyingProblemShape) * E, device, stream);
  torch::Tensor ptr_A_dev   = upload_bytes(ptr_A_host.data(),   sizeof(void*) * E, device, stream);
  torch::Tensor ptr_B_dev   = upload_bytes(ptr_B_host.data(),   sizeof(void*) * E, device, stream);
  torch::Tensor ptr_SFA_dev = upload_bytes(ptr_SFA_host.data(), sizeof(void*) * E, device, stream);
  torch::Tensor ptr_SFB_dev = upload_bytes(ptr_SFB_host.data(), sizeof(void*) * E, device, stream);
  torch::Tensor ptr_D_dev   = upload_bytes(ptr_D_host.data(),   sizeof(void*) * E, device, stream);
  torch::Tensor stride_A_dev   = upload_bytes(stride_A_host.data(),   sizeof(StrideA) * E, device, stream);
  torch::Tensor stride_B_dev   = upload_bytes(stride_B_host.data(),   sizeof(StrideB) * E, device, stream);
  torch::Tensor stride_D_dev   = upload_bytes(stride_D_host.data(),   sizeof(StrideD) * E, device, stream);
  torch::Tensor layout_SFA_dev = upload_bytes(layout_SFA_host.data(), sizeof(LayoutSFA) * E, device, stream);
  torch::Tensor layout_SFB_dev = upload_bytes(layout_SFB_host.data(), sizeof(LayoutSFB) * E, device, stream);

  // ---------------------------------------------------------------------------
  // Step 6: 填 KernelHardwareInfo —— 告诉 CUTLASS 目标 GPU 的信息。
  //   - device_id, sm_count : 用于 persistent kernel 调度
  //   - cluster_shape       : Blackwell 支持 thread-block cluster，需要一个首选形状
  //   - cluster_shape_fallback : 不满足首选时的降级形状（比如 SM 数不够整除）
  // (1,4,1) 表示 cluster 内 N 方向 4 个 block 组合成一个协作单元。
  // ---------------------------------------------------------------------------
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  hw_info.cluster_shape          = dim3(1, 4, 1);
  hw_info.cluster_shape_fallback = dim3(1, 2, 1);

  // ---------------------------------------------------------------------------
  // Step 7: 组装 CUTLASS Gemm::Arguments 结构。
  // 这是 CUTLASS 3.x 的统一入参，分为 4 段：
  //   (a) mode + problem shape
  //   (b) Mainloop 参数 (A/B/SFA/SFB 指针+stride+layout)
  //   (c) Epilogue 参数 (fusion_args, C, D)
  //   (d) hw_info
  // ---------------------------------------------------------------------------
  typename Gemm::Arguments arguments{
      // (a) 设成 kGrouped 模式，后跟 { group 数量, device 端 problem shape, host 端 problem shape }
      // host 端 shape 让 CUTLASS 在 launch 前做调度分析（可选；传 nullptr 也行）
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {E,
       reinterpret_cast<UnderlyingProblemShape*>(problem_sizes_dev.data_ptr()),
       problem_shapes_host.data()},
      // (b) Mainloop：块标量 GEMM 需要 4 组指针 (A, B, SFA, SFB) + 对应 stride/layout
      {reinterpret_cast<const ElementInput**>(ptr_A_dev.data_ptr()),
       reinterpret_cast<StrideA*>(stride_A_dev.data_ptr()),
       reinterpret_cast<const ElementInput**>(ptr_B_dev.data_ptr()),
       reinterpret_cast<StrideB*>(stride_B_dev.data_ptr()),
       reinterpret_cast<const ElementSF**>(ptr_SFA_dev.data_ptr()),
       reinterpret_cast<LayoutSFA*>(layout_SFA_dev.data_ptr()),
       reinterpret_cast<const ElementSF**>(ptr_SFB_dev.data_ptr()),
       reinterpret_cast<LayoutSFB*>(layout_SFB_dev.data_ptr())},
      // (c) Epilogue：没有 C 矩阵（设 nullptr），只有 D 输出。fusion_args 先用默认{},
      // 下面再写入 alpha=1, beta=0（等价于 D = acc，不读 C）
      {{},
       nullptr, nullptr,
       reinterpret_cast<ElementD**>(ptr_D_dev.data_ptr()),
       reinterpret_cast<StrideD*>(stride_D_dev.data_ptr())},
      // (d) 硬件信息
      hw_info};

  // 默认的 LinearCombination epilogue：D = alpha * acc + beta * C。
  // 设 alpha=1, beta=0 就是 "把累加结果直接写出"，不需要 C。
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta  = 0.0f;

  // ---------------------------------------------------------------------------
  // Step 8: 实例化、预检、分配 workspace、初始化、发射 kernel。
  // CUTLASS 标准三部曲：can_implement -> initialize -> run
  //   - can_implement : 静态 + 动态地检查 Arguments 是否被这个 kernel 类型支持
  //   - initialize    : 把 Arguments 编译成 device-side Params，并绑定 workspace
  //   - run           : 实际启动 kernel；enable_pdl=true 开启 Programmatic Dependent Launch
  //                     （让这个 kernel 和上一次 copy kernel 能在 SM 之间提前重叠）
  // ---------------------------------------------------------------------------
  Gemm gemm;
  // 8.1 预检：kernel 支不支持这组 Arguments（对齐/形状/schedule 约束等）
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // 8.2 查询 kernel 需要的临时 workspace 字节数，然后在 device 上分配一块。
  //     workspace 里通常放 tile scheduler 的原子计数器、cluster 同步变量等。
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device);
  torch::Tensor workspace = torch::empty(static_cast<long>(workspace_size), opts_u8);

  // 8.3 把 Arguments + workspace 装进 kernel，并在当前 stream 上排入 "初始化" 操作
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  // 8.4 真正启动 GEMM kernel。enable_pdl=true 让它与前一个 kernel 更好地重叠。
  CUTLASS_CHECK(gemm.run(stream, /*cuda_adapter=*/nullptr, /*enable_pdl=*/true));
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

}  // namespace hhw_mxfp8_group_gemm

void mxfp8_group_gemm(
    torch::Tensor& d,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  TORCH_CHECK(a.dim() == 2, "a must be a 2D tensor of shape (num_tokens, k)");
  TORCH_CHECK(b.dim() == 3, "b must be a 3D tensor of shape (num_experts, k, n)");
  TORCH_CHECK(problem_sizes.dim() == 2 && problem_sizes.size(1) == 3,
              "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0),
              "num_experts mismatch between problem_sizes and expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");
  TORCH_CHECK(expert_offsets.dtype() == torch::kInt32, "expert_offsets must be int32");
  TORCH_CHECK(blockscale_offsets.dtype() == torch::kInt32, "blockscale_offsets must be int32");
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn, "a must be float8_e4m3fn");
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn, "b must be float8_e4m3fn");
  TORCH_CHECK(d.dtype() == torch::kFloat16, "output dtype must be fp16");
  TORCH_CHECK(a.size(1) == b.size(1) && a.size(1) % 128 == 0, "K must be 128-aligned");
  TORCH_CHECK(b.size(2) % 128 == 0, "N must be 128-aligned");
  TORCH_CHECK(a.stride(1) == 1, "a must be row-major");
  TORCH_CHECK(b.stride(1) == 1, "b must be K-major per expert (stride(1) == 1)");

  auto stream = at::cuda::getCurrentCUDAStream();
  hhw_mxfp8_group_gemm::run_mxfp8_group_gemm(
      a, b, sfa, sfb, d, problem_sizes, expert_offsets, blockscale_offsets, stream);
#else
  TORCH_CHECK(false, "mxfp8_group_gemm requires CUTLASS_ARCH_MMA_SM100_SUPPORTED");
#endif
}
