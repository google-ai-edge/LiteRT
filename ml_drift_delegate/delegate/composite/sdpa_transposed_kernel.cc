// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"

namespace litert::ml_drift {

namespace {

// Launch and split-K parameters of the Flash-Decode kernels. They depend on the
// GPU, so they are chosen per device by `GetFlashDecodeTuning`.
struct FlashDecodeTuning {
  // Number of SIMD groups (waves of 32 lanes) cooperating on one query head in
  // the wave-SIMD kernel. At most 32, so wave 0 can reduce one value per wave.
  int num_simd_groups = 16;
  // Number of work items per work group in the work-group reduction kernel.
  int threads = 256;
  // Work items that combine the per-work-item values in the first stage of a
  // work-group reduction. Every work item then does the second stage itself.
  int reduce_threads = 32;
  // Maximum number of query heads sharing one K/V head to process per work
  // group in the work-group reduction kernel so per-head accumulators stay in
  // registers. Larger GQA groups are partitioned across multiple work groups.
  int max_heads_per_work_group = 8;
  // The keys of every query token and K/V head group are split across work
  // groups until there are at least `target_work_groups` work groups, but
  // every split keeps at least `min_split_keys` keys.
  int target_work_groups = 40;
  int min_split_keys = 256;
  // Local memory budget in bytes of the work-group reduction kernel. Apple GPUs
  // reject compute pipelines that use more than 32 KB of threadgroup memory,
  // and most OpenCL devices report at least 32 KB of local memory. WebGPU
  // devices report their limit (see `GetFlashDecodeTuning`).
  int local_memory_bytes = 32 * 1024;
};

// Returns the Flash-Decode tuning for `gpu_info`.
//
// The defaults were measured on Adreno 830 (work-group reduction kernel): with
// Qwen3 0.6B (8 K/V heads, 1280 cache entries), 5 splits of 256 keys read the
// KV cache at 58 GB/s, compared with 43 GB/s for one work group per K/V head.
// The wave-SIMD kernel uses 16 waves on Apple GPUs.
// TODO(b/553029558): Tune for Mali, Intel, Nvidia and AMD GPUs.
FlashDecodeTuning GetFlashDecodeTuning(const ::ml_drift::GpuInfo& gpu_info) {
  FlashDecodeTuning tuning;
  const int max_work_group_size = gpu_info.GetMaxWorkGroupTotalSize();
  if (max_work_group_size > 0) {
    // Keep both kernels within the work-group size limit of the device.
    while (tuning.threads > tuning.reduce_threads &&
           tuning.threads > max_work_group_size) {
      tuning.threads /= 2;
    }
    while (tuning.num_simd_groups > 1 &&
           tuning.num_simd_groups * 32 > max_work_group_size) {
      tuning.num_simd_groups /= 2;
    }
  }
  if (gpu_info.IsApiWebGpu() &&
      gpu_info.webgpu_info.max_compute_workgroup_storage_size > 0) {
    // WebGPU validates the workgroup storage of a pipeline against the limit
    // of the device: 16 KB by default, or the adapter limit when requested.
    tuning.local_memory_bytes =
        gpu_info.webgpu_info.max_compute_workgroup_storage_size;
  }
  // TODO(b/553029558): Use the local memory size of OpenCL
  // (`CL_DEVICE_LOCAL_MEM_SIZE`), Metal (`maxThreadgroupMemoryLength`) and
  // Vulkan (`maxComputeSharedMemorySize`) devices once `GpuInfo` reports it.
  return tuning;
}

// Keys scored by each work item per chunk of the work-group Flash-Decode
// kernel: at most as many as keep the probabilities of a chunk within 8 KB of
// local memory.
int MaxFlashDecodeKeysPerThread(int heads, int threads) {
  return std::clamp(2048 / (heads * threads), 1, 4);
}

// Upper bound of the local memory in bytes used by the work-group Flash-Decode
// kernel for `heads` query heads per work group and `slices` channel slices
// (head_dim / 4), assuming the largest chunk of probabilities: the float4 query
// slices, the probabilities of a chunk, the two reduction stages and the float4
// output accumulators (see `GenerateWorkGroupFlashDecodeCode`).
int FlashDecodeLocalMemoryBytes(const FlashDecodeTuning& tuning, int heads,
                                int slices) {
  constexpr int kFloatBytes = static_cast<int>(sizeof(float));
  const int threads = tuning.threads;
  const int chunk = threads * MaxFlashDecodeKeysPerThread(heads, threads);
  const int q_local = heads * slices * 4 * kFloatBytes;
  const int p_local = heads * chunk * kFloatBytes;
  const int red_local = heads * threads * kFloatBytes;
  const int red2_local = heads * tuning.reduce_threads * kFloatBytes;
  const int acc_local = threads * 4 * kFloatBytes;
  return q_local + p_local + red_local + red2_local + acc_local;
}

// Resolves `#pragma OPENCL EXTENSION ucl_wave_simd: enable` and the wave-SIMD
// intrinsics (`ucl::WaveSum`, `ucl::WaveMax`, `ucl::WaveShuffleXor`) to the
// target shading language, following `ResolveWaveMemory` in
// `wave_memory_util.cc`.
void ResolveWaveSimd(const ::ml_drift::GpuInfo& gpu_info, std::string* code) {
  const std::string kExtDecl = "#pragma OPENCL EXTENSION ucl_wave_simd: enable";
  const size_t ext_pos = code->find(kExtDecl);
  if (ext_pos != std::string::npos) {
    std::string patch;
    if (gpu_info.IsApiOpenCl()) {
      if (gpu_info.opencl_info.cl_version ==
              ::ml_drift::OpenClVersion::kCl2_0 ||
          gpu_info.SupportsExtension("cl_khr_subgroups")) {
        patch = "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
      } else if (gpu_info.SupportsExtension("cl_intel_subgroups")) {
        patch = "#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n";
      }
      if (gpu_info.SupportsExtension("cl_khr_subgroup_extended_types")) {
        absl::StrAppend(
            &patch,
            "#pragma OPENCL EXTENSION cl_khr_subgroup_extended_types : "
            "enable\n");
      }
      if (gpu_info.SupportsExtension("cl_khr_subgroup_shuffle")) {
        absl::StrAppend(
            &patch,
            "#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable\n");
      }
    } else if (gpu_info.IsGlsl()) {
      patch =
          "#extension GL_KHR_shader_subgroup_arithmetic : require\n"
          "#extension GL_KHR_shader_subgroup_shuffle : require\n";
      if (gpu_info.IsGlslSupportsExplicitFp16()) {
        absl::StrAppend(
            &patch,
            "#extension GL_EXT_shader_subgroup_extended_types_float16 : "
            "require\n");
      }
    } else if (gpu_info.IsApiWebGpu()) {
      patch = "enable subgroups;\n";
      if (gpu_info.webgpu_info.supports_fp16) {
        absl::StrAppend(&patch, "enable f16;\n");
      }
    }
    code->replace(ext_pos, kExtDecl.size(), patch);
  }

  absl::string_view wave_sum = "sub_group_reduce_add";
  absl::string_view wave_max = "sub_group_reduce_max";
  absl::string_view wave_shuffle_xor = "sub_group_shuffle_xor";
  if (gpu_info.IsApiMetal()) {
    wave_sum = "simd_sum";
    wave_max = "simd_max";
    wave_shuffle_xor = "simd_shuffle_xor";
  } else if (gpu_info.IsGlsl() || gpu_info.IsApiWebGpu()) {
    wave_sum = "subgroupAdd";
    wave_max = "subgroupMax";
    wave_shuffle_xor = "subgroupShuffleXor";
  }
  absl::StrReplaceAll({{"ucl::WaveSum", wave_sum},
                       {"ucl::WaveMax", wave_max},
                       {"ucl::WaveShuffleXor", wave_shuffle_xor}},
                      code);
}

// Whether the GPU supports fixed 32-lane wave-SIMD vector reductions
// (`ucl::WaveSum` on `half4` / `ucl::WaveMax`).
//
// Limited to Metal: OpenCL `sub_group_reduce_*` has no vector overloads and
// Adreno / Mali waves are not 32 lanes wide; WebGPU and Vulkan need the
// subgroup feature and a fixed subgroup size of 32, which `GpuInfo` does not
// guarantee.
// TODO(b/553029558): Support WebGPU / Vulkan when the device reports a fixed
// 32-lane subgroup size, and OpenCL with per-component reductions.
bool SupportsWave32Vec4SimdReduce(const ::ml_drift::GpuInfo& gpu_info) {
  return gpu_info.IsWaveSizeEqualTo32() && gpu_info.IsApiMetal();
}

// Whether the GPU supports the 8x8 wave-matrix MMA prefill kernel.
// TODO(b/553029558): Support more backends (e.g. Vulkan / WebGPU cooperative
// matrices and OpenCL subgroup matrix extensions).
bool SupportsWaveMatrixPrefill(const ::ml_drift::GpuInfo& gpu_info) {
  return gpu_info.IsApple() && gpu_info.IsApiMetal();
}

// Execution and work-group configuration for `FusedFlashDecodeSdpaOp`.
struct FlashDecodeConfig {
  // Whether to use the 32-lane wave-SIMD reduction kernel (`ucl::WaveSum` /
  // `ucl::WaveMax` across `tuning.num_simd_groups` waves of 32 lanes) or the
  // work-group reduction kernel with GQA head grouping and split-K.
  bool use_wave_simd = false;
  // Device-dependent launch parameters.
  FlashDecodeTuning tuning;
  // Query heads per work group (they share one K/V head).
  int heads = 1;
  // Number of work groups that split the keys of each query token and head
  // group. With more than one split, `FusedFlashDecodeCombineOp` combines the
  // partial results.
  int splits = 1;
};

FlashDecodeConfig GetFlashDecodeConfig(const ::ml_drift::GpuInfo& gpu_info,
                                       const ::ml_drift::BHWC& q_shape,
                                       const ::ml_drift::BHWC& k_shape) {
  FlashDecodeConfig config;
  config.tuning = GetFlashDecodeTuning(gpu_info);
  if (SupportsWave32Vec4SimdReduce(gpu_info) && q_shape.c == 128 &&
      config.tuning.num_simd_groups > 1) {
    config.use_wave_simd = true;
    return config;
  }
  if (k_shape.h <= 0 || q_shape.h % k_shape.h != 0) {
    return config;
  }
  // Process as many query heads of the GQA group per work group as divide the
  // group and keep the kernel within the local memory budget of the device.
  // With head dim 512 (128 slices), 8 heads need 37,888 B but 4 heads only
  // 25,088 B. If not even one head fits, `heads` stays 1 and
  // `IsSupportedFlashDecode` rejects the config, so the op falls back to the
  // decomposed graph.
  const int group_size = q_shape.h / k_shape.h;
  const int slices = q_shape.c / 4;
  for (int h = std::min(group_size, config.tuning.max_heads_per_work_group);
       h >= 1; --h) {
    if (group_size % h == 0 &&
        FlashDecodeLocalMemoryBytes(config.tuning, h, slices) <=
            config.tuning.local_memory_bytes) {
      config.heads = h;
      break;
    }
  }
  const int work_groups = (q_shape.h / config.heads) * q_shape.w;
  const int wanted_splits =
      (config.tuning.target_work_groups + work_groups - 1) / work_groups;
  const int max_splits =
      std::max(1, k_shape.w / config.tuning.min_split_keys);
  config.splits = std::clamp(wanted_splits, 1, max_splits);
  return config;
}

// Whether the fused Flash-Decode kernel supports these shapes. K and V must be
// BUFFER tensors in the packed layout written by `odml.cache_update`, the
// query heads must be a multiple of the K/V heads, and a mask must be shared by
// all heads with one row, or one row per query token, that covers every cache
// entry.
bool IsSupportedFlashDecode(const ::ml_drift::GpuInfo& gpu_info,
                            const ::ml_drift::BHWC& q_shape,
                            const ::ml_drift::TensorDescriptor& k_desc,
                            const ::ml_drift::TensorDescriptor& v_desc,
                            const ::ml_drift::TensorDescriptor* mask_desc) {
  if (k_desc.GetStorageType() != ::ml_drift::TensorStorageType::kBuffer ||
      v_desc.GetStorageType() != ::ml_drift::TensorStorageType::kBuffer) {
    return false;
  }
  const ::ml_drift::BHWC k_shape = k_desc.GetBHWCShape();
  if (q_shape.c % 4 != 0 || k_shape.h <= 0 || q_shape.h < k_shape.h ||
      q_shape.h % k_shape.h != 0 || k_shape.w % 4 != 0) {
    return false;
  }
  if (mask_desc != nullptr) {
    const ::ml_drift::BHWC mask_shape = mask_desc->GetBHWCShape();
    if (mask_shape.h != 1 || (mask_shape.w != 1 && mask_shape.w != q_shape.w) ||
        mask_shape.c < k_shape.w) {
      return false;
    }
  }
  const FlashDecodeConfig config =
      GetFlashDecodeConfig(gpu_info, q_shape, k_shape);
  if (config.use_wave_simd) {
    return true;
  }
  const int slices = q_shape.c / 4;
  const int threads = config.tuning.threads;
  return slices <= threads && threads % slices == 0 &&
         FlashDecodeLocalMemoryBytes(config.tuning, config.heads, slices) <=
             config.tuning.local_memory_bytes;
}

// TODO(b/552147487): Benchmark the kernel on Nvidia GPUs.
class FusedFlashDecodeSdpaOp : public ::ml_drift::GPUOperation {
 public:
  FusedFlashDecodeSdpaOp(const ::ml_drift::int3& work_group_size,
                         int head_groups, int splits = 1)
      : head_groups_(head_groups), splits_(splits) {
    work_group_size_ = work_group_size;
  }

  ::ml_drift::int3 GetGridSize() const override {
    if (work_group_size_.z > 1) {
      return ::ml_drift::int3(src_[0]->Width(),
                              head_groups_ * work_group_size_.y,
                              work_group_size_.z);
    }
    return ::ml_drift::int3(work_group_size_.x, head_groups_,
                            src_[0]->Width() * splits_);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  FusedFlashDecodeSdpaOp(FusedFlashDecodeSdpaOp&&) = default;
  FusedFlashDecodeSdpaOp& operator=(FusedFlashDecodeSdpaOp&&) = default;
  FusedFlashDecodeSdpaOp(const FusedFlashDecodeSdpaOp&) = delete;
  FusedFlashDecodeSdpaOp& operator=(const FusedFlashDecodeSdpaOp&) = delete;

 private:
  int head_groups_ = 1;
  int splits_ = 1;
};

// Combines the per-split partial results of `FusedFlashDecodeSdpaOp` when
// `splits > 1`. One work item handles one channel slice of one query head and
// token.
class FusedFlashDecodeCombineOp : public ::ml_drift::GPUOperation {
 public:
  FusedFlashDecodeCombineOp(int slices, int q_heads, int splits)
      : slices_(slices), q_heads_(q_heads), splits_(splits) {}

  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(slices_, q_heads_, src_[0]->Width() / splits_);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  FusedFlashDecodeCombineOp(FusedFlashDecodeCombineOp&&) = default;
  FusedFlashDecodeCombineOp& operator=(FusedFlashDecodeCombineOp&&) = default;
  FusedFlashDecodeCombineOp(const FusedFlashDecodeCombineOp&) = delete;
  FusedFlashDecodeCombineOp& operator=(const FusedFlashDecodeCombineOp&) =
      delete;

 private:
  int slices_ = 1;
  int q_heads_ = 1;
  int splits_ = 1;
};

// Generates the shader code that bounds `active_tokens` by the causal position
// of query `X` when the BOOL causal mask was pruned (`attr.is_causal` without a
// mask tensor). Expects `active_tokens` and, with `has_param`,
// `has_active_tokens` (whether `params` held an in-range active token count) to
// be defined.
//
// Multi-token queries clamp to `q_start + X + 1`. `q_start` is read from
// `param[0]` when it is in range (0 < q_start < active_tokens); otherwise the
// queries are assumed to be the last `q.Width()` active tokens.
//
// A single-token decode query is the newest token, so the active token count
// is already its causal bound. On a sliding-window ring-buffer cache,
// `param[0]` is the ring write offset (step % window) rather than an absolute
// position, and clamping to it would drop visible keys once the ring wraps. So
// `param[0]` only bounds a single query when the active token count is missing
// or out of range, and only if `param[0]` itself is in range
// (0 < param[0] < cache_size).
std::string GenerateImplicitCausalBoundCode(bool has_param,
                                            bool is_single_query) {
  if (is_single_query) {
    if (!has_param) return "";
    return R"(
  if (!has_active_tokens) {
    float4 p_start_vec = ucl::Convert<float4>(args.params.Read(0, 0, 0, 0));
    int start_val = (int)p_start_vec.x;
    if (start_val > 0 && start_val < active_tokens) {
      active_tokens = start_val + 1;
    }
  }
)";
  }
  if (!has_param) {
    // Without `params`, the query tokens are the last `q.Width()` entries.
    return R"(
  {
    int q_start = max(0, active_tokens - args.q.Width());
    active_tokens = min(active_tokens, q_start + X + 1);
  }
)";
  }
  return R"(
  {
    float4 p_start_vec = ucl::Convert<float4>(args.params.Read(0, 0, 0, 0));
    int start_val = (int)p_start_vec.x;
    int q_start = (start_val > 0 && start_val < active_tokens)
                      ? start_val
                      : max(0, active_tokens - args.q.Width());
    active_tokens = min(active_tokens, q_start + X + 1);
  }
)";
}

// Generates the wave-SIMD (32-lane) reduction Flash-Decode kernel source.
//
// Execution model:
// - `num_simd_groups` waves per work group (32 lanes per wave, e.g. 16 waves =
//   512 work items). Launch shape: (1, 32, num_simd_groups).
// - Grid: X = sequence index (1), Y = query head index.
// - Within each wave, the 32 lanes compute channel-parallel dot products
//   (dot(q_slice, k) over head_dim / 4 = 32 slices) reduced via `ucl::WaveSum`.
// - Across the waves, the KV cache sequence length is partitioned into
//   chunks of 16 keys (4 vector loads of 4 keys). Each wave maintains a
//   local online softmax (m_prev = running max, l_prev = running exp sum,
//   out_acc = accumulator).
// - Cross-wave reduction: wave 0 reduces s_m, s_l, and s_acc in local memory
//   using `ucl::WaveMax` and `ucl::WaveSum` to produce the normalized output.
//
// Masked keys and keys past `active_tokens` get the score -10000 (a boolean
// mask) or a score around -10000 (an additive mask); any score below -9000
// contributes a probability of exactly 0, so waves whose keys are all masked
// (e.g. a mask hiding a range below `active_tokens`) do not add
// `exp2(0) = 1` weights.
std::string GenerateWaveSimdFlashDecodeCode(
    int num_simd_groups, int slices, int gqa_ratio, int k_stride_head,
    int k_stride_slice, int v_stride_head, int v_stride_s, bool has_mask,
    bool is_bool_mask, int mask_width, bool has_param, bool is_causal,
    bool is_single_query, bool has_softcap, bool is_flattened_dst) {
  const int v_stride_2s = v_stride_s * 2;
  const int v_stride_3s = v_stride_s * 3;
  const int v_stride_4s = v_stride_s * 4;
  const std::string mask_row = mask_width == 1 ? "0" : "X";
  std::string op_code = absl::StrCat(R"(
#pragma OPENCL EXTENSION ucl_wave_simd: enable
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGroupId<1>();
  int simd_id = ucl::GetLocalId<2>();
  int tid = ucl::GetLocalId<1>();

  __local float s_m[$NSG];
  __local float s_l[$NSG];
  __local float s_w[$NSG];
  __local half4 s_acc[)",
                                     slices, R"(][$NSG];

  if (simd_id == 0 && tid < $NSG) {
    s_m[tid] = -10000.0f;
    s_l[tid] = 0.0f;
  }

  int active_tokens = args.cache_size;
)");

  if (has_param) {
    op_code += R"(
  int param_slice = args.src_end_ch_index / 4;
  int param_comp = args.src_end_ch_index % 4;
  float4 p_vec = ucl::Convert<float4>(args.params.Read(0, 0, param_slice, 0));
  float p_raw = (param_comp == 0) ? p_vec.x : ((param_comp == 1) ? p_vec.y : ((param_comp == 2) ? p_vec.z : p_vec.w));
  int param_val = (int)p_raw;
  bool has_active_tokens = param_val > 0 && param_val <= args.cache_size;
  if (has_active_tokens) {
    active_tokens = param_val;
  }
)";
  }
  if (!has_mask && is_causal) {
    op_code += GenerateImplicitCausalBoundCode(has_param, is_single_query);
  }

  absl::StrAppend(&op_code, R"(
  int total_chunks = (active_tokens + 3) / 4;
  int chunks_per_simd = (total_chunks + $NSG - 1) / $NSG;
  int chunk_start = simd_id * chunks_per_simd;
  int chunk_end = min(total_chunks, chunk_start + chunks_per_simd);
  int safe_chunk_end = max(chunk_start, min(chunk_end, (active_tokens / 16) * 4));

  // Note: This Flash-Decode SDPA kernel is optimized for float16 / half precision.
  half4 q_slice = (tid < args.slices) ? ucl::Convert<half4>(args.q.Read(X, Y, tid)) : half4(0.0h);
  half m_prev = -10000.0h;
  half l_prev = 0.0h;
  // Note: half4 output accumulator for maximum register efficiency on mobile GPUs.
  half4 out_acc = half4(0.0h);
  half inv_ln2 = 1.4426950408889634h;

  int kv_head = )",
                  (gqa_ratio > 1 ? absl::StrCat("Y / ", gqa_ratio) : "Y"), R"(;
  int k_base_head = kv_head * )",
                  k_stride_head, R"( + tid * )", k_stride_slice, R"(;
  int v_base_head = kv_head * )",
                  v_stride_head, R"( + tid * 4;

  int chunk = chunk_start;
  int k_idx = k_base_head + chunk * 4;
  int v_idx = v_base_head + chunk * )",
                  v_stride_s, R"(;

  for (; chunk + 3 < safe_chunk_end; chunk += 4) {
    half4 k0 = ucl::Convert<half4>(args.k.Read(k_idx + 0));
    half4 k1 = ucl::Convert<half4>(args.k.Read(k_idx + 1));
    half4 k2 = ucl::Convert<half4>(args.k.Read(k_idx + 2));
    half4 k3 = ucl::Convert<half4>(args.k.Read(k_idx + 3));
    half4 d0 = ucl::WaveSum(half4(dot(q_slice, k0), dot(q_slice, k1), dot(q_slice, k2), dot(q_slice, k3)));

    half4 k4 = ucl::Convert<half4>(args.k.Read(k_idx + 4));
    half4 k5 = ucl::Convert<half4>(args.k.Read(k_idx + 5));
    half4 k6 = ucl::Convert<half4>(args.k.Read(k_idx + 6));
    half4 k7 = ucl::Convert<half4>(args.k.Read(k_idx + 7));
    half4 d1 = ucl::WaveSum(half4(dot(q_slice, k4), dot(q_slice, k5), dot(q_slice, k6), dot(q_slice, k7)));

    half4 k8 = ucl::Convert<half4>(args.k.Read(k_idx + 8));
    half4 k9 = ucl::Convert<half4>(args.k.Read(k_idx + 9));
    half4 k10 = ucl::Convert<half4>(args.k.Read(k_idx + 10));
    half4 k11 = ucl::Convert<half4>(args.k.Read(k_idx + 11));
    half4 d2 = ucl::WaveSum(half4(dot(q_slice, k8), dot(q_slice, k9), dot(q_slice, k10), dot(q_slice, k11)));

    half4 k12 = ucl::Convert<half4>(args.k.Read(k_idx + 12));
    half4 k13 = ucl::Convert<half4>(args.k.Read(k_idx + 13));
    half4 k14 = ucl::Convert<half4>(args.k.Read(k_idx + 14));
    half4 k15 = ucl::Convert<half4>(args.k.Read(k_idx + 15));
    half4 d3 = ucl::WaveSum(half4(dot(q_slice, k12), dot(q_slice, k13), dot(q_slice, k14), dot(q_slice, k15)));
)");

  if (has_softcap) {
    op_code += R"(
    d0 = (half4)args.softcap * tanh(d0 / (half4)args.softcap);
    d1 = (half4)args.softcap * tanh(d1 / (half4)args.softcap);
    d2 = (half4)args.softcap * tanh(d2 / (half4)args.softcap);
    d3 = (half4)args.softcap * tanh(d3 / (half4)args.softcap);
)";
  }

  if (has_mask) {
    op_code += R"(
    half4 m_vec0 = ucl::Convert<half4>(args.mask.Read($MROW, 0, chunk + 0));
    half4 m_vec1 = ucl::Convert<half4>(args.mask.Read($MROW, 0, chunk + 1));
    half4 m_vec2 = ucl::Convert<half4>(args.mask.Read($MROW, 0, chunk + 2));
    half4 m_vec3 = ucl::Convert<half4>(args.mask.Read($MROW, 0, chunk + 3));
)";
    if (is_bool_mask) {
      op_code += R"(
    d0 = select(d0, half4(-10000.0h), m_vec0 < 0.5h);
    d1 = select(d1, half4(-10000.0h), m_vec1 < 0.5h);
    d2 = select(d2, half4(-10000.0h), m_vec2 < 0.5h);
    d3 = select(d3, half4(-10000.0h), m_vec3 < 0.5h);
)";
    } else {
      op_code += R"(
    d0 += m_vec0;
    d1 += m_vec1;
    d2 += m_vec2;
    d3 += m_vec3;
)";
    }
  }

  absl::StrAppend(&op_code, R"(
    // Online softmax tracking:
    // m_prev: running maximum of dot-product scores (for numerical stability).
    // l_prev: running sum of exponentiated scores (normalization denominator).
    // alpha:  rescaling factor (exp2(m_prev - m_new)) for previous accumulators when a new max is found.
    // p0..p3: unnormalized attention probabilities (exp2(d - m_new)) for keys in this chunk.
    half4 m_c01 = max(max(d0, d1), max(d2, d3));
    half m_chunk = max(max(m_c01.x, m_c01.y), max(m_c01.z, m_c01.w));
    half m_new = max(m_prev, m_chunk);
    half alpha = exp2((m_prev - m_new) * inv_ln2);

    half4 p0 = exp2((d0 - (half4)m_new) * (half4)inv_ln2);
    half4 p1 = exp2((d1 - (half4)m_new) * (half4)inv_ln2);
    half4 p2 = exp2((d2 - (half4)m_new) * (half4)inv_ln2);
    half4 p3 = exp2((d3 - (half4)m_new) * (half4)inv_ln2);
)");
  if (has_mask) {
    // When every key seen so far is masked, m_new is itself a masked score and
    // exp2(d - m_new) would be 1 for the masked keys.
    op_code += R"(
    p0 = select(p0, half4(0.0h), d0 < half4(-9000.0h));
    p1 = select(p1, half4(0.0h), d1 < half4(-9000.0h));
    p2 = select(p2, half4(0.0h), d2 < half4(-9000.0h));
    p3 = select(p3, half4(0.0h), d3 < half4(-9000.0h));
)";
  }
  absl::StrAppend(&op_code, R"(
    half4 p_sum01 = (p0 + p1) + (p2 + p3);
    half p_sum = (p_sum01.x + p_sum01.y) + (p_sum01.z + p_sum01.w);
    l_prev = fma(l_prev, alpha, p_sum);
    m_prev = m_new;

    half4 v0 = ucl::Convert<half4>(args.v.Read(v_idx + 0));
    half4 v1 = ucl::Convert<half4>(args.v.Read(v_idx + 1));
    half4 v2 = ucl::Convert<half4>(args.v.Read(v_idx + 2));
    half4 v3 = ucl::Convert<half4>(args.v.Read(v_idx + 3));
    half4 v_acc0 = fma((half4)p0.x, v0, fma((half4)p0.y, v1, fma((half4)p0.z, v2, (half4)p0.w * v3)));

    int v1_base = v_idx + )",
                  v_stride_s, R"(;
    half4 v4 = ucl::Convert<half4>(args.v.Read(v1_base + 0));
    half4 v5 = ucl::Convert<half4>(args.v.Read(v1_base + 1));
    half4 v6 = ucl::Convert<half4>(args.v.Read(v1_base + 2));
    half4 v7 = ucl::Convert<half4>(args.v.Read(v1_base + 3));
    half4 v_acc1 = fma((half4)p1.x, v4, fma((half4)p1.y, v5, fma((half4)p1.z, v6, (half4)p1.w * v7)));

    int v2_base = v_idx + )",
                  v_stride_2s, R"(;
    half4 v8 = ucl::Convert<half4>(args.v.Read(v2_base + 0));
    half4 v9 = ucl::Convert<half4>(args.v.Read(v2_base + 1));
    half4 v10 = ucl::Convert<half4>(args.v.Read(v2_base + 2));
    half4 v11 = ucl::Convert<half4>(args.v.Read(v2_base + 3));
    half4 v_acc2 = fma((half4)p2.x, v8, fma((half4)p2.y, v9, fma((half4)p2.z, v10, (half4)p2.w * v11)));

    int v3_base = v_idx + )",
                  v_stride_3s, R"(;
    half4 v12 = ucl::Convert<half4>(args.v.Read(v3_base + 0));
    half4 v13 = ucl::Convert<half4>(args.v.Read(v3_base + 1));
    half4 v14 = ucl::Convert<half4>(args.v.Read(v3_base + 2));
    half4 v15 = ucl::Convert<half4>(args.v.Read(v3_base + 3));
    half4 v_acc3 = fma((half4)p3.x, v12, fma((half4)p3.y, v13, fma((half4)p3.z, v14, (half4)p3.w * v15)));

    out_acc = fma(out_acc, (half4)alpha, (v_acc0 + v_acc1) + (v_acc2 + v_acc3));

    k_idx += 16;
    v_idx += )",
                  v_stride_4s, R"(;
  }
)");

  absl::StrAppend(&op_code, R"(
  for (; chunk < chunk_end; ++chunk) {
    half4 k0 = ucl::Convert<half4>(args.k.Read(k_idx + 0));
    half4 k1 = ucl::Convert<half4>(args.k.Read(k_idx + 1));
    half4 k2 = ucl::Convert<half4>(args.k.Read(k_idx + 2));
    half4 k3 = ucl::Convert<half4>(args.k.Read(k_idx + 3));

    half4 d = ucl::WaveSum(half4(dot(q_slice, k0), dot(q_slice, k1), dot(q_slice, k2), dot(q_slice, k3)));
)");

  if (has_softcap) {
    op_code += R"(
    d = (half4)args.softcap * tanh(d / (half4)args.softcap);
)";
  }

  if (has_mask) {
    op_code += R"(
    // Attention mask and cache boundary check: clamp masked-out or past-active-tokens
    // key scores to -10000.0h (-inf) so they contribute 0.0h to the softmax.
    half4 m_vec = ucl::Convert<half4>(args.mask.Read($MROW, 0, chunk));
    if (args.is_bool_mask) {
      if (m_vec.x < 0.5h || (chunk * 4 + 0) >= active_tokens) d.x = -10000.0h;
      if (m_vec.y < 0.5h || (chunk * 4 + 1) >= active_tokens) d.y = -10000.0h;
      if (m_vec.z < 0.5h || (chunk * 4 + 2) >= active_tokens) d.z = -10000.0h;
      if (m_vec.w < 0.5h || (chunk * 4 + 3) >= active_tokens) d.w = -10000.0h;
    } else {
      d += m_vec;
      if ((chunk * 4 + 0) >= active_tokens) d.x = -10000.0h;
      if ((chunk * 4 + 1) >= active_tokens) d.y = -10000.0h;
      if ((chunk * 4 + 2) >= active_tokens) d.z = -10000.0h;
      if ((chunk * 4 + 3) >= active_tokens) d.w = -10000.0h;
    }
)";
  } else {
    op_code += R"(
    // Cache boundary check: clamp keys beyond active_tokens to -10000.0h (-inf).
    if ((chunk * 4 + 0) >= active_tokens) d.x = -10000.0h;
    if ((chunk * 4 + 1) >= active_tokens) d.y = -10000.0h;
    if ((chunk * 4 + 2) >= active_tokens) d.z = -10000.0h;
    if ((chunk * 4 + 3) >= active_tokens) d.w = -10000.0h;
)";
  }

  absl::StrAppend(&op_code, R"(
    half m_new = max(m_prev, max(max(d.x, d.y), max(d.z, d.w)));
    half alpha = exp2((m_prev - m_new) * inv_ln2);
    half4 p = exp2((d - (half4)m_new) * (half4)inv_ln2);
    p = select(p, half4(0.0h), d < half4(-9000.0h));
    l_prev = fma(l_prev, alpha, (p.x + p.y) + (p.z + p.w));
    m_prev = m_new;

    // Cache entries past active_tokens may be unwritten; do not let them reach
    // the accumulator even with a zero probability (0 * NaN = NaN).
    half4 v0 = ucl::Convert<half4>(args.v.Read(v_idx + 0));
    half4 v1 = (chunk * 4 + 1 < active_tokens) ? ucl::Convert<half4>(args.v.Read(v_idx + 1)) : half4(0.0h);
    half4 v2 = (chunk * 4 + 2 < active_tokens) ? ucl::Convert<half4>(args.v.Read(v_idx + 2)) : half4(0.0h);
    half4 v3 = (chunk * 4 + 3 < active_tokens) ? ucl::Convert<half4>(args.v.Read(v_idx + 3)) : half4(0.0h);
    out_acc = fma(out_acc, (half4)alpha,
                  fma((half4)p.x, v0,
                  fma((half4)p.y, v1,
                  fma((half4)p.z, v2,
                      (half4)p.w * v3))));
    k_idx += 4;
    v_idx += )",
                  v_stride_s, R"(;
  }

  if (tid == 0) {
    s_m[simd_id] = (float)m_prev;
    s_l[simd_id] = (float)l_prev;
  }
  s_acc[tid][simd_id] = out_acc;

  ucl::SyncThreads<WorkGroup, Local>();

  if (simd_id == 0) {
    float m_val = (tid < $NSG) ? s_m[tid] : -10000.0f;
    float global_m = ucl::WaveMax(m_val);

    float sc = (tid < $NSG && s_l[tid] > 0.0f) ? exp2((s_m[tid] - global_m) * 1.44269504f) : 0.0f;
    float l_term = sc * ((tid < $NSG) ? s_l[tid] : 0.0f);
    float l_total = ucl::WaveSum(l_term);
    float inv_l = 1.0f / (l_total + 1e-10f);

    if (tid < $NSG) {
      s_w[tid] = sc * inv_l;
    }
  }

  ucl::SyncThreads<WorkGroup, Local>();

  if (simd_id == 0) {
)");
  // Weighted sum of the per-wave accumulators: pairs of waves with an fma,
  // then a balanced tree (`num_simd_groups` is a power of two).
  std::vector<std::string> terms;
  for (int i = 0; i + 1 < num_simd_groups; i += 2) {
    const std::string name = absl::StrCat("acc", i / 2);
    absl::StrAppend(&op_code, "    half4 ", name, " = fma((half4)s_w[", i,
                    "], s_acc[tid][", i, "], (half4)s_w[", i + 1,
                    "] * s_acc[tid][", i + 1, "]);\n");
    terms.push_back(name);
  }
  for (int level = 0; terms.size() > 1; ++level) {
    std::vector<std::string> next;
    for (size_t i = 0; i + 1 < terms.size(); i += 2) {
      const std::string name = absl::StrCat("sum", level, "_", i / 2);
      absl::StrAppend(&op_code, "    half4 ", name, " = ", terms[i], " + ",
                      terms[i + 1], ";\n");
      next.push_back(name);
    }
    terms = std::move(next);
  }
  absl::StrAppend(&op_code, "    half4 final_acc = ", terms[0], ";\n",
                  is_flattened_dst ? absl::StrFormat(R"(
    if (tid < args.slices) {
      int out_slice = Y * %d + tid;
      args.dst.Write(ucl::Convert<args.dst::type>(final_acc), X, 0, out_slice);
    }
)",
                                                     slices)
                                   : R"(
    if (tid < args.slices) {
      args.dst.Write(ucl::Convert<args.dst::type>(final_acc), X, Y, tid);
    }
)",
                  R"(
  }
}
)");
  return absl::StrReplaceAll(
      op_code,
      {{"$NSG", absl::StrCat(num_simd_groups)}, {"$MROW", mask_row}});
}

// Generates the work-group local-memory reduction and GQA split-K Flash-Decode
// kernel source.
//
// One work group handles one query token, `heads` query heads that share a
// K/V head (up to `max_heads_per_work_group`, so K/V elements are reused
// across GQA sibling heads) and one split of the keys. It walks its keys in
// chunks with an online softmax:
// 1. Each work item computes the full dot products of its keys against the
//    query heads of the work group, reading K slice by slice so that
//    consecutive work items read consecutive keys.
// 2. The work group reduces the chunk maximum and rescales the running sum and
//    the output accumulators.
// 3. The work items are remapped to (channel slice, key group) pairs to
//    accumulate the probability-weighted values, which reads V contiguously.
// Finally the key groups are reduced in local memory. With a single split the
// result is normalized and written out; otherwise every split writes its
// unnormalized output, running maximum and softmax normalizer, and
// `FusedFlashDecodeCombineOp` merges the splits.
std::string GenerateWorkGroupFlashDecodeCode(
    const ::ml_drift::GpuInfo& gpu_info, const FlashDecodeConfig& config,
    int slices, int group_size, int cache_size, int k_stride_head,
    int k_stride_slice, int v_stride_head, int v_stride_block, bool has_mask,
    bool is_bool_mask, int mask_width, bool has_param, bool is_causal,
    bool is_single_query, bool has_softcap, bool is_flattened_dst) {
  const int reduce_threads = config.tuning.reduce_threads;
  const int threads = config.tuning.threads;
  const int heads = config.heads;
  const int splits = config.splits;
  const int key_groups = threads / slices;
  // Keys scored by each work item per chunk: enough for the keys of a split,
  // but at most `MaxFlashDecodeKeysPerThread`, which the local memory budget
  // in `FlashDecodeLocalMemoryBytes` assumes.
  const int max_keys_per_thread = MaxFlashDecodeKeysPerThread(heads, threads);
  const int split_keys = (cache_size + splits - 1) / splits;
  const int keys_per_thread =
      std::clamp((split_keys + threads - 1) / threads, 1, max_keys_per_thread);
  const int chunk = threads * keys_per_thread;

  // Repeats `code` once per query head of the work group, with "$g" replaced
  // by the index of the head within the work group.
  auto per_head = [heads](absl::string_view code) {
    std::string out;
    for (int g = 0; g < heads; ++g) {
      absl::StrAppend(&out,
                      absl::StrReplaceAll(code, {{"$g", absl::StrCat(g)}}));
    }
    return out;
  };
  // Reduces the per-work-item `<value>$g` over the work group into
  // `<result>$g`, which the caller declares and every work item can read
  // afterwards.
  auto reduce = [&](absl::string_view value, absl::string_view result,
                    bool is_sum) {
    auto combine = [is_sum](absl::string_view a, absl::string_view b) {
      return is_sum ? absl::StrCat(a, " + ", b)
                    : absl::StrCat("max(", a, ", ", b, ")");
    };
    const std::string stage1 = per_head(absl::StrCat(
        "      float r$g = red_local[$g * ", threads, " + tid];\n",
        "      for (int i = 1; i < ", threads / reduce_threads, "; ++i) {\n",
        "        r$g = ",
        combine("r$g", absl::StrCat("red_local[$g * ", threads, " + i * ",
                                    reduce_threads, " + tid]")),
        ";\n      }\n      red2_local[$g * ", reduce_threads,
        " + tid] = r$g;\n"));
    const std::string stage2 = per_head(absl::StrCat(
        "    ", result, "$g = red2_local[$g * ", reduce_threads, "];\n",
        "    for (int i = 1; i < ", reduce_threads, "; ++i) {\n", "      ",
        result, "$g = ",
        combine(absl::StrCat(result, "$g"),
                absl::StrCat("red2_local[$g * ", reduce_threads, " + i]")),
        ";\n    }\n"));
    return absl::StrCat(
        per_head(absl::StrCat("    red_local[$g * ", threads,
                              " + tid] = ", value, "$g;\n")),
        "    ucl::SyncThreads<WorkGroup, Local>();\n    if (tid < ",
        reduce_threads, ") {\n", stage1,
        "    }\n    ucl::SyncThreads<WorkGroup, Local>();\n", stage2);
  };

  // The code is generated for exactly `threads` work items per work group.
  // Declaring that size on OpenCL lets the compiler budget registers for it;
  // otherwise Adreno may cap the work-group size of the kernel below
  // `threads`.
  std::string c;
  if (gpu_info.IsApiOpenCl()) {
    c = absl::StrCat("__attribute__((reqd_work_group_size(", threads,
                     ", 1, 1)))\n");
  }
  absl::StrAppend(&c, R"(MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int head0 = ucl::GetGroupId<1>() * )",
                  heads, R"(;
  int kv_head = head0 / )",
                  group_size, R"(;
  int X = ucl::GetGroupId<2>() / )",
                  splits, R"(;
  int split = ucl::GetGroupId<2>() % )",
                  splits, R"(;
  __local float4 q_local[)",
                  heads * slices, R"(];
  __local float p_local[)",
                  heads * chunk, R"(];
  __local float red_local[)",
                  heads * threads, R"(];
  __local float red2_local[)",
                  heads * reduce_threads, R"(];
  __local float4 acc_local[)",
                  threads, R"(];
  for (int i = tid; i < )",
                  heads * slices, "; i += ", threads, R"() {
    q_local[i] = ucl::Convert<float4>(args.q.Read(X, head0 + i / )",
                  slices, ", i % ", slices, R"());
  }
  int active_tokens = args.cache_size;
  bool has_active_tokens = false;
)");

  if (has_param) {
    absl::StrAppend(&c, R"(
  {
    int param_slice = args.src_end_ch_index / 4;
    int param_comp = args.src_end_ch_index % 4;
    float4 p_vec = ucl::Convert<float4>(args.params.Read(0, 0, param_slice, 0));
    float p_raw = param_comp == 0 ? p_vec.x
                : param_comp == 1 ? p_vec.y
                : param_comp == 2 ? p_vec.z
                                  : p_vec.w;
    int param_val = (int)p_raw;
    has_active_tokens = param_val > 0 && param_val <= args.cache_size;
    if (has_active_tokens) {
      active_tokens = param_val;
    }
  }
)");
  }
  if (!has_mask && is_causal) {
    absl::StrAppend(
        &c, GenerateImplicitCausalBoundCode(has_param, is_single_query));
  }

  absl::StrAppend(&c, per_head(R"(
  float m_run$g = -1.0e30f;
  float l_part$g = 0.0f;
  float4 acc$g = ucl::Init<float4>(0.0f);)"),
                  R"(
  int slice = tid % )",
                  slices, R"(;
  int key_group = tid / )",
                  slices, R"(;
  int k_head = kv_head * )",
                  k_stride_head, R"(;
  int v_head = kv_head * )",
                  v_stride_head, R"( + slice * 4;
  // Keys of this split, in whole blocks of 4 keys.
  int split_keys = ((active_tokens + )",
                  splits - 1, ") / ", splits, R"( + 3) / 4 * 4;
  int key_begin = split * split_keys;
  int key_end = min(active_tokens, key_begin + split_keys);
  ucl::SyncThreads<WorkGroup, Local>();

  for (int base = key_begin; base < key_end; base += )",
                  chunk, R"() {
)");
  // Scores of the keys base + r * threads + tid.
  for (int r = 0; r < keys_per_thread; ++r) {
    absl::StrAppend(&c, per_head(absl::StrCat("    float s", r, "_$g;\n")),
                    "    {\n      int key = base + ", r * threads, " + tid;\n",
                    per_head("      float d$g = 0.0f;\n"), R"(
      bool valid = key < key_end;
      if (valid) {
        int k_idx = k_head + key;
        for (int sl = 0; sl < )",
                    slices, R"(; ++sl) {
          float4 kv = ucl::Convert<float4>(args.k.Read(k_idx));
)",
                    per_head(absl::StrCat("          d$g += dot(q_local[$g * ",
                                          slices, " + sl], kv);\n")),
                    "          k_idx += ", k_stride_slice, ";\n        }\n");
    if (has_softcap) {
      absl::StrAppend(
          &c,
          per_head("        d$g = args.softcap * tanh(d$g / args.softcap);\n"));
    }
    if (has_mask) {
      const std::string mask_row = mask_width == 1 ? "0" : "X";
      if (is_bool_mask) {
        // Masked keys get the score of the keys outside the split. Comparing
        // the mask converted to float and clearing `valid` instead attended
        // to the masked keys on Adreno 830.
        absl::StrAppend(
            &c, "        bool4 mask_vec = args.mask.Read<bool>(", mask_row,
            R"(, 0, key / 4);
        int mask_comp = key % 4;
        int mask_val = mask_comp == 0 ? (int)mask_vec.x
                     : mask_comp == 1 ? (int)mask_vec.y
                     : mask_comp == 2 ? (int)mask_vec.z
                                      : (int)mask_vec.w;
)",
            per_head("        if (mask_val == 0) d$g = -1.0e30f;\n"));
      } else {
        absl::StrAppend(&c,
                        "        float4 mask_vec = "
                        "ucl::Convert<float4>(args.mask.Read(",
                        mask_row, R"(, 0, key / 4));
        int mask_comp = key % 4;
        float mask_val = mask_comp == 0 ? mask_vec.x
                       : mask_comp == 1 ? mask_vec.y
                       : mask_comp == 2 ? mask_vec.z
                                        : mask_vec.w;
)",
                        per_head("        d$g += mask_val;\n"));
      }
    }
    absl::StrAppend(
        &c, "      }\n",
        per_head(absl::StrCat("      s", r, "_$g = valid ? d$g : -1.0e30f;\n")),
        "    }\n");
  }

  // Chunk maximum, rescaling and probabilities.
  absl::StrAppend(&c, per_head("    float tmax$g = s0_$g;\n"));
  for (int r = 1; r < keys_per_thread; ++r) {
    absl::StrAppend(
        &c, per_head(absl::StrCat("    tmax$g = max(tmax$g, s", r, "_$g);\n")));
  }
  absl::StrAppend(&c, per_head("    float cmax$g;\n"),
                  reduce("tmax", "cmax", /*is_sum=*/false), per_head(R"(
    float m_new$g = max(m_run$g, cmax$g);
    float alpha$g = exp(m_run$g - m_new$g);
    m_run$g = m_new$g;
    float psum$g = 0.0f;
)"));
  for (int r = 0; r < keys_per_thread; ++r) {
    absl::StrAppend(
        &c, per_head(absl::StrCat(
                "    {\n      float p = s", r, "_$g > -1.0e29f ? exp(s", r,
                "_$g - m_new$g) : 0.0f;\n      p_local[$g * ", chunk, " + ",
                r * threads, " + tid] = p;\n      psum$g += p;\n    }\n")));
  }
  absl::StrAppend(&c, per_head("    l_part$g = l_part$g * alpha$g + psum$g;\n"),
                  "    ucl::SyncThreads<WorkGroup, Local>();\n",
                  per_head("    acc$g *= alpha$g;\n"), R"(
    // Probability-weighted values, 4 keys per block.
    int n_blocks = (min()",
                  chunk, R"(, key_end - base) + 3) / 4;
    int v_idx = v_head + (base / 4 + key_group) * )",
                  v_stride_block, R"(;
    for (int b = key_group; b < n_blocks; b += )",
                  key_groups, R"() {
      float4 v0 = ucl::Convert<float4>(args.v.Read(v_idx));
      float4 v1 = ucl::Convert<float4>(args.v.Read(v_idx + 1));
      float4 v2 = ucl::Convert<float4>(args.v.Read(v_idx + 2));
      float4 v3 = ucl::Convert<float4>(args.v.Read(v_idx + 3));
      // Cache entries past active_tokens may be unwritten; their probability
      // is 0, but 0 * NaN would still poison the accumulator.
      int block_key = base + b * 4;
      if (block_key + 4 > active_tokens) {
        if (block_key + 1 >= active_tokens) v1 = ucl::Init<float4>(0.0f);
        if (block_key + 2 >= active_tokens) v2 = ucl::Init<float4>(0.0f);
        if (block_key + 3 >= active_tokens) v3 = ucl::Init<float4>(0.0f);
      }
)",
                  per_head(absl::StrCat(
                      "      {\n        int p_idx = $g * ", chunk,
                      " + b * 4;\n        acc$g += p_local[p_idx] * v0 + "
                      "p_local[p_idx + 1] * v1 + p_local[p_idx + 2] * v2 + "
                      "p_local[p_idx + 3] * v3;\n      }\n")),
                  "      v_idx += ", key_groups * v_stride_block, R"(;
    }
  }

  // Softmax normalizer.
)",
                  per_head("  float l_total$g;\n"),
                  reduce("l_part", "l_total", /*is_sum=*/true));

  // Key-group reduction and output.
  for (int g = 0; g < heads; ++g) {
    const std::string head = absl::StrCat("head0 + ", g);
    std::string write;
    if (splits > 1) {
      const std::string part_row =
          absl::StrCat("X * ", splits, " + split, ", head);
      write = absl::StrCat(
          "    args.dst.Write(ucl::Convert<args.dst::type>(sum), ", part_row,
          ", tid);\n", "    if (tid == 0) {\n",
          "      float4 ml = ucl::Init<float4>(m_run", g, ", l_total", g,
          ", 0.0f, 0.0f);\n",
          "      args.dst.Write(ucl::Convert<args.dst::type>(ml), ", part_row,
          ", ", slices, ");\n", "    }\n");
    } else {
      write = absl::StrCat(
          "    float4 res = sum * (l_total", g, " > 0.0f ? 1.0f / l_total", g,
          " : 0.0f);\n",
          is_flattened_dst
              ? absl::StrCat("    args.dst.Write(ucl::Convert<args.dst::type>("
                             "res), X, 0, (",
                             head, ") * ", slices, " + tid);\n")
              : absl::StrCat("    args.dst.Write(ucl::Convert<args.dst::type>("
                             "res), X, ",
                             head, ", tid);\n"));
    }
    absl::StrAppend(&c, "  acc_local[tid] = acc", g, R"(;
  ucl::SyncThreads<WorkGroup, Local>();
  if (tid < )",
                    slices, R"() {
    float4 sum = acc_local[tid];
    for (int i = 1; i < )",
                    key_groups, R"(; ++i) {
      sum += acc_local[i * )",
                    slices, R"( + tid];
    }
)",
                    write, "  }\n");
    if (g + 1 < heads) {
      absl::StrAppend(&c, "  ucl::SyncThreads<WorkGroup, Local>();\n");
    }
  }
  absl::StrAppend(&c, "}\n");
  return c;
}

// Creates the fused Flash-Decode kernel for a config that
// `IsSupportedFlashDecode` accepts. Returns an error, instead of a kernel the
// device would reject, if the work-group kernel exceeds the local memory
// budget.
absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateFusedFlashDecodeSdpa(const ::ml_drift::GpuInfo& gpu_info,
                           const ::ml_drift::TensorDescriptor& q_desc,
                           const ::ml_drift::TensorDescriptor& k_desc,
                           const ::ml_drift::TensorDescriptor& v_desc,
                           const ::ml_drift::TensorDescriptor* mask_desc,
                           const ::ml_drift::TensorDescriptor* param_desc,
                           const ::ml_drift::TensorDescriptor& dst_desc,
                           const SdpaTransposedAttributes& attr,
                           bool is_flattened_dst,
                           const FlashDecodeConfig& config) {
  const int slices = q_desc.GetBHWCShape().c / 4;
  if (!config.use_wave_simd) {
    const int local_memory_bytes =
        FlashDecodeLocalMemoryBytes(config.tuning, config.heads, slices);
    if (local_memory_bytes > config.tuning.local_memory_bytes) {
      return absl::InternalError(absl::StrCat(
          "The work-group Flash-Decode SDPA kernel needs ", local_memory_bytes,
          " bytes of local memory for ", config.heads,
          " query head(s) of head dim ", slices * 4,
          " per work group, more than the ", config.tuning.local_memory_bytes,
          " bytes the device allows. IsSupportedFlashDecode should have "
          "rejected this config."));
    }
  }
  const int q_heads = q_desc.GetBHWCShape().h;
  const int kv_heads = k_desc.GetBHWCShape().h;
  const int group_size =
      (kv_heads > 0 && q_heads >= kv_heads && (q_heads % kv_heads == 0))
          ? (q_heads / kv_heads)
          : 1;
  const int cache_size = k_desc.GetBHWCShape().w;
  // Packed K: [kv_head][slice][key], one vector of 4 channels per key.
  // Packed V: [kv_head][key / 4][slice][key % 4], one vector of 4 channels per
  // key, so the 4 keys of a block are adjacent for every slice.
  const int cache_slices = (cache_size + 3) / 4;
  const int k_stride_slice = cache_slices * 4;
  const int k_stride_head = slices * k_stride_slice;
  const int v_stride_block = slices * 4;
  const int v_stride_head = cache_slices * v_stride_block;

  const ::ml_drift::int3 work_group_size =
      config.use_wave_simd
          ? ::ml_drift::int3(1, slices, config.tuning.num_simd_groups)
          : ::ml_drift::int3(config.tuning.threads, 1, 1);
  const int head_groups =
      config.use_wave_simd ? q_heads : (q_heads / config.heads);
  FusedFlashDecodeSdpaOp op(work_group_size, head_groups, config.splits);
  op.args_.AddInt("cache_size", cache_size);
  if (config.use_wave_simd) {
    op.args_.AddInt("slices", slices);
  }
  op.AddSrcTensor("q", q_desc);
  op.AddSrcTensor("k", k_desc);
  op.AddSrcTensor("v", v_desc);

  const bool has_mask = (mask_desc != nullptr);
  const bool is_bool_mask =
      has_mask && mask_desc->GetDataType() == ::ml_drift::DataType::kBool;
  if (has_mask) {
    if (config.use_wave_simd) {
      op.args_.AddInt("is_bool_mask", is_bool_mask ? 1 : 0);
    }
    op.AddSrcTensor("mask", *mask_desc);
  }

  const bool has_param =
      (param_desc != nullptr && attr.runtime_check.src_end_ch_index.has_value());
  if (has_param) {
    op.args_.AddInt("src_end_ch_index", *attr.runtime_check.src_end_ch_index);
    op.AddSrcTensor("params", *param_desc);
  }

  const bool has_softcap = (attr.softcap.has_value() && *attr.softcap > 0.0f);
  if (has_softcap) {
    op.args_.AddFloat("softcap", *attr.softcap);
  }

  op.AddDstTensor("dst", dst_desc);

  const int mask_width = has_mask ? mask_desc->GetBHWCShape().w : 1;
  // See `GenerateImplicitCausalBoundCode` for how a single-token decode query
  // range-checks `param[0]` instead of clamping to it.
  const bool is_single_query = q_desc.GetBHWCShape().w == 1;
  if (config.use_wave_simd) {
    op.code_ = GenerateWaveSimdFlashDecodeCode(
        config.tuning.num_simd_groups, slices, group_size, k_stride_head,
        k_stride_slice, v_stride_head, v_stride_block, has_mask, is_bool_mask,
        mask_width, has_param, attr.is_causal, is_single_query, has_softcap,
        is_flattened_dst);
  } else {
    op.code_ = GenerateWorkGroupFlashDecodeCode(
        gpu_info, config, slices, group_size, cache_size, k_stride_head,
        k_stride_slice, v_stride_head, v_stride_block, has_mask, is_bool_mask,
        mask_width, has_param, attr.is_causal, is_single_query, has_softcap,
        is_flattened_dst);
  }
  ResolveWaveSimd(gpu_info, &op.code_);
  return std::make_unique<FusedFlashDecodeSdpaOp>(std::move(op));
}

// Combines the partial results that `CreateFusedFlashDecodeSdpa` writes
// with more than one split into the normalized attention output.
std::unique_ptr<::ml_drift::GPUOperation> CreateFusedFlashDecodeCombine(
    const ::ml_drift::TensorDescriptor& part_desc,
    const ::ml_drift::TensorDescriptor& dst_desc, int slices, int q_heads,
    int splits, bool is_flattened_dst) {
  FusedFlashDecodeCombineOp op(slices, q_heads, splits);
  op.work_group_size_ = ::ml_drift::int3(slices, 1, 1);
  op.AddSrcTensor("part", part_desc);
  op.AddDstTensor("dst", dst_desc);
  const std::string ml_read =
      absl::StrCat("ucl::Convert<float4>(args.part.Read(X * ", splits,
                   " + i, head, ", slices, "))");
  op.code_ = absl::StrCat(
      R"(
MAIN_FUNCTION($0) {
  int slice = ucl::GetGlobalId<0>();
  int head = ucl::GetGlobalId<1>();
  int X = ucl::GetGlobalId<2>();
  if (slice >= )",
      slices, " || head >= ", q_heads, R"() {
    return;
  }
  float m_max = -1.0e30f;
  for (int i = 0; i < )",
      splits, R"(; ++i) {
    float4 ml = )",
      ml_read, R"(;
    if (ml.y > 0.0f) {
      m_max = max(m_max, ml.x);
    }
  }
  float l_sum = 0.0f;
  float4 acc = ucl::Init<float4>(0.0f);
  for (int i = 0; i < )",
      splits, R"(; ++i) {
    float4 ml = )",
      ml_read, R"(;
    if (ml.y > 0.0f) {
      float w = exp(ml.x - m_max);
      l_sum += w * ml.y;
      acc += w * ucl::Convert<float4>(args.part.Read(X * )",
      splits, R"( + i, head, slice));
    }
  }
  float4 res = acc * (l_sum > 0.0f ? 1.0f / l_sum : 0.0f);
)",
      is_flattened_dst ? absl::StrCat("  args.dst.Write(ucl::Convert<args.dst::"
                                      "type>(res), X, 0, head * ",
                                      slices, " + slice);\n")
                       : "  args.dst.Write(ucl::Convert<args.dst::type>"
                         "(res), X, head, slice);\n",
      "}\n");
  return std::make_unique<FusedFlashDecodeCombineOp>(std::move(op));
}

// Prefill tiling: Each threadgroup runs 4 SIMD groups (128 threads)
// and owns BQ=32 query rows
// (4 warps x 8 rows/warp) across GQA sibling query heads sharing the same KV
// head, stepping by BK=16 keys per iteration.
// Threadgroup memory is Q_smem (32x132 halfs = 8,448 B) + KV_smem (2,560 halfs
// = 5,120 B) = 13,568 B (< 16 KB), allowing 2 full threadgroups (8 warps = 256
// threads) per Apple GPU core while keeping live registers <= 48 per thread
// (zero register spills).
constexpr int kPrefillThreadsPerTg = 128;
constexpr int kPrefillTotalRows = 32;
constexpr int kPrefillKeyBlock = 16;

class FusedFlashAttentionPrefillOp : public ::ml_drift::GPUOperation {
 public:
  FusedFlashAttentionPrefillOp() = default;
  FusedFlashAttentionPrefillOp(int q_per_head, int heads_per_tg)
      : q_per_head_(q_per_head), heads_per_tg_(heads_per_tg) {}

  ::ml_drift::int3 GetGridSize() const override {
    const int num_q_tiles = (dst_[0]->Width() + q_per_head_ - 1) / q_per_head_;
    const int num_head_groups =
        (dst_[0]->Height() + heads_per_tg_ - 1) / heads_per_tg_;
    return ::ml_drift::int3(num_q_tiles,
                            num_head_groups * kPrefillThreadsPerTg, 1);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {::ml_drift::int3(1, kPrefillThreadsPerTg, 1)};
  }

  FusedFlashAttentionPrefillOp(FusedFlashAttentionPrefillOp&&) = default;
  FusedFlashAttentionPrefillOp& operator=(FusedFlashAttentionPrefillOp&&) =
      default;
  FusedFlashAttentionPrefillOp(const FusedFlashAttentionPrefillOp&) = delete;
  FusedFlashAttentionPrefillOp& operator=(const FusedFlashAttentionPrefillOp&) =
      delete;

 private:
  int q_per_head_ = kPrefillTotalRows;
  int heads_per_tg_ = 1;
};

std::unique_ptr<::ml_drift::GPUOperation> CreateFusedFlashAttentionPrefill(
    const ::ml_drift::GpuInfo& gpu_info,
    const ::ml_drift::TensorDescriptor& q_desc,
    const ::ml_drift::TensorDescriptor& k_desc,
    const ::ml_drift::TensorDescriptor& v_desc,
    const ::ml_drift::TensorDescriptor* mask_desc,
    const ::ml_drift::TensorDescriptor* param_desc,
    const ::ml_drift::TensorDescriptor& dst_desc,
    const SdpaTransposedAttributes& attr) {
  int slices = dst_desc.GetBHWCShape().c / 4;
  int v_stride_s = slices * 4;
  int k_o_slices = (k_desc.GetBHWCShape().w + 3) / 4;
  int k_stride_head = slices * k_o_slices * 4;
  int k_stride_slice = k_o_slices * 4;
  int v_stride_head = k_o_slices * v_stride_s;
  const int q_heads = q_desc.GetBHWCShape().h;
  const int kv_heads = k_desc.GetBHWCShape().h;
  const int gqa_ratio =
      (kv_heads > 0 && q_heads >= kv_heads && (q_heads % kv_heads == 0))
          ? (q_heads / kv_heads)
          : 1;

  // Step A: Group GQA sibling heads and query tokens into the 4 warps (32 query
  // rows) of the threadgroup so all warps share a single cooperative K/V load.
  int heads_per_tg = 1;
  if (gqa_ratio % 4 == 0) {
    heads_per_tg = 4;
  } else if (gqa_ratio % 2 == 0) {
    heads_per_tg = 2;
  }
  const int q_per_head = kPrefillTotalRows / heads_per_tg;

  FusedFlashAttentionPrefillOp custom_op(q_per_head, heads_per_tg);
  custom_op.work_group_size_ = ::ml_drift::int3(1, kPrefillThreadsPerTg, 1);
  custom_op.args_.AddInt("cache_size", k_desc.GetBHWCShape().w);
  custom_op.args_.AddInt("slices", slices);

  custom_op.AddSrcTensor("q", q_desc);
  custom_op.AddSrcTensor("k", k_desc);
  custom_op.AddSrcTensor("v", v_desc);

  bool has_mask = (mask_desc != nullptr);
  if (has_mask) {
    bool is_bool_mask =
        (mask_desc->GetDataType() == ::ml_drift::DataType::kBool);
    custom_op.args_.AddInt("is_bool_mask", is_bool_mask ? 1 : 0);
    custom_op.AddSrcTensor("mask", *mask_desc);
  }

  bool has_param = (param_desc != nullptr &&
                    attr.runtime_check.src_end_ch_index.has_value());
  if (has_param) {
    custom_op.args_.AddInt("src_end_ch_index",
                          *attr.runtime_check.src_end_ch_index);
    custom_op.AddSrcTensor("params", *param_desc);
  }

  bool has_softcap = (attr.softcap.has_value() && *attr.softcap > 0.0f);
  if (has_softcap) {
    custom_op.args_.AddFloat("softcap", *attr.softcap);
  }

  custom_op.AddDstTensor("dst", dst_desc);

  const int td = (slices + 1) / 2;  // 8-channel MMA tiles along head_dim
  const int bd_padded = td * 8;
  const int ldq = bd_padded + 4;    // 132 halfs (8,448 B for 32 rows)
  const int ldk = 20;               // 16 keys + 4 padding halfs
  const int ldv = bd_padded + 4;    // 132 halfs
  const int q_smem_size = kPrefillTotalRows * ldq;
  const int kv_smem_size = std::max(bd_padded * ldk, kPrefillKeyBlock * ldv);
  const int k_load_iters = (slices + 7) / 8;

  std::string op_code;
  absl::StrAppend(&op_code, R"(
#pragma OPENCL EXTENSION ucl_wave_simd: enable
#include <metal_simdgroup_matrix>

inline float2 mma_f16_f32_8x8(half2 A, half2 B, float2 C) {
  metal::simdgroup_matrix<float, 8, 8> D_mat;
  metal::simdgroup_matrix<half, 8, 8> A_mat;
  metal::simdgroup_matrix<half, 8, 8> B_mat;
  metal::simdgroup_matrix<float, 8, 8> C_mat;
  reinterpret_cast<thread half2&>(A_mat.thread_elements()) = A;
  reinterpret_cast<thread half2&>(B_mat.thread_elements()) = B;
  reinterpret_cast<thread float2&>(C_mat.thread_elements()) = C;
  metal::simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
  return reinterpret_cast<thread float2&>(D_mat.thread_elements());
}

MAIN_FUNCTION($0) {
  int tile_x = ucl::GetGlobalId<0>();
  int tg_y = ucl::GetGroupId<1>();
  int tid = ucl::GetLocalId<1>();
  int sg_id = tid >> 5;
  int lane_id = tid & 31;

  int Y0 = tg_y * )",
                  heads_per_tg, R"(;
  int X0 = tile_x * )",
                  q_per_head, R"(;
  int Y_warp = Y0 + (sg_id % )",
                  heads_per_tg, R"();
  int X_warp = X0 + (sg_id / )",
                  heads_per_tg, R"() * 8;

  int dst_w = args.dst.Width();
  int dst_h = args.dst.Height();
  if (X0 >= dst_w || Y0 >= dst_h) {
    return;
  }

  int active_tokens = args.cache_size;
  int q_start = 0;
)");

  if (has_param) {
    absl::StrAppend(&op_code, R"(
  int param_slice = args.src_end_ch_index / 4;
  int param_comp = args.src_end_ch_index % 4;
  float4 p_vec = ucl::Convert<float4>(args.params.Read(0, 0, param_slice, 0));
  float p_raw = (param_comp == 0) ? p_vec.x : ((param_comp == 1) ? p_vec.y : ((param_comp == 2) ? p_vec.z : p_vec.w));
  int param_val = (int)p_raw;
  if (param_val > 0 && param_val <= args.cache_size) {
    active_tokens = param_val;
  }
  float4 p_start_vec = ucl::Convert<float4>(args.params.Read(0, 0, 0, 0));
  int start_val = (int)p_start_vec.x;
  if (start_val > 0 && start_val < active_tokens) {
    q_start = start_val;
  }
)");
  }

  // When a fixed-width prefill signature (dst_w, e.g. 1024) processes a
  // partial chunk (active_tokens - q_start < dst_w, e.g. 128 tokens) and the
  // BOOL mask is pruned, query columns X >= valid_w are inactive padding.
  // Zero-fill those padded output columns and exit early for padded tiles.
  absl::StrAppend(&op_code, R"(
  int valid_w = min(dst_w, active_tokens - q_start);
  if (X0 >= valid_w) {
    if (lane_id < args.slices && Y_warp < dst_h) {
      for (int r = 0; r < 8; ++r) {
        if (X_warp + r < dst_w) {
          args.dst.Write(ucl::Convert<args.dst::type>(float4(0.0f)),
                         X_warp + r, Y_warp, lane_id);
        }
      }
    }
    return;
  }
)");

  const bool is_causal = attr.is_causal;
  if (is_causal) {
    absl::StrAppend(&op_code, "\n  int max_tokens = min(X0 + ", q_per_head,
                    " + q_start, active_tokens);\n");
  } else {
    absl::StrAppend(&op_code, "\n  int max_tokens = active_tokens;\n");
  }
  absl::StrAppend(&op_code, R"(  float inv_ln2 = 1.4426950408889634f;

  // Local memory: 8,448 B (Q_smem) + 5,120 B (KV_smem) = 13,568 B.
  __local half Q_smem[)",
                  q_smem_size, R"(];
  __local half KV_smem[)",
                  kv_smem_size, R"(];
  __local half* K_smem = KV_smem;
  __local half* V_smem = KV_smem;

  // Stage scaled Q into Q_smem[sg_id * 8 + r][lane_id * 4].
  for (int r = 0; r < 8; ++r) {
    int q_x = X_warp + r;
    int q_row_off = (sg_id * 8 + r) * )",
                  ldq, R"(;
    if (lane_id < )",
                  slices, R"() {
      bool q_ok = (q_x < valid_w && Y_warp < dst_h);
      half4 q_v = q_ok ? ucl::Convert<half4>(
                             ucl::Convert<float4>(
                                 args.q.Read(q_x, Y_warp, lane_id)) *
                             inv_ln2)
                       : half4(0.0h);
      *reinterpret_cast<__local half4*>(
          &Q_smem[q_row_off + lane_id * 4]) = q_v;
    }
)");
  if (bd_padded > slices * 4) {
    absl::StrAppend(&op_code, "    if (lane_id == 0) {\n");
    for (int c = slices * 4; c < bd_padded; ++c) {
      absl::StrAppend(&op_code, "      Q_smem[q_row_off + ", c, "] = 0.0h;\n");
    }
    absl::StrAppend(&op_code, "    }\n");
  }
  absl::StrAppend(&op_code, R"(  }

  // Apple simdgroup_matrix<T, 8, 8> lane-to-(row, col) fragment mapping.
  int qid = lane_id >> 2;
  int sm = (qid & 4) + ((lane_id >> 1) & 3);
  int sn = ((qid & 2) << 1) + ((lane_id & 1) << 1);
  int q_smem_base = (sg_id * 8 + sm) * )",
                  ldq, R"( + sn;

  int X_row = X_warp + sm;
  int P_row = X_row + q_start;
  bool row_valid = (X_row < valid_w && Y_warp < dst_h);

  float m_prev = -10000.0f;
  float l_prev = 0.0f;
)");
  for (int id = 0; id < td; ++id) {
    absl::StrAppend(&op_code, "  float2 o_frag", id, " = float2(0.0f);\n");
  }

  // KV buffer base addressing (Grouped-Query Attention).
  // K layout (WeightsLayout::kOSpatialIOGroupO4I4):
  //   [kv_heads, slices, k_o_slices, 4_keys, 4_channels] where each half4
  //   holds 4 channels (4*c_slice..4*c_slice+3) for one key at linear index
  //   k_head_base + c_slice * k_stride_slice + key_idx.
  // V layout (WeightsLayout::kOSpatialIOGroupI4O4):
  //   [kv_heads, k_o_slices, 4_keys, slices, 4_channels] where each half4
  //   holds 4 channels (4*c_slice..4*c_slice+3) for key (g*4 + t) at linear
  //   index v_head_base + (g*4 + v_k_sub) * slices + v_c_slice.
  absl::StrAppend(&op_code, "\n  int kv_head = ",
                  (gqa_ratio > 1 ? absl::StrCat("Y0 / ", gqa_ratio) : "Y0"),
                  ";\n  int k_head_base = kv_head * ", k_stride_head,
                  ";\n  int v_head_base = kv_head * ", v_stride_head,
                  ";\n  int k_s = tid & 15;\n  int k_cg = tid >> 4;\n"
                  "  int v_k_sub = tid & 3;\n  int v_c_slice = tid >> 2;\n"
                  "  int v_c_col = v_c_slice * 4;\n\n");

  auto emit_key_block_body = [&](bool is_interior) {
    absl::StrAppend(&op_code, "    ucl::SyncThreads<WorkGroup, Local>();\n");
    // 1. Coalesced K load (16 contiguous half4 keys across k_s = tid & 15,
    // 8 slices per iteration across k_cg = tid >> 4).
    if (is_interior && (slices % 8 == 0)) {
      absl::StrAppend(
          &op_code, "    #pragma unroll\n    for (int r = 0; r < ",
          k_load_iters, "; ++r) {\n      int c_sl = r * 8 + k_cg;\n",
          "      half4 kv = ucl::Convert<half4>(args.k.Read(k_head_base + "
          "c_sl * ",
          k_stride_slice, " + key_base + k_s));\n",
          "      int k_r = c_sl * 4;\n", "      K_smem[(k_r + 0) * ", ldk,
          " + k_s] = kv.x;\n", "      K_smem[(k_r + 1) * ", ldk,
          " + k_s] = kv.y;\n", "      K_smem[(k_r + 2) * ", ldk,
          " + k_s] = kv.z;\n", "      K_smem[(k_r + 3) * ", ldk,
          " + k_s] = kv.w;\n    }\n");
    } else {
      absl::StrAppend(
          &op_code, "    #pragma unroll\n    for (int r = 0; r < ",
          k_load_iters, "; ++r) {\n      int c_sl = r * 8 + k_cg;\n",
          "      bool k_ok = (c_sl < ", slices,
          " && (key_base + k_s) < active_tokens);\n",
          "      half4 kv = k_ok ? "
          "ucl::Convert<half4>(args.k.Read(k_head_base + c_sl * ",
          k_stride_slice, " + key_base + k_s)) : half4(0.0h);\n",
          "      if (c_sl < ", slices, ") {\n",
          "        int k_r = c_sl * 4;\n", "        K_smem[(k_r + 0) * ", ldk,
          " + k_s] = kv.x;\n", "        K_smem[(k_r + 1) * ", ldk,
          " + k_s] = kv.y;\n", "        K_smem[(k_r + 2) * ", ldk,
          " + k_s] = kv.z;\n", "        K_smem[(k_r + 3) * ", ldk,
          " + k_s] = kv.w;\n      }\n    }\n");
    }
    if (bd_padded > slices * 4) {
      absl::StrAppend(&op_code, "    if (tid < 16) {\n");
      for (int c = slices * 4; c < bd_padded; ++c) {
        absl::StrAppend(&op_code, "      K_smem[", c * ldk,
                        " + tid] = 0.0h;\n");
      }
      absl::StrAppend(&op_code, "    }\n");
    }
    absl::StrAppend(
        &op_code, "    ucl::SyncThreads<WorkGroup, Local>();\n\n",
        "    // 2. Hardware FP16->FP32 simdgroup_matrix Q * K^T (8x16 per "
        "warp).\n",
        "    float2 s_frag0 = float2(0.0f);\n",
        "    float2 s_frag1 = float2(0.0f);\n",
        "    #pragma unroll\n    for (int dd = 0; dd < ", td, "; ++dd) {\n",
        "      half2 qf = *reinterpret_cast<const __local "
        "half2*>(&Q_smem[q_smem_base + dd * 8]);\n",
        "      int k_off = (dd * 8 + sm) * ", ldk, " + sn;\n",
        "      half2 kf0 = *reinterpret_cast<const __local "
        "half2*>(&K_smem[k_off + 0]);\n",
        "      half2 kf1 = *reinterpret_cast<const __local "
        "half2*>(&K_smem[k_off + 8]);\n",
        "      s_frag0 = mma_f16_f32_8x8(qf, kf0, s_frag0);\n",
        "      s_frag1 = mma_f16_f32_8x8(qf, kf1, s_frag1);\n    }\n");

    // 3. Optional softcapping, mask, and active/causal key bounds.
    if (has_softcap) {
      for (int ik = 0; ik < 2; ++ik) {
        absl::StrAppend(
            &op_code, "    s_frag", ik,
            ".x = (float)args.softcap * tanh((s_frag", ik,
            ".x / inv_ln2) / (float)args.softcap) * inv_ln2;\n", "    s_frag",
            ik, ".y = (float)args.softcap * tanh((s_frag", ik,
            ".y / inv_ln2) / (float)args.softcap) * inv_ln2;\n");
      }
    }

    if (!is_interior) {
      for (int ik = 0; ik < 2; ++ik) {
        absl::StrAppend(&op_code, "    int col", ik, "_0 = key_base + ", ik * 8,
                        " + sn;\n", "    int col", ik, "_1 = col", ik,
                        "_0 + 1;\n");
        if (has_mask) {
          for (int c = 0; c < 2; ++c) {
            absl::StrAppend(
                &op_code, "    bool m_ok", ik, "_", c, " = (row_valid && col",
                ik, "_", c, " < active_tokens);\n", "    int msl", ik, "_", c,
                " = col", ik, "_", c, " >> 2;\n", "    int mcm", ik, "_", c,
                " = col", ik, "_", c, " & 3;\n", "    half4 mk", ik, "_", c,
                " = m_ok", ik, "_", c,
                " ? ucl::Convert<half4>(args.mask.Read(X_row, 0, msl", ik, "_",
                c, ")) : half4(0.0h);\n", "    float mv", ik, "_", c,
                " = (float)((mcm", ik, "_", c, " == 0) ? mk", ik, "_", c,
                ".x : ((mcm", ik, "_", c, " == 1) ? mk", ik, "_", c,
                ".y : ((mcm", ik, "_", c, " == 2) ? mk", ik, "_", c, ".z : mk",
                ik, "_", c, ".w)));\n", "    if (args.is_bool_mask) {\n",
                "      if (mv", ik, "_", c, " < 0.5f) s_frag", ik,
                (c == 0 ? ".x" : ".y"), " = -10000.0f;\n", "    } else {\n",
                "      s_frag", ik, (c == 0 ? ".x" : ".y"), " += mv", ik, "_",
                c, " * inv_ln2;\n", "    }\n");
          }
        }
        if (is_causal) {
          absl::StrAppend(&op_code, "    if (col", ik,
                          "_0 >= active_tokens || col", ik,
                          "_0 > P_row) s_frag", ik, ".x = -10000.0f;\n",
                          "    if (col", ik, "_1 >= active_tokens || col", ik,
                          "_1 > P_row) s_frag", ik, ".y = -10000.0f;\n");
        } else {
          absl::StrAppend(&op_code, "    if (col", ik,
                          "_0 >= active_tokens) s_frag", ik,
                          ".x = -10000.0f;\n", "    if (col", ik,
                          "_1 >= active_tokens) s_frag", ik,
                          ".y = -10000.0f;\n");
        }
      }
    }

    // 4. Online softmax reduction across the 4 threads sharing row `sm`
    // (lanes XOR 1 and XOR 8 in Apple's 8x8 simdgroup_matrix layout).
    absl::StrAppend(&op_code, R"(
    float row_max = max(max(s_frag0.x, s_frag0.y), max(s_frag1.x, s_frag1.y));
    row_max = max(row_max, ucl::WaveShuffleXor(row_max, 1u));
    row_max = max(row_max, ucl::WaveShuffleXor(row_max, 8u));
    float m_new = max(m_prev, row_max);
    float alp = exp2(m_prev - m_new);

    float2 p0_f = exp2(s_frag0 - m_new);
    float2 p1_f = exp2(s_frag1 - m_new);

    float row_sum = (p0_f.x + p0_f.y) + (p1_f.x + p1_f.y);
    row_sum += ucl::WaveShuffleXor(row_sum, 1u);
    row_sum += ucl::WaveShuffleXor(row_sum, 8u);
    l_prev = fma(l_prev, alp, row_sum);
    m_prev = m_new;

    half2 p_frag0 = half2(p0_f);
    half2 p_frag1 = half2(p1_f);
)");
    for (int id = 0; id < td; ++id) {
      absl::StrAppend(&op_code, "    o_frag", id, " *= alp;\n");
    }

    // 5. Coalesced half4 V load into V_smem[16][ldv] (placed after softmax so
    // s_frag0..1 are already dead, keeping register pressure low).
    absl::StrAppend(&op_code, "\n    ucl::SyncThreads<WorkGroup, Local>();\n");
    if (is_interior) {
      if (slices == 32) {
        absl::StrAppend(
            &op_code,
            "    {\n      int v_idx = v_head_base + (key_base / 4) * ",
            v_stride_s,
            " + tid;\n"
            "      #pragma unroll\n      for (int g = 0; g < 4; ++g) {\n"
            "        half4 vv = ucl::Convert<half4>(args.v.Read(v_idx + g * ",
            v_stride_s,
            "));\n"
            "        *reinterpret_cast<__local half4*>(&V_smem[(g * 4 + "
            "v_k_sub) * ",
            ldv, " + v_c_col]) = vv;\n      }\n    }\n");
      } else {
        absl::StrAppend(
            &op_code, "    if (v_c_slice < ", slices, ") {\n",
            "      int v_idx = v_head_base + (key_base / 4) * ", v_stride_s,
            " + tid;\n"
            "      #pragma unroll\n      for (int g = 0; g < 4; ++g) {\n"
            "        half4 vv = ucl::Convert<half4>(args.v.Read(v_idx + g * ",
            v_stride_s,
            "));\n"
            "        *reinterpret_cast<__local half4*>(&V_smem[(g * 4 + "
            "v_k_sub) * ",
            ldv, " + v_c_col]) = vv;\n      }\n    }\n");
      }
    } else {
      absl::StrAppend(
          &op_code, "    if (v_c_slice < ", slices, ") {\n",
          "      #pragma unroll\n      for (int g = 0; g < 4; ++g) {\n",
          "        int gk = key_base + g * 4 + v_k_sub;\n",
          "        half4 vv = (gk < active_tokens) ? "
          "ucl::Convert<half4>(args.v.Read(v_head_base + (key_base / 4 + g) * ",
          v_stride_s, " + tid)) : half4(0.0h);\n",
          "        *reinterpret_cast<__local half4*>(&V_smem[(g * 4 + "
          "v_k_sub) * ",
          ldv, " + v_c_col]) = vv;\n      }\n    }\n");
    }
    if (bd_padded > slices * 4) {
      absl::StrAppend(&op_code, "    if (tid < 16) {\n");
      for (int c = slices * 4; c < bd_padded; ++c) {
        absl::StrAppend(&op_code, "      V_smem[tid * ", ldv, " + ", c,
                        "] = 0.0h;\n");
      }
      absl::StrAppend(&op_code, "    }\n");
    }

    // 6. Wait for V_smem and compute P * V (8x128 per warp).
    absl::StrAppend(&op_code, "    ucl::SyncThreads<WorkGroup, Local>();\n");
    for (int id = 0; id < td; ++id) {
      absl::StrAppend(
          &op_code, "    {\n      int v_col = ", id * 8, " + sn;\n",
          "      half2 vf0 = *reinterpret_cast<const __local "
          "half2*>(&V_smem[(0 + sm) * ",
          ldv, " + v_col]);\n", "      o_frag", id,
          " = mma_f16_f32_8x8(p_frag0, vf0, o_frag", id, ");\n",
          "      half2 vf1 = *reinterpret_cast<const __local "
          "half2*>(&V_smem[(8 + sm) * ",
          ldv, " + v_col]);\n", "      o_frag", id,
          " = mma_f16_f32_8x8(p_frag1, vf1, o_frag", id, ");\n", "    }\n");
    }
  };

  // 6. Main key loop: fast interior loop (when no external mask) + boundary
  // tail.
  absl::StrAppend(&op_code, "  int key_base = 0;\n");
  if (!has_mask) {
    if (is_causal) {
      absl::StrAppend(&op_code,
                      "  int interior_end = min(X0 + q_start + 1, max_tokens)"
                      " - ",
                      kPrefillKeyBlock, ";\n");
    } else {
      absl::StrAppend(&op_code, "  int interior_end = max_tokens - ",
                      kPrefillKeyBlock, ";\n");
    }
    absl::StrAppend(&op_code,
                    "  for (; key_base <= interior_end; key_base += ",
                    kPrefillKeyBlock, ") {\n");
    emit_key_block_body(/*is_interior=*/true);
    absl::StrAppend(&op_code, "  }\n");
  }
  absl::StrAppend(&op_code, "  for (; key_base < max_tokens; key_base += ",
                  kPrefillKeyBlock, ") {\n");
  emit_key_block_body(/*is_interior=*/false);
  absl::StrAppend(&op_code, "  }\n\n");

  // 7. Normalize output and write float4 slices via Q_smem (reusing Q_smem so
  // KV_smem stays at 5,120 B).
  absl::StrAppend(&op_code, R"(  float inv_l = 1.0f / (l_prev + 1e-10f);
  ucl::SyncThreads<WorkGroup, Local>();
  __local half* warp_smem = &Q_smem[sg_id * 8 * )",
                  ldq, "];\n");
  for (int id = 0; id < td; ++id) {
    absl::StrAppend(&op_code,
                    "  *reinterpret_cast<__local half2*>(&warp_smem[sm * ",
                    ldq, " + ", id * 8, " + sn]) = half2(o_frag", id,
                    " * inv_l);\n");
  }
  absl::StrAppend(&op_code, R"(  ucl::SyncThreads<SubGroup, Local>();
  if (lane_id < args.slices && Y_warp < dst_h) {
    for (int r = 0; r < 8; ++r) {
      int out_x = X_warp + r;
      if (out_x < valid_w) {
        half4 out_v = *reinterpret_cast<const __local half4*>(
            &warp_smem[r * )",
                  ldq, R"( + lane_id * 4]);
        args.dst.Write(ucl::Convert<args.dst::type>(out_v), out_x, Y_warp,
                       lane_id);
      } else if (out_x < dst_w) {
        args.dst.Write(ucl::Convert<args.dst::type>(float4(0.0f)), out_x,
                       Y_warp, lane_id);
      }
    }
  }
}
)");

  ResolveWaveSimd(gpu_info, &op_code);
  custom_op.code_ = std::move(op_code);
  return std::make_unique<FusedFlashAttentionPrefillOp>(std::move(custom_op));
}

// Applies a BOOL attention mask to `logits`: `dst = mask ? logits : -10000`.
//
// This is an elementwise (linkable) operation, so `MergeNodes` fuses it into
// the epilogue of the operation producing `logits` (the QK^T GEMM) instead of
// running a separate pass over the logits like `GpuModelBuilder::SelectV2`.
// The mask is read with the standard elementwise broadcast rules, so a
// [1, 1, T, keys] mask applies to GQA-folded logits [b, kv_heads, gqa * T,
// keys] directly: folded row `w` reads mask row `w % T`, its query token.
::ml_drift::GpuModelBuilder::TensorHandle SelectBoolMask(
    ::ml_drift::GpuModelBuilder* model_builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& logits,
    const ::ml_drift::GpuModelBuilder::TensorHandle& mask) {
  ::ml_drift::ElementwiseDescriptor op_desc;
  // Use a large negative value to simulate -inf. std::limit<float>::min()
  // causes regression.
  op_desc.args.AddFloat("mask_value", -10000.0f,
                        logits.tensor_desc.GetDataType());
  op_desc.code = R"(
  out_value.x = in2_value.x ? in_value.x : args.mask_value;
  out_value.y = in2_value.y ? in_value.y : args.mask_value;
  out_value.z = in2_value.z ? in_value.z : args.mask_value;
  out_value.w = in2_value.w ? in_value.w : args.mask_value;
)";
  auto dst = model_builder->AddTensor(logits.tensor_desc);
  ::ml_drift::OperationDef definition;
  definition.src_tensors.push_back(logits.tensor_desc);
  definition.src_tensors.push_back(mask.tensor_desc);
  definition.dst_tensors.push_back(dst.tensor_desc);
  auto op = std::make_unique<::ml_drift::GPUOperation>(
      ::ml_drift::CreateGpuOperation(definition, std::move(op_desc),
                                     mask.tensor_desc.GetBHWCShape(),
                                     dst.tensor_desc.GetBHWCShape()));
  model_builder->AddGpuOperation({logits, mask}, {dst}, std::move(op),
                                 "sdpa_select_mask");
  return dst;
}

}  // namespace

bool SupportsFusedSdpaKernels(const ::ml_drift::GpuInfo& gpu_info) {
  return (gpu_info.IsApple() && gpu_info.IsApiMetal()) ||
         gpu_info.IsApiOpenCl();
}

absl::Status BuildSdpaTransposedGpuGraph(
    const std::vector<uint32_t>& input_ids, uint32_t output_id,
    const SdpaTransposedAttributes& attr,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (input_ids.size() < 3) {
    return absl::InvalidArgumentError(
        "SDPA transposed expects at least 3 inputs (Q, K, V).");
  }

  ABSL_ASSIGN_OR_RETURN(auto q, model_builder->GetTensor(input_ids[0]));
  ABSL_ASSIGN_OR_RETURN(auto k, model_builder->GetTensor(input_ids[1]));
  ABSL_ASSIGN_OR_RETURN(auto v, model_builder->GetTensor(input_ids[2]));

  const ::ml_drift::TensorDescriptor* mask_desc = nullptr;
  ::ml_drift::GpuModelBuilder::TensorHandle mask;
  const ::ml_drift::TensorDescriptor* param_desc = nullptr;
  ::ml_drift::GpuModelBuilder::TensorHandle param_tensor;

  if (input_ids.size() == 4) {
    ABSL_ASSIGN_OR_RETURN(auto mask_or_param,
                          model_builder->GetTensor(input_ids[3]));
    if (mask_or_param.tensor_desc.GetDataType() ==
        ::ml_drift::DataType::kInt32) {
      param_tensor = mask_or_param;
      param_desc = &param_tensor.tensor_desc;
    } else {
      mask = mask_or_param;
      mask_desc = &mask.tensor_desc;
    }
  } else if (input_ids.size() > 4) {
    ABSL_ASSIGN_OR_RETURN(mask, model_builder->GetTensor(input_ids[3]));
    mask_desc = &mask.tensor_desc;
    ABSL_ASSIGN_OR_RETURN(param_tensor, model_builder->GetTensor(input_ids[4]));
    param_desc = &param_tensor.tensor_desc;
  }

  const ::ml_drift::BHWC q_shape = q.tensor_desc.GetBHWCShape();
  const int head_dim = q_shape.c;
  const bool supports_fused_kernels =
      SupportsFusedSdpaKernels(model_builder->gpu_info());

  // The fused Flash-Attention prefill kernel indexes K and V directly in the
  // packed 4D layout produced by `odml.cache_update`, so it requires
  // `from_cache_update` and BUFFER storage. It also uses wave-matrix MMA and
  // 32-lane wave-SIMD intrinsics covering at most 32 channel slices
  // (head_dim <= 128). Everything else falls back to the multi-op graph below.
  const bool is_supported_flash_prefill =
      attr.is_prefill && attr.from_cache_update && head_dim % 4 == 0 &&
      head_dim <= 128 &&
      k.tensor_desc.GetStorageType() ==
          ::ml_drift::TensorStorageType::kBuffer &&
      v.tensor_desc.GetStorageType() ==
          ::ml_drift::TensorStorageType::kBuffer &&
      supports_fused_kernels &&
      SupportsWaveMatrixPrefill(model_builder->gpu_info());

  if (is_supported_flash_prefill) {
    auto dst =
        model_builder->AddTensor(q_shape, q.tensor_desc.GetDataType());
    auto op = CreateFusedFlashAttentionPrefill(
        model_builder->gpu_info(), q.tensor_desc, k.tensor_desc, v.tensor_desc,
        mask_desc, param_desc, dst.tensor_desc, attr);
    std::vector<::ml_drift::GpuModelBuilder::TensorHandle> src_tensors = {q, k,
                                                                          v};
    if (mask_desc) src_tensors.push_back(mask);
    if (param_desc) src_tensors.push_back(param_tensor);
    model_builder->AddGpuOperation(src_tensors, {dst}, std::move(op),
                                   "flash_prefill_sdpa");
    return model_builder->UpdateOutputTensor(dst, output_id);
  }

  const bool is_supported_flash_decode =
      attr.from_cache_update && !attr.is_prefill && supports_fused_kernels &&
      IsSupportedFlashDecode(model_builder->gpu_info(), q_shape, k.tensor_desc,
                             v.tensor_desc, mask_desc);

  if (is_supported_flash_decode) {
    const FlashDecodeConfig decode_config = GetFlashDecodeConfig(
        model_builder->gpu_info(), q_shape, k.tensor_desc.GetBHWCShape());
    ABSL_ASSIGN_OR_RETURN(auto output_ref, model_builder->GetTensor(output_id));
    const ::ml_drift::BHWC output_shape = output_ref.tensor_desc.GetBHWCShape();
    const bool is_flattened_dst =
        output_shape.h == 1 && output_shape.c == q_shape.h * q_shape.c;
    auto dst = model_builder->AddTensor(
        is_flattened_dst ? output_shape : q_shape, q.tensor_desc.GetDataType());
    std::vector<::ml_drift::GpuModelBuilder::TensorHandle> src_tensors = {q, k,
                                                                          v};
    if (mask_desc) src_tensors.push_back(mask);
    if (param_desc) src_tensors.push_back(param_tensor);
    if (decode_config.splits > 1) {
      const int slices = q_shape.c / 4;
      auto part = model_builder->AddTensor(
          ::ml_drift::BHWC(1, q_shape.h, q_shape.w * decode_config.splits,
                           (slices + 1) * 4),
          ::ml_drift::DataType::kFloat32);
      ABSL_ASSIGN_OR_RETURN(
          auto op, CreateFusedFlashDecodeSdpa(
                       model_builder->gpu_info(), q.tensor_desc, k.tensor_desc,
                       v.tensor_desc, mask_desc, param_desc, part.tensor_desc,
                       attr, is_flattened_dst, decode_config));
      model_builder->AddGpuOperation(src_tensors, {part}, std::move(op),
                                     "flash_decode_sdpa_split");
      auto combine = CreateFusedFlashDecodeCombine(
          part.tensor_desc, dst.tensor_desc, slices, q_shape.h,
          decode_config.splits, is_flattened_dst);
      model_builder->AddGpuOperation({part}, {dst}, std::move(combine),
                                     "flash_decode_sdpa_combine");
      return model_builder->UpdateOutputTensor(dst, output_id);
    }
    ABSL_ASSIGN_OR_RETURN(
        auto op, CreateFusedFlashDecodeSdpa(
                     model_builder->gpu_info(), q.tensor_desc, k.tensor_desc,
                     v.tensor_desc, mask_desc, param_desc, dst.tensor_desc,
                     attr, is_flattened_dst, decode_config));
    model_builder->AddGpuOperation(src_tensors, {dst}, std::move(op),
                                   "flash_decode_sdpa");
    return model_builder->UpdateOutputTensor(dst, output_id);
  }

  const auto k_shape = k.tensor_desc.GetBHWCShape();
  if (k_shape.h <= 0 || q_shape.h < k_shape.h || q_shape.h % k_shape.h != 0) {
    return absl::InvalidArgumentError(
        "Query heads must be a positive multiple of KV heads.");
  }
  const int gqa_ratio = q_shape.h / k_shape.h;

  ::ml_drift::GpuModelBuilder::TensorHandle q_for_bmm1 = q;
  // TODO(b/403337563): Remove this Reshape when storage type is BUFFER, since
  // folding gqa_ratio from H into W preserves the 1D DSHWBC4 buffer memory
  // layout.
  if (gqa_ratio > 1) {
    q_for_bmm1 = model_builder->Reshape(
        q, ::ml_drift::BHWC(q_shape.b, k_shape.h, gqa_ratio * q_shape.w,
                            q_shape.c));
  }

  ::ml_drift::GpuModelBuilder::TensorHandle logits;
  if (attr.from_cache_update) {
    ::ml_drift::WeightsDescription bmm1_desc = attr.bmm1_weights.desc;
    bmm1_desc.type = q.tensor_desc.GetDataType();
    const ::ml_drift::GpuModelBuilder::Weights bmm1_external_weights =
        ::ml_drift::CreateExternalWeights(k, bmm1_desc,
                                          attr.bmm1_weights.weights_shape);

    ::ml_drift::ConvRuntimeCheckDesc bmm1_runtime_check;
    if (param_desc) {
      bmm1_runtime_check.dst_end_ch_index = attr.runtime_check.src_end_ch_index;
    }

    ABSL_ASSIGN_OR_RETURN(
        logits, model_builder->FullyConnectedExternalWeights(
                    q_for_bmm1, bmm1_external_weights, /*biases=*/nullptr,
                    /*src_exp=*/nullptr, bmm1_runtime_check,
                    param_desc ? &param_tensor : nullptr));
  } else {
    ::ml_drift::BatchedMatMulAttributes bmm1_attr;
    bmm1_attr.transpose_left = false;
    bmm1_attr.transpose_right = true;
    ::ml_drift::ConvRuntimeCheckDesc bmm1_runtime_check;
    if (param_desc) {
      bmm1_runtime_check.dst_end_ch_index = attr.runtime_check.src_end_ch_index;
    }
    ABSL_ASSIGN_OR_RETURN(
        logits, model_builder->BatchedMatMul(
                    q_for_bmm1, k, bmm1_attr, /*src_exp=*/nullptr,
                    bmm1_runtime_check, param_desc ? &param_tensor : nullptr));
  }

  if (attr.softcap.has_value() && *attr.softcap > 0.0f) {
    const float cap_val = *attr.softcap;
    logits = model_builder->Multiplication(logits, 1.0f / cap_val);
    logits =
        model_builder->Elementwise(logits, ::ml_drift::OperationType::kTanh);
    logits = model_builder->Multiplication(logits, cap_val);
  }

  if (mask_desc != nullptr) {
    const auto mask_shape = mask.tensor_desc.GetBHWCShape();
    // With GQA the logits are folded to [b, kv_heads, gqa_ratio * T, keys]. A
    // [*, 1, T, keys] (or [*, 1, 1, keys]) mask applies to them directly: the
    // elementwise width broadcast reads mask row `w % T`, the query token of
    // folded row `w`. Other layouts (e.g. a mask per query head) need the
    // logits unfolded to [b, heads, T, keys].
    const bool mask_applies_to_folded_logits =
        mask_shape.h == 1 && (mask_shape.w == 1 || mask_shape.w == q_shape.w);
    const bool reshape_logits_for_mask =
        gqa_ratio > 1 && !mask_applies_to_folded_logits;
    if (reshape_logits_for_mask) {
      logits = model_builder->Reshape(
          logits, ::ml_drift::BHWC(q_shape.b, q_shape.h, q_shape.w, k_shape.w));
    }
    // Both ops are elementwise, so they get fused into the epilogue of the
    // operation producing `logits`.
    if (mask.tensor_desc.GetDataType() == ::ml_drift::DataType::kBool) {
      logits = SelectBoolMask(model_builder, logits, mask);
    } else {
      logits = model_builder->Add(logits, mask);
    }
    if (reshape_logits_for_mask) {
      logits = model_builder->Reshape(
          logits, ::ml_drift::BHWC(q_shape.b, k_shape.h, gqa_ratio * q_shape.w,
                                   k_shape.w));
    }
  }

  ::ml_drift::SoftmaxRuntimeCheckDesc softmax_runtime_check;
  if (param_desc) {
    softmax_runtime_check.end_ch_index = attr.runtime_check.src_end_ch_index;
  }

  ::ml_drift::GpuModelBuilder::TensorHandle output;
  if (attr.from_cache_update) {
    auto sfmx_partial = model_builder->SoftmaxReduce(
        logits, softmax_runtime_check, param_desc ? &param_tensor : nullptr);

    ::ml_drift::WeightsDescription bmm2_desc = attr.bmm2_weights.desc;
    bmm2_desc.type = logits.tensor_desc.GetDataType();
    const ::ml_drift::GpuModelBuilder::Weights bmm2_external_weights =
        ::ml_drift::CreateExternalWeights(v, bmm2_desc,
                                          attr.bmm2_weights.weights_shape);

    ::ml_drift::ConvRuntimeCheckDesc bmm2_runtime_check;
    if (param_desc) {
      bmm2_runtime_check.src_end_ch_index = attr.runtime_check.src_end_ch_index;
    }

    ABSL_ASSIGN_OR_RETURN(
        output,
        model_builder->FullyConnectedExternalWeights(
            logits, bmm2_external_weights, /*biases=*/nullptr, &sfmx_partial,
            bmm2_runtime_check, param_desc ? &param_tensor : nullptr));
  } else {
    auto probs = model_builder->Softmax(logits, softmax_runtime_check,
                                        param_desc ? &param_tensor : nullptr);

    ::ml_drift::BatchedMatMulAttributes bmm2_attr;
    bmm2_attr.transpose_left = false;
    bmm2_attr.transpose_right = true;
    ::ml_drift::ConvRuntimeCheckDesc bmm2_runtime_check;
    if (param_desc) {
      bmm2_runtime_check.src_end_ch_index = attr.runtime_check.src_end_ch_index;
    }
    ABSL_ASSIGN_OR_RETURN(
        output, model_builder->BatchedMatMul(
                    probs, v, bmm2_attr, /*src_exp=*/nullptr,
                    bmm2_runtime_check, param_desc ? &param_tensor : nullptr));
  }

  ABSL_ASSIGN_OR_RETURN(auto output_ref, model_builder->GetTensor(output_id));
  const auto output_shape = output_ref.tensor_desc.GetBHWCShape();
  if (output.tensor_desc.GetBHWCShape() != output_shape) {
    output = model_builder->Reshape(output, output_shape);
  }
  return model_builder->UpdateOutputTensor(output, output_id);
}

absl::Status CreateSdpaTransposedFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  const SdpaTransposedAttributes& attr =
      std::any_cast<const SdpaTransposedAttributes&>(node.operation.attributes);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildSdpaTransposedGpuGraph(input_ids, outputs[0]->id, attr,
                                     model_builder);
}

absl::Status CreateSdpaTransposedFromIrOp(
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& node,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const SdpaTransposedAttributes& attr =
      std::any_cast<const SdpaTransposedAttributes&>(node.attr);
  std::vector<uint32_t> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto* input : inputs) input_ids.push_back(input->id);
  return BuildSdpaTransposedGpuGraph(input_ids, outputs[0]->id, attr,
                                     model_builder);
}

}  // namespace litert::ml_drift
