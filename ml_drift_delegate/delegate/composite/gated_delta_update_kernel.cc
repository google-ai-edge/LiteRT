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

#include "ml_drift_delegate/delegate/composite/gated_delta_update_kernel.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/tensor_handle.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/gated_delta_update_parser.h"

namespace litert::ml_drift {
namespace {

// Subgroup size for the 2D register-only shuffle shader.
constexpr int kSubgroupSize = 32;

// Maximum number of unrolled D_k rounds per thread in the subgroup shader.
// Each round holds 4 StateType (float4) recurrent state vectors (16 floats),
// so capping k_rounds at 4 bounds state registers to at most 16 float4 (64
// VGPRs) per thread and prevents register spilling when D_k is large.
constexpr int kMaxKRounds = 4;

// Maximum workgroup size along X for the shared-memory fallback shader
// (supports up to D_k = 1024 = 256 vec4 slices; each thread holds only 1
// round = 4 float4 state registers).
constexpr int kMaxSharedMemWorkGroupSize = 256;

// Resolves `#pragma OPENCL EXTENSION ucl_wave_simd: enable` and the wave-SIMD
// intrinsics (`ucl::WaveSum`, `ucl::WaveMax`, `ucl::WaveBroadcast`,
// `ucl::WaveShuffle`, `ucl::WaveShuffleXor`) to the target shading language,
// following `ResolveWaveMemory` in `wave_memory_util.cc` and `ResolveWaveSimd`
// in `sdpa_transposed_kernel.cc`.
void ResolveWaveSimd(const ::ml_drift::GpuInfo& gpu_info, std::string* code) {
  const std::string kExtDecl =
      "#pragma OPENCL EXTENSION ucl_wave_simd: enable\n";
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
  absl::string_view wave_broadcast = "sub_group_broadcast";
  absl::string_view wave_shuffle = "sub_group_shuffle";
  absl::string_view wave_shuffle_xor = "sub_group_shuffle_xor";
  if (gpu_info.IsApiMetal()) {
    wave_sum = "simd_sum";
    wave_max = "simd_max";
    wave_broadcast = "simd_broadcast";
    wave_shuffle = "simd_shuffle";
    wave_shuffle_xor = "simd_shuffle_xor";
  } else if (gpu_info.IsGlsl() || gpu_info.IsApiWebGpu()) {
    wave_sum = "subgroupAdd";
    wave_max = "subgroupMax";
    wave_broadcast = "subgroupBroadcast";
    wave_shuffle = "subgroupShuffle";
    wave_shuffle_xor = "subgroupShuffleXor";
  }
  absl::StrReplaceAll({{"ucl::WaveSum", wave_sum},
                       {"ucl::WaveMax", wave_max},
                       {"ucl::WaveBroadcast", wave_broadcast},
                       {"ucl::WaveShuffleXor", wave_shuffle_xor}},
                      code);
  absl::StrReplaceAll({{"ucl::WaveShuffle", wave_shuffle}}, code);
}

// Whether the GPU supports 32-lane wave-SIMD vec4 shuffle operations
// (`ucl::WaveShuffle` and `ucl::WaveShuffleXor` on `float4` / `half4`) with a
// fixed subgroup size of 32.
bool SupportsWave32Vec4Shuffle(const ::ml_drift::GpuInfo& gpu_info) {
  if (gpu_info.IsApiMetal()) {
    return gpu_info.IsWaveSizeEqualTo32() ||
           gpu_info.supported_wave_sizes.empty();
  }
  if (gpu_info.IsApiWebGpu()) {
    return gpu_info.webgpu_info.supports_subgroups &&
           (gpu_info.IsWaveSizeEqualTo32() ||
            (gpu_info.webgpu_info.min_subgroup_size == 32 &&
             gpu_info.webgpu_info.max_subgroup_size == 32));
  }
  if (gpu_info.IsApiOpenCl()) {
    const bool has_subgroups =
        gpu_info.opencl_info.cl_version == ::ml_drift::OpenClVersion::kCl2_0 ||
        gpu_info.SupportsExtension("cl_khr_subgroups") ||
        gpu_info.SupportsExtension("cl_intel_subgroups");
    return gpu_info.IsWaveSizeEqualTo32() && has_subgroups &&
           gpu_info.SupportsExtension("cl_khr_subgroup_shuffle") &&
           gpu_info.SupportsExtension("cl_khr_subgroup_extended_types");
  }
  if (gpu_info.IsGlsl()) {
    return gpu_info.IsWaveSizeEqualTo32() &&
           gpu_info.SupportsExtension("GL_KHR_shader_subgroup_shuffle");
  }
  return false;
}

class GatedDeltaUpdateOp : public ::ml_drift::GPUOperation {
 public:
  GatedDeltaUpdateOp(int batch_size, int num_heads, int v_groups,
                     int work_group_size_x)
      : batch_size_(batch_size),
        num_heads_(num_heads),
        v_groups_(v_groups),
        work_group_size_x_(work_group_size_x) {
    // The shaders index threads by their local id and reduce across the
    // workgroup, so they need this work group size even without tuning.
    work_group_size_ = ::ml_drift::int3(work_group_size_x, 1, 1);
  }

  ::ml_drift::int3 GetGridSize() const override {
    // Grid: X=WorkGroupSizeX * Value Groups, Y=Height (Heads), Z=Batch
    return ::ml_drift::int3(work_group_size_x_ * v_groups_, num_heads_,
                            batch_size_);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {::ml_drift::int3(work_group_size_x_, 1, 1)};
  }

  GatedDeltaUpdateOp(GatedDeltaUpdateOp&& operation) = default;
  GatedDeltaUpdateOp& operator=(GatedDeltaUpdateOp&& operation) = default;
  GatedDeltaUpdateOp(const GatedDeltaUpdateOp&) = delete;
  GatedDeltaUpdateOp& operator=(const GatedDeltaUpdateOp&) = delete;

 private:
  int batch_size_ = 1;
  int num_heads_ = 1;
  int v_groups_ = 1;
  int work_group_size_x_ = 4;
};

// 2D (D_k x D_v) subgroup-parallel shader (register-only reduction, zero
// shared memory, zero barriers). Decomposes each subgroup of SUBGROUP_THREADS
// (= min(HEAD_K_DIM_SLICES, 32)) lanes into V_TILE groups of K_THREADS =
// SUBGROUP_THREADS / V_TILE lanes (e.g., 8 x 4 for D_k=128, V_TILE=4; 16 x 2
// for D_k=256, V_TILE=2; 32 x 1 for D_k=512, V_TILE=1):
//   - k_lane = tid % K_THREADS processes K_ROUNDS = HEAD_K_DIM_SLICES /
//     K_THREADS (bounded by kMaxKRounds = 4) strided rounds of 4 rows along
//     D_k (k_slice = k_lane + m * K_THREADS).
//   - v_lane = tid / K_THREADS owns 1 v_slice (4 columns along D_v).
// By scaling V_TILE down (4 -> 2 -> 1) when D_k > 128 and falling back to the
// shared-memory shader when D_k > 512, each thread holds at most 16 float4
// recurrent state registers (4 rounds x 1 v_slice = 64 VGPRs), preventing
// register spilling for large head dimensions. Each thread loads beta_t and g_t
// uniformly, loads its own v_slice directly without divergent branches or
// broadcasts, coalesces k_t/q_t global loads across all lanes (distributing via
// ucl::WaveShuffle -> simd_shuffle / subgroupShuffle / sub_group_shuffle), and
// reduces a single float4 kv_mem and my_attn_out across K_THREADS lanes via
// log2(K_THREADS) ucl::WaveShuffleXor -> simd_shuffle_xor / subgroupShuffleXor
// / sub_group_shuffle_xor butterfly steps.
//
// Final rendered shader example — Variant 1 (Metal, D_k=128, D_v=128, V_TILE=4,
// K_THREADS=8, SEQ_LEN_EXPR=128, GQA_RATIO=3, StateType=float4,
// ActivationType=float4):
// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// ```msl
// MAIN_FUNCTION($0) {
//   int tid = ucl::GetLocalId<0>();
//   int k_lane = tid % 8;
//   int v_lane = tid / 8;
//   int v_slice = ucl::GetGroupId<0>() * 4 + v_lane;
//   int h = ucl::GetGroupId<1>();
//   int h_k = h / 3;
//   int b = ucl::GetGroupId<2>();
//
//   int k_base_0 = k_lane * 4;
//   float4 s0_0 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_0 + 0, h, v_slice, b));
//   float4 s1_0 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_0 + 1, h, v_slice, b));
//   float4 s2_0 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_0 + 2, h, v_slice, b));
//   float4 s3_0 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_0 + 3, h, v_slice, b));
//   int k_base_1 = (k_lane + 1 * 8) * 4;
//   float4 s0_1 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_1 + 0, h, v_slice, b));
//   float4 s1_1 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_1 + 1, h, v_slice, b));
//   float4 s2_1 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_1 + 2, h, v_slice, b));
//   float4 s3_1 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_1 + 3, h, v_slice, b));
//   int k_base_2 = (k_lane + 2 * 8) * 4;
//   float4 s0_2 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_2 + 0, h, v_slice, b));
//   float4 s1_2 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_2 + 1, h, v_slice, b));
//   float4 s2_2 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_2 + 2, h, v_slice, b));
//   float4 s3_2 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_2 + 3, h, v_slice, b));
//   int k_base_3 = (k_lane + 3 * 8) * 4;
//   float4 s0_3 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_3 + 0, h, v_slice, b));
//   float4 s1_3 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_3 + 1, h, v_slice, b));
//   float4 s2_3 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_3 + 2, h, v_slice, b));
//   float4 s3_3 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base_3 + 3, h, v_slice, b));
//
//   int valid_len = clamp(args.valid_len.Read<int>(0, 0, 0, b).x, 0, 128);
//
//   // Process active sequence length sequentially
//   for (int t = 0; t < valid_len; ++t) {
//     // Uniform load across subgroup: all lanes read identical (h, 0, t_slice, b)
//     int t_slice = t / 4;
//     int t_elem = t % 4;
//     float4 beta_t_vec = ucl::Convert<float4>(args.beta_t.Read(h, 0, t_slice, b));
//     float beta_val = beta_t_vec.x;
//     if (t_elem == 1) {
//       beta_val = beta_t_vec.y;
//     } else if (t_elem == 2) {
//       beta_val = beta_t_vec.z;
//     } else if (t_elem == 3) {
//       beta_val = beta_t_vec.w;
//     }
//
//     float4 g_t_vec = ucl::Convert<float4>(args.g_t.Read(h, 0, t_slice, b));
//     float g_val = g_t_vec.x;
//     if (t_elem == 1) {
//       g_val = g_t_vec.y;
//     } else if (t_elem == 2) {
//       g_val = g_t_vec.z;
//     } else if (t_elem == 3) {
//       g_val = g_t_vec.w;
//     }
//
//     float decay_scalar = exp(ucl::Convert<float>(g_val));
//     float4 decay_vec = ucl::Init<float4>(decay_scalar);
//
//     s0_0 = s0_0 * decay_vec;
//     s1_0 = s1_0 * decay_vec;
//     s2_0 = s2_0 * decay_vec;
//     s3_0 = s3_0 * decay_vec;
//     s0_1 = s0_1 * decay_vec;
//     s1_1 = s1_1 * decay_vec;
//     s2_1 = s2_1 * decay_vec;
//     s3_1 = s3_1 * decay_vec;
//     s0_2 = s0_2 * decay_vec;
//     s1_2 = s1_2 * decay_vec;
//     s2_2 = s2_2 * decay_vec;
//     s3_2 = s3_2 * decay_vec;
//     s0_3 = s0_3 * decay_vec;
//     s1_3 = s1_3 * decay_vec;
//     s2_3 = s2_3 * decay_vec;
//     s3_3 = s3_3 * decay_vec;
//
//     // Coalesced chunked load of full D_k across 32 lanes
//     float4 k_loaded_0 = ucl::Convert<float4>(args.k_t.Read(t, h_k, tid, b));
//
//     float4 kv_mem = ucl::Init<float4>(0.0);
//     {
//       float4 k_val_0 = simd_shuffle(k_loaded_0, ucl::Convert<uint>(k_lane));
//       kv_mem += s0_0 * ucl::Init<float4>(k_val_0.x) +
//                 s1_0 * ucl::Init<float4>(k_val_0.y) +
//                 s2_0 * ucl::Init<float4>(k_val_0.z) +
//                 s3_0 * ucl::Init<float4>(k_val_0.w);
//     }
//     {
//       float4 k_val_1 = simd_shuffle(k_loaded_0, ucl::Convert<uint>(k_lane + 1 * 8));
//       kv_mem += s0_1 * ucl::Init<float4>(k_val_1.x) +
//                 s1_1 * ucl::Init<float4>(k_val_1.y) +
//                 s2_1 * ucl::Init<float4>(k_val_1.z) +
//                 s3_1 * ucl::Init<float4>(k_val_1.w);
//     }
//     {
//       float4 k_val_2 = simd_shuffle(k_loaded_0, ucl::Convert<uint>(k_lane + 2 * 8));
//       kv_mem += s0_2 * ucl::Init<float4>(k_val_2.x) +
//                 s1_2 * ucl::Init<float4>(k_val_2.y) +
//                 s2_2 * ucl::Init<float4>(k_val_2.z) +
//                 s3_2 * ucl::Init<float4>(k_val_2.w);
//     }
//     {
//       float4 k_val_3 = simd_shuffle(k_loaded_0, ucl::Convert<uint>(k_lane + 3 * 8));
//       kv_mem += s0_3 * ucl::Init<float4>(k_val_3.x) +
//                 s1_3 * ucl::Init<float4>(k_val_3.y) +
//                 s2_3 * ucl::Init<float4>(k_val_3.z) +
//                 s3_3 * ucl::Init<float4>(k_val_3.w);
//     }
//
//     // Butterfly reduction across K_THREADS lanes sharing the same v_lane
//     kv_mem += simd_shuffle_xor(kv_mem, 1u);
//     kv_mem += simd_shuffle_xor(kv_mem, 2u);
//     kv_mem += simd_shuffle_xor(kv_mem, 4u);
//
//     // Each v_lane group loads its own v_slice directly (no branch or broadcast)
//     float4 v_vec = ucl::Convert<float4>(args.v_t.Read(t, h, v_slice, b));
//
//     float4 beta_factor = ucl::Init<float4>(ucl::Convert<float>(beta_val));
//     float4 delta_slice = (v_vec - kv_mem) * beta_factor;
//
//     {
//       float4 k_val_0 = simd_shuffle(k_loaded_0, ucl::Convert<uint>(k_lane));
//       s0_0 = s0_0 + delta_slice * ucl::Init<float4>(k_val_0.x);
//       s1_0 = s1_0 + delta_slice * ucl::Init<float4>(k_val_0.y);
//       s2_0 = s2_0 + delta_slice * ucl::Init<float4>(k_val_0.z);
//       s3_0 = s3_0 + delta_slice * ucl::Init<float4>(k_val_0.w);
//     }
//     {
//       float4 k_val_1 = simd_shuffle(k_loaded_0, ucl::Convert<uint>(k_lane + 1 * 8));
//       s0_1 = s0_1 + delta_slice * ucl::Init<float4>(k_val_1.x);
//       s1_1 = s1_1 + delta_slice * ucl::Init<float4>(k_val_1.y);
//       s2_1 = s2_1 + delta_slice * ucl::Init<float4>(k_val_1.z);
//       s3_1 = s3_1 + delta_slice * ucl::Init<float4>(k_val_1.w);
//     }
//     {
//       float4 k_val_2 = simd_shuffle(k_loaded_0, ucl::Convert<uint>(k_lane + 2 * 8));
//       s0_2 = s0_2 + delta_slice * ucl::Init<float4>(k_val_2.x);
//       s1_2 = s1_2 + delta_slice * ucl::Init<float4>(k_val_2.y);
//       s2_2 = s2_2 + delta_slice * ucl::Init<float4>(k_val_2.z);
//       s3_2 = s3_2 + delta_slice * ucl::Init<float4>(k_val_2.w);
//     }
//     {
//       float4 k_val_3 = simd_shuffle(k_loaded_0, ucl::Convert<uint>(k_lane + 3 * 8));
//       s0_3 = s0_3 + delta_slice * ucl::Init<float4>(k_val_3.x);
//       s1_3 = s1_3 + delta_slice * ucl::Init<float4>(k_val_3.y);
//       s2_3 = s2_3 + delta_slice * ucl::Init<float4>(k_val_3.z);
//       s3_3 = s3_3 + delta_slice * ucl::Init<float4>(k_val_3.w);
//     }
//
//     // Coalesced chunked load of full D_k Q across 32 lanes
//     float4 q_loaded_0 = ucl::Convert<float4>(args.q_t.Read(t, h_k, tid, b));
//
//     float4 my_attn_out = ucl::Init<float4>(0.0);
//     {
//       float4 q_val_0 = simd_shuffle(q_loaded_0, ucl::Convert<uint>(k_lane));
//       my_attn_out += s0_0 * ucl::Init<float4>(q_val_0.x) +
//                      s1_0 * ucl::Init<float4>(q_val_0.y) +
//                      s2_0 * ucl::Init<float4>(q_val_0.z) +
//                      s3_0 * ucl::Init<float4>(q_val_0.w);
//     }
//     {
//       float4 q_val_1 = simd_shuffle(q_loaded_0, ucl::Convert<uint>(k_lane + 1 * 8));
//       my_attn_out += s0_1 * ucl::Init<float4>(q_val_1.x) +
//                      s1_1 * ucl::Init<float4>(q_val_1.y) +
//                      s2_1 * ucl::Init<float4>(q_val_1.z) +
//                      s3_1 * ucl::Init<float4>(q_val_1.w);
//     }
//     {
//       float4 q_val_2 = simd_shuffle(q_loaded_0, ucl::Convert<uint>(k_lane + 2 * 8));
//       my_attn_out += s0_2 * ucl::Init<float4>(q_val_2.x) +
//                      s1_2 * ucl::Init<float4>(q_val_2.y) +
//                      s2_2 * ucl::Init<float4>(q_val_2.z) +
//                      s3_2 * ucl::Init<float4>(q_val_2.w);
//     }
//     {
//       float4 q_val_3 = simd_shuffle(q_loaded_0, ucl::Convert<uint>(k_lane + 3 * 8));
//       my_attn_out += s0_3 * ucl::Init<float4>(q_val_3.x) +
//                      s1_3 * ucl::Init<float4>(q_val_3.y) +
//                      s2_3 * ucl::Init<float4>(q_val_3.z) +
//                      s3_3 * ucl::Init<float4>(q_val_3.w);
//     }
//
//     // Butterfly reduction across K_THREADS lanes sharing the same v_lane
//     my_attn_out += simd_shuffle_xor(my_attn_out, 1u);
//     my_attn_out += simd_shuffle_xor(my_attn_out, 2u);
//     my_attn_out += simd_shuffle_xor(my_attn_out, 4u);
//
//     if (k_lane == 0) {
//       args.output.Write(ucl::Convert<float4>(my_attn_out), t, h, v_slice, b);
//     }
//   }
//
//   if (k_lane == 0) {
//     for (int t = valid_len; t < 128; ++t) {
//       args.output.Write(ucl::Init<float4>(0.0), t, h, v_slice, b);
//     }
//   }
//
//   // Write out final evolved recurrent state for this thread's owned rows and v_slice
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s0_0), k_base_0 + 0, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s1_0), k_base_0 + 1, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s2_0), k_base_0 + 2, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s3_0), k_base_0 + 3, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s0_1), k_base_1 + 0, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s1_1), k_base_1 + 1, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s2_1), k_base_1 + 2, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s3_1), k_base_1 + 3, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s0_2), k_base_2 + 0, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s1_2), k_base_2 + 1, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s2_2), k_base_2 + 2, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s3_2), k_base_2 + 3, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s0_3), k_base_3 + 0, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s1_3), k_base_3 + 1, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s2_3), k_base_3 + 2, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s3_3), k_base_3 + 3, h, v_slice, b);
// }
// ```
//
// Final rendered shader example — Variant 2 (WebGPU WGSL, D_k=128, D_v=128,
// V_TILE=4, K_THREADS=8, SEQ_LEN_EXPR=128, GQA_RATIO=3, StateType=vec4<f32>,
// ActivationType=vec4<f32>):
// ```wgsl
// enable subgroups;
// MAIN_FUNCTION($0) {
//   int tid = ucl::GetLocalId<0>();
//   int k_lane = tid % 8;
//   int v_lane = tid / 8;
//   int v_slice = ucl::GetGroupId<0>() * 4 + v_lane;
//   int h = ucl::GetGroupId<1>();
//   int h_k = h / 3;
//   int b = ucl::GetGroupId<2>();
//
//   int k_base_0 = k_lane * 4;
//   vec4<f32> s0_0 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_0 + 0, h, v_slice, b));
//   vec4<f32> s1_0 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_0 + 1, h, v_slice, b));
//   vec4<f32> s2_0 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_0 + 2, h, v_slice, b));
//   vec4<f32> s3_0 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_0 + 3, h, v_slice, b));
//   int k_base_1 = (k_lane + 1 * 8) * 4;
//   vec4<f32> s0_1 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_1 + 0, h, v_slice, b));
//   vec4<f32> s1_1 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_1 + 1, h, v_slice, b));
//   vec4<f32> s2_1 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_1 + 2, h, v_slice, b));
//   vec4<f32> s3_1 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_1 + 3, h, v_slice, b));
//   int k_base_2 = (k_lane + 2 * 8) * 4;
//   vec4<f32> s0_2 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_2 + 0, h, v_slice, b));
//   vec4<f32> s1_2 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_2 + 1, h, v_slice, b));
//   vec4<f32> s2_2 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_2 + 2, h, v_slice, b));
//   vec4<f32> s3_2 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_2 + 3, h, v_slice, b));
//   int k_base_3 = (k_lane + 3 * 8) * 4;
//   vec4<f32> s0_3 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_3 + 0, h, v_slice, b));
//   vec4<f32> s1_3 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_3 + 1, h, v_slice, b));
//   vec4<f32> s2_3 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_3 + 2, h, v_slice, b));
//   vec4<f32> s3_3 = ucl::Convert<vec4<f32>>(args.recurrent_state_out.Read(k_base_3 + 3, h, v_slice, b));
//
//   int valid_len = clamp(args.valid_len.Read<int>(0, 0, 0, b).x, 0, 128);
//
//   // Process active sequence length sequentially
//   for (int t = 0; t < valid_len; ++t) {
//     // Uniform load across subgroup: all lanes read identical (h, 0, t_slice, b)
//     int t_slice = t / 4;
//     int t_elem = t % 4;
//     vec4<f32> beta_t_vec = ucl::Convert<vec4<f32>>(args.beta_t.Read(h, 0, t_slice, b));
//     f32 beta_val = beta_t_vec.x;
//     if (t_elem == 1) {
//       beta_val = beta_t_vec.y;
//     } else if (t_elem == 2) {
//       beta_val = beta_t_vec.z;
//     } else if (t_elem == 3) {
//       beta_val = beta_t_vec.w;
//     }
//
//     vec4<f32> g_t_vec = ucl::Convert<vec4<f32>>(args.g_t.Read(h, 0, t_slice, b));
//     f32 g_val = g_t_vec.x;
//     if (t_elem == 1) {
//       g_val = g_t_vec.y;
//     } else if (t_elem == 2) {
//       g_val = g_t_vec.z;
//     } else if (t_elem == 3) {
//       g_val = g_t_vec.w;
//     }
//
//     f32 decay_scalar = exp(ucl::Convert<f32>(g_val));
//     vec4<f32> decay_vec = ucl::Init<vec4<f32>>(decay_scalar);
//
//     s0_0 = s0_0 * decay_vec;
//     s1_0 = s1_0 * decay_vec;
//     s2_0 = s2_0 * decay_vec;
//     s3_0 = s3_0 * decay_vec;
//     s0_1 = s0_1 * decay_vec;
//     s1_1 = s1_1 * decay_vec;
//     s2_1 = s2_1 * decay_vec;
//     s3_1 = s3_1 * decay_vec;
//     s0_2 = s0_2 * decay_vec;
//     s1_2 = s1_2 * decay_vec;
//     s2_2 = s2_2 * decay_vec;
//     s3_2 = s3_2 * decay_vec;
//     s0_3 = s0_3 * decay_vec;
//     s1_3 = s1_3 * decay_vec;
//     s2_3 = s2_3 * decay_vec;
//     s3_3 = s3_3 * decay_vec;
//
//     // Coalesced load of full D_k across 32 lanes (1 vec4 per lane)
//     vec4<f32> k_loaded = ucl::Convert<vec4<f32>>(args.k_t.Read(t, h_k, tid, b));
//
//     vec4<f32> kv_mem = ucl::Init<vec4<f32>>(0.0);
//     {
//       vec4<f32> k_val_0 = subgroupShuffle(k_loaded, ucl::Convert<uint>(k_lane));
//       kv_mem += s0_0 * ucl::Init<vec4<f32>>(k_val_0.x) +
//                 s1_0 * ucl::Init<vec4<f32>>(k_val_0.y) +
//                 s2_0 * ucl::Init<vec4<f32>>(k_val_0.z) +
//                 s3_0 * ucl::Init<vec4<f32>>(k_val_0.w);
//     }
//     {
//       vec4<f32> k_val_1 = subgroupShuffle(k_loaded, ucl::Convert<uint>(k_lane + 1 * 8));
//       kv_mem += s0_1 * ucl::Init<vec4<f32>>(k_val_1.x) +
//                 s1_1 * ucl::Init<vec4<f32>>(k_val_1.y) +
//                 s2_1 * ucl::Init<vec4<f32>>(k_val_1.z) +
//                 s3_1 * ucl::Init<vec4<f32>>(k_val_1.w);
//     }
//     {
//       vec4<f32> k_val_2 = subgroupShuffle(k_loaded, ucl::Convert<uint>(k_lane + 2 * 8));
//       kv_mem += s0_2 * ucl::Init<vec4<f32>>(k_val_2.x) +
//                 s1_2 * ucl::Init<vec4<f32>>(k_val_2.y) +
//                 s2_2 * ucl::Init<vec4<f32>>(k_val_2.z) +
//                 s3_2 * ucl::Init<vec4<f32>>(k_val_2.w);
//     }
//     {
//       vec4<f32> k_val_3 = subgroupShuffle(k_loaded, ucl::Convert<uint>(k_lane + 3 * 8));
//       kv_mem += s0_3 * ucl::Init<vec4<f32>>(k_val_3.x) +
//                 s1_3 * ucl::Init<vec4<f32>>(k_val_3.y) +
//                 s2_3 * ucl::Init<vec4<f32>>(k_val_3.z) +
//                 s3_3 * ucl::Init<vec4<f32>>(k_val_3.w);
//     }
//
//     // Butterfly reduction across K_THREADS lanes sharing the same v_lane
//     kv_mem += subgroupShuffleXor(kv_mem, 1u);
//     kv_mem += subgroupShuffleXor(kv_mem, 2u);
//     kv_mem += subgroupShuffleXor(kv_mem, 4u);
//
//     // Each v_lane group loads its own v_slice directly (no branch or broadcast)
//     vec4<f32> v_vec = ucl::Convert<vec4<f32>>(args.v_t.Read(t, h, v_slice, b));
//
//     vec4<f32> beta_factor = ucl::Init<vec4<f32>>(ucl::Convert<f32>(beta_val));
//     vec4<f32> delta_slice = (v_vec - kv_mem) * beta_factor;
//
//     {
//       vec4<f32> k_val_0 = subgroupShuffle(k_loaded, ucl::Convert<uint>(k_lane));
//       s0_0 = s0_0 + delta_slice * ucl::Init<vec4<f32>>(k_val_0.x);
//       s1_0 = s1_0 + delta_slice * ucl::Init<vec4<f32>>(k_val_0.y);
//       s2_0 = s2_0 + delta_slice * ucl::Init<vec4<f32>>(k_val_0.z);
//       s3_0 = s3_0 + delta_slice * ucl::Init<vec4<f32>>(k_val_0.w);
//     }
//     {
//       vec4<f32> k_val_1 = subgroupShuffle(k_loaded, ucl::Convert<uint>(k_lane + 1 * 8));
//       s0_1 = s0_1 + delta_slice * ucl::Init<vec4<f32>>(k_val_1.x);
//       s1_1 = s1_1 + delta_slice * ucl::Init<vec4<f32>>(k_val_1.y);
//       s2_1 = s2_1 + delta_slice * ucl::Init<vec4<f32>>(k_val_1.z);
//       s3_1 = s3_1 + delta_slice * ucl::Init<vec4<f32>>(k_val_1.w);
//     }
//     {
//       vec4<f32> k_val_2 = subgroupShuffle(k_loaded, ucl::Convert<uint>(k_lane + 2 * 8));
//       s0_2 = s0_2 + delta_slice * ucl::Init<vec4<f32>>(k_val_2.x);
//       s1_2 = s1_2 + delta_slice * ucl::Init<vec4<f32>>(k_val_2.y);
//       s2_2 = s2_2 + delta_slice * ucl::Init<vec4<f32>>(k_val_2.z);
//       s3_2 = s3_2 + delta_slice * ucl::Init<vec4<f32>>(k_val_2.w);
//     }
//     {
//       vec4<f32> k_val_3 = subgroupShuffle(k_loaded, ucl::Convert<uint>(k_lane + 3 * 8));
//       s0_3 = s0_3 + delta_slice * ucl::Init<vec4<f32>>(k_val_3.x);
//       s1_3 = s1_3 + delta_slice * ucl::Init<vec4<f32>>(k_val_3.y);
//       s2_3 = s2_3 + delta_slice * ucl::Init<vec4<f32>>(k_val_3.z);
//       s3_3 = s3_3 + delta_slice * ucl::Init<vec4<f32>>(k_val_3.w);
//     }
//
//     // Coalesced load of full D_k Q across 32 lanes (1 vec4 per lane)
//     vec4<f32> q_loaded = ucl::Convert<vec4<f32>>(args.q_t.Read(t, h_k, tid, b));
//
//     vec4<f32> my_attn_out = ucl::Init<vec4<f32>>(0.0);
//     {
//       vec4<f32> q_val_0 = subgroupShuffle(q_loaded, ucl::Convert<uint>(k_lane));
//       my_attn_out += s0_0 * ucl::Init<vec4<f32>>(q_val_0.x) +
//                      s1_0 * ucl::Init<vec4<f32>>(q_val_0.y) +
//                      s2_0 * ucl::Init<vec4<f32>>(q_val_0.z) +
//                      s3_0 * ucl::Init<vec4<f32>>(q_val_0.w);
//     }
//     {
//       vec4<f32> q_val_1 = subgroupShuffle(q_loaded, ucl::Convert<uint>(k_lane + 1 * 8));
//       my_attn_out += s0_1 * ucl::Init<vec4<f32>>(q_val_1.x) +
//                      s1_1 * ucl::Init<vec4<f32>>(q_val_1.y) +
//                      s2_1 * ucl::Init<vec4<f32>>(q_val_1.z) +
//                      s3_1 * ucl::Init<vec4<f32>>(q_val_1.w);
//     }
//     {
//       vec4<f32> q_val_2 = subgroupShuffle(q_loaded, ucl::Convert<uint>(k_lane + 2 * 8));
//       my_attn_out += s0_2 * ucl::Init<vec4<f32>>(q_val_2.x) +
//                      s1_2 * ucl::Init<vec4<f32>>(q_val_2.y) +
//                      s2_2 * ucl::Init<vec4<f32>>(q_val_2.z) +
//                      s3_2 * ucl::Init<vec4<f32>>(q_val_2.w);
//     }
//     {
//       vec4<f32> q_val_3 = subgroupShuffle(q_loaded, ucl::Convert<uint>(k_lane + 3 * 8));
//       my_attn_out += s0_3 * ucl::Init<vec4<f32>>(q_val_3.x) +
//                      s1_3 * ucl::Init<vec4<f32>>(q_val_3.y) +
//                      s2_3 * ucl::Init<vec4<f32>>(q_val_3.z) +
//                      s3_3 * ucl::Init<vec4<f32>>(q_val_3.w);
//     }
//
//     // Butterfly reduction across K_THREADS lanes sharing the same v_lane
//     my_attn_out += subgroupShuffleXor(my_attn_out, 1u);
//     my_attn_out += subgroupShuffleXor(my_attn_out, 2u);
//     my_attn_out += subgroupShuffleXor(my_attn_out, 4u);
//
//     if (k_lane == 0) {
//       args.output.Write(ucl::Convert<vec4<f32>>(my_attn_out), t, h, v_slice, b);
//     }
//   }
//
//   if (k_lane == 0) {
//     for (int t = valid_len; t < 128; ++t) {
//       args.output.Write(ucl::Init<vec4<f32>>(0.0), t, h, v_slice, b);
//     }
//   }
//
//   // Write out final evolved recurrent state for this thread's owned rows and v_slice
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s0_0), k_base_0 + 0, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s1_0), k_base_0 + 1, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s2_0), k_base_0 + 2, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s3_0), k_base_0 + 3, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s0_1), k_base_1 + 0, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s1_1), k_base_1 + 1, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s2_1), k_base_1 + 2, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s3_1), k_base_1 + 3, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s0_2), k_base_2 + 0, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s1_2), k_base_2 + 1, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s2_2), k_base_2 + 2, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s3_2), k_base_2 + 3, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s0_3), k_base_3 + 0, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s1_3), k_base_3 + 1, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s2_3), k_base_3 + 2, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<vec4<f32>>(s3_3), k_base_3 + 3, h, v_slice, b);
// }
// ```
// NOLINTEND(whitespace/line_length)
// clang-format on
constexpr char kGatedDeltaUpdateShuffleShader[] = R"(
#pragma OPENCL EXTENSION ucl_wave_simd: enable
MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int k_lane = tid % K_THREADS;
  int v_lane = tid / K_THREADS;
  int v_slice = ucl::GetGroupId<0>() * V_TILE + v_lane;
  int h = ucl::GetGroupId<1>();
  int h_k = h / GQA_RATIO;
  int b = ucl::GetGroupId<2>();

  STATE_INIT_DECLS
  VALID_LEN_INIT

  // Process active sequence length sequentially
  for (int t = 0; t < LOOP_BOUND_EXPR; ++t) {
    // Uniform load across subgroup: all lanes read identical (h, 0, t_slice, b)
    int t_slice = t / 4;
    int t_elem = t % 4;
    ActivationType beta_t_vec = ucl::Convert<ActivationType>(args.beta_t.Read(h, 0, t_slice, b));
    ActivationScalarType beta_val = beta_t_vec.x;
    if (t_elem == 1) {
      beta_val = beta_t_vec.y;
    } else if (t_elem == 2) {
      beta_val = beta_t_vec.z;
    } else if (t_elem == 3) {
      beta_val = beta_t_vec.w;
    }

    ActivationType g_t_vec = ucl::Convert<ActivationType>(args.g_t.Read(h, 0, t_slice, b));
    ActivationScalarType g_val = g_t_vec.x;
    if (t_elem == 1) {
      g_val = g_t_vec.y;
    } else if (t_elem == 2) {
      g_val = g_t_vec.z;
    } else if (t_elem == 3) {
      g_val = g_t_vec.w;
    }

    StateScalarType decay_scalar = exp(ucl::Convert<StateScalarType>(g_val));
    StateType decay_vec = ucl::Init<StateType>(decay_scalar);

    DECAY_STEP

    K_LOAD_STEP

    StateType kv_mem = ucl::Init<StateType>(0.0);
    KV_MEM_STEP

    // Butterfly reduction across K_THREADS lanes sharing the same v_lane
    SIMD_REDUCE_KV_MEM

    // Each v_lane group loads its own v_slice directly (no branch or broadcast)
    StateType v_vec = ucl::Convert<StateType>(args.v_t.Read(t, h, v_slice, b));

    StateType beta_factor = ucl::Init<StateType>(ucl::Convert<StateScalarType>(beta_val));
    StateType delta_slice = (v_vec - kv_mem) * beta_factor;

    UPDATE_STATE_STEP

    Q_LOAD_STEP

    StateType my_attn_out = ucl::Init<StateType>(0.0);
    ATTN_OUT_STEP

    // Butterfly reduction across K_THREADS lanes sharing the same v_lane
    SIMD_REDUCE_ATTN_OUT

    if (k_lane == 0) {
      args.output.Write(ucl::Convert<ActivationType>(my_attn_out), t, h, v_slice, b);
    }
  }

  ZERO_PAD_TAIL_STEP

  // Write out final evolved recurrent state for this thread's owned rows and v_slice
  STATE_WRITE_OUT_STEP
}
)";

// Shared memory-based shader (single-phase 2-barrier reduction for OpenCL /
// WebGPU without 32-lane subgroups, or when D_k > 512 exceeds the 4-round
// per-thread register budget of a single 32-lane subgroup).
//
// Final rendered shader example — Variant 3 (Shared-Memory Fallback, D_k=128,
// D_v=128, HEAD_K_DIM_SLICES=32, SEQ_LEN_EXPR=128, GQA_RATIO=3,
// StateType=float4, ActivationType=float4):
// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// ```cl
// MAIN_FUNCTION($0) {
//   int k_slice = ucl::GetLocalId<0>();
//   int v_slice = ucl::GetGroupId<0>();
//   int h = ucl::GetGroupId<1>();
//   int h_k = h / 3;
//   int b = ucl::GetGroupId<2>();
//
//   __local float4 scratch_kv[32];
//   __local float4 scratch_out[32];
//   __local float scratch_kq[32];
//
//   // Each thread owns 4 rows along D_k (k_slice * 4 + {0, 1, 2, 3}) for this v_slice.
//   // Held entirely in 4 registers. Zero register spilling.
//   int k_base = k_slice * 4;
//   float4 s0 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 0, h, v_slice, b));
//   float4 s1 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 1, h, v_slice, b));
//   float4 s2 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 2, h, v_slice, b));
//   float4 s3 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 3, h, v_slice, b));
//   int valid_len = clamp(args.valid_len.Read<int>(0, 0, 0, b).x, 0, 128);
//
//   // Process active sequence length sequentially
//   for (int t = 0; t < valid_len; ++t) {
//     // Read beta and g for this step (mapped from TFLite 3D shape [B, H, L] to MLDrift BHWC [B, 1, H, L])
//     int t_slice = t / 4;
//     int t_elem = t % 4;
//     float4 beta_t_vec = ucl::Convert<float4>(args.beta_t.Read(h, 0, t_slice, b));
//     float beta_val = beta_t_vec.x;
//     if (t_elem == 1) {
//       beta_val = beta_t_vec.y;
//     } else if (t_elem == 2) {
//       beta_val = beta_t_vec.z;
//     } else if (t_elem == 3) {
//       beta_val = beta_t_vec.w;
//     }
//
//     float4 g_t_vec = ucl::Convert<float4>(args.g_t.Read(h, 0, t_slice, b));
//     float g_val = g_t_vec.x;
//     if (t_elem == 1) {
//       g_val = g_t_vec.y;
//     } else if (t_elem == 2) {
//       g_val = g_t_vec.z;
//     } else if (t_elem == 3) {
//       g_val = g_t_vec.w;
//     }
//
//     float decay_scalar = exp(ucl::Convert<float>(g_val));
//     float4 decay_vec = ucl::Init<float4>(decay_scalar);
//
//     // Apply decay to this thread's 4 rows
//     s0 = s0 * decay_vec;
//     s1 = s1 * decay_vec;
//     s2 = s2 * decay_vec;
//     s3 = s3 * decay_vec;
//
//     // Load full vec4 K and Q for this thread's k_slice directly
//     float4 k_val = ucl::Convert<float4>(args.k_t.Read(t, h_k, k_slice, b));
//     float4 q_val = ucl::Convert<float4>(args.q_t.Read(t, h_k, k_slice, b));
//
//     // Compute partial dot products before barrier
//     float4 kv_mem = s0 * ucl::Init<float4>(k_val.x) +
//                     s1 * ucl::Init<float4>(k_val.y) +
//                     s2 * ucl::Init<float4>(k_val.z) +
//                     s3 * ucl::Init<float4>(k_val.w);
//     float4 sq_mem = s0 * ucl::Init<float4>(q_val.x) +
//                     s1 * ucl::Init<float4>(q_val.y) +
//                     s2 * ucl::Init<float4>(q_val.z) +
//                     s3 * ucl::Init<float4>(q_val.w);
//     float kq_dot = dot(k_val, q_val);
//
//     scratch_kv[k_slice] = kv_mem;
//     scratch_out[k_slice] = sq_mem;
//     scratch_kq[k_slice] = kq_dot;
//     ucl::SyncThreads<WorkGroup, Local>();
//     // Fold the 32 partial sums across 4 quarters (lanes 0..7 accumulate lanes
//     // k_slice + {0, 8, 16, 24}) in a single barrier step instead of 5 tree-reduction
//     // barriers, then sum the remaining 8 partial sums directly.
//     if (k_slice < 8) {
//       scratch_kv[k_slice] += scratch_kv[k_slice + 8] +
//                              scratch_kv[k_slice + 16] +
//                              scratch_kv[k_slice + 24];
//       scratch_out[k_slice] += scratch_out[k_slice + 8] +
//                               scratch_out[k_slice + 16] +
//                               scratch_out[k_slice + 24];
//       scratch_kq[k_slice] += scratch_kq[k_slice + 8] +
//                              scratch_kq[k_slice + 16] +
//                              scratch_kq[k_slice + 24];
//     }
//     ucl::SyncThreads<WorkGroup, Local>();
//     kv_mem = scratch_kv[0] + scratch_kv[1] + scratch_kv[2] + scratch_kv[3] +
//              scratch_kv[4] + scratch_kv[5] + scratch_kv[6] + scratch_kv[7];
//
//     // Load V vector slice for this step
//     float4 v_vec = ucl::Convert<float4>(args.v_t.Read(t, h, v_slice, b));
//     float4 beta_factor = ucl::Init<float4>(ucl::Convert<float>(beta_val));
//     float4 delta_slice = (v_vec - kv_mem) * beta_factor;
//
//     // Update recurrent state in-place with outer product: S += delta * k^T
//     s0 = s0 + delta_slice * ucl::Init<float4>(k_val.x);
//     s1 = s1 + delta_slice * ucl::Init<float4>(k_val.y);
//     s2 = s2 + delta_slice * ucl::Init<float4>(k_val.z);
//     s3 = s3 + delta_slice * ucl::Init<float4>(k_val.w);
//
//     if (k_slice == 0) {
//       float4 sq_sum = scratch_out[0] + scratch_out[1] + scratch_out[2] +
//                       scratch_out[3] + scratch_out[4] + scratch_out[5] +
//                       scratch_out[6] + scratch_out[7];
//       float kq_sum = scratch_kq[0] + scratch_kq[1] + scratch_kq[2] +
//                      scratch_kq[3] + scratch_kq[4] + scratch_kq[5] +
//                      scratch_kq[6] + scratch_kq[7];
//       float4 out_sum = sq_sum + delta_slice * ucl::Init<float4>(kq_sum);
//       args.output.Write(ucl::Convert<float4>(out_sum), t, h, v_slice, b);
//     }
//   }
//
//   if (k_slice == 0) {
//     for (int t = valid_len; t < 128; ++t) {
//       args.output.Write(ucl::Init<float4>(0.0), t, h, v_slice, b);
//     }
//   }
//
//   // Write out final evolved recurrent state for this thread's 4 rows
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s0), k_base + 0, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s1), k_base + 1, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s2), k_base + 2, h, v_slice, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s3), k_base + 3, h, v_slice, b);
// }
// ```
// NOLINTEND(whitespace/line_length)
// clang-format on
constexpr char kGatedDeltaUpdateSharedMemShader[] = R"(
MAIN_FUNCTION($0) {
  int k_slice = ucl::GetLocalId<0>();
  int v_slice = ucl::GetGroupId<0>();
  int h = ucl::GetGroupId<1>();
  int h_k = h / GQA_RATIO;
  int b = ucl::GetGroupId<2>();

  __local StateType scratch_kv[HEAD_K_DIM_SLICES];
  __local StateType scratch_out[HEAD_K_DIM_SLICES];
  __local StateScalarType scratch_kq[HEAD_K_DIM_SLICES];

  // Each thread owns 4 rows along D_k (k_slice * 4 + {0, 1, 2, 3}) for this v_slice.
  // Held entirely in 4 registers. Zero register spilling.
  int k_base = k_slice * 4;
  StateType s0 = ucl::Convert<StateType>(args.recurrent_state_out.Read(k_base + 0, h, v_slice, b));
  StateType s1 = ucl::Convert<StateType>(args.recurrent_state_out.Read(k_base + 1, h, v_slice, b));
  StateType s2 = ucl::Convert<StateType>(args.recurrent_state_out.Read(k_base + 2, h, v_slice, b));
  StateType s3 = ucl::Convert<StateType>(args.recurrent_state_out.Read(k_base + 3, h, v_slice, b));
  VALID_LEN_INIT

  // Process active sequence length sequentially
  for (int t = 0; t < LOOP_BOUND_EXPR; ++t) {
    // Read beta and g for this step (mapped from TFLite 3D shape [B, H, L] to MLDrift BHWC [B, 1, H, L])
    int t_slice = t / 4;
    int t_elem = t % 4;
    ActivationType beta_t_vec = ucl::Convert<ActivationType>(args.beta_t.Read(h, 0, t_slice, b));
    ActivationScalarType beta_val = beta_t_vec.x;
    if (t_elem == 1) {
      beta_val = beta_t_vec.y;
    } else if (t_elem == 2) {
      beta_val = beta_t_vec.z;
    } else if (t_elem == 3) {
      beta_val = beta_t_vec.w;
    }

    ActivationType g_t_vec = ucl::Convert<ActivationType>(args.g_t.Read(h, 0, t_slice, b));
    ActivationScalarType g_val = g_t_vec.x;
    if (t_elem == 1) {
      g_val = g_t_vec.y;
    } else if (t_elem == 2) {
      g_val = g_t_vec.z;
    } else if (t_elem == 3) {
      g_val = g_t_vec.w;
    }

    StateScalarType decay_scalar = exp(ucl::Convert<StateScalarType>(g_val));
    StateType decay_vec = ucl::Init<StateType>(decay_scalar);

    // Apply decay to this thread's 4 rows
    s0 = s0 * decay_vec;
    s1 = s1 * decay_vec;
    s2 = s2 * decay_vec;
    s3 = s3 * decay_vec;

    // Load full vec4 K and Q for this thread's k_slice directly
    StateType k_val = ucl::Convert<StateType>(args.k_t.Read(t, h_k, k_slice, b));
    StateType q_val = ucl::Convert<StateType>(args.q_t.Read(t, h_k, k_slice, b));

    // Compute partial dot products before barrier
    StateType kv_mem = s0 * ucl::Init<StateType>(k_val.x) +
                       s1 * ucl::Init<StateType>(k_val.y) +
                       s2 * ucl::Init<StateType>(k_val.z) +
                       s3 * ucl::Init<StateType>(k_val.w);
    StateType sq_mem = s0 * ucl::Init<StateType>(q_val.x) +
                       s1 * ucl::Init<StateType>(q_val.y) +
                       s2 * ucl::Init<StateType>(q_val.z) +
                       s3 * ucl::Init<StateType>(q_val.w);
    StateScalarType kq_dot = dot(k_val, q_val);

    SHARED_REDUCE_KV_MEM

    // Load V vector slice for this step
    StateType v_vec = ucl::Convert<StateType>(args.v_t.Read(t, h, v_slice, b));
    StateType beta_factor = ucl::Init<StateType>(ucl::Convert<StateScalarType>(beta_val));
    StateType delta_slice = (v_vec - kv_mem) * beta_factor;

    // Update recurrent state in-place with outer product: S += delta * k^T
    s0 = s0 + delta_slice * ucl::Init<StateType>(k_val.x);
    s1 = s1 + delta_slice * ucl::Init<StateType>(k_val.y);
    s2 = s2 + delta_slice * ucl::Init<StateType>(k_val.z);
    s3 = s3 + delta_slice * ucl::Init<StateType>(k_val.w);

    SHARED_REDUCE_ATTN_OUT
  }

  ZERO_PAD_TAIL_STEP

  // Write out final evolved recurrent state for this thread's 4 rows
  args.recurrent_state_out.Write(ucl::Convert<StateType>(s0), k_base + 0, h, v_slice, b);
  args.recurrent_state_out.Write(ucl::Convert<StateType>(s1), k_base + 1, h, v_slice, b);
  args.recurrent_state_out.Write(ucl::Convert<StateType>(s2), k_base + 2, h, v_slice, b);
  args.recurrent_state_out.Write(ucl::Convert<StateType>(s3), k_base + 3, h, v_slice, b);
}
)";

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaUpdate(const ::ml_drift::OperationDef& definition, int mode,
                       const ::ml_drift::GpuInfo* gpu_info = nullptr) {
  const auto& q_t = definition.src_tensors[0];
  const auto& k_t = definition.src_tensors[1];
  const auto& v_t = definition.src_tensors[2];
  const auto& beta_t = definition.src_tensors[3];
  const auto& g_t = definition.src_tensors[4];
  const bool has_valid_len = definition.src_tensors.size() >= 6;

  const auto& output = definition.dst_tensors[0];
  const auto& rec_state_out = definition.dst_tensors[1];
  if (rec_state_out.GetStorageType() !=
      ::ml_drift::TensorStorageType::kBuffer) {
    return absl::InvalidArgumentError(
        "gated_delta_update requires kBuffer storage for recurrent_state_out.");
  }

  // Get dimensions from shape
  auto q_shape = q_t.GetBHWCShape();
  auto v_shape = v_t.GetBHWCShape();

  int B = v_shape.b;
  int H = v_shape.h;
  int seq_len = v_shape.w;
  int H_k = q_shape.h;
  if (H_k <= 0 || H < H_k || (H % H_k != 0)) {
    return absl::InvalidArgumentError(
        "gated_delta_update requires H_v to be a positive multiple of H_k.");
  }
  if (has_valid_len) {
    auto valid_len_shape = definition.src_tensors[5].GetBHWCShape();
    if (valid_len_shape.b != B || valid_len_shape.DimensionsProduct() != B) {
      return absl::InvalidArgumentError(
          "gated_delta_update requires valid_len to have B elements with "
          "batch dimension B.");
    }
  }
  int gqa_ratio = H / H_k;
  int D_k = q_shape.c;
  int D_v = v_shape.c;

  bool d_k_valid =
      (D_k >= 16) && (D_k % 4 == 0) && (((D_k / 4) & ((D_k / 4) - 1)) == 0);
  bool d_v_valid =
      (D_v >= 16) && (D_v % 4 == 0) && (((D_v / 4) & ((D_v / 4) - 1)) == 0);
  if (!d_k_valid || !d_v_valid) {
    return absl::InvalidArgumentError(
        "gated_delta_update requires D_k and D_v to be powers of 2 (at least "
        "16) and multiples of 4.");
  }

  int q_slices = D_k / 4;
  int v_slices = D_v / 4;
  if (q_slices > kMaxSharedMemWorkGroupSize) {
    return absl::InvalidArgumentError(
        "gated_delta_update requires D_k <= 1024 to bound workgroup size and "
        "register pressure.");
  }

  bool is_power_of_two = (q_slices > 0) && ((q_slices & (q_slices - 1)) == 0);
  // A single 32-lane subgroup can process at most kSubgroupSize * kMaxKRounds
  // (32 * 4 = 128) slices (D_k <= 512) while keeping k_rounds <= kMaxKRounds
  // (at most 16 float4 recurrent state registers per thread). Larger D_k
  // (D_k = 1024) falls back to the workgroup shared-memory reduction shader
  // where each of the q_slices threads holds only 1 round (4 float4 registers).
  // Setting mode == 2 explicitly forces the shared-memory reduction shader for
  // testing across all supported D_k values.
  const bool force_shared_mem = (mode == 2);
  bool can_use_shuffle = !force_shared_mem && gpu_info != nullptr &&
                         SupportsWave32Vec4Shuffle(*gpu_info) &&
                         is_power_of_two &&
                         (q_slices <= kSubgroupSize * kMaxKRounds);

  int subgroup_threads = std::min(q_slices, kSubgroupSize);
  int v_tile = 1;
  if (can_use_shuffle) {
    // Select the largest v_tile (4, 2, or 1 v_slices per subgroup) such that
    // each thread processes at most kMaxKRounds (4) rounds along D_k. Each
    // round holds 1 k_slice (4 rows of D_k) x 1 v_slice (4 cols of D_v) = 4
    // float4 registers (16 scalar floats). For example, at D_k = 256 (q_slices
    // = 64): v_tile = 4 leaves only 32 / 4 = 8 k_threads, requiring 64 / 8 = 8
    // rounds (8 * 4 = 32 float4 registers = 128 scalar VGPRs per thread).
    // Stepping down to v_tile = 2 gives 32 / 2 = 16 k_threads and 64 / 16 = 4
    // rounds <= kMaxKRounds (16 float4 registers = 64 scalar VGPRs per thread).
    for (int candidate_v_tile : {4, 2, 1}) {
      if (v_slices % candidate_v_tile == 0 &&
          subgroup_threads >= candidate_v_tile) {
        int candidate_k_threads = subgroup_threads / candidate_v_tile;
        int candidate_k_rounds = q_slices / candidate_k_threads;
        if (candidate_k_rounds <= kMaxKRounds) {
          v_tile = candidate_v_tile;
          break;
        }
      }
    }
  }
  int work_group_size_x = can_use_shuffle ? subgroup_threads : q_slices;
  int v_groups = v_slices / v_tile;
  int k_threads = work_group_size_x / v_tile;
  int k_rounds = q_slices / k_threads;
  // Number of vec4 chunks each thread in the workgroup/subgroup must load so
  // that all work_group_size_x lanes together cover all q_slices of D_k (e.g.,
  // for D_k = 256 -> q_slices = 64 and work_group_size_x = 32 -> 2 chunks per
  // thread).
  int k_load_chunks = q_slices / work_group_size_x;

  auto op =
      std::make_unique<GatedDeltaUpdateOp>(B, H, v_groups, work_group_size_x);

  op->AddSrcTensor("q_t", q_t);
  op->AddSrcTensor("k_t", k_t);
  op->AddSrcTensor("v_t", v_t);
  op->AddSrcTensor("beta_t", beta_t);
  op->AddSrcTensor("g_t", g_t);
  if (has_valid_len) {
    op->AddSrcTensor("valid_len", definition.src_tensors[5]);
  }

  op->AddDstTensor("output", output);
  op->AddDstTensor("recurrent_state_out", rec_state_out);

  std::string code = can_use_shuffle ? kGatedDeltaUpdateShuffleShader
                                     : kGatedDeltaUpdateSharedMemShader;

  std::string state_init_decls;
  std::string decay_step;
  std::string k_load_step;
  std::string kv_mem_step;
  std::string simd_reduce_kv_mem;
  std::string update_state_step;
  std::string q_load_step;
  std::string attn_out_step;
  std::string simd_reduce_attn_out;
  std::string state_write_out_step;

  k_load_step = "    // Coalesced chunked load of full D_k across 32 lanes\n";
  q_load_step = "    // Coalesced chunked load of full D_k Q across 32 lanes\n";
  for (int c = 0; c < k_load_chunks; ++c) {
    std::string cs = std::to_string(c);
    std::string slice_idx =
        (c == 0) ? "tid" : "tid + " + std::to_string(c * work_group_size_x);
    k_load_step += "    StateType k_loaded_" + cs +
                   " = ucl::Convert<StateType>(args.k_t.Read(t, h_k, " +
                   slice_idx + ", b));\n";
    q_load_step += "    StateType q_loaded_" + cs +
                   " = ucl::Convert<StateType>(args.q_t.Read(t, h_k, " +
                   slice_idx + ", b));\n";
  }

  for (int m = 0; m < k_rounds; ++m) {
    std::string ms = std::to_string(m);
    std::string k_slice_expr =
        (m == 0) ? "k_lane" : "(k_lane + " + ms + " * K_THREADS)";
    int chunk_idx = m / v_tile;
    int m_in_chunk = m % v_tile;
    std::string k_loaded_var = "k_loaded_" + std::to_string(chunk_idx);
    std::string q_loaded_var = "q_loaded_" + std::to_string(chunk_idx);
    std::string lane_expr =
        (m_in_chunk == 0)
            ? "k_lane"
            : "k_lane + " + std::to_string(m_in_chunk) + " * K_THREADS";
    std::string shuffle_k_expr =
        (v_tile == 1) ? k_loaded_var
                      : "ucl::WaveShuffle(" + k_loaded_var +
                            ", ucl::Convert<uint>(" + lane_expr + "))";
    std::string shuffle_q_expr =
        (v_tile == 1) ? q_loaded_var
                      : "ucl::WaveShuffle(" + q_loaded_var +
                            ", ucl::Convert<uint>(" + lane_expr + "))";

    state_init_decls +=
        "  int k_base_" + ms + " = " + k_slice_expr + " * 4;\n" +
        "  StateType s0_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_out.Read(k_base_" +
        ms + " + 0, h, v_slice, b));\n" + "  StateType s1_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_out.Read(k_base_" +
        ms + " + 1, h, v_slice, b));\n" + "  StateType s2_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_out.Read(k_base_" +
        ms + " + 2, h, v_slice, b));\n" + "  StateType s3_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_out.Read(k_base_" +
        ms + " + 3, h, v_slice, b));\n";

    decay_step += "    s0_" + ms + " = s0_" + ms + " * decay_vec;\n" +
                  "    s1_" + ms + " = s1_" + ms + " * decay_vec;\n" +
                  "    s2_" + ms + " = s2_" + ms + " * decay_vec;\n" +
                  "    s3_" + ms + " = s3_" + ms + " * decay_vec;\n";

    kv_mem_step +=
        "    {\n"
        "      StateType k_val_" +
        ms + " = " + shuffle_k_expr +
        ";\n"
        "      kv_mem += s0_" +
        ms + " * ucl::Init<StateType>(k_val_" + ms + ".x) +\n" +
        "                s1_" + ms + " * ucl::Init<StateType>(k_val_" + ms +
        ".y) +\n" + "                s2_" + ms +
        " * ucl::Init<StateType>(k_val_" + ms + ".z) +\n" +
        "                s3_" + ms + " * ucl::Init<StateType>(k_val_" + ms +
        ".w);\n"
        "    }\n";

    update_state_step +=
        "    {\n"
        "      StateType k_val_" +
        ms + " = " + shuffle_k_expr +
        ";\n"
        "      s0_" +
        ms + " = s0_" + ms + " + delta_slice * ucl::Init<StateType>(k_val_" +
        ms + ".x);\n" + "      s1_" + ms + " = s1_" + ms +
        " + delta_slice * ucl::Init<StateType>(k_val_" + ms + ".y);\n" +
        "      s2_" + ms + " = s2_" + ms +
        " + delta_slice * ucl::Init<StateType>(k_val_" + ms + ".z);\n" +
        "      s3_" + ms + " = s3_" + ms +
        " + delta_slice * ucl::Init<StateType>(k_val_" + ms +
        ".w);\n"
        "    }\n";

    attn_out_step +=
        "    {\n"
        "      StateType q_val_" +
        ms + " = " + shuffle_q_expr +
        ";\n"
        "      my_attn_out += s0_" +
        ms + " * ucl::Init<StateType>(q_val_" + ms + ".x) +\n" +
        "                     s1_" + ms + " * ucl::Init<StateType>(q_val_" +
        ms + ".y) +\n" + "                     s2_" + ms +
        " * ucl::Init<StateType>(q_val_" + ms + ".z) +\n" +
        "                     s3_" + ms + " * ucl::Init<StateType>(q_val_" +
        ms +
        ".w);\n"
        "    }\n";

    state_write_out_step +=
        "  args.recurrent_state_out.Write(ucl::Convert<StateType>(s0_" + ms +
        "), k_base_" + ms + " + 0, h, v_slice, b);\n" +
        "  args.recurrent_state_out.Write(ucl::Convert<StateType>(s1_" + ms +
        "), k_base_" + ms + " + 1, h, v_slice, b);\n" +
        "  args.recurrent_state_out.Write(ucl::Convert<StateType>(s2_" + ms +
        "), k_base_" + ms + " + 2, h, v_slice, b);\n" +
        "  args.recurrent_state_out.Write(ucl::Convert<StateType>(s3_" + ms +
        "), k_base_" + ms + " + 3, h, v_slice, b);\n";
  }

  for (int offset = 1; offset < k_threads; offset *= 2) {
    std::string os = std::to_string(offset);
    simd_reduce_kv_mem +=
        "    kv_mem += ucl::WaveShuffleXor(kv_mem, " + os + "u);\n";
    simd_reduce_attn_out +=
        "    my_attn_out += ucl::WaveShuffleXor(my_attn_out, " + os + "u);\n";
  }

  std::string shared_reduce_kv_mem = (q_slices == 32)
                                         ? R"(scratch_kv[k_slice] = kv_mem;
    scratch_out[k_slice] = sq_mem;
    scratch_kq[k_slice] = kq_dot;
    ucl::SyncThreads<WorkGroup, Local>();
    // Fold the 32 partial sums across 4 quarters (lanes 0..7 accumulate lanes
    // k_slice + {0, 8, 16, 24}) in a single barrier step instead of 5 tree-reduction
    // barriers, then sum the remaining 8 partial sums directly.
    if (k_slice < 8) {
      scratch_kv[k_slice] += scratch_kv[k_slice + 8] +
                             scratch_kv[k_slice + 16] +
                             scratch_kv[k_slice + 24];
      scratch_out[k_slice] += scratch_out[k_slice + 8] +
                              scratch_out[k_slice + 16] +
                              scratch_out[k_slice + 24];
      scratch_kq[k_slice] += scratch_kq[k_slice + 8] +
                             scratch_kq[k_slice + 16] +
                             scratch_kq[k_slice + 24];
    }
    ucl::SyncThreads<WorkGroup, Local>();
    kv_mem = scratch_kv[0] + scratch_kv[1] + scratch_kv[2] + scratch_kv[3] +
             scratch_kv[4] + scratch_kv[5] + scratch_kv[6] + scratch_kv[7];)"
                                         : R"(scratch_kv[k_slice] = kv_mem;
    scratch_out[k_slice] = sq_mem;
    scratch_kq[k_slice] = kq_dot;
    ucl::SyncThreads<WorkGroup, Local>();
    for (int stride = HEAD_K_DIM_SLICES / 2; stride > 0; stride /= 2) {
      if (k_slice < stride) {
        scratch_kv[k_slice] += scratch_kv[k_slice + stride];
        scratch_out[k_slice] += scratch_out[k_slice + stride];
        scratch_kq[k_slice] += scratch_kq[k_slice + stride];
      }
      ucl::SyncThreads<WorkGroup, Local>();
    }
    kv_mem = scratch_kv[0];)";

  std::string shared_reduce_attn_out = (q_slices == 32) ? R"(if (k_slice == 0) {
      StateType sq_sum = scratch_out[0] + scratch_out[1] + scratch_out[2] +
                         scratch_out[3] + scratch_out[4] + scratch_out[5] +
                         scratch_out[6] + scratch_out[7];
      StateScalarType kq_sum = scratch_kq[0] + scratch_kq[1] + scratch_kq[2] +
                               scratch_kq[3] + scratch_kq[4] + scratch_kq[5] +
                               scratch_kq[6] + scratch_kq[7];
      StateType out_sum = sq_sum + delta_slice * ucl::Init<StateType>(kq_sum);
      args.output.Write(ucl::Convert<ActivationType>(out_sum), t, h, v_slice, b);
    })"
                                                        : R"(if (k_slice == 0) {
      StateType out_sum = scratch_out[0] + delta_slice * ucl::Init<StateType>(scratch_kq[0]);
      args.output.Write(ucl::Convert<ActivationType>(out_sum), t, h, v_slice, b);
    })";

  std::string valid_len_init =
      has_valid_len
          ? "int valid_len = clamp(args.valid_len.Read<int>(0, 0, 0, b).x, 0, "
            "SEQ_LEN_EXPR);"
          : "";
  std::string loop_bound_expr = has_valid_len ? "valid_len" : "SEQ_LEN_EXPR";
  std::string zero_pad_lane_cond =
      can_use_shuffle ? "k_lane == 0" : "k_slice == 0";
  std::string zero_pad_tail_step =
      has_valid_len
          ? "if (" + zero_pad_lane_cond +
                ") {\n"
                "    for (int t = valid_len; t < SEQ_LEN_EXPR; ++t) {\n"
                "      args.output.Write(ucl::Init<ActivationType>(0.0), t, h, "
                "v_slice, b);\n"
                "    }\n"
                "  }"
          : "";

  absl::StrReplaceAll(
      {{"STATE_INIT_DECLS", absl::StripAsciiWhitespace(state_init_decls)},
       {"VALID_LEN_INIT", valid_len_init},
       {"LOOP_BOUND_EXPR", loop_bound_expr},
       {"DECAY_STEP", absl::StripAsciiWhitespace(decay_step)},
       {"K_LOAD_STEP", absl::StripAsciiWhitespace(k_load_step)},
       {"KV_MEM_STEP", absl::StripAsciiWhitespace(kv_mem_step)},
       {"SIMD_REDUCE_KV_MEM", absl::StripAsciiWhitespace(simd_reduce_kv_mem)},
       {"UPDATE_STATE_STEP", absl::StripAsciiWhitespace(update_state_step)},
       {"Q_LOAD_STEP", absl::StripAsciiWhitespace(q_load_step)},
       {"ATTN_OUT_STEP", absl::StripAsciiWhitespace(attn_out_step)},
       {"SIMD_REDUCE_ATTN_OUT",
        absl::StripAsciiWhitespace(simd_reduce_attn_out)},
       {"ZERO_PAD_TAIL_STEP", zero_pad_tail_step},
       {"STATE_WRITE_OUT_STEP",
        absl::StripAsciiWhitespace(state_write_out_step)},
       {"SHARED_REDUCE_KV_MEM", shared_reduce_kv_mem},
       {"SHARED_REDUCE_ATTN_OUT", shared_reduce_attn_out}},
      &code);

  absl::StrReplaceAll(
      {{"SEQ_LEN_EXPR", std::to_string(seq_len)},
       {"V_TILE", std::to_string(v_tile)},
       {"K_THREADS", std::to_string(k_threads)},
       {"StateScalarType",
        ::ml_drift::ToUclDataType(rec_state_out.GetDataType(), 1)},
       {"StateType", ::ml_drift::ToUclDataType(rec_state_out.GetDataType(), 4)},
       {"ActivationScalarType",
        ::ml_drift::ToUclDataType(output.GetDataType(), 1)},
       {"ActivationType", ::ml_drift::ToUclDataType(output.GetDataType(), 4)},
       {"HEAD_K_DIM_SLICES", std::to_string(q_slices)},
       {"GQA_RATIO", std::to_string(gqa_ratio)}},
      &code);

  if (can_use_shuffle) {
    ResolveWaveSimd(*gpu_info, &code);
  }

  op->code_ = std::move(code);
  return op;
}

// `gated_delta_update` block mode: the Qwen3.5 GatedDeltaNet block after the
// short conv. It covers the q / k / v and z / b / a splits, beta = sigmoid(b)
// and g = -exp(a_log) * softplus(a + dt_bias), the valid mask, the q / k L2
// norms (with the 1 / sqrt(D_k) readout scale folded into q), the gated delta
// rule and the gated RMSNorm x SiLU(z). The kernels read the short conv output
// and
// the fused [z | b | a] projection at channel offsets, keep all math in float32
// and update the recurrent state in place. A token with a zero valid mask gets
// q = 0, beta = 0 and g = 0, so it leaves the state unchanged and outputs
// zeros.
//
// Execution paths:
//   - Single-token decode (T = 1): a single fused kernel per layer —
//     `kGatedDeltaNetDecodeShader` (one workgroup of V_GROUPS 32-lane subgroups
//     per value head) when 32-lane wave shuffles are supported, or
//     `kGatedDeltaNetDecodeSharedMemShader` (one workgroup of DEC_WG_SIZE
//     threads per value head streaming across D_v in DEC_VCHUNKS tiles with
//     workgroup-memory reductions) otherwise.
//   - Multi-token recurrence (T > 1):
//     1. `kGatedDeltaNetPrepShader` runs across all (token, key_head) pairs in
//        parallel to mask and L2-normalize q / k into `qk_norm` and compute
//        per-value-head (beta, decay, mask) into `gates`, avoiding redundant
//        L2-norm reductions and transcendental ops across v_slice workgroups
//        inside the sequential token loop.
//     2. `kGatedDeltaNetPrefillShader` (32-lane wave shuffles, V_TILE = 4) or
//        `kGatedDeltaNetSharedMemShader` (64-thread workgroup-memory
//        reductions, V_TILE = 2, double-buffered 2 barriers per token) updates
//        the recurrent state and writes the un-normalized output in float32.
//     3. `kGatedDeltaNetNormShader` applies the gated RMSNorm x SiLU(z).

constexpr int kGdnVTile = 4;
constexpr int kGdnKThreads = kSubgroupSize / kGdnVTile;
constexpr int kGdnSharedVTile = 2;

// `beta_val` and `decay_scalar` of one token, from `tok_mask` and the zba
// slices `beta_vec` / `gate_vec` that hold b and a.
constexpr char kGdnScalarGates[] = R"(
    float beta_raw = beta_vec.x;
    if (beta_elem == 1) {
      beta_raw = beta_vec.y;
    } else if (beta_elem == 2) {
      beta_raw = beta_vec.z;
    } else if (beta_elem == 3) {
      beta_raw = beta_vec.w;
    }
    float gate_raw = gate_vec.x;
    if (gate_elem == 1) {
      gate_raw = gate_vec.y;
    } else if (gate_elem == 2) {
      gate_raw = gate_vec.z;
    } else if (gate_elem == 3) {
      gate_raw = gate_vec.w;
    }
    float beta_val = tok_mask / (1.0f + exp(-beta_raw));
    float gate_in = gate_raw + dt_bias_h;
    // Overflow-free softplus: max(x, 0) + log(1 + exp(-|x|)).
    float softplus_val = max(gate_in, 0.0f) + log(1.0f + exp(min(gate_in, -gate_in)));
    float decay_scalar = exp(neg_a_h * softplus_val * tok_mask);
)";

constexpr char kGatedDeltaNetDecodeShader[] = R"(
#pragma OPENCL EXTENSION ucl_wave_simd: enable
MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int lane = tid % 32;
  int sg = tid / 32;
  int k_lane = lane % K_THREADS;
  int v_lane = lane / K_THREADS;
  int v_slice = sg * V_TILE + v_lane;
  int h = ucl::GetGroupId<1>();
  int h_k = h / GQA_RATIO;
  int b = ucl::GetGroupId<2>();
  float neg_a_h = -exp(A_LOG_EXPR);
  float dt_bias_h = DT_BIAS_EXPR;
  int beta_ch = BETA_CH_OFFSET + h;
  int beta_slice = beta_ch / 4;
  int beta_elem = beta_ch % 4;
  int gate_ch = GATE_CH_OFFSET + h;
  int gate_slice = gate_ch / 4;
  int gate_elem = gate_ch % 4;
  int q_in_slice = h_k * HEAD_K_DIM_SLICES + lane;
  int k_in_slice = K_SLICE_OFFSET + q_in_slice;
  int v_in_slice = V_SLICE_OFFSET + h * HEAD_V_DIM_SLICES + v_slice;
  int out_slice = h * HEAD_V_DIM_SLICES + v_slice;
  float tok_mask = MASK_EXPR;
  float4 beta_vec = ucl::Convert<float4>(args.zba.Read(0, 0, beta_slice, b));
  float4 gate_vec = ucl::Convert<float4>(args.zba.Read(0, 0, gate_slice, b));
GDN_SCALAR_GATES
  float4 decay_vec = ucl::Init<float4>(decay_scalar);
  float4 q_loaded_0 = ucl::Convert<float4>(args.conv_out.Read(0, 0, q_in_slice, b)) * ucl::Init<float4>(tok_mask);
  float4 k_loaded_0 = ucl::Convert<float4>(args.conv_out.Read(0, 0, k_in_slice, b));
  float q_sumsq = ucl::WaveSum(dot(q_loaded_0, q_loaded_0));
  float k_sumsq = ucl::WaveSum(dot(k_loaded_0, k_loaded_0));
  q_loaded_0 = q_loaded_0 * ucl::Init<float4>(rsqrt(q_sumsq + args.eps) * args.q_scale);
  k_loaded_0 = k_loaded_0 * ucl::Init<float4>(rsqrt(k_sumsq + args.eps));
STATE_INIT_DECLS
DECAY_STEP
  float4 kv_mem = ucl::Init<float4>(0.0f);
KV_MEM_STEP
SIMD_REDUCE_KV_MEM
  float4 v_vec = ucl::Convert<float4>(args.conv_out.Read(0, 0, v_in_slice, b));
  float4 delta_slice = (v_vec - kv_mem) * ucl::Init<float4>(beta_val);
UPDATE_STATE_STEP
  float4 my_attn_out = ucl::Init<float4>(0.0f);
ATTN_OUT_STEP
SIMD_REDUCE_ATTN_OUT
STATE_WRITE_OUT_STEP
  // Gated RMSNorm over the D_v outputs of head h; lanes with k_lane == 0 hold
  // one v_slice each.
  float sumsq_part = 0.0f;
  if (k_lane == 0) {
    sumsq_part = dot(my_attn_out, my_attn_out);
  }
  float sg_sumsq = ucl::WaveSum(sumsq_part);
  __local float norm_partial[V_GROUPS];
  if (lane == 0) {
    norm_partial[sg] = sg_sumsq;
  }
  ucl::SyncThreads<WorkGroup, Local>();
  float head_sumsq = NORM_PARTIAL_SUM;
  float inv_rms = rsqrt(head_sumsq * args.inv_head_v_dim + args.norm_eps);
  if (k_lane == 0) {
    float4 norm_w = NORM_WEIGHT_EXPR;
    float4 z_vec = ucl::Convert<float4>(args.zba.Read(0, 0, out_slice, b));
    float4 z_gate = z_vec / (ucl::Init<float4>(1.0f) + exp(-z_vec));
    float4 gated_out = my_attn_out * ucl::Init<float4>(inv_rms) * norm_w * z_gate;
    args.output.Write(ucl::Convert<ActivationType>(gated_out), 0, 0, out_slice, b);
  }
}
)";

// Single-token fused decode kernel for GPUs without 32-lane wave shuffles (e.g.
// Adreno wave64 OpenCL / Vulkan WebGPU): one workgroup of DEC_WG_SIZE (= 32 *
// DEC_VTILE, 128 or 256) threads per value head streams across D_v in
// DEC_VCHUNKS tiles so each thread holds only 4 float4 state registers at a
// time (~9 KB shared memory, well within WebGPU's 16 KB minimum), then applies
// the gated RMSNorm x SiLU(z) in the same kernel.
constexpr char kGatedDeltaNetDecodeSharedMemShader[] = R"(
MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int lane = tid % 32;
  int v_local = tid / 32;
  int h = ucl::GetGroupId<1>();
  int h_k = h / GQA_RATIO;
  int b = ucl::GetGroupId<2>();
  float neg_a_h = -exp(A_LOG_EXPR);
  float dt_bias_h = DT_BIAS_EXPR;
  int beta_ch = BETA_CH_OFFSET + h;
  int beta_slice = beta_ch / 4;
  int beta_elem = beta_ch % 4;
  int gate_ch = GATE_CH_OFFSET + h;
  int gate_slice = gate_ch / 4;
  int gate_elem = gate_ch % 4;
  float tok_mask = MASK_EXPR;
  float4 beta_vec = ucl::Convert<float4>(args.zba.Read(0, 0, beta_slice, b));
  float4 gate_vec = ucl::Convert<float4>(args.zba.Read(0, 0, gate_slice, b));
GDN_SCALAR_GATES
  float4 decay_vec = ucl::Init<float4>(decay_scalar);
  int q_in_slice = h_k * HEAD_K_DIM_SLICES + lane;
  int k_in_slice = K_SLICE_OFFSET + q_in_slice;
  float4 q_val = ucl::Convert<float4>(args.conv_out.Read(0, 0, q_in_slice, b)) * ucl::Init<float4>(tok_mask);
  float4 k_val = ucl::Convert<float4>(args.conv_out.Read(0, 0, k_in_slice, b));
  __local float4 scratch_kv[DEC_WG_SIZE];
  __local float4 scratch_sq[DEC_WG_SIZE];
  __local float4 head_out[HEAD_V_DIM_SLICES];
  __local float scratch_red[96];
  if (v_local == 0) {
    scratch_red[lane] = dot(q_val, q_val);
    scratch_red[32 + lane] = dot(k_val, k_val);
    scratch_red[64 + lane] = dot(k_val, q_val);
  }
  ucl::SyncThreads<WorkGroup, Local>();
  if (tid < 8) {
    scratch_red[tid] += scratch_red[tid + 8] + scratch_red[tid + 16] + scratch_red[tid + 24];
    scratch_red[32 + tid] += scratch_red[40 + tid] + scratch_red[48 + tid] + scratch_red[56 + tid];
    scratch_red[64 + tid] += scratch_red[72 + tid] + scratch_red[80 + tid] + scratch_red[88 + tid];
  }
  ucl::SyncThreads<WorkGroup, Local>();
  float q_sumsq = scratch_red[0] + scratch_red[1] + scratch_red[2] + scratch_red[3] +
                  scratch_red[4] + scratch_red[5] + scratch_red[6] + scratch_red[7];
  float k_sumsq = scratch_red[32] + scratch_red[33] + scratch_red[34] + scratch_red[35] +
                  scratch_red[36] + scratch_red[37] + scratch_red[38] + scratch_red[39];
  float kq_raw = scratch_red[64] + scratch_red[65] + scratch_red[66] + scratch_red[67] +
                 scratch_red[68] + scratch_red[69] + scratch_red[70] + scratch_red[71];
  float q_norm_scale = rsqrt(q_sumsq + args.eps) * args.q_scale;
  float k_norm_scale = rsqrt(k_sumsq + args.eps);
  float4 q_hat = q_val * ucl::Init<float4>(q_norm_scale);
  float4 k_hat = k_val * ucl::Init<float4>(k_norm_scale);
  float kq_sum = kq_raw * q_norm_scale * k_norm_scale;
  int k_base = lane * 4;
  for (int c = 0; c < DEC_VCHUNKS; ++c) {
    int v_slice = c * DEC_VTILE + v_local;
    int v_in_slice = V_SLICE_OFFSET + h * HEAD_V_DIM_SLICES + v_slice;
    float4 s0 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 0, h, v_slice, b)) * decay_vec;
    float4 s1 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 1, h, v_slice, b)) * decay_vec;
    float4 s2 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 2, h, v_slice, b)) * decay_vec;
    float4 s3 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 3, h, v_slice, b)) * decay_vec;
    int sh_base = v_local * 32;
    scratch_kv[sh_base + lane] = s0 * ucl::Init<float4>(k_hat.x) + s1 * ucl::Init<float4>(k_hat.y) +
                                 s2 * ucl::Init<float4>(k_hat.z) + s3 * ucl::Init<float4>(k_hat.w);
    scratch_sq[sh_base + lane] = s0 * ucl::Init<float4>(q_hat.x) + s1 * ucl::Init<float4>(q_hat.y) +
                                 s2 * ucl::Init<float4>(q_hat.z) + s3 * ucl::Init<float4>(q_hat.w);
    ucl::SyncThreads<WorkGroup, Local>();
    if (lane < 8) {
      int idx = sh_base + lane;
      scratch_kv[idx] += scratch_kv[idx + 8] + scratch_kv[idx + 16] + scratch_kv[idx + 24];
      scratch_sq[idx] += scratch_sq[idx + 8] + scratch_sq[idx + 16] + scratch_sq[idx + 24];
    }
    ucl::SyncThreads<WorkGroup, Local>();
    float4 kv_sum = scratch_kv[sh_base + 0] + scratch_kv[sh_base + 1] + scratch_kv[sh_base + 2] +
                    scratch_kv[sh_base + 3] + scratch_kv[sh_base + 4] + scratch_kv[sh_base + 5] +
                    scratch_kv[sh_base + 6] + scratch_kv[sh_base + 7];
    float4 v_vec = ucl::Convert<float4>(args.conv_out.Read(0, 0, v_in_slice, b));
    float4 delta_slice = (v_vec - kv_sum) * ucl::Init<float4>(beta_val);
    args.recurrent_state_out.Write(s0 + delta_slice * ucl::Init<float4>(k_hat.x), k_base + 0, h, v_slice, b);
    args.recurrent_state_out.Write(s1 + delta_slice * ucl::Init<float4>(k_hat.y), k_base + 1, h, v_slice, b);
    args.recurrent_state_out.Write(s2 + delta_slice * ucl::Init<float4>(k_hat.z), k_base + 2, h, v_slice, b);
    args.recurrent_state_out.Write(s3 + delta_slice * ucl::Init<float4>(k_hat.w), k_base + 3, h, v_slice, b);
    if (lane == 0) {
      float4 sq_sum = scratch_sq[sh_base + 0] + scratch_sq[sh_base + 1] + scratch_sq[sh_base + 2] +
                      scratch_sq[sh_base + 3] + scratch_sq[sh_base + 4] + scratch_sq[sh_base + 5] +
                      scratch_sq[sh_base + 6] + scratch_sq[sh_base + 7];
      head_out[v_slice] = sq_sum + delta_slice * ucl::Init<float4>(kq_sum);
    }
    ucl::SyncThreads<WorkGroup, Local>();
  }
  if (tid < 32) {
    float sumsq = 0.0f;
    if (tid < HEAD_V_DIM_SLICES) {
      float4 o_part = head_out[tid];
      sumsq = dot(o_part, o_part);
    }
    scratch_red[tid] = sumsq;
  }
  ucl::SyncThreads<WorkGroup, Local>();
  if (tid < 8) {
    scratch_red[tid] += scratch_red[tid + 8] + scratch_red[tid + 16] + scratch_red[tid + 24];
  }
  ucl::SyncThreads<WorkGroup, Local>();
  float head_sumsq = scratch_red[0] + scratch_red[1] + scratch_red[2] + scratch_red[3] +
                     scratch_red[4] + scratch_red[5] + scratch_red[6] + scratch_red[7];
  float inv_rms = rsqrt(head_sumsq * args.inv_head_v_dim + args.norm_eps);
  if (tid < HEAD_V_DIM_SLICES) {
    int v_slice = tid;
    int out_slice = h * HEAD_V_DIM_SLICES + v_slice;
    float4 norm_w = NORM_WEIGHT_EXPR;
    float4 z_vec = ucl::Convert<float4>(args.zba.Read(0, 0, out_slice, b));
    float4 z_gate = z_vec / (ucl::Init<float4>(1.0f) + exp(-z_vec));
    float4 gated_out = head_out[v_slice] * ucl::Init<float4>(inv_rms) * norm_w * z_gate;
    args.output.Write(ucl::Convert<ActivationType>(gated_out), 0, 0, out_slice, b);
  }
}
)";

// Parallel token preparation: one workgroup of 32 threads per (token, key_head)
// masks and L2-normalizes q and k (with 1 / sqrt(D_k) folded into q) and
// computes (beta, decay, mask, 0) for each associated value head.
constexpr char kGatedDeltaNetPrepShader[] = R"(
MAIN_FUNCTION($0) {
  int lane = ucl::GetLocalId<0>();
  int t = ucl::GetGroupId<0>();
  int h_k = ucl::GetGroupId<1>();
  int b = ucl::GetGroupId<2>();
MASK_INIT
  int q_slice = h_k * HEAD_K_DIM_SLICES + lane;
  int k_slice = K_SLICE_OFFSET + q_slice;
  float4 q_val = ucl::Convert<float4>(args.conv_out.Read(t, 0, q_slice, b)) * ucl::Init<float4>(tok_mask);
  float4 k_val = ucl::Convert<float4>(args.conv_out.Read(t, 0, k_slice, b));
  __local float scratch_qq[32];
  __local float scratch_kk[32];
  scratch_qq[lane] = dot(q_val, q_val);
  scratch_kk[lane] = dot(k_val, k_val);
  ucl::SyncThreads<WorkGroup, Local>();
  if (lane < 8) {
    scratch_qq[lane] += scratch_qq[lane + 8] + scratch_qq[lane + 16] + scratch_qq[lane + 24];
    scratch_kk[lane] += scratch_kk[lane + 8] + scratch_kk[lane + 16] + scratch_kk[lane + 24];
  }
  ucl::SyncThreads<WorkGroup, Local>();
  float q_sumsq = scratch_qq[0] + scratch_qq[1] + scratch_qq[2] + scratch_qq[3] +
                  scratch_qq[4] + scratch_qq[5] + scratch_qq[6] + scratch_qq[7];
  float k_sumsq = scratch_kk[0] + scratch_kk[1] + scratch_kk[2] + scratch_kk[3] +
                  scratch_kk[4] + scratch_kk[5] + scratch_kk[6] + scratch_kk[7];
  float4 q_hat = q_val * ucl::Init<float4>(rsqrt(q_sumsq + args.eps) * args.q_scale);
  float4 k_hat = k_val * ucl::Init<float4>(rsqrt(k_sumsq + args.eps));
  args.qk_norm.Write(q_hat, t, 0, q_slice, b);
  args.qk_norm.Write(k_hat, t, 0, k_slice, b);
  for (int r = lane; r < GQA_RATIO; r += 32) {
    int h = h_k * GQA_RATIO + r;
    float neg_a_h = -exp(A_LOG_EXPR);
    float dt_bias_h = DT_BIAS_EXPR;
    int beta_ch = BETA_CH_OFFSET + h;
    int beta_slice = beta_ch / 4;
    int beta_elem = beta_ch % 4;
    int gate_ch = GATE_CH_OFFSET + h;
    int gate_slice = gate_ch / 4;
    int gate_elem = gate_ch % 4;
    float4 beta_vec = ucl::Convert<float4>(args.zba.Read(t, 0, beta_slice, b));
    float4 gate_vec = ucl::Convert<float4>(args.zba.Read(t, 0, gate_slice, b));
GDN_SCALAR_GATES
    args.gates.Write(ucl::Init<float4>(beta_val, decay_scalar, tok_mask, 0.0f), t, 0, h, b);
  }
}
)";

constexpr char kGatedDeltaNetPrefillShader[] = R"(
#pragma OPENCL EXTENSION ucl_wave_simd: enable
MAIN_FUNCTION($0) {
  int lane = ucl::GetLocalId<0>();
  int k_lane = lane % K_THREADS;
  int v_lane = lane / K_THREADS;
  int v_slice = ucl::GetGroupId<0>() * V_TILE + v_lane;
  int h = ucl::GetGroupId<1>();
  int h_k = h / GQA_RATIO;
  int b = ucl::GetGroupId<2>();
  int q_in_slice = h_k * HEAD_K_DIM_SLICES + lane;
  int k_in_slice = K_SLICE_OFFSET + q_in_slice;
  int v_in_slice = V_SLICE_OFFSET + h * HEAD_V_DIM_SLICES + v_slice;
  int out_slice = h * HEAD_V_DIM_SLICES + v_slice;
STATE_INIT_DECLS
  // Inputs of token 0. Every iteration prefetches the inputs of the next token,
  // so that their load latency hides behind the recurrence.
  float4 gate_next = ucl::Convert<float4>(args.gates.Read(0, 0, h, b));
  float4 q_next = ucl::Convert<float4>(args.qk_norm.Read(0, 0, q_in_slice, b));
  float4 k_next = ucl::Convert<float4>(args.qk_norm.Read(0, 0, k_in_slice, b));
  float4 v_next = ucl::Convert<float4>(args.conv_out.Read(0, 0, v_in_slice, b));
  for (int t = 0; t < SEQ_LEN; ++t) {
    float4 gate_cur = gate_next;
    float4 q_loaded_0 = q_next;
    float4 k_loaded_0 = k_next;
    float4 v_vec = v_next;
    if (t + 1 < SEQ_LEN) {
      gate_next = ucl::Convert<float4>(args.gates.Read(t + 1, 0, h, b));
      q_next = ucl::Convert<float4>(args.qk_norm.Read(t + 1, 0, q_in_slice, b));
      k_next = ucl::Convert<float4>(args.qk_norm.Read(t + 1, 0, k_in_slice, b));
      v_next = ucl::Convert<float4>(args.conv_out.Read(t + 1, 0, v_in_slice, b));
    }
    if (gate_cur.z != 0.0f) {
      float beta_val = gate_cur.x;
      float4 decay_vec = ucl::Init<float4>(gate_cur.y);
DECAY_STEP
      float4 kv_mem = ucl::Init<float4>(0.0f);
KV_MEM_STEP
SIMD_REDUCE_KV_MEM
      float4 delta_slice = (v_vec - kv_mem) * ucl::Init<float4>(beta_val);
UPDATE_STATE_STEP
      float4 my_attn_out = ucl::Init<float4>(0.0f);
ATTN_OUT_STEP
SIMD_REDUCE_ATTN_OUT
      if (k_lane == 0) {
        args.raw_out.Write(my_attn_out, t, 0, out_slice, b);
      }
    } else if (k_lane == 0) {
      args.raw_out.Write(ucl::Init<float4>(0.0f), t, 0, out_slice, b);
    }
  }
STATE_WRITE_OUT_STEP
}
)";

// One workgroup of 64 threads (2 v_slices x 32 k_slices, filling one full 64-
// lane hardware wave on Adreno or two 32-lane subgroups) per (2 v_slices, value
// head); thread `tid` owns rows 4 * (tid % 32) .. +3 of `v_slice =
// group_id_x * 2 + (tid / 32)` (4 float4 state registers per thread).
// Double-buffering the 1D workgroup scratch arrays by `t % 2` keeps every
// token step to 2 barriers in uniform control flow while avoiding write-after-
// read hazards across iterations.
constexpr char kGatedDeltaNetSharedMemShader[] = R"(
MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int lane = tid % 32;
  int v_local = tid / 32;
  int v_slice = ucl::GetGroupId<0>() * 2 + v_local;
  int h = ucl::GetGroupId<1>();
  int h_k = h / GQA_RATIO;
  int b = ucl::GetGroupId<2>();
  int q_in_slice = h_k * HEAD_K_DIM_SLICES + lane;
  int k_in_slice = K_SLICE_OFFSET + q_in_slice;
  int v_in_slice = V_SLICE_OFFSET + h * HEAD_V_DIM_SLICES + v_slice;
  int out_slice = h * HEAD_V_DIM_SLICES + v_slice;
  __local float4 scratch_kv[128];
  __local float4 scratch_sq[128];
  __local float scratch_kq[64];
  int k_base = lane * 4;
  float4 s0 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 0, h, v_slice, b));
  float4 s1 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 1, h, v_slice, b));
  float4 s2 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 2, h, v_slice, b));
  float4 s3 = ucl::Convert<float4>(args.recurrent_state_out.Read(k_base + 3, h, v_slice, b));
  float4 gate_next = ucl::Convert<float4>(args.gates.Read(0, 0, h, b));
  float4 q_next = ucl::Convert<float4>(args.qk_norm.Read(0, 0, q_in_slice, b));
  float4 k_next = ucl::Convert<float4>(args.qk_norm.Read(0, 0, k_in_slice, b));
  float4 v_next = ucl::Convert<float4>(args.conv_out.Read(0, 0, v_in_slice, b));
  for (int t = 0; t < SEQ_LEN; ++t) {
    float4 gate_cur = gate_next;
    float4 q_val = q_next;
    float4 k_val = k_next;
    float4 v_vec = v_next;
    if (t + 1 < SEQ_LEN) {
      gate_next = ucl::Convert<float4>(args.gates.Read(t + 1, 0, h, b));
      q_next = ucl::Convert<float4>(args.qk_norm.Read(t + 1, 0, q_in_slice, b));
      k_next = ucl::Convert<float4>(args.qk_norm.Read(t + 1, 0, k_in_slice, b));
      v_next = ucl::Convert<float4>(args.conv_out.Read(t + 1, 0, v_in_slice, b));
    }
    float4 decay_vec = ucl::Init<float4>(gate_cur.y);
    s0 = s0 * decay_vec;
    s1 = s1 * decay_vec;
    s2 = s2 * decay_vec;
    s3 = s3 * decay_vec;
    int buf = t % 2;
    int sh_base = (buf * 2 + v_local) * 32;
    int kq_base = buf * 32;
    scratch_kv[sh_base + lane] = s0 * ucl::Init<float4>(k_val.x) + s1 * ucl::Init<float4>(k_val.y) +
                                 s2 * ucl::Init<float4>(k_val.z) + s3 * ucl::Init<float4>(k_val.w);
    scratch_sq[sh_base + lane] = s0 * ucl::Init<float4>(q_val.x) + s1 * ucl::Init<float4>(q_val.y) +
                                 s2 * ucl::Init<float4>(q_val.z) + s3 * ucl::Init<float4>(q_val.w);
    if (v_local == 0) {
      scratch_kq[kq_base + lane] = dot(k_val, q_val);
    }
    ucl::SyncThreads<WorkGroup, Local>();
    // Fold the 32 partial sums per v_local into 8 (lanes 0..7 add +8, +16, +24).
    if (lane < 8) {
      int idx = sh_base + lane;
      scratch_kv[idx] += scratch_kv[idx + 8] + scratch_kv[idx + 16] + scratch_kv[idx + 24];
      scratch_sq[idx] += scratch_sq[idx + 8] + scratch_sq[idx + 16] + scratch_sq[idx + 24];
      if (v_local == 0) {
        int kq_idx = kq_base + lane;
        scratch_kq[kq_idx] += scratch_kq[kq_idx + 8] + scratch_kq[kq_idx + 16] +
                              scratch_kq[kq_idx + 24];
      }
    }
    ucl::SyncThreads<WorkGroup, Local>();
    float4 kv_sum = scratch_kv[sh_base + 0] + scratch_kv[sh_base + 1] + scratch_kv[sh_base + 2] +
                    scratch_kv[sh_base + 3] + scratch_kv[sh_base + 4] + scratch_kv[sh_base + 5] +
                    scratch_kv[sh_base + 6] + scratch_kv[sh_base + 7];
    float4 delta_slice = (v_vec - kv_sum) * ucl::Init<float4>(gate_cur.x);
    s0 = s0 + delta_slice * ucl::Init<float4>(k_val.x);
    s1 = s1 + delta_slice * ucl::Init<float4>(k_val.y);
    s2 = s2 + delta_slice * ucl::Init<float4>(k_val.z);
    s3 = s3 + delta_slice * ucl::Init<float4>(k_val.w);
    if (lane == 0) {
      float4 sq_sum = scratch_sq[sh_base + 0] + scratch_sq[sh_base + 1] + scratch_sq[sh_base + 2] +
                      scratch_sq[sh_base + 3] + scratch_sq[sh_base + 4] + scratch_sq[sh_base + 5] +
                      scratch_sq[sh_base + 6] + scratch_sq[sh_base + 7];
      float kq_sum = scratch_kq[kq_base + 0] + scratch_kq[kq_base + 1] + scratch_kq[kq_base + 2] +
                     scratch_kq[kq_base + 3] + scratch_kq[kq_base + 4] + scratch_kq[kq_base + 5] +
                     scratch_kq[kq_base + 6] + scratch_kq[kq_base + 7];
      float4 attn_out = sq_sum + delta_slice * ucl::Init<float4>(kq_sum);
      args.raw_out.Write(attn_out, t, 0, out_slice, b);
    }
  }
  args.recurrent_state_out.Write(s0, k_base + 0, h, v_slice, b);
  args.recurrent_state_out.Write(s1, k_base + 1, h, v_slice, b);
  args.recurrent_state_out.Write(s2, k_base + 2, h, v_slice, b);
  args.recurrent_state_out.Write(s3, k_base + 3, h, v_slice, b);
}
)";

// Gated RMSNorm x SiLU(z) of the un-normalized float32 output: one workgroup of
// 32 threads per (token, value head), thread `v_slice` owns one slice of D_v.
constexpr char kGatedDeltaNetNormShader[] = R"(
MAIN_FUNCTION($0) {
  int v_slice = ucl::GetLocalId<0>();
  int t = ucl::GetGroupId<0>();
  int h = ucl::GetGroupId<1>();
  int b = ucl::GetGroupId<2>();
  __local float scratch_sumsq[32];
  int out_slice = h * HEAD_V_DIM_SLICES + min(v_slice, HEAD_V_DIM_SLICES - 1);
  float4 o = ucl::Convert<float4>(args.raw_out.Read(t, 0, out_slice, b));
  float sumsq = 0.0f;
  if (v_slice < HEAD_V_DIM_SLICES) {
    sumsq = dot(o, o);
  }
  scratch_sumsq[v_slice] = sumsq;
  ucl::SyncThreads<WorkGroup, Local>();
  if (v_slice < 8) {
    scratch_sumsq[v_slice] += scratch_sumsq[v_slice + 8] + scratch_sumsq[v_slice + 16] +
                              scratch_sumsq[v_slice + 24];
  }
  ucl::SyncThreads<WorkGroup, Local>();
  float head_sumsq = scratch_sumsq[0] + scratch_sumsq[1] + scratch_sumsq[2] + scratch_sumsq[3] +
                     scratch_sumsq[4] + scratch_sumsq[5] + scratch_sumsq[6] + scratch_sumsq[7];
  float inv_rms = rsqrt(head_sumsq * args.inv_head_v_dim + args.norm_eps);
  if (v_slice < HEAD_V_DIM_SLICES) {
    float4 norm_w = NORM_WEIGHT_EXPR;
    float4 z_vec = ucl::Convert<float4>(args.zba.Read(t, 0, out_slice, b));
    float4 z_gate = z_vec / (ucl::Init<float4>(1.0f) + exp(-z_vec));
    float4 gated_out = o * ucl::Init<float4>(inv_rms) * norm_w * z_gate;
    args.output.Write(ucl::Convert<ActivationType>(gated_out), t, 0, out_slice, b);
  }
}
)";

// Shader expression of element `index_expr` of the 1D constant `name` with
// `size` elements. TFLite 1D tensors map to BHWC (size, 1, 1, 1).
absl::StatusOr<std::string> ConstElementExpr(
    absl::string_view name, const ::ml_drift::TensorDescriptor& desc, int size,
    absl::string_view index_expr) {
  const auto shape = desc.GetBHWCShape();
  if (shape.b == size && shape.h == 1 && shape.w == 1 && shape.c == 1) {
    return absl::StrCat("args.", name, ".Read<float>(0, 0, 0, ", index_expr,
                        ").x");
  }
  if (shape.b == 1 && shape.h == 1 && shape.w == size && shape.c == 1) {
    return absl::StrCat("args.", name, ".Read<float>(", index_expr,
                        ", 0, 0, 0).x");
  }
  return absl::InvalidArgumentError(
      absl::StrCat("gated_delta_update: unsupported layout of ", name, " (",
                   shape.b, ", ", shape.h, ", ", shape.w, ", ", shape.c, ")."));
}

// float4 of the gated RMSNorm weight at D_v elements [4 * v_slice, +4).
absl::StatusOr<std::string> NormWeightExpr(
    const ::ml_drift::TensorDescriptor& desc, int head_v_dim) {
  std::string expr = "ucl::Init<float4>(";
  for (int i = 0; i < 4; ++i) {
    ABSL_ASSIGN_OR_RETURN(const std::string element,
                          ConstElementExpr("norm_weight", desc, head_v_dim,
                                           absl::StrCat("v_slice * 4 + ", i)));
    absl::StrAppend(&expr, i > 0 ? ", " : "", element);
  }
  return absl::StrCat(expr, ")");
}

// Validated dimensions of an `gated_delta_update` operation.
struct GdnDims {
  int seq_len = 1;
  int key_dim = 0;
  int value_dim = 0;
  int q_slices = 0;
  int v_slices = 0;
};

absl::StatusOr<GdnDims> GetGdnConvDims(
    const ::ml_drift::TensorDescriptor& conv_out,
    const GatedDeltaNetAttributes& attr) {
  ABSL_RETURN_IF_ERROR(CheckGatedDeltaNetAttributes(attr));
  GdnDims dims;
  dims.key_dim = attr.num_k_heads * attr.head_k_dim;
  dims.value_dim = attr.num_v_heads * attr.head_v_dim;
  dims.q_slices = attr.head_k_dim / 4;
  dims.v_slices = attr.head_v_dim / 4;
  const auto conv_shape = conv_out.GetBHWCShape();
  dims.seq_len = conv_shape.w;
  if (conv_shape.b != 1 || conv_shape.h != 1 || dims.seq_len <= 0 ||
      conv_shape.c != 2 * dims.key_dim + dims.value_dim) {
    return absl::InvalidArgumentError(
        "gated_delta_update: conv_out must be [1, T, 2 * key_dim + "
        "value_dim].");
  }
  return dims;
}

absl::Status CheckGdnZba(const ::ml_drift::TensorDescriptor& zba,
                         const GdnDims& dims,
                         const GatedDeltaNetAttributes& attr) {
  const auto zba_shape = zba.GetBHWCShape();
  if (zba_shape.b != 1 || zba_shape.h != 1 || zba_shape.w != dims.seq_len ||
      zba_shape.c != dims.value_dim + 2 * attr.num_v_heads) {
    return absl::InvalidArgumentError(
        "gated_delta_update: zba must be [1, T, value_dim + 2 * H_v].");
  }
  return absl::OkStatus();
}

absl::Status CheckGdnState(const ::ml_drift::TensorDescriptor& state,
                           const GatedDeltaNetAttributes& attr) {
  if (state.GetStorageType() != ::ml_drift::TensorStorageType::kBuffer ||
      state.GetDataType() != ::ml_drift::DataType::kFloat32) {
    return absl::InvalidArgumentError(
        "gated_delta_update requires a float32 kBuffer recurrent state.");
  }
  const auto state_shape = state.GetBHWCShape();
  if (state_shape.b != 1 || state_shape.h != attr.num_v_heads ||
      state_shape.w != attr.head_k_dim || state_shape.c != attr.head_v_dim) {
    return absl::InvalidArgumentError(
        "gated_delta_update: recurrent state must be [1, H_v, D_k, D_v].");
  }
  return absl::OkStatus();
}

// Recurrent state code of the wave shaders.
struct GdnWaveStateCode {
  std::string init;
  std::string decay;
  std::string kv_mem;
  std::string reduce_kv_mem;
  std::string update;
  std::string attn_out;
  std::string reduce_attn_out;
  std::string write_out;
};

GdnWaveStateCode BuildGdnWaveStateCode(int k_rounds, absl::string_view indent) {
  static constexpr absl::string_view kComponents[] = {"x", "y", "z", "w"};
  GdnWaveStateCode c;
  for (int m = 0; m < k_rounds; ++m) {
    const std::string ms = absl::StrCat(m);
    const std::string lane_expr =
        m == 0 ? "k_lane" : absl::StrCat("k_lane + ", m * kGdnKThreads);
    const std::string k_shuffle = absl::StrCat(
        "ucl::WaveShuffle(k_loaded_0, ucl::Convert<uint>(", lane_expr, "))");
    const std::string q_shuffle = absl::StrCat(
        "ucl::WaveShuffle(q_loaded_0, ucl::Convert<uint>(", lane_expr, "))");
    absl::StrAppend(&c.init, "  int k_base_", ms, " = (", lane_expr,
                    ") * 4;\n");
    for (int i = 0; i < 4; ++i) {
      absl::StrAppend(&c.init, "  float4 s", i, "_", ms,
                      " = ucl::Convert<float4>(args.recurrent_state_out.Read("
                      "k_base_",
                      ms, " + ", i, ", h, v_slice, b));\n");
      absl::StrAppend(&c.decay, indent, "s", i, "_", ms, " = s", i, "_", ms,
                      " * decay_vec;\n");
      absl::StrAppend(&c.write_out, "  args.recurrent_state_out.Write(s", i,
                      "_", ms, ", k_base_", ms, " + ", i,
                      ", h, v_slice, b);\n");
    }
    absl::StrAppend(&c.kv_mem, indent, "{\n", indent, "  float4 k_val_", ms,
                    " = ", k_shuffle, ";\n", indent, "  kv_mem += ");
    absl::StrAppend(&c.update, indent, "{\n", indent, "  float4 k_val_", ms,
                    " = ", k_shuffle, ";\n");
    absl::StrAppend(&c.attn_out, indent, "{\n", indent, "  float4 q_val_", ms,
                    " = ", q_shuffle, ";\n", indent, "  my_attn_out += ");
    for (int i = 0; i < 4; ++i) {
      const absl::string_view comp = kComponents[i];
      absl::StrAppend(&c.kv_mem, i > 0 ? " + " : "", "s", i, "_", ms,
                      " * ucl::Init<float4>(k_val_", ms, ".", comp, ")");
      absl::StrAppend(&c.update, indent, "  s", i, "_", ms, " = s", i, "_", ms,
                      " + delta_slice * ucl::Init<float4>(k_val_", ms, ".",
                      comp, ");\n");
      absl::StrAppend(&c.attn_out, i > 0 ? " + " : "", "s", i, "_", ms,
                      " * ucl::Init<float4>(q_val_", ms, ".", comp, ")");
    }
    absl::StrAppend(&c.kv_mem, ";\n", indent, "}\n");
    absl::StrAppend(&c.update, indent, "}\n");
    absl::StrAppend(&c.attn_out, ";\n", indent, "}\n");
  }
  for (int offset = 1; offset < kGdnKThreads; offset *= 2) {
    absl::StrAppend(&c.reduce_kv_mem, indent,
                    "kv_mem += ucl::WaveShuffleXor(kv_mem, ", offset, "u);\n");
    absl::StrAppend(&c.reduce_attn_out, indent,
                    "my_attn_out += ucl::WaveShuffleXor(my_attn_out, ", offset,
                    "u);\n");
  }
  return c;
}

void SubstituteGdnCommonDims(const GatedDeltaNetAttributes& attr,
                             const GdnDims& dims,
                             const GdnWaveStateCode* state_code,
                             ::ml_drift::DataType output_type,
                             std::string* code) {
  absl::StrReplaceAll({{"GDN_SCALAR_GATES", kGdnScalarGates}}, code);
  if (state_code != nullptr) {
    absl::StrReplaceAll({{"STATE_INIT_DECLS", state_code->init},
                         {"DECAY_STEP", state_code->decay},
                         {"KV_MEM_STEP", state_code->kv_mem},
                         {"SIMD_REDUCE_KV_MEM", state_code->reduce_kv_mem},
                         {"UPDATE_STATE_STEP", state_code->update},
                         {"ATTN_OUT_STEP", state_code->attn_out},
                         {"SIMD_REDUCE_ATTN_OUT", state_code->reduce_attn_out},
                         {"STATE_WRITE_OUT_STEP", state_code->write_out}},
                        code);
  }
  absl::StrReplaceAll(
      {{"V_TILE", absl::StrCat(kGdnVTile)},
       {"K_THREADS", absl::StrCat(kGdnKThreads)},
       {"V_GROUPS", absl::StrCat(dims.v_slices / kGdnVTile)},
       {"GQA_RATIO", absl::StrCat(attr.num_v_heads / attr.num_k_heads)},
       {"BETA_CH_OFFSET", absl::StrCat(dims.value_dim)},
       {"GATE_CH_OFFSET", absl::StrCat(dims.value_dim + attr.num_v_heads)},
       {"HEAD_K_DIM_SLICES", absl::StrCat(dims.q_slices)},
       {"HEAD_V_DIM_SLICES", absl::StrCat(dims.v_slices)},
       {"K_SLICE_OFFSET", absl::StrCat(dims.key_dim / 4)},
       {"V_SLICE_OFFSET", absl::StrCat(2 * dims.key_dim / 4)},
       {"SEQ_LEN", absl::StrCat(dims.seq_len)},
       {"ActivationType", ::ml_drift::ToUclDataType(output_type, 4)}},
      code);
}

}  // namespace

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaNetDecode(const ::ml_drift::OperationDef& definition,
                          const GatedDeltaNetAttributes& attr,
                          const ::ml_drift::GpuInfo& gpu_info,
                          bool use_wave_ops) {
  if (use_wave_ops && !SupportsWave32Vec4Shuffle(gpu_info)) {
    return absl::UnimplementedError(
        "gated_delta_update decode wave kernel requires 32-lane wave "
        "shuffles.");
  }
  if (definition.src_tensors.size() != 5 &&
      definition.src_tensors.size() != 6) {
    return absl::InvalidArgumentError(
        "gated_delta_update decode kernel expects 5 or 6 inputs.");
  }
  if (definition.dst_tensors.size() != 2) {
    return absl::InvalidArgumentError(
        "gated_delta_update decode kernel expects 2 outputs.");
  }
  const auto& conv_out = definition.src_tensors[0];
  const auto& zba = definition.src_tensors[1];
  const auto& a_log = definition.src_tensors[2];
  const auto& dt_bias = definition.src_tensors[3];
  const auto& norm_weight = definition.src_tensors[4];
  const bool has_mask = definition.src_tensors.size() == 6;
  const auto& output = definition.dst_tensors[0];
  const auto& state = definition.dst_tensors[1];
  ABSL_ASSIGN_OR_RETURN(const GdnDims dims, GetGdnConvDims(conv_out, attr));
  ABSL_RETURN_IF_ERROR(CheckGdnZba(zba, dims, attr));
  ABSL_RETURN_IF_ERROR(CheckGdnState(state, attr));
  if (dims.seq_len != 1) {
    return absl::InvalidArgumentError(
        "gated_delta_update decode kernel takes one token.");
  }
  if (has_mask &&
      definition.src_tensors[5].GetBHWCShape().DimensionsProduct() != 1) {
    return absl::InvalidArgumentError(
        "gated_delta_update decode kernel requires a 1-element valid_mask.");
  }
  const int v_groups = dims.v_slices / kGdnVTile;
  const int decode_v_tile = (dims.v_slices % 8 == 0) ? 8 : 4;
  const int decode_wg_size = kSubgroupSize * decode_v_tile;
  const int decode_v_chunks = dims.v_slices / decode_v_tile;
  const int work_group_size_x =
      use_wave_ops ? kSubgroupSize * v_groups : decode_wg_size;
  auto op = std::make_unique<GatedDeltaUpdateOp>(
      /*batch_size=*/1, attr.num_v_heads, /*v_groups=*/1, work_group_size_x);
  op->AddSrcTensor("conv_out", conv_out);
  op->AddSrcTensor("zba", zba);
  op->AddSrcTensor("a_log", a_log);
  op->AddSrcTensor("dt_bias", dt_bias);
  op->AddSrcTensor("norm_weight", norm_weight);
  if (has_mask) {
    op->AddSrcTensor("valid_mask", definition.src_tensors[5]);
  }
  op->AddDstTensor("output", output);
  op->AddDstTensor("recurrent_state_out", state);
  op->args_.AddFloat("eps", attr.eps);
  op->args_.AddFloat("q_scale",
                     1.0f / std::sqrt(static_cast<float>(attr.head_k_dim)));
  op->args_.AddFloat("norm_eps", attr.norm_eps);
  op->args_.AddFloat("inv_head_v_dim",
                     1.0f / static_cast<float>(attr.head_v_dim));

  ABSL_ASSIGN_OR_RETURN(
      const std::string a_log_expr,
      ConstElementExpr("a_log", a_log, attr.num_v_heads, "h"));
  ABSL_ASSIGN_OR_RETURN(
      const std::string dt_bias_expr,
      ConstElementExpr("dt_bias", dt_bias, attr.num_v_heads, "h"));
  ABSL_ASSIGN_OR_RETURN(const std::string norm_weight_expr,
                        NormWeightExpr(norm_weight, attr.head_v_dim));
  const std::string mask_expr =
      has_mask ? "args.valid_mask.Read<float>(0, 0, 0, b).x" : "1.0f";

  std::string code;
  GdnWaveStateCode state_code;
  if (use_wave_ops) {
    std::string norm_partial_sum;
    for (int g = 0; g < v_groups; ++g) {
      absl::StrAppend(&norm_partial_sum, g > 0 ? " + " : "", "norm_partial[", g,
                      "]");
    }
    state_code =
        BuildGdnWaveStateCode(dims.q_slices / kGdnKThreads, /*indent=*/"  ");
    code = kGatedDeltaNetDecodeShader;
    absl::StrReplaceAll({{"A_LOG_EXPR", a_log_expr},
                         {"DT_BIAS_EXPR", dt_bias_expr},
                         {"MASK_EXPR", mask_expr},
                         {"NORM_WEIGHT_EXPR", norm_weight_expr},
                         {"NORM_PARTIAL_SUM", norm_partial_sum}},
                        &code);
    SubstituteGdnCommonDims(attr, dims, &state_code, output.GetDataType(),
                            &code);
    ResolveWaveSimd(gpu_info, &code);
  } else {
    code = kGatedDeltaNetDecodeSharedMemShader;
    absl::StrReplaceAll({{"DEC_WG_SIZE", absl::StrCat(decode_wg_size)},
                         {"DEC_VTILE", absl::StrCat(decode_v_tile)},
                         {"DEC_VCHUNKS", absl::StrCat(decode_v_chunks)},
                         {"A_LOG_EXPR", a_log_expr},
                         {"DT_BIAS_EXPR", dt_bias_expr},
                         {"MASK_EXPR", mask_expr},
                         {"NORM_WEIGHT_EXPR", norm_weight_expr}},
                        &code);
    SubstituteGdnCommonDims(attr, dims, /*state_code=*/nullptr,
                            output.GetDataType(), &code);
  }
  op->code_ = std::move(code);
  return op;
}

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaNetPrep(const ::ml_drift::OperationDef& definition,
                        const GatedDeltaNetAttributes& attr) {
  if (definition.src_tensors.size() != 4 &&
      definition.src_tensors.size() != 5) {
    return absl::InvalidArgumentError(
        "gated_delta_update prep expects 4 or 5 inputs.");
  }
  if (definition.dst_tensors.size() != 2) {
    return absl::InvalidArgumentError(
        "gated_delta_update prep expects 2 outputs.");
  }
  const auto& conv_out = definition.src_tensors[0];
  const auto& zba = definition.src_tensors[1];
  const auto& a_log = definition.src_tensors[2];
  const auto& dt_bias = definition.src_tensors[3];
  const bool has_mask = definition.src_tensors.size() == 5;
  const auto& qk_norm = definition.dst_tensors[0];
  const auto& gates = definition.dst_tensors[1];
  ABSL_ASSIGN_OR_RETURN(const GdnDims dims, GetGdnConvDims(conv_out, attr));
  ABSL_RETURN_IF_ERROR(CheckGdnZba(zba, dims, attr));
  const auto qk_shape = qk_norm.GetBHWCShape();
  if (qk_norm.GetDataType() != ::ml_drift::DataType::kFloat32 ||
      qk_shape.b != 1 || qk_shape.h != 1 || qk_shape.w != dims.seq_len ||
      qk_shape.c != 2 * dims.key_dim) {
    return absl::InvalidArgumentError(
        "gated_delta_update prep: qk_norm must be float32 [1, T, "
        "2 * key_dim].");
  }
  const auto gates_shape = gates.GetBHWCShape();
  if (gates.GetDataType() != ::ml_drift::DataType::kFloat32 ||
      gates_shape.b != 1 || gates_shape.h != 1 ||
      gates_shape.w != dims.seq_len || gates_shape.c != 4 * attr.num_v_heads) {
    return absl::InvalidArgumentError(
        "gated_delta_update prep: gates must be float32 [1, T, 4 * H_v].");
  }

  std::string mask_init = "  float tok_mask = 1.0f;";
  if (has_mask) {
    const auto mask_shape = definition.src_tensors[4].GetBHWCShape();
    if (mask_shape.b == 1 && mask_shape.h == 1 && mask_shape.w == 1 &&
        mask_shape.c == dims.seq_len) {
      mask_init =
          R"(  float4 mask_vec = ucl::Convert<float4>(args.valid_mask.Read(0, 0, t / 4, b));
  int mask_elem = t % 4;
  float tok_mask = mask_vec.x;
  if (mask_elem == 1) {
    tok_mask = mask_vec.y;
  } else if (mask_elem == 2) {
    tok_mask = mask_vec.z;
  } else if (mask_elem == 3) {
    tok_mask = mask_vec.w;
  })";
    } else if (mask_shape.b == 1 && mask_shape.h == 1 &&
               mask_shape.w == dims.seq_len && mask_shape.c == 1) {
      mask_init =
          "  float tok_mask = ucl::Convert<float4>(args.valid_mask.Read(t, 0, "
          "0, b)).x;";
    } else {
      return absl::InvalidArgumentError(
          "gated_delta_update prep: valid_mask must be [1, T] or [1, T, 1].");
    }
  }

  auto op = std::make_unique<GatedDeltaUpdateOp>(
      /*batch_size=*/1, attr.num_k_heads, /*v_groups=*/dims.seq_len,
      /*work_group_size_x=*/kSubgroupSize);
  op->AddSrcTensor("conv_out", conv_out);
  op->AddSrcTensor("zba", zba);
  op->AddSrcTensor("a_log", a_log);
  op->AddSrcTensor("dt_bias", dt_bias);
  if (has_mask) {
    op->AddSrcTensor("valid_mask", definition.src_tensors[4]);
  }
  op->AddDstTensor("qk_norm", qk_norm);
  op->AddDstTensor("gates", gates);
  op->args_.AddFloat("eps", attr.eps);
  op->args_.AddFloat("q_scale",
                     1.0f / std::sqrt(static_cast<float>(attr.head_k_dim)));

  ABSL_ASSIGN_OR_RETURN(
      const std::string a_log_expr,
      ConstElementExpr("a_log", a_log, attr.num_v_heads, "h"));
  ABSL_ASSIGN_OR_RETURN(
      const std::string dt_bias_expr,
      ConstElementExpr("dt_bias", dt_bias, attr.num_v_heads, "h"));
  std::string code = kGatedDeltaNetPrepShader;
  absl::StrReplaceAll({{"MASK_INIT", mask_init},
                       {"A_LOG_EXPR", a_log_expr},
                       {"DT_BIAS_EXPR", dt_bias_expr}},
                      &code);
  SubstituteGdnCommonDims(attr, dims, /*state_code=*/nullptr,
                          qk_norm.GetDataType(), &code);
  op->code_ = std::move(code);
  return op;
}

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaNetRecurrence(const ::ml_drift::OperationDef& definition,
                              const GatedDeltaNetAttributes& attr,
                              const ::ml_drift::GpuInfo& gpu_info,
                              bool use_wave_ops) {
  if (use_wave_ops && !SupportsWave32Vec4Shuffle(gpu_info)) {
    return absl::UnimplementedError(
        "gated_delta_update: the GPU has no 32-lane wave shuffles.");
  }
  if (definition.src_tensors.size() != 3) {
    return absl::InvalidArgumentError(
        "gated_delta_update recurrence expects 3 inputs (conv_out, qk_norm, "
        "gates).");
  }
  if (definition.dst_tensors.size() != 2) {
    return absl::InvalidArgumentError(
        "gated_delta_update recurrence expects 2 outputs.");
  }
  const auto& conv_out = definition.src_tensors[0];
  const auto& qk_norm = definition.src_tensors[1];
  const auto& gates = definition.src_tensors[2];
  const auto& raw_out = definition.dst_tensors[0];
  const auto& state = definition.dst_tensors[1];
  ABSL_ASSIGN_OR_RETURN(const GdnDims dims, GetGdnConvDims(conv_out, attr));
  ABSL_RETURN_IF_ERROR(CheckGdnState(state, attr));
  const auto qk_shape = qk_norm.GetBHWCShape();
  if (qk_norm.GetDataType() != ::ml_drift::DataType::kFloat32 ||
      qk_shape.b != 1 || qk_shape.h != 1 || qk_shape.w != dims.seq_len ||
      qk_shape.c != 2 * dims.key_dim) {
    return absl::InvalidArgumentError(
        "gated_delta_update: qk_norm must be float32 [1, T, 2 * key_dim].");
  }
  const auto gates_shape = gates.GetBHWCShape();
  if (gates.GetDataType() != ::ml_drift::DataType::kFloat32 ||
      gates_shape.b != 1 || gates_shape.h != 1 ||
      gates_shape.w != dims.seq_len || gates_shape.c != 4 * attr.num_v_heads) {
    return absl::InvalidArgumentError(
        "gated_delta_update: gates must be float32 [1, T, 4 * H_v].");
  }
  const auto raw_shape = raw_out.GetBHWCShape();
  if (raw_out.GetDataType() != ::ml_drift::DataType::kFloat32 ||
      raw_shape.b != 1 || raw_shape.h != 1 || raw_shape.w != dims.seq_len ||
      raw_shape.c != dims.value_dim) {
    return absl::InvalidArgumentError(
        "gated_delta_update: the recurrence output must be float32 [1, T, "
        "value_dim].");
  }

  // Wave: one 32-lane workgroup per (4 v_slices, value head).
  // Shared memory: one 64-thread workgroup per (2 v_slices, value head).
  const int v_groups = use_wave_ops ? dims.v_slices / kGdnVTile
                                    : dims.v_slices / kGdnSharedVTile;
  const int work_group_size_x =
      use_wave_ops ? kSubgroupSize : kSubgroupSize * kGdnSharedVTile;
  auto op = std::make_unique<GatedDeltaUpdateOp>(
      /*batch_size=*/1, attr.num_v_heads, v_groups, work_group_size_x);
  op->AddSrcTensor("conv_out", conv_out);
  op->AddSrcTensor("qk_norm", qk_norm);
  op->AddSrcTensor("gates", gates);
  op->AddDstTensor("raw_out", raw_out);
  op->AddDstTensor("recurrent_state_out", state);

  std::string code;
  GdnWaveStateCode state_code;
  if (use_wave_ops) {
    code = kGatedDeltaNetPrefillShader;
    state_code = BuildGdnWaveStateCode(dims.q_slices / kGdnKThreads,
                                       /*indent=*/"      ");
  } else {
    code = kGatedDeltaNetSharedMemShader;
  }
  SubstituteGdnCommonDims(attr, dims, use_wave_ops ? &state_code : nullptr,
                          raw_out.GetDataType(), &code);
  if (use_wave_ops) {
    ResolveWaveSimd(gpu_info, &code);
  }
  op->code_ = std::move(code);
  return op;
}

absl::StatusOr<std::unique_ptr<::ml_drift::GPUOperation>>
CreateGatedDeltaNetNorm(const ::ml_drift::OperationDef& definition,
                        const GatedDeltaNetAttributes& attr) {
  ABSL_RETURN_IF_ERROR(CheckGatedDeltaNetAttributes(attr));
  if (definition.src_tensors.size() != 3 ||
      definition.dst_tensors.size() != 1) {
    return absl::InvalidArgumentError(
        "gated_delta_update norm expects 3 inputs and 1 output.");
  }
  const auto& raw_out = definition.src_tensors[0];
  const auto& zba = definition.src_tensors[1];
  const auto& norm_weight = definition.src_tensors[2];
  const auto& output = definition.dst_tensors[0];
  const int value_dim = attr.num_v_heads * attr.head_v_dim;
  const auto shape = raw_out.GetBHWCShape();
  const auto out_shape = output.GetBHWCShape();
  if (shape.b != 1 || shape.h != 1 || shape.c != value_dim ||
      out_shape.b != 1 || out_shape.h != 1 || out_shape.w != shape.w ||
      out_shape.c != value_dim) {
    return absl::InvalidArgumentError(
        "gated_delta_update norm: input and output must be [1, T, "
        "value_dim].");
  }
  auto op = std::make_unique<GatedDeltaUpdateOp>(
      /*batch_size=*/1, attr.num_v_heads, /*v_groups=*/shape.w,
      /*work_group_size_x=*/kSubgroupSize);
  op->AddSrcTensor("raw_out", raw_out);
  op->AddSrcTensor("zba", zba);
  op->AddSrcTensor("norm_weight", norm_weight);
  op->AddDstTensor("output", output);
  op->args_.AddFloat("norm_eps", attr.norm_eps);
  op->args_.AddFloat("inv_head_v_dim",
                     1.0f / static_cast<float>(attr.head_v_dim));
  ABSL_ASSIGN_OR_RETURN(const std::string norm_weight_expr,
                        NormWeightExpr(norm_weight, attr.head_v_dim));
  std::string code = kGatedDeltaNetNormShader;
  absl::StrReplaceAll(
      {{"NORM_WEIGHT_EXPR", norm_weight_expr},
       {"HEAD_V_DIM_SLICES", absl::StrCat(attr.head_v_dim / 4)},
       {"ActivationType", ::ml_drift::ToUclDataType(output.GetDataType(), 4)}},
      &code);
  op->code_ = std::move(code);
  return op;
}

namespace {

template <typename T>
std::vector<::ml_drift::ValueId> GetTensorIds(const std::vector<T>& tensors) {
  std::vector<::ml_drift::ValueId> ids;
  ids.reserve(tensors.size());
  for (const auto& t : tensors) {
    ids.push_back(t->id);
  }
  return ids;
}

absl::Status BuildGatedDeltaUpdateGpuGraph(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<::ml_drift::ValueId>& src_ids,
    const std::vector<::ml_drift::ValueId>& dst_ids, int mode,
    const ::ml_drift::GpuInfo* gpu_info,
    ::ml_drift::GpuModelBuilder* model_builder) {
  ABSL_ASSIGN_OR_RETURN(auto op,
                        CreateGatedDeltaUpdate(op_def, mode, gpu_info));
  model_builder->AddGpuOperation(src_ids, dst_ids, std::move(op),
                                 "gated_delta_update");
  return absl::OkStatus();
}

absl::Status BuildGatedDeltaNetGpuGraph(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<::ml_drift::ValueId>& src_ids,
    const std::vector<::ml_drift::ValueId>& dst_ids,
    const GatedDeltaNetAttributes& attr, const ::ml_drift::GpuInfo* gpu_info,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (gpu_info == nullptr) {
    return absl::InvalidArgumentError(
        "gated_delta_update block mode requires the GPU info.");
  }
  if ((src_ids.size() != 5 && src_ids.size() != 6) || dst_ids.size() != 2 ||
      op_def.src_tensors.size() != src_ids.size() ||
      op_def.dst_tensors.size() != dst_ids.size()) {
    return absl::InvalidArgumentError(
        "gated_delta_update block mode expects 5 or 6 GPU inputs and 2 "
        "outputs.");
  }
  const bool use_wave_ops = SupportsWave32Vec4Shuffle(*gpu_info);
  const bool has_mask = src_ids.size() == 6;
  const int seq_len = op_def.src_tensors[0].GetBHWCShape().w;
  if (seq_len == 1) {
    ABSL_ASSIGN_OR_RETURN(auto op, CreateGatedDeltaNetDecode(
                                       op_def, attr, *gpu_info, use_wave_ops));
    model_builder->AddGpuOperation(src_ids, dst_ids, std::move(op),
                                   "gated_delta_update");
    return absl::OkStatus();
  }

  ABSL_RETURN_IF_ERROR(CheckGatedDeltaNetAttributes(attr));
  auto add_f32_buffer = [&](int channels) {
    ::ml_drift::TensorDescriptor desc = op_def.dst_tensors[0];
    desc.SetDataType(::ml_drift::DataType::kFloat32);
    desc.SetStorageType(::ml_drift::TensorStorageType::kBuffer);
    desc.SetBHWCShape(::ml_drift::BHWC(1, 1, seq_len, channels));
    return model_builder->AddTensor(desc);
  };
  const ::ml_drift::TensorHandle qk_norm =
      add_f32_buffer(2 * attr.num_k_heads * attr.head_k_dim);
  const ::ml_drift::TensorHandle gates = add_f32_buffer(4 * attr.num_v_heads);
  const ::ml_drift::TensorHandle raw =
      add_f32_buffer(attr.num_v_heads * attr.head_v_dim);

  ::ml_drift::OperationDef prep_def = op_def;
  prep_def.src_tensors = {op_def.src_tensors[0], op_def.src_tensors[1],
                          op_def.src_tensors[2], op_def.src_tensors[3]};
  std::vector<::ml_drift::ValueId> prep_src_ids = {src_ids[0], src_ids[1],
                                                   src_ids[2], src_ids[3]};
  if (has_mask) {
    prep_def.src_tensors.push_back(op_def.src_tensors[5]);
    prep_src_ids.push_back(src_ids[5]);
  }
  prep_def.dst_tensors = {qk_norm.tensor_desc, gates.tensor_desc};
  ABSL_ASSIGN_OR_RETURN(auto prep_op, CreateGatedDeltaNetPrep(prep_def, attr));
  model_builder->AddGpuOperation(prep_src_ids, {qk_norm.id, gates.id},
                                 std::move(prep_op), "gated_delta_update_prep");

  ::ml_drift::OperationDef recurrence_def = op_def;
  recurrence_def.src_tensors = {op_def.src_tensors[0], qk_norm.tensor_desc,
                                gates.tensor_desc};
  recurrence_def.dst_tensors = {raw.tensor_desc, op_def.dst_tensors[1]};
  ABSL_ASSIGN_OR_RETURN(auto recurrence_op,
                        CreateGatedDeltaNetRecurrence(recurrence_def, attr,
                                                      *gpu_info, use_wave_ops));
  model_builder->AddGpuOperation({src_ids[0], qk_norm.id, gates.id},
                                 {raw.id, dst_ids[1]}, std::move(recurrence_op),
                                 "gated_delta_update_recurrence");

  ::ml_drift::OperationDef norm_def = op_def;
  norm_def.src_tensors = {raw.tensor_desc, op_def.src_tensors[1],
                          op_def.src_tensors[4]};
  norm_def.dst_tensors = {op_def.dst_tensors[0]};
  ABSL_ASSIGN_OR_RETURN(auto norm_op, CreateGatedDeltaNetNorm(norm_def, attr));
  model_builder->AddGpuOperation({raw.id, src_ids[1], src_ids[4]}, {dst_ids[0]},
                                 std::move(norm_op), "gated_delta_update_norm");
  return absl::OkStatus();
}

}  // namespace

absl::Status CreateGatedDeltaUpdateFromNode(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, const ::ml_drift::GpuInfo* gpu_info,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* attr =
      std::any_cast<GatedDeltaUpdateAttributes>(&node.operation.attributes);
  if (!attr) {
    return absl::InvalidArgumentError(
        "Missing attributes for GatedDeltaUpdate operation.");
  }
  if (attr->block.has_value()) {
    return BuildGatedDeltaNetGpuGraph(op_def, GetTensorIds(inputs),
                                      GetTensorIds(outputs), *attr->block,
                                      gpu_info, model_builder);
  }
  return BuildGatedDeltaUpdateGpuGraph(op_def, GetTensorIds(inputs),
                                       GetTensorIds(outputs), attr->mode,
                                       gpu_info, model_builder);
}

absl::Status CreateGatedDeltaUpdateFromIrOp(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& ir_op, const ::ml_drift::GpuInfo* gpu_info,
    ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* attr = std::any_cast<GatedDeltaUpdateAttributes>(&ir_op.attr);
  if (!attr) {
    return absl::InvalidArgumentError(
        "Missing attributes for GatedDeltaUpdate IR operation.");
  }
  if (attr->block.has_value()) {
    return BuildGatedDeltaNetGpuGraph(op_def, GetTensorIds(inputs),
                                      GetTensorIds(outputs), *attr->block,
                                      gpu_info, model_builder);
  }
  return BuildGatedDeltaUpdateGpuGraph(op_def, GetTensorIds(inputs),
                                       GetTensorIds(outputs), attr->mode,
                                       gpu_info, model_builder);
}

}  // namespace litert::ml_drift
