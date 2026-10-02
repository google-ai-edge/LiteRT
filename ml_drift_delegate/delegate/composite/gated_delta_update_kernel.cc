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

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/gated_delta_update_parser.h"

namespace litert::ml_drift {
namespace {

class GatedDeltaUpdateOp : public ::ml_drift::GPUOperation {
 public:
  GatedDeltaUpdateOp(int batch_size, int num_heads, int v_groups, int k_slices)
      : batch_size_(batch_size),
        num_heads_(num_heads),
        v_groups_(v_groups),
        k_slices_(k_slices) {}

  ::ml_drift::int3 GetGridSize() const override {
    // Grid: X=K Slices * Value Groups, Y=Height (Heads), Z=Batch
    return ::ml_drift::int3(k_slices_ * v_groups_, num_heads_, batch_size_);
  }

  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    return {::ml_drift::int3(k_slices_, 1, 1)};
  }

  GatedDeltaUpdateOp(GatedDeltaUpdateOp&& operation) = default;
  GatedDeltaUpdateOp& operator=(GatedDeltaUpdateOp&& operation) = default;
  GatedDeltaUpdateOp(const GatedDeltaUpdateOp&) = delete;
  GatedDeltaUpdateOp& operator=(const GatedDeltaUpdateOp&) = delete;

 private:
  int batch_size_ = 1;
  int num_heads_ = 1;
  int v_groups_ = 1;
  int k_slices_ = 4;
};

// SIMD shuffle-based shader (register-only reduction, zero shared
// memory, zero barriers). Collectively processes V_TILE v_slices per subgroup
// so q_t/k_t loads are reused, beta_t/g_t are loaded uniformly across the
// subgroup, v_t loads are distributed across lanes and shared via
// simd_broadcast, and lanes 0..V_TILE-1 write consecutive v_slices of attn_out
// in parallel.
//
// Final rendered shader example (Metal, D_k=128, D_v=128, V_TILE=4,
// SEQ_LEN_EXPR=128, GQA_RATIO=3, StateType=float4, ActivationType=float4):
// clang-format off
// NOLINTBEGIN(whitespace/line_length)
// ```msl
// MAIN_FUNCTION($0) {
//   int k_slice = ucl::GetLocalId<0>();
//   int k_base = k_slice * 4;
//   int v_base = ucl::GetGroupId<0>() * 4;
//   int h = ucl::GetGlobalId<1>();
//   int h_k = h / 3;
//   int b = ucl::GetGlobalId<2>();
//
//   float4 s0_0 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 0, h, v_base + 0, b));
//   float4 s1_0 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 1, h, v_base + 0, b));
//   float4 s2_0 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 2, h, v_base + 0, b));
//   float4 s3_0 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 3, h, v_base + 0, b));
//   float4 s0_1 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 0, h, v_base + 1, b));
//   float4 s1_1 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 1, h, v_base + 1, b));
//   float4 s2_1 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 2, h, v_base + 1, b));
//   float4 s3_1 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 3, h, v_base + 1, b));
//   float4 s0_2 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 0, h, v_base + 2, b));
//   float4 s1_2 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 1, h, v_base + 2, b));
//   float4 s2_2 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 2, h, v_base + 2, b));
//   float4 s3_2 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 3, h, v_base + 2, b));
//   float4 s0_3 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 0, h, v_base + 3, b));
//   float4 s1_3 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 1, h, v_base + 3, b));
//   float4 s2_3 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 2, h, v_base + 3, b));
//   float4 s3_3 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base + 3, h, v_base + 3, b));
//
//   // Process sequence length sequentially
//   for (int t = 0; t < 128; ++t) {
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
//     // Load full vec4 K once for this thread's k_slice and share across V_TILE
//     float4 k_val = ucl::Convert<float4>(args.k_t.Read(t, h_k, k_slice, b));
//
//     float4 kv_mem_0 = ucl::Init<float4>(0.0);
//     float4 kv_mem_1 = ucl::Init<float4>(0.0);
//     float4 kv_mem_2 = ucl::Init<float4>(0.0);
//     float4 kv_mem_3 = ucl::Init<float4>(0.0);
//     s0_0 = s0_0 * decay_vec;
//     s1_0 = s1_0 * decay_vec;
//     s2_0 = s2_0 * decay_vec;
//     s3_0 = s3_0 * decay_vec;
//     kv_mem_0 = s0_0 * ucl::Init<float4>(k_val.x) +
//                 s1_0 * ucl::Init<float4>(k_val.y) +
//                 s2_0 * ucl::Init<float4>(k_val.z) +
//                 s3_0 * ucl::Init<float4>(k_val.w);
//     s0_1 = s0_1 * decay_vec;
//     s1_1 = s1_1 * decay_vec;
//     s2_1 = s2_1 * decay_vec;
//     s3_1 = s3_1 * decay_vec;
//     kv_mem_1 = s0_1 * ucl::Init<float4>(k_val.x) +
//                 s1_1 * ucl::Init<float4>(k_val.y) +
//                 s2_1 * ucl::Init<float4>(k_val.z) +
//                 s3_1 * ucl::Init<float4>(k_val.w);
//     s0_2 = s0_2 * decay_vec;
//     s1_2 = s1_2 * decay_vec;
//     s2_2 = s2_2 * decay_vec;
//     s3_2 = s3_2 * decay_vec;
//     kv_mem_2 = s0_2 * ucl::Init<float4>(k_val.x) +
//                 s1_2 * ucl::Init<float4>(k_val.y) +
//                 s2_2 * ucl::Init<float4>(k_val.z) +
//                 s3_2 * ucl::Init<float4>(k_val.w);
//     s0_3 = s0_3 * decay_vec;
//     s1_3 = s1_3 * decay_vec;
//     s2_3 = s2_3 * decay_vec;
//     s3_3 = s3_3 * decay_vec;
//     kv_mem_3 = s0_3 * ucl::Init<float4>(k_val.x) +
//                 s1_3 * ucl::Init<float4>(k_val.y) +
//                 s2_3 * ucl::Init<float4>(k_val.z) +
//                 s3_3 * ucl::Init<float4>(k_val.w);
//
//     kv_mem_0 = simd_sum(kv_mem_0);
//     kv_mem_1 = simd_sum(kv_mem_1);
//     kv_mem_2 = simd_sum(kv_mem_2);
//     kv_mem_3 = simd_sum(kv_mem_3);
//
//     // Distribute v_slice loads across lanes 0..V_TILE-1 and share via subgroup broadcast
//     float4 v_loaded = ucl::Init<float4>(0.0);
//     if (k_slice < 4) {
//       v_loaded = ucl::Convert<float4>(args.v_t.Read(t, h, v_base + k_slice, b));
//     }
//     float4 v_vec_0 = simd_broadcast(v_loaded, 0);
//     float4 v_vec_1 = simd_broadcast(v_loaded, 1);
//     float4 v_vec_2 = simd_broadcast(v_loaded, 2);
//     float4 v_vec_3 = simd_broadcast(v_loaded, 3);
//
//     float4 beta_factor = ucl::Init<float4>(ucl::Convert<float>(beta_val));
//
//     // Load full vec4 Q once for this thread's k_slice and share across V_TILE
//     float4 q_val = ucl::Convert<float4>(args.q_t.Read(t, h_k, k_slice, b));
//
//     float4 my_attn_out_0 = ucl::Init<float4>(0.0);
//     float4 my_attn_out_1 = ucl::Init<float4>(0.0);
//     float4 my_attn_out_2 = ucl::Init<float4>(0.0);
//     float4 my_attn_out_3 = ucl::Init<float4>(0.0);
//     float4 delta_slice_0 = (v_vec_0 - kv_mem_0) * beta_factor;
//     s0_0 = s0_0 + delta_slice_0 * ucl::Init<float4>(k_val.x);
//     s1_0 = s1_0 + delta_slice_0 * ucl::Init<float4>(k_val.y);
//     s2_0 = s2_0 + delta_slice_0 * ucl::Init<float4>(k_val.z);
//     s3_0 = s3_0 + delta_slice_0 * ucl::Init<float4>(k_val.w);
//     my_attn_out_0 = s0_0 * ucl::Init<float4>(q_val.x) +
//                     s1_0 * ucl::Init<float4>(q_val.y) +
//                     s2_0 * ucl::Init<float4>(q_val.z) +
//                     s3_0 * ucl::Init<float4>(q_val.w);
//     float4 delta_slice_1 = (v_vec_1 - kv_mem_1) * beta_factor;
//     s0_1 = s0_1 + delta_slice_1 * ucl::Init<float4>(k_val.x);
//     s1_1 = s1_1 + delta_slice_1 * ucl::Init<float4>(k_val.y);
//     s2_1 = s2_1 + delta_slice_1 * ucl::Init<float4>(k_val.z);
//     s3_1 = s3_1 + delta_slice_1 * ucl::Init<float4>(k_val.w);
//     my_attn_out_1 = s0_1 * ucl::Init<float4>(q_val.x) +
//                     s1_1 * ucl::Init<float4>(q_val.y) +
//                     s2_1 * ucl::Init<float4>(q_val.z) +
//                     s3_1 * ucl::Init<float4>(q_val.w);
//     float4 delta_slice_2 = (v_vec_2 - kv_mem_2) * beta_factor;
//     s0_2 = s0_2 + delta_slice_2 * ucl::Init<float4>(k_val.x);
//     s1_2 = s1_2 + delta_slice_2 * ucl::Init<float4>(k_val.y);
//     s2_2 = s2_2 + delta_slice_2 * ucl::Init<float4>(k_val.z);
//     s3_2 = s3_2 + delta_slice_2 * ucl::Init<float4>(k_val.w);
//     my_attn_out_2 = s0_2 * ucl::Init<float4>(q_val.x) +
//                     s1_2 * ucl::Init<float4>(q_val.y) +
//                     s2_2 * ucl::Init<float4>(q_val.z) +
//                     s3_2 * ucl::Init<float4>(q_val.w);
//     float4 delta_slice_3 = (v_vec_3 - kv_mem_3) * beta_factor;
//     s0_3 = s0_3 + delta_slice_3 * ucl::Init<float4>(k_val.x);
//     s1_3 = s1_3 + delta_slice_3 * ucl::Init<float4>(k_val.y);
//     s2_3 = s2_3 + delta_slice_3 * ucl::Init<float4>(k_val.z);
//     s3_3 = s3_3 + delta_slice_3 * ucl::Init<float4>(k_val.w);
//     my_attn_out_3 = s0_3 * ucl::Init<float4>(q_val.x) +
//                     s1_3 * ucl::Init<float4>(q_val.y) +
//                     s2_3 * ucl::Init<float4>(q_val.z) +
//                     s3_3 * ucl::Init<float4>(q_val.w);
//
//     my_attn_out_0 = simd_sum(my_attn_out_0);
//     my_attn_out_1 = simd_sum(my_attn_out_1);
//     my_attn_out_2 = simd_sum(my_attn_out_2);
//     my_attn_out_3 = simd_sum(my_attn_out_3);
//     float4 out_to_write = my_attn_out_0;
//     if (k_slice == 1) {
//       out_to_write = my_attn_out_1;
//     }
//     if (k_slice == 2) {
//       out_to_write = my_attn_out_2;
//     }
//     if (k_slice == 3) {
//       out_to_write = my_attn_out_3;
//     }
//     if (k_slice < 4) {
//       args.output.Write(ucl::Convert<float4>(out_to_write), t, h, v_base + k_slice, b);
//     }
//   }
//
//   // Write out final evolved recurrent state for this thread's 4 rows across V_TILE
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s0_0), k_base + 0, h, v_base + 0, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s1_0), k_base + 1, h, v_base + 0, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s2_0), k_base + 2, h, v_base + 0, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s3_0), k_base + 3, h, v_base + 0, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s0_1), k_base + 0, h, v_base + 1, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s1_1), k_base + 1, h, v_base + 1, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s2_1), k_base + 2, h, v_base + 1, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s3_1), k_base + 3, h, v_base + 1, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s0_2), k_base + 0, h, v_base + 2, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s1_2), k_base + 1, h, v_base + 2, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s2_2), k_base + 2, h, v_base + 2, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s3_2), k_base + 3, h, v_base + 2, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s0_3), k_base + 0, h, v_base + 3, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s1_3), k_base + 1, h, v_base + 3, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s2_3), k_base + 2, h, v_base + 3, b);
//   args.recurrent_state_out.Write(ucl::Convert<float4>(s3_3), k_base + 3, h, v_base + 3, b);
// }
// ```
// NOLINTEND(whitespace/line_length)
// clang-format on
constexpr char kGatedDeltaUpdateShuffleShader[] = R"(
MAIN_FUNCTION($0) {
  int k_slice = ucl::GetLocalId<0>();
  int k_base = k_slice * 4;
  int v_base = ucl::GetGroupId<0>() * V_TILE;
  int h = ucl::GetGlobalId<1>();
  int h_k = h / GQA_RATIO;
  int b = ucl::GetGlobalId<2>();

  STATE_INIT_DECLS

  // Process sequence length sequentially
  for (int t = 0; t < SEQ_LEN_EXPR; ++t) {
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

    // Load full vec4 K once for this thread's k_slice and share across V_TILE
    StateType k_val = ucl::Convert<StateType>(args.k_t.Read(t, h_k, k_slice, b));

    KV_MEM_INIT_DECLS
    DECAY_AND_KV_MEM_STEP

    SIMD_REDUCE_KV_MEM

    // Distribute v_slice loads across lanes 0..V_TILE-1 and share via subgroup broadcast
    StateType v_loaded = ucl::Init<StateType>(0.0);
    if (k_slice < V_TILE) {
      v_loaded = ucl::Convert<StateType>(args.v_t.Read(t, h, v_base + k_slice, b));
    }
    V_BROADCAST_DECLS

    StateType beta_factor = ucl::Init<StateType>(ucl::Convert<StateScalarType>(beta_val));

    // Load full vec4 Q once for this thread's k_slice and share across V_TILE
    StateType q_val = ucl::Convert<StateType>(args.q_t.Read(t, h_k, k_slice, b));

    ATTN_OUT_INIT_DECLS
    UPDATE_STATE_AND_ATTN_OUT_STEP

    SIMD_REDUCE_ATTN_OUT
  }

  // Write out final evolved recurrent state for this thread's 4 rows across V_TILE
  STATE_WRITE_OUT_STEP
}
)";

// Shared memory-based shader (single-phase 2-barrier reduction for OpenCL /
// WebGPU)
constexpr char kGatedDeltaUpdateSharedMemShader[] = R"(
MAIN_FUNCTION($0) {
  int k_slice = ucl::GetLocalId<0>();
  int v_slice = ucl::GetGroupId<0>();
  int h = ucl::GetGlobalId<1>();
  int h_k = h / GQA_RATIO;
  int b = ucl::GetGlobalId<2>();

  __local StateType scratch_kv[HEAD_K_DIM_SLICES];
  __local StateType scratch_out[HEAD_K_DIM_SLICES];
  __local StateScalarType scratch_kq[HEAD_K_DIM_SLICES];

  // Each thread owns 4 rows along D_k (k_slice * 4 + {0, 1, 2, 3}) for this v_slice.
  // Held entirely in 4 registers. Zero register spilling.
  int k_base = k_slice * 4;
  StateType s0 = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base + 0, h, v_slice, b));
  StateType s1 = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base + 1, h, v_slice, b));
  StateType s2 = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base + 2, h, v_slice, b));
  StateType s3 = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base + 3, h, v_slice, b));

  // Process sequence length sequentially
  for (int t = 0; t < SEQ_LEN_EXPR; ++t) {
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
  const auto& rec_state_in = definition.src_tensors[5];

  const auto& output = definition.dst_tensors[0];
  const auto& rec_state_out = definition.dst_tensors[1];

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

  bool is_power_of_two = (q_slices > 0) && ((q_slices & (q_slices - 1)) == 0);
  bool can_use_metal_shuffle =
      gpu_info && gpu_info->IsApiMetal() && is_power_of_two && (q_slices <= 32);

  int v_tile = 1;
  if (can_use_metal_shuffle) {
    if (v_slices % 4 == 0 && q_slices >= 4) {
      v_tile = 4;
    } else if (v_slices % 2 == 0 && q_slices >= 2) {
      v_tile = 2;
    }
  }
  int v_groups = v_slices / v_tile;

  auto op = std::make_unique<GatedDeltaUpdateOp>(B, H, v_groups, q_slices);

  op->AddSrcTensor("q_t", q_t);
  op->AddSrcTensor("k_t", k_t);
  op->AddSrcTensor("v_t", v_t);
  op->AddSrcTensor("beta_t", beta_t);
  op->AddSrcTensor("g_t", g_t);
  op->AddSrcTensor("recurrent_state_in", rec_state_in);

  op->AddDstTensor("output", output);
  op->AddDstTensor("recurrent_state_out", rec_state_out);

  std::string code = can_use_metal_shuffle ? kGatedDeltaUpdateShuffleShader
                                           : kGatedDeltaUpdateSharedMemShader;

  std::string state_init_decls;
  std::string kv_mem_init_decls;
  std::string decay_and_kv_mem_step;
  std::string simd_reduce_kv_mem;
  std::string v_broadcast_decls;
  std::string attn_out_init_decls;
  std::string update_state_and_attn_out_step;
  std::string simd_reduce_attn_out;
  std::string state_write_out_step;

  for (int m = 0; m < v_tile; ++m) {
    std::string ms = std::to_string(m);
    state_init_decls +=
        "  StateType s0_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base + 0, "
        "h, v_base + " +
        ms + ", b));\n" + "  StateType s1_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base + 1, "
        "h, v_base + " +
        ms + ", b));\n" + "  StateType s2_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base + 2, "
        "h, v_base + " +
        ms + ", b));\n" + "  StateType s3_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base + 3, "
        "h, v_base + " +
        ms + ", b));\n";

    kv_mem_init_decls +=
        "    StateType kv_mem_" + ms + " = ucl::Init<StateType>(0.0);\n";

    decay_and_kv_mem_step +=
        "    s0_" + ms + " = s0_" + ms + " * decay_vec;\n" + "    s1_" + ms +
        " = s1_" + ms + " * decay_vec;\n" + "    s2_" + ms + " = s2_" + ms +
        " * decay_vec;\n" + "    s3_" + ms + " = s3_" + ms + " * decay_vec;\n" +
        "    kv_mem_" + ms + " = s0_" + ms +
        " * ucl::Init<StateType>(k_val.x) +\n" + "                s1_" + ms +
        " * ucl::Init<StateType>(k_val.y) +\n" + "                s2_" + ms +
        " * ucl::Init<StateType>(k_val.z) +\n" + "                s3_" + ms +
        " * ucl::Init<StateType>(k_val.w);\n";

    if (q_slices == 32) {
      simd_reduce_kv_mem +=
          "    kv_mem_" + ms + " = simd_sum(kv_mem_" + ms + ");\n";
      simd_reduce_attn_out +=
          "    my_attn_out_" + ms + " = simd_sum(my_attn_out_" + ms + ");\n";
    } else {
      simd_reduce_kv_mem +=
          "    for (int offset = 1; offset < HEAD_K_DIM_SLICES; offset *= 2) "
          "{\n"
          "      kv_mem_" +
          ms + " += simd_shuffle_xor(kv_mem_" + ms + ", offset);\n    }\n";
      simd_reduce_attn_out +=
          "    for (int offset = 1; offset < HEAD_K_DIM_SLICES; offset *= 2) "
          "{\n"
          "      my_attn_out_" +
          ms + " += simd_shuffle_xor(my_attn_out_" + ms + ", offset);\n    }\n";
    }
    v_broadcast_decls += "    StateType v_vec_" + ms +
                         " = simd_broadcast(v_loaded, " + ms + ");\n";

    attn_out_init_decls +=
        "    StateType my_attn_out_" + ms + " = ucl::Init<StateType>(0.0);\n";

    update_state_and_attn_out_step +=
        "    StateType delta_slice_" + ms + " = (v_vec_" + ms + " - kv_mem_" +
        ms + ") * beta_factor;\n" + "    s0_" + ms + " = s0_" + ms +
        " + delta_slice_" + ms + " * ucl::Init<StateType>(k_val.x);\n" +
        "    s1_" + ms + " = s1_" + ms + " + delta_slice_" + ms +
        " * ucl::Init<StateType>(k_val.y);\n" + "    s2_" + ms + " = s2_" + ms +
        " + delta_slice_" + ms + " * ucl::Init<StateType>(k_val.z);\n" +
        "    s3_" + ms + " = s3_" + ms + " + delta_slice_" + ms +
        " * ucl::Init<StateType>(k_val.w);\n" + "    my_attn_out_" + ms +
        " = s0_" + ms + " * ucl::Init<StateType>(q_val.x) +\n" +
        "                    s1_" + ms +
        " * ucl::Init<StateType>(q_val.y) +\n" + "                    s2_" +
        ms + " * ucl::Init<StateType>(q_val.z) +\n" +
        "                    s3_" + ms + " * ucl::Init<StateType>(q_val.w);\n";

    state_write_out_step +=
        "  args.recurrent_state_out.Write(ucl::Convert<StateType>(s0_" + ms +
        "), k_base + 0, h, v_base + " + ms + ", b);\n" +
        "  args.recurrent_state_out.Write(ucl::Convert<StateType>(s1_" + ms +
        "), k_base + 1, h, v_base + " + ms + ", b);\n" +
        "  args.recurrent_state_out.Write(ucl::Convert<StateType>(s2_" + ms +
        "), k_base + 2, h, v_base + " + ms + ", b);\n" +
        "  args.recurrent_state_out.Write(ucl::Convert<StateType>(s3_" + ms +
        "), k_base + 3, h, v_base + " + ms + ", b);\n";
  }

  simd_reduce_attn_out += "    StateType out_to_write = my_attn_out_0;\n";
  for (int m = 1; m < v_tile; ++m) {
    std::string ms = std::to_string(m);
    simd_reduce_attn_out += "    if (k_slice == " + ms +
                            ") {\n      out_to_write = my_attn_out_" + ms +
                            ";\n    }\n";
  }
  simd_reduce_attn_out +=
      "    if (k_slice < V_TILE) {\n"
      "      args.output.Write(ucl::Convert<ActivationType>(out_to_write), t, "
      "h, v_base + k_slice, b);\n"
      "    }\n";

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

  absl::StrReplaceAll(
      {{"STATE_INIT_DECLS", absl::StripAsciiWhitespace(state_init_decls)},
       {"KV_MEM_INIT_DECLS", absl::StripAsciiWhitespace(kv_mem_init_decls)},
       {"DECAY_AND_KV_MEM_STEP",
        absl::StripAsciiWhitespace(decay_and_kv_mem_step)},
       {"SIMD_REDUCE_KV_MEM", absl::StripAsciiWhitespace(simd_reduce_kv_mem)},
       {"V_BROADCAST_DECLS", absl::StripAsciiWhitespace(v_broadcast_decls)},
       {"ATTN_OUT_INIT_DECLS", absl::StripAsciiWhitespace(attn_out_init_decls)},
       {"UPDATE_STATE_AND_ATTN_OUT_STEP",
        absl::StripAsciiWhitespace(update_state_and_attn_out_step)},
       {"SIMD_REDUCE_ATTN_OUT",
        absl::StripAsciiWhitespace(simd_reduce_attn_out)},
       {"STATE_WRITE_OUT_STEP",
        absl::StripAsciiWhitespace(state_write_out_step)},
       {"SHARED_REDUCE_KV_MEM", shared_reduce_kv_mem},
       {"SHARED_REDUCE_ATTN_OUT", shared_reduce_attn_out}},
      &code);

  absl::StrReplaceAll(
      {{"SEQ_LEN_EXPR", std::to_string(seq_len)},
       {"V_TILE", std::to_string(v_tile)},
       {"StateScalarType",
        ::ml_drift::ToUclDataType(rec_state_in.GetDataType(), 1)},
       {"StateType", ::ml_drift::ToUclDataType(rec_state_in.GetDataType(), 4)},
       {"ActivationScalarType",
        ::ml_drift::ToUclDataType(output.GetDataType(), 1)},
       {"ActivationType", ::ml_drift::ToUclDataType(output.GetDataType(), 4)},
       {"HEAD_K_DIM_SLICES", std::to_string(q_slices)},
       {"GQA_RATIO", std::to_string(gqa_ratio)}},
      &code);

  op->code_ = std::move(code);
  return op;
}

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
  return BuildGatedDeltaUpdateGpuGraph(op_def, GetTensorIds(inputs),
                                       GetTensorIds(outputs), attr->mode,
                                       gpu_info, model_builder);
}

}  // namespace litert::ml_drift
