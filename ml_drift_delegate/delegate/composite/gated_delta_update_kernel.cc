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

// 2D (D_k x D_v) subgroup-parallel shader (register-only reduction, zero
// shared memory, zero barriers). Decomposes each 32-lane subgroup into V_TILE
// groups of K_THREADS = HEAD_K_DIM_SLICES / V_TILE lanes (e.g., 8 x 4 for
// D_k=128, V_TILE=4):
//   - k_lane = tid % K_THREADS (0..7) processes K_ROUNDS = HEAD_K_DIM_SLICES /
//     K_THREADS (4) strided rounds of 4 rows along D_k (k_slice = k_lane +
//     m * K_THREADS).
//   - v_lane = tid / K_THREADS (0..3) owns 1 v_slice (4 columns along D_v).
// Each thread holds 16 float4 recurrent state registers (4 rounds x 1 v_slice),
// loads beta_t and g_t uniformly, loads its own v_slice directly without
// divergent branches or broadcasts, coalesces k_t/q_t global loads across all
// 32 lanes (distributing via simd_shuffle(loaded, k_lane + m * K_THREADS)),
// and reduces a single float4 kv_mem and my_attn_out across K_THREADS lanes via
// log2(K_THREADS) = 3 simd_shuffle_xor butterfly steps.
//
// Final rendered shader example (Metal, D_k=128, D_v=128, V_TILE=4,
// K_THREADS=8, SEQ_LEN_EXPR=128, GQA_RATIO=3, StateType=float4,
// ActivationType=float4):
// NOLINTBEGIN(whitespace/line_length)
// ```msl
// MAIN_FUNCTION($0) {
//   int tid = ucl::GetLocalId<0>();
//   int k_lane = tid % 8;
//   int v_lane = tid / 8;
//   int v_slice = ucl::GetGroupId<0>() * 4 + v_lane;
//   int h = ucl::GetGlobalId<1>();
//   int h_k = h / 3;
//   int b = ucl::GetGlobalId<2>();
//
//   int k_base_0 = k_lane * 4;
//   float4 s0_0 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_0 + 0, h, v_slice, b));
//   float4 s1_0 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_0 + 1, h, v_slice, b));
//   float4 s2_0 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_0 + 2, h, v_slice, b));
//   float4 s3_0 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_0 + 3, h, v_slice, b));
//   int k_base_1 = (k_lane + 1 * 8) * 4;
//   float4 s0_1 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_1 + 0, h, v_slice, b));
//   float4 s1_1 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_1 + 1, h, v_slice, b));
//   float4 s2_1 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_1 + 2, h, v_slice, b));
//   float4 s3_1 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_1 + 3, h, v_slice, b));
//   int k_base_2 = (k_lane + 2 * 8) * 4;
//   float4 s0_2 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_2 + 0, h, v_slice, b));
//   float4 s1_2 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_2 + 1, h, v_slice, b));
//   float4 s2_2 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_2 + 2, h, v_slice, b));
//   float4 s3_2 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_2 + 3, h, v_slice, b));
//   int k_base_3 = (k_lane + 3 * 8) * 4;
//   float4 s0_3 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_3 + 0, h, v_slice, b));
//   float4 s1_3 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_3 + 1, h, v_slice, b));
//   float4 s2_3 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_3 + 2, h, v_slice, b));
//   float4 s3_3 = ucl::Convert<float4>(args.recurrent_state_in.Read(k_base_3 + 3, h, v_slice, b));
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
//     float4 k_loaded = ucl::Convert<float4>(args.k_t.Read(t, h_k, tid, b));
//
//     float4 kv_mem = ucl::Init<float4>(0.0);
//     {
//       float4 k_val_0 = simd_shuffle(k_loaded, k_lane);
//       kv_mem += s0_0 * ucl::Init<float4>(k_val_0.x) +
//                 s1_0 * ucl::Init<float4>(k_val_0.y) +
//                 s2_0 * ucl::Init<float4>(k_val_0.z) +
//                 s3_0 * ucl::Init<float4>(k_val_0.w);
//     }
//     {
//       float4 k_val_1 = simd_shuffle(k_loaded, (k_lane + 1 * 8));
//       kv_mem += s0_1 * ucl::Init<float4>(k_val_1.x) +
//                 s1_1 * ucl::Init<float4>(k_val_1.y) +
//                 s2_1 * ucl::Init<float4>(k_val_1.z) +
//                 s3_1 * ucl::Init<float4>(k_val_1.w);
//     }
//     {
//       float4 k_val_2 = simd_shuffle(k_loaded, (k_lane + 2 * 8));
//       kv_mem += s0_2 * ucl::Init<float4>(k_val_2.x) +
//                 s1_2 * ucl::Init<float4>(k_val_2.y) +
//                 s2_2 * ucl::Init<float4>(k_val_2.z) +
//                 s3_2 * ucl::Init<float4>(k_val_2.w);
//     }
//     {
//       float4 k_val_3 = simd_shuffle(k_loaded, (k_lane + 3 * 8));
//       kv_mem += s0_3 * ucl::Init<float4>(k_val_3.x) +
//                 s1_3 * ucl::Init<float4>(k_val_3.y) +
//                 s2_3 * ucl::Init<float4>(k_val_3.z) +
//                 s3_3 * ucl::Init<float4>(k_val_3.w);
//     }
//
//     // Butterfly reduction across K_THREADS lanes sharing the same v_lane
//     kv_mem += simd_shuffle_xor(kv_mem, 1);
//     kv_mem += simd_shuffle_xor(kv_mem, 2);
//     kv_mem += simd_shuffle_xor(kv_mem, 4);
//
//     // Each v_lane group loads its own v_slice directly (no branch or broadcast)
//     float4 v_vec = ucl::Convert<float4>(args.v_t.Read(t, h, v_slice, b));
//
//     float4 beta_factor = ucl::Init<float4>(ucl::Convert<float>(beta_val));
//     float4 delta_slice = (v_vec - kv_mem) * beta_factor;
//
//     {
//       float4 k_val_0 = simd_shuffle(k_loaded, k_lane);
//       s0_0 = s0_0 + delta_slice * ucl::Init<float4>(k_val_0.x);
//       s1_0 = s1_0 + delta_slice * ucl::Init<float4>(k_val_0.y);
//       s2_0 = s2_0 + delta_slice * ucl::Init<float4>(k_val_0.z);
//       s3_0 = s3_0 + delta_slice * ucl::Init<float4>(k_val_0.w);
//     }
//     {
//       float4 k_val_1 = simd_shuffle(k_loaded, (k_lane + 1 * 8));
//       s0_1 = s0_1 + delta_slice * ucl::Init<float4>(k_val_1.x);
//       s1_1 = s1_1 + delta_slice * ucl::Init<float4>(k_val_1.y);
//       s2_1 = s2_1 + delta_slice * ucl::Init<float4>(k_val_1.z);
//       s3_1 = s3_1 + delta_slice * ucl::Init<float4>(k_val_1.w);
//     }
//     {
//       float4 k_val_2 = simd_shuffle(k_loaded, (k_lane + 2 * 8));
//       s0_2 = s0_2 + delta_slice * ucl::Init<float4>(k_val_2.x);
//       s1_2 = s1_2 + delta_slice * ucl::Init<float4>(k_val_2.y);
//       s2_2 = s2_2 + delta_slice * ucl::Init<float4>(k_val_2.z);
//       s3_2 = s3_2 + delta_slice * ucl::Init<float4>(k_val_2.w);
//     }
//     {
//       float4 k_val_3 = simd_shuffle(k_loaded, (k_lane + 3 * 8));
//       s0_3 = s0_3 + delta_slice * ucl::Init<float4>(k_val_3.x);
//       s1_3 = s1_3 + delta_slice * ucl::Init<float4>(k_val_3.y);
//       s2_3 = s2_3 + delta_slice * ucl::Init<float4>(k_val_3.z);
//       s3_3 = s3_3 + delta_slice * ucl::Init<float4>(k_val_3.w);
//     }
//
//     // Coalesced load of full D_k Q across 32 lanes (1 vec4 per lane)
//     float4 q_loaded = ucl::Convert<float4>(args.q_t.Read(t, h_k, tid, b));
//
//     float4 my_attn_out = ucl::Init<float4>(0.0);
//     {
//       float4 q_val_0 = simd_shuffle(q_loaded, k_lane);
//       my_attn_out += s0_0 * ucl::Init<float4>(q_val_0.x) +
//                      s1_0 * ucl::Init<float4>(q_val_0.y) +
//                      s2_0 * ucl::Init<float4>(q_val_0.z) +
//                      s3_0 * ucl::Init<float4>(q_val_0.w);
//     }
//     {
//       float4 q_val_1 = simd_shuffle(q_loaded, (k_lane + 1 * 8));
//       my_attn_out += s0_1 * ucl::Init<float4>(q_val_1.x) +
//                      s1_1 * ucl::Init<float4>(q_val_1.y) +
//                      s2_1 * ucl::Init<float4>(q_val_1.z) +
//                      s3_1 * ucl::Init<float4>(q_val_1.w);
//     }
//     {
//       float4 q_val_2 = simd_shuffle(q_loaded, (k_lane + 2 * 8));
//       my_attn_out += s0_2 * ucl::Init<float4>(q_val_2.x) +
//                      s1_2 * ucl::Init<float4>(q_val_2.y) +
//                      s2_2 * ucl::Init<float4>(q_val_2.z) +
//                      s3_2 * ucl::Init<float4>(q_val_2.w);
//     }
//     {
//       float4 q_val_3 = simd_shuffle(q_loaded, (k_lane + 3 * 8));
//       my_attn_out += s0_3 * ucl::Init<float4>(q_val_3.x) +
//                      s1_3 * ucl::Init<float4>(q_val_3.y) +
//                      s2_3 * ucl::Init<float4>(q_val_3.z) +
//                      s3_3 * ucl::Init<float4>(q_val_3.w);
//     }
//
//     // Butterfly reduction across K_THREADS lanes sharing the same v_lane
//     my_attn_out += simd_shuffle_xor(my_attn_out, 1);
//     my_attn_out += simd_shuffle_xor(my_attn_out, 2);
//     my_attn_out += simd_shuffle_xor(my_attn_out, 4);
//
//     if (k_lane == 0) {
//       args.output.Write(ucl::Convert<float4>(my_attn_out), t, h, v_slice, b);
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
// NOLINTEND(whitespace/line_length)
constexpr char kGatedDeltaUpdateShuffleShader[] = R"(
MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int k_lane = tid % K_THREADS;
  int v_lane = tid / K_THREADS;
  int v_slice = ucl::GetGroupId<0>() * V_TILE + v_lane;
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

    DECAY_STEP

    // Coalesced load of full D_k across 32 lanes (1 vec4 per lane)
    StateType k_loaded = ucl::Convert<StateType>(args.k_t.Read(t, h_k, tid, b));

    StateType kv_mem = ucl::Init<StateType>(0.0);
    KV_MEM_STEP

    // Butterfly reduction across K_THREADS lanes sharing the same v_lane
    SIMD_REDUCE_KV_MEM

    // Each v_lane group loads its own v_slice directly (no branch or broadcast)
    StateType v_vec = ucl::Convert<StateType>(args.v_t.Read(t, h, v_slice, b));

    StateType beta_factor = ucl::Init<StateType>(ucl::Convert<StateScalarType>(beta_val));
    StateType delta_slice = (v_vec - kv_mem) * beta_factor;

    UPDATE_STATE_STEP

    // Coalesced load of full D_k Q across 32 lanes (1 vec4 per lane)
    StateType q_loaded = ucl::Convert<StateType>(args.q_t.Read(t, h_k, tid, b));

    StateType my_attn_out = ucl::Init<StateType>(0.0);
    ATTN_OUT_STEP

    // Butterfly reduction across K_THREADS lanes sharing the same v_lane
    SIMD_REDUCE_ATTN_OUT

    if (k_lane == 0) {
      args.output.Write(ucl::Convert<ActivationType>(my_attn_out), t, h, v_slice, b);
    }
  }

  // Write out final evolved recurrent state for this thread's owned rows and v_slice
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
  int k_threads = q_slices / v_tile;
  int k_rounds = q_slices / k_threads;

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
  std::string decay_step;
  std::string kv_mem_step;
  std::string simd_reduce_kv_mem;
  std::string update_state_step;
  std::string attn_out_step;
  std::string simd_reduce_attn_out;
  std::string state_write_out_step;

  for (int m = 0; m < k_rounds; ++m) {
    std::string ms = std::to_string(m);
    std::string k_slice_expr =
        (m == 0) ? "k_lane" : "(k_lane + " + ms + " * K_THREADS)";
    state_init_decls +=
        "  int k_base_" + ms + " = " + k_slice_expr + " * 4;\n" +
        "  StateType s0_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base_" + ms +
        " + 0, h, v_slice, b));\n" + "  StateType s1_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base_" + ms +
        " + 1, h, v_slice, b));\n" + "  StateType s2_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base_" + ms +
        " + 2, h, v_slice, b));\n" + "  StateType s3_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base_" + ms +
        " + 3, h, v_slice, b));\n";

    decay_step += "    s0_" + ms + " = s0_" + ms + " * decay_vec;\n" +
                  "    s1_" + ms + " = s1_" + ms + " * decay_vec;\n" +
                  "    s2_" + ms + " = s2_" + ms + " * decay_vec;\n" +
                  "    s3_" + ms + " = s3_" + ms + " * decay_vec;\n";

    kv_mem_step += "    {\n"
                   "      StateType k_val_" +
                   ms + " = simd_shuffle(k_loaded, " + k_slice_expr +
                   ");\n"
                   "      kv_mem += s0_" +
                   ms + " * ucl::Init<StateType>(k_val_" + ms + ".x) +\n" +
                   "                s1_" + ms +
                   " * ucl::Init<StateType>(k_val_" + ms + ".y) +\n" +
                   "                s2_" + ms +
                   " * ucl::Init<StateType>(k_val_" + ms + ".z) +\n" +
                   "                s3_" + ms +
                   " * ucl::Init<StateType>(k_val_" + ms +
                   ".w);\n"
                   "    }\n";

    update_state_step += "    {\n"
                         "      StateType k_val_" +
                         ms + " = simd_shuffle(k_loaded, " + k_slice_expr +
                         ");\n"
                         "      s0_" +
                         ms + " = s0_" + ms +
                         " + delta_slice * ucl::Init<StateType>(k_val_" + ms +
                         ".x);\n" + "      s1_" + ms + " = s1_" + ms +
                         " + delta_slice * ucl::Init<StateType>(k_val_" + ms +
                         ".y);\n" + "      s2_" + ms + " = s2_" + ms +
                         " + delta_slice * ucl::Init<StateType>(k_val_" + ms +
                         ".z);\n" + "      s3_" + ms + " = s3_" + ms +
                         " + delta_slice * ucl::Init<StateType>(k_val_" + ms +
                         ".w);\n"
                         "    }\n";

    attn_out_step += "    {\n"
                     "      StateType q_val_" +
                     ms + " = simd_shuffle(q_loaded, " + k_slice_expr +
                     ");\n"
                     "      my_attn_out += s0_" +
                     ms + " * ucl::Init<StateType>(q_val_" + ms + ".x) +\n" +
                     "                     s1_" + ms +
                     " * ucl::Init<StateType>(q_val_" + ms + ".y) +\n" +
                     "                     s2_" + ms +
                     " * ucl::Init<StateType>(q_val_" + ms + ".z) +\n" +
                     "                     s3_" + ms +
                     " * ucl::Init<StateType>(q_val_" + ms +
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
        "    kv_mem += simd_shuffle_xor(kv_mem, " + os + ");\n";
    simd_reduce_attn_out +=
        "    my_attn_out += simd_shuffle_xor(my_attn_out, " + os + ");\n";
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

  absl::StrReplaceAll(
      {{"  STATE_INIT_DECLS\n", state_init_decls},
       {"    DECAY_STEP\n", decay_step},
       {"    KV_MEM_STEP\n", kv_mem_step},
       {"    SIMD_REDUCE_KV_MEM\n", simd_reduce_kv_mem},
       {"    UPDATE_STATE_STEP\n", update_state_step},
       {"    ATTN_OUT_STEP\n", attn_out_step},
       {"    SIMD_REDUCE_ATTN_OUT\n", simd_reduce_attn_out},
       {"  STATE_WRITE_OUT_STEP\n", state_write_out_step},
       {"SHARED_REDUCE_KV_MEM", shared_reduce_kv_mem},
       {"SHARED_REDUCE_ATTN_OUT", shared_reduce_attn_out}},
      &code);

  absl::StrReplaceAll(
      {{"SEQ_LEN_EXPR", std::to_string(seq_len)},
       {"V_TILE", std::to_string(v_tile)},
       {"K_THREADS", std::to_string(k_threads)},
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
