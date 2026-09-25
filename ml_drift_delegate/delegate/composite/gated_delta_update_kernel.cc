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
  GatedDeltaUpdateOp() = default;
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

  int batch_size_ = 1;
  int num_heads_ = 1;
  int v_slices_ = 1;
  int v_groups_ = 1;
  int k_slices_ = 4;
};

// SIMD shuffle-based shader (register-only 2D (D_k x D_v) warp decomposition,
// zero shared memory, zero barriers). Maps the subgroup of HEAD_K_DIM_SLICES
// lanes as a 2D grid of (K_THREADS x V_TILE):
//   - v_lane = tid / K_THREADS (0..V_TILE-1) assigns each sub-warp its own
//     independent v_slice = v_base + v_lane along D_v.
//   - k_lane = tid % K_THREADS (0..K_THREADS-1) partitions D_k across
//     K_THREADS lanes (each owning V_TILE k_slices = 4 * V_TILE rows of D_k).
// All 32 lanes coalesce-load k_t and q_t (1 vec4 per lane) and exchange slices
// via simd_shuffle; beta_t, g_t, and v_t are read directly without branching or
// broadcast; and kv_mem / my_attn_out reduce across K_THREADS lanes in
// log2(K_THREADS) butterfly simd_shuffle_xor steps for all V_TILE v_slices
// simultaneously.
constexpr char kGatedDeltaUpdateShuffleShader[] = R"(
SUBGROUP_HEADER
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
    // Uniform read of beta and g across the subgroup (no lane-0 branch or broadcast)
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

    bool is_active = (beta_val != 0.0 || g_val != 0.0);
    INACTIVE_EARLY_CONTINUE
    StateType k_loaded = ucl::Init<StateType>(0.0);
    StateType kv_mem = ucl::Init<StateType>(0.0);

    if (is_active) {
      StateScalarType decay_scalar = 1.0;
      if (g_val <= 0.0) {
        decay_scalar = exp(ucl::Convert<StateScalarType>(g_val));
      }
      StateType decay_vec = ucl::Init<StateType>(decay_scalar);

      // Coalesced load of full D_k vector across all lanes (1 vec4 per lane)
      k_loaded = ucl::Convert<StateType>(args.k_t.Read(t, h_k, tid, b));

      DECAY_AND_KV_MEM_STEP
    }

    SIMD_REDUCE_KV_MEM

    StateType my_attn_out = ucl::Init<StateType>(0.0);
    if (is_active) {
      // Direct v_t load per v_lane sub-warp (no branch or broadcast)
      StateType v_vec = ucl::Convert<StateType>(args.v_t.Read(t, h, v_slice, b));

      StateScalarType beta_clean = 0.0;
      if (beta_val > 0.0) {
        beta_clean = ucl::Convert<StateScalarType>(beta_val);
      }
      StateType beta_factor = ucl::Init<StateType>(beta_clean);
      StateType delta_slice = (v_vec - kv_mem) * beta_factor;

      // Coalesced load of full D_k query vector across all lanes (1 vec4 per lane)
      StateType q_loaded = ucl::Convert<StateType>(args.q_t.Read(t, h_k, tid, b));

      UPDATE_STATE_AND_ATTN_OUT_STEP
    }

    SIMD_REDUCE_ATTN_OUT
  }

  // Write out final evolved recurrent state for this thread's 4 * V_TILE rows of v_slice
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

    StateScalarType decay_scalar = 1.0;
    if (g_val <= 0.0) {
      decay_scalar = exp(ucl::Convert<StateScalarType>(g_val));
    }
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
    StateScalarType kq_dot = k_val.x * q_val.x + k_val.y * q_val.y +
                             k_val.z * q_val.z + k_val.w * q_val.w;

    SHARED_REDUCE_KV_MEM

    // Load V vector slice for this step
    StateType v_vec = ucl::Convert<StateType>(args.v_t.Read(t, h, v_slice, b));
    StateScalarType beta_clean = 0.0;
    if (beta_val > 0.0) {
      beta_clean = ucl::Convert<StateScalarType>(beta_val);
    }
    StateType beta_factor = ucl::Init<StateType>(beta_clean);
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
  auto op = std::make_unique<GatedDeltaUpdateOp>();

  const auto& q_t = definition.src_tensors[0];
  const auto& k_t = definition.src_tensors[1];
  const auto& v_t = definition.src_tensors[2];
  const auto& beta_t = definition.src_tensors[3];
  const auto& g_t = definition.src_tensors[4];
  const auto& rec_state_in = definition.src_tensors[5];

  const auto& output = definition.dst_tensors[0];
  const auto& rec_state_out = definition.dst_tensors[1];

  op->AddSrcTensor("q_t", q_t);
  op->AddSrcTensor("k_t", k_t);
  op->AddSrcTensor("v_t", v_t);
  op->AddSrcTensor("beta_t", beta_t);
  op->AddSrcTensor("g_t", g_t);
  op->AddSrcTensor("recurrent_state_in", rec_state_in);

  op->AddDstTensor("output", output);
  op->AddDstTensor("recurrent_state_out", rec_state_out);

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

  op->batch_size_ = B;
  op->num_heads_ = H;
  op->v_slices_ = D_v / 4;
  op->k_slices_ = q_slices;

  bool is_power_of_two = (q_slices > 0) && ((q_slices & (q_slices - 1)) == 0);
  bool is_metal_shuffle =
      gpu_info && gpu_info->IsApiMetal() && is_power_of_two && (q_slices <= 32);
  bool is_webgpu_subgroup = gpu_info && gpu_info->IsApiWebGpu() &&
                            gpu_info->SupportsSubGroupWithSize(32) &&
                            is_power_of_two && (q_slices <= 32);
  bool can_use_shuffle = is_metal_shuffle || is_webgpu_subgroup;

  int v_tile = 1;
  if (can_use_shuffle) {
    if (op->v_slices_ % 4 == 0 && q_slices >= 4) {
      v_tile = 4;
    } else if (op->v_slices_ % 2 == 0 && q_slices >= 2) {
      v_tile = 2;
    }
  }
  int k_threads = q_slices / v_tile;
  op->v_groups_ = op->v_slices_ / v_tile;

  std::string code = can_use_shuffle ? kGatedDeltaUpdateShuffleShader
                                     : kGatedDeltaUpdateSharedMemShader;
  std::string subgroup_header = is_webgpu_subgroup ? "enable subgroups;\n" : "";

  std::string state_init_decls;
  std::string decay_and_kv_mem_step;
  std::string update_state_and_attn_out_step;
  std::string state_write_out_step;

  for (int m = 0; m < v_tile; ++m) {
    std::string ms = std::to_string(m);
    std::string shuffle_k_expr =
        is_webgpu_subgroup
            ? "subgroupShuffle(k_loaded, u32(k_lane * V_TILE + " + ms + "))"
            : "simd_shuffle(k_loaded, k_lane * V_TILE + " + ms + ")";
    std::string shuffle_q_expr =
        is_webgpu_subgroup
            ? "subgroupShuffle(q_loaded, u32(k_lane * V_TILE + " + ms + "))"
            : "simd_shuffle(q_loaded, k_lane * V_TILE + " + ms + ")";

    state_init_decls +=
        "  int k_base_" + ms + " = (k_lane * V_TILE + " + ms + ") * 4;\n" +
        "  StateType s0_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base_" + ms +
        " + 0, h, v_slice, b));\n" + "  StateType s1_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base_" + ms +
        " + 1, h, v_slice, b));\n" + "  StateType s2_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base_" + ms +
        " + 2, h, v_slice, b));\n" + "  StateType s3_" + ms +
        " = ucl::Convert<StateType>(args.recurrent_state_in.Read(k_base_" + ms +
        " + 3, h, v_slice, b));\n";

    decay_and_kv_mem_step +=
        "    StateType k_val_" + ms + " = " + shuffle_k_expr + ";\n" +
        "    s0_" + ms + " = s0_" + ms + " * decay_vec;\n" + "    s1_" + ms +
        " = s1_" + ms + " * decay_vec;\n" + "    s2_" + ms + " = s2_" + ms +
        " * decay_vec;\n" + "    s3_" + ms + " = s3_" + ms + " * decay_vec;\n" +
        "    kv_mem += s0_" + ms + " * ucl::Init<StateType>(k_val_" + ms +
        ".x) +\n" + "              s1_" + ms +
        " * ucl::Init<StateType>(k_val_" + ms + ".y) +\n" +
        "              s2_" + ms + " * ucl::Init<StateType>(k_val_" + ms +
        ".z) +\n" + "              s3_" + ms +
        " * ucl::Init<StateType>(k_val_" + ms + ".w);\n";

    update_state_and_attn_out_step +=
        "    StateType k_val_" + ms + " = " + shuffle_k_expr + ";\n" +
        "    StateType q_val_" + ms + " = " + shuffle_q_expr + ";\n" +
        "    s0_" + ms + " = s0_" + ms +
        " + delta_slice * ucl::Init<StateType>(k_val_" + ms + ".x);\n" +
        "    s1_" + ms + " = s1_" + ms +
        " + delta_slice * ucl::Init<StateType>(k_val_" + ms + ".y);\n" +
        "    s2_" + ms + " = s2_" + ms +
        " + delta_slice * ucl::Init<StateType>(k_val_" + ms + ".z);\n" +
        "    s3_" + ms + " = s3_" + ms +
        " + delta_slice * ucl::Init<StateType>(k_val_" + ms + ".w);\n" +
        "    my_attn_out += s0_" + ms + " * ucl::Init<StateType>(q_val_" + ms +
        ".x) +\n" + "                   s1_" + ms +
        " * ucl::Init<StateType>(q_val_" + ms + ".y) +\n" +
        "                   s2_" + ms + " * ucl::Init<StateType>(q_val_" + ms +
        ".z) +\n" + "                   s3_" + ms +
        " * ucl::Init<StateType>(q_val_" + ms + ".w);\n";

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

  std::string simd_reduce_kv_mem =
      is_webgpu_subgroup
          ? "    for (uint offset = 1u; offset < u32(K_THREADS); offset *= "
            "2u) {\n"
            "      kv_mem += subgroupShuffleXor(kv_mem, offset);\n"
            "    }\n"
          : "    for (int offset = 1; offset < K_THREADS; offset *= 2) {\n"
            "      kv_mem += simd_shuffle_xor(kv_mem, offset);\n"
            "    }\n";

  std::string simd_reduce_attn_out =
      (is_webgpu_subgroup
           ? std::string(
                 "    for (uint offset = 1u; offset < u32(K_THREADS); offset "
                 "*= 2u) {\n"
                 "      my_attn_out += subgroupShuffleXor(my_attn_out, "
                 "offset);\n"
                 "    }\n")
           : std::string(
                 "    for (int offset = 1; offset < K_THREADS; offset *= 2) "
                 "{\n"
                 "      my_attn_out += simd_shuffle_xor(my_attn_out, offset);\n"
                 "    }\n")) +
      "    if (k_lane == 0) {\n"
      "      args.output.Write(ucl::Convert<ActivationType>(my_attn_out), t, "
      "h, v_slice, b);\n"
      "    }\n";

  std::string shared_reduce_kv_mem = (q_slices == 32)
                                         ? R"(scratch_kv[k_slice] = kv_mem;
    scratch_out[k_slice] = sq_mem;
    scratch_kq[k_slice] = kq_dot;
    ucl::SyncThreads<WorkGroup, Local>();
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

  std::string inactive_early_continue =
      is_webgpu_subgroup ? "" : R"(if (!is_active) {
      if (k_lane == 0) {
        args.output.Write(ucl::Init<ActivationType>(0.0), t, h, v_slice, b);
      }
      continue;
    })";

  absl::StrReplaceAll(
      {{"SUBGROUP_HEADER", subgroup_header},
       {"INACTIVE_EARLY_CONTINUE", inactive_early_continue},
       {"STATE_INIT_DECLS", state_init_decls},
       {"DECAY_AND_KV_MEM_STEP", decay_and_kv_mem_step},
       {"SIMD_REDUCE_KV_MEM", simd_reduce_kv_mem},
       {"UPDATE_STATE_AND_ATTN_OUT_STEP", update_state_and_attn_out_step},
       {"SIMD_REDUCE_ATTN_OUT", simd_reduce_attn_out},
       {"STATE_WRITE_OUT_STEP", state_write_out_step},
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
