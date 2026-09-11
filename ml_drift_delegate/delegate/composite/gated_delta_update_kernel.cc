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
    // Grid: X=K Slices * Value Slices, Y=Height (Heads), Z=Batch
    return ::ml_drift::int3(k_slices_ * v_slices_, num_heads_, batch_size_);
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
  int k_slices_ = 4;
};

// SIMD shuffle-based shader (register-only butterfly reduction, zero shared
// memory, zero barriers).
constexpr char kGatedDeltaUpdateShuffleShader[] = R"(
MAIN_FUNCTION($0) {
  int k_slice = ucl::GetLocalId<0>();
  int v_slice = ucl::GetGroupId<0>();
  int h = ucl::GetGlobalId<1>();
  int b = ucl::GetGlobalId<2>();

  int seq_len = args.output.Width();

  // Each thread owns 4 rows along D_k (k_slice * 4 + {0, 1, 2, 3}) for this v_slice.
  // Held entirely in 4 registers. Zero register spilling.
  int k_base = k_slice * 4;
  Type s0 = args.recurrent_state_in.Read(k_base + 0, h, v_slice, b);
  Type s1 = args.recurrent_state_in.Read(k_base + 1, h, v_slice, b);
  Type s2 = args.recurrent_state_in.Read(k_base + 2, h, v_slice, b);
  Type s3 = args.recurrent_state_in.Read(k_base + 3, h, v_slice, b);

  // Process sequence length sequentially
  for (int t = 0; t < seq_len; ++t) {
    // Read beta and g for this step (mapped from TFLite 3D shape [B, H, L] to MLDrift BHWC [1, B, H, L])
    int t_slice = t / 4;
    int t_elem = t % 4;
    Type beta_t_vec = args.beta_t.Read(h, b, t_slice, 0);
    ScalarType beta_val = beta_t_vec.x;
    if (t_elem == 1) {
      beta_val = beta_t_vec.y;
    } else if (t_elem == 2) {
      beta_val = beta_t_vec.z;
    } else if (t_elem == 3) {
      beta_val = beta_t_vec.w;
    }

    Type g_t_vec = args.g_t.Read(h, b, t_slice, 0);
    ScalarType g_val = g_t_vec.x;
    if (t_elem == 1) {
      g_val = g_t_vec.y;
    } else if (t_elem == 2) {
      g_val = g_t_vec.z;
    } else if (t_elem == 3) {
      g_val = g_t_vec.w;
    }

    ScalarType decay_scalar = exp(g_val);
    Type decay_vec = ucl::Init<Type>(decay_scalar);

    // Apply decay to this thread's 4 rows
    s0 = s0 * decay_vec;
    s1 = s1 * decay_vec;
    s2 = s2 * decay_vec;
    s3 = s3 * decay_vec;

    // Load full vec4 K for this thread's k_slice directly
    Type k_val = args.k_t.Read(t, h, k_slice, b);

    // Compute this thread's partial dot product: sum_i (S[i] * k[i])
    Type kv_mem = s0 * ucl::Init<Type>(k_val.x) +
                  s1 * ucl::Init<Type>(k_val.y) +
                  s2 * ucl::Init<Type>(k_val.z) +
                  s3 * ucl::Init<Type>(k_val.w);

    // Register-only butterfly XOR shuffle reduction across lanes
    for (int offset = 1; offset < HEAD_K_DIM_SLICES; offset *= 2) {
      kv_mem += simd_shuffle_xor(kv_mem, offset);
    }

    // Load V vector slice for this step
    Type v_vec = args.v_t.Read(t, h, v_slice, b);
    Type beta_factor = ucl::Init<Type>(beta_val);
    Type delta_slice = (v_vec - kv_mem) * beta_factor;

    // Update recurrent state in-place with outer product: S += delta * k^T
    s0 = s0 + delta_slice * ucl::Init<Type>(k_val.x);
    s1 = s1 + delta_slice * ucl::Init<Type>(k_val.y);
    s2 = s2 + delta_slice * ucl::Init<Type>(k_val.z);
    s3 = s3 + delta_slice * ucl::Init<Type>(k_val.w);

    // Load full vec4 Q for this thread's k_slice directly
    Type q_val = args.q_t.Read(t, h, k_slice, b);

    // Compute attention output slice: sum_i (S[i] * q[i])
    Type my_attn_out = s0 * ucl::Init<Type>(q_val.x) +
                       s1 * ucl::Init<Type>(q_val.y) +
                       s2 * ucl::Init<Type>(q_val.z) +
                       s3 * ucl::Init<Type>(q_val.w);

    for (int offset = 1; offset < HEAD_K_DIM_SLICES; offset *= 2) {
      my_attn_out += simd_shuffle_xor(my_attn_out, offset);
    }
    if (k_slice == 0) {
      args.output.Write(my_attn_out, t, h, v_slice, b);
    }
  }

  // Write out final evolved recurrent state for this thread's 4 rows
  args.recurrent_state_out.Write(s0, k_base + 0, h, v_slice, b);
  args.recurrent_state_out.Write(s1, k_base + 1, h, v_slice, b);
  args.recurrent_state_out.Write(s2, k_base + 2, h, v_slice, b);
  args.recurrent_state_out.Write(s3, k_base + 3, h, v_slice, b);
}
)";

// Shared memory-based shader (tree reduction with barriers for OpenCL / WebGPU)
constexpr char kGatedDeltaUpdateSharedMemShader[] = R"(
MAIN_FUNCTION($0) {
  int k_slice = ucl::GetLocalId<0>();
  int v_slice = ucl::GetGroupId<0>();
  int h = ucl::GetGlobalId<1>();
  int b = ucl::GetGlobalId<2>();

  __local Type scratch[HEAD_K_DIM_SLICES];
  int seq_len = args.output.Width();

  // Each thread owns 4 rows along D_k (k_slice * 4 + {0, 1, 2, 3}) for this v_slice.
  // Held entirely in 4 registers. Zero register spilling.
  int k_base = k_slice * 4;
  Type s0 = args.recurrent_state_in.Read(k_base + 0, h, v_slice, b);
  Type s1 = args.recurrent_state_in.Read(k_base + 1, h, v_slice, b);
  Type s2 = args.recurrent_state_in.Read(k_base + 2, h, v_slice, b);
  Type s3 = args.recurrent_state_in.Read(k_base + 3, h, v_slice, b);

  // Process sequence length sequentially
  for (int t = 0; t < seq_len; ++t) {
    // Read beta and g for this step (mapped from TFLite 3D shape [B, H, L] to MLDrift BHWC [1, B, H, L])
    int t_slice = t / 4;
    int t_elem = t % 4;
    Type beta_t_vec = args.beta_t.Read(h, b, t_slice, 0);
    ScalarType beta_val = beta_t_vec.x;
    if (t_elem == 1) {
      beta_val = beta_t_vec.y;
    } else if (t_elem == 2) {
      beta_val = beta_t_vec.z;
    } else if (t_elem == 3) {
      beta_val = beta_t_vec.w;
    }

    Type g_t_vec = args.g_t.Read(h, b, t_slice, 0);
    ScalarType g_val = g_t_vec.x;
    if (t_elem == 1) {
      g_val = g_t_vec.y;
    } else if (t_elem == 2) {
      g_val = g_t_vec.z;
    } else if (t_elem == 3) {
      g_val = g_t_vec.w;
    }

    ScalarType decay_scalar = exp(g_val);
    Type decay_vec = ucl::Init<Type>(decay_scalar);

    // Apply decay to this thread's 4 rows
    s0 = s0 * decay_vec;
    s1 = s1 * decay_vec;
    s2 = s2 * decay_vec;
    s3 = s3 * decay_vec;

    // Load full vec4 K for this thread's k_slice directly
    Type k_val = args.k_t.Read(t, h, k_slice, b);

    // Compute this thread's partial dot product: sum_i (S[i] * k[i])
    Type kv_mem = s0 * ucl::Init<Type>(k_val.x) +
                  s1 * ucl::Init<Type>(k_val.y) +
                  s2 * ucl::Init<Type>(k_val.z) +
                  s3 * ucl::Init<Type>(k_val.w);

    // Shared memory tree reduction across workgroup
    scratch[k_slice] = kv_mem;
    ucl::SyncThreads<WorkGroup, Local>();
    for (int stride = HEAD_K_DIM_SLICES / 2; stride > 0; stride /= 2) {
      if (k_slice < stride) {
        scratch[k_slice] += scratch[k_slice + stride];
      }
      ucl::SyncThreads<WorkGroup, Local>();
    }
    kv_mem = scratch[0];
    ucl::SyncThreads<WorkGroup, Local>();

    // Load V vector slice for this step
    Type v_vec = args.v_t.Read(t, h, v_slice, b);
    Type beta_factor = ucl::Init<Type>(beta_val);
    Type delta_slice = (v_vec - kv_mem) * beta_factor;

    // Update recurrent state in-place with outer product: S += delta * k^T
    s0 = s0 + delta_slice * ucl::Init<Type>(k_val.x);
    s1 = s1 + delta_slice * ucl::Init<Type>(k_val.y);
    s2 = s2 + delta_slice * ucl::Init<Type>(k_val.z);
    s3 = s3 + delta_slice * ucl::Init<Type>(k_val.w);

    // Load full vec4 Q for this thread's k_slice directly
    Type q_val = args.q_t.Read(t, h, k_slice, b);

    // Compute attention output slice: sum_i (S[i] * q[i])
    Type my_attn_out = s0 * ucl::Init<Type>(q_val.x) +
                       s1 * ucl::Init<Type>(q_val.y) +
                       s2 * ucl::Init<Type>(q_val.z) +
                       s3 * ucl::Init<Type>(q_val.w);

    // Shared memory tree reduction across workgroup
    scratch[k_slice] = my_attn_out;
    ucl::SyncThreads<WorkGroup, Local>();
    for (int stride = HEAD_K_DIM_SLICES / 2; stride > 0; stride /= 2) {
      if (k_slice < stride) {
        scratch[k_slice] += scratch[k_slice + stride];
      }
      ucl::SyncThreads<WorkGroup, Local>();
    }
    if (k_slice == 0) {
      args.output.Write(scratch[0], t, h, v_slice, b);
    }
    ucl::SyncThreads<WorkGroup, Local>();
  }

  // Write out final evolved recurrent state for this thread's 4 rows
  args.recurrent_state_out.Write(s0, k_base + 0, h, v_slice, b);
  args.recurrent_state_out.Write(s1, k_base + 1, h, v_slice, b);
  args.recurrent_state_out.Write(s2, k_base + 2, h, v_slice, b);
  args.recurrent_state_out.Write(s3, k_base + 3, h, v_slice, b);
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

  int B = q_shape.b;
  int H = q_shape.h;
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
  bool can_use_metal_shuffle =
      gpu_info && gpu_info->IsApiMetal() && is_power_of_two && (q_slices <= 32);

  std::string code = can_use_metal_shuffle ? kGatedDeltaUpdateShuffleShader
                                           : kGatedDeltaUpdateSharedMemShader;

  absl::StrReplaceAll(
      {{"ScalarType", ::ml_drift::ToUclDataType(output.GetDataType(), 1)},
       {"Type", ::ml_drift::ToUclDataType(output.GetDataType(), 4)},
       {"HEAD_K_DIM_SLICES", std::to_string(q_slices)}},
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
