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

#include "ml_drift_delegate/delegate/composite/ir/litert_op_selector.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/selectors/operation_selector.h"  // from @ml_drift
#include "ml_drift/common/selectors/special_selector.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_kernel.h"
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_kernel.h"
#include "ml_drift_delegate/delegate/composite/gated_delta_update_kernel.h"
#include "ml_drift_delegate/delegate/composite/glu_kernel.h"
#include "ml_drift_delegate/delegate/composite/moe_experts_kernel.h"
#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_kernel.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_kernel.h"
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.h"
#include "ml_drift_delegate/delegate/composite/short_conv_step_kernel.h"

namespace litert::ml_drift::ir {

LiteRtOpSelector::LiteRtOpSelector(
    const ::ml_drift::CreateGpuModelInfo* create_info,
    const ::ml_drift::GpuInfo* gpu_info)
    : create_info_(*create_info), gpu_info_(*gpu_info) {}

void LiteRtOpSelector::TensorToBuffer(
    const ::ml_drift::ir::IrTensor& tensor, bool copy_from_original,
    ::ml_drift::GpuModelBuilder* model_builder) {
  auto tensor_id = tensor.id;
  if (replaced_tensors_.contains(tensor_id)) {
    return;
  }
  auto handle = model_builder->GetTensor(tensor_id);
  if (!handle.ok()) {
    return;
  }
  if (handle->tensor_desc.GetStorageType() ==
      ::ml_drift::TensorStorageType::kBuffer) {
    return;
  }
  ::ml_drift::TensorDescriptor new_desc = handle->tensor_desc;
  new_desc.SetStorageType(::ml_drift::TensorStorageType::kBuffer);
  auto new_tensor = model_builder->AddTensor(new_desc);
  if (copy_from_original) {
    // Emit an explicit copy instead of re-pointing the producer of `tensor_id`
    // at `new_tensor`. The param tensor can be read by other consumers that
    // expect the original (non-BUFFER) storage type, e.g. the channel slices
    // that the ring-buffer SDPA path applies to it. Re-pointing the producer
    // would leave `tensor_id` without any producer, so those consumers would
    // read uninitialized memory.
    //
    // When the param tensor has no other consumer, `LinkNodes()` merges this
    // copy back into the producer, which yields the same graph as re-pointing
    // the producer would have.
    model_builder->Copy(*handle, new_tensor);
  }
  auto replaced = std::make_unique<::ml_drift::ir::IrTensor>(tensor);
  replaced->id = new_tensor.id;
  replaced->desc = new_desc;
  replaced_tensors_[tensor_id] = std::move(replaced);
}

std::vector<const ::ml_drift::ir::IrTensor*>
LiteRtOpSelector::ResolveInputsWithBufferParam(
    int param_index, const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (param_index >= 0) {
    // Ensure param tensor is a buffer tensor as kernel programs expect so.
    TensorToBuffer(*inputs[param_index], /*copy_from_original=*/true,
                   model_builder);
  }
  // Substitute any input tensor that was replaced in `replaced_tensors_`. In
  // addition to `inputs[param_index]`, upstream `add_values_to_cache` nodes may
  // have promoted their non-external KV cache output tensors to BUFFER storage,
  // which are consumed here as KV inputs even though the op's own input
  // descriptors were originally created with the default storage type.
  std::vector<const ::ml_drift::ir::IrTensor*> resolved_inputs = inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (auto it = replaced_tensors_.find(inputs[i]->id);
        it != replaced_tensors_.end()) {
      resolved_inputs[i] = it->second.get();
    }
  }
  return resolved_inputs;
}

absl::Status LiteRtOpSelector::GPUOperationFromNode(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<const ::ml_drift::ir::IrTensor*>& inputs,
    const std::vector<const ::ml_drift::ir::IrTensor*>& outputs,
    const ::ml_drift::ir::IrOp& op,
    ::ml_drift::GpuModelBuilder* model_builder) {
  if (op.name == "add_values_to_cache") {
    ::ml_drift::OperationDef cache_op_def = op_def;
    std::vector<const ::ml_drift::ir::IrTensor*> cache_inputs =
        ResolveInputsWithBufferParam(/*param_index=*/2, inputs, model_builder);
    std::vector<::ml_drift::ValueId> src_ids(cache_inputs.size());
    for (size_t i = 0; i < cache_inputs.size(); ++i) {
      src_ids[i] = cache_inputs[i]->id;
      cache_op_def.src_tensors[i] = cache_inputs[i]->desc;
    }
    std::vector<::ml_drift::ValueId> dst_ids(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      dst_ids[i] = outputs[i]->id;
      const bool is_external =
          create_info_.predefined.contains(outputs[i]->id) ||
          create_info_.external_immutable_tensors.contains(outputs[i]->id) ||
          create_info_.external_mutable_tensors.contains(outputs[i]->id);
      if (!is_external) {
        // Unlike input param tensors, `add_values_to_cache` is itself the
        // producer of `outputs[i]` and writes the packed KV cache directly into
        // the new BUFFER tensor (which downstream `sdpa_transposed` and
        // `runtime_bmm` consumers require), so `copy_from_original` is false.
        TensorToBuffer(*outputs[i], /*copy_from_original=*/false,
                       model_builder);
        if (auto it = replaced_tensors_.find(outputs[i]->id);
            it != replaced_tensors_.end()) {
          dst_ids[i] = it->second->id;
          cache_op_def.dst_tensors[i] = it->second->desc;
        }
      }
    }
    ABSL_ASSIGN_OR_RETURN(auto gpu_op,
                          CreateAddValuesToCacheFromNode(cache_op_def, op));
    model_builder->AddGpuOperation(src_ids, dst_ids, std::move(gpu_op),
                                   "add_values_to_cache");
    return absl::OkStatus();
  } else if (op.name == "fused_sdpa_cache_update") {
    int param_index = inputs.size() > 6 ? 6 : -1;
    std::vector<const ::ml_drift::ir::IrTensor*> sdpa_inputs =
        ResolveInputsWithBufferParam(param_index, inputs, model_builder);
    return CreateFusedSdpaCacheUpdateFromIrOp(sdpa_inputs, outputs, op,
                                              model_builder);
  } else if (op.name == "gated_delta_update") {
    return CreateGatedDeltaUpdateFromIrOp(op_def, inputs, outputs, op,
                                          &gpu_info_, model_builder);
  } else if (op.name == "moe_experts") {
    return CreateMoeExpertsFromIrOp(create_info_, inputs, outputs, op,
                                    model_builder);
  } else if (op.name == "runtime_batched_matmul") {
    int param_index =
        inputs.size() > 2 ? static_cast<int>(inputs.size()) - 1 : -1;
    std::vector<const ::ml_drift::ir::IrTensor*> bmm_inputs =
        ResolveInputsWithBufferParam(param_index, inputs, model_builder);
    return CreateRuntimeBatchedMatMulFromIrOp(bmm_inputs, outputs, op,
                                              model_builder);
  } else if (op.name == "sdpa_transposed") {
    int param_index = -1;
    if (inputs.size() == 4 &&
        inputs[3]->desc.GetDataType() == ::ml_drift::DataType::kInt32) {
      param_index = 3;
    } else if (inputs.size() > 4) {
      param_index = 4;
    }
    std::vector<const ::ml_drift::ir::IrTensor*> sdpa_inputs =
        ResolveInputsWithBufferParam(param_index, inputs, model_builder);
    return CreateSdpaTransposedFromIrOp(sdpa_inputs, outputs, op,
                                        model_builder);
  } else if (op.name == "swiglu") {
    return CreateGluFromIrOp(inputs, outputs, op, model_builder);
  } else if (op.name == "qkv_norm_rope") {
    return CreateQkvNormRopeFromIrOp(inputs, outputs, op, model_builder);
  } else if (op.name == "short_conv_step") {
    return CreateShortConvStepFromIrOp(inputs, outputs, op, model_builder);
  }

  return ::ml_drift::GPUOperationFromNode(gpu_info_, op_def, create_info_,
                                          inputs, outputs, op, model_builder);
}

absl::Status LiteRtOpSelector::GPUSubgraphFromIrModel(
    const ::ml_drift::ir::IrModel& ir_model, ::ml_drift::ir::IrOpId first_op_id,
    const absl::flat_hash_set<::ml_drift::ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<::ml_drift::ir::IrOpId>* new_consumed_ops,
    ::ml_drift::GpuModelBuilder* model_builder) {
  return ::ml_drift::GPUSubgraphFromIrModel(create_info_.hints, gpu_info_,
                                            ir_model, first_op_id, consumed_ops,
                                            new_consumed_ops, model_builder);
}

}  // namespace litert::ml_drift::ir
