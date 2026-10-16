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

#include "ml_drift_delegate/delegate/composite/litert_op_selector.h"

#include <any>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/selectors/operation_selector.h"  // from @ml_drift
#include "ml_drift/common/selectors/special_selector.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_kernel.h"
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_kernel.h"
#include "ml_drift_delegate/delegate/composite/fused_sdpa_cache_update_parser.h"
#include "ml_drift_delegate/delegate/composite/gated_delta_update_kernel.h"
#include "ml_drift_delegate/delegate/composite/gated_delta_update_parser.h"
#include "ml_drift_delegate/delegate/composite/glu_kernel.h"
#include "ml_drift_delegate/delegate/composite/glu_parser.h"
#include "ml_drift_delegate/delegate/composite/moe_experts_kernel.h"
#include "ml_drift_delegate/delegate/composite/moe_experts_parser.h"
#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_kernel.h"
#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_parser.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_kernel.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.h"
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"
#include "ml_drift_delegate/delegate/composite/short_conv_step_kernel.h"
#include "ml_drift_delegate/delegate/composite/short_conv_step_parser.h"

namespace litert::ml_drift {

namespace {

absl::Status CreateRoPEFromNode(
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  ::ml_drift::RoPEAttributes attr;
  if (node.operation.attributes.has_value()) {
    if (auto* rope_attr = std::any_cast<::ml_drift::RoPEAttributes>(
            &node.operation.attributes)) {
      attr = *rope_attr;
    }
  }
  if (inputs.size() == 2) {
    if (outputs.size() != 1) {
      return absl::InvalidArgumentError("RoPE expects 1 output for 2 inputs.");
    }
    ABSL_ASSIGN_OR_RETURN(auto src_handle,
                          model_builder->GetTensor(inputs[0]->id));
    ABSL_ASSIGN_OR_RETURN(auto position_handle,
                          model_builder->GetTensor(inputs[1]->id));
    auto dst_handle =
        model_builder->SplitRoPEConcat(src_handle, position_handle, attr);
    ABSL_RETURN_IF_ERROR(
        model_builder->UpdateOutputTensor(dst_handle, outputs[0]->id));
    return absl::OkStatus();
  } else if (inputs.size() == 3) {
    if (outputs.size() != 2) {
      return absl::InvalidArgumentError("RoPE expects 2 outputs for 3 inputs.");
    }
    ABSL_ASSIGN_OR_RETURN(auto src_l_handle,
                          model_builder->GetTensor(inputs[0]->id));
    ABSL_ASSIGN_OR_RETURN(auto src_r_handle,
                          model_builder->GetTensor(inputs[1]->id));
    ABSL_ASSIGN_OR_RETURN(auto position_handle,
                          model_builder->GetTensor(inputs[2]->id));
    auto dst_handles =
        model_builder->RoPE(src_l_handle, src_r_handle, position_handle, attr);
    ABSL_RETURN_IF_ERROR(model_builder->UpdateOutputTensors(
        dst_handles, {outputs[0]->id, outputs[1]->id}));
    return absl::OkStatus();
  } else {
    return absl::InvalidArgumentError("RoPE expects 2 or 3 inputs.");
  }
}

}  // namespace

LiteRtOpSelector::LiteRtOpSelector(
    const ::ml_drift::CreateGpuModelInfo* create_info,
    const ::ml_drift::GpuInfo* gpu_info)
    : create_info_(*create_info), gpu_info_(*gpu_info) {}

void LiteRtOpSelector::TensorToBuffer(
    const ::ml_drift::Value& value, bool copy_from_original,
    ::ml_drift::GpuModelBuilder* model_builder) {
  auto tensor_id = value.id;
  if (replaced_tensors_.contains(tensor_id)) {
    return;
  }
  auto tensor = model_builder->GetTensor(tensor_id);
  if (!tensor.ok()) {
    return;
  }
  if (tensor->tensor_desc.GetStorageType() ==
      ::ml_drift::TensorStorageType::kBuffer) {
    return;
  }
  ::ml_drift::TensorDescriptor new_desc = tensor->tensor_desc;
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
    model_builder->Copy(*tensor, new_tensor);
  }
  replaced_tensors_[tensor_id] = std::make_unique<::ml_drift::Value>(
      ::ml_drift::Value{new_tensor.id, value.tensor, value.quant_params});
}

std::vector<::ml_drift::Value*> LiteRtOpSelector::ResolveInputsWithBufferParam(
    int param_index, const std::vector<::ml_drift::Value*>& inputs,
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
  std::vector<::ml_drift::Value*> resolved_inputs = inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    if (auto it = replaced_tensors_.find(inputs[i]->id);
        it != replaced_tensors_.end()) {
      resolved_inputs[i] = it->second.get();
    }
  }
  return resolved_inputs;
}

absl::Status LiteRtOpSelector::GPUOperationFromNode(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  if (node.operation.type == kAddValuesToCacheType) {
    ::ml_drift::OperationDef cache_op_def = op_def;
    std::vector<::ml_drift::ValueId> src_ids(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      src_ids[i] = inputs[i]->id;
    }
    int param_index = 2;
    // Ensure param tensor is a buffer tensor as kernel programs expect so.
    TensorToBuffer(*inputs[param_index], /*copy_from_original=*/true,
                   model_builder);
    if (auto it = replaced_tensors_.find(inputs[param_index]->id);
        it != replaced_tensors_.end()) {
      src_ids[param_index] = it->second->id;
      ABSL_ASSIGN_OR_RETURN(auto param_handle,
                            model_builder->GetTensor(src_ids[param_index]));
      cache_op_def.src_tensors[param_index] = param_handle.tensor_desc;
    }
    std::vector<::ml_drift::ValueId> dst_ids(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
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
          ABSL_ASSIGN_OR_RETURN(auto dst_handle,
                                model_builder->GetTensor(dst_ids[i]));
          cache_op_def.dst_tensors[i] = dst_handle.tensor_desc;
        }
      }
    }
    ABSL_ASSIGN_OR_RETURN(auto op,
                          CreateAddValuesToCacheFromNode(cache_op_def, node));
    model_builder->AddGpuOperation(src_ids, dst_ids, std::move(op),
                                   node.operation.type);
    return absl::OkStatus();
  }
  if (node.operation.type == kGatedDeltaUpdateType) {
    return CreateGatedDeltaUpdateFromNode(op_def, inputs, outputs, node,
                                          &gpu_info_, model_builder);
  }
  if (node.operation.type == kFusedSdpaCacheUpdateType) {
    int param_index = inputs.size() > 6 ? 6 : -1;
    std::vector<::ml_drift::Value*> sdpa_inputs =
        ResolveInputsWithBufferParam(param_index, inputs, model_builder);
    return CreateFusedSdpaCacheUpdateFromNode(sdpa_inputs, outputs, node,
                                              model_builder);
  }
  if (node.operation.type == kRuntimeBatchedMatMulType) {
    int param_index =
        inputs.size() > 2 ? static_cast<int>(inputs.size()) - 1 : -1;
    std::vector<::ml_drift::Value*> bmm_inputs =
        ResolveInputsWithBufferParam(param_index, inputs, model_builder);
    return CreateRuntimeBatchedMatMulFromNode(bmm_inputs, outputs, node,
                                              model_builder);
  }
  if (node.operation.type == kSdpaTransposedType) {
    int param_index = -1;
    if (inputs.size() == 4 &&
        inputs[3]->tensor.type == ::ml_drift::DataType::kInt32) {
      param_index = 3;
    } else if (inputs.size() > 4) {
      param_index = 4;
    }
    std::vector<::ml_drift::Value*> sdpa_inputs =
        ResolveInputsWithBufferParam(param_index, inputs, model_builder);
    return CreateSdpaTransposedFromNode(sdpa_inputs, outputs, node,
                                        model_builder);
  }
  if (node.operation.type == kMoeExpertsType) {
    return CreateMoeExpertsFromNode(create_info_, inputs, outputs, node,
                                    model_builder);
  }
  if (node.operation.type == kSwigluType) {
    return CreateGluFromNode(inputs, outputs, node, model_builder);
  }
  if (node.operation.type == kQkvNormRopeType) {
    return CreateQkvNormRopeFromNode(inputs, outputs, node, model_builder);
  }
  if (node.operation.type == kShortConvStepType) {
    return CreateShortConvStepFromNode(inputs, outputs, node, model_builder);
  }
  if (node.operation.type == ToString(::ml_drift::OperationType::kRope)) {
    return CreateRoPEFromNode(inputs, outputs, node, model_builder);
  }
  return ::ml_drift::GPUOperationFromNode(gpu_info_, op_def, create_info_,
                                          inputs, outputs, node, model_builder);
}

absl::Status LiteRtOpSelector::GPUSubgraphFromGraph(
    const ::ml_drift::GraphFloat32& graph, ::ml_drift::NodeId first_node_id,
    const std::set<::ml_drift::NodeId>& consumed_nodes,
    std::set<::ml_drift::NodeId>* new_consumed_nodes,
    ::ml_drift::GpuModelBuilder* model_builder) {
  return ::ml_drift::GPUSubgraphFromGraph(create_info_.hints, gpu_info_, graph,
                                          first_node_id, consumed_nodes,
                                          new_consumed_nodes, model_builder);
}

}  // namespace litert::ml_drift
