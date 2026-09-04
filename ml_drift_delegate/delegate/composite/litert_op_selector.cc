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
#include "ml_drift_delegate/delegate/composite/moe_experts_kernel.h"
#include "ml_drift_delegate/delegate/composite/moe_experts_parser.h"
#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_kernel.h"
#include "ml_drift_delegate/delegate/composite/qkv_norm_rope_parser.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_kernel.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_kernel.h"
#include "ml_drift_delegate/delegate/composite/sdpa_transposed_parser.h"
#include "ml_drift_delegate/delegate/composite/swiglu_kernel.h"
#include "ml_drift_delegate/delegate/composite/swiglu_parser.h"

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

void LiteRtOpSelector::EnsureTensorIsBuffer(
    int tensor_index, const std::vector<::ml_drift::Value*>& values,
    ::ml_drift::GpuModelBuilder* model_builder) {
  auto tensor_id = values[tensor_index]->id;
  if (replaced_tensors_.contains(tensor_id)) {
    return;
  }
  auto tensor_handle_or = model_builder->GetTensor(tensor_id);
  if (!tensor_handle_or.ok()) {
    return;
  }
  auto tensor = tensor_handle_or.value();
  if (tensor.tensor_desc.GetStorageType() ==
      ::ml_drift::TensorStorageType::BUFFER) {
    return;
  }
  ::ml_drift::TensorDescriptor new_desc = tensor.tensor_desc;
  new_desc.SetStorageType(::ml_drift::TensorStorageType::BUFFER);
  auto new_tensor = model_builder->AddTensor(new_desc);
  if (!model_builder->UpdateOutputTensor(tensor, new_tensor.id).ok()) {
    model_builder->Copy(tensor, new_tensor);
  }
  replaced_tensors_[tensor_id] = std::make_unique<::ml_drift::Value>(
      ::ml_drift::Value{new_tensor.id, values[tensor_index]->tensor,
                        values[tensor_index]->quant_params});
}

absl::Status LiteRtOpSelector::GPUOperationFromNode(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  if (node.operation.type == kAddValuesToCacheType) {
    ::ml_drift::OperationDef custom_op_def = op_def;
    for (int i = 0; i < custom_op_def.dst_tensors.size(); ++i) {
      custom_op_def.dst_tensors[i].SetStorageType(
          ::ml_drift::TensorStorageType::BUFFER);
    }
    ABSL_ASSIGN_OR_RETURN(auto op,
                          CreateAddValuesToCacheFromNode(custom_op_def, node));
    std::vector<::ml_drift::ValueId> src_ids(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      src_ids[i] = inputs[i]->id;
    }
    int param_index = 2;
    // Ensure param tensor is a buffer tensor as kernel programs expect so.
    EnsureTensorIsBuffer(param_index, inputs, model_builder);
    if (replaced_tensors_.contains(inputs[param_index]->id)) {
      src_ids[param_index] = replaced_tensors_[inputs[param_index]->id]->id;
    }
    std::vector<::ml_drift::ValueId> dst_ids(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
      EnsureTensorIsBuffer(i, outputs, model_builder);
      if (replaced_tensors_.contains(outputs[i]->id)) {
        dst_ids[i] = replaced_tensors_[outputs[i]->id]->id;
      } else {
        dst_ids[i] = outputs[i]->id;
      }
    }
    model_builder->AddGpuOperation(src_ids, dst_ids, std::move(op),
                                   node.operation.type);
    return absl::OkStatus();
  }
  if (node.operation.type == kRuntimeBatchedMatMulType) {
    std::vector<::ml_drift::Value*> bmm_inputs = inputs;
    if (inputs.size() > 2) {
      int param_index = inputs.size() - 1;
      // Ensure param tensor is a buffer tensor as kernel programs expect so.
      EnsureTensorIsBuffer(param_index, inputs, model_builder);
    }
    for (int i = 0; i < inputs.size(); ++i) {
      if (replaced_tensors_.contains(inputs[i]->id)) {
        bmm_inputs[i] = replaced_tensors_[inputs[i]->id].get();
      }
    }
    return CreateRuntimeBatchedMatMulFromNode(bmm_inputs, outputs, node,
                                              model_builder);
  }
  if (node.operation.type == kSdpaTransposedType) {
    std::vector<::ml_drift::Value*> sdpa_inputs = inputs;
    if (inputs.size() > 4) {
      int param_index = 4;
      // Ensure param tensor is a buffer tensor as kernel programs expect so.
      EnsureTensorIsBuffer(param_index, inputs, model_builder);
    }
    for (int i = 0; i < inputs.size(); ++i) {
      if (replaced_tensors_.contains(inputs[i]->id)) {
        sdpa_inputs[i] = replaced_tensors_[inputs[i]->id].get();
      }
    }
    return CreateSdpaTransposedFromNode(sdpa_inputs, outputs, node,
                                        model_builder);
  }
  if (node.operation.type == kMoeExpertsType) {
    return CreateMoeExpertsFromNode(create_info_, inputs, outputs, node,
                                    model_builder);
  }
  if (node.operation.type == kSwigluType) {
    return CreateSwigluFromNode(inputs, outputs, node, model_builder);
  }
  if (node.operation.type == kQkvNormRopeType) {
    return CreateQkvNormRopeFromNode(inputs, outputs, node, model_builder);
  }
  if (node.operation.type == ToString(::ml_drift::OperationType::ROPE)) {
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
