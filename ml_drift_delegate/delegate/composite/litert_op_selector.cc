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

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/selectors/operation_selector.h"  // from @ml_drift
#include "ml_drift/common/selectors/special_selector.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_kernel.h"
#include "ml_drift_delegate/delegate/composite/add_values_to_cache_parser.h"
#include "ml_drift_delegate/delegate/composite/moe_experts_kernel.h"
#include "ml_drift_delegate/delegate/composite/moe_experts_parser.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_kernel.h"
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"

namespace litert::ml_drift {

LiteRtOpSelector::LiteRtOpSelector(
    const ::ml_drift::CreateGpuModelInfo* create_info,
    const ::ml_drift::GpuInfo* gpu_info)
    : create_info_(*create_info), gpu_info_(*gpu_info) {}

void LiteRtOpSelector::ParamTensorToBuffer(
    int param_index, const std::vector<::ml_drift::Value*>& inputs,
    ::ml_drift::GpuModelBuilder* model_builder) {
  auto param_id = inputs[param_index]->id;
  if (replaced_tensors_.contains(param_id)) {
    return;
  }
  auto param_tensor_handle_or = model_builder->GetTensor(param_id);
  if (!param_tensor_handle_or.ok()) {
    return;
  }
  auto param_tensor = param_tensor_handle_or.value();
  if (param_tensor.tensor_desc.GetStorageType() ==
      ::ml_drift::TensorStorageType::BUFFER) {
    return;
  }
  ::ml_drift::TensorDescriptor new_desc = param_tensor.tensor_desc;
  new_desc.SetStorageType(::ml_drift::TensorStorageType::BUFFER);
  auto new_param_tensor = model_builder->AddTensor(new_desc);
  if (!model_builder->UpdateOutputTensor(param_tensor, new_param_tensor.id)
           .ok()) {
    return;
  }
  replaced_tensors_[param_id] = std::make_unique<::ml_drift::Value>(
      ::ml_drift::Value{new_param_tensor.id, inputs[param_index]->tensor,
                        inputs[param_index]->quant_params});
}

absl::Status LiteRtOpSelector::GPUOperationFromNode(
    const ::ml_drift::OperationDef& op_def,
    const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    const ::ml_drift::Node& node, ::ml_drift::GpuModelBuilder* model_builder) {
  if (node.operation.type == kAddValuesToCacheType) {
    ASSIGN_OR_RETURN(auto op, CreateAddValuesToCacheFromNode(op_def, node));
    std::vector<::ml_drift::ValueId> src_ids(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      src_ids[i] = inputs[i]->id;
    }
    int param_index = 2;
    // Ensure param tensor is a buffer tensor as kernel programs expect so.
    ParamTensorToBuffer(param_index, inputs, model_builder);
    if (replaced_tensors_.contains(inputs[param_index]->id)) {
      src_ids[param_index] = replaced_tensors_[inputs[param_index]->id]->id;
    }
    std::vector<::ml_drift::ValueId> dst_ids(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
      dst_ids[i] = outputs[i]->id;
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
      ParamTensorToBuffer(param_index, inputs, model_builder);
      if (replaced_tensors_.contains(inputs[param_index]->id)) {
        bmm_inputs[param_index] =
            replaced_tensors_[inputs[param_index]->id].get();
      }
    }
    return CreateRuntimeBatchedMatMulFromNode(bmm_inputs, outputs, node,
                                              model_builder);
  }
  if (node.operation.type == kMoeExpertsType) {
    return CreateMoeExpertsFromNode(create_info_, inputs, outputs, node,
                                    model_builder);
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
