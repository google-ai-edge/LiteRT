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

#include "third_party/odml/litert/ml_drift/delegate/composite/litert_op_selector.h"

#include <set>
#include <utility>
#include <vector>

#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/selectors/operation_selector.h"  // from @ml_drift
#include "ml_drift/common/selectors/special_selector.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/delegate/composite/add_values_to_cache_kernel.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/add_values_to_cache_parser.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/moe_experts_kernel.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/moe_experts_parser.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/runtime_batched_matmul_kernel.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/runtime_batched_matmul_parser.h"

namespace litert::ml_drift {

LiteRtOpSelector::LiteRtOpSelector(
    const ::ml_drift::CreateGpuModelInfo* create_info,
    const ::ml_drift::GpuInfo* gpu_info)
    : create_info_(*create_info), gpu_info_(*gpu_info) {}

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
    std::vector<::ml_drift::ValueId> dst_ids(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
      dst_ids[i] = outputs[i]->id;
    }
    model_builder->AddGpuOperation(src_ids, dst_ids, std::move(op),
                                   node.operation.type);
    return absl::OkStatus();
  }
  if (node.operation.type == kRuntimeBatchedMatMulType) {
    return CreateRuntimeBatchedMatMulFromNode(inputs, outputs, node,
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
