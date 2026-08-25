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

#include "ml_drift_delegate/delegate/shared_memory_manager/gf32_graph_adapter.h"

#include <cstdint>
#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift

namespace ml_drift {

BHWC GraphFloat32Adapter::GetValueShape(uint32_t value_id) const {
  return graph_.GetValue(value_id)->tensor.shape;
}

void GraphFloat32Adapter::SetValueType(uint32_t value_id, DataType type) {
  graph_.GetValue(value_id)->tensor.type = type;
}

void GraphFloat32Adapter::SetValueShapeAndType(uint32_t value_id,
                                               const BHWC& shape,
                                               DataType type) {
  Value* value = graph_.GetValue(value_id);
  value->tensor.shape = shape;
  value->tensor.type = type;
}

std::vector<uint32_t> GraphFloat32Adapter::FindConsumerOps(
    uint32_t value_id) const {
  std::vector<Node*> consumers = graph_.FindConsumers(value_id);
  std::vector<uint32_t> op_ids;
  op_ids.reserve(consumers.size());
  for (const Node* node : consumers) {
    op_ids.push_back(node->id);
  }
  return op_ids;
}

std::string GraphFloat32Adapter::GetOpTypeName(uint32_t op_id) const {
  return graph_.GetNode(op_id)->operation.type;
}

bool GraphFloat32Adapter::OpHasInputs(uint32_t op_id) const {
  return !graph_.FindInputs(op_id).empty();
}

BHWC GraphFloat32Adapter::GetOpFirstInputShape(uint32_t op_id) const {
  return graph_.FindInputs(op_id)[0]->tensor.shape;
}

DataType GraphFloat32Adapter::GetOpFirstInputType(uint32_t op_id) const {
  return graph_.FindInputs(op_id)[0]->tensor.type;
}

uint32_t GraphFloat32Adapter::AddConstantInput(uint32_t global_tensor_id,
                                               const BHWC& shape, DataType type,
                                               uint32_t consumer_op_id) {
  Value* value = graph_.NewValue();
  value->tensor.ref = global_tensor_id;
  value->tensor.type = type;
  value->tensor.shape = shape;
  value->tensor.is_variable_input = false;
  graph_.AddConsumer(consumer_op_id, value->id);
  return value->id;
}

}  // namespace ml_drift
