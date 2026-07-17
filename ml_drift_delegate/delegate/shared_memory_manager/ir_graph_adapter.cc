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

#include "ml_drift_delegate/delegate/shared_memory_manager/ir_graph_adapter.h"

#include <cstdint>
#include <string>
#include <vector>

#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace ml_drift {

BHWC IrModelAdapter::GetValueShape(uint32_t value_id) const {
  return graph_.tensor(value_id)->desc.GetBHWCShape();
}

void IrModelAdapter::SetValueType(uint32_t value_id, DataType type) {
  graph_.GetMutableTensor(value_id)->desc.SetDataType(type);
}

void IrModelAdapter::SetValueShapeAndType(uint32_t value_id, const BHWC& shape,
                                          DataType type) {
  // Mutate the descriptor in place, preserving storage type / layout and
  // keeping the tensor id stable (matching GraphFloat32's in-place mutation).
  ir::IrTensor* tensor = graph_.GetMutableTensor(value_id);
  tensor->desc.SetBHWCShape(shape);
  tensor->desc.SetDataType(type);
}

std::vector<uint32_t> IrModelAdapter::FindConsumerOps(uint32_t value_id) const {
  std::vector<ir::IrOp*> consumers = graph_.FindConsumers(value_id);
  std::vector<uint32_t> op_ids;
  op_ids.reserve(consumers.size());
  for (const ir::IrOp* op : consumers) {
    op_ids.push_back(static_cast<uint32_t>(op->id));
  }
  return op_ids;
}

std::string IrModelAdapter::GetOpTypeName(uint32_t op_id) const {
  return graph_.op(op_id)->name;
}

bool IrModelAdapter::OpHasInputs(uint32_t op_id) const {
  return !graph_.op(op_id)->inputs.empty();
}

BHWC IrModelAdapter::GetOpFirstInputShape(uint32_t op_id) const {
  const ir::IrTensorId input_id = graph_.op(op_id)->inputs[0];
  return graph_.tensor(input_id)->desc.GetBHWCShape();
}

DataType IrModelAdapter::GetOpFirstInputType(uint32_t op_id) const {
  const ir::IrTensorId input_id = graph_.op(op_id)->inputs[0];
  return graph_.tensor(input_id)->desc.GetDataType();
}

uint32_t IrModelAdapter::AddConstantInput(uint32_t /*global_tensor_id*/,
                                          const BHWC& shape, DataType type,
                                          uint32_t consumer_op_id) {
  // ir::IrModel does not track a global tensor reference on the value, so
  // `global_tensor_id` is intentionally unused here.
  ir::IrTensor* value = graph_.add_tensor(ml_drift::TensorDescriptor{});
  value->desc.SetDataType(type);
  value->desc.SetBHWCShape(shape);
  graph_.AddConsumer(value->id, consumer_op_id);
  return static_cast<uint32_t>(value->id);
}

}  // namespace ml_drift
