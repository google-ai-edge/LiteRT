// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/tflite/convert/convert_clamp.h"

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertClamp(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  // TFLite StablehloClamp inputs: 0: min, 1: operand, 2: max
  const int min_id = node.inputs->data[0];
  const int operand_id = node.inputs->data[1];
  const int max_id = node.inputs->data[2];
  const int output_id = node.outputs->data[0];

  // Create MAXIMUM op: max(operand, min)
  ::ml_drift::ir::IrOp* max_op = ir_model.add_op();
  max_op->name = ToString(::ml_drift::OperationType::MAXIMUM);
  ir_model.AddConsumer(tensor_map.at(operand_id), max_op->id);
  ir_model.AddConsumer(tensor_map.at(min_id), max_op->id);

  // Create intermediate tensor
  const auto& operand_tensor = *ir_model.tensor(tensor_map.at(operand_id));
  ::ml_drift::ir::IrTensor* interim_tensor =
      ir_model.add_tensor(operand_tensor.desc);
  ir_model.SetProducer(interim_tensor->id, max_op->id);

  // Create MINIMUM op: min(interim, max)
  ::ml_drift::ir::IrOp* min_op = ir_model.add_op();
  min_op->name = ToString(::ml_drift::OperationType::MINIMUM);
  ir_model.AddConsumer(interim_tensor->id, min_op->id);
  ir_model.AddConsumer(tensor_map.at(max_id), min_op->id);

  ir_model.SetProducer(tensor_map.at(output_id), min_op->id);
}

}  // namespace litert::ml_drift::ir
