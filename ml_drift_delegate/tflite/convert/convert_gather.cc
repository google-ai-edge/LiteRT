// Copyright 2026 The ML Drift Authors.
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

#include "ml_drift_delegate/tflite/convert/convert_gather.h"

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

void ConvertGather(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const int input_id = node.inputs->data[0];
  const int indices_id = node.inputs->data[1];
  const int output_id = node.outputs->data[0];
  const TfLiteTensor* input_tensor = context.tensors + input_id;
  const TfLiteTensor* indices_tensor = context.tensors + indices_id;

  const bool indices_are_const = ::tflite::IsConstantTensor(indices_tensor);
  const bool indices_are_1d = indices_tensor->dims->size == 1;

  ::ml_drift::ir::IrTensorId final_indices_id = tensor_map[indices_id];

  // Insert a RESHAPE if indices tensor [N] is mis-auto-expanded to [N,1,1,1].
  if (!indices_are_const && indices_are_1d) {
    ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
    reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);

    ::ml_drift::ReshapeAttributes reshape_attr;
    ::ml_drift::BHWC new_shape(1, 1, 1, indices_tensor->dims->data[0]);
    reshape_attr.new_shape = new_shape;
    reshape_op->attr = std::move(reshape_attr);

    ir_model.AddConsumer(tensor_map[indices_id], reshape_op->id);

    ::ml_drift::ir::IrTensor* new_indices_tensor = ir_model.add_tensor(
        ir_model.tensor(tensor_map[indices_id])->desc.GetDataType(), new_shape);
    final_indices_id = new_indices_tensor->id;

    ir_model.SetProducer(final_indices_id, reshape_op->id);
  }

  ::ml_drift::GatherAttributes attr;
  const auto* params =
      static_cast<const TfLiteGatherParams*>(node.builtin_data);
  attr.axis = ExtractAxisFromIndex(*input_tensor, params->axis);

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::GATHER);
  op->attr = std::move(attr);
  ir_model.AddConsumer(tensor_map[input_id], op->id);
  ir_model.AddConsumer(final_indices_id, op->id);
  ir_model.SetProducer(tensor_map[output_id], op->id);
}

}  // namespace litert::ml_drift::ir
