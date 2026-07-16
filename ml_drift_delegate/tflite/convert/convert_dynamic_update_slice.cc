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

#include "ml_drift_delegate/tflite/convert/convert_dynamic_update_slice.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {
namespace {

::ml_drift::ir::IrTensorId AddReshapeOp(::ml_drift::ir::IrModel& ir_model,
                                        ::ml_drift::ir::IrTensorId input_id,
                                        const ::ml_drift::BHWC& new_shape) {
  auto* reshape_op = ir_model.add_op();
  reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
  ::ml_drift::ReshapeAttributes attr;
  attr.new_shape = new_shape;
  reshape_op->attr = attr;

  auto* in_tensor = ir_model.tensor(input_id);
  auto* out_tensor =
      ir_model.add_tensor(in_tensor->desc.GetDataType(), new_shape);

  ir_model.AddConsumer(input_id, reshape_op->id);
  ir_model.SetProducer(out_tensor->id, reshape_op->id);

  return out_tensor->id;
}

::ml_drift::ir::IrTensorId GetRightAlignedInput(
    const TfLiteContext& context, int tfl_input_id,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const TfLiteTensor* tfl_tensor = &context.tensors[tfl_input_id];
  ::ml_drift::ir::IrTensorId current_id;

  if (tflite::IsConstantTensor(tfl_tensor)) {
    current_id = AddConstInput(context, tfl_input_id, ir_model, {})->id;
  } else {
    current_id = tensor_map[tfl_input_id];
  }

  std::vector<int32_t> dims(tfl_tensor->dims->data,
                            tfl_tensor->dims->data + tfl_tensor->dims->size);
  ::ml_drift::BHWC right_aligned_shape = GetRightAlignedBHWC(dims, 1);

  if (ir_model.tensor(current_id)->desc.GetBHWCShape() != right_aligned_shape) {
    current_id = AddReshapeOp(ir_model, current_id, right_aligned_shape);
  }

  return current_id;
}

}  // namespace

void ConvertDynamicUpdateSlice(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrTensorId operand_id =
      GetRightAlignedInput(context, node.inputs->data[0], tensor_map, ir_model);

  ::ml_drift::ir::IrTensorId update_id =
      GetRightAlignedInput(context, node.inputs->data[1], tensor_map, ir_model);

  const int start_indices_tfl_id = node.inputs->data[2];
  ::ml_drift::ir::IrTensorId start_indices_id;
  if (tflite::IsConstantTensor(&context.tensors[start_indices_tfl_id])) {
    start_indices_id =
        AddConstInput(context, start_indices_tfl_id, ir_model, {})->id;
  } else {
    start_indices_id = tensor_map[start_indices_tfl_id];
  }

  ::ml_drift::ir::IrOp* ir_op = ir_model.add_op();
  ir_op->name = ToString(::ml_drift::OperationType::DYNAMIC_UPDATE_SLICE);
  ir_model.AddConsumer(operand_id, ir_op->id);
  ir_model.AddConsumer(update_id, ir_op->id);
  ir_model.AddConsumer(start_indices_id, ir_op->id);

  const int output_tfl_id = node.outputs->data[0];
  const TfLiteTensor* output_tensor = &context.tensors[output_tfl_id];
  std::vector<int32_t> out_dims(
      output_tensor->dims->data,
      output_tensor->dims->data + output_tensor->dims->size);
  ::ml_drift::BHWC right_aligned_out_shape = GetRightAlignedBHWC(out_dims, 1);

  ::ml_drift::ir::IrTensorId original_output_id = tensor_map[output_tfl_id];
  const ::ml_drift::BHWC original_out_shape =
      ir_model.tensor(original_output_id)->desc.GetBHWCShape();

  if (original_out_shape != right_aligned_out_shape) {
    ::ml_drift::ir::IrTensor* right_out_tensor = ir_model.add_tensor(
        ir_model.tensor(original_output_id)->desc.GetDataType(),
        right_aligned_out_shape);
    ir_model.SetProducer(right_out_tensor->id, ir_op->id);

    // Reshape right-aligned output back to the original shape
    ::ml_drift::ir::IrOp* output_reshape_op = ir_model.add_op();
    output_reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
    ::ml_drift::ReshapeAttributes out_attr;
    out_attr.new_shape = original_out_shape;
    output_reshape_op->attr = out_attr;

    ir_model.AddConsumer(right_out_tensor->id, output_reshape_op->id);
    ir_model.SetProducer(original_output_id, output_reshape_op->id);
  } else {
    // 4D case, no reshape needed
    ir_model.SetProducer(original_output_id, ir_op->id);
  }
}

}  // namespace litert::ml_drift::ir
