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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_unpack.h"

#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/convert/convert_aux.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertUnpack(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const TfLiteUnpackParams* unpack_params =
      reinterpret_cast<const TfLiteUnpackParams*>(node.builtin_data);
  const int axis = unpack_params->axis;
  const int num = unpack_params->num;

  const int input_tfl_id = node.inputs->data[0];
  const ::ml_drift::ir::IrTensorId input_ir_id = tensor_map[input_tfl_id];
  const TfLiteTensor* input_tensor = context.tensors + input_tfl_id;
  const ::ml_drift::Axis ml_drift_axis =
      ExtractAxisFromIndex(*input_tensor, axis);

  if (num == 1) {
    ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
    reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
    const int output_id = node.outputs->data[0];
    const ::ml_drift::ir::IrTensorId ir_output_id = tensor_map[output_id];

    ir_model.AddConsumer(input_ir_id, reshape_op->id);
    ir_model.SetProducer(ir_output_id, reshape_op->id);

    const TfLiteTensor* output_tensor = context.tensors + output_id;
    if (output_tensor->dims->size == 5) {
      ::ml_drift::Reshape3DAttributes attr;
      attr.new_shape = ir_model.tensor(ir_output_id)->desc.GetBHWDCShape();
      reshape_op->attr = std::move(attr);
    } else {
      ::ml_drift::ReshapeAttributes attr;
      attr.new_shape = ir_model.tensor(ir_output_id)->desc.GetBHWCShape();
      reshape_op->attr = std::move(attr);
    }
    return;
  }

  ::ml_drift::SplitAttributes split_attr;
  split_attr.axis = ml_drift_axis;

  ::ml_drift::ir::IrOp* split_op = ir_model.add_op();
  split_op->name = ToString(::ml_drift::OperationType::SPLIT);
  split_op->attr = split_attr;

  ir_model.AddConsumer(input_ir_id, split_op->id);

  const bool is_5d_input = input_tensor->dims->size == 5;
  ::ml_drift::BHWDC split_shape_5d;
  ::ml_drift::BHWC split_shape_4d;
  if (is_5d_input) {
    split_shape_5d = ir_model.tensor(input_ir_id)->desc.GetBHWDCShape();
    split_shape_5d.set(ml_drift_axis, 1);
  } else {
    split_shape_4d = ir_model.tensor(input_ir_id)->desc.GetBHWCShape();
    split_shape_4d.set(ml_drift_axis, 1);
  }

  for (int i = 0; i < node.outputs->size; ++i) {
    const int output_tfl_id = node.outputs->data[i];
    const ::ml_drift::ir::IrTensorId ir_output_id = tensor_map[output_tfl_id];

    ::ml_drift::ir::IrTensor* split_tensor;
    if (is_5d_input) {
      split_tensor = ir_model.add_tensor(
          ir_model.tensor(ir_output_id)->desc.GetDataType(), split_shape_5d);
    } else {
      split_tensor = ir_model.add_tensor(
          ir_model.tensor(ir_output_id)->desc.GetDataType(), split_shape_4d);
    }

    ir_model.SetProducer(split_tensor->id, split_op->id);

    ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
    reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);

    const TfLiteTensor* output_tensor = context.tensors + output_tfl_id;
    if (output_tensor->dims->size == 5) {
      ::ml_drift::Reshape3DAttributes attr;
      attr.new_shape = ir_model.tensor(ir_output_id)->desc.GetBHWDCShape();
      reshape_op->attr = std::move(attr);
    } else {
      ::ml_drift::ReshapeAttributes attr;
      attr.new_shape = ir_model.tensor(ir_output_id)->desc.GetBHWCShape();
      reshape_op->attr = std::move(attr);
    }

    ir_model.AddConsumer(split_tensor->id, reshape_op->id);
    ir_model.SetProducer(ir_output_id, reshape_op->id);
  }
}

}  // namespace litert::ml_drift::ir
