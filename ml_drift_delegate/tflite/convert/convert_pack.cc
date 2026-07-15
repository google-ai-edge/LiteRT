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

#include "ml_drift_delegate/tflite/convert/convert_pack.h"

#include <utility>
#include <vector>

#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertPack(const TfLiteContext& context, const TfLiteNode& node,
                 const TfLiteRegistration& registration,
                 ::ml_drift::ir::TensorMap& tensor_map,
                 ::ml_drift::ir::IrModel& ir_model) {
  const TfLiteTensor* output_tensor = context.tensors + node.outputs->data[0];
  const ::ml_drift::ir::IrTensorId ir_output_id =
      tensor_map[node.outputs->data[0]];

  if (node.inputs->size == 1) {
    ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
    reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
    const int input_id = node.inputs->data[0];
    ir_model.AddConsumer(tensor_map[input_id], reshape_op->id);
    ir_model.SetProducer(ir_output_id, reshape_op->id);

    ::ml_drift::ReshapeAttributes attr;
    attr.new_shape = ir_model.tensor(ir_output_id)->desc.GetBHWCShape();
    reshape_op->attr = std::move(attr);
    return;
  }

  const TfLitePackParams* pack_params =
      reinterpret_cast<const TfLitePackParams*>(node.builtin_data);
  const int axis = pack_params->axis;
  const ::ml_drift::BHWC output_shape =
      ir_model.tensor(ir_output_id)->desc.GetBHWCShape();
  const ::ml_drift::Axis ml_drift_axis =
      ExtractAxisFromIndex(*output_tensor, axis);

  std::vector<::ml_drift::ir::IrTensorId> reshaped_inputs;
  for (int i = 0; i < node.inputs->size; ++i) {
    const int input_tfl_id = node.inputs->data[i];
    const ::ml_drift::ir::IrTensorId input_ir_id = tensor_map[input_tfl_id];

    ::ml_drift::ir::IrOp* reshape_op = ir_model.add_op();
    reshape_op->name = ToString(::ml_drift::OperationType::RESHAPE);
    ::ml_drift::ReshapeAttributes attr;
    attr.new_shape = output_shape;
    attr.new_shape.set(ml_drift_axis, 1);
    reshape_op->attr = attr;
    ir_model.AddConsumer(input_ir_id, reshape_op->id);

    ::ml_drift::ir::IrTensor* reshaped_tensor = ir_model.add_tensor(
        ir_model.tensor(input_ir_id)->desc.GetDataType(), attr.new_shape);
    ir_model.SetProducer(reshaped_tensor->id, reshape_op->id);
    reshaped_inputs.push_back(reshaped_tensor->id);
  }

  ::ml_drift::ir::IrOp* concat_op = ir_model.add_op();
  concat_op->name = ToString(::ml_drift::OperationType::CONCAT);
  ::ml_drift::ConcatAttributes concat_attr;
  concat_attr.axis = ml_drift_axis;
  concat_op->attr = std::move(concat_attr);
  for (auto input_id : reshaped_inputs) {
    ir_model.AddConsumer(input_id, concat_op->id);
  }

  ir_model.SetProducer(ir_output_id, concat_op->id);
}

}  // namespace litert::ml_drift::ir
