// Copyright 2025 Google LLC.
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

#include "ml_drift_delegate/tflite/convert/convert_reshape.h"

#include <utility>

#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertReshape(const TfLiteContext& context, const TfLiteNode& node,
                    const TfLiteRegistration& registration,
                    ::ml_drift::ir::TensorMap& tensor_map,
                    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* ir_op = ir_model.add_op();
  ir_op->name = ToString(::ml_drift::OperationType::RESHAPE);
  const int input_id = node.inputs->data[0];
  ir_model.AddConsumer(tensor_map[input_id], ir_op->id);
  const int output_id = node.outputs->data[0];
  ir_model.SetProducer(tensor_map[output_id], ir_op->id);

  const TfLiteTensor& output_tensor = context.tensors[node.outputs->data[0]];
  if (output_tensor.dims->size == 5) {
    ::ml_drift::Reshape3DAttributes attr3d;
    attr3d.new_shape =
        ir_model.tensor(tensor_map[output_id])->desc.GetBHWDCShape();
    ir_op->attr = std::move(attr3d);
  } else {
    ::ml_drift::ReshapeAttributes attr;
    attr.new_shape =
        ir_model.tensor(tensor_map[output_id])->desc.GetBHWCShape();
    ir_op->attr = std::move(attr);
  }
}

}  // namespace litert::ml_drift::ir
