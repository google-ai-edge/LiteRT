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

#include "ml_drift_delegate/tflite/convert/convert_reverse.h"

#include <set>
#include <utility>

#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

void ConvertReverse(const TfLiteContext& context, const TfLiteNode& node,
                    const TfLiteRegistration& registration,
                    ::ml_drift::ir::TensorMap& tensor_map,
                    ::ml_drift::ir::IrModel& ir_model) {
  const TfLiteTensor* input_tensor = context.tensors + node.inputs->data[0];
  const TfLiteTensor* axes_tensor = context.tensors + node.inputs->data[1];
  const int output_id = node.outputs->data[0];
  const ::ml_drift::ir::IrTensorId ir_output_id = tensor_map[output_id];

  ::ml_drift::ReverseAttributes attr;
  const int num_axes = tflite::NumElements(axes_tensor->dims);
  for (int i = 0; i < num_axes; ++i) {
    int axis_index = axes_tensor->data.i32[i];
    attr.axes.insert(ExtractAxisFromIndex(*input_tensor, axis_index));
  }

  ::ml_drift::ir::IrOp* reverse_op = ir_model.add_op();
  reverse_op->name = ToString(::ml_drift::OperationType::REVERSE);
  reverse_op->attr = std::move(attr);

  ir_model.AddConsumer(tensor_map[node.inputs->data[0]], reverse_op->id);
  ir_model.SetProducer(ir_output_id, reverse_op->id);
}

}  // namespace litert::ml_drift::ir
