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

#include "ml_drift_delegate/tflite/convert/convert_cumsum.h"

#include <cstdint>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/internal/tensor_ctypes.h"

namespace litert::ml_drift::ir {

void ConvertCumsum(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const int input_id = node.inputs->data[0];
  const int axis_id = node.inputs->data[1];
  const int output_id = node.outputs->data[0];

  const TfLiteTensor& axis_tensor = context.tensors[axis_id];
  const int tflite_axis = tflite::GetTensorData<int32_t>(&axis_tensor)[0];

  const TfLiteTensor& input_tensor = context.tensors[input_id];

  ::ml_drift::CumsumAttributes attr;
  attr.axis = ExtractAxisFromIndex(input_tensor, tflite_axis);

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::CUMSUM);
  op->attr = attr;

  ir_model.AddConsumer(tensor_map.at(input_id), op->id);
  ir_model.SetProducer(tensor_map.at(output_id), op->id);
}

}  // namespace litert::ml_drift::ir
