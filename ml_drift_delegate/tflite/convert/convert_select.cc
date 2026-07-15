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

#include "ml_drift_delegate/tflite/convert/convert_select.h"

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/convert/convert_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

void ConvertSelect(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  const int cond_id = node.inputs->data[0];
  const int if_id = node.inputs->data[1];
  const int else_id = node.inputs->data[2];
  const int output_id = node.outputs->data[0];

  ::ml_drift::SelectV2Attributes attr;

  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::SELECT_V2);
  op->attr = attr;

  // Handle input 0 (cond)
  if (tflite::IsConstantTensor(context.tensors + cond_id)) {
    ::ml_drift::ir::IrTensor* const_tensor =
        AddConstInput(context, cond_id, ir_model, /*layout=*/{});
    ir_model.AddConsumer(const_tensor->id, op->id);
  } else {
    ir_model.AddConsumer(tensor_map.at(cond_id), op->id);
  }

  // num_dims == 3; convert HWC to 1HWC for constant tensors
  SizedLayout constants_layout;
  constants_layout.layout_3d = ::ml_drift::Layout::HWC;

  // Handle input 1 (if)
  if (tflite::IsConstantTensor(context.tensors + if_id)) {
    ::ml_drift::ir::IrTensor* const_tensor =
        AddConstInput(context, if_id, ir_model, constants_layout);
    ir_model.AddConsumer(const_tensor->id, op->id);
  } else {
    ir_model.AddConsumer(tensor_map.at(if_id), op->id);
  }

  // Handle input 2 (else)
  if (tflite::IsConstantTensor(context.tensors + else_id)) {
    ::ml_drift::ir::IrTensor* const_tensor =
        AddConstInput(context, else_id, ir_model, constants_layout);
    ir_model.AddConsumer(const_tensor->id, op->id);
  } else {
    ir_model.AddConsumer(tensor_map.at(else_id), op->id);
  }

  ir_model.SetProducer(tensor_map.at(output_id), op->id);
}

}  // namespace litert::ml_drift::ir
