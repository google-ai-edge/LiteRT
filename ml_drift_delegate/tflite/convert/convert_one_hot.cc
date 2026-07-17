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

#include "ml_drift_delegate/tflite/convert/convert_one_hot.h"

#include <utility>

#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertOneHot(const TfLiteContext& context, const TfLiteNode& node,
                   const TfLiteRegistration& registration,
                   ::ml_drift::ir::TensorMap& tensor_map,
                   ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* one_hot_op = ir_model.add_op();
  one_hot_op->name = ToString(::ml_drift::OperationType::ONE_HOT);

  const int input_id = tensor_map[node.inputs->data[0]];
  ir_model.AddConsumer(input_id, one_hot_op->id);

  ::ml_drift::OneHotAttributes attr;
  const TfLiteTensor& on_tensor = context.tensors[node.inputs->data[2]];
  const TfLiteTensor& off_tensor = context.tensors[node.inputs->data[3]];

  attr.on_value = on_tensor.data.f[0];
  attr.off_value = off_tensor.data.f[0];

  one_hot_op->attr = std::move(attr);
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], one_hot_op->id);
}

}  // namespace litert::ml_drift::ir
