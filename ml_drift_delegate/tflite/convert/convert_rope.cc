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

#include "ml_drift_delegate/tflite/convert/convert_rope.h"

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertRoPE(
    const TfLiteContext& context, const TfLiteNode& node,
    const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = ToString(::ml_drift::OperationType::ROPE);

  // Either 2 inputs -> 1 output, or 3 inputs -> 2 outputs.
  for (int i = 0; i < node.inputs->size; ++i) {
    ir_model.AddConsumer(tensor_map[node.inputs->data[i]], op->id);
  }

  for (int i = 0; i < node.outputs->size; ++i) {
    ir_model.SetProducer(tensor_map[node.outputs->data[i]], op->id);
  }
}

}  // namespace litert::ml_drift::ir
