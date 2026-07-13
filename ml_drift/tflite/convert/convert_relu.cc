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

#include "third_party/odml/litert/ml_drift/tflite/convert/convert_relu.h"

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

void ConvertRelu(
    const TfLiteNode& node, const TfLiteRegistration& registration,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* relu_op = ir_model.add_op();
  relu_op->name = ToString(::ml_drift::OperationType::RELU);
  ir_model.AddConsumer(tensor_map[node.inputs->data[0]], relu_op->id);
  ir_model.SetProducer(tensor_map[node.outputs->data[0]], relu_op->id);

  ::ml_drift::ReLUAttributes attr;
  switch (registration.builtin_code) {
    case kTfLiteBuiltinRelu:
      attr.activation_min = 0;
      attr.activation_max = 0;
      break;
    case kTfLiteBuiltinRelu6:
      attr.activation_min = 0;
      attr.activation_max = 6;
      break;
    case kTfLiteBuiltinRelu0To1:
      attr.activation_min = 0.0f;
      attr.activation_max = 1.0f;
      break;
    case kTfLiteBuiltinReluN1To1:
      attr.activation_min = -1.0f;
      attr.activation_max = 1.0f;
      break;
    case kTfLiteBuiltinLeakyRelu: {
      const auto* params =
          static_cast<const TfLiteLeakyReluParams*>(node.builtin_data);
      attr.alpha = params ? params->alpha : 0;
      attr.activation_min = 0;
      attr.activation_max = 0;
      break;
    }
    default:
      break;
  }
  relu_op->attr = attr;
}

}  // namespace litert::ml_drift::ir
