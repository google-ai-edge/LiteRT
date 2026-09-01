// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/composite/ir/swiglu_parser.h"

#include <cstdint>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/swiglu_parser.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {
namespace {

absl::Status SwigluIsSupported(const TfLiteContext* context,
                               const TfLiteNode* tflite_node,
                               const TfLiteRegistration* /*registration*/) {
  int num_runtime_inputs = 0;
  for (int i = 0; i < tflite_node->inputs->size; ++i) {
    if (tflite_node->inputs->data[i] != kTfLiteOptionalTensor &&
        !::tflite::IsConstantTensor(
            &context->tensors[tflite_node->inputs->data[i]])) {
      ++num_runtime_inputs;
    }
  }

  if (num_runtime_inputs != 1 && num_runtime_inputs != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "SwiGLU expects 1 or 2 inputs, but got ", num_runtime_inputs));
  }

  if (tflite_node->outputs->size != 1) {
    return absl::InvalidArgumentError("SwiGLU expects 1 output.");
  }

  return absl::OkStatus();
}

void SwigluConvert(
    const TfLiteContext& /*context*/, const TfLiteNode& tflite_node,
    const TfLiteRegistration& /*registration*/,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& /*options*/,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* swiglu_op = ir_model.add_op();
  swiglu_op->name = "swiglu";

  for (int i = 0; i < tflite_node.inputs->size; ++i) {
    if (tflite_node.inputs->data[i] != kTfLiteOptionalTensor) {
      ir_model.AddConsumer(tensor_map[tflite_node.inputs->data[i]],
                           swiglu_op->id);
    }
  }
  ir_model.SetProducer(tensor_map[tflite_node.outputs->data[0]], swiglu_op->id);

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node.builtin_data);
  ::litert::ml_drift::SwigluAttributes attr;
  if (params && params->attributes && params->attributes_size > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(params->attributes),
            params->attributes_size)
            .AsMap();
    if (!flexbuffer_map["gate_size"].IsNull()) {
      attr.gate_size = flexbuffer_map["gate_size"].AsInt32();
    }
  } else if (tflite_node.custom_initial_data &&
             tflite_node.custom_initial_data_size > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(tflite_node.custom_initial_data),
            tflite_node.custom_initial_data_size)
            .AsMap();
    if (!flexbuffer_map["gate_size"].IsNull()) {
      attr.gate_size = flexbuffer_map["gate_size"].AsInt32();
    }
  }
  swiglu_op->attr = attr;
}

}  // namespace

CustomIrOpParser GetSwigluParser() {
  return {SwigluIsSupported, SwigluConvert};
}

}  // namespace litert::ml_drift::ir
