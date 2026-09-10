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

#include "ml_drift_delegate/delegate/composite/ir/short_conv_step_parser.h"

#include <cstdint>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/ir_model.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/short_conv_step_parser.h"
#include "ml_drift_delegate/tflite/custom_ir_operation_parser.h"
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {
namespace {

absl::Status ShortConvStepIsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* /*registration*/) {
  if (tflite_node->inputs->size != 3 && tflite_node->inputs->size != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("ShortConvStep expects 3 or 4 inputs, but got ",
                     tflite_node->inputs->size));
  }

  if (tflite_node->outputs->size != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("ShortConvStep expects 2 outputs, but got ",
                     tflite_node->outputs->size));
  }

  return absl::OkStatus();
}

void ShortConvStepConvert(
    const TfLiteContext& /*context*/, const TfLiteNode& tflite_node,
    const TfLiteRegistration& /*registration*/,
    absl::flat_hash_map<int, ::ml_drift::ir::IrTensorId>& tensor_map,
    const IrModelBuilderOptions& /*options*/,
    ::ml_drift::ir::IrModel& ir_model) {
  ::ml_drift::ir::IrOp* op = ir_model.add_op();
  op->name = "short_conv_step";

  for (int i = 0; i < tflite_node.inputs->size; ++i) {
    if (tflite_node.inputs->data[i] != kTfLiteOptionalTensor) {
      ir_model.AddConsumer(tensor_map[tflite_node.inputs->data[i]], op->id);
    }
  }
  for (int i = 0; i < tflite_node.outputs->size; ++i) {
    ir_model.SetProducer(tensor_map[tflite_node.outputs->data[i]], op->id);
  }

  const auto* params = static_cast<const TfLiteStablehloCompositeParams*>(
      tflite_node.builtin_data);
  ::litert::ml_drift::ShortConvStepAttributes attr;
  if (params && params->attributes && params->attributes_size > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(params->attributes),
            params->attributes_size)
            .AsMap();
    if (!flexbuffer_map["conv_L_cache"].IsNull()) {
      attr.conv_L_cache = flexbuffer_map["conv_L_cache"].AsInt32();
    }
  } else if (tflite_node.custom_initial_data &&
             tflite_node.custom_initial_data_size > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(tflite_node.custom_initial_data),
            tflite_node.custom_initial_data_size)
            .AsMap();
    if (!flexbuffer_map["conv_L_cache"].IsNull()) {
      attr.conv_L_cache = flexbuffer_map["conv_L_cache"].AsInt32();
    }
  }
  op->attr = attr;
}

}  // namespace

CustomIrOpParser GetShortConvStepParser() {
  return {ShortConvStepIsSupported, ShortConvStepConvert};
}

}  // namespace litert::ml_drift::ir
