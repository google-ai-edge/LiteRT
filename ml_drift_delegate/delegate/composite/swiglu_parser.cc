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

#include "ml_drift_delegate/delegate/composite/swiglu_parser.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

absl::Status SwigluOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  if (tflite_node->inputs->size != 1 && tflite_node->inputs->size != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("SwiGLU expects 1 or 2 inputs, but got ",
                     tflite_node->inputs->size));
  }
  if (tflite_node->outputs->size != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("SwiGLU expects 1 output, but got ",
                     tflite_node->outputs->size));
  }
  for (int i = 0; i < tflite_node->inputs->size; ++i) {
    ABSL_RETURN_IF_ERROR(
        PreCheckRuntimeOrConstantInput(context, tflite_node, i));
  }
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
  return absl::OkStatus();
}

void SwigluOperationParser::Parse(const TfLiteNode* tflite_node,
                                  const TfLiteRegistration*,
                                  ::ml_drift::GraphFloat32* graph,
                                  ObjectReader* reader) {
  auto* node = graph->NewNode();
  node->operation.type = kSwigluType;
  for (int i = 0; i < tflite_node->inputs->size; ++i) {
    if (reader->IsConstantTensor(i)) {
      const ::ml_drift::Value* input = reader->AddConstInput(i, /*layout=*/{});
      graph->AddConsumer(node->id, input->id);
    } else {
      reader->AddInput(node, i);
    }
  }
  reader->AddOutputs(node);

  SwigluAttributes attr;
  const uint8_t* buffer_t = nullptr;
  size_t length = 0;
  if (tflite_node->custom_initial_data &&
      tflite_node->custom_initial_data_size > 0) {
    buffer_t =
        reinterpret_cast<const uint8_t*>(tflite_node->custom_initial_data);
    length = tflite_node->custom_initial_data_size;
  } else if (tflite_node->builtin_data) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(
            tflite_node->builtin_data);
    if (composite_params && composite_params->attributes &&
        composite_params->attributes_size > 0) {
      buffer_t =
          reinterpret_cast<const uint8_t*>(composite_params->attributes);
      length = composite_params->attributes_size;
    }
  }
  if (buffer_t && length > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(buffer_t, length).AsMap();
    if (!flexbuffer_map["gate_size"].IsNull()) {
      attr.gate_size = flexbuffer_map["gate_size"].AsInt32();
    }
  }
  node->operation.attributes = std::move(attr);
}

}  // namespace litert::ml_drift
