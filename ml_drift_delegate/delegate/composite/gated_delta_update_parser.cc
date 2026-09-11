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

#include "ml_drift_delegate/delegate/composite/gated_delta_update_parser.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

absl::Status GatedDeltaUpdateOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  if (tflite_node->inputs->size != 6) {
    return absl::UnavailableError("gated_delta_update expects 6 inputs.");
  }
  for (int i = 0; i < 6; ++i) {
    ABSL_RETURN_IF_ERROR(PreCheckReadValue(context, tflite_node, i));
  }
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));

  if (tflite_node->outputs->size != 2) {
    return absl::InvalidArgumentError("gated_delta_update expects 2 outputs.");
  }

  const TfLiteTensor* q = nullptr;
  ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 0, &q));
  const TfLiteTensor* v = nullptr;
  ABSL_RETURN_IF_ERROR(PreGetInputTensor(context, tflite_node, 2, &v));
  if (q && q->dims && q->dims->size >= 4 && v && v->dims &&
      v->dims->size >= 4) {
    int D_k = q->dims->data[q->dims->size - 1];
    int D_v = v->dims->data[v->dims->size - 1];
    bool d_k_valid =
        (D_k >= 16) && (D_k % 4 == 0) && (((D_k / 4) & ((D_k / 4) - 1)) == 0);
    bool d_v_valid =
        (D_v >= 16) && (D_v % 4 == 0) && (((D_v / 4) & ((D_v / 4) - 1)) == 0);
    if (!d_k_valid || !d_v_valid) {
      return absl::InvalidArgumentError(
          "gated_delta_update requires D_k and D_v to be powers of 2 (at "
          "least 16) and multiples of 4.");
    }
  }

  return absl::OkStatus();
}

void GatedDeltaUpdateOperationParser::Parse(const TfLiteNode* tflite_node,
                                            const TfLiteRegistration*,
                                            ::ml_drift::GraphFloat32* graph,
                                            ObjectReader* reader) {
  auto* node = graph->NewNode();
  node->operation.type = kGatedDeltaUpdateType;
  for (int i = 0; i < 6; ++i) {
    if (reader->CanReadValue(i)) {
      reader->AddInput(node, i);
    } else {
      const ::ml_drift::Value* input = reader->AddConstInput(i, /*layout=*/{});
      graph->AddConsumer(node->id, input->id);
    }
  }
  reader->AddOutputs(node);

  GatedDeltaUpdateAttributes attr;
  const uint8_t* buffer_t = nullptr;
  size_t length = 0;
  if (tflite_node->custom_initial_data &&
      tflite_node->custom_initial_data_size > 0) {
    buffer_t =
        reinterpret_cast<const uint8_t*>(tflite_node->custom_initial_data);
    length = tflite_node->custom_initial_data_size;
  }
  if (buffer_t && length > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(buffer_t, length).AsMap();
    if (!flexbuffer_map["mode"].IsNull()) {
      attr.mode = flexbuffer_map["mode"].AsInt32();
    }
  }
  node->operation.attributes = std::move(attr);
}

}  // namespace litert::ml_drift
