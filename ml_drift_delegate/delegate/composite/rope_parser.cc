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

#include "ml_drift_delegate/delegate/composite/rope_parser.h"

#include <cstdint>
#include <cstddef>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

absl::Status RopeOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration* registration) {
  if (tflite_node->inputs->size == 2) {
    if (tflite_node->outputs->size != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid number of outputs: ",
                       tflite_node->outputs->size, ", while expected 1."));
    }
  } else if (tflite_node->inputs->size == 3) {
    if (tflite_node->outputs->size != 2) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid number of outputs: ",
                       tflite_node->outputs->size, ", while expected 2."));
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid number of inputs: ", tflite_node->inputs->size,
                     ", while expected 2 or 3."));
  }

  ABSL_RETURN_IF_ERROR(
      PreCheckRuntimeOrConstantInput(context, tflite_node, 0));
  ABSL_RETURN_IF_ERROR(
      PreCheckRuntimeOrConstantInput(context, tflite_node, 1));
  if (tflite_node->inputs->size != 2) {
    ABSL_RETURN_IF_ERROR(
        PreCheckRuntimeOrConstantInput(context, tflite_node, 2));
  }
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
  return absl::OkStatus();
}

void RopeOperationParser::Parse(const TfLiteNode* tflite_node,
                                const TfLiteRegistration* registration,
                                ::ml_drift::GraphFloat32* graph,
                                ObjectReader* reader) {
  auto* node = graph->NewNode();
  node->operation.type = ToString(::ml_drift::OperationType::ROPE);
  {
    constexpr int kIndex = 0;
    if (reader->IsConstantTensor(kIndex)) {
      const ::ml_drift::Value* input =
          reader->AddConstInput(kIndex, /*layout=*/{});
      graph->AddConsumer(node->id, input->id);
    } else {
      reader->AddInput(node, kIndex);
    }
  }
  {
    constexpr int kIndex = 1;
    if (reader->IsConstantTensor(kIndex)) {
      const ::ml_drift::Value* input =
          reader->AddConstInput(kIndex, /*layout=*/{});
      graph->AddConsumer(node->id, input->id);
    } else {
      reader->AddInput(node, kIndex);
    }
  }
  if (tflite_node->inputs->size > 2) {
    constexpr int kIndex = 2;
    if (reader->IsConstantTensor(kIndex)) {
      const ::ml_drift::Value* input =
          reader->AddConstInput(kIndex, /*layout=*/{});
      graph->AddConsumer(node->id, input->id);
    } else {
      reader->AddInput(node, kIndex);
    }
  }
  reader->AddOutputs(node);

  ::ml_drift::RoPEAttributes attr;
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
    if (!flexbuffer_map["min_timescale"].IsNull()) {
      attr.min_timescale = flexbuffer_map["min_timescale"].AsFloat();
    }
    if (!flexbuffer_map["max_timescale"].IsNull()) {
      attr.max_timescale = flexbuffer_map["max_timescale"].AsFloat();
    } else if (!flexbuffer_map["base"].IsNull()) {
      attr.max_timescale = flexbuffer_map["base"].AsFloat();
    } else if (!flexbuffer_map["rope_theta"].IsNull()) {
      attr.max_timescale = flexbuffer_map["rope_theta"].AsFloat();
    } else if (!flexbuffer_map["theta"].IsNull()) {
      attr.max_timescale = flexbuffer_map["theta"].AsFloat();
    }
    if (!flexbuffer_map["proportion"].IsNull()) {
      attr.proportion = flexbuffer_map["proportion"].AsFloat();
    } else if (!flexbuffer_map["partial_rotary_factor"].IsNull()) {
      attr.proportion = flexbuffer_map["partial_rotary_factor"].AsFloat();
    }
    if (!flexbuffer_map["kernel_type"].IsNull()) {
      attr.kernel_type = static_cast<::ml_drift::RoPEKernelType>(
          flexbuffer_map["kernel_type"].AsInt32());
    }
  }
  node->operation.attributes = std::move(attr);
}

}  // namespace litert::ml_drift
