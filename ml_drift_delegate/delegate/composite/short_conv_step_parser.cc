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

#include "ml_drift_delegate/delegate/composite/short_conv_step_parser.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/object_reader.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

ShortConvStepAttributes ParseShortConvStepAttributes(
    const TfLiteNode& tflite_node) {
  ShortConvStepAttributes attr;
  const uint8_t* buffer_t = nullptr;
  size_t length = 0;
  if (tflite_node.custom_initial_data &&
      tflite_node.custom_initial_data_size > 0) {
    buffer_t =
        reinterpret_cast<const uint8_t*>(tflite_node.custom_initial_data);
    length = tflite_node.custom_initial_data_size;
  } else if (tflite_node.builtin_data) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(
            tflite_node.builtin_data);
    if (composite_params && composite_params->attributes &&
        composite_params->attributes_size > 0) {
      buffer_t = reinterpret_cast<const uint8_t*>(composite_params->attributes);
      length = composite_params->attributes_size;
    }
  }
  if (buffer_t && length > 0) {
    const flexbuffers::Map flexbuffer_map =
        flexbuffers::GetRoot(buffer_t, length).AsMap();
    if (!flexbuffer_map["conv_L_cache"].IsNull()) {
      attr.conv_L_cache = flexbuffer_map["conv_L_cache"].AsInt32();
    }
    if (!flexbuffer_map["is_gated"].IsNull()) {
      attr.is_gated = flexbuffer_map["is_gated"].AsBool();
    }
    if (!flexbuffer_map["use_silu"].IsNull()) {
      attr.use_silu = flexbuffer_map["use_silu"].AsBool();
    }
    if (!flexbuffer_map["channels_last"].IsNull()) {
      attr.channels_last = flexbuffer_map["channels_last"].AsBool();
    }
  }
  return attr;
}

namespace {

// Returns the dims of `tensor`, or an error naming `name` if it has none.
absl::StatusOr<const TfLiteIntArray*> GetDims(const TfLiteContext* context,
                                              const TfLiteIntArray* indices,
                                              int index,
                                              absl::string_view name) {
  const int tensor_index = indices->data[index];
  if (tensor_index < 0 || context == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("ShortConvStep is missing ", name, "."));
  }
  const TfLiteIntArray* dims = context->tensors[tensor_index].dims;
  if (dims == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("ShortConvStep requires a static ", name, " shape."));
  }
  return dims;
}

std::string DimsToString(const TfLiteIntArray& dims) {
  std::string s = "[";
  for (int i = 0; i < dims.size; ++i) {
    absl::StrAppend(&s, i > 0 ? ", " : "", dims.data[i]);
  }
  return s + "]";
}

}  // namespace

absl::Status CheckShortConvStepSupported(const TfLiteContext* context,
                                         const TfLiteNode* tflite_node) {
  if (tflite_node->inputs->size < 3 || tflite_node->inputs->size > 5) {
    return absl::InvalidArgumentError(
        absl::StrCat("ShortConvStep expects 3 to 5 inputs, but got ",
                     tflite_node->inputs->size));
  }
  if (tflite_node->outputs->size != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("ShortConvStep expects 2 outputs, but got ",
                     tflite_node->outputs->size));
  }
  const ShortConvStepAttributes attr =
      ParseShortConvStepAttributes(*tflite_node);
  if (attr.conv_L_cache < 2 || attr.conv_L_cache > 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("ShortConvStep supports conv_L_cache in [2, 4], but got ",
                     attr.conv_L_cache));
  }
  if (!attr.channels_last) {
    return absl::UnimplementedError(
        "ShortConvStep supports only channels-last conv_state and "
        "conv_weight (channels_last = false).");
  }

  // conv_state is [1, L - 1, H] and conv_weight [1, L, H]; H comes from the
  // state so that in_proj may be wider than the output.
  const int taps = attr.conv_L_cache;
  ABSL_ASSIGN_OR_RETURN(const TfLiteIntArray* state,
                        GetDims(context, tflite_node->inputs, 1, "conv_state"));
  ABSL_ASSIGN_OR_RETURN(
      const TfLiteIntArray* weight,
      GetDims(context, tflite_node->inputs, 2, "conv_weight"));
  if (state->size != 3 || state->data[0] != 1 || state->data[1] != taps - 1 ||
      state->data[2] <= 0) {
    return absl::UnimplementedError(absl::StrCat(
        "ShortConvStep supports only a channels-last conv_state [1, ", taps - 1,
        ", H], but got ", DimsToString(*state)));
  }
  const int hidden_size = state->data[2];
  if (weight->size != 3 || weight->data[0] != 1 || weight->data[1] != taps ||
      weight->data[2] != hidden_size) {
    return absl::UnimplementedError(absl::StrCat(
        "ShortConvStep supports only a channels-last conv_weight [1, ", taps,
        ", ", hidden_size, "], but got ", DimsToString(*weight)));
  }
  ABSL_ASSIGN_OR_RETURN(const TfLiteIntArray* out,
                        GetDims(context, tflite_node->outputs, 0, "output"));
  if (out->size == 0 || out->data[out->size - 1] != hidden_size) {
    return absl::UnimplementedError(
        absl::StrCat("ShortConvStep expects ", hidden_size,
                     " output channels, but got ", DimsToString(*out)));
  }
  return absl::OkStatus();
}

absl::Status ShortConvStepOperationParser::IsSupported(
    const TfLiteContext* context, const TfLiteNode* tflite_node,
    const TfLiteRegistration*) {
  ABSL_RETURN_IF_ERROR(CheckShortConvStepSupported(context, tflite_node));
  for (int i = 0; i < tflite_node->inputs->size; ++i) {
    ABSL_RETURN_IF_ERROR(
        PreCheckRuntimeOrConstantInput(context, tflite_node, i));
  }
  ABSL_RETURN_IF_ERROR(PreCheckOutputs(context, tflite_node));
  return absl::OkStatus();
}

void ShortConvStepOperationParser::Parse(const TfLiteNode* tflite_node,
                                         const TfLiteRegistration*,
                                         ::ml_drift::GraphFloat32* graph,
                                         ObjectReader* reader) {
  auto* node = graph->NewNode();
  node->operation.type = kShortConvStepType;
  for (int i = 0; i < tflite_node->inputs->size; ++i) {
    // Input 1 (conv_state) is not added to the GPU node; the kernel reads and
    // updates output 1 (next_state) in-place on GPU, matching
    // odml.cache_update.
    if (i == 1) continue;
    if (reader->IsConstantTensor(i)) {
      const ::ml_drift::Value* input = reader->AddConstInput(i, /*layout=*/{});
      graph->AddConsumer(node->id, input->id);
    } else {
      reader->AddInput(node, i);
    }
  }
  reader->AddOutputs(node);
  node->operation.attributes = ParseShortConvStepAttributes(*tflite_node);
}

}  // namespace litert::ml_drift
