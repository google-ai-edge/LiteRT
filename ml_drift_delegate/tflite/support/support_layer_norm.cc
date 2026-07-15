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

#include "ml_drift_delegate/tflite/support/support_layer_norm.h"

#include <optional>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"

namespace litert::ml_drift::ir {

bool IsLayerNormSupported(const TfLiteContext* absl_nonnull context,
                          const TfLiteNode* absl_nonnull node,
                          const TfLiteRegistration* absl_nonnull registration,
                          std::string* absl_nonnull error) {
  // Check version.
  if (registration->version != 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check number of inputs.
  const TfLiteIntArray* inputs = node->inputs;
  if (inputs->size < 1 || inputs->size > 3) {
    *error = absl::StrCat("Invalid number of inputs: ", inputs->size,
                          ", should be between 1 and 3");
    return false;
  }
  // Check number of outputs.
  const TfLiteIntArray* outputs = node->outputs;
  if (outputs->size != 1) {
    *error = absl::StrCat("Invalid number of outputs: ", outputs->size,
                          ", which should be 1");
    return false;
  }
  // Validate tensor IDs.
  const int input_id = inputs->data[0];
  if (!ValidateTensorId(*context, input_id, "inputs", *error)) return false;
  const TfLiteTensor* input = context->tensors + input_id;

  std::optional<int> scale_id;
  if (inputs->size > 1) {
    scale_id = inputs->data[1];
    if (!ValidateTensorId(*context, *scale_id, "scale", *error)) return false;
  }
  const TfLiteTensor* scale =
      scale_id.has_value() ? context->tensors + *scale_id : nullptr;
  std::optional<int> bias_id;
  if (inputs->size > 2) {
    bias_id = inputs->data[2];
    if (!ValidateTensorId(*context, *bias_id, "bias", *error)) return false;
  }
  const TfLiteTensor* bias =
      bias_id.has_value() ? context->tensors + *bias_id : nullptr;

  const int output_id = outputs->data[0];
  if (!ValidateTensorId(*context, output_id, "output", *error)) return false;
  const TfLiteTensor* output = context->tensors + output_id;

  // Check dtype.
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteFloat16,
      kTfLiteFloat32,
      // go/keep-sorted end
      // clang-format on
  };
  if (!CheckTensorDtype(*input, supported_dtypes, "input", *error))
    return false;
  if (scale != nullptr &&
      !CheckTensorDtype(*scale, supported_dtypes, "scale", *error)) {
    return false;
  }
  if (scale != nullptr) {
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>(
            scale);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  }
  if (bias != nullptr &&
      !CheckTensorDtype(*bias, supported_dtypes, "bias", *error)) {
    return false;
  }
  if (bias != nullptr) {
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>(
            bias);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  }
  if (!CheckTensorDtype(*output, supported_dtypes, "output", *error)) {
    return false;
  }

  // Check const inputs.
  if (!CheckNotConstant(*input, "input", *error)) return false;

  // Check shapes
  if (input->dims->size < 2 || input->dims->size > 4) {
    *error = absl::StrCat("Invalid number of input dims: ", input->dims->size,
                          ", should be between 2 and 4");
    return false;
  }
  if (scale != nullptr) {
    if (!scale->dims || scale->dims->size != 1) {
      *error = absl::StrCat("Invalid number of scale dims: ", scale->dims->size,
                            ", should be 1");
      return false;
    }
    if (input->dims->data[input->dims->size - 1] != scale->dims->data[0]) {
      *error = "scale tensor length doesn't match input tensor channels.";
      return false;
    }
  }
  if (bias != nullptr) {
    if (!bias->dims || bias->dims->size != 1) {
      *error = absl::StrCat("Invalid number of bias dims: ", bias->dims->size,
                            ", should be 1");
      return false;
    }
    if (input->dims->data[input->dims->size - 1] != bias->dims->data[0]) {
      *error = "bias tensor length doesn't match input tensor channels.";
      return false;
    }
  }

  // Parse params
  const auto* params =
      static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
  if (params == nullptr) {
    *error = "LayerNorm is missing params.";
    return false;
  }
  const flexbuffers::Map flexbuffer_map =
      flexbuffers::GetRoot(params->attributes, params->attributes_size).AsMap();

  if (flexbuffer_map["epsilon"].IsNull()) {
    *error = "LayerNorm is missing epsilon.";
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
