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

#include "ml_drift_delegate/tflite/support/support_group_norm.h"

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

bool IsGroupNormSupported(const TfLiteContext* absl_nonnull context,
                          const TfLiteNode* absl_nonnull node,
                          const TfLiteRegistration* absl_nonnull registration,
                          std::string* absl_nonnull error) {
  if (registration->version != 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  const TfLiteIntArray* inputs = node->inputs;
  if (inputs->size < 1 || inputs->size > 3) {
    *error = absl::StrCat("Invalid number of inputs: ", inputs->size,
                          ", should be between 1 and 3");
    return false;
  }

  const TfLiteIntArray* outputs = node->outputs;
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/inputs->size,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int input_id = inputs->data[0];
  const TfLiteTensor& input = context->tensors[input_id];

  std::optional<int> gamma_id;
  if (inputs->size > 1) {
    gamma_id = inputs->data[1];
  }
  const TfLiteTensor* gamma =
      gamma_id.has_value() ? &context->tensors[*gamma_id] : nullptr;
  std::optional<int> beta_id;
  if (inputs->size > 2) {
    beta_id = inputs->data[2];
  }
  const TfLiteTensor* beta =
      beta_id.has_value() ? &context->tensors[*beta_id] : nullptr;

  const int output_id = outputs->data[0];
  const TfLiteTensor& output = context->tensors[output_id];

  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteFloat16,
      kTfLiteFloat32,
      // go/keep-sorted end
      // clang-format on
  };

  if (!CheckTensorDtype(input, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  if (gamma != nullptr) {
    if (!CheckTensorDtype(*gamma, supported_dtypes, "inputs[1]", *error)) {
      return false;
    }
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>(
            gamma);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  }
  if (beta != nullptr) {
    if (!CheckTensorDtype(*beta, supported_dtypes, "inputs[2]", *error)) {
      return false;
    }
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>(
            beta);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  }
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }

  if (!CheckNotConstant(input, "inputs[0]", *error)) {
    return false;
  }

  if (!CheckTensorDims(input, /*min_dims=*/4, /*max_dims=*/4, "inputs[0]",
                       *error)) {
    return false;
  }
  if (gamma != nullptr) {
    if (!CheckTensorDims(*gamma, /*min_dims=*/1, /*max_dims=*/1, "inputs[1]",
                         *error)) {
      return false;
    }
    if (input.dims->data[3] != gamma->dims->data[0]) {
      *error = "Gamma tensor length doesn't match input tensor channels.";
      return false;
    }
  }
  if (beta != nullptr) {
    if (!CheckTensorDims(*beta, /*min_dims=*/1, /*max_dims=*/1, "inputs[2]",
                         *error)) {
      return false;
    }
    if (input.dims->data[3] != beta->dims->data[0]) {
      *error = "beta tensor length doesn't match input tensor channels.";
      return false;
    }
  }

  const auto* params =
      static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
  if (params == nullptr) {
    *error = "GroupNorm is missing params.";
    return false;
  }
  const flexbuffers::Map flexbuffer_map =
      flexbuffers::GetRoot(params->attributes, params->attributes_size).AsMap();
  const int tensor_dims_size = input.dims->size;

  if (!flexbuffer_map["_TENSOR_V1_reduction_axes"].IsNull()) {
    const flexbuffers::Vector reduction_axes_vec =
        flexbuffer_map["_TENSOR_V1_reduction_axes"]
            .AsMap()["TENSOR_DATA"]
            .AsVector();
    if (reduction_axes_vec.size() != 1 ||
        reduction_axes_vec[0].AsInt64() != tensor_dims_size - 1) {
      *error = "Only reduction on the last axis is supported for GroupNorm.";
      return false;
    }
  }

  if (!flexbuffer_map["channel_axis"].IsNull() &&
      flexbuffer_map["channel_axis"].AsInt32() != tensor_dims_size - 1) {
    *error = "Only channel-last tensor is supported for GroupNorm.";
    return false;
  }
  if (flexbuffer_map["num_groups"].IsNull()) {
    *error = "GroupNorm is missing num_groups.";
    return false;
  }
  if (flexbuffer_map["epsilon"].IsNull()) {
    *error = "GroupNorm is missing epsilon.";
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
