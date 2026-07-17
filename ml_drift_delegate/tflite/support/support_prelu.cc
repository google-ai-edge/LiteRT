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

#include "ml_drift_delegate/tflite/support/support_prelu.h"

#include <array>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

bool IsPReLUSupported(const TfLiteContext* absl_nonnull context,
                      const TfLiteNode* absl_nonnull node,
                      const TfLiteRegistration* absl_nonnull registration,
                      int supported_max_version,
                      std::string* absl_nonnull error) {
  // Check version.
  if (registration->version < 1 ||
      registration->version > supported_max_version) {
    *error = absl::StrCat("Unsupported version: ", registration->version,
                          ", supported versions are [1, ",
                          supported_max_version, "]");
    return false;
  }

  // Check number of inputs and outputs.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/2,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  const TfLiteIntArray* inputs = node->inputs;
  const TfLiteIntArray* outputs = node->outputs;

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int input_id = inputs->data[0];
  const int alpha_id = inputs->data[1];
  const int output_id = outputs->data[0];

  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteFloat16,
      kTfLiteFloat32,
      // go/keep-sorted end
      // clang-format on
  };
  const TfLiteTensor& input = context->tensors[input_id];
  if (!CheckTensorDtype(input, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& alpha = context->tensors[alpha_id];
  if (!CheckTensorDtype(alpha, supported_dtypes, "inputs[1]", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }

  if (!CheckNotConstant(input, "inputs[0]", *error)) return false;
  if (!CheckIsConstant(alpha, "inputs[1]", *error)) return false;

  if (!alpha.dims) {
    *error = "Alpha tensor has no dims.";
    return false;
  }

  // Check dimensions.
  if (!CheckTensorDims(input, 1, 4, "inputs[0]", *error)) return false;
  if (!CheckTensorDims(output, 1, 4, "outputs[0]", *error)) return false;

  // Input and output shapes must be equal.
  if (!::tflite::HaveSameShapes(&input, &output)) {
    *error = "Input and output shapes must be equal.";
    return false;
  }

  const int input_dims_size = input.dims->size;
  const int input_channels = input.dims->data[input_dims_size - 1];

  if (alpha.dims->size == 1) {
    if (alpha.dims->data[0] != input_channels) {
      *error = absl::StrCat(
          "Linear alpha shape does not match the number of input channels: ",
          alpha.dims->data[0], " vs ", input_channels);
      return false;
    }
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::FLOAT32>(
            &alpha);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  } else {
    auto get_input_hwc = [](const TfLiteIntArray* dims) -> std::array<int, 3> {
      const int size = dims->size;
      if (size == 0) return {1, 1, 1};
      if (size == 1) return {1, 1, 1};
      if (size == 2) return {1, 1, dims->data[1]};
      if (size == 3) return {1, dims->data[1], dims->data[2]};
      return {dims->data[1], dims->data[2], dims->data[3]};
    };

    auto get_alpha_hwc = [](const TfLiteIntArray* dims) -> std::array<int, 3> {
      const int size = dims->size;
      if (size == 3) return {dims->data[0], dims->data[1], dims->data[2]};
      if (size == 4) return {dims->data[1], dims->data[2], dims->data[3]};
      return {0, 0, 0};
    };

    if (get_input_hwc(input.dims) != get_alpha_hwc(alpha.dims)) {
      *error = "Alpha shape does not match input shape.";
      return false;
    }
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::HWC, ::ml_drift::DataType::FLOAT32>(
            &alpha);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  }

  return true;
}

}  // namespace litert::ml_drift::ir
