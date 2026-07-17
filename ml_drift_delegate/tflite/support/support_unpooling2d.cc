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

#include "ml_drift_delegate/tflite/support/support_unpooling2d.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"

namespace litert::ml_drift::ir {

bool IsUnpooling2dSupported(const TfLiteContext* absl_nonnull context,
                            const TfLiteNode* absl_nonnull node,
                            const TfLiteRegistration* absl_nonnull registration,
                            std::string* absl_nonnull error) {
  // Check number of inputs.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/2,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }
  // Validate tensor IDs.
  const TfLiteIntArray* inputs = node->inputs;
  const TfLiteIntArray* outputs = node->outputs;

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int input_id = inputs->data[0];
  const int indices_id = inputs->data[1];
  const int output_id = outputs->data[0];

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
  const TfLiteTensor& input = context->tensors[input_id];
  if (!CheckTensorDtype(input, supported_dtypes, "input", *error)) {
    return false;
  }
  const TfLiteTensor& indices = context->tensors[indices_id];
  if (!CheckTensorDtype(indices, {kTfLiteInt32}, "indices", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "output", *error)) {
    return false;
  }

  // Check const inputs.
  if (!CheckNotConstant(input, "input", *error)) {
    return false;
  }

  // Check shapes
  if (input.dims->size < 3 || input.dims->size > 5) {
    *error = absl::StrCat("Invalid number of input dims: ", input.dims->size,
                          ", should be between 3 and 5");
    return false;
  }
  if (!TfLiteIntArrayEqual(input.dims, indices.dims)) {
    *error = "Input and indices dimensions must match.";
    return false;
  }
  if (output.dims->size != input.dims->size) {
    *error = absl::StrCat("Output and input dimensions must match: ",
                          output.dims->size, " != ", input.dims->size);
    return false;
  }

  // Parse params
  const auto* params =
      static_cast<const TfLitePoolParams*>(node->custom_initial_data);
  if (params == nullptr) {
    *error = "Unpooling2d is missing params.";
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
