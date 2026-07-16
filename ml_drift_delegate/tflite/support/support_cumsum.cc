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

#include "ml_drift_delegate/tflite/support/support_cumsum.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

bool IsCumsumSupported(const TfLiteContext* absl_nonnull context,
                       const TfLiteNode* absl_nonnull node,
                       const TfLiteRegistration* absl_nonnull registration,
                       std::string* absl_nonnull error) {
  // Check version.
  if (registration->version != 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
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
  const int axis_id = inputs->data[1];
  const int output_id = outputs->data[0];

  // Check dtype.
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteBool,
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
      kTfLiteInt16,
      kTfLiteInt32,
      kTfLiteUInt8,
      kTfLiteUInt16,
      kTfLiteUInt32,
      // go/keep-sorted end
      // clang-format on
  };
  const TfLiteTensor& input = context->tensors[input_id];
  if (!CheckTensorDtype(input, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& axis = context->tensors[axis_id];
  if (!CheckTensorDtype(axis, {kTfLiteInt32}, "inputs[1]", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }
  // Check const inputs.
  if (!CheckNotConstant(input, "inputs[0]", *error)) {
    return false;
  }
  if (!CheckIsConstant(axis, "inputs[1]", *error)) {
    return false;
  }
  // Check axis value.
  if (!CheckTensorDims(axis, /*min_dims=*/0, /*max_dims=*/1, "inputs[1]",
                       *error)) {
    return false;
  }
  if (axis.dims->size == 1 && axis.dims->data[0] != 1) {
    *error = absl::StrCat("Invalid number of axis values: ", axis.dims->data[0],
                          ", should be 1");
    return false;
  }
  const int axis_value =
      ResolveNegativeIndex(axis.data.i32[0], input.dims->size);
  if (axis_value < 0 || axis_value >= input.dims->size) {
    *error = absl::StrCat("Invalid axis value: ", axis_value);
    return false;
  }
  // Input shape must match output shape.
  if (!TfLiteIntArrayEqual(input.dims, output.dims)) {
    *error = absl::StrCat("Input and output dimensions must match: ",
                          input.dims->size, " != ", output.dims->size);
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
