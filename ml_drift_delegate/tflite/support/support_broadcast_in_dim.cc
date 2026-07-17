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

#include "ml_drift_delegate/tflite/support/support_broadcast_in_dim.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

bool IsBroadcastInDimSupported(
    const TfLiteContext* absl_nonnull context,
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
  const absl::Status status =
      CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>(
          &axis);
  if (!status.ok()) {
    *error = status.message();
    return false;
  }
  if (!CheckTensorDims(input, /*min_dims=*/1, /*max_dims=*/5, "inputs[0]",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(axis, /*min_dims=*/1, /*max_dims=*/1, "inputs[1]",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(output, /*min_dims=*/1, /*max_dims=*/5, "outputs[0]",
                       *error)) {
    return false;
  }
  // Check axis value.
  if (axis.dims->data[0] != input.dims->size) {
    *error = absl::StrCat("Axis dims size ", axis.dims->data[0],
                          " does not match input dims size ", input.dims->size);
    return false;
  }
  for (int i = 0; i < axis.dims->data[0]; ++i) {
    const int axis_value = axis.data.i32[i];
    if (axis_value < 0 || axis_value >= output.dims->size) {
      *error = absl::StrCat("Invalid axis value: ", axis_value,
                            ", should be in [0, ", output.dims->size, ")");
      return false;
    }
    if (input.dims->data[i] != 1 &&
        input.dims->data[i] != output.dims->data[axis_value]) {
      *error = absl::StrCat("Input dims must be 1 at axis ", axis_value,
                            " or equal to output dims: ", input.dims->data[i],
                            " != ", output.dims->data[axis_value]);
      return false;
    }
  }
  // Any dims that aren't mapped to by the input must be 1.
  for (int i = 0; i < output.dims->size; ++i) {
    // Check if they are broadcasted to
    bool found = false;
    for (int j = 0; j < axis.dims->data[0]; ++j) {
      if (i == axis.data.i32[j]) {
        found = true;
        break;
      }
    }
    if (!found && output.dims->data[i] != 1) {
      *error = absl::StrCat("Output dims must be 1 at index ", i,
                            " or equal to output dims: ", output.dims->data[i],
                            " != 1");
      return false;
    }
  }
  return true;
}

}  // namespace litert::ml_drift::ir
