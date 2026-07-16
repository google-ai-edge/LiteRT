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

#include "ml_drift_delegate/tflite/support/support_one_hot.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"

using ::tflite::IsConstantTensor;

namespace litert::ml_drift::ir {

constexpr int kMaxDims = 4;

bool IsOneHotSupported(const TfLiteContext* absl_nonnull context,
                       const TfLiteNode* absl_nonnull node,
                       const TfLiteRegistration* absl_nonnull registration,
                       std::string* absl_nonnull error) {
  // Check version
  if (registration->version != 1) {
    *error = "Unsupported version.";
    return false;
  }

  // Check number of inputs.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/4,
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
  const int on_id = inputs->data[2];
  const int off_id = inputs->data[3];
  const int output_id = outputs->data[0];

  // Check dtype.
  const TfLiteTensor& input = context->tensors[input_id];
  if (!CheckTensorDtype(input, {kTfLiteInt32}, "input", *error)) {
    return false;
  }
  // depth (input tensor 2) is unused by ml_drift.
  const TfLiteTensor& on = context->tensors[on_id];
  if (!CheckTensorDtype(on, {kTfLiteFloat32}, "on", *error)) {
    return false;
  }
  const TfLiteTensor& off = context->tensors[off_id];
  if (!CheckTensorDtype(off, {kTfLiteFloat32}, "off", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, {kTfLiteInt32}, "output", *error)) {
    return false;
  }

  // Check const inputs.
  if (IsConstantTensor(&input)) {
    return false;
  }

  // Check number of dims.
  if (!CheckTensorDims(input, /*min_dims=*/1, /*max_dims=*/kMaxDims, "input",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(on, /*min_dims=*/1, /*max_dims=*/1, "on", *error)) {
    return false;
  }
  if (!CheckTensorDims(off, /*min_dims=*/1, /*max_dims=*/1, "off", *error)) {
    return false;
  }
  if (!CheckTensorDims(output, /*min_dims=*/1, /*max_dims=*/kMaxDims, "output",
                       *error)) {
    return false;
  }
  // Check on / off tensor have a single value
  if (on.dims->data[0] != 1) {
    *error = "On tensor must have a single value";
    return false;
  }
  if (off.dims->data[0] != 1) {
    *error = "Off tensor must have a single value";
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
