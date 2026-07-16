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

#include "ml_drift_delegate/tflite/support/support_select.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

bool IsSelectSupported(const TfLiteContext* absl_nonnull context,
                       const TfLiteNode* absl_nonnull node,
                       const TfLiteRegistration* absl_nonnull registration,
                       std::string* absl_nonnull error) {
  // Check version.
  if (registration->version > 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version,
                          ", max supported version is 1");
    return false;
  }

  // Check number of inputs and outputs.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/3,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  const TfLiteIntArray* inputs = node->inputs;
  const TfLiteIntArray* outputs = node->outputs;

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int cond_id = inputs->data[0];
  const int if_id = inputs->data[1];
  const int else_id = inputs->data[2];
  const int output_id = outputs->data[0];

  // Check dtypes.
  const absl::flat_hash_set<TfLiteType> supported_cond_dtypes = {
      kTfLiteBool,
      kTfLiteFloat16,
      kTfLiteFloat32,
  };
  const TfLiteTensor& cond = context->tensors[cond_id];
  if (!CheckTensorDtype(cond, supported_cond_dtypes, "inputs[0]", *error)) {
    return false;
  }

  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      kTfLiteFloat16, kTfLiteFloat32,  kTfLiteInt8,   kTfLiteInt16,
      kTfLiteInt32,   kTfLiteUInt8,    kTfLiteUInt16, kTfLiteUInt32,
      kTfLiteBool,    kTfLiteBFloat16,
  };
  const TfLiteTensor& if_tensor = context->tensors[if_id];
  if (!CheckTensorDtype(if_tensor, supported_dtypes, "inputs[1]", *error)) {
    return false;
  }
  const TfLiteTensor& else_tensor = context->tensors[else_id];
  if (!CheckTensorDtype(else_tensor, supported_dtypes, "inputs[2]", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }

  // Check tensor dimensions.
  if (!CheckTensorDims(cond, 0, 4, "inputs[0]", *error)) return false;
  if (!CheckTensorDims(if_tensor, 0, 4, "inputs[1]", *error)) return false;
  if (!CheckTensorDims(else_tensor, 0, 4, "inputs[2]", *error)) return false;
  if (!CheckTensorDims(output, 0, 4, "outputs[0]", *error)) return false;

  // Check shapes.
  const auto is_invalid_shape = [&](const TfLiteTensor& tensor) {
    return tensor.dims->size != 0 &&
           !TfLiteIntArrayEqual(tensor.dims, output.dims) &&
           (tensor.dims->size > 1 || tensor.dims->data[0] > 1);
  };
  if (is_invalid_shape(if_tensor) || is_invalid_shape(else_tensor)) {
    *error =
        "The 'if' and 'else' tensors must have the same shape as the output, "
        "be a scalar, or have a single element.";
    return false;
  }

  return true;
}

}  // namespace litert::ml_drift::ir
