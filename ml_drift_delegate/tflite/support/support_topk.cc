// Copyright 2026 The ML Drift Authors.
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

#include "ml_drift_delegate/tflite/support/support_topk.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

bool IsTopKSupported(const TfLiteContext* absl_nonnull context,
                     const TfLiteNode* absl_nonnull node,
                     const TfLiteRegistration* absl_nonnull registration,
                     std::string* absl_nonnull error) {
  // Check version.
  if (registration->version < 1 || registration->version > 2) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check number of inputs and outputs.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/2,
                              /*expected_outputs=*/2, *error)) {
    return false;
  }

  const TfLiteIntArray* inputs = node->inputs;
  const TfLiteIntArray* outputs = node->outputs;

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int input_id = inputs->data[0];
  const int k_id = inputs->data[1];
  const int output_id_values = outputs->data[0];
  const int output_id_indices = outputs->data[1];

  // Check dtype.
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      kTfLiteFloat16,
      kTfLiteFloat32,
  };
  const TfLiteTensor& input = context->tensors[input_id];
  if (!CheckTensorDtype(input, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& k_tensor = context->tensors[k_id];
  if (!CheckTensorDtype(k_tensor, {kTfLiteInt32}, "inputs[1]", *error)) {
    return false;
  }
  const TfLiteTensor& output_values = context->tensors[output_id_values];
  if (!CheckTensorDtype(output_values, supported_dtypes, "outputs[0]",
                        *error)) {
    return false;
  }
  const TfLiteTensor& output_indices = context->tensors[output_id_indices];
  if (!CheckTensorDtype(output_indices, {kTfLiteInt32}, "outputs[1]", *error)) {
    return false;
  }

  // Check const inputs.
  if (!::tflite::IsConstantTensor(&k_tensor)) {
    *error = "TopK k-tensor must be a constant tensor.";
    return false;
  }

  if (::tflite::NumElements(&k_tensor) != 1) {
    *error = "TopK k-tensor must be a scalar.";
    return false;
  }

  return true;
}

}  // namespace litert::ml_drift::ir
