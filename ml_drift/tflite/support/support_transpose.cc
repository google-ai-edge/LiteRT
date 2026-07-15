// Copyright 2026 Google LLC.
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

#include "third_party/odml/litert/ml_drift/tflite/support/support_transpose.h"

#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

bool IsTransposeSupported(const TfLiteContext* absl_nonnull context,
                          const TfLiteNode* absl_nonnull node,
                          const TfLiteRegistration* absl_nonnull registration,
                          std::string* absl_nonnull error) {
  // Check version.
  if (registration->version > 9) {
    *error = absl::StrCat("Unsupported version: ", registration->version,
                          ", max supported version is 9");
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
  const TfLiteTensor& input_tensor = context->tensors[input_id];

  const std::vector<TfLiteType> supported_dtypes_vec = {
      kTfLiteFloat32, kTfLiteFloat16, kTfLiteInt32, kTfLiteUInt32, kTfLiteInt16,
      kTfLiteUInt16,  kTfLiteInt8,    kTfLiteUInt8, kTfLiteBool};
  absl::flat_hash_set<TfLiteType> supported_dtypes(supported_dtypes_vec.begin(),
                                                   supported_dtypes_vec.end());
  if (!CheckTensorDtype(input_tensor, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }

  const int perm_id = inputs->data[1];
  const TfLiteTensor& perm_tensor = context->tensors[perm_id];

  // Check permutation tensor is constant.
  if (!tflite::IsConstantTensor(&perm_tensor)) {
    *error = "Permutation tensor must be constant";
    return false;
  }

  // Check permutation tensor precision.
  if (perm_tensor.type != kTfLiteInt32) {
    *error = "Permutation tensor must be INT32";
    return false;
  }

  // Check permutation tensor shape and size.
  const int num_elements = tflite::NumElements(&perm_tensor);
  if (num_elements > 5 || num_elements < 2) {
    *error = absl::StrCat("Permutation for transpose is invalid. Size: ",
                          num_elements);
    return false;
  }

  return true;
}

}  // namespace litert::ml_drift::ir
