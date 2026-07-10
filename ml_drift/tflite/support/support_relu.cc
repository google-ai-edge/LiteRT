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

#include "third_party/odml/litert/ml_drift/tflite/support/support_relu.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

bool IsReluSupported(const TfLiteContext* absl_nonnull context,
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
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/1,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  const TfLiteIntArray* inputs = node->inputs;
  const TfLiteIntArray* outputs = node->outputs;

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int input_id = inputs->data[0];
  const int output_id = outputs->data[0];

  // Check dtype.
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
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }

  // Check const inputs.
  if (!CheckNotConstant(input, "inputs[0]", *error)) {
    return false;
  }

  // Check params for Leaky ReLU.
  if (registration->builtin_code == kTfLiteBuiltinLeakyRelu) {
    const auto* params =
        static_cast<const TfLiteLeakyReluParams*>(node->builtin_data);
    if (!params) {
      *error = "Missing TfLiteLeakyReluParams.";
      return false;
    }
  }

  // Input and output shapes must be equal.
  if (!TfLiteIntArrayEqual(input.dims, output.dims)) {
    *error = "Input and output shapes must be equal.";
    return false;
  }

  return true;
}

}  // namespace litert::ml_drift::ir
