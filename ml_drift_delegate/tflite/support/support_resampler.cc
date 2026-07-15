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

#include "ml_drift_delegate/tflite/support/support_resampler.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

bool IsResamplerSupported(const TfLiteContext* absl_nonnull context,
                          const TfLiteNode* absl_nonnull node,
                          const TfLiteRegistration* absl_nonnull registration,
                          std::string* absl_nonnull error) {
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

  const int input0_id = inputs->data[0];
  const int input1_id = inputs->data[1];  // Warp.
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
  const TfLiteTensor& input0 = context->tensors[input0_id];
  if (!CheckTensorDtype(input0, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& input1 = context->tensors[input1_id];
  if (!CheckTensorDtype(input1, supported_dtypes, "inputs[1]", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }

  // Check const inputs.
  if (!CheckNotConstant(input0, "inputs[0]", *error)) {
    return false;
  }
  if (!CheckNotConstant(input1, "inputs[1]", *error)) {
    return false;
  }

  // Check dims. Sizes should match.
  if (input0.dims->size != input1.dims->size) {
    *error = absl::StrCat("Input 0 and input 1 dimensions must match: ",
                          input0.dims->size, " != ", input1.dims->size);
    return false;
  }
  if (input0.dims->size != output.dims->size) {
    *error = absl::StrCat("Input 0 and output dimensions must match: ",
                          input0.dims->size, " != ", output.dims->size);
    return false;
  }

  // Input 1 and output should match except for the channel dimension.
  for (int i = 0; i < input1.dims->size - 1; ++i) {
    if (input1.dims->data[i] != output.dims->data[i]) {
      *error = absl::StrCat("Input 1 and output dimensions must match: ",
                            input1.dims->data[i], " != ", output.dims->data[i]);
      return false;
    }
  }

  // The channel dimension should be 2 for input 1 (warp).
  if (input1.dims->data[input1.dims->size - 1] != 2) {
    *error = absl::StrCat("Input 1 channel dimension must be 2: ",
                          input1.dims->data[input1.dims->size - 1], " != 2");
    return false;
  }

  return true;
}

}  // namespace litert::ml_drift::ir
