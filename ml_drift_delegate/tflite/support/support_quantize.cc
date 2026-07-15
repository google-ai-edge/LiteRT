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

#include "ml_drift_delegate/tflite/support/support_quantize.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

bool IsQuantizeSupported(const TfLiteContext* absl_nonnull context,
                         const TfLiteNode* absl_nonnull node,
                         const TfLiteRegistration* absl_nonnull registration,
                         std::string* absl_nonnull error) {
  if (registration->version > 2) {
    *error = "Unsupported version.";
    return false;
  }
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/1,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  const int input_id = node->inputs->data[0];
  const int output_id = node->outputs->data[0];
  if (!ValidateTensorId(*context, input_id, "input", *error)) {
    return false;
  }
  if (!ValidateTensorId(*context, output_id, "output", *error)) {
    return false;
  }

  const TfLiteTensor* input = context->tensors + input_id;
  if (!CheckTensorDtype(*input, {kTfLiteFloat16, kTfLiteFloat32}, "input",
                        *error)) {
    return false;
  }

  const TfLiteTensor* output = context->tensors + output_id;
  if (!CheckTensorDtype(
          *output,
          {kTfLiteInt8, kTfLiteUInt8, kTfLiteInt4, kTfLiteUInt4, kTfLiteInt2},
          "output", *error)) {
    return false;
  }

  if (output->quantization.params == nullptr) {
    *error = "Encountered Quantize output with no quant params";
    return false;
  }

  return true;
}

}  // namespace litert::ml_drift::ir
