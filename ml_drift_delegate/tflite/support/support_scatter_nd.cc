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

#include "ml_drift_delegate/tflite/support/support_scatter_nd.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

bool IsScatterNdSupported(const TfLiteContext* absl_nonnull context,
                          const TfLiteNode* absl_nonnull node,
                          const TfLiteRegistration* absl_nonnull registration,
                          std::string* absl_nonnull error) {
  if (registration->version != 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/3,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }
  if (!ValidateTensorIds(*context, *node->inputs, "inputs", *error)) {
    return false;
  }
  if (!ValidateTensorIds(*context, *node->outputs, "outputs", *error)) {
    return false;
  }
  const TfLiteTensor& indices = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& updates = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& shape = context->tensors[node->inputs->data[2]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  if (!CheckTensorDtype(indices, {kTfLiteInt32}, "inputs[0]", *error) ||
      !CheckTensorDtype(updates, {kTfLiteFloat32}, "inputs[1]", *error) ||
      !CheckTensorDtype(shape, {kTfLiteInt32}, "inputs[2]", *error) ||
      !CheckTensorDtype(output, {kTfLiteFloat32}, "outputs[0]", *error)) {
    return false;
  }
  if (!CheckNotConstant(indices, "inputs[0]", *error) ||
      !CheckNotConstant(updates, "inputs[1]", *error) ||
      !CheckIsConstant(shape, "inputs[2]", *error)) {
    return false;
  }
  if (indices.dims->size != 2 || indices.dims->data[1] != 1) {
    *error = "ScatterND indices must have shape [N, 1].";
    return false;
  }
  if (updates.dims->size != 4 || output.dims->size != 4) {
    *error = "ScatterND updates and output must be 4D.";
    return false;
  }
  if (updates.dims->data[0] != indices.dims->data[0]) {
    *error = "ScatterND updates and indices must have the same first dim.";
    return false;
  }
  for (int i = 1; i < 4; ++i) {
    if (updates.dims->data[i] != output.dims->data[i]) {
      *error = "ScatterND updates and output must match on the last 3 dims.";
      return false;
    }
  }
  return true;
}

}  // namespace litert::ml_drift::ir
