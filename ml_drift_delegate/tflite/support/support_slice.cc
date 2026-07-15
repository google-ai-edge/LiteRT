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

#include "ml_drift_delegate/tflite/support/support_slice.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

bool IsSliceSupported(const TfLiteContext* absl_nonnull context,
                      const TfLiteNode* absl_nonnull node,
                      const TfLiteRegistration* absl_nonnull registration,
                      std::string* absl_nonnull error) {
  // Check version.
  if (registration->version > 8) {
    absl::StrAppend(error, "Unsupported version: ", registration->version);
    return false;
  }

  // Check inputs and outputs count
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/3,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  // Get and validate tensors.
  if (!ValidateTensorIds(*context, *node->inputs, "inputs", *error)) {
    return false;
  }
  if (!ValidateTensorIds(*context, *node->outputs, "outputs", *error)) {
    return false;
  }

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& begin = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& size = context->tensors[node->inputs->data[2]];

  if (!CheckNotConstant(input, "inputs[0]", *error)) {
    return false;
  }
  if (!CheckIsConstant(begin, "inputs[1]", *error)) {
    return false;
  }
  absl::Status status =
      CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>(
          &begin);
  if (!status.ok()) {
    *error = status.message();
    return false;
  }
  if (!CheckIsConstant(size, "inputs[2]", *error)) {
    return false;
  }
  status = CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>(
      &size);
  if (!status.ok()) {
    *error = status.message();
    return false;
  }
  if (!CheckTensorDims(input, /*min_dims=*/0, /*max_dims=*/5, "inputs[0]",
                       *error)) {
    return false;
  }

  if (!CheckTensorDims(begin, /*min_dims=*/1, /*max_dims=*/1, "inputs[1]",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(size, /*min_dims=*/1, /*max_dims=*/1, "inputs[2]",
                       *error)) {
    return false;
  }
  if (begin.dims->data[0] != size.dims->data[0]) {
    absl::StrAppend(
        error, "Begin and size tensors must have the same number of elements");
    return false;
  }
  if (begin.dims->data[0] != input.dims->size) {
    absl::StrAppend(error,
                    "Begin tensor elements must be equal to input tensor rank");
    return false;
  }

  // Check for out-of-bounds slice.
  for (int i = 0; i < input.dims->size; ++i) {
    if (begin.data.i32[i] + size.data.i32[i] > input.dims->data[i]) {
      absl::StrAppend(error, "Slice is out of bounds for dimension ", i);
      return false;
    }
  }

  return true;
}

}  // namespace litert::ml_drift::ir
