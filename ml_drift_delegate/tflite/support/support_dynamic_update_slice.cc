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

#include "ml_drift_delegate/tflite/support/support_dynamic_update_slice.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

bool IsDynamicUpdateSliceSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    std::string* absl_nonnull error) {
  // Check version
  if (registration->version > 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check inputs and outputs count
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/3,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  const TfLiteTensor* operand = context->tensors + node->inputs->data[0];
  const TfLiteTensor* update = context->tensors + node->inputs->data[1];
  const TfLiteTensor* start_indices = context->tensors + node->inputs->data[2];
  const TfLiteTensor* output = context->tensors + node->outputs->data[0];

  if (!CheckNotConstant(*start_indices, "start_indices", *error)) {
    return false;
  }

  if (!CheckTensorDims(*operand, /*min_dims=*/0, /*max_dims=*/4, "operand",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(*update, /*min_dims=*/0, /*max_dims=*/4, "update",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(*start_indices, /*min_dims=*/0, /*max_dims=*/4,
                       "start_indices", *error)) {
    return false;
  }
  if (!CheckTensorDims(*output, /*min_dims=*/0, /*max_dims=*/4, "output",
                       *error)) {
    return false;
  }

  // The rank of update must be less than or equal to the rank of operand.
  if (update->dims->size > operand->dims->size) {
    absl::StrAppend(
        error,
        "Update tensor rank cannot be greater than operand tensor rank.");
    return false;
  }

  // Compare shapes from the trailing dimensions.
  for (int i = 1; i <= update->dims->size; ++i) {
    int update_dim = update->dims->data[update->dims->size - i];
    int operand_dim = operand->dims->data[operand->dims->size - i];
    if (update_dim > operand_dim) {
      absl::StrAppend(
          error,
          "Update shape must be less than or equal to the operand shape.");
      return false;
    }
  }

  return true;
}

}  // namespace litert::ml_drift::ir
