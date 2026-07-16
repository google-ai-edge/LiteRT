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

#include "ml_drift_delegate/tflite/support/support_strided_slice.h"

#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"

namespace litert::ml_drift::ir {

bool IsStridedSliceSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    std::string* absl_nonnull error) {
  // Check version.
  if (registration->version > 2) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check inputs count.
  if (node->inputs->size < 4) {
    absl::StrAppend(error, "Requires at least 4 inputs, got ",
                    node->inputs->size);
    return false;
  }

  // Check parameters.
  const auto* params =
      static_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
  if (!params) {
    absl::StrAppend(error, "Missing TfLiteStridedSliceParams");
    return false;
  }
  if (params->ellipsis_mask) {
    absl::StrAppend(error, "ellipsis_mask is not supported");
    return false;
  }
  if (params->new_axis_mask) {
    absl::StrAppend(error, "new_axis_mask is not supported");
    return false;
  }
  if (params->shrink_axis_mask) {
    absl::StrAppend(error, "shrink_axis_mask are not supported");
    return false;
  }

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *node->inputs, "inputs", *error)) {
    return false;
  }
  if (!ValidateTensorIds(*context, *node->outputs, "outputs", *error)) {
    return false;
  }

  // Get and validate tensors.
  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& begin = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& end = context->tensors[node->inputs->data[2]];
  const TfLiteTensor& strides = context->tensors[node->inputs->data[3]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  if (!CheckNotConstant(input, "inputs[0]", *error)) return false;
  if (!CheckIsConstant(begin, "inputs[1]", *error)) return false;
  absl::Status status =
      CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>(
          &begin);
  if (!status.ok()) {
    *error = status.message();
    return false;
  }
  if (!CheckIsConstant(end, "inputs[2]", *error)) return false;
  status = CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>(
      &end);
  if (!status.ok()) {
    *error = status.message();
    return false;
  }
  if (!CheckIsConstant(strides, "inputs[3]", *error)) return false;
  status = CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::INT32>(
      &strides);
  if (!status.ok()) {
    *error = status.message();
    return false;
  }
  if (!CheckTensorDims(input, /*min_dims=*/0, /*max_dims=*/4, "inputs[0]",
                       *error)) {
    return false;
  }

  if (!CheckTensorDtype(begin, {kTfLiteInt32}, "inputs[1]", *error)) {
    return false;
  }
  if (!CheckTensorDtype(end, {kTfLiteInt32}, "inputs[2]", *error)) return false;
  if (!CheckTensorDtype(strides, {kTfLiteInt32}, "inputs[3]", *error)) {
    return false;
  }

  // Check that slice parameter tensors are 1D and have the same size.
  if (!CheckTensorDims(begin, /*min_dims=*/1, /*max_dims=*/1, "inputs[1]",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(end, /*min_dims=*/1, /*max_dims=*/1, "inputs[2]",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(strides, /*min_dims=*/1, /*max_dims=*/1, "inputs[3]",
                       *error)) {
    return false;
  }
  if (begin.dims->data[0] != end.dims->data[0] ||
      begin.dims->data[0] != strides.dims->data[0]) {
    absl::StrAppend(error,
                    "Begin, end, and strides tensors must have the same "
                    "number of elements");
    return false;
  }

  // Read slice parameter tensors.
  std::vector<int> starts(begin.data.i32, begin.data.i32 + begin.dims->data[0]);
  std::vector<int> ends(end.data.i32, end.data.i32 + end.dims->data[0]);
  std::vector<int> strides_vec(strides.data.i32,
                               strides.data.i32 + strides.dims->data[0]);

  // Dimension check with implicit batch handling.
  int begin_mask = params->begin_mask;
  int end_mask = params->end_mask;
  const int input_rank = input.dims->size;
  const int params_rank = starts.size();
  if (params_rank == input_rank - 1) {
    // If the batch dimension is missing, pad the parameters.
    starts.insert(starts.begin(), 0);
    ends.insert(ends.begin(), input.dims->data[0]);
    strides_vec.insert(strides_vec.begin(), 1);
    // Shift masks to align with the now-padded dimensions.
    begin_mask <<= 1;
    end_mask <<= 1;
  } else if (params_rank != input_rank) {
    absl::StrAppend(error,
                    "Slice parameter rank must match input rank or be one "
                    "less (for implicit batch)");
    return false;
  }

  // Apply masks and resolve negative indices.
  ResolveNegativeIndices(*input.dims, starts);
  ResolveNegativeIndices(*input.dims, ends);
  UpdateWithMask(begin_mask, end_mask, *input.dims, starts, ends);

  // Validate strides and calculate expected output shape.
  std::vector<int> expected_shape;
  for (int i = 0; i < input.dims->size; ++i) {
    if (strides_vec[i] <= 0) {
      absl::StrAppend(error, "Strides must be positive. Negative strides (",
                      strides_vec[i], ") are not supported.");
      return false;
    }
    expected_shape.push_back(
        ::ml_drift::DivideRoundUp(ends[i] - starts[i], strides_vec[i]));
  }

  // Compare with actual output shape.
  if (output.dims->size != expected_shape.size()) {
    absl::StrAppend(error, "Output tensor rank mismatch");
    return false;
  }
  for (int i = 0; i < output.dims->size; ++i) {
    if (output.dims->data[i] != expected_shape[i]) {
      absl::StrAppend(error, "Output shape mismatch at dimension ", i,
                      ". Expected ", expected_shape[i], ", got ",
                      output.dims->data[i]);
      return false;
    }
  }

  return true;
}

}  // namespace litert::ml_drift::ir
