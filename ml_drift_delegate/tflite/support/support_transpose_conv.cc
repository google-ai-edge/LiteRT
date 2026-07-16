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

#include "ml_drift_delegate/tflite/support/support_transpose_conv.h"

#include <optional>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"

using ::tflite::IsConstantTensor;

namespace litert::ml_drift::ir {

constexpr int kMaxDims = 4;

bool IsTransposeConvSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    std::string* absl_nonnull error) {
  // We have two TransposeConv ops.
  // The differences are the input order and param storage location.
  // Builtin input order: {output shape(i32), weights, input, bias (optional)}
  // Custom input order: {input, weights, bias (optional)}

  // Check number of inputs.
  const bool builtin_op =
      registration->builtin_code == kTfLiteBuiltinTransposeConv;
  if (builtin_op) {
    if (!CheckInputOutputCounts(*node, /*expected_inputs=*/3,
                                /*expected_outputs=*/1, *error) &&
        !CheckInputOutputCounts(*node, /*expected_inputs=*/4,
                                /*expected_outputs=*/1, *error)) {
      return false;
    }
  } else {  // custom
    if (!CheckInputOutputCounts(*node, /*expected_inputs=*/2,
                                /*expected_outputs=*/1, *error) &&
        !CheckInputOutputCounts(*node, /*expected_inputs=*/3,
                                /*expected_outputs=*/1, *error)) {
      return false;
    }
  }

  // Validate tensor IDs.
  const TfLiteIntArray* inputs = node->inputs;
  const TfLiteIntArray* outputs = node->outputs;

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int input_id = builtin_op ? inputs->data[2] : inputs->data[0];
  const int weights_id = inputs->data[1];
  std::optional<int> bias_id = std::nullopt;
  if (inputs->size == 4) {  // must be builtin op
    bias_id = inputs->data[3];
  } else if (!builtin_op && inputs->size == 3) {
    bias_id = inputs->data[2];
  }
  std::optional<int> output_shape_id = std::nullopt;
  if (builtin_op) {
    output_shape_id = inputs->data[0];
  }
  const int output_id = outputs->data[0];

  // Check dtype.
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
      kTfLiteUInt8,
      // go/keep-sorted end
      // clang-format on
  };
  const TfLiteTensor& input = context->tensors[input_id];
  if (!CheckTensorDtype(input, supported_dtypes, "input", *error)) {
    return false;
  }
  const absl::flat_hash_set<TfLiteType> supported_weights_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
      kTfLiteUInt8,
      // go/keep-sorted end
      // clang-format on
  };
  const TfLiteTensor& weights = context->tensors[weights_id];
  if (!CheckTensorDtype(weights, supported_weights_dtypes, "weights", *error)) {
    return false;
  }
  if (IsConstantTensor(&weights)) {
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>(
            &weights);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  }
  const TfLiteTensor* bias =
      bias_id.has_value() ? &context->tensors[bias_id.value()] : nullptr;
  if (bias_id.has_value()) {
    const absl::flat_hash_set<TfLiteType> supported_bias_dtypes = {
        // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteFloat16,
      kTfLiteFloat32,
      // go/keep-sorted end
        // clang-format on
    };
    if (!CheckTensorDtype(*bias, supported_bias_dtypes, "bias", *error)) {
      return false;
    }
  }
  const TfLiteTensor* output_shape =
      output_shape_id.has_value() ? &context->tensors[output_shape_id.value()]
                                  : nullptr;
  if (output_shape) {
    if (!CheckTensorDtype(*output_shape, {kTfLiteInt32}, "output_shape",
                          *error)) {
      return false;
    }
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "output", *error)) {
    return false;
  }

  // Check const inputs.
  if (IsConstantTensor(&input) && IsConstantTensor(&weights) &&
      ((bias && IsConstantTensor(bias)) || !bias)) {
    absl::StrAppend(error, "Invalid constant inputs: ", node->inputs->data[0],
                    ", ", node->inputs->data[1]);
    if (bias) absl::StrAppend(error, ", ", node->inputs->data[2]);
    return false;
  }

  // Check number of dims.
  if (!CheckTensorDims(input, /*min_dims=*/2, /*max_dims=*/kMaxDims, "input",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(weights, /*min_dims=*/2, /*max_dims=*/kMaxDims,
                       "weights", *error)) {
    return false;
  }
  if (bias && !CheckTensorDims(*bias, /*min_dims=*/2, /*max_dims=*/kMaxDims,
                               "bias", *error)) {
    return false;
  }
  if (!CheckTensorDims(output, /*min_dims=*/2, /*max_dims=*/kMaxDims, "output",
                       *error)) {
    return false;
  }
  // Check output shape.
  if (output_shape_id.has_value()) {
    if (output_shape->dims->size != 1) {
      *error = "Output shape must be a 1D tensor.";
      return false;
    }
    if (output_shape->dims->data[0] != output.dims->size) {
      *error =
          "Passed output shape must have same number of elements as output "
          "dimensions.";
      return false;
    }
    for (int i = 0; i < output.dims->size; ++i) {
      if (output_shape->data.i32[i] != output.dims->data[i]) {
        *error = "Passed output shape must be equal to output dimensions";
        return false;
      }
    }
  }
  return true;
}

}  // namespace litert::ml_drift::ir
