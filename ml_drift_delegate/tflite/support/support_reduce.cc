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

#include "ml_drift_delegate/tflite/support/support_reduce.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

bool IsReduceSupported(const TfLiteContext* absl_nonnull context,
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

  const int input_id = inputs->data[0];
  const int axes_id = inputs->data[1];
  const int output_id = outputs->data[0];

  // Check dtype.
  const absl::flat_hash_set<TfLiteType> supported_dtypes_b = {kTfLiteBool};
  const absl::flat_hash_set<TfLiteType> supported_dtypes_fi = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
      kTfLiteInt16,
      kTfLiteInt32
      // go/keep-sorted end
      // clang-format on
  };
  const bool is_bool_op =
      registration->builtin_code == kTfLiteBuiltinReduceAll ||
      registration->builtin_code == kTfLiteBuiltinReduceAny;
  const absl::flat_hash_set<TfLiteType> supported_dtypes =
      is_bool_op ? supported_dtypes_b : supported_dtypes_fi;

  const TfLiteTensor& input = context->tensors[input_id];
  if (!CheckTensorDtype(input, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& axes = context->tensors[axes_id];
  if (!CheckTensorDtype(axes, {kTfLiteInt32}, "inputs[1]", *error)) {
    return false;
  }

  // Check const inputs.
  if (!CheckNotConstant(input, "inputs[0]", *error)) {
    return false;
  }
  if (!CheckIsConstant(axes, "inputs[1]", *error)) {
    return false;
  }

  // Check params.
  const auto* params =
      static_cast<const TfLiteReducerParams*>(node->builtin_data);
  if (!params) {
    *error = "Missing TfLiteReducerParams.";
    return false;
  }

  // Check axes values.
  const int* axes_data = axes.data.i32;
  const int num_axes = axes.bytes / sizeof(int);
  const int num_dims = input.dims->size;
  for (int i = 0; i < num_axes; ++i) {
    int axis = ResolveNegativeIndex(axes_data[i], num_dims);
    if (axis < 0 || axis >= num_dims) {
      *error =
          absl::StrCat("Invalid axis value: ", axes_data[i],
                       ", for input tensor with ", num_dims, " dimensions.");
      return false;
    }
  }

  return true;
}

}  // namespace litert::ml_drift::ir
