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

#include "ml_drift_delegate/tflite/support/support_pooling2d.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

bool IsPooling2dSupported(const TfLiteContext* absl_nonnull context,
                          const TfLiteNode* absl_nonnull node,
                          const TfLiteRegistration* absl_nonnull registration,
                          std::string* absl_nonnull error) {
  // Check version.
  if (registration->version < 1 || registration->version > 2) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check params and expected outputs.
  const TfLitePoolParams* params = nullptr;
  int expected_outputs = 1;
  if (registration->builtin_code == kTfLiteBuiltinCustom) {
    if (node->outputs->size != 2) {
      *error = "Custom pooling op must have 2 outputs.";
      return false;
    }
    if (node->custom_initial_data == nullptr) {
      *error = "Custom pooling op must have custom initial data.";
      return false;
    }
    params = static_cast<const TfLitePoolParams*>(node->custom_initial_data);
    expected_outputs = 2;
  } else {
    if (node->outputs->size != 1) {
      *error = "Builtin pooling op must have exactly 1 output.";
      return false;
    }
    params = static_cast<const TfLitePoolParams*>(node->builtin_data);
  }

  if (!params) {
    *error = "Params should not be null.";
    return false;
  }
  if (params->filter_height <= 0 || params->filter_width <= 0) {
    *error = absl::StrCat("Unsupported filter size: ", params->filter_height,
                          "x", params->filter_width);
    return false;
  }
  if (params->stride_height <= 0 || params->stride_width <= 0) {
    *error = absl::StrCat("Unsupported stride size: ", params->stride_height,
                          "x", params->stride_width);
    return false;
  }

  // Check number of inputs and outputs.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/1, expected_outputs,
                              *error)) {
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
      kTfLiteBFloat16,
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

  // Check rank.
  if (!CheckTensorDims(input, /*min_dims=*/3, /*max_dims=*/4, "inputs[0]",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(output, /*min_dims=*/3, /*max_dims=*/4, "outputs[0]",
                       *error)) {
    return false;
  }
  if (expected_outputs == 2) {
    const TfLiteTensor& output1 = context->tensors[outputs->data[1]];
    if (!CheckTensorDtype(output1, supported_dtypes, "outputs[1]", *error)) {
      return false;
    }
    if (!CheckTensorDims(output1, /*min_dims=*/3, /*max_dims=*/4, "outputs[1]",
                         *error)) {
      return false;
    }
  }

  // Check const inputs.
  if (!CheckNotConstant(input, "inputs[0]", *error)) {
    return false;
  }

  // Check fused activation.
  if (!CheckFusedActivationSkipSize(params->activation).ok()) {
    *error = absl::StrCat("Unsupported fused activation: ", params->activation);
    return false;
  }

  return true;
}

}  // namespace litert::ml_drift::ir
