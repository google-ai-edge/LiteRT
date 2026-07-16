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

#include "ml_drift_delegate/tflite/support/support_resize2d.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

bool IsResize2DSupported(const TfLiteContext* absl_nonnull context,
                         const TfLiteNode* absl_nonnull node,
                         const TfLiteRegistration* absl_nonnull registration,
                         std::string* absl_nonnull error) {
  // Check version.
  if (registration->version < 1 || registration->version > 3) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check number of inputs.
  // Input 0: input tensor.
  // Input 1: size tensor (unused, size calculated based on output tensor).
  const TfLiteIntArray* inputs = node->inputs;
  if (inputs->size < 1 || inputs->size > 2) {
    *error = absl::StrCat("Invalid number of inputs: ", inputs->size);
    return false;
  }
  // Check number of outputs.
  const TfLiteIntArray* outputs = node->outputs;
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/inputs->size,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

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
  // Check const inputs.
  if (!CheckNotConstant(input, "inputs[0]", *error)) {
    return false;
  }
  // Check dims. Should <= 4.
  if (!CheckTensorDims(input, /*min_dims=*/0, /*max_dims=*/4, "inputs[0]",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(output, /*min_dims=*/0, /*max_dims=*/4, "outputs[0]",
                       *error)) {
    return false;
  }
  // Check params.
  const int op_code = registration->builtin_code;
  if (op_code == kTfLiteBuiltinResizeBilinear) {
    const auto* params =
        static_cast<const TfLiteResizeBilinearParams*>(node->builtin_data);
    if (!params) {
      *error = "Invalid params.";
      return false;
    }
    if (params->align_corners && params->half_pixel_centers) {
      *error = "If half_pixel_centers is True, align_corners must be False.";
      return false;
    }
  } else if (op_code == kTfLiteBuiltinResizeNearestNeighbor) {
    const auto* params = static_cast<const TfLiteResizeNearestNeighborParams*>(
        node->builtin_data);
    if (!params) {
      *error = "Invalid params.";
      return false;
    }
  } else {
    *error = absl::StrCat("Unsupported op code: ", op_code);
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
