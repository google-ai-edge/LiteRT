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

#include "ml_drift_delegate/tflite/support/support_reshape.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

constexpr int kMaxDims = 5;

namespace {

bool CheckInt4Quantization(const TfLiteTensor& tensor, const char* tensor_name,
                           std::string& error) {
  if (tensor.type != kTfLiteInt4) {
    return true;
  }
  if (tensor.quantization.type != kTfLiteAffineQuantization ||
      tensor.quantization.params == nullptr) {
    error =
        absl::StrCat(tensor_name, " INT4 tensor requires affine quantization");
    return false;
  }
  const auto* quant =
      static_cast<const TfLiteAffineQuantization*>(tensor.quantization.params);
  if (quant->scale == nullptr || quant->scale->size != 1 ||
      quant->zero_point == nullptr || quant->zero_point->size != 1) {
    error = absl::StrCat(tensor_name,
                         " INT4 tensor requires per-tensor quantization");
    return false;
  }
  return true;
}

}  // namespace

bool IsReshapeSupported(const TfLiteContext* absl_nonnull context,
                        const TfLiteNode* absl_nonnull node,
                        const TfLiteRegistration* absl_nonnull registration,
                        std::string* absl_nonnull error) {
  // Check version.
  if (registration->version != 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check number of inputs.
  // Input 0: input tensor.
  // Input 1: shape tensor (unused, shape calculated based on output tensor).
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
      kTfLiteBool,
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt4,
      kTfLiteInt8,
      kTfLiteInt16,
      kTfLiteInt32,
      kTfLiteUInt8,
      kTfLiteUInt16,
      kTfLiteUInt32,
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
  if ((input.type == kTfLiteInt4) != (output.type == kTfLiteInt4)) {
    *error = "Mismatched INT4 input and output types";
    return false;
  }
  if (!CheckInt4Quantization(input, "inputs[0]", *error)) return false;
  if (!CheckInt4Quantization(output, "outputs[0]", *error)) return false;
  // Check const inputs.
  if (!CheckNotConstant(input, "inputs[0]", *error)) {
    return false;
  }
  // Check dims. Should <= kMaxDims.
  if (!CheckTensorDims(input, /*min_dims=*/0, /*max_dims=*/kMaxDims,
                       "inputs[0]", *error)) {
    return false;
  }
  if (!CheckTensorDims(output, /*min_dims=*/0, /*max_dims=*/kMaxDims,
                       "outputs[0]", *error)) {
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
