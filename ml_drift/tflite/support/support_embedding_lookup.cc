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

#include "third_party/odml/litert/ml_drift/tflite/support/support_embedding_lookup.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

using ::tflite::IsConstantTensor;

bool IsEmbeddingLookupSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    std::string* absl_nonnull error) {
  // Check version.
  if (registration->version != 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }
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

  const int src_id = inputs->data[0];
  const int weights_id = inputs->data[1];
  const int output_id = outputs->data[0];

  const TfLiteTensor& input = context->tensors[src_id];
  if (!CheckTensorDtype(input, {kTfLiteFloat16, kTfLiteFloat32, kTfLiteInt32},
                        "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& weights = context->tensors[weights_id];
  const absl::flat_hash_set<TfLiteType> supported_weights_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteFloat32,
      kTfLiteInt2,
      kTfLiteInt4,
      kTfLiteInt8,
      // go/keep-sorted end
      // clang-format on
  };
  if (!CheckTensorDtype(weights, supported_weights_dtypes, "inputs[1]",
                        *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, {kTfLiteFloat16, kTfLiteFloat32}, "outputs[0]",
                        *error)) {
    return false;
  }
  // Check input dims.
  if (!CheckTensorDims(input, /*min_dims=*/1, /*max_dims=*/4, "inputs[0]",
                       *error)) {
    return false;
  }
  // Check weights
  switch (weights.type) {
    case kTfLiteInt2:
    case kTfLiteInt4:
    case kTfLiteInt8:
      if (weights.quantization.params == nullptr) {
        *error = "Quantization params are null for weights";
        return false;
      }
      if (weights.quantization.type != kTfLiteAffineQuantization) {
        *error = "Unsupported quantization type for weights: " +
                 std::to_string(weights.quantization.type);
        return false;
      }
      if (weights.dims->size == 4) {
        if (weights.dims->data[1] != 1 || weights.dims->data[2] != 1) {
          *error = absl::StrCat(
              "Unsupported HW weight dimensions for quantized "
              "embedding lookup: ",
              weights.dims->data[1], " , ", weights.dims->data[2]);
          return false;
        }
      } else if (weights.dims->size != 2) {
        *error = absl::StrCat("Unsupported ", weights.dims->size,
                              "D weights for quantized embedding lookup");
        return false;
      }
      break;
    case kTfLiteFloat32:
      if (!CheckTensorDims(weights, /*min_dims=*/1, /*max_dims=*/4, "inputs[1]",
                           *error)) {
        return false;
      }
      break;
    default:
      *error = absl::StrCat("Unsupported weights type: ", weights.type);
      return false;
  }
  // Check output dims.
  if (!CheckTensorDims(output, /*min_dims=*/1, /*max_dims=*/4, "outputs[0]",
                       *error)) {
    return false;
  }
  // Check const inputs.
  if (IsConstantTensor(&input) && IsConstantTensor(&weights)) {
    *error =
        absl::StrCat("Invalid constant inputs: ", src_id, ", ", weights_id);
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
