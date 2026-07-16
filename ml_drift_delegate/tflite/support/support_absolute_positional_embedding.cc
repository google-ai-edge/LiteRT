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

#include "ml_drift_delegate/tflite/support/support_absolute_positional_embedding.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

constexpr int kMaxDims = 4;

bool IsAbsolutePositionalEmbeddingSupported(
    const TfLiteContext* absl_nonnull context,
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

  // Check dtype.
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
      kTfLiteInt16,
      kTfLiteInt32,
      kTfLiteUInt8,
      kTfLiteUInt16,
      kTfLiteUInt32,
      // go/keep-sorted end
      // clang-format on
  };
  for (int i = 0; i < inputs->size; ++i) {
    const TfLiteTensor& input = context->tensors[inputs->data[i]];
    if (!CheckTensorDtype(input, supported_dtypes,
                          absl::StrCat("inputs[", i, "]"), *error)) {
      return false;
    }
  }
  for (int i = 0; i < outputs->size; ++i) {
    const TfLiteTensor& output = context->tensors[outputs->data[i]];
    if (!CheckTensorDtype(output, supported_dtypes,
                          absl::StrCat("outputs[", i, "]"), *error)) {
      return false;
    }
  }

  // Check dims.
  for (int i = 0; i < inputs->size; ++i) {
    const TfLiteTensor& input = context->tensors[inputs->data[i]];
    if (!CheckTensorDims(input, /*min_dims=*/0, /*max_dims=*/kMaxDims,
                         absl::StrCat("inputs[", i, "]"), *error)) {
      return false;
    }
  }

  const TfLiteTensor* src = context->tensors + inputs->data[0];
  const TfLiteTensor* pos = context->tensors + inputs->data[1];

  auto get_width = [](const TfLiteTensor* t) {
    return t->dims->size == 4 ? t->dims->data[2] :
           t->dims->size == 3 ? t->dims->data[1] : 1;
  };

  if (get_width(src) != get_width(pos)) {
    *error = "src and pos must have the same width for PositionalEmbedding.";
    return false;
  }

  return true;
}

}  // namespace litert::ml_drift::ir
