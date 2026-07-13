// Copyright 2026 Google LLC.
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

#include "third_party/odml/litert/ml_drift/tflite/support/support_tile.h"

#include <cstdint>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/c/common.h"

namespace litert::ml_drift::ir {

namespace {

constexpr int kMinVersion = 1;
constexpr int kMaxVersion = 2;
constexpr int kMaxDims = 5;

const absl::flat_hash_set<TfLiteType>& SupportedDtypes() {
  static const auto* const kSupportedDtypes =
      new absl::flat_hash_set<TfLiteType>({
          kTfLiteFloat32,
          kTfLiteFloat16,
          kTfLiteInt32,
          kTfLiteInt8,
          kTfLiteUInt8,
          kTfLiteBool,
      });
  return *kSupportedDtypes;
}

}  // namespace

bool IsTileSupported(const TfLiteContext* absl_nonnull context,
                     const TfLiteNode* absl_nonnull node,
                     const TfLiteRegistration* absl_nonnull registration,
                     std::string* absl_nonnull error) {
  if (registration->version < kMinVersion ||
      registration->version > kMaxVersion) {
    *error = "Unsupported version.";
    return false;
  }

  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/2,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  if (!ValidateTensorIds(*context, *node->inputs, "inputs", *error) ||
      !ValidateTensorIds(*context, *node->outputs, "outputs", *error)) {
    return false;
  }

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& multiples = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  if (!CheckNotConstant(input, "input", *error)) {
    return false;
  }

  if (!CheckIsConstant(multiples, "multiples", *error)) {
    return false;
  }

  if (!CheckTensorDtype(input, SupportedDtypes(), "input", *error) ||
      !CheckTensorDtype(output, SupportedDtypes(), "output", *error)) {
    return false;
  }

  if (multiples.type != kTfLiteInt32) {
    *error = "multiples must be kTfLiteInt32.";
    return false;
  }

  if (!CheckTensorDims(input, 1, kMaxDims, "input", *error) ||
      !CheckTensorDims(output, 1, kMaxDims, "output", *error)) {
    return false;
  }

  if (!CheckTensorDims(multiples, 1, 1, "multiples", *error)) {
    return false;
  }

  if (multiples.dims->size != 1 ||
      multiples.dims->data[0] != input.dims->size) {
    *error = "multiples length must match input rank.";
    return false;
  }

  if (multiples.data.i32 == nullptr) {
    *error = "multiples data is null.";
    return false;
  }

  // Validate output shape matches input_shape * multiples if possible
  const int32_t* multiples_data = multiples.data.i32;
  for (int i = 0; i < input.dims->size; ++i) {
    if (output.dims->data[i] != input.dims->data[i] * multiples_data[i]) {
      *error = "output shape does not match (input_shape * multiples).";
      return false;
    }
  }

  return true;
}

}  // namespace litert::ml_drift::ir
