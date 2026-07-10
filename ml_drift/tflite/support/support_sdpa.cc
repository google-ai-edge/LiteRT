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

#include "third_party/odml/litert/ml_drift/tflite/support/support_sdpa.h"

#include <optional>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

using ::tflite::IsConstantTensor;

bool IsSdpaSupported(const TfLiteContext* absl_nonnull context,
                     const TfLiteNode* absl_nonnull node,
                     const TfLiteRegistration* absl_nonnull registration,
                     std::string* absl_nonnull error) {
  // Check version.
  if (registration->version != 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check number of inputs.
  const TfLiteIntArray* inputs = node->inputs;
  if (inputs->size < 3 || inputs->size > 4) {
    *error = absl::StrCat("Invalid number of inputs: ", inputs->size,
                          ", should be either 3 or 4");
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

  const int q_id = inputs->data[0];
  const TfLiteTensor& q = context->tensors[q_id];

  const int k_id = inputs->data[1];
  const TfLiteTensor& k = context->tensors[k_id];

  const int v_id = inputs->data[2];
  const TfLiteTensor& v = context->tensors[v_id];

  std::optional<int> mask_id;
  if (inputs->size == 4) {
    mask_id = inputs->data[3];
  }
  const TfLiteTensor* mask =
      mask_id.has_value() ? &context->tensors[*mask_id] : nullptr;

  const int output_id = outputs->data[0];
  const TfLiteTensor& output = context->tensors[output_id];

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
  if (!CheckTensorDtype(q, supported_dtypes, "q", *error)) return false;
  if (!CheckTensorDtype(k, supported_dtypes, "k", *error)) return false;
  if (!CheckTensorDtype(v, supported_dtypes, "v", *error)) return false;
  if (mask != nullptr &&
      !CheckTensorDtype(*mask, supported_dtypes, "mask", *error)) {
    return false;
  }
  if (!CheckTensorDtype(output, supported_dtypes, "output", *error)) {
    return false;
  }

  // Check const inputs.
  if (IsConstantTensor(&q) && IsConstantTensor(&k) && IsConstantTensor(&v)) {
    *error = "All inputs are constant.";
    return false;
  }

  // Check dims.
  if (!CheckTensorDims(q, /*min_dims=*/1, /*max_dims=*/4, "q", *error)) {
    return false;
  }
  if (!CheckTensorDims(k, /*min_dims=*/1, /*max_dims=*/4, "k", *error)) {
    return false;
  }
  if (!CheckTensorDims(v, /*min_dims=*/1, /*max_dims=*/4, "v", *error)) {
    return false;
  }
  if (mask != nullptr &&
      !CheckTensorDims(*mask, /*min_dims=*/1, /*max_dims=*/4, "mask", *error)) {
    return false;
  }
  if (!CheckTensorDims(output, /*min_dims=*/1, /*max_dims=*/4, "output",
                       *error)) {
    return false;
  }

  const int q_last_dim = q.dims->size > 0 ? q.dims->data[q.dims->size - 1] : 1;
  const int k_last_dim = k.dims->size > 0 ? k.dims->data[k.dims->size - 1] : 1;
  const int v_last_dim = v.dims->size > 0 ? v.dims->data[v.dims->size - 1] : 1;
  if (q_last_dim != k_last_dim || q_last_dim != v_last_dim) {
    *error = "Input channels mismatch.";
    return false;
  }
  if (!TfLiteIntArrayEqual(k.dims, v.dims)) {
    *error = "K and V must have the same shape.";
    return false;
  }
  if (mask != nullptr) {
    const int k_height = k.dims->size > 2 ? k.dims->data[k.dims->size - 3] : 1;
    const int mask_channels =
        mask->dims->size > 0 ? mask->dims->data[mask->dims->size - 1] : 1;
    if (k_height != mask_channels) {
      *error = absl::StrCat(
          "Mask channels must match K tensor's height, instead got ",
          mask_channels, " != ", k_height);
      return false;
    }
  }
  return true;
}

}  // namespace litert::ml_drift::ir
