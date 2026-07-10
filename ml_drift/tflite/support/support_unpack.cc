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

#include "third_party/odml/litert/ml_drift/tflite/support/support_unpack.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/c/builtin_op_data.h"
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

bool IsUnpackSupported(const TfLiteContext* absl_nonnull context,
                       const TfLiteNode* absl_nonnull node,
                       const TfLiteRegistration* absl_nonnull registration,
                       std::string* absl_nonnull error) {
  if (registration->version < kMinVersion ||
      registration->version > kMaxVersion) {
    *error = "Unsupported version.";
    return false;
  }

  const auto* params =
      static_cast<const TfLiteUnpackParams*>(node->builtin_data);
  if (params == nullptr) {
    *error = "Missing TfLiteUnpackParams.";
    return false;
  }

  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/1,
                              /*expected_outputs=*/params->num, *error)) {
    return false;
  }

  if (!ValidateTensorIds(*context, *node->inputs, "inputs", *error) ||
      !ValidateTensorIds(*context, *node->outputs, "outputs", *error)) {
    return false;
  }

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];

  if (!CheckNotConstant(input, "input", *error)) {
    return false;
  }

  if (!CheckTensorDtype(input, SupportedDtypes(), "input", *error)) {
    return false;
  }

  if (!CheckTensorDims(input, 1, kMaxDims, "input", *error)) {
    return false;
  }

  int axis = ResolveNegativeIndex(params->axis, input.dims->size);
  if (axis < 0 || axis >= input.dims->size) {
    *error = "Invalid axis.";
    return false;
  }

  if (input.dims->data[axis] != params->num) {
    *error = "input.dims[axis] must match params->num.";
    return false;
  }

  for (int i = 0; i < node->outputs->size; ++i) {
    const TfLiteTensor& output = context->tensors[node->outputs->data[i]];
    if (!CheckTensorDtype(output, SupportedDtypes(),
                          absl::StrCat("output[", i, "]"), *error)) {
      return false;
    }

    if (output.dims->size != input.dims->size - 1) {
      *error = absl::StrCat("output[", i, "] rank mismatch.");
      return false;
    }

    int output_dim_idx = 0;
    for (int input_dim_idx = 0; input_dim_idx < input.dims->size;
         ++input_dim_idx) {
      if (input_dim_idx == axis) continue;
      if (output.dims->data[output_dim_idx] !=
          input.dims->data[input_dim_idx]) {
        *error = absl::StrCat("output[", i, "] shape mismatch.");
        return false;
      }
      output_dim_idx++;
    }
  }

  return true;
}

}  // namespace litert::ml_drift::ir
