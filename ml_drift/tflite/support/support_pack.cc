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

#include "third_party/odml/litert/ml_drift/tflite/support/support_pack.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

using ::tflite::IsConstantTensor;

bool IsPackSupported(const TfLiteContext* absl_nonnull context,
                     const TfLiteNode* absl_nonnull node,
                     const TfLiteRegistration* absl_nonnull registration,
                     std::string* absl_nonnull error) {
  // Check version.
  if (registration->version != 1) {
    *error = absl::StrCat("Unsupported version. Should be 1, is instead: ",
                          registration->version);
    return false;
  }

  // Check number of inputs.
  const TfLiteIntArray* inputs = node->inputs;
  if (inputs->size < 1) {
    *error = absl::StrCat("Invalid number of inputs: ", inputs->size,
                          ", should be at least 1");
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

  const int output_id = outputs->data[0];

  // Check dtype.
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteBool,
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
  std::vector<const TfLiteTensor*> input_tensors(inputs->size);
  for (int i = 0; i < inputs->size; ++i) {
    input_tensors[i] = &context->tensors[inputs->data[i]];
    if (!CheckTensorDtype(*input_tensors[i], supported_dtypes,
                          absl::StrCat("inputs[", i, "]"), *error)) {
      return false;
    }
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }
  // Check const inputs.
  if (std::all_of(input_tensors.begin(), input_tensors.end(),
                  IsConstantTensor)) {
    *error = absl::StrCat("Invalid all constant inputs: [", inputs->data[0],
                          ", ", inputs->data[inputs->size - 1], "]");
    return false;
  }
  // Check dims. Should all be the same number of dims.
  for (int i = 0; i < inputs->size; ++i) {
    if (!CheckTensorDims(*input_tensors[i], /*min_dims=*/1, /*max_dims=*/4,
                         absl::StrCat("inputs[", i, "]"), *error)) {
      return false;
    }
  }
  if (!CheckTensorDims(output, /*min_dims=*/1, /*max_dims=*/4, "outputs[0]",
                       *error)) {
    return false;
  }
  if (inputs->size == 1) {
    return true;
  }  // Use reshape op.
  // Check params
  const auto* params =
      reinterpret_cast<const TfLitePackParams*>(node->builtin_data);
  if (!params) {
    *error = "Incompatible node->builtin_data";
    return false;
  }
  // Check axis value.
  int axis_value = params->axis;
  const int output_rank = output.dims->size;
  if (axis_value < 0) {
    axis_value += output_rank;
  }
  if (axis_value < 0 || axis_value >= output_rank) {
    *error = absl::StrCat("Invalid axis value: ", axis_value,
                          ", should be in [0, ", output_rank, ")");
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
