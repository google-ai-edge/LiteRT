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

#include "third_party/odml/litert/ml_drift/tflite/support/support_splitv.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/ir_model_builder_helper.h"
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

using ::tflite::IsConstantTensor;

bool IsSplitVSupported(const TfLiteContext* absl_nonnull context,
                       const TfLiteNode* absl_nonnull node,
                       const TfLiteRegistration* absl_nonnull registration,
                       std::string* absl_nonnull error) {
  // Check version.
  if (registration->version != 1) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check number of inputs.
  // Input tensor map:
  // 0: input tensor (non-const)
  // 1: num_split tensor (const). We can ignore this tensor since num_split is
  // parsed into params->num_splits.
  // 2: axis tensor (const)
  const TfLiteIntArray& inputs = *node->inputs;
  if (inputs.size != 3) {
    *error = absl::StrCat("Invalid number of inputs: ", inputs.size,
                          ", should be 3");
    return false;
  }
  // Check param
  const auto* params =
      reinterpret_cast<TfLiteSplitVParams*>(node->builtin_data);
  if (!params) {
    *error = "Incompatible node->builtin_data";
    return false;
  }
  if (params->num_splits <= 0) {
    *error = absl::StrCat("Invalid num_splits: ", params->num_splits,
                          ", should be > 0");
    return false;
  }
  // Check number of outputs.
  const TfLiteIntArray& outputs = *node->outputs;
  if (outputs.size != params->num_splits) {
    *error = absl::StrCat("Invalid number of outputs: ", outputs.size,
                          ", which should be: ", params->num_splits);
    return false;
  }
  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, outputs, "outputs", *error)) return false;

  const int input_id = inputs.data[0];
  const int num_splits_id = inputs.data[1];
  const int axis_id = inputs.data[2];

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
  const TfLiteTensor& input_tensor = context->tensors[input_id];
  if (!CheckTensorDtype(input_tensor, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& num_splits_tensor = context->tensors[num_splits_id];
  const TfLiteTensor& axis_tensor = context->tensors[axis_id];
  if (!CheckTensorDtype(axis_tensor, {kTfLiteInt32}, "inputs[2]", *error)) {
    return false;
  }
  std::vector<const TfLiteTensor*> output_tensors;
  output_tensors.reserve(outputs.size);
  for (int i = 0; i < outputs.size; ++i) {
    output_tensors.push_back(&context->tensors[outputs.data[i]]);
    if (!CheckTensorDtype(*output_tensors[i], supported_dtypes,
                          absl::StrCat("outputs[", i, "]"), *error)) {
      return false;
    }
  }
  // Check const inputs.
  if (!CheckNotConstant(input_tensor, "inputs[0]", *error)) {
    return false;
  }
  if (!CheckIsConstant(num_splits_tensor, "inputs[1]", *error)) {
    return false;
  }
  if (!CheckIsConstant(axis_tensor, "inputs[2]", *error)) {
    return false;
  }
  for (int i = 0; i < outputs.size; ++i) {
    if (IsConstantTensor(output_tensors[i])) {
      *error = "Output tensor is constant";
      return false;
    }
  }
  // Check dims.
  if (!CheckTensorDims(input_tensor, /*min_dims=*/1, /*max_dims=*/4,
                       "inputs[0]", *error)) {
    return false;
  }
  if (!CheckTensorDims(axis_tensor, /*min_dims=*/0, /*max_dims=*/1, "inputs[2]",
                       *error)) {
    return false;
  }
  for (int i = 0; i < outputs.size; ++i) {
    if (!CheckTensorDims(*output_tensors[i], /*min_dims=*/1, /*max_dims=*/4,
                         absl::StrCat("outputs[", i, "]"), *error)) {
      return false;
    }
  }

  // Axis can be negative.
  const int32_t axis =
      ResolveNegativeIndex(axis_tensor.data.i32[0], input_tensor.dims->size);
  if (axis < 0 || axis >= input_tensor.dims->size) {
    *error = absl::StrCat("Invalid axis: ", axis);
    return false;
  }
  // Make sure we agree on all dims except the split dim.
  for (int dim = 0; dim < input_tensor.dims->size; ++dim) {
    if (dim == axis) {
      continue;
    }
    for (int i = 0; i < outputs.size; ++i) {
      if (input_tensor.dims->data[dim] != output_tensors[i]->dims->data[dim]) {
        *error = absl::StrCat("Input and output tensor[", i,
                              "] dims do not agree on dim ", dim);
        return false;
      }
    }
  }

  // Along split dim, make sure all output tensors are the same, and multiply
  // to equal the input tensor along that dim.
  for (int i = 1; i < outputs.size; ++i) {
    if (output_tensors[i]->dims->data[axis] !=
        output_tensors[0]->dims->data[axis]) {
      *error =
          absl::StrCat("Output tensor[", i,
                       "] split dim is: ", output_tensors[i]->dims->data[axis],
                       ", should be: ", output_tensors[0]->dims->data[axis]);
      return false;
    }
  }
  if (output_tensors[0]->dims->data[axis] * output_tensors.size() !=
      input_tensor.dims->data[axis]) {
    *error = absl::StrCat("Output tensors do not cover the input tensor. ",
                          output_tensors[0]->dims->data[axis], " * ",
                          output_tensors.size(),
                          " != ", input_tensor.dims->data[axis]);
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
