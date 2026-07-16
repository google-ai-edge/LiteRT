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

#include "ml_drift_delegate/tflite/support/support_concat.h"

#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/ir_model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

using ::tflite::IsConstantTensor;

namespace litert::ml_drift::ir {

constexpr int kMaxDims = 5;

bool IsConcatSupported(const TfLiteContext* absl_nonnull context,
                       const TfLiteNode* absl_nonnull node,
                       const TfLiteRegistration* absl_nonnull registration,
                       std::string* absl_nonnull error) {
  // Check version.
  if (registration->version < 1 || registration->version > 6) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check number of inputs.
  const TfLiteIntArray* inputs = node->inputs;
  if (inputs->size <= 1) {
    *error = absl::StrCat("Invalid number of inputs: ", inputs->size,
                          ", should be at least 2");
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
  for (int i = 0; i < inputs->size; ++i) {
    if (!IsConstantTensor(input_tensors[i])) {
      break;
    }
    if (i == inputs->size - 1) {
      *error = absl::StrCat("Invalid all constant inputs: [", inputs->data[0],
                            ", ", inputs->data[inputs->size - 1], "]");
      return false;
    }
  }
  // Check params
  const auto* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);
  if (!params) {
    *error = "Incompatible node->builtin_data";
    return false;
  }
  // Check dims.
  for (int i = 0; i < inputs->size; ++i) {
    if (!CheckTensorDims(*input_tensors[i], /*min_dims=*/1,
                         /*max_dims=*/kMaxDims, absl::StrCat("inputs[", i, "]"),
                         *error)) {
      return false;
    }
  }

  // Get axis
  std::vector<::ml_drift::BHWDC> input_shapes;
  input_shapes.reserve(input_tensors.size());
  for (const auto& tensor : input_tensors) {
    input_shapes.push_back(
        ::litert::ml_drift::ir::ExtractTensorShape(tensor->dims));
  }
  const ::ml_drift::BHWDC output_shape =
      ::litert::ml_drift::ir::ExtractTensorShape(output.dims);
  const ::ml_drift::Axis axis =
      ::litert::ml_drift::ir::GetConcatAxis(input_shapes, output_shape);
  if (axis == ::ml_drift::Axis::UNKNOWN) {
    *error = "Invalid axis";
    return false;
  }
  // Check shapes. Need to agree on all dims but axis.
  auto check_dim = [&](::ml_drift::Axis dim_axis) {
    const std::string dim_name = ToString(dim_axis);
    if (axis == dim_axis) return true;
    for (int i = 1; i < input_shapes.size(); ++i) {
      if (input_shapes[i].get(dim_axis) != input_shapes[0].get(dim_axis)) {
        *error = absl::StrCat("Input[", i, "] dim ", dim_name,
                              " is: ", input_shapes[i].get(dim_axis),
                              ", should be: ", input_shapes[0].get(dim_axis));
        return false;
      }
    }
    if (output_shape.get(dim_axis) != input_shapes[0].get(dim_axis)) {
      *error = absl::StrCat("Output dim ", dim_name,
                            " is: ", output_shape.get(dim_axis),
                            ", should be: ", input_shapes[0].get(dim_axis));
      return false;
    }
    return true;
  };

  if (!check_dim(::ml_drift::Axis::BATCH)) return false;
  if (!check_dim(::ml_drift::Axis::HEIGHT)) return false;
  if (!check_dim(::ml_drift::Axis::WIDTH)) return false;
  if (!check_dim(::ml_drift::Axis::DEPTH)) return false;
  if (!check_dim(::ml_drift::Axis::CHANNELS)) return false;

  // Check output concat dim is same size as sum of input concat dims.
  int dim_sum = 0;
  for (int i = 0; i < input_shapes.size(); ++i) {
    dim_sum += input_shapes[i].get(axis);
  }
  if (output_shape.get(axis) != dim_sum) {
    *error = absl::StrCat("Output concat dim is: ", output_shape.get(axis),
                          ", should be: ", dim_sum);
    return false;
  }
  if (!CheckFusedActivation(node, params->activation)
           .ok()) {
    *error = absl::StrCat("Unsupported fused activation: ", params->activation);
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
