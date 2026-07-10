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

#include "third_party/odml/litert/ml_drift/tflite/support/support_argmax.h"

#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

using ::tflite::GetShapeDebugString;

namespace litert::ml_drift::ir {
namespace {

bool IsScalar(const TfLiteTensor* tensor) {
  const TfLiteIntArray* dimensions = tensor->dims;
  if (dimensions->size < 0) {
    return false;
  }
  for (int i = 0; i < dimensions->size; ++i) {
    if (dimensions->data[i] != 1) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool IsArgMaxSupported(const TfLiteContext* absl_nonnull context,
                       const TfLiteNode* absl_nonnull node,
                       const TfLiteRegistration* absl_nonnull registration,
                       std::string* absl_nonnull error) {
  // Check version.
  // ArgMax currently does not have version restrictions.

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

  const int input0_id = inputs->data[0];
  const int input1_id = inputs->data[1];
  const int output_id = outputs->data[0];

  // Check dtype.
  const TfLiteTensor& dim_tensor = context->tensors[input1_id];
  if (!CheckTensorDtype(dim_tensor, {kTfLiteInt32},
                        "inputs[1]" /*dimension tensor*/, *error)) {
    return false;
  }

  // Check dims.
  const TfLiteTensor& input = context->tensors[input0_id];
  if (!CheckTensorDims(input, /*min_dims=*/0, /*max_dims=*/4, "inputs[0]",
                       *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDims(output, /*min_dims=*/0, /*max_dims=*/4, "outputs[0]",
                       *error)) {
    return false;
  }
  if (!IsScalar(&dim_tensor)) {
    *error = absl::StrCat("Dimension tensor is not a scalar, with shape: ",
                          GetShapeDebugString(dim_tensor.dims));
    return false;
  }
  // Check const inputs.
  if (!CheckNotConstant(input, "inputs[0]", *error)) return false;
  if (!CheckIsConstant(dim_tensor, "inputs[1]", *error)) {
    return false;
  }
  const absl::Status status =
      CheckPopulateTensor<::ml_drift::Scalar, ::ml_drift::DataType::INT32>(
          &dim_tensor);
  if (!status.ok()) {
    *error = status.message();
    return false;
  }
  // Check shapes.
  int dim_index = dim_tensor.data.i32[0];
  if (dim_index < 0) {
    dim_index = input.dims->size + dim_index;
  }
  if (dim_index < 0 || dim_index >= input.dims->size) {
    *error =
        absl::StrCat("Index for axis out of range: ", dim_tensor.data.i32[0],
                     ", while input has ", input.dims->size, " dimensions");
    return false;
  }
  std::vector<int> expected_output_dims(input.dims->size);
  for (int i = 0; i < input.dims->size; ++i) {
    expected_output_dims[i] = i == dim_index ? 1 : input.dims->data[i];
  }

  if (output.dims->size == expected_output_dims.size()) {
    for (int i = 0; i < output.dims->size; ++i) {
      if (i != dim_index && output.dims->data[i] != expected_output_dims[i]) {
        *error = absl::StrCat(
            "Output dimension mismatch: ", GetShapeDebugString(output.dims),
            ", ", GetShapeDebugString(input.dims));
        return false;
      } else if (i == dim_index && output.dims->data[i] != 1) {
        *error = absl::StrCat(
            "Output dimension mismatch: ", GetShapeDebugString(output.dims),
            ", ", GetShapeDebugString(input.dims));
        return false;
      }
    }
  } else if (output.dims->size < expected_output_dims.size()) {
    const int delta = expected_output_dims.size() - output.dims->size;
    for (int i = 0; i < expected_output_dims.size(); ++i) {
      if (i < delta && expected_output_dims[i] != 1) {
        *error = absl::StrCat(
            "Output dimension mismatch: ", GetShapeDebugString(output.dims),
            ", ", GetShapeDebugString(input.dims));
        return false;
      } else if (i >= delta && i != dim_index &&
                 output.dims->data[i - delta] != expected_output_dims[i]) {
        *error = absl::StrCat(
            "Output dimension mismatch: ", GetShapeDebugString(output.dims),
            ", ", GetShapeDebugString(input.dims));
        return false;
      } else if (i == dim_index && output.dims->data[i - delta] != 1) {
        *error = absl::StrCat(
            "Output dimension mismatch: ", GetShapeDebugString(output.dims),
            ", ", GetShapeDebugString(input.dims));
        return false;
      }
    }
  } else {
    const int delta = output.dims->size - expected_output_dims.size();
    for (int i = 0; i < output.dims->size; ++i) {
      if (i < delta && output.dims->data[i] != 1) {
        *error = absl::StrCat(
            "Output dimension mismatch: ", GetShapeDebugString(output.dims),
            ", ", GetShapeDebugString(input.dims));
        return false;
      } else if (i >= delta && (i - delta) != dim_index &&
                 output.dims->data[i] != expected_output_dims[i - delta]) {
        *error = absl::StrCat(
            "Output dimension mismatch: ", GetShapeDebugString(output.dims),
            ", ", GetShapeDebugString(input.dims));
        return false;
      } else if ((i - delta) == dim_index && output.dims->data[i] != 1) {
        *error = absl::StrCat(
            "Output dimension mismatch: ", GetShapeDebugString(output.dims),
            ", ", GetShapeDebugString(input.dims));
        return false;
      }
    }
  }
  return true;
}

}  // namespace litert::ml_drift::ir
