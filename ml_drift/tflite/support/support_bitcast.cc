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

#include "third_party/odml/litert/ml_drift/tflite/support/support_bitcast.h"

#include <cstddef>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"
#include "tflite/util.h"

namespace litert::ml_drift::ir {

using ::tflite::GetShapeDebugString;
using ::tflite::GetSizeOfType;
using ::tflite::NumElements;

bool IsBitcastSupported(const TfLiteContext* absl_nonnull context,
                        const TfLiteNode* absl_nonnull node,
                        const TfLiteRegistration* absl_nonnull registration,
                        std::string* absl_nonnull error) {
  // No version check for bitcast.
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/1,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  const int input_id = node->inputs->data[0];
  const int output_id = node->outputs->data[0];
  if (!ValidateTensorId(*context, input_id, "input", *error)) {
    return false;
  }
  if (!ValidateTensorId(*context, output_id, "output", *error)) {
    return false;
  }

  const TfLiteTensor* input = context->tensors + input_id;
  const TfLiteTensor* output = context->tensors + output_id;

  if (!CheckTensorDims(*input, /*min_dims=*/0, /*max_dims=*/4, "input",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(*output, /*min_dims=*/0, /*max_dims=*/4, "output",
                       *error)) {
    return false;
  }

  // Check size match.
  size_t input_elem_size, output_elem_size;
  if (GetSizeOfType(/*context=*/nullptr, input->type, &input_elem_size) !=
      kTfLiteOk) {
    *error = "Could not parse input type";
    return false;
  }
  if (GetSizeOfType(/*context=*/nullptr, output->type, &output_elem_size) !=
      kTfLiteOk) {
    *error = "Could not parse output type";
    return false;
  }
  if (input_elem_size == output_elem_size) {
    if (NumElements(input->dims) != NumElements(output->dims)) {
      *error =
          "If input and output types have the same element size, they must "
          "have the same number of elements";
      return false;
    }
  } else if (input_elem_size > output_elem_size) {
    if (input->dims->size + 1 != output->dims->size) {
      *error =
          "If input element size is greater than output element size, "
          "require that output rank is one greater than input rank";
      return false;
    }
    for (int d = 0; d < input->dims->size; ++d) {
      if (input->dims->data[d] != output->dims->data[d]) {
        *error = absl::StrCat("Shapes must match in all but last dim, but got ",
                              GetShapeDebugString(input->dims), " vs ",
                              GetShapeDebugString(output->dims));
        return false;
      }
    }
    if (output->dims->data[output->dims->size - 1] * output_elem_size !=
        input_elem_size) {
      *error = absl::StrCat(
          "Last output dim(", output->dims->data[output->dims->size - 1],
          ") must be equal to input element size(", input_elem_size,
          ") divided by output element size(", output_elem_size, ")");
      return false;
    }
  } else {  // output_elem_size > input_elem_size
    if (input->dims->size != output->dims->size + 1) {
      *error =
          "If output element size is greater than input element size, "
          "require that input rank is on greater than output rank";
      return false;
    }
    for (int d = 0; d < output->dims->size; ++d) {
      if (input->dims->data[d] != output->dims->data[d]) {
        *error = absl::StrCat("Shapes must match in all but last dim, but got ",
                              GetShapeDebugString(input->dims), " vs ",
                              GetShapeDebugString(output->dims));
        return false;
      }
    }
    if (input->dims->data[input->dims->size - 1] * input_elem_size !=
        output_elem_size) {
      *error = absl::StrCat(
          "Last input dim(", input->dims->data[input->dims->size - 1],
          ") must be equal to output element size(", output_elem_size,
          ") divided by input element size(", input_elem_size, ")");
      return false;
    }
  }

  if (!CheckNotConstant(*input, "input", *error)) {
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
