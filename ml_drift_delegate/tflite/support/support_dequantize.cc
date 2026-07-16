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

#include "ml_drift_delegate/tflite/support/support_dequantize.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

bool IsDequantizeSupported(const TfLiteContext* absl_nonnull context,
                           const TfLiteNode* absl_nonnull node,
                           const TfLiteRegistration* absl_nonnull registration,
                           std::string* absl_nonnull error) {
  if (registration->version > 3) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

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

  if (::tflite::IsConstantTensor(input)) {
    if (!CheckTensorDims(*input, /*min_dims=*/2, /*max_dims=*/4, "input",
                         *error)) {
      return false;
    }
    const absl::flat_hash_set<TfLiteType> supported_input_dtypes = {
        kTfLiteFloat32, kTfLiteFloat16, kTfLiteInt4, kTfLiteInt2,
        kTfLiteInt8,    kTfLiteUInt8,   kTfLiteInt32};
    if (!CheckTensorDtype(*input, supported_input_dtypes, "input", *error)) {
      return false;
    }
    if (input->sparsity) {
      *error = "ML Drift doesn't support sparsity.";
      return false;
    }
    if (input->dims->size == 2) {
      const absl::Status status =
          CheckPopulateTensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32>(
              input);
      if (!status.ok()) {
        *error = status.message();
        return false;
      }
    } else if (input->dims->size == 3) {
      const absl::Status status =
          CheckPopulateTensor<::ml_drift::HWC, ::ml_drift::DataType::FLOAT32>(
              input);
      if (!status.ok()) {
        *error = status.message();
        return false;
      }
    } else if (input->dims->size == 4) {
      const absl::Status status =
          CheckPopulateTensor<::ml_drift::BHWC, ::ml_drift::DataType::FLOAT32>(
              input);
      if (!status.ok()) {
        *error = status.message();
        return false;
      }
    }
    return true;
  }

  const TfLiteAffineQuantization* quantization_data =
      reinterpret_cast<const TfLiteAffineQuantization*>(
          input->quantization.params);
  if (quantization_data) {
    if (quantization_data->scale->size > 1) {
      *error = "Unsupported quantization scale size";
      return false;
    }
  } else {
    if (::tflite::IsConstantTensor(input)) {
      *error = "Encountered Dequantize input with no quant params";
      return false;
    }
  }

  return true;
}

}  // namespace litert::ml_drift::ir
