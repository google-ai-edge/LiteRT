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

#include "ml_drift_delegate/tflite/support/support_batch_matmul.h"

#include <string>

#include "absl/base/no_destructor.h"  // from @com_google_absl
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
namespace {

using ::tflite::GetShapeDebugString;
using ::tflite::IsConstantTensor;

const absl::flat_hash_set<TfLiteType>& GetSupportedDtypes() {
  static const absl::NoDestructor<absl::flat_hash_set<TfLiteType>>
      supported_dtypes(absl::flat_hash_set<TfLiteType>{
          // clang-format off
        // go/keep-sorted start numeric=yes
        kTfLiteBFloat16,
        kTfLiteFloat16,
        kTfLiteFloat32,
        kTfLiteInt8,
        kTfLiteUInt8,
        // go/keep-sorted end
          // clang-format on
      });
  return *supported_dtypes;
}

bool IsRuntimeInputSupported(const TfLiteTensor& input,
                             const std::string& tensor_name,
                             std::string* error) {
  if (!CheckTensorDtype(input, GetSupportedDtypes(), tensor_name, *error)) {
    return false;
  }
  if ((input.type == kTfLiteUInt8 || input.type == kTfLiteInt8) &&
      input.quantization.type == kTfLiteAffineQuantization) {
    TfLiteAffineQuantization* quantization_data =
        reinterpret_cast<TfLiteAffineQuantization*>(input.quantization.params);
    if (quantization_data->scale->size > 1) {
      *error = absl::StrCat(
          "Only tensor-wise quantized input is supported, but the "
          "size of scale is not 1: ",
          quantization_data->scale->size);
      return false;
    }
  }
  return true;
}

}  // namespace

bool IsBatchMatMulSupported(const TfLiteContext* absl_nonnull context,
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

  const int input0_id = inputs->data[0];
  const int input1_id = inputs->data[1];
  const int output_id = outputs->data[0];

  // Check dtype for inputs.
  const TfLiteTensor& input0 = context->tensors[input0_id];
  if (!IsRuntimeInputSupported(input0, "inputs[0]", error)) {
    return false;
  }
  const TfLiteTensor& input1 = context->tensors[input1_id];
  if (IsConstantTensor(&input1)) {
    // Only 2D is currently supported because we will reduce the op to
    // FullyConnected to support the case that the second input is a constant
    // tensor.
    if (input1.dims->size != 2) {
      absl::StrAppend(error,
                      "If the second input is a constant tensor, only 2D "
                      "tensor is supported, but got ",
                      input1.dims->size, "D");
      return false;
    }
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::HW, ::ml_drift::DataType::FLOAT32>(
            &input1);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  } else if (!IsRuntimeInputSupported(input1, "inputs[1]", error)) {
    return false;
  }
  // Check dtype for output.
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, GetSupportedDtypes(), "outputs[0]", *error)) {
    return false;
  }
  // Check dims and shapes.
  // The shapes should be:
  //   input0: [..., M, K]
  //   input1: [..., K, N]
  //   output: [..., M, N]
  if (!CheckTensorDims(input0, /*min_dims=*/2, /*max_dims=*/5, "inputs[0]",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(input1, /*min_dims=*/2, /*max_dims=*/5, "inputs[1]",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(output, /*min_dims=*/2, /*max_dims=*/5, "outputs[0]",
                       *error)) {
    return false;
  }
  // Attributes.
  const TfLiteBatchMatMulParams* params =
      reinterpret_cast<TfLiteBatchMatMulParams*>(node->builtin_data);
  bool transpose_left = params ? params->adj_x : false;
  bool transpose_right = params ? params->adj_y : false;
  // Dimensions check. Account for transpose.
  const int l_dim1 = transpose_left ? 1 : 2;   // M
  const int l_dim0 = transpose_left ? 2 : 1;   // K
  const int r_dim1 = transpose_right ? 1 : 2;  // K
  const int r_dim0 = transpose_right ? 2 : 1;  // N
  const int input0_m = input0.dims->data[input0.dims->size - l_dim1];
  const int input0_k = input0.dims->data[input0.dims->size - l_dim0];
  const int input1_k = input1.dims->data[input1.dims->size - r_dim1];
  const int input1_n = input1.dims->data[input1.dims->size - r_dim0];
  const int output_m = output.dims->data[output.dims->size - 2];
  const int output_n = output.dims->data[output.dims->size - 1];
  if (input0_k != input1_k) {
    absl::StrAppend(
        error, "The inner dimensions of the first and second inputs mismatch: ",
        GetShapeDebugString(input0.dims), " vs ",
        GetShapeDebugString(input1.dims));
    return false;
  }
  if (input0_m != output_m) {
    absl::StrAppend(error,
                    "The dimensions of the first input and output mismatch: ",
                    GetShapeDebugString(input0.dims), " vs ",
                    GetShapeDebugString(output.dims));
    return false;
  }
  if (input1_n != output_n) {
    absl::StrAppend(error,
                    "The dimensions of the second input and output mismatch: ",
                    GetShapeDebugString(input1.dims), " vs ",
                    GetShapeDebugString(output.dims));
    return false;
  }

  // Check const inputs.
  if (!CheckNotConstant(input0, "inputs[0]", *error)) {
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
