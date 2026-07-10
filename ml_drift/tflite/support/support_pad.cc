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

#include "third_party/odml/litert/ml_drift/tflite/support/support_pad.h"

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "third_party/odml/litert/ml_drift/tflite/support/support_aux.h"
#include "tflite/builtin_ops.h"
#include "tflite/c/builtin_op_data.h"
#include "tflite/c/common.h"
#include "tflite/kernels/kernel_util.h"

namespace litert::ml_drift::ir {

constexpr int kMaxDims = 4;

using ::tflite::NumElements;

bool IsPadSupported(const TfLiteContext* absl_nonnull context,
                    const TfLiteNode* absl_nonnull node,
                    const TfLiteRegistration* absl_nonnull registration,
                    std::string* absl_nonnull error) {
  // Check version.
  if (registration->version > 5) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }

  // Check number of inputs.
  // input 0: input tensor
  // input 1: paddings tensor
  // input 2: constant_values tensor (optional, for PADV2)
  const TfLiteIntArray* inputs = node->inputs;
  if (inputs->size < 2 || inputs->size > 3) {
    *error = absl::StrCat("Invalid number of inputs: ", inputs->size,
                          ", should be 2 or 3");
    return false;
  }
  if ((registration->builtin_code == kTfLiteBuiltinPad ||
       registration->builtin_code == kTfLiteBuiltinMirrorPad) &&
      inputs->size != 2) {
    *error = absl::StrCat("Invalid number of inputs: ", inputs->size,
                          ", should be 2");
    return false;
  }

  // Check number of outputs.
  const TfLiteIntArray* outputs = node->outputs;
  if (!CheckInputOutputCounts(*node, /*expected_inputs=*/inputs->size,
                              /*expected_outputs=*/1, *error)) {
    return false;
  }

  if (registration->builtin_code == kTfLiteBuiltinMirrorPad) {
    const auto* params =
        static_cast<TfLiteMirrorPaddingParams*>(node->builtin_data);
    if (!params) {
      *error = "Incompatible node->builtin_data for MirrorPad";
      return false;
    }
    if (params->mode != kTfLiteMirrorPaddingReflect) {
      *error = "Only Reflective padding is supported for Mirror Pad operation.";
      return false;
    }
  }

  // Validate tensor IDs.
  if (!ValidateTensorIds(*context, *inputs, "inputs", *error)) return false;
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) return false;

  const int input_id = inputs->data[0];
  const int paddings_id = inputs->data[1];
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
  const TfLiteTensor& input_tensor = context->tensors[input_id];
  if (!CheckTensorDtype(input_tensor, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& paddings_tensor = context->tensors[paddings_id];
  if (!CheckTensorDtype(paddings_tensor, {kTfLiteInt32}, "inputs[1]", *error)) {
    return false;
  }
  const absl::Status status =
      CheckPopulateTensor<::ml_drift::HW, ::ml_drift::DataType::INT32>(
          &paddings_tensor);
  if (!status.ok()) {
    *error = status.message();
    return false;
  }
  if (inputs->size == 3) {
    const TfLiteTensor& const_val_tensor = context->tensors[inputs->data[2]];
    if (!CheckTensorDtype(const_val_tensor, {input_tensor.type}, "inputs[2]",
                          *error)) {
      return false;
    }
    if (NumElements(&const_val_tensor) != 1) {
      *error = "constant_values tensor must be scalar.";
      return false;
    }
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::Scalar, ::ml_drift::DataType::FLOAT32>(
            &const_val_tensor);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  }

  const TfLiteTensor& output_tensor = context->tensors[output_id];
  if (!CheckTensorDtype(output_tensor, supported_dtypes, "outputs[0]",
                        *error)) {
    return false;
  }

  // Check const inputs.
  if (!CheckNotConstant(input_tensor, "inputs[0]", *error)) {
    return false;
  }
  if (!CheckIsConstant(paddings_tensor, "inputs[1]", *error)) {
    return false;
  }
  if (inputs->size == 3 && !CheckIsConstant(context->tensors[inputs->data[2]],
                                            "inputs[2]", *error)) {
    return false;
  }

  // Paddings shape
  if (!CheckTensorDims(paddings_tensor, /*min_dims=*/2, /*max_dims=*/2,
                       "inputs[1]", *error)) {
    return false;
  }
  if (paddings_tensor.dims->data[1] != 2) {
    *error = "Paddings tensor second dimension must be 2";
    return false;
  }
  if (paddings_tensor.dims->data[0] != input_tensor.dims->size) {
    *error = "Paddings tensor first dimension must be equal to input rank.";
    return false;
  }

  if (!CheckTensorDims(input_tensor, /*min_dims=*/1, /*max_dims=*/kMaxDims,
                       "inputs[0]", *error)) {
    return false;
  }

  return true;
}

}  // namespace litert::ml_drift::ir
