// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law of or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/tflite/support/support_depthwise_conv.h"

#include <algorithm>
#include <optional>
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

using ::tflite::GetShapeDebugString;
using ::tflite::IsConstantTensor;

namespace litert::ml_drift::ir {
namespace {

int GetPadding(const TfLitePadding padding, const int input,
               const int dilated_kernel, const int stride) {
  if (padding == kTfLitePaddingValid) return 0;
  const int output = (input + stride - 1) / stride;  // = ceil(input / stride)
  return std::max(0, (output - 1) * stride + dilated_kernel - input);
}

int GetOutputDim(const TfLitePadding padding, const int input, const int kernel,
                 const int dilation, const int stride) {
  const int dilated_kernel = (kernel - 1) * dilation + 1;
  const int total_padding = GetPadding(padding, input, dilated_kernel, stride);
  return (input + total_padding - dilated_kernel) / stride + 1;
}

}  // namespace

bool IsDepthwiseConv2dSupported(
    const TfLiteContext* absl_nonnull context,
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
  if (inputs->size < 2 || inputs->size > 3) {
    absl::StrAppend(error, "Invalid number of inputs: ", inputs->size);
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

  const int input0_id = inputs->data[0];
  const int input1_id = inputs->data[1];
  std::optional<int> input2_id;
  if (inputs->size == 3) input2_id = inputs->data[2];
  const int output_id = outputs->data[0];

  // Check dtype.
  const absl::flat_hash_set<TfLiteType> supported_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteBFloat16,
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
      kTfLiteUInt8,
      // go/keep-sorted end
      // clang-format on
  };
  const TfLiteTensor& input = context->tensors[input0_id];
  if (!CheckTensorDtype(input, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& weights = context->tensors[input1_id];
  const absl::flat_hash_set<TfLiteType> supported_weights_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt8,
      kTfLiteUInt8,
      // go/keep-sorted end
      // clang-format on
  };
  if (!CheckTensorDtype(weights, supported_weights_dtypes, "inputs[1]",
                        *error)) {
    return false;
  }
  if (IsConstantTensor(&weights)) {
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>(
            &weights);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  }
  const TfLiteTensor* bias =
      input2_id.has_value() ? &context->tensors[*input2_id] : nullptr;
  if (bias && !IsConstantTensor(bias)) {
    *error = "Unsupported runtime bias tensor";
    return false;
  }
  const absl::flat_hash_set<TfLiteType> supported_bias_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteFloat16,
      kTfLiteFloat32,
      // go/keep-sorted end
      // clang-format on
  };
  if (bias &&
      !CheckTensorDtype(*bias, supported_bias_dtypes, "inputs[2]", *error)) {
    return false;
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }
  // Check dims.
  if (!CheckTensorDims(input, /*min_dims=*/4, /*max_dims=*/4, "inputs[0]",
                       *error)) {
    return false;
  }
  if (!CheckTensorDims(weights, /*min_dims=*/4, /*max_dims=*/4, "inputs[1]",
                       *error)) {
    return false;
  }
  if (bias && !CheckTensorDims(*bias, /*min_dims=*/1, /*max_dims=*/1,
                               "inputs[2]", *error)) {
    return false;
  }
  if (!CheckTensorDims(output, /*min_dims=*/4, /*max_dims=*/4, "outputs[0]",
                       *error)) {
    return false;
  }
  const int ib = input.dims->data[0];
  const int ih = input.dims->data[1];
  const int iw = input.dims->data[2];
  const int ic = input.dims->data[3];
  const int wo = weights.dims->data[0];
  const int wh = weights.dims->data[1];
  const int ww = weights.dims->data[2];
  const int wc = weights.dims->data[3];
  if (wo != 1) {
    absl::StrAppend(error, "Weights output channel must be 1: ",
                    GetShapeDebugString(weights.dims));
    return false;
  }
  const auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  if (!params) {
    absl::StrAppend(error, "Incompatible node->builtin_data");
    return false;
  }
  if (wc != ic * params->depth_multiplier) {
    absl::StrAppend(
        error, "Input channel, weights channel and depth multiplier mismatch: ",
        GetShapeDebugString(input.dims), ", ",
        GetShapeDebugString(weights.dims), ", ", params->depth_multiplier);
    return false;
  }
  const int bl = bias ? bias->dims->data[0] : 0;
  if (bias && wc != bl) {
    absl::StrAppend(error, "Weights channel and bias mismatch: ",
                    GetShapeDebugString(weights.dims), ", ",
                    GetShapeDebugString(bias->dims));
    return false;
  }
  const int ob = output.dims->data[0];
  const int oh = output.dims->data[1];
  const int ow = output.dims->data[2];
  const int oc = output.dims->data[3];
  if (ib != ob) {
    absl::StrAppend(error, "Input and output batch mismatch: ",
                    GetShapeDebugString(input.dims), ", ",
                    GetShapeDebugString(output.dims));
    return false;
  }
  if (oc != wc) {
    absl::StrAppend(error, "Output and weights channel mismatch: ",
                    GetShapeDebugString(output.dims), ", ",
                    GetShapeDebugString(weights.dims));
    return false;
  }
  const int dh = params->dilation_height_factor;
  const int dw = params->dilation_width_factor;
  const int sh = params->stride_height;
  const int sw = params->stride_width;
  if (params->padding != kTfLitePaddingSame &&
      params->padding != kTfLitePaddingValid) {
    absl::StrAppend(error, "Incompatible padding: ", params->padding);
    return false;
  }
  if (oh != GetOutputDim(params->padding, ih, wh, dh, sh)) {
    absl::StrAppend(error, "Input, weights, output height mismatch: ",
                    GetShapeDebugString(input.dims), ", ",
                    GetShapeDebugString(weights.dims), ", ",
                    GetShapeDebugString(output.dims), ", padding: ",
                    params->padding == kTfLitePaddingSame ? "same" : "valid",
                    ", dilation: ", dh, ", stride: ", sh);
    return false;
  }
  if (ow != GetOutputDim(params->padding, iw, ww, dw, sw)) {
    absl::StrAppend(error, "Input, weights, output width mismatch: ",
                    GetShapeDebugString(input.dims), ", ",
                    GetShapeDebugString(weights.dims), ", ",
                    GetShapeDebugString(output.dims), ", padding: ",
                    params->padding == kTfLitePaddingSame ? "same" : "valid",
                    ", dilation: ", dw, ", stride: ", sw);
    return false;
  }
  // Check const inputs.
  if (IsConstantTensor(&input) && IsConstantTensor(&weights) &&
      ((bias && IsConstantTensor(bias)) || !bias)) {
    absl::StrAppend(error, "Invalid constant inputs: ", node->inputs->data[0],
                    ", ", node->inputs->data[1]);
    if (bias) absl::StrAppend(error, ", ", node->inputs->data[2]);
    return false;
  }
  // Check fused activation.
  if (!CheckFusedActivation(node, params->activation)
           .ok()) {
    *error = absl::StrCat("Unsupported fused activation: ", params->activation);
    return false;
  }
  return true;
}

}  // namespace litert::ml_drift::ir
