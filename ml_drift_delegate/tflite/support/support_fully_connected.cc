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

#include "ml_drift_delegate/tflite/support/support_fully_connected.h"

#include <optional>
#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "ml_drift_delegate/tflite/support/support_aux.h"
#include "tflite/c/common.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/kernels/kernel_util.h"

using ::tflite::IsConstantTensor;

namespace litert::ml_drift::ir {

bool IsFullyConnectedSupported(
    const TfLiteContext* absl_nonnull context,
    const TfLiteNode* absl_nonnull node,
    const TfLiteRegistration* absl_nonnull registration,
    std::string* absl_nonnull error) {
  // Check version.
  if (registration->version < 1 || registration->version > 17) {
    *error = absl::StrCat("Unsupported version: ", registration->version);
    return false;
  }
  // Check number of inputs.
  const TfLiteIntArray* inputs = node->inputs;
  if (inputs->size < 2 || inputs->size > 3) {
    *error = absl::StrCat("Invalid number of inputs: ", inputs->size);
    return false;
  }
  // Check number of outputs.
  const TfLiteIntArray* outputs = node->outputs;
  // Check number of outputs.
  if (outputs->size != 1) {
    *error = absl::StrCat("Invalid number of outputs: ", outputs->size,
                          ", must be 1");
    return false;
  }

  // Validate tensor IDs.
  for (int i = 0; i < 2; ++i) {
    if (!ValidateTensorId(*context, inputs->data[i],
                          absl::StrCat("inputs[", i, "]"), *error)) {
      return false;
    }
  }
  std::optional<int> bias_id;
  if (inputs->size == 3 && inputs->data[2] != kTfLiteOptionalTensor) {
    if (!ValidateTensorId(*context, inputs->data[2], "inputs[2]", *error)) {
      return false;
    }
    bias_id = inputs->data[2];
  }
  if (!ValidateTensorIds(*context, *outputs, "outputs", *error)) {
    return false;
  }

  const int src_id = inputs->data[0];
  const int weights_id = inputs->data[1];
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
  const TfLiteTensor& input = context->tensors[src_id];
  if (!CheckTensorDtype(input, supported_dtypes, "inputs[0]", *error)) {
    return false;
  }
  const TfLiteTensor& weights = context->tensors[weights_id];
  const absl::flat_hash_set<TfLiteType> supported_weights_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt2,
      kTfLiteInt4,
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
        CheckPopulateTensor<::ml_drift::OHWI, ::ml_drift::DataType::kFloat32>(
            &weights);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  }
  const TfLiteTensor* bias =
      bias_id.has_value() ? &context->tensors[*bias_id] : nullptr;
  const absl::flat_hash_set<TfLiteType> supported_bias_dtypes = {
      // clang-format off
      // go/keep-sorted start numeric=yes
      kTfLiteFloat16,
      kTfLiteFloat32,
      kTfLiteInt32,
      // go/keep-sorted end
      // clang-format on
  };
  if (bias &&
      !CheckTensorDtype(*bias, supported_bias_dtypes, "inputs[2]", *error)) {
    return false;
  }
  if (bias && IsConstantTensor(bias)) {
    const absl::Status status =
        CheckPopulateTensor<::ml_drift::Linear, ::ml_drift::DataType::kFloat32>(
            bias);
    if (!status.ok()) {
      *error = status.message();
      return false;
    }
  }
  const TfLiteTensor& output = context->tensors[output_id];
  if (!CheckTensorDtype(output, supported_dtypes, "outputs[0]", *error)) {
    return false;
  }
  // Check dims.
  if (!CheckTensorDims(input, /*min_dims=*/0, /*max_dims=*/4, "inputs[0]",
                       *error)) {
    return false;
  }
  if (weights.quantization.type == kTfLiteAffineQuantization) {
    if (weights.type != kTfLiteInt8 && weights.type != kTfLiteInt4 &&
        weights.type != kTfLiteInt2) {
      *error = absl::StrCat("Unsupported quantization type for weights: ",
                            weights.quantization.type);
      return false;
    }
    const TfLiteIntArray& weight_dims = *weights.dims;
    if (weight_dims.size == 4) {
      if (weight_dims.data[1] != 1 || weight_dims.data[2] != 1) {
        *error =
            absl::StrCat("Unsupported HW weight dimensions for quantized FC: ",
                         weight_dims.data[1], " , ", weight_dims.data[2]);
        return false;
      }
    } else if (weight_dims.size != 2) {
      *error = absl::StrCat("Unsupported ", weight_dims.size,
                            "D weights for quantized FC");
      return false;
    }
  } else if (weights.quantization.type == kTfLiteBlockwiseQuantizationV2) {
    // N-D blockwise quantization (FULLY_CONNECTED versions 16 and 17): the
    // scales and zero points are separate tensors, e.g. centered int2.
    if (!IsConstantTensor(&weights)) {
      *error = "Expected constant weights for N-D blockwise quantization.";
      return false;
    }
    // Standard TFLite FULLY_CONNECTED weights are 2-D [O, I]; batched 3-D/4-D
    // blockwise v2 weights are handled by dedicated composite ops.
    if (weights.dims == nullptr || weights.dims->size != 2) {
      *error = "Expected 2D weights for N-D blockwise quantization.";
      return false;
    }
    const absl::StatusOr<::litert::ml_drift::BlockwiseV2Params>
        blockwise_params = ::litert::ml_drift::GetBlockwiseV2Params(
            context->tensors, context->tensors_size, weights);
    if (!blockwise_params.ok()) {
      *error = blockwise_params.status().message();
      return false;
    }
    // Weights that a quantized FullyConnected node can't consume are
    // dequantized on the host during conversion, which needs their data.
    ::litert::ml_drift::BlockwiseV2WeightsInfo blockwise_info;
    if (weights.data.raw_const == nullptr &&
        !::litert::ml_drift::GetBlockwiseV2WeightsInfo(
             weights, *blockwise_params, &blockwise_info)
             .ok()) {
      *error =
          "N-D blockwise quantized weights without host data can't be "
          "dequantized on the host.";
      return false;
    }
    if (bias != nullptr) {
      *error = "Bias is not supported with N-D blockwise quantized weights.";
      return false;
    }
  } else {
    if (!CheckTensorDims(weights, /*min_dims=*/0, /*max_dims=*/4, "inputs[1]",
                         *error)) {
      return false;
    }
    const int total_weight_elements = tflite::NumElements(&weights);
    int input_feature_size_flat = 1;
    for (int i = 1; i < input.dims->size; ++i) {
      input_feature_size_flat *= input.dims->data[i];
    }
    const int expected_weight_elements_flat =
        input_feature_size_flat * output.dims->data[output.dims->size - 1];

    const int expected_weight_elements_batch =
        input.dims->data[input.dims->size - 1] *
        output.dims->data[output.dims->size - 1];

    if (total_weight_elements != expected_weight_elements_flat &&
        total_weight_elements != expected_weight_elements_batch) {
      *error = absl::StrCat("Weights size mismatch for FC reshape. Got ",
                            total_weight_elements, " but expected ",
                            expected_weight_elements_flat, " (flat) or ",
                            expected_weight_elements_batch, " (batch)");
      return false;
    }
  }
  if (bias && !CheckTensorDims(*bias, /*min_dims=*/1, /*max_dims=*/1,
                               "inputs[2]", *error)) {
    return false;
  }
  if (!CheckTensorDims(output, /*min_dims=*/0, /*max_dims=*/4, "outputs[0]",
                       *error)) {
    return false;
  }

  const auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  if (!params) {
    *error = "Incompatible node->builtin_data";
    return false;
  }
  if (params->weights_format != kTfLiteFullyConnectedWeightsFormatDefault) {
    *error =
        absl::StrCat("Unsupported weights format: ", params->weights_format);
    return false;
  }
  if (params->quant_spec_size > 0) {
    // FULLY_CONNECTED version 15: a non-standard quantization contract.
    *error = "Fully connected with a quantization spec is not supported.";
    return false;
  }
  // Check const inputs.
  if (IsConstantTensor(&input)) {
    *error = absl::StrCat(
        "Constant input[0] is not supported for FullyConnected: ", src_id);
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
