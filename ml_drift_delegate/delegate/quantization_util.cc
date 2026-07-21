// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/quantization_util.h"

#include <stdint.h>

#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"

namespace ml_drift {
namespace delegate {
namespace {

void DequantizeInput(
    TfLiteContext* context, int input_index,
    const absl::flat_hash_map<int, int>& quant_conversion_map) {
  if (quant_conversion_map.find(input_index) == quant_conversion_map.end()) {
    return;
  }
  int original_tensor_idx = quant_conversion_map.at(input_index);
  const TfLiteTensor& dequantized_tflite_tensor = context->tensors[input_index];
  const TfLiteTensor& original_tflite_tensor =
      context->tensors[original_tensor_idx];
  tflite::DequantizationParams op_params;
  op_params.zero_point = original_tflite_tensor.params.zero_point;
  op_params.scale = original_tflite_tensor.params.scale;
  if (original_tflite_tensor.type == kTfLiteInt8) {
    tflite::optimized_ops::Dequantize(
        op_params, tflite::GetTensorShape(&original_tflite_tensor),
        original_tflite_tensor.data.int8,
        tflite::GetTensorShape(&original_tflite_tensor),
        dequantized_tflite_tensor.data.f);
  } else if (original_tflite_tensor.type == kTfLiteUInt8) {
    tflite::optimized_ops::Dequantize(
        op_params, tflite::GetTensorShape(&original_tflite_tensor),
        original_tflite_tensor.data.uint8,
        tflite::GetTensorShape(&original_tflite_tensor),
        dequantized_tflite_tensor.data.f);
  }
}

void QuantizeOutput(TfLiteContext* context, int output_index,
                    const absl::flat_hash_map<int, int>& quant_conversion_map) {
  if (quant_conversion_map.find(output_index) == quant_conversion_map.end()) {
    return;
  }
  int original_tensor_idx = quant_conversion_map.at(output_index);
  const TfLiteTensor& dequantized_tflite_tensor =
      context->tensors[output_index];
  const TfLiteTensor& original_tflite_tensor =
      context->tensors[original_tensor_idx];
  tflite::QuantizationParams op_params;
  op_params.zero_point = original_tflite_tensor.params.zero_point;
  op_params.scale = original_tflite_tensor.params.scale;
  if (original_tflite_tensor.type == kTfLiteInt8) {
    tflite::optimized_ops::AffineQuantize(
        op_params, tflite::GetTensorShape(&original_tflite_tensor),
        dequantized_tflite_tensor.data.f,
        tflite::GetTensorShape(&original_tflite_tensor),
        original_tflite_tensor.data.int8);
  } else if (original_tflite_tensor.type == kTfLiteUInt8) {
    tflite::optimized_ops::AffineQuantize(
        op_params, tflite::GetTensorShape(&original_tflite_tensor),
        dequantized_tflite_tensor.data.f,
        tflite::GetTensorShape(&original_tflite_tensor),
        original_tflite_tensor.data.uint8);
  }
}

}  // namespace

absl::Status DequantizeInputs(
    TfLiteContext* context, const std::vector<int64_t>& input_indices,
    const absl::flat_hash_map<int, int>& quant_conversion_map) {
  for (auto index : input_indices) {
    DequantizeInput(context, static_cast<int>(index), quant_conversion_map);
  }
  return absl::OkStatus();
}

absl::Status QuantizeOutputs(
    TfLiteContext* context, const std::vector<int64_t>& output_indices,
    const absl::flat_hash_map<int, int>& quant_conversion_map) {
  for (auto index : output_indices) {
    QuantizeOutput(context, static_cast<int>(index), quant_conversion_map);
  }

  return absl::OkStatus();
}

}  // namespace delegate
}  // namespace ml_drift
