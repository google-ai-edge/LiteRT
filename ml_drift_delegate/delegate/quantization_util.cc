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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#include "tflite/kernels/internal/runtime_shape.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/util.h"

namespace ml_drift {
namespace delegate {
namespace {

struct ValidatedQuantizedLayout {
  size_t element_count = 0;
  size_t dequantized_bytes = 0;
};

struct QuantizedTensorPair {
  TfLiteTensor* orig_tensor = nullptr;
  absl::Span<float> fp32_span;
};

absl::StatusOr<ValidatedQuantizedLayout> ValidateQuantizedTensor(
    const TfLiteTensor& quantized) {
  if (quantized.type != kTfLiteInt8 && quantized.type != kTfLiteUInt8) {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported quantized tensor type: ", quantized.type));
  }
  int checked_count_int = 0;
  if (tflite::CheckedNumElements(quantized.dims, checked_count_int) !=
      kTfLiteOk) {
    return absl::InvalidArgumentError(
        "Quantized tensor has null, negative, or overflowing dimensions.");
  }
  const size_t element_count = static_cast<size_t>(checked_count_int);
  size_t dequantized_bytes = 0;
  if (tflite::MultiplyAndCheckOverflow(element_count, sizeof(float),
                                       &dequantized_bytes) != kTfLiteOk) {
    return absl::InvalidArgumentError(
        "Quantized tensor dequantized byte size overflows size_t.");
  }
  if (element_count == 0) {
    return ValidatedQuantizedLayout{
        .element_count = 0,
        .dequantized_bytes = 0,
    };
  }
  if (quantized.bytes < element_count) {
    return absl::FailedPreconditionError(
        absl::StrCat("Quantized tensor byte size (", quantized.bytes,
                     ") is smaller than element count (", element_count, ")."));
  }
  if (quantized.data.raw == nullptr) {
    return absl::FailedPreconditionError(
        "Quantized tensor has null data pointer for non-empty shape.");
  }
  return ValidatedQuantizedLayout{
      .element_count = element_count,
      .dequantized_bytes = dequantized_bytes,
  };
}

absl::Status ValidateTensorIndex(const TfLiteContext& context, int64_t index,
                                 absl::string_view label) {
  if (index < 0 || index > std::numeric_limits<int>::max() ||
      static_cast<uint64_t>(index) >= context.tensors_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Out-of-range ", label, " tensor index ", index,
                     " (tensors_size=", context.tensors_size, ")."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<QuantizedTensorPair>> ResolveQuantizedTensorPair(
    TfLiteContext& context, int64_t fp32_index, absl::string_view io_label,
    const absl::flat_hash_map<int, int>& quant_conversion_map) {
  ABSL_RETURN_IF_ERROR(ValidateTensorIndex(context, fp32_index, io_label));
  const auto quant_it = quant_conversion_map.find(static_cast<int>(fp32_index));
  TfLiteTensor& fp32_tensor = context.tensors[fp32_index];
  if (quant_it == quant_conversion_map.end() ||
      fp32_tensor.type != kTfLiteFloat32) {
    return std::nullopt;
  }
  ABSL_RETURN_IF_ERROR(
      ValidateTensorIndex(context, quant_it->second, "original quantized"));
  const size_t float_count =
      fp32_tensor.data.f != nullptr ? fp32_tensor.bytes / sizeof(float) : 0;
  return QuantizedTensorPair{
      .orig_tensor = &context.tensors[quant_it->second],
      .fp32_span = absl::MakeSpan(fp32_tensor.data.f, float_count),
  };
}

}  // namespace

absl::StatusOr<size_t> ValidatedDequantizedBytes(
    const TfLiteTensor& quantized) {
  ABSL_ASSIGN_OR_RETURN(const ValidatedQuantizedLayout layout,
                        ValidateQuantizedTensor(quantized));
  return layout.dequantized_bytes;
}

absl::Status DequantizeTo(const TfLiteTensor& quantized,
                          absl::Span<float> dst) {
  ABSL_ASSIGN_OR_RETURN(const ValidatedQuantizedLayout layout,
                        ValidateQuantizedTensor(quantized));
  if (layout.element_count == 0) {
    return absl::OkStatus();
  }
  if (dst.data() == nullptr || dst.size() < layout.element_count) {
    return absl::InvalidArgumentError(
        absl::StrCat("Destination float span size (", dst.size(),
                     ") is smaller than quantized element count (",
                     layout.element_count, ")."));
  }
  const tflite::RuntimeShape shape = tflite::GetTensorShape(&quantized);
  tflite::DequantizationParams op_params;
  op_params.zero_point = quantized.params.zero_point;
  op_params.scale = quantized.params.scale;
  if (quantized.type == kTfLiteInt8) {
    tflite::optimized_ops::Dequantize(op_params, shape, quantized.data.int8,
                                      shape, dst.data());
  } else {
    tflite::optimized_ops::Dequantize(op_params, shape, quantized.data.uint8,
                                      shape, dst.data());
  }
  return absl::OkStatus();
}

absl::Status QuantizeFrom(absl::Span<const float> src,
                          TfLiteTensor& quantized) {
  ABSL_ASSIGN_OR_RETURN(const ValidatedQuantizedLayout layout,
                        ValidateQuantizedTensor(quantized));
  if (layout.element_count == 0) {
    return absl::OkStatus();
  }
  if (src.data() == nullptr || src.size() < layout.element_count) {
    return absl::InvalidArgumentError(
        absl::StrCat("Source float span size (", src.size(),
                     ") is smaller than quantized element count (",
                     layout.element_count, ")."));
  }
  const tflite::RuntimeShape shape = tflite::GetTensorShape(&quantized);
  tflite::QuantizationParams op_params;
  op_params.zero_point = quantized.params.zero_point;
  op_params.scale = quantized.params.scale;
  if (quantized.type == kTfLiteInt8) {
    tflite::optimized_ops::AffineQuantize(op_params, shape, src.data(), shape,
                                          quantized.data.int8);
  } else {
    tflite::optimized_ops::AffineQuantize(op_params, shape, src.data(), shape,
                                          quantized.data.uint8);
  }
  return absl::OkStatus();
}

absl::Status DequantizeInputs(
    TfLiteContext* context, const std::vector<int64_t>& input_indices,
    const absl::flat_hash_map<int, int>& quant_conversion_map) {
  if (context == nullptr || context->tensors == nullptr) {
    return absl::InvalidArgumentError(
        "Null TfLiteContext or context->tensors.");
  }
  for (const int64_t index : input_indices) {
    ABSL_ASSIGN_OR_RETURN(const std::optional<QuantizedTensorPair> pair,
                          ResolveQuantizedTensorPair(*context, index, "input",
                                                     quant_conversion_map));
    if (pair.has_value()) {
      ABSL_RETURN_IF_ERROR(DequantizeTo(*pair->orig_tensor, pair->fp32_span));
    }
  }
  return absl::OkStatus();
}

absl::Status QuantizeOutputs(
    TfLiteContext* context, const std::vector<int64_t>& output_indices,
    const absl::flat_hash_map<int, int>& quant_conversion_map) {
  if (context == nullptr || context->tensors == nullptr) {
    return absl::InvalidArgumentError(
        "Null TfLiteContext or context->tensors.");
  }
  for (const int64_t index : output_indices) {
    ABSL_ASSIGN_OR_RETURN(const std::optional<QuantizedTensorPair> pair,
                          ResolveQuantizedTensorPair(*context, index, "output",
                                                     quant_conversion_map));
    if (pair.has_value()) {
      ABSL_RETURN_IF_ERROR(QuantizeFrom(pair->fp32_span, *pair->orig_tensor));
    }
  }
  return absl::OkStatus();
}

}  // namespace delegate
}  // namespace ml_drift
