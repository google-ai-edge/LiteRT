/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/common_nnpack/utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

NnpackActivationBounds GetActivationBounds(FusedActivation activation) {
  switch (activation) {
    case FusedActivation::kActRelu:
      return {.min = 0.0f, .max = std::numeric_limits<float>::infinity()};
    case FusedActivation::kActReluN1To1:
      return {.min = -1.0f, .max = 1.0f};
    case FusedActivation::kActRelu6:
      return {.min = 0.0f, .max = 6.0f};
    case FusedActivation::kActNone:
    default:
      return {.min = -std::numeric_limits<float>::infinity(),
              .max = std::numeric_limits<float>::infinity()};
  }
}

absl::StatusOr<NnpackPadding> ComputePadding(
    Padding padding, size_t input_height, size_t input_width,
    size_t kernel_height, size_t kernel_width, size_t stride_h, size_t stride_w,
    size_t dilation_h, size_t dilation_w) {
  if (padding != kPaddingSame) {
    return NnpackPadding{};
  }

  const size_t effective_kernel_height = (kernel_height - 1) * dilation_h + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation_w + 1;

  const size_t output_height = (input_height + stride_h - 1) / stride_h;
  const size_t output_width = (input_width + stride_w - 1) / stride_w;

  const size_t total_padding_h =
      std::max<size_t>(0, (output_height - 1) * stride_h +
                              effective_kernel_height - input_height);
  const size_t total_padding_w = std::max<size_t>(
      0, (output_width - 1) * stride_w + effective_kernel_width - input_width);

  return NnpackPadding{
      .top = total_padding_h / 2,
      .right = total_padding_w - total_padding_w / 2,
      .bottom = total_padding_h - total_padding_h / 2,
      .left = total_padding_w / 2,
  };
}

absl::StatusOr<NnpackTransposeConvPadding> ComputeTransposeConvPadding(
    Padding padding, size_t input_height, size_t input_width,
    size_t kernel_height, size_t kernel_width, size_t stride_h, size_t stride_w,
    size_t output_height, size_t output_width) {
  if (stride_h == 0 || stride_w == 0) {
    return absl::InvalidArgumentError("Strides must be non-zero");
  }
  if (input_height == 0 || input_width == 0) {
    return absl::InvalidArgumentError("Input dimensions must be non-zero");
  }

  const size_t expected_output_h =
      (input_height - 1) * stride_h + kernel_height;
  const size_t expected_output_w = (input_width - 1) * stride_w + kernel_width;

  if (padding == kPaddingValid) {
    if (output_height < expected_output_h || output_width < expected_output_w) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Requested output size (%zu, %zu) is smaller than "
          "base output size (%zu, %zu) for VALID padding",
          output_height, output_width, expected_output_h, expected_output_w));
    }
    const size_t adj_h = output_height - expected_output_h;
    const size_t adj_w = output_width - expected_output_w;
    if (adj_h >= stride_h || adj_w >= stride_w) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Output adjustment (%zu, %zu) exceeds stride (%zu, %zu)", adj_h,
          adj_w, stride_h, stride_w));
    }
    NnpackTransposeConvPadding result;
    result.top = 0;
    result.right = 0;
    result.bottom = 0;
    result.left = 0;
    result.adj_h = adj_h;
    result.adj_w = adj_w;
    return result;
  }

  if (padding == kPaddingSame) {
    const size_t same_output_h = input_height * stride_h;
    const size_t same_output_w = input_width * stride_w;
    if (output_height < same_output_h || output_width < same_output_w) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Requested output size (%zu, %zu) is smaller than "
          "SAME output size (%zu, %zu)",
          output_height, output_width, same_output_h, same_output_w));
    }
    const size_t adj_h = output_height - same_output_h;
    const size_t adj_w = output_width - same_output_w;
    if (adj_h >= stride_h || adj_w >= stride_w) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Output adjustment (%zu, %zu) exceeds stride (%zu, %zu)", adj_h,
          adj_w, stride_h, stride_w));
    }

    const size_t total_padding_h =
        kernel_height > stride_h ? kernel_height - stride_h : 0;
    const size_t total_padding_w =
        kernel_width > stride_w ? kernel_width - stride_w : 0;

    NnpackTransposeConvPadding result;
    result.top = total_padding_h / 2;
    result.right = total_padding_w - total_padding_w / 2;
    result.bottom = total_padding_h - total_padding_h / 2;
    result.left = total_padding_w / 2;
    result.adj_h = adj_h;
    result.adj_w = adj_w;
    return result;
  }

  return absl::InvalidArgumentError("Unsupported padding mode");
}

absl::StatusOr<std::vector<size_t>> ToNnpackDims(
    absl::Span<const int32_t> shape) {
  std::vector<size_t> dims;
  dims.reserve(shape.size());
  for (int32_t dim : shape) {
    if (dim < 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Dynamic/negative dimension %d not supported", dim));
    }
    dims.push_back(static_cast<size_t>(dim));
  }
  return dims;
}

absl::Status ValidateFp32OrQuantizedConstantWeights(
    absl::string_view op_name, const graph::TensorInformation& weights_info,
    bool is_external) {
  if (is_external) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: weights must be a constant tensor", op_name));
  }
  if (weights_info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: weights buffer must not be null", op_name));
  }
  LRT_TENSOR_RETURN_IF_ERROR(ValidateTensorType(
      op_name, weights_info,
      absl::Span<const Type>({Type::kFP32, Type::kI8, Type::kI4, Type::kI2})));
  return absl::OkStatus();
}

}  // namespace litert::tensor
