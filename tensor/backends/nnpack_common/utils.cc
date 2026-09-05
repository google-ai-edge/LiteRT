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

#include "tensor/backends/nnpack_common/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

absl::StatusOr<ActivationBounds> GetActivationBounds(
    FusedActivation activation, absl::string_view op_name) {
  ActivationBounds b;
  switch (activation) {
    case kActNone:
      b.output_min = -kInf;
      b.output_max = kInf;
      break;
    case kActRelu:
      b.output_min = 0.0f;
      b.output_max = kInf;
      break;
    case kActRelu6:
      b.output_min = 0.0f;
      b.output_max = 6.0f;
      break;
    case kActReluN1To1:
      b.output_min = -1.0f;
      b.output_max = 1.0f;
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("%s: unsupported fused activation %d", op_name,
                          static_cast<int>(activation)));
  }
  return b;
}

PaddingValues ComputePadding(Padding padding, int input_h, int input_w,
                             int filter_h, int filter_w, int stride_h,
                             int stride_w, int dilation_h, int dilation_w) {
  PaddingValues p{0, 0, 0, 0};
  if (padding == kPaddingSame) {
    const int eff_filter_h = (filter_h - 1) * dilation_h + 1;
    const int eff_filter_w = (filter_w - 1) * dilation_w + 1;
    const int out_h = static_cast<int>(
        std::ceil(static_cast<float>(input_h) / static_cast<float>(stride_h)));
    const int out_w = static_cast<int>(
        std::ceil(static_cast<float>(input_w) / static_cast<float>(stride_w)));
    const int pad_h =
        std::max(0, (out_h - 1) * stride_h + eff_filter_h - input_h);
    const int pad_w =
        std::max(0, (out_w - 1) * stride_w + eff_filter_w - input_w);
    p.top = pad_h / 2;
    p.bottom = pad_h - p.top;
    p.left = pad_w / 2;
    p.right = pad_w - p.left;
  }
  return p;
}

absl::StatusOr<TransposeConvPaddingValues> ComputeTransposeConvPadding(
    Padding padding, int input_h, int input_w, int filter_h, int filter_w,
    int stride_h, int stride_w, int output_h, int output_w) {
  if (input_h <= 0 || input_w <= 0 || filter_h <= 0 || filter_w <= 0 ||
      stride_h <= 0 || stride_w <= 0 || output_h <= 0 || output_w <= 0) {
    return absl::InvalidArgumentError(
        "TransposeConv expects positive input/filter/stride/output sizes");
  }

  auto compute_dim = [&](int input, int filter, int stride, int output,
                         uint32_t* pad_before, uint32_t* pad_after,
                         uint32_t* adj,
                         absl::string_view dim_name) -> absl::Status {
    const int base = (input - 1) * stride + filter;
    int pad_total = 0;
    int adj_local = 0;
    if (output >= base) {
      adj_local = output - base;
    } else {
      pad_total = base - output;
    }
    if (adj_local >= stride) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TransposeConv %s adjustment (%d) must be < stride (%d)", dim_name,
          adj_local, stride));
    }
    if (padding == kPaddingValid && pad_total != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TransposeConv %s padding (%d) not allowed for VALID padding",
          dim_name, pad_total));
    }
    const int pad_before_local = pad_total / 2;
    const int pad_after_local = pad_total - pad_before_local;
    *pad_before = pad_before_local;
    *pad_after = pad_after_local;
    *adj = adj_local;
    return absl::OkStatus();
  };

  TransposeConvPaddingValues res{0, 0, 0, 0, 0, 0};
  LRT_TENSOR_RETURN_IF_ERROR(compute_dim(input_h, filter_h, stride_h, output_h,
                                         &res.top, &res.bottom, &res.adj_h,
                                         "height"));
  LRT_TENSOR_RETURN_IF_ERROR(compute_dim(input_w, filter_w, stride_w, output_w,
                                         &res.left, &res.right, &res.adj_w,
                                         "width"));
  return res;
}

absl::StatusOr<std::vector<size_t>> ToNnpackDims(
    absl::Span<const int32_t> shape) {
  std::vector<size_t> dims;
  dims.reserve(shape.size());
  for (int32_t dim : shape) {
    if (dim < 0) {
      return absl::InvalidArgumentError("Negative tensor dimension.");
    }
    dims.push_back(static_cast<size_t>(dim));
  }
  return dims;
}

absl::Status ValidateFp32OrQuantizedConstantWeights(const graph::Tensor& tensor,
                                                    absl::string_view op_name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
  if (info.type == Type::kFP32) {
    return absl::OkStatus();
  }
  if (info.type != Type::kI8) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s only supports FP32 weights or quantized INT8 "
                        "constant weights. Got type id %d.",
                        op_name, static_cast<int>(info.type)));
  }
  if (info.buffer == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s INT8 weights must be constant tensors.", op_name));
  }
  if (info.quantization == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s INT8 weights require quantization metadata.", op_name));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& quantization,
      info.quantization->As<const graph::PerChannelAffineQuantization>());
  if (quantization.scales.empty() || quantization.zero_points.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s INT8 weights require non-empty scales and "
                        "zero-points.",
                        op_name));
  }
  return absl::OkStatus();
}

}  // namespace litert::tensor
