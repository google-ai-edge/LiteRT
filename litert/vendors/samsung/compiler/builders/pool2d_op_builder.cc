// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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
#include "litert/vendors/samsung/compiler/builders/pool2d_op_builder.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"
#include "tflite/schema/schema_generated.h"

namespace litert::samsung {
constexpr int32_t kInputIndex = 0;
constexpr int32_t kOutputIndex = 0;
constexpr int32_t kHeightIndex = 1;
constexpr int32_t kWidthIndex = 2;

Expected<OpWrapper> BuildGeneralPool2dOp(const Op& op, const std::string& type,
                                         int32_t stride_h, int32_t stride_w,
                                         int32_t filter_h, int32_t filter_w,
                                         uint32_t padding,
                                         uint32_t tfl_fused_activation = 0) {
  OpWrapper op_wrapper(type);

  for (const auto& input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto& output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }

  const int32_t dilation_h = 1, dilation_w = 1;
  std::vector<int32_t> strides = {stride_h, stride_w};
  std::vector<int32_t> filter = {filter_h, filter_w};
  op_wrapper.AddParam("strides", strides)
      .AddParam("kernel_shape", filter)
      .AddParam("dilations", std::vector<int32_t>{dilation_h, dilation_w});

  auto input_dimensions = GetDimensions(op.Inputs()[kInputIndex]);
  auto output_dimensions = GetDimensions(op.Outputs()[kOutputIndex]);

  std::vector<int32_t> explicit_paddings(4, 0);
  if (padding == ::tflite::Padding_SAME) {
    const auto [padding_top, padding_bottom] = GetExplicitPadding(
        input_dimensions[kHeightIndex], filter_h,
        output_dimensions[kHeightIndex], stride_h, dilation_h);
    const auto [padding_left, padding_right] = GetExplicitPadding(
        input_dimensions[kWidthIndex], filter_w, output_dimensions[kWidthIndex],
        stride_w, dilation_w);
    explicit_paddings = {padding_top, padding_left, padding_bottom,
                         padding_right};
  }
  op_wrapper.AddParam("pads", explicit_paddings);
  op_wrapper.AddParam("count_include_pad", false);

  return op_wrapper;
}

Expected<OpWrapper> BuildMaxPool2dOp(const Op& op) {
  int32_t stride_h;
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dStrideHOption(op.Get(), &stride_h));

  int32_t stride_w;
  LITERT_RETURN_IF_ERROR(LiteRtGetMaxPool2dStrideWOption(op.Get(), &stride_w));

  int32_t filter_h;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFilterHeightOption(op.Get(), &filter_h));

  int32_t filter_w;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFilterWidthOption(op.Get(), &filter_w));

  uint32_t padding;
  LiteRtGetMaxPool2dPaddingOption(op.Get(), &padding);

  uint32_t tfl_fused_activation;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetMaxPool2dFusedActivationOption(op.Get(), &tfl_fused_activation));

  return BuildGeneralPool2dOp(op, "MaxPool", stride_h, stride_w, filter_h,
                              filter_w, padding, tfl_fused_activation);
}

Expected<OpWrapper> BuildAvgPool2dOp(const Op& op) {
  int32_t stride_h;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dStrideHOption(op.Get(), &stride_h));

  int32_t stride_w;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dStrideWOption(op.Get(), &stride_w));

  int32_t filter_h;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dFilterHeightOption(op.Get(), &filter_h));

  int32_t filter_w;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetAveragePool2dFilterWidthOption(op.Get(), &filter_w));

  uint32_t padding;
  LiteRtGetAveragePool2dPaddingOption(op.Get(), &padding);

  uint32_t tfl_fused_activation;
  LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dFusedActivationOption(
      op.Get(), &tfl_fused_activation));

  return BuildGeneralPool2dOp(op, "AvgPool", stride_h, stride_w, filter_h,
                              filter_w, padding, tfl_fused_activation);
}

}  // namespace litert::samsung
