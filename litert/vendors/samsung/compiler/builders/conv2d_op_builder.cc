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

#include "litert/vendors/samsung/compiler/builders/conv2d_op_builder.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"
#include "tflite/schema/schema_generated.h"

namespace litert::samsung {

constexpr int32_t kInputIndex = 0;
constexpr int32_t kKernelIndex = 1;
constexpr int32_t kBiasIndex = 2;
constexpr int32_t kOutputIndex = 0;
constexpr int32_t kIOHeightIndex = 1;
constexpr int32_t kIOWidthIndex = 2;
constexpr int32_t kIOChannelIndex = 3;
constexpr int32_t kKernelHeightIndex = 1;
constexpr int32_t kKernelWidthIndex = 2;

Expected<OpWrapper> BuildGeneralConvOp(const Op &op, const std::string &type,
                                       int32_t stride_h, int32_t stride_w,
                                       int32_t dilation_h, int32_t dilation_w,
                                       int32_t padding,
                                       int32_t depth_multiplier,
                                       uint32_t tfl_fused_activation = 0) {
  OpWrapper op_wrapper(type);

  for (const auto &input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto &output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }

  auto kernel_dimensions = GetDimensions(op.Inputs()[kKernelIndex]);

  int32_t kernel_h = kernel_dimensions[kKernelHeightIndex];
  int32_t kernel_w = kernel_dimensions[kKernelWidthIndex];
  op_wrapper.AddParam("kernel_h", kernel_h);
  op_wrapper.AddParam("kernel_w", kernel_w);

  op_wrapper.AddParam("stride_w", stride_w);
  op_wrapper.AddParam("stride_h", stride_h);
  op_wrapper.AddParam("dilation_w", dilation_w);
  op_wrapper.AddParam("dilation_h", dilation_h);

  auto input_dimensions = GetDimensions(op.Inputs()[kInputIndex]);
  auto output_dimensions = GetDimensions(op.Outputs()[kOutputIndex]);
  op_wrapper.AddParam("in_channels", input_dimensions[kIOChannelIndex]);
  op_wrapper.AddParam("out_channels", output_dimensions[kIOChannelIndex]);

  std::vector<int32_t> explicit_paddings(4, 0);
  if (padding == ::tflite::Padding_SAME) {
    const auto [padding_top, padding_bottom] = GetExplicitPadding(
        input_dimensions[kIOHeightIndex], kernel_h,
        output_dimensions[kIOHeightIndex], stride_h, dilation_h);
    const auto [padding_left, padding_right] = GetExplicitPadding(
        input_dimensions[kIOWidthIndex], kernel_w,
        output_dimensions[kIOWidthIndex], stride_w, dilation_w);
    explicit_paddings = {padding_top, padding_left, padding_bottom,
                         padding_right};
  }
  op_wrapper.AddParam("padding", "EXPLICIT");
  op_wrapper.AddParam("padding_type", "CONSTANT");
  op_wrapper.AddParam("explicit_padding", explicit_paddings);
  op_wrapper.AddParam("group", depth_multiplier);

  auto activation = GetFusedActivationName(tfl_fused_activation);
  if (!activation) {
    return activation.Error();
  }
  op_wrapper.AddParam("activation", *activation);

  return op_wrapper;
}

Expected<OpWrapper> BuildConv2dOp(const Op &op) {
  int32_t stride_h = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideHOption(op.Get(), &stride_h));

  int32_t stride_w = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideWOption(op.Get(), &stride_w));

  int32_t dilation_h = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dDilationHOption(op.Get(), &dilation_h));

  int32_t dilation_w = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dDilationWOption(op.Get(), &dilation_w));

  uint32_t padding = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetConv2dPaddingOption(op.Get(), &padding));

  uint32_t tfl_fused_activation;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetConv2dFusedActivationOption(op.Get(), &tfl_fused_activation));

  const int32_t depth_multiplier = 1;

  return BuildGeneralConvOp(op, "CONV2D", stride_h, stride_w, dilation_h,
                            dilation_w, padding, depth_multiplier,
                            tfl_fused_activation);
}

Expected<OpWrapper> BuildDepthwiseConv2dOp(const Op &op) {
  int32_t stride_h = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dStrideHOption(op.Get(), &stride_h));

  int32_t stride_w = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dStrideWOption(op.Get(), &stride_w));

  int32_t dilation_h = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dDilationHOption(op.Get(), &dilation_h));

  int32_t dilation_w = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dDilationWOption(op.Get(), &dilation_w));

  uint32_t padding = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDepthwiseConv2dPaddingOption(op.Get(), &padding));

  int32_t depth_multiplier = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dDepthMultiplierOption(
      op.Get(), &depth_multiplier));

  uint32_t tfl_fused_activation;
  LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dFusedActivationOption(
      op.Get(), &tfl_fused_activation));

  return BuildGeneralConvOp(op, "DWCONV2D", stride_h, stride_w, dilation_h,
                            dilation_w, padding, depth_multiplier,
                            tfl_fused_activation);
}

}  // namespace litert::samsung
