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

#include "litert/vendors/samsung/compiler/builders/transpose_conv_op_builder.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"
#include "tflite/schema/schema_generated.h"

namespace litert::samsung {

constexpr int32_t kOutputShapeIndex = 0;
constexpr int32_t kInputIndex = 2;
constexpr int32_t kKernelIndex = 1;
constexpr int32_t kBiasIndex = 3;
constexpr int32_t kOutputIndex = 0;
constexpr int32_t kIOHeightIndex = 1;
constexpr int32_t kIOWidthIndex = 2;
constexpr int32_t kIOChannelIndex = 3;
constexpr int32_t kKernelHeightIndex = 1;
constexpr int32_t kKernelWidthIndex = 2;

Expected<OpWrapper> BuildTransposeConvOp(const Op& op) {
  OpWrapper op_wrapper("DECONV2D");

  const auto input = std::move(op.Inputs()[kInputIndex]);
  const auto kernel = std::move(op.Inputs()[kKernelIndex]);
  op_wrapper.AddInput(input).AddInput(kernel);
  if (op.Inputs().size() > kBiasIndex) {
    op_wrapper.AddInput(op.Inputs()[kBiasIndex]);
  }
  for (const auto& output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }

  auto kernel_dimensions = GetDimensions(kernel);
  int32_t kernel_h = kernel_dimensions[kKernelHeightIndex];
  int32_t kernel_w = kernel_dimensions[kKernelWidthIndex];
  op_wrapper.AddParam("kernel_h", kernel_h);
  op_wrapper.AddParam("kernel_w", kernel_w);

  int32_t stride_h = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTransposeConvStrideHOption(op.Get(), &stride_h));

  int32_t stride_w = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTransposeConvStrideWOption(op.Get(), &stride_w));

  op_wrapper.AddParam("stride_w", stride_w);
  op_wrapper.AddParam("stride_h", stride_h);

  auto input_dimensions = GetDimensions(input);
  auto output_dimensions = GetDimensions(op.Outputs()[kOutputIndex]);
  op_wrapper.AddParam("in_channels", input_dimensions[kIOChannelIndex]);
  op_wrapper.AddParam("out_channels", output_dimensions[kIOChannelIndex]);

  uint32_t padding = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTransposeConvPaddingOption(op.Get(), &padding));
  // Compute explicit padding according to padding type
  std::vector<int32_t> explicit_paddings(4, 0);
  if (padding == ::tflite::Padding_SAME) {
    const auto [padding_top, padding_bottom] =
        GetExplicitPadding(output_dimensions[kIOHeightIndex], kernel_h,
                           input_dimensions[kIOHeightIndex], stride_h, 1);
    const auto [padding_left, padding_right] =
        GetExplicitPadding(output_dimensions[kIOWidthIndex], kernel_w,
                           input_dimensions[kIOWidthIndex], stride_w, 1);
    explicit_paddings = {padding_top, padding_left, padding_bottom,
                         padding_right};
  }
  op_wrapper.AddParam("padding", "EXPLICIT");
  op_wrapper.AddParam("padding_type", "CONSTANT");
  op_wrapper.AddParam("explicit_padding", explicit_paddings);

  uint32_t tfl_fused_activation;
  LITERT_RETURN_IF_ERROR(LiteRtGetTransposeConvFusedActivationOption(
      op.Get(), &tfl_fused_activation));

  LITERT_ASSIGN_OR_RETURN(auto activation,
                          GetFusedActivationName(tfl_fused_activation));
  op_wrapper.AddParam("activation", activation);

  return op_wrapper;
}

}  // namespace litert::samsung
