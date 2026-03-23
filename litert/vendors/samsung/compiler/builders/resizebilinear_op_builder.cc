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

#include "litert/vendors/samsung/compiler/builders/resizebilinear_op_builder.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {
constexpr int32_t kInputIndex = 0;
constexpr int32_t kOutputIndex = 0;
constexpr int32_t kHeightIndex = 2;
constexpr int32_t kWidthIndex = 3;

Expected<OpWrapper> BuildResizeBilinearOp(const Op &op) {
  OpWrapper op_wrapper("ResizeBilinear");

  for (const auto &input : op.Inputs()) {
    op_wrapper.AddInput(input);
  }
  for (const auto &output : op.Outputs()) {
    op_wrapper.AddOutput(output);
  }

  bool align_corners;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetResizeBilinearAlignCornersOption(op.Get(), &align_corners));

  bool half_pixel_centers;
  LITERT_RETURN_IF_ERROR(LiteRtGetResizeBilinearHalfPixelCenterOption(
      op.Get(), &half_pixel_centers));

  auto input_dimensions = GetDimensions(op.Inputs()[kInputIndex]);
  auto output_dimensions = GetDimensions(op.Outputs()[kOutputIndex]);
  std::vector<double> upsampling_factor;

  upsampling_factor.push_back(
      static_cast<double>(output_dimensions[kHeightIndex]) /
      input_dimensions[kHeightIndex]);
  upsampling_factor.push_back(
      static_cast<double>(output_dimensions[kWidthIndex]) /
      input_dimensions[kWidthIndex]);

  op_wrapper.AddParam("align_corners", align_corners)
      .AddParam("half_pixel_centers", half_pixel_centers)
      .AddParam("upsampling_factor", upsampling_factor);

  return op_wrapper;
}
}  // namespace litert::samsung
