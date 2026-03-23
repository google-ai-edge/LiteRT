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

#include "litert/vendors/samsung/compiler/builders/strided_slice_op_builder.h"

#include "litert/c/litert_op_options.h"
#include "litert/vendors/samsung/compiler/builders/utils.h"

namespace litert::samsung {

constexpr int kIOIndex = 0;
constexpr int kBeginIndex = 1;
constexpr int kEndIndex = 2;
constexpr int kStridesIndex = 3;

Expected<OpWrapper> BuildStridedSliceOp(const Op &op) {
  OpWrapper op_wrapper("StridedSlice");

  op_wrapper.AddInput(op.Inputs()[kIOIndex]);
  op_wrapper.AddOutput(op.Outputs()[kIOIndex]);

  int32_t begin_mask = 0, end_mask = 0, ellipsis_mask = 0, new_axis_mask = 0,
          shrink_axis_mask = 0;
  bool offset = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceBeginMaskOption(op.Get(), &begin_mask));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceEndMaskOption(op.Get(), &end_mask));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceEllipsisMaskOption(op.Get(), &ellipsis_mask));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceNewAxisMaskOption(op.Get(), &new_axis_mask));
  LITERT_RETURN_IF_ERROR(
      LiteRtGetStridedSliceShrinkAxisMaskOption(op.Get(), &shrink_axis_mask));
  LITERT_RETURN_IF_ERROR(LiteRtGetStridedSliceOffsetOption(op.Get(), &offset));

  if (offset) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Doesn't support StridedSlice offset is true");
  }

  LITERT_ASSIGN_OR_RETURN(auto begin,
                          GetWeightDataAs<int32_t>(op.Inputs()[kBeginIndex]));
  LITERT_ASSIGN_OR_RETURN(auto end,
                          GetWeightDataAs<int32_t>(op.Inputs()[kEndIndex]));
  LITERT_ASSIGN_OR_RETURN(auto strides,
                          GetWeightDataAs<int32_t>(op.Inputs()[kStridesIndex]));

  op_wrapper.AddParam("ellipsis_mask", ellipsis_mask);
  op_wrapper.AddParam("new_axis_mask", new_axis_mask);
  op_wrapper.AddParam("shrink_axis_mask", shrink_axis_mask);
  op_wrapper.AddParam("offset", offset);

  if (begin.size() != end.size()) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "The begin and end of StridedSlice should be same");
  }

  auto input_dimensions = GetDimensions(op.Inputs()[kIOIndex]);
  for (int idx = 0; idx < begin.size(); ++idx) {
    if ((begin_mask & (1 << idx)) != 0) {
      begin[idx] = strides[idx] > 0 ? 0 : input_dimensions[idx] - 1;
    }
    if ((end_mask & (1 << idx)) != 0) {
      end[idx] = strides[idx] > 0 ? input_dimensions[idx] : -1;
    }
  }

  op_wrapper.AddParam("starts", begin);
  op_wrapper.AddParam("ends", end);
  op_wrapper.AddParam("steps", strides);

  return op_wrapper;
}
}  // namespace litert::samsung
