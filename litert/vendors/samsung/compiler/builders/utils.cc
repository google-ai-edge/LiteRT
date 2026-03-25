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
#include "litert/vendors/samsung/compiler/builders/utils.h"

#include "tflite/schema/schema_generated.h"

namespace litert::samsung {

Expected<std::string> GetFusedActivationName(uint32_t tfl_fused_activation) {
  if (tfl_fused_activation > tflite::ActivationFunctionType_MAX) {
    return Error(kLiteRtStatusErrorIndexOOB, "Invalid activation");
  }

  if (!tflite::EnumNamesActivationFunctionType()[tfl_fused_activation]) {
    return Error(kLiteRtStatusErrorUnsupported, "Unsupported fused activation");
  }

  std::string activation_type =
      tflite::EnumNamesActivationFunctionType()[tfl_fused_activation];
  return activation_type;
}

absl::InlinedVector<int32_t, kExpectedMaxTensorRank> GetDimensions(
    const Tensor &t) {
  auto tensor_type = t.RankedTensorType();
  auto dimensions = tensor_type->Layout().Dimensions();
  // TODO: Remove this comment
  //  Have no idea why get incorrect dimensions when use Span.
  return absl::InlinedVector<int32_t, kExpectedMaxTensorRank>(
      dimensions.begin(), dimensions.end());
}

Expected<const char *> MapToElementTypeStr(ElementType element_type) {
  switch (element_type) {
    case ElementType::Bool:
      return "BOOL";
    case ElementType::Int4:
      return "INT4";
    case ElementType::Int8:
      return "INT8";
    case ElementType::UInt8:
      return "UINT8";
    case ElementType::Int16:
      return "INT16";
    case ElementType::UInt16:
      return "UINT16";
    case ElementType::Int32:
      return "INT32";
    case ElementType::Int64:
      return "INT64";
    case ElementType::Float16:
      return "FLOAT16";
    case ElementType::Float32:
      return "FLOAT32";
    default:
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Element Type not supported");
  }
}

std::pair<int32_t, int32_t> GetExplicitPadding(int32_t input_size,
                                               int32_t filter_size,
                                               int32_t output_size,
                                               int32_t stride,
                                               int32_t dilation) {
  int32_t padding_t_or_l = 0;
  int32_t padding_b_or_r = 0;
  int32_t dilated_filter_size = (filter_size - 1) * dilation + 1;
  int32_t total_pad =
      ((output_size - 1) * stride + dilated_filter_size - input_size);

  if (total_pad < 0) {
    total_pad = 0;
  }

  padding_t_or_l = total_pad / 2;
  padding_b_or_r = total_pad - padding_t_or_l;

  return {padding_t_or_l, padding_b_or_r};
}

}  // namespace litert::samsung
