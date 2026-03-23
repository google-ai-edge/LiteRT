// Copyright 2024 Google LLC.
// Copyright (C) Samsung Electronics Co. LTD. All rights reserved
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

absl::InlinedVector<int32_t, kExpectedMaxTensorRank>
GetDimensions(const Tensor &t) {
  auto tensor_type = t.RankedTensorType();
  auto dimensions = tensor_type->Layout().Dimensions();
  // TODO: Remove this comment
  //  Have no idea why get incorrect dimensions when use Span.
  return absl::InlinedVector<int32_t, kExpectedMaxTensorRank>(
      dimensions.begin(), dimensions.end());
}

} // namespace litert::samsung
