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

enum DATA_TYPE {
    UNDEFINED = 0,
    FLOAT32,
    UINT8,
    INT8,
    UINT16,
    INT16,
    INT32,
    INT64,
    STRING,
    BOOL,
    FLOAT16,
    DOUBLE,
    UINT32,
    UINT64,
    COMPLEX64,
    COMPLEX128,
    BFLOAT16,
    FLOAT8E4M3FN,
    FLOAT8E4M3FNUZ,
    FLOAT8E5M2,
    FLOAT8E5M2FNUZ,
    UINT4,
    INT4,
    FLOAT4E2M1,
    FLOAT8E8M0,
    UINT2,
    INT2
};

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

Expected<const uint32_t> ConvertElementTypeToInt(ElementType element_type) {
  switch (element_type) {
  case ElementType::Bool:
    return DATA_TYPE::BOOL;
  case ElementType::Int4:
    return DATA_TYPE::INT4;
  case ElementType::Int8:
    return DATA_TYPE::INT8;
  case ElementType::UInt8:
    return DATA_TYPE::UINT8;
  case ElementType::Int16:
    return DATA_TYPE::UINT16;
  case ElementType::UInt16:
    return DATA_TYPE::UINT16;
  case ElementType::Int32:
    return DATA_TYPE::INT32;
  case ElementType::Int64:
    return DATA_TYPE::INT64;
  case ElementType::Float16:
    return DATA_TYPE::FLOAT16;
  case ElementType::Float32:
    return DATA_TYPE::FLOAT32;
  default:
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Element Type not supported");
  }
}

} // namespace litert::samsung
