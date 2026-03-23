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
#ifndef ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_BUILDERS_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_BUILDERS_UTILS_H_

#include <cstdint>
#include <string>

#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"

namespace litert::samsung {

Expected<std::string> GetFusedActivationName(uint32_t tfl_fused_activation);

absl::InlinedVector<int32_t, kExpectedMaxTensorRank> GetDimensions(
    const Tensor &t);

template <typename T>
Expected<std::vector<T>> GetWeightDataAs(const Tensor &t) {
  LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type, t.RankedTensorType());

  std::vector<T> data;
  switch (ranked_tensor_type.ElementType()) {
#define GET_AND_FILL(element_type, arith_type)                        \
  case element_type: {                                                \
    LITERT_ASSIGN_OR_RETURN(auto value, t.WeightsData<arith_type>()); \
                                                                      \
    data.resize(value.size());                                        \
    std::transform(value.begin(), value.end(), data.begin(),          \
                   [](auto val) { return static_cast<T>(val); });     \
    return data;                                                      \
  }
    GET_AND_FILL(ElementType::Bool, bool);
    GET_AND_FILL(ElementType::Int8, int8_t);
    GET_AND_FILL(ElementType::UInt8, uint8_t);
    GET_AND_FILL(ElementType::Int16, int16_t);
    GET_AND_FILL(ElementType::UInt16, uint16_t);
    GET_AND_FILL(ElementType::Int32, int32_t);
    GET_AND_FILL(ElementType::UInt32, uint32_t);
    GET_AND_FILL(ElementType::Int64, int64_t);
    GET_AND_FILL(ElementType::Float32, float);
    GET_AND_FILL(ElementType::Float64, double);

#undef GET_AND_FILL
    default:
      return Error(kLiteRtStatusErrorUnsupported, "Unsupported element type.");
  }
}

Expected<const uint32_t> ConvertElementTypeToInt(ElementType element_type);

std::pair<int32_t, int32_t> GetExplicitPadding(int32_t input_size,
                                               int32_t filter_size,
                                               int32_t output_size,
                                               int32_t stride, int32_t dilation,
                                               bool is_transposed = false);

} // namespace litert::samsung

#endif  // ODML_LITERT_LITERT_VENDORS_SAMSUNG_COMPILER_BUILDERS_MUL_OP_BUILDER_H_
