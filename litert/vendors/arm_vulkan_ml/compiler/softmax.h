// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_SOFTMAX_H_
#define ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_SOFTMAX_H_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "litert/c/options/litert_arm_vulkan_ml_options.h"
#include "tosa/Graph.hpp"

namespace litert::arm_vulkan_ml {
using SoftmaxLutType = LrtArmVulkanMLSoftmaxLutType;
const char* SoftmaxLutTypeName(SoftmaxLutType type);
std::optional<SoftmaxLutType> ParseSoftmaxLutType(const std::string& name);

struct QuantizedSoftmaxPlan {
  SoftmaxLutType lut_type = kLrtArmVulkanMLSoftmaxLutInt8;
  int8_t normalization_shift = 9;
  uint32_t channels = 0;
  double beta = 1;
  // d=-255..0, addressed by d+255. Both layouts reconstruct these same
  // nonnegative Q31 values; there is no gemmlowp approximation/dependency.
  std::array<int32_t, 256> exponential{};
};
std::optional<QuantizedSoftmaxPlan> AnalyzeQuantizedSoftmax(
    uint32_t channels, double input_scale, double beta, SoftmaxLutType type,
    std::string& reason);
tfsc::tosa::ResId EmitQuantizedSoftmax(tfsc::tosa::Graph& graph,
    tfsc::tosa::ResId input, const std::vector<uint32_t>& shape,
    const QuantizedSoftmaxPlan& plan);
}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_SOFTMAX_H_
