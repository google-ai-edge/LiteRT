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

#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "tosa/Graph.hpp"

namespace litert::arm_vulkan_ml {
enum class SoftmaxLutType {
  kAuto = 0,
  kInt8 = 1,
  kInt16 = 2,
};
const char* SoftmaxLutTypeName(SoftmaxLutType type);
std::optional<SoftmaxLutType> ParseSoftmaxLutType(const std::string& name);

struct QuantizedSoftmaxPlan {
  SoftmaxLutType lut_type = SoftmaxLutType::kInt8;
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
                                       tfsc::tosa::ResId input,
                                       const std::vector<uint32_t>& shape,
                                       const QuantizedSoftmaxPlan& plan);

struct Int16SoftmaxPlan {
  uint32_t channels = 0;
  double beta = 1.0;
  Scale32 diff_scale{0, 0};
  Scale32 output_scale{0, 0};
  int32_t diff_min = -65535;
  bool clamp_diff = false;
  std::array<int16_t, 513> exp_lut{};
  std::array<int16_t, 513> one_over_one_plus_x_lut{};
};
std::optional<Int16SoftmaxPlan> AnalyzeInt16Softmax(
    uint32_t channels, double input_scale, double output_scale, double beta,
    std::string& reason);
tfsc::tosa::ResId EmitInt16Softmax(tfsc::tosa::Graph& graph,
                                   tfsc::tosa::ResId input,
                                   const std::vector<uint32_t>& shape,
                                   const Int16SoftmaxPlan& plan,
                                   GraphConstantPool& constants);
}  // namespace litert::arm_vulkan_ml
#endif  // ODML_LITERT_LITERT_VENDORS_ARM_VULKAN_ML_COMPILER_SOFTMAX_H_
