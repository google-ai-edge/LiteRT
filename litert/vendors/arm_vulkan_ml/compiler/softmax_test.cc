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

#include "litert/vendors/arm_vulkan_ml/compiler/softmax.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "TosaForSpirvCodegen.hpp"
#include "spirv/Module.hpp"
#include "spirv-tools/libspirv.hpp"
#include "tosa/OperatorEnum.hpp"

namespace litert::arm_vulkan_ml {
namespace {
TEST(QuantizedSoftmax, ProvesSignedSumAndRoundedNumeratorForAllRows) {
  for (uint32_t channels : {1, 2, 3, 255, 256, 511, 512, 1000, 1001, 1024, 2048, 4095}) {
    std::string reason;
    auto plan = AnalyzeQuantizedSoftmax(channels, .125, 1, kLrtArmVulkanMLSoftmaxLutAuto, reason);
    ASSERT_TRUE(plan) << reason;
    EXPECT_EQ(plan->lut_type, kLrtArmVulkanMLSoftmaxLutInt8);
    const int shift = plan->normalization_shift;
    int64_t maximum = (int64_t{INT32_MAX} + (int64_t{1} << (shift - 1))) >> shift;
    EXPECT_LE(maximum * channels, INT32_MAX);
    EXPECT_LE(maximum * 256 + maximum * channels / 2, INT32_MAX);
    EXPECT_GT(maximum, 0);
  }
}
TEST(QuantizedSoftmax, RejectsUnsupportedDimensionsAndNonfiniteParameters) {
  std::string reason;
  for (uint32_t channels : {0, 4096, 8192})
    EXPECT_FALSE(AnalyzeQuantizedSoftmax(channels, .125, 1, kLrtArmVulkanMLSoftmaxLutAuto, reason));
  for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(AnalyzeQuantizedSoftmax(5, value, 1, kLrtArmVulkanMLSoftmaxLutAuto, reason));
    EXPECT_FALSE(AnalyzeQuantizedSoftmax(5, .125, value, kLrtArmVulkanMLSoftmaxLutAuto, reason));
  }
  EXPECT_FALSE(AnalyzeQuantizedSoftmax(5, .125, 1, static_cast<SoftmaxLutType>(99), reason));
}
TEST(QuantizedSoftmax, BothLayoutsRetainTheSameNonnegativeQ31Exponential) {
  for (double scale : {1e-12, 1.0 / 256, .125, 1.0, 100.0}) for (double beta : {.1, 1.0, 2.0}) {
    std::string reason;
    auto a = AnalyzeQuantizedSoftmax(1000, scale, beta, kLrtArmVulkanMLSoftmaxLutInt8, reason);
    auto b = AnalyzeQuantizedSoftmax(1000, scale, beta, kLrtArmVulkanMLSoftmaxLutInt16, reason);
    ASSERT_TRUE(a); ASSERT_TRUE(b);
    EXPECT_EQ(a->exponential, b->exponential);
    EXPECT_EQ(a->exponential.back(), INT32_MAX);
    EXPECT_TRUE(std::is_sorted(a->exponential.begin(), a->exponential.end()));
    for (int32_t value : a->exponential) {
      EXPECT_GE(value, 0);
      // The INT16 TABLE interpolation fraction is zero at these addresses;
      // decoding its three exact chunks must equal the four-byte decoding.
      int64_t upper = ((value >> 21) & 0x7ff) * int64_t{128};
      int64_t middle = ((value >> 10) & 0x7ff) * int64_t{128};
      int64_t lower = (value & 0x3ff) * int64_t{128};
      EXPECT_EQ((upper << 14) + (middle << 3) + (lower >> 7), value);
    }
  }
}
TEST(QuantizedSoftmax, EmitsIntegerNormalizationAndSelectableTableCounts) {
  using O = tfsc::tosa::OperatorEnum;
  for (auto type : {kLrtArmVulkanMLSoftmaxLutInt8, kLrtArmVulkanMLSoftmaxLutInt16}) {
    std::string reason;
    auto plan = AnalyzeQuantizedSoftmax(5, .125, .7, type, reason); ASSERT_TRUE(plan);
    auto module = tfsc::CreateModule(); tfsc::tosa::Graph graph(module, "main");
    auto input = graph.AddInput(tfsc::tosa::Tensor(tfsc::tosa::DataType::int8_t, {2, 5}), 0);
    auto output = EmitQuantizedSoftmax(graph, input, {2, 5}, *plan);
    ASSERT_TRUE(output); ASSERT_FALSE(tfsc::IsError(output)) << tfsc::GetErrorMsg(output);
    graph.AddOutput(output, 1); graph.FinalizeGraph();
    const auto binary = tfsc::WriteToBinary(module);
    ASSERT_FALSE(binary.empty());
    spvtools::SpirvTools validator(SPV_ENV_UNIVERSAL_1_6);
    std::string diagnostics;
    validator.SetMessageConsumer([&](spv_message_level_t, const char*,
        const spv_position_t&, const char* message) { diagnostics += message; });
    EXPECT_TRUE(validator.Validate(binary)) << diagnostics;
    int tables = 0, divides = 0;
    for (const auto& instruction : module->GetSpirvGraph()) if (instruction.m_Opcode == spv::Op::OpExtInst) {
      const auto opcode = static_cast<O>(instruction.m_Operands[3].m_LiteralWord);
      tables += opcode == O::Table; divides += opcode == O::IntDiv;
      EXPECT_NE(opcode, O::Exp); EXPECT_NE(opcode, O::Reciprocal);
    }
    EXPECT_EQ(tables, type == kLrtArmVulkanMLSoftmaxLutInt8 ? 4 : 3);
    EXPECT_EQ(divides, 1);
  }
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
