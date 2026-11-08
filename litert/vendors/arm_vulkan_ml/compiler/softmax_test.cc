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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "TosaForSpirvCodegen.hpp"
#include <gtest/gtest.h>
#include "spirv/unified1/spirv.hpp"  // from @arm_vulkan_ml_dep_spirv_headers
#include "spirv-tools/libspirv.h"  // from @arm_vulkan_ml_tool_spirv_tools
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "tosa-for-spirv-codegen/tosa/Tensor.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/Graph.hpp"  // from @tosa_for_spirv_codegen
#include "spirv-tools/libspirv.hpp"  // from @arm_vulkan_ml_tool_spirv_tools
#include "tosa/OperatorEnum.hpp"

namespace litert::arm_vulkan_ml {
namespace {
TEST(QuantizedSoftmax, ProvesSignedSumAndRoundedNumeratorForAllRows) {
  for (uint32_t channels :
       {1, 2, 3, 255, 256, 511, 512, 1000, 1001, 1024, 2048, 4095}) {
    std::string reason;
    auto plan =
        AnalyzeQuantizedSoftmax(channels, .125, 1, SoftmaxLutType::kAuto, reason);
    ASSERT_TRUE(plan) << reason;
    EXPECT_EQ(plan->lut_type, SoftmaxLutType::kInt8);
    const int shift = plan->normalization_shift;
    int64_t maximum =
        (int64_t{INT32_MAX} + (int64_t{1} << (shift - 1))) >> shift;
    EXPECT_LE(maximum * channels, INT32_MAX);
    EXPECT_LE(maximum * 256 + maximum * channels / 2, INT32_MAX);
    EXPECT_GT(maximum, 0);
  }
}
TEST(QuantizedSoftmax, RejectsUnsupportedDimensionsAndNonfiniteParameters) {
  std::string reason;
  for (uint32_t channels : {0, 4096, 8192})
    EXPECT_FALSE(AnalyzeQuantizedSoftmax(channels, .125, 1,
                                         SoftmaxLutType::kAuto, reason));
  for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(
        AnalyzeQuantizedSoftmax(5, value, 1, SoftmaxLutType::kAuto, reason));
    EXPECT_FALSE(
        AnalyzeQuantizedSoftmax(5, .125, value, SoftmaxLutType::kAuto, reason));
  }
  EXPECT_FALSE(AnalyzeQuantizedSoftmax(5, .125, 1,
                                       static_cast<SoftmaxLutType>(3), reason));
}
TEST(QuantizedSoftmax, BothLayoutsRetainTheSameNonnegativeQ31Exponential) {
  for (double scale : {1e-12, 1.0 / 256, .125, 1.0, 100.0})
    for (double beta : {.1, 1.0, 2.0}) {
      std::string reason;
      auto a = AnalyzeQuantizedSoftmax(1000, scale, beta, SoftmaxLutType::kInt8,
                                       reason);
      auto b = AnalyzeQuantizedSoftmax(1000, scale, beta,
                                       SoftmaxLutType::kInt16, reason);
      ASSERT_TRUE(a);
      ASSERT_TRUE(b);
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
  for (auto type : {SoftmaxLutType::kInt8, SoftmaxLutType::kInt16}) {
    std::string reason;
    auto plan = AnalyzeQuantizedSoftmax(5, .125, .7, type, reason);
    ASSERT_TRUE(plan);
    auto module = tfsc::CreateModule();
    tfsc::tosa::Graph graph(module, "main");
    auto input = graph.AddInput(
        tfsc::tosa::Tensor(tfsc::tosa::DataType::int8_t, {2, 5}), 0);
    auto output = EmitQuantizedSoftmax(graph, input, {2, 5}, *plan);
    ASSERT_TRUE(output);
    ASSERT_FALSE(tfsc::IsError(output)) << tfsc::GetErrorMsg(output);
    graph.AddOutput(output, 1);
    graph.FinalizeGraph();
    const auto binary = tfsc::WriteToBinary(module);
    ASSERT_FALSE(binary.empty());
    spvtools::SpirvTools validator(SPV_ENV_UNIVERSAL_1_6);
    std::string diagnostics;
    validator.SetMessageConsumer(
        [&](spv_message_level_t, const char*, const spv_position_t&,
            const char* message) { diagnostics += message; });
    EXPECT_TRUE(validator.Validate(binary)) << diagnostics;
    int tables = 0, divides = 0;
    for (const auto& instruction : module->GetSpirvGraph())
      if (instruction.m_Opcode == spv::Op::OpExtInst) {
        const auto opcode =
            static_cast<O>(instruction.m_Operands[3].m_LiteralWord);
        tables += opcode == O::Table;
        divides += opcode == O::IntDiv;
        EXPECT_NE(opcode, O::Exp);
        EXPECT_NE(opcode, O::Reciprocal);
      }
    EXPECT_EQ(tables, type == SoftmaxLutType::kInt8 ? 4 : 3);
    EXPECT_EQ(divides, 1);
  }
}

TEST(Int16Softmax, EmitsValidClzTwoTableSpirvWithoutInt32ClampOrIntDiv) {
  using O = tfsc::tosa::OperatorEnum;
  for (double input_scale : {0.0001, 0.025, 1.0}) {
    for (int bits : {16, 32}) {
      std::string reason;
      auto plan = AnalyzeInt16Softmax(197, input_scale, 1.0 / 32768.0, 1.0,
                                      reason);
      ASSERT_TRUE(plan) << reason;
      EXPECT_EQ(plan->clamp_diff, input_scale * (65535.0 / 10.0) > 1.0);
      auto module = tfsc::CreateModule();
      tfsc::tosa::Graph graph(module, "main");
      auto input = graph.AddInput(
          tfsc::tosa::Tensor(tfsc::tosa::DataType::int16_t, {1, 12, 197, 197}),
          0);
      GraphConstantPool constants;
      constants.rescale_multiplier_bits = bits;
      auto output =
          EmitInt16Softmax(graph, input, {1, 12, 197, 197}, *plan, constants);
      ASSERT_TRUE(output);
      ASSERT_FALSE(tfsc::IsError(output)) << tfsc::GetErrorMsg(output);
      graph.AddOutput(output, 1);
      graph.FinalizeGraph();
      const auto binary = tfsc::WriteToBinary(module);
      ASSERT_FALSE(binary.empty());
      spvtools::SpirvTools validator(SPV_ENV_UNIVERSAL_1_6);
      std::string diagnostics;
      validator.SetMessageConsumer(
          [&](spv_message_level_t, const char*, const spv_position_t&,
              const char* message) { diagnostics += message; });
      EXPECT_TRUE(validator.Validate(binary)) << diagnostics;

      int tables = 0, clz = 0, divides = 0, clamps = 0, maximums = 0;
      for (const auto& instruction : module->GetSpirvGraph()) {
        if (instruction.m_Opcode == spv::Op::OpExtInst) {
          const auto opcode =
              static_cast<O>(instruction.m_Operands[3].m_LiteralWord);
          tables += opcode == O::Table;
          clz += opcode == O::Clz;
          divides += opcode == O::IntDiv;
          clamps += opcode == O::Clamp;
          maximums += opcode == O::Maximum;
          EXPECT_NE(opcode, O::Exp);
          EXPECT_NE(opcode, O::Reciprocal);
        }
      }
      EXPECT_EQ(tables, 2);
      EXPECT_EQ(clz, 1);
      EXPECT_EQ(divides, 0);
      EXPECT_EQ(clamps, 0);
      EXPECT_EQ(maximums, plan->clamp_diff ? 1 : 0);
      size_t lut_constants = 0;
      for (const auto& c : constants.values) {
        if (c.bytes.size() == 513 * sizeof(int16_t)) ++lut_constants;
      }
      EXPECT_EQ(lut_constants, 2);
    }
  }
}

TEST(Int16Softmax, RejectsInvalidChannelCountAndOutputScale) {
  std::string reason;
  EXPECT_TRUE(AnalyzeInt16Softmax(8192, 0.05, 1.0 / 32768.0, 1.0, reason));
  for (uint32_t channels : {0u, 8193u, 16384u}) {
    EXPECT_FALSE(
        AnalyzeInt16Softmax(channels, 0.05, 1.0 / 32768.0, 1.0, reason));
  }
  EXPECT_FALSE(AnalyzeInt16Softmax(197, 0.05, 1.0 / 256.0, 1.0, reason));
  for (double invalid : {0.0, -0.5, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(
        AnalyzeInt16Softmax(197, invalid, 1.0 / 32768.0, 1.0, reason));
    EXPECT_FALSE(
        AnalyzeInt16Softmax(197, 0.05, 1.0 / 32768.0, invalid, reason));
  }
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
