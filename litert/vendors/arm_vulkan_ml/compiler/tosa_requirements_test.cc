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

#include "litert/vendors/arm_vulkan_ml/compiler/tosa_requirements.h"

#include <vector>

#include "gtest/gtest.h"
#include "TosaForSpirvCodegen.hpp"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/portable_ops.h"
#include "spirv/Module.hpp"
#include "tosa/Graph.hpp"

namespace litert::arm_vulkan_ml {
namespace {
using tfsc::tosa::Attribute;
using tfsc::tosa::DataType;
using tfsc::tosa::Tensor;
using O = tfsc::tosa::OperatorEnum;

OperationPlan FloatConvolution(bool depthwise) {
  OperationPlan p{};
  p.code = depthwise ? kLiteRtOpCodeTflDepthwiseConv2d : kLiteRtOpCodeTflConv2d;
  p.inputs = {{{1, 3, 3, 2}, 1.0, 0, true}};
  p.output = {{1, 3, 3, 2}, 1.0, 0, true};
  p.weight_shape = depthwise ? std::vector<uint32_t>{1, 1, 1, 2}
                            : std::vector<uint32_t>{2, 1, 1, 2};
  p.float_weights.assign(depthwise ? 2 : 4, 1.0f);
  p.float_bias.assign(2, 0.0f);
  p.stride = p.dilation = {1, 1};
  p.padding = {0, 0, 0, 0};
  p.cropped_shape = p.inputs[0].shape;
  return p;
}

TEST(TosaRequirementsTest, FloatingZeroPointsHaveTheSpecificationElementType) {
  // Check emitted operands independently of SPIRV-Tools and the emulator,
  // which accepted the old mixed FP32/INT8 tuples.
  for (int variant = 0; variant < 5; ++variant) {
    auto module = tfsc::CreateModule();
    tfsc::tosa::Graph graph(module, "main");
    auto p = FloatConvolution(variant == 1);
    std::vector<tfsc::tosa::ResId> inputs;
    tfsc::tosa::ResId result;
    GraphConstantPool constants;
    O expected;
    size_t first_zero;
    if (variant < 2) {
      inputs.push_back(graph.AddInput(Tensor(DataType::float32_t, p.inputs[0].shape), 0));
      result = EmitOperation(graph, inputs, p, constants);
      expected = variant == 0 ? O::Conv2d : O::DepthwiseConv2d;
      first_zero = 12;
    } else {
      PortablePlan portable{};
      portable.shapes = {{1, 4, 4, 1}};
      portable.outputs = {{1, 2, 2, 1}};
      portable.code = kLiteRtOpCodeTflAveragePool2d;
      portable.kernel = portable.strides = {2, 2};
      portable.padding = {0, 0, 0, 0};
      expected = O::AvgPool2d;
      first_zero = 9;
      if (variant == 4) {
        portable.code = kLiteRtOpCodeTflBatchMatmul;
        portable.shapes = {{1, 2, 3}, {1, 3, 2}};
        portable.outputs = {{1, 2, 2}};
        expected = O::Matmul;
        first_zero = 6;
      }
      for (size_t i = 0; i < portable.shapes.size(); ++i)
        inputs.push_back(graph.AddInput(Tensor(DataType::float32_t, portable.shapes[i]), static_cast<uint32_t>(i)));
      if (variant == 2) {
        p.code = kLiteRtOpCodeTflAveragePool2d;
        p.inputs[0].shape = portable.shapes[0];
        p.output.shape = portable.outputs[0];
        p.kernel = portable.kernel;
        p.stride = portable.strides;
        result = EmitOperation(graph, inputs, p, constants);
      } else {
        auto outputs = EmitPortableOperation(graph, inputs, portable);
        ASSERT_EQ(outputs.size(), 1);
        result = outputs[0];
      }
    }
    ASSERT_NE(result, nullptr);
    ASSERT_FALSE(tfsc::IsError(result));
    graph.AddOutput(result, static_cast<uint32_t>(inputs.size()));
    graph.FinalizeGraph();
    size_t checked = 0;
    for (const auto& instruction : module->GetSpirvGraph()) {
      const auto& a = instruction.m_Operands;
      if (instruction.m_Opcode != spv::Op::OpExtInst ||
          a[3].m_LiteralWord != static_cast<uint32_t>(expected)) continue;
      ASSERT_GT(a.size(), first_zero + 1);
      for (size_t i : {first_zero, first_zero + 1}) {
        const auto* zero = a[i].m_InstructionPtr;
        ASSERT_EQ(zero->m_Opcode, spv::Op::OpConstantNull);
        const auto* tensor = zero->m_Operands[0].m_InstructionPtr;
        ASSERT_EQ(tensor->m_Opcode, spv::Op::OpTypeTensorARM);
        const auto* scalar = tensor->m_Operands[1].m_InstructionPtr;
        EXPECT_EQ(scalar->m_Opcode, spv::Op::OpTypeFloat);
        EXPECT_EQ(scalar->m_Operands[1].m_LiteralWord, 32);
      }
      ++checked;
    }
    EXPECT_EQ(checked, 1);
    TosaLevelRequirements requirements;
    std::string reason;
    EXPECT_TRUE(CollectTosaLevelRequirements(*module, requirements, reason)) << reason;
  }
}

TEST(TosaRequirementsTest, CollectsStrideAndEffectiveDilatedKernel) {
  for (bool dilation : {false, true}) {
    auto p = FloatConvolution(false);
    p.inputs[0].shape = p.output.shape = p.cropped_shape = {1, 1, 1, 1};
    p.weight_shape = dilation ? std::vector<uint32_t>{1, 2, 2, 1}
                              : std::vector<uint32_t>{1, 1, 1, 1};
    p.float_weights.assign(dilation ? 4 : 1, 1.0f);
    p.float_bias = {0.0f};
    p.stride = dilation ? std::vector<int32_t>{1, 1} : std::vector<int32_t>{16384, 16384};
    p.dilation = dilation ? std::vector<int32_t>{8192, 8192} : std::vector<int32_t>{1, 1};
    p.padding.assign(4, dilation ? 4096 : 0);
    auto module = tfsc::CreateModule();
    tfsc::tosa::Graph graph(module, "main");
    auto input = graph.AddInput(Tensor(DataType::float32_t, p.inputs[0].shape), 0);
    GraphConstantPool constants;
    auto output = EmitOperation(graph, {input}, p, constants);
    ASSERT_FALSE(tfsc::IsError(output));
    graph.AddOutput(output, 1);
    graph.FinalizeGraph();
    TosaLevelRequirements r;
    std::string reason;
    ASSERT_TRUE(CollectTosaLevelRequirements(*module, r, reason)) << reason;
    EXPECT_TRUE(r.checked);
    EXPECT_EQ(dilation ? r.max_kernel : r.max_stride, 16384);
  }
}

TEST(TosaRequirementsTest, IncludesGeneratedListsAndIntermediateTensorByteSizes) {
  auto module = tfsc::CreateModule();
  tfsc::tosa::Graph graph(module, "main");
  auto input = graph.AddInput(Tensor(DataType::float32_t, {16384, 16384, 2}), 0);
  std::vector<tfsc::tosa::ResId> pieces(65, input);
  auto output = graph.AddConcatOperator(pieces, Attribute(int32_t{0}),
      Tensor(DataType::float32_t, {65 * 16384, 16384, 2}));
  graph.AddOutput(output, 1);
  graph.FinalizeGraph();
  TosaLevelRequirements r;
  std::string reason;
  ASSERT_TRUE(CollectTosaLevelRequirements(*module, r, reason)) << reason;
  EXPECT_EQ(r.max_tensor_list_size, 65);
  EXPECT_EQ(r.max_tensor_bytes, uint64_t{65} * (uint64_t{1} << 31));
  EXPECT_EQ(r.max_rank, 3);
  EXPECT_EQ(r.max_nesting, 1);
}

TEST(TosaRequirementsTest, ComputesResizeScaleRatioRatherThanRawNumerator) {
  auto module = tfsc::CreateModule();
  tfsc::tosa::Graph graph(module, "main");
  auto input = graph.AddInput(Tensor(DataType::float32_t, {1, 2, 2, 1}), 0);
  auto output = graph.AddResizeOperator(input,
      graph.AddTensorConstant(Attribute(std::vector<int32_t>{514, 2, 514, 2})),
      graph.AddTensorConstant(Attribute(std::vector<int32_t>{0, 0})),
      graph.AddTensorConstant(Attribute(std::vector<int32_t>{512, 512})),
      Attribute(int32_t{2}), Tensor(DataType::float32_t, {1, 514, 514, 1}));
  graph.AddOutput(output, 1);
  graph.FinalizeGraph();
  TosaLevelRequirements r;
  std::string reason;
  ASSERT_TRUE(CollectTosaLevelRequirements(*module, r, reason)) << reason;
  EXPECT_EQ(r.max_scale, 257);
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
