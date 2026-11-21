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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "TosaForSpirvCodegen.hpp"
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tosa-for-spirv-codegen/spirv/Instruction.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/spirv/Module.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/spirv/Operand.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/OperatorEnum.hpp"  // from @tosa_for_spirv_codegen
#include "tosa-for-spirv-codegen/tosa/Tensor.hpp"  // from @tosa_for_spirv_codegen
#include "litert/c/litert_op_code.h"
#include "litert/vendors/arm_vulkan_ml/common/arm_vulkan_ml_jit_executable.h"
#include "litert/vendors/arm_vulkan_ml/compiler/conv2d.h"
#include "litert/vendors/arm_vulkan_ml/compiler/operation.h"
#include "litert/vendors/arm_vulkan_ml/compiler/portable_ops.h"
#include "spirv/unified1/spirv.hpp"  // from @arm_vulkan_ml_dep_spirv_headers
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

struct MinimalModule {
  std::shared_ptr<tfsc::spirv::Module> module;
  const tfsc::spirv::Instruction* tosa_import = nullptr;
  const tfsc::spirv::Instruction* tensor_type = nullptr;
  tfsc::tosa::ResId input = nullptr;
};

// Builds a minimal finalized single-input INT32 graph for SPIR-V validation
// tests.
MinimalModule CreateMinimalValidModule() {
  MinimalModule result;
  result.module = tfsc::CreateModule();
  tfsc::tosa::Graph graph(result.module, "main");
  result.input = graph.AddInput(Tensor(DataType::int32_t, {2, 3}), 0);
  graph.AddOutput(result.input, 1);
  graph.FinalizeGraph();
  tfsc::spirv::Module::InstructionRange imports =
      result.module->GetInstructionsOfType(spv::Op::OpExtInstImport);
  if (imports.first != imports.second) result.tosa_import = &*imports.first;
  tfsc::spirv::Module::InstructionRange tensors =
      result.module->GetInstructionsOfType(spv::Op::OpTypeTensorARM);
  if (tensors.first != tensors.second) result.tensor_type = &*tensors.first;
  return result;
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
      inputs.push_back(
          graph.AddInput(Tensor(DataType::float32_t, p.inputs[0].shape), 0));
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
        inputs.push_back(
            graph.AddInput(Tensor(DataType::float32_t, portable.shapes[i]),
                           static_cast<uint32_t>(i)));
      if (variant == 2) {
        p.code = kLiteRtOpCodeTflAveragePool2d;
        p.inputs[0].shape = portable.shapes[0];
        p.output.shape = portable.outputs[0];
        p.kernel = portable.kernel;
        p.stride = portable.strides;
        result = EmitOperation(graph, inputs, p, constants);
      } else {
        GraphConstantPool constants;
        auto outputs =
            EmitPortableOperation(graph, inputs, portable, constants);
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
          a[3].m_LiteralWord != static_cast<uint32_t>(expected))
        continue;
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
    EXPECT_TRUE(CollectTosaLevelRequirements(*module, requirements, reason))
        << reason;
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
    p.stride = dilation ? std::vector<int32_t>{1, 1}
                        : std::vector<int32_t>{16384, 16384};
    p.dilation = dilation ? std::vector<int32_t>{8192, 8192}
                          : std::vector<int32_t>{1, 1};
    p.padding.assign(4, dilation ? 4096 : 0);
    auto module = tfsc::CreateModule();
    tfsc::tosa::Graph graph(module, "main");
    auto input =
        graph.AddInput(Tensor(DataType::float32_t, p.inputs[0].shape), 0);
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

TEST(TosaRequirementsTest,
     IncludesGeneratedListsAndIntermediateTensorByteSizes) {
  auto module = tfsc::CreateModule();
  tfsc::tosa::Graph graph(module, "main");
  auto input =
      graph.AddInput(Tensor(DataType::float32_t, {16384, 16384, 2}), 0);
  std::vector<tfsc::tosa::ResId> pieces(65, input);
  auto output = graph.AddConcatOperator(
      pieces, Attribute(int32_t{0}),
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
  auto output = graph.AddResizeOperator(
      input,
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

TEST(TosaRequirementsTest, RejectsSameTypeCastOperators) {
  static constexpr DataType kTypes[] = {
      DataType::int32_t,   DataType::int8_t,     DataType::float32_t,
      DataType::float16_t, DataType::bfloat16_t, DataType::bool_t,
  };
  for (DataType dtype : kTypes) {
    SCOPED_TRACE(tfsc::tosa::GetDataTypeName(dtype));
    std::shared_ptr<tfsc::spirv::Module> module = tfsc::CreateModule();
    tfsc::tosa::Graph graph(module, "main");
    tfsc::tosa::ResId input = graph.AddInput(Tensor(dtype, {2, 3}), 0);
    tfsc::tosa::ResId output =
        graph.AddCastOperator(input, Tensor(dtype, {2, 3}));
    if (output == nullptr || tfsc::IsError(output)) {
      ADD_FAILURE() << "Failed to emit CAST operator";
      continue;
    }
    graph.AddOutput(output, 1);
    graph.FinalizeGraph();
    TosaLevelRequirements requirements;
    std::string reason;
    EXPECT_FALSE(CollectTosaLevelRequirements(*module, requirements, reason));
    EXPECT_EQ(reason,
              "TOSA CAST requires distinct input and output element types");
  }
}

TEST(TosaRequirementsTest, AcceptsCrossTypeCastOperators) {
  static constexpr std::pair<DataType, DataType> kPairs[] = {
      {DataType::int32_t, DataType::int8_t},
      {DataType::int8_t, DataType::int32_t},
      {DataType::float32_t, DataType::int32_t},
      {DataType::int8_t, DataType::bool_t},
      {DataType::float32_t, DataType::float16_t},
      {DataType::float16_t, DataType::bfloat16_t},
  };
  for (const auto& [in_type, out_type] : kPairs) {
    SCOPED_TRACE(absl::StrCat(tfsc::tosa::GetDataTypeName(in_type), "->",
                              tfsc::tosa::GetDataTypeName(out_type)));
    std::shared_ptr<tfsc::spirv::Module> module = tfsc::CreateModule();
    tfsc::tosa::Graph graph(module, "main");
    tfsc::tosa::ResId input = graph.AddInput(Tensor(in_type, {2, 3}), 0);
    tfsc::tosa::ResId output =
        graph.AddCastOperator(input, Tensor(out_type, {2, 3}));
    if (output == nullptr || tfsc::IsError(output)) {
      ADD_FAILURE() << "Failed to emit CAST operator";
      continue;
    }
    graph.AddOutput(output, 1);
    graph.FinalizeGraph();
    TosaLevelRequirements requirements;
    std::string reason;
    EXPECT_TRUE(CollectTosaLevelRequirements(*module, requirements, reason))
        << reason;
    EXPECT_TRUE(requirements.checked);
  }
}

TEST(TosaRequirementsTest, RejectsMalformedCastOperator) {
  // Missing input operand (4 operands instead of 5).
  {
    MinimalModule base = CreateMinimalValidModule();
    ASSERT_NE(base.tosa_import, nullptr);
    ASSERT_NE(base.tensor_type, nullptr);
    base.module->EmplaceInstructionNonUnique(
        spv::Op::OpExtInst,
        std::vector<tfsc::spirv::Operand>{
            tfsc::spirv::Operand(base.tensor_type),
            tfsc::spirv::RESID,
            tfsc::spirv::Operand(base.tosa_import),
            tfsc::spirv::Operand(
                static_cast<uint32_t>(tfsc::tosa::OperatorEnum::Cast)),
        });
    TosaLevelRequirements requirements;
    std::string reason;
    EXPECT_FALSE(
        CollectTosaLevelRequirements(*base.module, requirements, reason));
    EXPECT_EQ(reason, "Cannot determine emitted TOSA level requirements");
  }

  // Non-instruction input operand.
  {
    MinimalModule base = CreateMinimalValidModule();
    ASSERT_NE(base.tosa_import, nullptr);
    ASSERT_NE(base.tensor_type, nullptr);
    base.module->EmplaceInstructionNonUnique(
        spv::Op::OpExtInst,
        std::vector<tfsc::spirv::Operand>{
            tfsc::spirv::Operand(base.tensor_type),
            tfsc::spirv::RESID,
            tfsc::spirv::Operand(base.tosa_import),
            tfsc::spirv::Operand(
                static_cast<uint32_t>(tfsc::tosa::OperatorEnum::Cast)),
            tfsc::spirv::Operand(0u),
        });
    TosaLevelRequirements requirements;
    std::string reason;
    EXPECT_FALSE(
        CollectTosaLevelRequirements(*base.module, requirements, reason));
    EXPECT_EQ(reason, "Cannot determine emitted TOSA level requirements");
  }

  // Input operand points to an instruction with empty operands.
  {
    MinimalModule base = CreateMinimalValidModule();
    ASSERT_NE(base.tosa_import, nullptr);
    ASSERT_NE(base.tensor_type, nullptr);
    tfsc::spirv::Instruction empty_input(spv::Op::OpNop, {});
    base.module->EmplaceInstructionNonUnique(
        spv::Op::OpExtInst,
        std::vector<tfsc::spirv::Operand>{
            tfsc::spirv::Operand(base.tensor_type),
            tfsc::spirv::RESID,
            tfsc::spirv::Operand(base.tosa_import),
            tfsc::spirv::Operand(
                static_cast<uint32_t>(tfsc::tosa::OperatorEnum::Cast)),
            tfsc::spirv::Operand(&empty_input),
        });
    TosaLevelRequirements requirements;
    std::string reason;
    EXPECT_FALSE(
        CollectTosaLevelRequirements(*base.module, requirements, reason));
    EXPECT_EQ(reason, "Cannot determine emitted TOSA level requirements");
  }

  // Non-instruction output type operand.
  {
    MinimalModule base = CreateMinimalValidModule();
    ASSERT_NE(base.tosa_import, nullptr);
    base.module->EmplaceInstructionNonUnique(
        spv::Op::OpExtInst,
        std::vector<tfsc::spirv::Operand>{
            tfsc::spirv::Operand(0u),
            tfsc::spirv::RESID,
            tfsc::spirv::Operand(base.tosa_import),
            tfsc::spirv::Operand(
                static_cast<uint32_t>(tfsc::tosa::OperatorEnum::Cast)),
            tfsc::spirv::Operand(base.input),
        });
    TosaLevelRequirements requirements;
    std::string reason;
    EXPECT_FALSE(
        CollectTosaLevelRequirements(*base.module, requirements, reason));
    EXPECT_EQ(reason, "Cannot determine emitted TOSA level requirements");
  }

  // Output type operand points to a 4-operand non-OpTypeTensorARM instruction
  // carrying a valid distinct element type at operand 1.
  {
    MinimalModule base = CreateMinimalValidModule();
    ASSERT_NE(base.tosa_import, nullptr);
    tfsc::spirv::Instruction int8_type(
        spv::Op::OpTypeInt, {tfsc::spirv::RESID, tfsc::spirv::Operand(8u),
                             tfsc::spirv::Operand(0u)});
    tfsc::spirv::Instruction non_tensor(
        spv::Op::OpNop,
        {tfsc::spirv::RESID, tfsc::spirv::Operand(&int8_type),
         tfsc::spirv::Operand(0u), tfsc::spirv::Operand(0u)});
    base.module->EmplaceInstructionNonUnique(
        spv::Op::OpExtInst,
        std::vector<tfsc::spirv::Operand>{
            tfsc::spirv::Operand(&non_tensor),
            tfsc::spirv::RESID,
            tfsc::spirv::Operand(base.tosa_import),
            tfsc::spirv::Operand(
                static_cast<uint32_t>(tfsc::tosa::OperatorEnum::Cast)),
            tfsc::spirv::Operand(base.input),
        });
    TosaLevelRequirements requirements;
    std::string reason;
    EXPECT_FALSE(
        CollectTosaLevelRequirements(*base.module, requirements, reason));
    EXPECT_EQ(reason, "Cannot determine emitted TOSA level requirements");
  }

  // Output type operand points to a truncated OpTypeTensorARM (2 operands)
  // carrying a valid distinct element type at operand 1.
  {
    MinimalModule base = CreateMinimalValidModule();
    ASSERT_NE(base.tosa_import, nullptr);
    tfsc::spirv::Instruction int8_type(
        spv::Op::OpTypeInt, {tfsc::spirv::RESID, tfsc::spirv::Operand(8u),
                             tfsc::spirv::Operand(0u)});
    tfsc::spirv::Instruction truncated_tensor(
        spv::Op::OpTypeTensorARM,
        {tfsc::spirv::RESID, tfsc::spirv::Operand(&int8_type)});
    base.module->EmplaceInstructionNonUnique(
        spv::Op::OpExtInst,
        std::vector<tfsc::spirv::Operand>{
            tfsc::spirv::Operand(&truncated_tensor),
            tfsc::spirv::RESID,
            tfsc::spirv::Operand(base.tosa_import),
            tfsc::spirv::Operand(
                static_cast<uint32_t>(tfsc::tosa::OperatorEnum::Cast)),
            tfsc::spirv::Operand(base.input),
        });
    TosaLevelRequirements requirements;
    std::string reason;
    EXPECT_FALSE(
        CollectTosaLevelRequirements(*base.module, requirements, reason));
    EXPECT_EQ(reason, "Cannot determine emitted TOSA level requirements");
  }
}

TEST(TosaRequirementsTest, RejectsInvalidTensorElementTypes) {
  struct InvalidElementCase {
    const char* name;
    spv::Op opcode;
    std::vector<tfsc::spirv::Operand> operands;
    bool use_literal_element_operand = false;
  };
  const std::vector<InvalidElementCase> cases = {
      {"NonInstructionElementOperand", spv::Op::OpNop, {}, true},
      {"UnsupportedOpcodeVoid", spv::Op::OpTypeVoid, {tfsc::spirv::RESID}},
      {"TruncatedIntType", spv::Op::OpTypeInt, {tfsc::spirv::RESID}},
      {"OverlongIntType",
       spv::Op::OpTypeInt,
       {tfsc::spirv::RESID, tfsc::spirv::Operand(32u),
        tfsc::spirv::Operand(0u), tfsc::spirv::Operand(0u)}},
      {"NonLiteralIntBitWidth",
       spv::Op::OpTypeInt,
       {tfsc::spirv::RESID, tfsc::spirv::RESID, tfsc::spirv::Operand(0u)}},
      {"NonLiteralFloatExtraOperand",
       spv::Op::OpTypeFloat,
       {tfsc::spirv::RESID, tfsc::spirv::Operand(16u), tfsc::spirv::RESID}},
  };
  for (const InvalidElementCase& tc : cases) {
    SCOPED_TRACE(tc.name);
    MinimalModule base = CreateMinimalValidModule();
    if (base.tensor_type == nullptr) {
      ADD_FAILURE() << "Missing OpTypeTensorARM in baseline graph";
      continue;
    }
    tfsc::spirv::Instruction bad_element(tc.opcode, tc.operands);
    tfsc::spirv::Operand element_operand =
        tc.use_literal_element_operand ? tfsc::spirv::Operand(0u)
                                       : tfsc::spirv::Operand(&bad_element);
    base.module->EmplaceInstructionNonUnique(
        spv::Op::OpTypeTensorARM,
        std::vector<tfsc::spirv::Operand>{
            tfsc::spirv::RESID,
            element_operand,
            base.tensor_type->m_Operands[2],
            base.tensor_type->m_Operands[3],
        });
    TosaLevelRequirements requirements;
    std::string reason;
    EXPECT_FALSE(
        CollectTosaLevelRequirements(*base.module, requirements, reason));
    EXPECT_EQ(reason, "Cannot determine emitted TOSA level requirements");
  }
}

TEST(TosaRequirementsTest, DistinguishesCastElementTypesByOpcodeAndExtra) {
  struct CastTypePairCase {
    const char* name;
    spv::Op in_opcode;
    std::vector<tfsc::spirv::Operand> in_operands;
    spv::Op out_opcode;
    std::vector<tfsc::spirv::Operand> out_operands;
    bool expect_valid;
  };
  const std::vector<CastTypePairCase> cases = {
      {"IntSignednessMismatch",
       spv::Op::OpTypeInt,
       {tfsc::spirv::RESID, tfsc::spirv::Operand(32u),
        tfsc::spirv::Operand(0u)},
       spv::Op::OpTypeInt,
       {tfsc::spirv::RESID, tfsc::spirv::Operand(32u),
        tfsc::spirv::Operand(1u)},
       true},
      {"IntSameSignednessRejected",
       spv::Op::OpTypeInt,
       {tfsc::spirv::RESID, tfsc::spirv::Operand(32u),
        tfsc::spirv::Operand(1u)},
       spv::Op::OpTypeInt,
       {tfsc::spirv::RESID, tfsc::spirv::Operand(32u),
        tfsc::spirv::Operand(1u)},
       false},
      {"IntExtraPresenceMismatch",
       spv::Op::OpTypeInt,
       {tfsc::spirv::RESID, tfsc::spirv::Operand(32u)},
       spv::Op::OpTypeInt,
       {tfsc::spirv::RESID, tfsc::spirv::Operand(32u),
        tfsc::spirv::Operand(0u)},
       true},
      {"IntVsFloatSameWidthAndNoExtra",
       spv::Op::OpTypeInt,
       {tfsc::spirv::RESID, tfsc::spirv::Operand(32u)},
       spv::Op::OpTypeFloat,
       {tfsc::spirv::RESID, tfsc::spirv::Operand(32u)},
       true},
  };
  for (const CastTypePairCase& tc : cases) {
    SCOPED_TRACE(tc.name);
    MinimalModule base = CreateMinimalValidModule();
    if (base.tosa_import == nullptr || base.tensor_type == nullptr) {
      ADD_FAILURE() << "Missing baseline SPIR-V instructions";
      continue;
    }
    tfsc::spirv::Instruction in_elem(tc.in_opcode, tc.in_operands);
    tfsc::spirv::Instruction out_elem(tc.out_opcode, tc.out_operands);
    tfsc::spirv::Instruction in_tensor(
        spv::Op::OpTypeTensorARM,
        {tfsc::spirv::RESID, tfsc::spirv::Operand(&in_elem),
         base.tensor_type->m_Operands[2], base.tensor_type->m_Operands[3]});
    tfsc::spirv::Instruction out_tensor(
        spv::Op::OpTypeTensorARM,
        {tfsc::spirv::RESID, tfsc::spirv::Operand(&out_elem),
         base.tensor_type->m_Operands[2], base.tensor_type->m_Operands[3]});
    tfsc::spirv::Instruction cast_input(
        spv::Op::OpUndef,
        {tfsc::spirv::Operand(&in_tensor), tfsc::spirv::RESID});
    base.module->EmplaceInstructionNonUnique(
        spv::Op::OpExtInst,
        std::vector<tfsc::spirv::Operand>{
            tfsc::spirv::Operand(&out_tensor),
            tfsc::spirv::RESID,
            tfsc::spirv::Operand(base.tosa_import),
            tfsc::spirv::Operand(
                static_cast<uint32_t>(tfsc::tosa::OperatorEnum::Cast)),
            tfsc::spirv::Operand(&cast_input),
        });
    TosaLevelRequirements requirements;
    std::string reason;
    if (tc.expect_valid) {
      EXPECT_TRUE(
          CollectTosaLevelRequirements(*base.module, requirements, reason))
          << reason;
      EXPECT_TRUE(requirements.checked);
    } else {
      EXPECT_FALSE(
          CollectTosaLevelRequirements(*base.module, requirements, reason));
      EXPECT_EQ(reason,
                "TOSA CAST requires distinct input and output element types");
    }
  }
}
}  // namespace
}  // namespace litert::arm_vulkan_ml
