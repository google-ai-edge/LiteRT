// Copyright 2025 Google LLC.
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

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"

namespace litert::internal {
namespace {

static constexpr auto kScale = 1.0f;
static constexpr auto kZero = 1L;
static constexpr auto kTensorName = "test_tensor";
static constexpr absl::string_view kData = "911GT3RS";

using ::testing::ElementsAreArray;

TEST(BuilderTest, InitializeBuilder) {
  LiteRtSubgraphT subgraph;
  subgraph.EmplaceOp();
  LiteRtBuilderT builder;

  EXPECT_EQ(subgraph.Ops().size(), 1);
}

TEST(BuilderTest, BuildRuntimeTensorWithoutWeights) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;
  auto weights = LiteRtWeightsT();
  const auto quant = MakePerTensorQuantization(kScale, kZero);
  const auto tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2});
  auto& tensor = builder.BuildTensor(weights, quant, tensor_type, kTensorName);

  auto built_tensor_type = tensor.Type();

  EXPECT_EQ(built_tensor_type.first, kLiteRtRankedTensorType);
  EXPECT_EQ(built_tensor_type.second.ranked_tensor_type.element_type,
            kLiteRtElementTypeInt32);
  EXPECT_EQ(built_tensor_type.second.ranked_tensor_type.layout.rank, 3);
  EXPECT_THAT(
      absl::MakeConstSpan(
          built_tensor_type.second.ranked_tensor_type.layout.dimensions, 3),
      ElementsAreArray({2, 2, 2}));

  auto built_quant = tensor.Qparams();

  EXPECT_EQ(built_quant.first, kLiteRtQuantizationPerTensor);
  EXPECT_EQ(built_quant.second.per_tensor.scale, kScale);
  EXPECT_EQ(built_quant.second.per_tensor.zero_point, kZero);
  EXPECT_EQ(tensor.Name(), kTensorName);
}

inline LiteRtTensorT BuildSimpleTensor() {
  LiteRtTensorT tensor;
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
  tensor.SetQarams(MakePerTensorQuantization(kScale, kZero));
  tensor.SetName(kTensorName);
  return tensor;
}

TEST(BuilderTest, BuildRuntimeTensorFromExistingTensor) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;
  auto tensor = BuildSimpleTensor();
  auto& built_tensor = builder.BuildTensor(tensor);
  auto built_tensor_type = built_tensor.Type();

  EXPECT_EQ(built_tensor_type.first, kLiteRtRankedTensorType);
  EXPECT_EQ(built_tensor_type.second.ranked_tensor_type.element_type,
            kLiteRtElementTypeInt32);
  EXPECT_EQ(built_tensor_type.second.ranked_tensor_type.layout.rank, 3);
  EXPECT_THAT(
      absl::MakeConstSpan(
          built_tensor_type.second.ranked_tensor_type.layout.dimensions, 3),
      ElementsAreArray({2, 2, 2}));
  EXPECT_EQ(builder.IsTensorAllocated(&built_tensor), true);
  EXPECT_EQ(builder.Subgraph().Tensors().size(), 1);
}

TEST(BuilderTest, BuildTensorWithExistingTensorWithWeights) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  {
    OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(tensor.Weights(), std::move(buf));
  }
  LiteRtBuilderT builder;
  auto& built_tensor = builder.BuildTensor(tensor);

  EXPECT_EQ(built_tensor.Weights().GetBufferId(), 1);
  EXPECT_EQ(built_tensor.Weights().GetBufferManager(),
            tensor.Weights().GetBufferManager());
  EXPECT_EQ(built_tensor.Weights().Buffer().Size(), kData.size());
}

TEST(BuilderTest, BuildTensorWithWeights) {
  LiteRtSubgraphT subgraph;
  BufferManager manager;
  LiteRtWeightsT weights;
  {
    weights.SetBufferManager(&manager);
    OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }
  LiteRtBuilderT builder;
  auto& built_tensor = builder.BuildTensor(
      weights, MakePerTensorQuantization(kScale, kZero),
      MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}), kTensorName);

  EXPECT_EQ(built_tensor.Weights().GetBufferId(), 1);
  EXPECT_EQ(built_tensor.Weights().GetBufferManager(), &manager);
  EXPECT_EQ(built_tensor.Weights().Buffer().Size(), kData.size());
}

TEST(BuilderTest, BuildOp) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;
  auto tensor = BuildSimpleTensor();
  auto& built_tensor_0 = builder.BuildTensor(tensor);
  auto& built_tensor_1 = builder.BuildTensor(tensor);
  auto& built_op = builder.BuildOp(kLiteRtOpCodeTflAdd, {&built_tensor_0},
                                   {&built_tensor_1});

  EXPECT_EQ(built_op.OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(built_op.Inputs().size(), 1);
  EXPECT_EQ(built_op.Outputs().size(), 1);
  EXPECT_EQ(built_tensor_0.GetUse(0).first, &built_op);
  EXPECT_EQ(built_tensor_0.GetUse(0).second, 0);
  EXPECT_EQ(built_tensor_1.DefiningOp(), &built_op);
}

TEST(BuilderTest, BuildOpFromExistingOp) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;
  auto tensor = BuildSimpleTensor();
  auto& built_tensor_0 = builder.BuildTensor(tensor);
  auto& built_tensor_1 = builder.BuildTensor(tensor);

  auto op = LiteRtOpT();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& built_op = builder.BuildOp(op, {&built_tensor_0}, {&built_tensor_1});

  EXPECT_EQ(built_op.OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(built_op.Inputs().size(), 1);
  EXPECT_EQ(built_op.Outputs().size(), 1);
  EXPECT_EQ(built_tensor_0.GetUse(0).first, &built_op);
  EXPECT_EQ(built_tensor_0.GetUse(0).second, 0);
  EXPECT_EQ(built_tensor_1.DefiningOp(), &built_op);
  EXPECT_EQ(builder.IsOpAllocated(&built_op), true);
  EXPECT_EQ(builder.Subgraph().Ops().size(), 1);
}

TEST(BuilderTest, EraseOp) {
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder;
  auto& op_to_erase = subgraph.EmplaceOp();
  builder.EraseOp(&op_to_erase);
  EXPECT_EQ(builder.Erases().size(), 1);
  EXPECT_EQ(builder.Erases().contains(&op_to_erase), true);
}

TEST(BuilderTest, UncommittedTransformation) {
  LiteRtSubgraphT subgraph;
  auto& op_to_replace = subgraph.EmplaceOp();
  op_to_replace.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& tensor_0 = subgraph.EmplaceTensor();
  auto& tensor_1 = subgraph.EmplaceTensor();
  AttachInput(&tensor_0, op_to_replace);
  AttachOutput(&tensor_1, op_to_replace);

  LiteRtBuilderT builder;
  builder.BuildOp(kLiteRtOpCodeTflMul, {&tensor_0}, {&tensor_1});
  builder.EraseOp(&op_to_replace);

  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(&subgraph.Ops().front()->Input(0), &tensor_0);
  EXPECT_EQ(&subgraph.Ops().front()->Output(0), &tensor_1);
  EXPECT_EQ(subgraph.Ops().front()->Input(0).Users().front(), &op_to_replace);
  EXPECT_EQ(subgraph.Ops().front()->Output(0).DefiningOp(), &op_to_replace);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(subgraph.Tensors().size(), 2);
}

TEST(BuilderTest, AddOpToMulOpTransformation) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op_to_replace = subgraph.EmplaceOp();
  op_to_replace.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& tensor_0 = subgraph.EmplaceTensor();
  auto& tensor_1 = subgraph.EmplaceTensor();
  tensor_0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
  tensor_1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}));
  AttachInput(&tensor_0, op_to_replace);
  AttachOutput(&tensor_1, op_to_replace);

  LiteRtBuilderT builder;
  builder.BuildOp(kLiteRtOpCodeTflMul, {&tensor_0}, {&tensor_1});
  builder.EraseOp(&op_to_replace);
  builder.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->Inputs().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->Outputs().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflMul);
  EXPECT_EQ(subgraph.Tensors().size(), 2);

  // Serialize and deserialize the model to verify the changes.
  auto serialized = SerializeModel(std::move(model));
  ASSERT_TRUE(VerifyFlatbuffer(serialized->Span()));
  auto model_wrap = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(model_wrap);
  EXPECT_EQ(model_wrap->get()->Unpack()->subgraphs.size(), 1);
}

TEST(BuilderTest, AddOpToMulOpAndAddOpTransformation) {
  LiteRtSubgraphT subgraph;
  auto& op_to_replace = subgraph.EmplaceOp();
  op_to_replace.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& tensor_0 = subgraph.EmplaceTensor();
  auto& tensor_1 = subgraph.EmplaceTensor();
  AttachInput(&tensor_0, op_to_replace);
  AttachOutput(&tensor_1, op_to_replace);

  LiteRtBuilderT builder;
  auto& built_tensor_0 = builder.BuildTensor(tensor_0);
  builder.BuildOp(kLiteRtOpCodeTflMul, {&tensor_0}, {&built_tensor_0});
  builder.BuildOp(kLiteRtOpCodeTflAdd, {&built_tensor_0}, {&tensor_1});
  builder.EraseOp(&op_to_replace);
  builder.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 2);
  EXPECT_EQ(subgraph.Ops().at(0)->OpCode(), kLiteRtOpCodeTflMul);
  EXPECT_EQ(subgraph.Ops().at(1)->OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(subgraph.Tensors().size(), 3);
}

TEST(BuilderTest, AddOpAndMulOpToDivOpTransformation) {
  LiteRtSubgraphT subgraph;
  auto& add_op_to_replace = subgraph.EmplaceOp();
  auto& mul_op_to_replace = subgraph.EmplaceOp();
  add_op_to_replace.SetOpCode(kLiteRtOpCodeTflAdd);
  mul_op_to_replace.SetOpCode(kLiteRtOpCodeTflMul);
  auto& tensor_0 = subgraph.EmplaceTensor();
  auto& tensor_1 = subgraph.EmplaceTensor();
  auto& tensor_2 = subgraph.EmplaceTensor();
  AttachInput(&tensor_0, add_op_to_replace);
  AttachOutput(&tensor_1, add_op_to_replace);
  AttachInput(&tensor_1, mul_op_to_replace);
  AttachOutput(&tensor_2, mul_op_to_replace);

  LiteRtBuilderT builder;
  builder.BuildOp(kLiteRtOpCodeTflDiv, {&tensor_0}, {&tensor_2});
  builder.EraseOp(&add_op_to_replace);
  builder.EraseOp(&mul_op_to_replace);
  builder.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->Inputs().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->Outputs().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflDiv);
  EXPECT_EQ(subgraph.Tensors().size(), 2);
}

TEST(BuilderTest, BuildWeightsSuccess) {
  static constexpr absl::string_view kData = "911GT3RS";
  absl::Span<const uint8_t> data = absl::MakeConstSpan(
      reinterpret_cast<const uint8_t*>(kData.data()), kData.size());
  LiteRtBuilderT builder;
  LiteRtWeightsT null_weights;
  null_weights.SetBufferManager(nullptr);
  auto& tensor =
      builder.BuildTensor(null_weights, Quantization(), TensorType());
  auto& weights = builder.BuildWeights(data.data(), data.size(), &tensor);
  EXPECT_EQ(weights.Buffer().Size(), kData.size());
  EXPECT_EQ(tensor.Weights().Buffer().Size(), kData.size());
  EXPECT_THAT(
      absl::MakeConstSpan(weights.Buffer().Data(), weights.Buffer().Size()),
      testing::ElementsAreArray(reinterpret_cast<const uint8_t*>(kData.data()),
                                kData.size()));
}

TEST(BuilderTest, TransferWeightsAfterApplyingChangesSuccess) {
  static constexpr absl::string_view kData = "911GT3RS";
  absl::Span<const uint8_t> data = absl::MakeConstSpan(
      reinterpret_cast<const uint8_t*>(kData.data()), kData.size());
  BufferManager manager;
  LiteRtSubgraphT subgraph = LiteRtSubgraphT(&manager);
  // Scope to ensure that the builder is destroyed after the changes are
  // applied.
  {
    LiteRtBuilderT builder;
    LiteRtTensorT input_tensor;
    LiteRtTensorT output_tensor;
    LiteRtWeightsT null_weights;
    null_weights.SetBufferManager(nullptr);
    auto& const_tensor = builder.BuildTensor(null_weights, Quantization(),
                                             TensorType(), kTensorName);

    auto& weights =
        builder.BuildWeights(data.data(), data.size(), &const_tensor);
    EXPECT_THAT(
        absl::MakeConstSpan(weights.Buffer().Data(), weights.Buffer().Size()),
        testing::ElementsAreArray(
            reinterpret_cast<const uint8_t*>(kData.data()), kData.size()));
    builder.BuildOp(kLiteRtOpCodeTflAdd, {&const_tensor, &input_tensor},
                    {&output_tensor});
    builder.ApplyChanges(&subgraph);
  }
  ASSERT_EQ(subgraph.Ops().size(), 1);
  ASSERT_EQ(subgraph.Ops().front()->Inputs().size(), 2);
  auto const_tensor_after_apply_changes =
      subgraph.Ops().front()->Inputs().front();
  EXPECT_EQ(const_tensor_after_apply_changes->Weights().Buffer().Size(),
            kData.size());
  EXPECT_THAT(
      absl::MakeConstSpan(
          const_tensor_after_apply_changes->Weights().Buffer().Data(),
          const_tensor_after_apply_changes->Weights().Buffer().Size()),
      testing::ElementsAreArray(reinterpret_cast<const uint8_t*>(kData.data()),
                                kData.size()));
}

TEST(BuilderTest, ApplyChangesConnectivityIndexing) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& op_to_replace = subgraph.EmplaceOp();
  op_to_replace.SetOpCode(kLiteRtOpCodeTflAdd);

  auto& input0 = subgraph.EmplaceTensor();
  auto& input1 = subgraph.EmplaceTensor();
  auto& output = subgraph.EmplaceTensor();

  // Setup inputs/outputs for the op to replace
  input0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2}));
  input1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2}));
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {2}));

  AttachInput(&input0, op_to_replace);
  AttachInput(&input1, op_to_replace);
  AttachOutput(&output, op_to_replace);

  // Builder: Replace Add with Mul
  LiteRtBuilderT builder;
  builder.BuildOp(kLiteRtOpCodeTflMul, {&input0, &input1}, {&output});
  builder.EraseOp(&op_to_replace);

  builder.ApplyChanges(&subgraph);

  // Verify connectivity
  ASSERT_EQ(subgraph.Ops().size(), 1);
  auto& new_op = *subgraph.Ops().front();
  EXPECT_EQ(new_op.OpCode(), kLiteRtOpCodeTflMul);

  // Verify inputs
  ASSERT_EQ(new_op.Inputs().size(), 2);
  EXPECT_EQ(new_op.Inputs()[0], &input0);
  EXPECT_EQ(new_op.Inputs()[1], &input1);

  // Verify back-edges (Users and UserArgInds) - This catches the bug
  // input0 should have user new_op at arg index 0
  ASSERT_EQ(input0.Users().size(), 1);
  EXPECT_EQ(input0.Users()[0], &new_op);
  EXPECT_EQ(input0.UserArgInds()[0], 0);

  // input1 should have user new_op at arg index 1
  ASSERT_EQ(input1.Users().size(), 1);
  EXPECT_EQ(input1.Users()[0], &new_op);
  EXPECT_EQ(input1.UserArgInds()[0], 1);
}

TEST(BuilderTest, ApplyChangesSpliceIndexAfterDCE) {
  // Test case where upstream dead op is removed, shifting indices.
  // Original: DeadOp(0) -> ReplacedOp(1) -> DownstreamOp(2)
  // DCE removes DeadOp and ReplacedOp. DownstreamOp shifts to 0.
  // Old splice_index = 1.
  // Correct insertion: 0 (before DownstreamOp).
  // Incorrect insertion: 1 (after DownstreamOp).

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  // 1. Dead Op (Op0) - Floating op that will be removed by DCE
  auto& dead_op = subgraph.EmplaceOp();
  dead_op.SetOpCode(kLiteRtOpCodeTflAdd);

  // 2. Op to replace (Op1)
  auto& op_to_replace = subgraph.EmplaceOp();
  op_to_replace.SetOpCode(kLiteRtOpCodeTflSub);
  auto& input = subgraph.EmplaceTensor();  // Graph input
  auto& replace_out = subgraph.EmplaceTensor();
  AttachInput(&input, op_to_replace);
  AttachOutput(&replace_out, op_to_replace);

  // 3. Downstream Op (Op2)
  auto& downstream_op = subgraph.EmplaceOp();
  downstream_op.SetOpCode(kLiteRtOpCodeTflMul);
  auto& final_out = subgraph.EmplaceTensor();  // Graph output
  AttachInput(&replace_out, downstream_op);    // Connect to op_to_replace
  AttachOutput(&final_out, downstream_op);

  // Verify initial indices (implicitly 0, 1, 2 due to emplace order)
  ASSERT_EQ(subgraph.Ops().size(), 3);

  LiteRtBuilderT builder;
  // Build new op using same input and output as replaced op
  builder.BuildOp(kLiteRtOpCodeTflDiv, {&input}, {&replace_out});
  builder.EraseOp(&op_to_replace);

  // Apply changes
  // DCE will remove dead_op (unused) and op_to_replace (erased).
  // downstream_op remains.
  // new_op (Div) should be inserted BEFORE downstream_op.
  builder.ApplyChanges(&subgraph);

  ASSERT_EQ(subgraph.Ops().size(), 2);
  // Expected order: Div, Mul
  EXPECT_EQ(subgraph.Ops()[0]->OpCode(), kLiteRtOpCodeTflDiv);
  EXPECT_EQ(subgraph.Ops()[1]->OpCode(), kLiteRtOpCodeTflMul);
}

// Helpers for complex graph tests
LiteRtTensorT& MakeTensor(LiteRtSubgraphT& subgraph) {
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2}));
  return tensor;
}

LiteRtOpT& MakeOp(LiteRtSubgraphT& subgraph, LiteRtOpCode code,
                  const std::vector<LiteRtTensorT*>& inputs,
                  const std::vector<LiteRtTensorT*>& outputs) {
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(code);
  for (auto* input : inputs) {
    AttachInput(input, op);
  }
  for (auto* output : outputs) {
    AttachOutput(output, op);
  }
  return op;
}

TEST(BuilderTest, GridTopologyReplaceCenter) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  // Create 3x3 grid of tensors and ops
  // T00 T01 T02
  //  |   |   |
  // Op00 Op01 Op02
  //  |   |   |
  // T10 T11 T12
  //  |   |   |
  // Op10 Op11 Op12
  //  |   |   |
  // T20 T21 T22
  //  |   |   |
  // Op20 Op21 Op22
  //  |   |   |
  // T30 T31 T32

  std::vector<std::vector<LiteRtTensorT*>> tensors(
      4, std::vector<LiteRtTensorT*>(3));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      tensors[i][j] = &MakeTensor(subgraph);
    }
  }

  std::vector<std::vector<LiteRtOpT*>> ops(3, std::vector<LiteRtOpT*>(3));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // Op connects T(i, j) to T(i+1, j)
      ops[i][j] = &MakeOp(subgraph, kLiteRtOpCodeTflAdd, {tensors[i][j]},
                          {tensors[i + 1][j]});
    }
  }

  // Target: Center Op ops[1][1] (Middle row, middle col)
  auto* target_op = ops[1][1];
  auto* input = tensors[1][1];
  auto* output = tensors[2][1];

  LiteRtBuilderT builder;
  // Replace with Mul
  builder.BuildOp(kLiteRtOpCodeTflMul, {input}, {output});
  builder.EraseOp(target_op);

  builder.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 9);

  // Verify by finding the op that produces 'output'.
  auto* defining_op = output->DefiningOp();
  ASSERT_NE(defining_op, nullptr);
  EXPECT_EQ(defining_op->OpCode(), kLiteRtOpCodeTflMul);
}

TEST(BuilderTest, TreeTopologyDeleteLeaves) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  // Binary Tree Depth 2
  //       Root
  //      /    \
  //    L1      R1
  //   /  \    /  \
  // L2a L2b  R2a R2b

  auto& root_in = MakeTensor(subgraph);
  auto& l1_in = MakeTensor(subgraph);
  auto& r1_in = MakeTensor(subgraph);
  auto& l2a_in = MakeTensor(subgraph);
  auto& l2b_in = MakeTensor(subgraph);
  auto& r2a_in = MakeTensor(subgraph);
  auto& r2b_in = MakeTensor(subgraph);

  MakeOp(subgraph, kLiteRtOpCodeTflSplit, {&root_in}, {&l1_in, &r1_in});
  MakeOp(subgraph, kLiteRtOpCodeTflSplit, {&l1_in}, {&l2a_in, &l2b_in});
  MakeOp(subgraph, kLiteRtOpCodeTflSplit, {&r1_in}, {&r2a_in, &r2b_in});

  // Ops at leaves (consumers)
  auto& op_l2a = MakeOp(subgraph, kLiteRtOpCodeTflAbs, {&l2a_in}, {});
  auto& op_l2b = MakeOp(subgraph, kLiteRtOpCodeTflAbs, {&l2b_in}, {});
  auto& op_r2a = MakeOp(subgraph, kLiteRtOpCodeTflAbs, {&r2a_in}, {});
  auto& op_r2b = MakeOp(subgraph, kLiteRtOpCodeTflAbs, {&r2b_in}, {});

  ASSERT_EQ(subgraph.Ops().size(), 7);

  LiteRtBuilderT builder;
  // Delete all leaf ops
  builder.EraseOp(&op_l2a);
  builder.EraseOp(&op_l2b);
  builder.EraseOp(&op_r2a);
  builder.EraseOp(&op_r2b);

  builder.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 3);
  // Verify remaining ops are splits
  for (auto* op : subgraph.Ops()) {
    EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflSplit);
  }
}

class BuilderRandomGraphTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override { rng_.seed(GetParam()); }
  std::mt19937 rng_;
};

TEST_P(BuilderRandomGraphTest, RandomMutations) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  // 1. Generate Random DAG
  // Structure: N layers, each layer has W nodes.
  // Each node in layer i connects to a random node in layer i+1.
  int num_layers = 10;
  int width = 3;

  std::vector<std::vector<LiteRtTensorT*>> tensors(num_layers + 1);
  for (int i = 0; i <= num_layers; ++i) {
    for (int j = 0; j < width; ++j) {
      tensors[i].push_back(&MakeTensor(subgraph));
    }
  }

  std::vector<LiteRtOpT*> ops;
  for (int i = 0; i < num_layers; ++i) {
    for (int j = 0; j < width; ++j) {
      // Op connecting Layer i, Node j -> Layer i+1, Random Node k
      std::uniform_int_distribution<int> dist(0, width - 1);
      int next_node = dist(rng_);

      std::vector<LiteRtTensorT*> inputs = {tensors[i][j]};
      if (width > 1) {
        inputs.push_back(tensors[i][next_node]);
      }
      std::vector<LiteRtTensorT*> outputs = {tensors[i + 1][j]};

      // OpCode doesn't matter much, use ADD
      ops.push_back(&MakeOp(subgraph, kLiteRtOpCodeTflAdd, inputs, outputs));
    }
  }

  size_t original_op_count = ops.size();
  ASSERT_EQ(original_op_count, num_layers * width);

  // 2. Perform Random Mutations
  LiteRtBuilderT builder;
  int num_mutations = 10;
  size_t expected_op_count_change = 0;

  std::vector<LiteRtOpT*> erased_ops;

  for (int k = 0; k < num_mutations; ++k) {
    std::uniform_int_distribution<int> op_dist(0, ops.size() - 1);
    int op_idx = op_dist(rng_);
    auto* op = ops[op_idx];

    // Check if already erased
    bool already_erased = false;
    for (auto* e : erased_ops)
      if (e == op) already_erased = true;
    if (already_erased) continue;

    std::uniform_int_distribution<int> action_dist(0, 2);
    int action = action_dist(rng_);

    if (action == 0) {  // ERASE
      builder.EraseOp(op);
      erased_ops.push_back(op);
      expected_op_count_change--;
    } else if (action == 1) {  // REPLACE
      // Build new op with same inputs/outputs
      // Use MUL instead of ADD
      builder.BuildOp(kLiteRtOpCodeTflMul, op->Inputs(), op->Outputs());
      builder.EraseOp(op);
      erased_ops.push_back(op);
      // Count change is 0 ( -1 + 1 )
    } else {  // ADD PARALLEL
      // Add another op with same inputs/outputs
      builder.BuildOp(kLiteRtOpCodeTflSub, op->Inputs(), op->Outputs());
      expected_op_count_change++;
    }
  }

  builder.ApplyChanges(&subgraph);

  // 3. Verify
  size_t expected_ops = original_op_count + expected_op_count_change;
  EXPECT_EQ(subgraph.Ops().size(), expected_ops);

  // Verify that erased ops are NOT in the subgraph
  for (auto* op : subgraph.Ops()) {
    for (auto* erased : erased_ops) {
      EXPECT_NE(op, erased);
    }
  }
}

TEST_P(BuilderRandomGraphTest, RandomChainCollapse) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  // Create a long chain
  // In -> Op0 -> T0 -> Op1 -> T1 ... -> OpN -> Out

  std::uniform_int_distribution<int> length_dist(2, 10);
  int chain_length = length_dist(rng_);

  auto* input_tensor = &MakeTensor(subgraph);
  subgraph.Inputs().push_back(input_tensor);  // Mark as subgraph input

  std::vector<LiteRtOpT*> chain_ops;
  std::vector<LiteRtTensorT*> intermediate_tensors;

  LiteRtTensorT* current_input = input_tensor;
  for (int i = 0; i < chain_length; ++i) {
    auto* output_tensor = &MakeTensor(subgraph);
    auto* op = &MakeOp(subgraph, kLiteRtOpCodeTflAdd, {current_input},
                       {output_tensor});
    chain_ops.push_back(op);
    intermediate_tensors.push_back(output_tensor);
    current_input = output_tensor;
  }

  auto* final_output_tensor = intermediate_tensors.back();
  subgraph.Outputs().push_back(final_output_tensor);  // Mark as subgraph output

  // Collapse Chain: Replace all ops with a single MUL op
  // New Op: Input -> MUL -> Out

  LiteRtBuilderT builder;
  builder.BuildOp(kLiteRtOpCodeTflMul, {input_tensor}, {final_output_tensor});

  for (auto* op : chain_ops) {
    builder.EraseOp(op);
  }

  builder.ApplyChanges(&subgraph);

  // Verify
  EXPECT_EQ(subgraph.Ops().size(), 1);
  auto* new_op = subgraph.Ops().front();
  EXPECT_EQ(new_op->OpCode(), kLiteRtOpCodeTflMul);

  // Check Connectivity
  ASSERT_EQ(new_op->Inputs().size(), 1);
  EXPECT_EQ(new_op->Inputs()[0], input_tensor);

  ASSERT_EQ(new_op->Outputs().size(), 1);
  EXPECT_EQ(new_op->Outputs()[0], final_output_tensor);

  // Check Subgraph I/O are preserved
  EXPECT_EQ(subgraph.Inputs().size(), 1);
  EXPECT_EQ(subgraph.Inputs()[0], input_tensor);
  EXPECT_EQ(subgraph.Outputs().size(), 1);
  EXPECT_EQ(subgraph.Outputs()[0], final_output_tensor);

  // Check Use-Def chains
  ASSERT_EQ(input_tensor->Users().size(), 1);
  EXPECT_EQ(input_tensor->Users()[0], new_op);

  EXPECT_EQ(final_output_tensor->DefiningOp(), new_op);
}

TEST_P(BuilderRandomGraphTest, RandomChainExpansion) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  // Single Op: In -> Op -> Out
  auto* input_tensor = &MakeTensor(subgraph);
  auto* output_tensor = &MakeTensor(subgraph);
  subgraph.Inputs().push_back(input_tensor);
  subgraph.Outputs().push_back(output_tensor);

  auto* original_op =
      &MakeOp(subgraph, kLiteRtOpCodeTflAdd, {input_tensor}, {output_tensor});

  // Expand to Chain: In -> Op0 -> T0 -> Op1 -> Out
  // Length M
  std::uniform_int_distribution<int> length_dist(2, 5);
  int new_chain_length = length_dist(rng_);

  LiteRtBuilderT builder;
  BufferManager manager;

  LiteRtTensorT* current_input = input_tensor;
  std::vector<LiteRtOpT*> new_ops;

  for (int i = 0; i < new_chain_length; ++i) {
    LiteRtTensorT* current_output;
    if (i == new_chain_length - 1) {
      current_output = output_tensor;  // Final op writes to original output
    } else {
      LiteRtWeightsT weights(&manager);
      auto& new_tensor_ref = builder.BuildTensor(
          weights, MakePerTensorQuantization(1.0f, 0),
          MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2}),
          "chain_intermediate");
      current_output = &new_tensor_ref;
    }

    // We want to verify replacing chain with chain.
    // So we use BuildOp.
    auto& new_op_ref =
        builder.BuildOp(kLiteRtOpCodeTflSub, {current_input}, {current_output});
    new_ops.push_back(&new_op_ref);
    current_input = current_output;
  }

  builder.EraseOp(original_op);
  builder.ApplyChanges(&subgraph);

  // Verify
  EXPECT_EQ(subgraph.Ops().size(), new_chain_length);

  // Verify chain connectivity by traversing
  LiteRtOpT* current_op = input_tensor->Users()[0];
  int ops_traversed = 0;
  while (true) {
    ops_traversed++;
    // Check Op Code
    EXPECT_EQ(current_op->OpCode(), kLiteRtOpCodeTflSub);

    // Check Output
    ASSERT_EQ(current_op->Outputs().size(), 1);
    auto* out_tensor = current_op->Outputs()[0];

    if (out_tensor == output_tensor) {
      break;  // Reached end
    }

    // Move to next
    ASSERT_EQ(out_tensor->Users().size(), 1);
    current_op = out_tensor->Users()[0];
  }
  EXPECT_EQ(ops_traversed, new_chain_length);
}

TEST_P(BuilderRandomGraphTest, RandomBoundaryMutation) {
  // Test replacing ops at the very edge of the graph
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();

  int num_inputs = 3;
  int num_outputs = 3;

  std::vector<LiteRtTensorT*> inputs;
  std::vector<LiteRtTensorT*> outputs;

  for (int i = 0; i < num_inputs; ++i) {
    auto* t = &MakeTensor(subgraph);
    inputs.push_back(t);
    subgraph.Inputs().push_back(t);
  }
  for (int i = 0; i < num_outputs; ++i) {
    auto* t = &MakeTensor(subgraph);
    outputs.push_back(t);
    subgraph.Outputs().push_back(t);
  }

  // Create Ops connecting Inputs -> Outputs directly (bipartite)
  // Each input connected to random output via an Op
  std::vector<LiteRtOpT*> boundary_ops;
  for (int i = 0; i < num_inputs; ++i) {
    // Input[i] -> Op -> Output[i % num_outputs]
    auto* op = &MakeOp(subgraph, kLiteRtOpCodeTflAdd, {inputs[i]},
                       {outputs[i % num_outputs]});
    boundary_ops.push_back(op);
  }

  LiteRtBuilderT builder;
  // Randomly mutate these boundary ops
  for (auto* op : boundary_ops) {
    std::uniform_int_distribution<int> action_dist(0, 1);
    if (action_dist(rng_) == 0) {
      builder.BuildOp(kLiteRtOpCodeTflMul, op->Inputs(), op->Outputs());
      builder.EraseOp(op);
    } else {
      builder.BuildOp(kLiteRtOpCodeTflSub, op->Inputs(), op->Outputs());
      builder.EraseOp(op);
    }
  }

  builder.ApplyChanges(&subgraph);

  // Verify Inputs and Outputs are still the same tensor pointers
  EXPECT_EQ(subgraph.Inputs().size(), num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    EXPECT_EQ(subgraph.Inputs()[i], inputs[i]);
    // Verify input is used
    EXPECT_FALSE(inputs[i]->Users().empty());
  }

  EXPECT_EQ(subgraph.Outputs().size(), num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    EXPECT_EQ(subgraph.Outputs()[i], outputs[i]);
    // Verify output has definition
    EXPECT_NE(outputs[i]->DefiningOp(), nullptr);
  }
}

INSTANTIATE_TEST_SUITE_P(RandomGraphTests, BuilderRandomGraphTest,
                         ::testing::Range(0, 50));

}  // namespace
}  // namespace litert::internal
