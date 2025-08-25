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

#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model.h"
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

TEST(RewriterTest, InitializeRewriter) {
  LiteRtSubgraphT subgraph;
  subgraph.EmplaceOp();
  LiteRtRewriterT rewriter;

  EXPECT_EQ(subgraph.Ops().size(), 1);
}

TEST(RewriterTest, BuildRuntimeTensorWithoutWeights) {
  LiteRtSubgraphT subgraph;
  LiteRtRewriterT rewriter;
  auto weights = LiteRtWeightsT();
  const auto quant = MakePerTensorQuantization(kScale, kZero);
  const auto tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2});
  auto& tensor = rewriter.BuildTensor(weights, quant, tensor_type, kTensorName);

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

TEST(RewriterTest, BuildRuntimeTensorFromExistingTensor) {
  LiteRtSubgraphT subgraph;
  LiteRtRewriterT rewriter;
  auto tensor = BuildSimpleTensor();
  auto& built_tensor = rewriter.BuildTensor(tensor);
  auto built_tensor_type = built_tensor.Type();

  EXPECT_EQ(built_tensor_type.first, kLiteRtRankedTensorType);
  EXPECT_EQ(built_tensor_type.second.ranked_tensor_type.element_type,
            kLiteRtElementTypeInt32);
  EXPECT_EQ(built_tensor_type.second.ranked_tensor_type.layout.rank, 3);
  EXPECT_THAT(
      absl::MakeConstSpan(
          built_tensor_type.second.ranked_tensor_type.layout.dimensions, 3),
      ElementsAreArray({2, 2, 2}));
  EXPECT_EQ(rewriter.IsTensorAllocated(&built_tensor), true);
  EXPECT_EQ(rewriter.Subgraph().Tensors().size(), 1);
}

TEST(RewriterTest, BuildTensorWithExistingTensorWithWeights) {
  LiteRtSubgraphT subgraph;
  auto& tensor = subgraph.EmplaceTensor();
  {
    OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(tensor.Weights(), std::move(buf));
  }
  LiteRtRewriterT rewriter;
  auto& built_tensor = rewriter.BuildTensor(tensor);

  EXPECT_EQ(built_tensor.Weights().GetBufferId(), 1);
  EXPECT_EQ(built_tensor.Weights().GetBufferManager(),
            tensor.Weights().GetBufferManager());
  EXPECT_EQ(built_tensor.Weights().Buffer().Size(), kData.size());
}

TEST(RewriterTest, BuildTensorWithWeights) {
  LiteRtSubgraphT subgraph;
  BufferManager manager;
  LiteRtWeightsT weights;
  {
    weights.SetBufferManager(&manager);
    OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }
  LiteRtRewriterT rewriter;
  auto& built_tensor = rewriter.BuildTensor(
      weights, MakePerTensorQuantization(kScale, kZero),
      MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2, 2}), kTensorName);

  EXPECT_EQ(built_tensor.Weights().GetBufferId(), 1);
  EXPECT_EQ(built_tensor.Weights().GetBufferManager(), &manager);
  EXPECT_EQ(built_tensor.Weights().Buffer().Size(), kData.size());
}

TEST(RewriterTest, BuildOp) {
  LiteRtSubgraphT subgraph;
  LiteRtRewriterT rewriter;
  auto tensor = BuildSimpleTensor();
  auto& built_tensor_0 = rewriter.BuildTensor(tensor);
  auto& built_tensor_1 = rewriter.BuildTensor(tensor);
  auto& built_op = rewriter.BuildOp(kLiteRtOpCodeTflAdd, {&built_tensor_0},
                                    {&built_tensor_1});

  EXPECT_EQ(built_op.OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(built_op.Inputs().size(), 1);
  EXPECT_EQ(built_op.Outputs().size(), 1);
  EXPECT_EQ(built_tensor_0.GetUse(0).first, &built_op);
  EXPECT_EQ(built_tensor_0.GetUse(0).second, 0);
  EXPECT_EQ(built_tensor_1.DefiningOp(), &built_op);
}

TEST(RewriterTest, BuildOpFromExistingOp) {
  LiteRtSubgraphT subgraph;
  LiteRtRewriterT rewriter;
  auto tensor = BuildSimpleTensor();
  auto& built_tensor_0 = rewriter.BuildTensor(tensor);
  auto& built_tensor_1 = rewriter.BuildTensor(tensor);

  auto op = LiteRtOpT();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& built_op = rewriter.BuildOp(op, {&built_tensor_0}, {&built_tensor_1});

  EXPECT_EQ(built_op.OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(built_op.Inputs().size(), 1);
  EXPECT_EQ(built_op.Outputs().size(), 1);
  EXPECT_EQ(built_tensor_0.GetUse(0).first, &built_op);
  EXPECT_EQ(built_tensor_0.GetUse(0).second, 0);
  EXPECT_EQ(built_tensor_1.DefiningOp(), &built_op);
  EXPECT_EQ(rewriter.IsOpAllocated(&built_op), true);
  EXPECT_EQ(rewriter.Subgraph().Ops().size(), 1);
}

TEST(RewriterTest, EraseOp) {
  LiteRtSubgraphT subgraph;
  LiteRtRewriterT rewriter;
  auto& op_to_erase = subgraph.EmplaceOp();
  rewriter.EraseOp(&op_to_erase);
  EXPECT_EQ(rewriter.Erases().size(), 1);
  EXPECT_EQ(rewriter.Erases().contains(&op_to_erase), true);
}

TEST(RewriterTest, UncommittedTransformation) {
  LiteRtSubgraphT subgraph;
  auto& op_to_replace = subgraph.EmplaceOp();
  op_to_replace.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& tensor_0 = subgraph.EmplaceTensor();
  auto& tensor_1 = subgraph.EmplaceTensor();
  AttachInput(&tensor_0, op_to_replace);
  AttachOutput(&tensor_1, op_to_replace);

  LiteRtRewriterT rewriter;
  rewriter.BuildOp(kLiteRtOpCodeTflMul, {&tensor_0}, {&tensor_1});
  rewriter.EraseOp(&op_to_replace);

  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(&subgraph.Ops().front()->Input(0), &tensor_0);
  EXPECT_EQ(&subgraph.Ops().front()->Output(0), &tensor_1);
  EXPECT_EQ(subgraph.Ops().front()->Input(0).Users().front(), &op_to_replace);
  EXPECT_EQ(subgraph.Ops().front()->Output(0).DefiningOp(), &op_to_replace);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(subgraph.Tensors().size(), 2);
}

TEST(RewriterTest, AddOpToMulOpTransformation) {
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

  LiteRtRewriterT rewriter;
  rewriter.BuildOp(kLiteRtOpCodeTflMul, {&tensor_0}, {&tensor_1});
  rewriter.EraseOp(&op_to_replace);
  rewriter.ApplyChanges(&subgraph);

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

TEST(RewriterTest, AddOpToMulOpAndAddOpTransformation) {
  LiteRtSubgraphT subgraph;
  auto& op_to_replace = subgraph.EmplaceOp();
  op_to_replace.SetOpCode(kLiteRtOpCodeTflAdd);
  auto& tensor_0 = subgraph.EmplaceTensor();
  auto& tensor_1 = subgraph.EmplaceTensor();
  AttachInput(&tensor_0, op_to_replace);
  AttachOutput(&tensor_1, op_to_replace);

  LiteRtRewriterT rewriter;
  auto& built_tensor_0 = rewriter.BuildTensor(tensor_0);
  rewriter.BuildOp(kLiteRtOpCodeTflMul, {&tensor_0}, {&built_tensor_0});
  rewriter.BuildOp(kLiteRtOpCodeTflAdd, {&built_tensor_0}, {&tensor_1});
  rewriter.EraseOp(&op_to_replace);
  rewriter.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 2);
  EXPECT_EQ(subgraph.Ops().at(0)->OpCode(), kLiteRtOpCodeTflMul);
  EXPECT_EQ(subgraph.Ops().at(1)->OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(subgraph.Tensors().size(), 3);
}

TEST(RewriterTest, AddOpAndMulOpToDivOpTransformation) {
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

  LiteRtRewriterT rewriter;
  rewriter.BuildOp(kLiteRtOpCodeTflDiv, {&tensor_0}, {&tensor_2});
  rewriter.EraseOp(&add_op_to_replace);
  rewriter.EraseOp(&mul_op_to_replace);
  rewriter.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->Inputs().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->Outputs().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflDiv);
  EXPECT_EQ(subgraph.Tensors().size(), 2);
}
}  // namespace
}  // namespace litert::internal
