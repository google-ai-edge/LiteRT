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

}  // namespace
}  // namespace litert::internal
