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

#include "litert/cc/internal/litert_builder.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"

namespace litert {

template <typename T>
RankedTensorSpec MakeRankedTensorSpec(absl::Span<const int32_t> dims) {
  return RankedTensorSpecBuilder(
             RankedTensorType(
                 GetElementType<T>(),
                 Layout(BuildLayout(dims.data(), dims.data() + dims.size()))))
      .Build();
}

namespace {

static constexpr absl::string_view kTensorName = "M3";
static constexpr absl::string_view kData = "Nurburgring";
constexpr int32_t kTensorDimensions[] = {1, 2, 3};
constexpr LiteRtLayout kLayout = BuildLayout(kTensorDimensions);
constexpr LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    /*.layout=*/kLayout,
};

//===----------------------------------------------------------------------===//
//                               CC Builder                                  //
//===----------------------------------------------------------------------===//

TEST(CcRankedTensorSpecBuilderTest, TestBuild) {
  auto ranked_tensor_spec =
      RankedTensorSpecBuilder(RankedTensorType(kTensorType))
          .WithTensorName(std::string(kTensorName))
          .Build();
  EXPECT_EQ(ranked_tensor_spec.ranked_tensor_type.ElementType(),
            ElementType::Float32);
  EXPECT_EQ(ranked_tensor_spec.tensor_name, kTensorName);
}

TEST(CcBuilderTest, TestBuildUnrankedTensor) {
  LiteRtBuilderT builder;
  Builder cc_builder(&builder);
  auto tensor = cc_builder.BuildScalar(kLiteRtElementTypeFloat32,
                                       std::string(kTensorName));
  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->Name(), kTensorName);
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
}

TEST(CcBuilderTest, TestBuildRankedTensor) {
  LiteRtBuilderT builder;
  Builder cc_builder(&builder);
  RankedTensorType tensor_type(kTensorType);
  auto ranked_tensor_spec = RankedTensorSpecBuilder(tensor_type)
                                .WithTensorName(std::string(kTensorName))
                                .Build();
  auto tensor = cc_builder.BuildTensor(ranked_tensor_spec);

  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->Name(), kTensorName);
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
  auto built_tensor_type = tensor->RankedTensorType();
  EXPECT_EQ(built_tensor_type->ElementType(), ElementType::Float32);
  EXPECT_EQ(built_tensor_type->Layout().Rank(), 3);
  EXPECT_THAT(built_tensor_type->Layout().Dimensions(),
              ::testing::ElementsAreArray({1, 2, 3}));
}

TEST(CcBuilderTest, TestCloneTensor) {
  LiteRtBuilderT builder;
  Builder cc_builder(&builder);
  RankedTensorType tensor_type(kTensorType);
  auto ranked_tensor_spec = RankedTensorSpecBuilder(tensor_type)
                                .WithTensorName(std::string(kTensorName))
                                .Build();
  auto tensor = cc_builder.BuildTensor(ranked_tensor_spec);
  ASSERT_TRUE(tensor.HasValue());

  auto cloned_tensor = cc_builder.CloneTensor(*tensor);
  ASSERT_TRUE(cloned_tensor.HasValue());
  EXPECT_EQ(cloned_tensor->Name(), kTensorName);
  EXPECT_EQ(cloned_tensor->ElementType(), ElementType::Float32);
  auto cloned_tensor_type = cloned_tensor->RankedTensorType();
  ASSERT_TRUE(cloned_tensor_type.HasValue());
  EXPECT_EQ(cloned_tensor_type->Layout().Rank(), 3);
}

TEST(CcBuilderTest, TestCloneTensorWithQuantization) {
  LiteRtBuilderT builder;
  Builder cc_builder(&builder);
  RankedTensorType tensor_type(kTensorType);
  auto per_tensor_quantization = MakePerTensorQuantization(1.0, 1);
  auto ranked_tensor_spec =
      RankedTensorSpecBuilder(tensor_type)
          .WithPerTensorQuantization(per_tensor_quantization.second.per_tensor)
          .Build();
  auto tensor = cc_builder.BuildTensor(ranked_tensor_spec);
  ASSERT_TRUE(tensor.HasValue());

  auto cloned_tensor = cc_builder.CloneTensor(*tensor);
  ASSERT_TRUE(cloned_tensor.HasValue());
  EXPECT_EQ(cloned_tensor->ElementType(), ElementType::Float32);
  EXPECT_EQ(cloned_tensor->PerTensorQuantization().scale, 1.0);
  EXPECT_EQ(cloned_tensor->PerTensorQuantization().zero_point, 1);
}

TEST(CcBuilderTest, TestBuildRankedTensorWithWeights) {
  LiteRtBuilderT builder;
  Builder cc_builder(&builder);
  RankedTensorType tensor_type(kTensorType);

  litert::internal::BufferManager manager;
  LiteRtWeightsT weights;
  {
    weights.SetBufferManager(&manager);
    litert::OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }
  Weights cc_weights = Weights(&weights);
  auto ranked_tensor_spec = RankedTensorSpecBuilder(tensor_type)
                                .WithWeights(Weights(&weights))
                                .Build();
  auto tensor = cc_builder.BuildTensor(ranked_tensor_spec);
  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
  EXPECT_EQ(tensor->Weights().Get()->Buffer().StrView(), kData);
}

TEST(CcBuilderTest, TestBuildRankedTensorWithPerTensorQuantization) {
  LiteRtBuilderT builder;
  Builder cc_builder(&builder);
  RankedTensorType tensor_type(kTensorType);
  auto per_tensor_quantization = MakePerTensorQuantization(1.0, 1);
  auto ranked_tensor_spec =
      RankedTensorSpecBuilder(tensor_type)
          .WithPerTensorQuantization(per_tensor_quantization.second.per_tensor)
          .Build();
  auto tensor = cc_builder.BuildTensor(ranked_tensor_spec);
  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
  EXPECT_EQ(tensor->PerTensorQuantization().scale, 1.0);
  EXPECT_EQ(tensor->PerTensorQuantization().zero_point, 1);
}

TEST(CcBuilderTest, TestBuildRankedTensorWithPerChannelQuantization) {
  constexpr auto kNumChannels = 2;
  constexpr auto kQuantizedDimension = 0;
  constexpr float kScales[kNumChannels] = {1.0, 2.0};
  constexpr int64_t kZeroPoints[kNumChannels] = {0, 0};

  LiteRtBuilderT builder;
  Builder cc_builder(&builder);
  RankedTensorType tensor_type(kTensorType);
  LiteRtTensorT per_channel_quantized_tensor;
  auto per_channel_quantization = MakePerChannelQuantization(
      kScales, kZeroPoints, kQuantizedDimension, per_channel_quantized_tensor);
  auto ranked_tensor_spec = RankedTensorSpecBuilder(tensor_type)
                                .WithPerChannelQuantization(
                                    per_channel_quantization.second.per_channel)
                                .Build();
  auto tensor = cc_builder.BuildTensor(ranked_tensor_spec);
  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
  EXPECT_EQ(tensor->PerChannelQuantization().scales[0], 1.0);
  EXPECT_EQ(tensor->PerChannelQuantization().scales[1], 2.0);
  EXPECT_EQ(tensor->PerChannelQuantization().zero_points[0], 0);
  EXPECT_EQ(tensor->PerChannelQuantization().zero_points[1], 0);
  EXPECT_EQ(tensor->PerChannelQuantization().num_channels, 2);
  EXPECT_EQ(tensor->PerChannelQuantization().quantized_dimension, 0);
}

TEST(CcBuilderTest, TestBuildOp) {
  LiteRtBuilderT builder;
  Builder cc_builder(&builder);
  LiteRtTensorT litert_tensor_0;
  LiteRtTensorT litert_tensor_1;
  LiteRtTensorT litert_tensor_2;
  std::vector<Tensor> inputs;
  inputs.push_back(Tensor(&litert_tensor_0));
  inputs.push_back(Tensor(&litert_tensor_1));
  std::vector<Tensor> outputs;
  outputs.push_back(Tensor(&litert_tensor_2));
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflAdd, inputs, outputs);
  EXPECT_EQ(op.Inputs().size(), 2);
  EXPECT_EQ(op.Outputs().size(), 1);
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflAdd);
}

TEST(CcBuilderTest, TestBuildWeights) {
  const float kData[] = {1.0f, 2.0f, 3.0f};
  absl::Span<const float> data = absl::MakeConstSpan(kData);

  LiteRtBuilderT builder;
  Builder cc_builder(&builder);
  LiteRtTensorT litert_tensor_0;
  RankedTensorType tensor_type(kTensorType);

  auto tensor_spec = RankedTensorSpecBuilder(tensor_type).Build();
  auto tensor = cc_builder.BuildTensor(tensor_spec);
  auto weights = cc_builder.BuildWeights<float>(data, tensor.Value());

  ASSERT_TRUE(weights.HasValue());
  EXPECT_EQ(weights.Value().Get()->Buffer().Size(),
            data.size() * sizeof(float));
  const float* weights_data =
      reinterpret_cast<const float*>(weights.Value().Get()->Buffer().Data());
  for (int i = 0; i < data.size(); ++i) {
    EXPECT_EQ(weights_data[i], data[i]);
  }
}

TEST(CcBuilderTest, TestSetOpOptions) {
  LiteRtBuilderT builder;
  Builder cc_builder(&builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflAdd, inputs, outputs);
  {
    AddOptions add_options;
    add_options.fused_activation_function = kActivationFunctionTypeRelu;
    cc_builder.SetOpOptions<AddOptions>(op, std::move(add_options));
  }
  auto res = GetOptionsAs<AddOptions>(op.Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeRelu);
}

//===----------------------------------------------------------------------===//
//                       Builder Extended API Tests                          //
//===----------------------------------------------------------------------===//

class BuilderExtendedTest : public ::testing::Test {
 protected:
  LiteRtBuilderT builder_impl_;
  Builder builder_{&builder_impl_};
};

// Create Single Op (Add)
TEST_F(BuilderExtendedTest, CreateAddOp) {
  auto t1_res = builder_.BuildTensor(MakeRankedTensorSpec<float>({2}));
  auto t2_res = builder_.BuildTensor(MakeRankedTensorSpec<float>({2}));
  ASSERT_TRUE(t1_res.HasValue());
  ASSERT_TRUE(t2_res.HasValue());
  auto out_res =
      builder_.CreateOpWithOutputSpec(kLiteRtOpCodeTflAdd, {*t1_res, *t2_res},
                                      MakeRankedTensorSpec<float>({2}));
  ASSERT_TRUE(out_res.HasValue());
  auto out = std::move(*out_res);

  auto def_op = out.DefiningOp();
  ASSERT_TRUE(def_op.has_value());
  EXPECT_EQ(Op(def_op->op).Code(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(Op(def_op->op).Inputs().size(), 2);
}

// Create Multi-Output Op (Split)
TEST_F(BuilderExtendedTest, CreateSplitOp) {
  auto axis_res = builder_.BuildTensor(MakeRankedTensorSpec<int32_t>({1}));
  auto input_res = builder_.BuildTensor(MakeRankedTensorSpec<float>({4}));
  ASSERT_TRUE(axis_res.HasValue());
  ASSERT_TRUE(input_res.HasValue());

  std::vector<RankedTensorSpec> output_specs;
  output_specs.push_back(MakeRankedTensorSpec<float>({2}));
  output_specs.push_back(MakeRankedTensorSpec<float>({2}));

  auto outs_res = builder_.CreateOpWithOutputSpec(
      kLiteRtOpCodeTflSplit, {*axis_res, *input_res}, output_specs);
  ASSERT_TRUE(outs_res.HasValue());
  auto outs = std::move(*outs_res);

  ASSERT_EQ(outs.size(), 2);
  auto def0 = outs[0].DefiningOp();
  auto def1 = outs[1].DefiningOp();
  EXPECT_EQ(def0->op, def1->op);
  EXPECT_EQ(Op(def0->op).Code(), kLiteRtOpCodeTflSplit);
}

// ReplaceOp (Simple)
TEST_F(BuilderExtendedTest, ReplaceOp) {
  LiteRtSubgraphT subgraph;
  auto& input_tensor = subgraph.EmplaceTensor();
  auto& output_tensor = subgraph.EmplaceTensor();

  auto& existing_op = subgraph.EmplaceOp();
  existing_op.SetOpCode(kLiteRtOpCodeTflAdd);

  litert::internal::AttachInput(&input_tensor, existing_op);
  litert::internal::AttachOutput(&output_tensor, existing_op);

  LiteRtBuilderT buidler_typed;
  Builder b(&buidler_typed);

  Op op_to_replace(&existing_op);
  Tensor input(&input_tensor);

  auto new_op = b.ReplaceOp(op_to_replace, kLiteRtOpCodeTflNeg, {input});

  EXPECT_EQ(new_op.Code(), kLiteRtOpCodeTflNeg);
  EXPECT_EQ(new_op.Inputs().size(), 1);
  EXPECT_EQ(new_op.Inputs()[0].Get(), &input_tensor);
  EXPECT_EQ(new_op.Outputs().size(), 1);
  EXPECT_EQ(new_op.Outputs()[0].Get(), &output_tensor);
}

// EraseOp
TEST_F(BuilderExtendedTest, EraseOp) {
  auto in_res = builder_.BuildTensor(MakeRankedTensorSpec<float>({1}));
  ASSERT_TRUE(in_res.HasValue());
  auto out_res = builder_.CreateOpWithOutputSpec(
      kLiteRtOpCodeTflAbs, {*in_res}, MakeRankedTensorSpec<float>({1}));
  ASSERT_TRUE(out_res.HasValue());
  auto def = out_res->DefiningOp();
  auto op = Op(def->op);

  builder_.EraseOp(op);
}

// Variadic Input (Concat)
TEST_F(BuilderExtendedTest, CreateVariadicOp) {
  auto t1 = builder_.BuildTensor(MakeRankedTensorSpec<float>({1}));
  auto t2 = builder_.BuildTensor(MakeRankedTensorSpec<float>({1}));
  auto t3 = builder_.BuildTensor(MakeRankedTensorSpec<float>({1}));
  ASSERT_TRUE(t1.HasValue());
  ASSERT_TRUE(t2.HasValue());
  ASSERT_TRUE(t3.HasValue());

  auto out_res = builder_.CreateOpWithOutputSpec(
      kLiteRtOpCodeTflConcatenation, {*t1, *t2, *t3},
      MakeRankedTensorSpec<float>({3}));
  ASSERT_TRUE(out_res.HasValue());
  auto def = out_res->DefiningOp();
  EXPECT_EQ(Op(def->op).Inputs().size(), 3);
}

// Replace with different input count
TEST_F(BuilderExtendedTest, ReplaceOpDiffInputs) {
  LiteRtSubgraphT subgraph;
  auto& t1 = subgraph.EmplaceTensor();
  auto& t2 = subgraph.EmplaceTensor();
  auto& out = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  litert::internal::AttachInput(&t1, op);
  litert::internal::AttachInput(&t2, op);
  litert::internal::AttachOutput(&out, op);

  LiteRtBuilderT buidler_typed;
  Builder b(&buidler_typed);

  Op op_to_replace(&op);
  Tensor input(&t1);

  // New: Abs(t1) - 1 input
  auto new_op = b.ReplaceOp(op_to_replace, kLiteRtOpCodeTflAbs, {input});

  EXPECT_EQ(new_op.Code(), kLiteRtOpCodeTflAbs);
  EXPECT_EQ(new_op.Inputs().size(), 1);
  EXPECT_EQ(new_op.Outputs()[0].Get(), &out);
}

// Complex Topology: ResNet Block-ish
TEST_F(BuilderExtendedTest, CreateResNetBlock) {
  auto input_res =
      builder_.BuildTensor(MakeRankedTensorSpec<float>({1, 224, 224, 3}));
  auto axis_res = builder_.BuildTensor(MakeRankedTensorSpec<int32_t>({1}));
  ASSERT_TRUE(input_res.HasValue());
  ASSERT_TRUE(axis_res.HasValue());

  // Split
  std::vector<RankedTensorSpec> output_specs;
  output_specs.push_back(MakeRankedTensorSpec<float>({1, 224, 224, 3}));
  output_specs.push_back(MakeRankedTensorSpec<float>({1, 224, 224, 3}));
  auto splits_res = builder_.CreateOpWithOutputSpec(
      kLiteRtOpCodeTflSplit, {*axis_res, *input_res}, output_specs);
  ASSERT_TRUE(splits_res.HasValue());
  auto splits = std::move(*splits_res);

  // Path 1: Conv
  auto conv_out_res = builder_.CreateOpWithOutputSpec(
      kLiteRtOpCodeTflConv2d, {splits[0]},
      MakeRankedTensorSpec<float>({1, 224, 224, 3}));
  ASSERT_TRUE(conv_out_res.HasValue());

  // Join: Add(Conv, Path 2)
  auto add_out_res = builder_.CreateOpWithOutputSpec(
      kLiteRtOpCodeTflAdd, {*conv_out_res, splits[1]},
      MakeRankedTensorSpec<float>({1, 224, 224, 3}));
  ASSERT_TRUE(add_out_res.HasValue());

  auto def = add_out_res->DefiningOp();
  EXPECT_EQ(Op(def->op).Inputs().size(), 2);
}

// Empty Inputs
TEST_F(BuilderExtendedTest, CreateOpNoInputs) {
  // E.g. Custom op that generates data?
  auto out_res = builder_.CreateOpWithOutputSpec(
      kLiteRtOpCodeTflCustom, {}, MakeRankedTensorSpec<float>({1}));
  ASSERT_TRUE(out_res.HasValue());
  auto def = out_res->DefiningOp();
  EXPECT_EQ(Op(def->op).Inputs().size(), 0);
}

// Empty Outputs
TEST_F(BuilderExtendedTest, CreateOpNoOutputs) {
  auto in_res = builder_.BuildTensor(MakeRankedTensorSpec<float>({1}));
  ASSERT_TRUE(in_res.HasValue());
  auto outs_res = builder_.CreateOpWithOutputSpec(
      kLiteRtOpCodeTflCustom, {*in_res}, std::vector<RankedTensorSpec>{});
  ASSERT_TRUE(outs_res.HasValue());
  EXPECT_TRUE(outs_res->empty());
}

// Large Fan-In
TEST_F(BuilderExtendedTest, LargeFanIn) {
  std::vector<Tensor> inputs;
  for (int i = 0; i < 100; ++i) {
    auto t = builder_.BuildTensor(MakeRankedTensorSpec<float>({1}));
    ASSERT_TRUE(t.HasValue());
    inputs.push_back(std::move(*t));
  }
  auto out_res =
      builder_.CreateOpWithOutputSpec(kLiteRtOpCodeTflConcatenation, inputs,
                                      MakeRankedTensorSpec<float>({100}));
  ASSERT_TRUE(out_res.HasValue());
  auto def = out_res->DefiningOp();
  EXPECT_EQ(Op(def->op).Inputs().size(), 100);
}

// Dynamic Shapes (Wildcards)
TEST_F(BuilderExtendedTest, CreateDynamicShape) {
  auto t_res = builder_.BuildTensor(MakeRankedTensorSpec<float>({-1, 10}));
  ASSERT_TRUE(t_res.HasValue());
  auto type = t_res->RankedTensorType();
  ASSERT_TRUE(type.HasValue());
  EXPECT_EQ(type->Layout().Dimensions()[0], -1);
}

// ReplaceOp with more inputs (Neg -> Add)
TEST_F(BuilderExtendedTest, ReplaceNegWithAdd) {
  LiteRtSubgraphT subgraph;
  auto& t1 = subgraph.EmplaceTensor();
  auto& out = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflNeg);
  litert::internal::AttachInput(&t1, op);
  litert::internal::AttachOutput(&out, op);

  LiteRtBuilderT buidler_typed;
  Builder b(&buidler_typed);

  Op op_to_replace(&op);
  Tensor input(&t1);

  // New: Add(t1, t1)
  auto new_op = b.ReplaceOp(op_to_replace, kLiteRtOpCodeTflAdd, {input, input});

  EXPECT_EQ(new_op.Code(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(new_op.Inputs().size(), 2);
  EXPECT_EQ(new_op.Outputs()[0].Get(), &out);
}

// ReplaceOp preserving outputs (Multi-output)
TEST_F(BuilderExtendedTest, ReplaceSplitWithSplit) {
  LiteRtSubgraphT subgraph;
  auto& t = subgraph.EmplaceTensor();
  auto& axis = subgraph.EmplaceTensor();
  auto& out1 = subgraph.EmplaceTensor();
  auto& out2 = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSplit);
  litert::internal::AttachInput(&axis, op);
  litert::internal::AttachInput(&t, op);
  litert::internal::AttachOutput(&out1, op);
  litert::internal::AttachOutput(&out2, op);

  LiteRtBuilderT buidler_typed;
  Builder b(&buidler_typed);

  Op op_to_replace(&op);
  Tensor input_t(&t);
  Tensor input_axis(&axis);  // Or use a new axis tensor

  // Create a new axis tensor for replacement test
  auto axis2_res = b.BuildTensor(MakeRankedTensorSpec<int32_t>({1}));
  ASSERT_TRUE(axis2_res.HasValue());

  // Replace with same op but different inputs
  auto new_op =
      b.ReplaceOp(op_to_replace, kLiteRtOpCodeTflSplit, {*axis2_res, input_t});

  EXPECT_EQ(new_op.Code(), kLiteRtOpCodeTflSplit);
  EXPECT_EQ(new_op.Inputs()[0].Get(), axis2_res->Get());
  EXPECT_EQ(new_op.Outputs()[0].Get(), &out1);
}

// Mixed Types
TEST_F(BuilderExtendedTest, MixedTypes) {
  auto f = builder_.BuildTensor(MakeRankedTensorSpec<float>({1}));
  auto i = builder_.BuildTensor(MakeRankedTensorSpec<int32_t>({1}));
  ASSERT_TRUE(f.HasValue());
  ASSERT_TRUE(i.HasValue());
  // Cast Op
  auto cast_res = builder_.CreateOpWithOutputSpec(
      kLiteRtOpCodeTflCast, {*f}, MakeRankedTensorSpec<int32_t>({1}));
  ASSERT_TRUE(cast_res.HasValue());
  auto type = cast_res->RankedTensorType();
  ASSERT_TRUE(type.HasValue());
  EXPECT_EQ(type->ElementType(),
            static_cast<ElementType>(kLiteRtElementTypeInt32));
}

// Quantized Types (Manual construction)
TEST_F(BuilderExtendedTest, QuantizedTypeConstruction) {
  std::vector<int> dims = {1};
  RankedTensorSpec qtype =
      RankedTensorSpecBuilder(
          RankedTensorType(
              static_cast<ElementType>(kLiteRtElementTypeInt8),
              Layout(BuildLayout(dims.data(), dims.data() + dims.size()))))
          .Build();
  auto t_res = builder_.BuildTensor(qtype);
  ASSERT_TRUE(t_res.HasValue());
  auto type = t_res->RankedTensorType();
  ASSERT_TRUE(type.HasValue());
  EXPECT_EQ(type->ElementType(),
            static_cast<ElementType>(kLiteRtElementTypeInt8));
}

// Integration: Matcher Pattern
TEST_F(BuilderExtendedTest, PatternMatchReplace) {
  LiteRtSubgraphT subgraph;
  auto& x = subgraph.EmplaceTensor();
  auto& out = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  litert::internal::AttachInput(&x, op);
  litert::internal::AttachInput(&x, op);
  litert::internal::AttachOutput(&out, op);

  LiteRtBuilderT buidler_typed;
  Builder b(&buidler_typed);

  Op op_to_replace(&op);
  Tensor input_x(&x);

  // Pattern: Add(a, a) -> Mul(a, 2)
  if (op_to_replace.Code() == kLiteRtOpCodeTflAdd &&
      op_to_replace.Inputs()[0].Get() == op_to_replace.Inputs()[1].Get()) {
    auto two = b.BuildTensor(MakeRankedTensorSpec<float>({1}));
    ASSERT_TRUE(two.HasValue());
    auto new_op =
        b.ReplaceOp(op_to_replace, kLiteRtOpCodeTflMul, {input_x, *two});
    EXPECT_EQ(new_op.Code(), kLiteRtOpCodeTflMul);
    EXPECT_EQ(new_op.Outputs()[0].Get(), &out);
  } else {
    FAIL() << "Pattern not matched";
  }
}

}  // namespace
}  // namespace litert
