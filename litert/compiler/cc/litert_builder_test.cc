// Copyright 2026 Google LLC.
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

#include "litert/compiler/cc/litert_builder.h"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_tfl_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"

namespace litert::compiler {

template <typename T>
RankedTensorSpec MakeRankedTensorSpec(absl::Span<const int32_t> dims) {
  return RankedTensorSpecBuilder(
             RankedTensorType(
                 GetElementType<T>(),
                 Layout(BuildLayout(dims.data(), dims.data() + dims.size()))))
      .Build();
}

template <typename T>
RankedTensorSpec MakeRankedTensorSpec(std::initializer_list<int32_t> dims) {
  return MakeRankedTensorSpec<T>(
      absl::Span<const int32_t>(dims.begin(), dims.size()));
}

namespace {

using ::MakePerChannelQuantization;
using ::MakePerTensorQuantization;
using ::SetWeightsFromOwnedBuffer;
using ::litert::ElementType;
using ::litert::Layout;
using ::litert::OwningBufferRef;
using ::litert::RankedTensorType;
using ::litert::internal::AttachInput;
using ::litert::internal::AttachOutput;

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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  auto tensor = cc_builder.BuildScalar(kLiteRtElementTypeFloat32,
                                       std::string(kTensorName));
  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->Name(), kTensorName);
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
}

TEST(CcBuilderTest, TestBuildRankedTensor) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  RankedTensorType tensor_type(kTensorType);

  ::litert::internal::BufferManager manager;
  LiteRtWeightsT weights;
  {
    weights.SetBufferManager(&manager);
    OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }
  Weights cc_weights(ctx, &weights);
  auto ranked_tensor_spec = RankedTensorSpecBuilder(tensor_type)
                                .WithWeights(Weights(ctx, &weights))
                                .Build();
  auto tensor = cc_builder.BuildTensor(ranked_tensor_spec);
  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
  EXPECT_EQ(tensor->Weights().StrView(), kData);
}

TEST(CcBuilderTest, TestBuildRankedTensorWithPerTensorQuantization) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  constexpr auto kNumChannels = 2;
  constexpr auto kQuantizedDimension = 0;
  constexpr float kScales[kNumChannels] = {1.0, 2.0};
  constexpr int64_t kZeroPoints[kNumChannels] = {0, 0};

  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
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
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  LiteRtTensorT litert_tensor_0;
  LiteRtTensorT litert_tensor_1;
  LiteRtTensorT litert_tensor_2;
  std::vector<Tensor> inputs;
  inputs.push_back(Tensor(ctx, &litert_tensor_0));
  inputs.push_back(Tensor(ctx, &litert_tensor_1));
  std::vector<Tensor> outputs;
  outputs.push_back(Tensor(ctx, &litert_tensor_2));
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflAdd, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  EXPECT_EQ(op->Inputs().size(), 2);
  EXPECT_EQ(op->Outputs().size(), 1);
  EXPECT_EQ(op->Code(), kLiteRtOpCodeTflAdd);
}

TEST(CcBuilderTest, TestBuildWeights) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  const float kData[] = {1.0f, 2.0f, 3.0f};
  absl::Span<const float> data = absl::MakeConstSpan(kData);

  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  RankedTensorType tensor_type(kTensorType);

  auto tensor_spec = RankedTensorSpecBuilder(tensor_type).Build();
  auto tensor = cc_builder.BuildTensor(tensor_spec);
  auto weights = cc_builder.BuildWeights<float>(data, *tensor);

  ASSERT_TRUE(weights.HasValue());
  EXPECT_EQ(weights->Bytes().size(), data.size() * sizeof(float));
  const float* weights_data =
      reinterpret_cast<const float*>(weights->Bytes().data());
  for (int i = 0; i < data.size(); ++i) {
    EXPECT_EQ(weights_data[i], data[i]);
  }
}

TEST(CcBuilderTest, TestSetOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflAdd, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    AddOptions add_options;
    add_options.fused_activation_function = kActivationFunctionTypeRelu;
    cc_builder.SetOpOptions<AddOptions>(*op, std::move(add_options));
  }
  auto res = GetOptionsAs<AddOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeRelu);
}

TEST(CcBuilderTest, TestSetCompositeOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeShloComposite, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    CompositeOptions options;
    options.name = "odml.rms_norm";
    options.subgraph = 2;
    options.version = 1;
    auto res =
        cc_builder.SetOpOptions<CompositeOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<CompositeOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().name, "odml.rms_norm");
  EXPECT_EQ(res.Value().subgraph, 2);
  EXPECT_EQ(res.Value().version, 1);
}

TEST(CcBuilderTest, TestSetRmsNormOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeShloComposite, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    RmsNormOpts options;
    options.subgraph = 2;
    options.version = 1;
    options.epsilon = 1e-4f;
    auto res = cc_builder.SetOpOptions<RmsNormOpts>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<RmsNormOpts>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().name, CompositeOptions::kRmsNorm);
  EXPECT_EQ(res.Value().subgraph, 2);
  EXPECT_EQ(res.Value().version, 1);
  EXPECT_NEAR(res.Value().epsilon, 1e-4f, 1e-6f);
}

TEST(CcBuilderTest, TestSetGatherOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflGather, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    GatherOptions options;
    options.axis = 1;
    options.batch_dims = 0;
    auto res = cc_builder.SetOpOptions<GatherOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<GatherOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().axis, 1);
  EXPECT_EQ(res.Value().batch_dims, 0);
}

TEST(CcBuilderTest, TestSetReduceMaxOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflReduceMax, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    ReduceMaxOptions options;
    options.keep_dims = true;
    auto res =
        cc_builder.SetOpOptions<ReduceMaxOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<ReduceMaxOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().keep_dims, true);
}

TEST(CcBuilderTest, TestSetMulOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflMul, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    MulOptions options;
    options.fused_activation_function = kActivationFunctionTypeRelu;
    auto res = cc_builder.SetOpOptions<MulOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<MulOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeRelu);
}

TEST(CcBuilderTest, TestSetConcatenationOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflConcatenation, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    ConcatenationOptions options;
    options.axis = 2;
    options.fused_activation_function = kActivationFunctionTypeNone;
    auto res =
        cc_builder.SetOpOptions<ConcatenationOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<ConcatenationOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().axis, 2);
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeNone);
}

TEST(CcBuilderTest, TestSetReshapeOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflReshape, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    ReshapeOptions options;
    options.new_shape = {1, 2, 3};
    auto res = cc_builder.SetOpOptions<ReshapeOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<ReshapeOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().new_shape, (std::vector<int32_t>{1, 2, 3}));
}

TEST(CcBuilderTest, TestSetOneHotOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflOneHot, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    OneHotOptions options;
    options.axis = 3;
    auto res = cc_builder.SetOpOptions<OneHotOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<OneHotOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().axis, 3);
}

TEST(CcBuilderTest, TestSetSplitVOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflSplitV, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    SplitVOptions options;
    options.num_splits = 4;
    auto res = cc_builder.SetOpOptions<SplitVOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<SplitVOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().num_splits, 4);
}

TEST(CcBuilderTest, TestSetFullyConnectedOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflFullyConnected, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    FullyConnectedOptions options;
    options.fused_activation_function = kActivationFunctionTypeRelu6;
    options.weights_format = kFullyConnectedOptionsWeightsFormatDefault;
    options.keep_num_dims = true;
    options.quantized_bias_type = kLiteRtElementTypeInt32;
    options.asymmetric_quantize_input = false;
    auto res =
        cc_builder.SetOpOptions<FullyConnectedOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<FullyConnectedOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().fused_activation_function,
            kActivationFunctionTypeRelu6);
  EXPECT_EQ(res.Value().weights_format,
            kFullyConnectedOptionsWeightsFormatDefault);
  EXPECT_EQ(res.Value().keep_num_dims, true);
  EXPECT_EQ(res.Value().quantized_bias_type, kLiteRtElementTypeInt32);
  EXPECT_EQ(res.Value().asymmetric_quantize_input, false);
}

TEST(CcBuilderTest, TestSetStridedSliceOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflStridedSlice, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    StridedSliceOptions options;
    options.begin_mask = 1;
    options.end_mask = 2;
    options.ellipsis_mask = 0;
    options.new_axis_mask = 0;
    options.shrink_axis_mask = 4;
    options.offset = false;
    auto res =
        cc_builder.SetOpOptions<StridedSliceOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<StridedSliceOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().begin_mask, 1);
  EXPECT_EQ(res.Value().end_mask, 2);
  EXPECT_EQ(res.Value().shrink_axis_mask, 4);
  EXPECT_EQ(res.Value().offset, false);
}

TEST(CcBuilderTest, TestSetBatchMatmulOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflBatchMatmul, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    BatchMatmulOptions options;
    options.adj_x = true;
    options.adj_y = false;
    options.asymmetric_quantize_input = true;
    auto res =
        cc_builder.SetOpOptions<BatchMatmulOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<BatchMatmulOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().adj_x, true);
  EXPECT_EQ(res.Value().adj_y, false);
  EXPECT_EQ(res.Value().asymmetric_quantize_input, true);
}

TEST(CcBuilderTest, TestSetDivOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflDiv, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    DivOptions options;
    options.fused_activation_function = kActivationFunctionTypeRelu;
    auto res = cc_builder.SetOpOptions<DivOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<DivOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeRelu);
}

TEST(CcBuilderTest, TestSetSoftmaxOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflSoftmax, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    SoftmaxOptions options;
    options.beta = 2.0f;
    auto res = cc_builder.SetOpOptions<SoftmaxOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<SoftmaxOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_FLOAT_EQ(res.Value().beta, 2.0f);
}

TEST(CcBuilderTest, TestSetSubOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflSub, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    SubOptions options;
    options.fused_activation_function = kActivationFunctionTypeRelu6;
    auto res = cc_builder.SetOpOptions<SubOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<SubOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().fused_activation_function,
            kActivationFunctionTypeRelu6);
}

TEST(CcBuilderTest, TestSetSumOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflSum, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    SumOptions options;
    options.keep_dims = true;
    auto res = cc_builder.SetOpOptions<SumOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<SumOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().keep_dims, true);
}

TEST(CcBuilderTest, TestSetReduceMinOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflReduceMin, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    ReduceMinOptions options;
    options.keep_dims = true;
    auto res =
        cc_builder.SetOpOptions<ReduceMinOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<ReduceMinOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().keep_dims, true);
}

TEST(CcBuilderTest, TestSetReduceAnyOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflReduceAny, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    ReduceAnyOptions options;
    options.keep_dims = false;
    auto res =
        cc_builder.SetOpOptions<ReduceAnyOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<ReduceAnyOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().keep_dims, false);
}

TEST(CcBuilderTest, TestSetReduceAllOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflReduceAll, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    ReduceAllOptions options;
    options.keep_dims = true;
    auto res =
        cc_builder.SetOpOptions<ReduceAllOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<ReduceAllOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().keep_dims, true);
}

TEST(CcBuilderTest, TestSetPackOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  auto in1 = cc_builder.BuildScalar(kLiteRtElementTypeFloat32);
  auto in2 = cc_builder.BuildScalar(kLiteRtElementTypeFloat32);
  ASSERT_TRUE(in1.HasValue() && in2.HasValue());
  std::vector<Tensor> inputs = {*in1, *in2};
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflPack, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    PackOptions options;
    options.axis = 1;
    auto res = cc_builder.SetOpOptions<PackOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<PackOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().axis, 1);
}

TEST(CcBuilderTest, TestSetUnpackOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflUnpack, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    UnpackOptions options;
    options.axis = 0;
    options.num = 2;
    auto res = cc_builder.SetOpOptions<UnpackOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<UnpackOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().axis, 0);
  EXPECT_EQ(res.Value().num, 2);
}

TEST(CcBuilderTest, TestSetMeanOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflMean, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    MeanOptions options;
    options.keep_dims = true;
    auto res = cc_builder.SetOpOptions<MeanOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<MeanOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().keep_dims, true);
}

TEST(CcBuilderTest, TestSetSplitOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflSplit, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    SplitOptions options;
    options.num_splits = 3;
    auto res = cc_builder.SetOpOptions<SplitOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<SplitOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().num_splits, 3);
}

TEST(CcBuilderTest, TestSetConv2dOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflConv2d, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    Conv2dOptions options;
    options.padding = kPaddingSame;
    options.stride_w = 2;
    options.stride_h = 2;
    options.dilation_w_factor = 1;
    options.dilation_h_factor = 1;
    options.fused_activation_function = kActivationFunctionTypeRelu;
    auto res = cc_builder.SetOpOptions<Conv2dOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<Conv2dOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().padding, kPaddingSame);
  EXPECT_EQ(res.Value().stride_w, 2);
  EXPECT_EQ(res.Value().stride_h, 2);
  EXPECT_EQ(res.Value().dilation_w_factor, 1);
  EXPECT_EQ(res.Value().dilation_h_factor, 1);
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeRelu);
}

TEST(CcBuilderTest, TestSetConv3dOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflConv3d, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    Conv3dOptions options;
    options.padding = kPaddingValid;
    options.stride_w = 1;
    options.stride_h = 1;
    options.stride_d = 1;
    options.dilation_w_factor = 1;
    options.dilation_h_factor = 1;
    options.dilation_d_factor = 1;
    options.fused_activation_function = kActivationFunctionTypeNone;
    auto res = cc_builder.SetOpOptions<Conv3dOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<Conv3dOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().padding, kPaddingValid);
  EXPECT_EQ(res.Value().stride_w, 1);
  EXPECT_EQ(res.Value().stride_h, 1);
  EXPECT_EQ(res.Value().stride_d, 1);
  EXPECT_EQ(res.Value().dilation_w_factor, 1);
  EXPECT_EQ(res.Value().dilation_h_factor, 1);
  EXPECT_EQ(res.Value().dilation_d_factor, 1);
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeNone);
}

TEST(CcBuilderTest, TestSetDepthwiseConv2dOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op =
      cc_builder.BuildOp(kLiteRtOpCodeTflDepthwiseConv2d, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    DepthwiseConv2dOptions options;
    options.padding = kPaddingSame;
    options.stride_w = 1;
    options.stride_h = 1;
    options.depth_multiplier = 1;
    options.fused_activation_function = kActivationFunctionTypeRelu;
    options.dilation_w_factor = 1;
    options.dilation_h_factor = 1;
    auto res = cc_builder.SetOpOptions<DepthwiseConv2dOptions>(
        *op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<DepthwiseConv2dOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().padding, kPaddingSame);
  EXPECT_EQ(res.Value().depth_multiplier, 1);
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeRelu);
}

TEST(CcBuilderTest, TestSetTransposeConvOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflTransposeConv, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    TransposeConvOptions options;
    options.padding = kPaddingValid;
    options.stride_w = 2;
    options.stride_h = 2;
    options.fused_activation_function = kActivationFunctionTypeNone;
    auto res =
        cc_builder.SetOpOptions<TransposeConvOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<TransposeConvOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().padding, kPaddingValid);
  EXPECT_EQ(res.Value().stride_w, 2);
  EXPECT_EQ(res.Value().stride_h, 2);
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeNone);
}

TEST(CcBuilderTest, TestSetAveragePool2dOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflAveragePool2d, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    AveragePool2dOptions options;
    options.padding = kPaddingSame;
    options.stride_w = 1;
    options.stride_h = 1;
    options.filter_width = 3;
    options.filter_height = 3;
    options.fused_activation_function = kActivationFunctionTypeRelu;
    auto res =
        cc_builder.SetOpOptions<AveragePool2dOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<AveragePool2dOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().padding, kPaddingSame);
  EXPECT_EQ(res.Value().filter_width, 3);
  EXPECT_EQ(res.Value().filter_height, 3);
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeRelu);
}

TEST(CcBuilderTest, TestSetMaxPool2dOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflMaxPool2d, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    MaxPool2dOptions options;
    options.padding = kPaddingValid;
    options.stride_w = 2;
    options.stride_h = 2;
    options.filter_width = 2;
    options.filter_height = 2;
    options.fused_activation_function = kActivationFunctionTypeNone;
    auto res =
        cc_builder.SetOpOptions<MaxPool2dOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<MaxPool2dOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().padding, kPaddingValid);
  EXPECT_EQ(res.Value().filter_width, 2);
  EXPECT_EQ(res.Value().filter_height, 2);
  EXPECT_EQ(res.Value().fused_activation_function, kActivationFunctionTypeNone);
}

TEST(CcBuilderTest, TestSetL2Pool2dOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflL2Pool2d, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    L2Pool2dOptions options;
    options.padding = kPaddingSame;
    options.stride_w = 1;
    options.stride_h = 1;
    options.filter_width = 2;
    options.filter_height = 2;
    options.fused_activation_function = kActivationFunctionTypeNone;
    auto res =
        cc_builder.SetOpOptions<L2Pool2dOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<L2Pool2dOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().padding, kPaddingSame);
  EXPECT_EQ(res.Value().filter_width, 2);
  EXPECT_EQ(res.Value().filter_height, 2);
}

TEST(CcBuilderTest, TestSetResizeBilinearOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflResizeBilinear, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    ResizeBilinearOptions options;
    options.align_corners = true;
    options.half_pixel_centers = false;
    auto res =
        cc_builder.SetOpOptions<ResizeBilinearOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<ResizeBilinearOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().align_corners, true);
  EXPECT_EQ(res.Value().half_pixel_centers, false);
}

TEST(CcBuilderTest, TestSetLeakyReluOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflLeakyRelu, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    LeakyReluOptions options;
    options.alpha = 0.2f;
    auto res =
        cc_builder.SetOpOptions<LeakyReluOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<LeakyReluOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_FLOAT_EQ(res.Value().alpha, 0.2f);
}

TEST(CcBuilderTest, TestSetSpaceToDepthOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflSpaceToDepth, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    SpaceToDepthOptions options;
    options.block_size = 2;
    auto res =
        cc_builder.SetOpOptions<SpaceToDepthOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<SpaceToDepthOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().block_size, 2);
}

TEST(CcBuilderTest, TestSetDepthToSpaceOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflDepthToSpace, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    DepthToSpaceOptions options;
    options.block_size = 2;
    auto res =
        cc_builder.SetOpOptions<DepthToSpaceOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<DepthToSpaceOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().block_size, 2);
}

TEST(CcBuilderTest, TestSetResizeNearestNeighborOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflResizeNearestNeighbor, inputs,
                               outputs);
  ASSERT_TRUE(op.HasValue());
  {
    ResizeNearestNeighborOptions options;
    options.align_corners = false;
    options.half_pixel_centers = true;
    auto res = cc_builder.SetOpOptions<ResizeNearestNeighborOptions>(
        *op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res =
      GetOptionsAs<ResizeNearestNeighborOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().align_corners, false);
  EXPECT_EQ(res.Value().half_pixel_centers, true);
}

TEST(CcBuilderTest, TestSetCumSumOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflCumsum, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    CumSumOptions options;
    options.exclusive = true;
    options.reverse = false;
    auto res = cc_builder.SetOpOptions<CumSumOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<CumSumOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().exclusive, true);
  EXPECT_EQ(res.Value().reverse, false);
}

TEST(CcBuilderTest, TestSetGeluOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflGelu, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    GeluOptions options;
    options.approximate = true;
    auto res = cc_builder.SetOpOptions<GeluOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<GeluOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().approximate, true);
}

TEST(CcBuilderTest, TestSetMirrorPadOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflMirrorPad, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    MirrorPadOptions options;
    options.mode = kMirrorPadModeReflect;
    auto res =
        cc_builder.SetOpOptions<MirrorPadOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<MirrorPadOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_EQ(res.Value().mode, kMirrorPadModeReflect);
}

TEST(CcBuilderTest, TestSetSqueezeOpOptions) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  auto op = cc_builder.BuildOp(kLiteRtOpCodeTflSqueeze, inputs, outputs);
  ASSERT_TRUE(op.HasValue());
  {
    SqueezeOptions options;
    options.squeeze_dims = {1, 2};
    auto res = cc_builder.SetOpOptions<SqueezeOptions>(*op, std::move(options));
    ASSERT_TRUE(res.HasValue());
  }
  auto res = GetOptionsAs<SqueezeOptions>(op->Context(), op->Get());
  ASSERT_TRUE(res.HasValue());
  EXPECT_THAT(res.Value().squeeze_dims, ::testing::ElementsAreArray({1, 2}));
}

//===----------------------------------------------------------------------===//
//                       Builder Extended API Tests                          //
//===----------------------------------------------------------------------===//

class BuilderExtendedTest : public ::testing::Test {
 protected:
  const LiteRtCompilerContext* ctx_{LrtGetCompilerContext()};
  LiteRtBuilderT builder_impl_;
  Builder builder_{ctx_, &builder_impl_};
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
  EXPECT_EQ(Op(ctx_, def_op->op).Code(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(Op(ctx_, def_op->op).Inputs().size(), 2);
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
  EXPECT_EQ(Op(ctx_, def0->op).Code(), kLiteRtOpCodeTflSplit);
}

// ReplaceOp (Simple)
TEST_F(BuilderExtendedTest, ReplaceOp) {
  LiteRtSubgraphT subgraph;
  auto& input_tensor = subgraph.EmplaceTensor();
  auto& output_tensor = subgraph.EmplaceTensor();

  auto& existing_op = subgraph.EmplaceOp();
  existing_op.SetOpCode(kLiteRtOpCodeTflAdd);

  AttachInput(&input_tensor, existing_op);
  AttachOutput(&output_tensor, existing_op);

  LiteRtBuilderT buidler_typed;
  Builder b(ctx_, &buidler_typed);

  Op op_to_replace(ctx_, &existing_op);
  Tensor input(ctx_, &input_tensor);

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
  auto op = Op(ctx_, def->op);

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
  EXPECT_EQ(Op(ctx_, def->op).Inputs().size(), 3);
}

// Replace with different input count
TEST_F(BuilderExtendedTest, ReplaceOpDiffInputs) {
  LiteRtSubgraphT subgraph;
  auto& t1 = subgraph.EmplaceTensor();
  auto& t2 = subgraph.EmplaceTensor();
  auto& out = subgraph.EmplaceTensor();
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&t1, op);
  AttachInput(&t2, op);
  AttachOutput(&out, op);

  LiteRtBuilderT buidler_typed;
  Builder b(ctx_, &buidler_typed);

  Op op_to_replace(ctx_, &op);
  Tensor input(ctx_, &t1);

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
  EXPECT_EQ(Op(ctx_, def->op).Inputs().size(), 2);
}

// Empty Inputs
TEST_F(BuilderExtendedTest, CreateOpNoInputs) {
  // E.g. Custom op that generates data?
  auto out_res = builder_.CreateOpWithOutputSpec(
      kLiteRtOpCodeTflCustom, {}, MakeRankedTensorSpec<float>({1}));
  ASSERT_TRUE(out_res.HasValue());
  auto def = out_res->DefiningOp();
  EXPECT_EQ(Op(ctx_, def->op).Inputs().size(), 0);
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
  EXPECT_EQ(Op(ctx_, def->op).Inputs().size(), 100);
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
  AttachInput(&t1, op);
  AttachOutput(&out, op);

  LiteRtBuilderT buidler_typed;
  Builder b(ctx_, &buidler_typed);

  Op op_to_replace(ctx_, &op);
  Tensor input(ctx_, &t1);

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
  AttachInput(&axis, op);
  AttachInput(&t, op);
  AttachOutput(&out1, op);
  AttachOutput(&out2, op);

  LiteRtBuilderT buidler_typed;
  Builder b(ctx_, &buidler_typed);

  Op op_to_replace(ctx_, &op);
  Tensor input_t(ctx_, &t);

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
  AttachInput(&x, op);
  AttachInput(&x, op);
  AttachOutput(&out, op);

  LiteRtBuilderT buidler_typed;
  Builder b(ctx_, &buidler_typed);

  Op op_to_replace(ctx_, &op);
  Tensor input_x(ctx_, &x);

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
}  // namespace litert::compiler
