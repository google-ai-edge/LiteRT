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

#include "litert/cc/internal/litert_rewriter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_layout.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"

namespace litert {

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
//                               CC Rewriter                                  //
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

TEST(CcRewriterTest, TestBuildUnrankedTensor) {
  LiteRtRewriterT rewriter;
  Rewriter cc_rewriter(&rewriter);
  auto tensor = cc_rewriter.BuildScalar(kLiteRtElementTypeFloat32,
                                        std::string(kTensorName));
  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->Name(), kTensorName);
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
}

TEST(CcRewriterTest, TestBuildRankedTensor) {
  LiteRtRewriterT rewriter;
  Rewriter cc_rewriter(&rewriter);
  RankedTensorType tensor_type(kTensorType);
  auto ranked_tensor_spec = RankedTensorSpecBuilder(tensor_type)
                                .WithTensorName(std::string(kTensorName))
                                .Build();
  auto tensor = cc_rewriter.BuildTensor(ranked_tensor_spec);

  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->Name(), kTensorName);
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
  auto built_tensor_type = tensor->RankedTensorType();
  EXPECT_EQ(built_tensor_type->ElementType(), ElementType::Float32);
  EXPECT_EQ(built_tensor_type->Layout().Rank(), 3);
  EXPECT_THAT(built_tensor_type->Layout().Dimensions(),
              ::testing::ElementsAreArray({1, 2, 3}));
}

TEST(CcRewriterTest, TestBuildRankedTensorWithWeights) {
  LiteRtRewriterT rewriter;
  Rewriter cc_rewriter(&rewriter);
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
  auto tensor = cc_rewriter.BuildTensor(ranked_tensor_spec);
  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
  EXPECT_EQ(tensor->Weights().Get()->Buffer().StrView(), kData);
}

TEST(CcRewriterTest, TestBuildRankedTensorWithPerTensorQuantization) {
  LiteRtRewriterT rewriter;
  Rewriter cc_rewriter(&rewriter);
  RankedTensorType tensor_type(kTensorType);
  auto per_tensor_quantization = MakePerTensorQuantization(1.0, 1);
  auto ranked_tensor_spec =
      RankedTensorSpecBuilder(tensor_type)
          .WithPerTensorQuantization(per_tensor_quantization.second.per_tensor)
          .Build();
  auto tensor = cc_rewriter.BuildTensor(ranked_tensor_spec);
  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
  EXPECT_EQ(tensor->PerTensorQuantization().scale, 1.0);
  EXPECT_EQ(tensor->PerTensorQuantization().zero_point, 1);
}

TEST(CcRewriterTest, TestBuildRankedTensorWithPerChannelQuantization) {
  constexpr auto kNumChannels = 2;
  constexpr auto kQuantizedDimension = 0;
  constexpr float kScales[kNumChannels] = {1.0, 2.0};
  constexpr int64_t kZeroPoints[kNumChannels] = {0, 0};

  LiteRtRewriterT rewriter;
  Rewriter cc_rewriter(&rewriter);
  RankedTensorType tensor_type(kTensorType);
  LiteRtTensorT per_channel_quantized_tensor;
  auto per_channel_quantization = MakePerChannelQuantization(
      kScales, kZeroPoints, kQuantizedDimension, per_channel_quantized_tensor);
  auto ranked_tensor_spec = RankedTensorSpecBuilder(tensor_type)
                                .WithPerChannelQuantization(
                                    per_channel_quantization.second.per_channel)
                                .Build();
  auto tensor = cc_rewriter.BuildTensor(ranked_tensor_spec);
  ASSERT_TRUE(tensor.HasValue());
  EXPECT_EQ(tensor->ElementType(), ElementType::Float32);
  EXPECT_EQ(tensor->PerChannelQuantization().scales[0], 1.0);
  EXPECT_EQ(tensor->PerChannelQuantization().scales[1], 2.0);
  EXPECT_EQ(tensor->PerChannelQuantization().zero_points[0], 0);
  EXPECT_EQ(tensor->PerChannelQuantization().zero_points[1], 0);
  EXPECT_EQ(tensor->PerChannelQuantization().num_channels, 2);
  EXPECT_EQ(tensor->PerChannelQuantization().quantized_dimension, 0);
}

TEST(CcRewriterTest, TestBuildOp) {
  LiteRtRewriterT rewriter;
  Rewriter cc_rewriter(&rewriter);
  LiteRtTensorT litert_tensor_0;
  LiteRtTensorT litert_tensor_1;
  LiteRtTensorT litert_tensor_2;
  OpInputs inputs;
  inputs.push_back(Tensor(&litert_tensor_0));
  inputs.push_back(Tensor(&litert_tensor_1));
  OpOutputs outputs;
  outputs.push_back(Tensor(&litert_tensor_2));
  auto op = cc_rewriter.BuildOp(kLiteRtOpCodeTflAdd, inputs, outputs);
  EXPECT_EQ(op.Inputs().size(), 2);
  EXPECT_EQ(op.Outputs().size(), 1);
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflAdd);
}

}  // namespace
}  // namespace litert
