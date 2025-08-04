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

#include "litert/c/litert_rewriter.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"
#include "litert/test/matchers.h"

namespace {

TEST(LiteRtRewriterTest, EraseOp) {
  LiteRtRewriterT rewriter;
  LiteRtOpT op;
  LITERT_ASSERT_OK(LiteRtRewriterEraseOp(&op, &rewriter));
  EXPECT_TRUE(rewriter.Erases().contains(&op));
}

TEST(LiteRtRewriterTest, EraseOpNullOp) {
  LiteRtRewriterT rewriter;
  EXPECT_EQ(LiteRtRewriterEraseOp(nullptr, &rewriter),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtRewriterTest, EraseOpNullRewriter) {
  EXPECT_EQ(LiteRtRewriterEraseOp(nullptr, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtRewriterTest, BuildUnrankedTensor) {
  LiteRtRewriterT rewriter;
  LiteRtTensor tensor;
  std::string name = "GT";
  LiteRtUnrankedTensorType unranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32};
  LITERT_ASSERT_OK(LiteRtRewriterBuildTensor(
      kLiteRtUnrankedTensorType, LiteRtRankedTensorType(), unranked_tensor_type,
      LiteRtWeights(), kLiteRtQuantizationNone, LiteRtQuantizationPerTensor(),
      LiteRtQuantizationPerChannel(), &rewriter, name.c_str(), name.size(),
      &tensor));
  EXPECT_EQ(tensor->Type().first, kLiteRtUnrankedTensorType);
  EXPECT_EQ(tensor->Type().second.unranked_tensor_type.element_type,
            kLiteRtElementTypeFloat32);
  EXPECT_EQ(tensor->Name(), name);
}

TEST(LiteRtRewriterTest, BuildUnrankedTensorWithWeights) {
  static constexpr absl::string_view kData = "CircuitDeMonaco";
  litert::internal::BufferManager manager;
  LiteRtWeightsT weights;
  {
    litert::OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }
  weights.SetBufferId(weights.GetBufferId());
  weights.SetBufferManager(&manager);
  LiteRtRewriterT rewriter;
  LiteRtTensor tensor;
  LiteRtUnrankedTensorType unranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32};
  EXPECT_EQ(LiteRtRewriterBuildTensor(
                kLiteRtUnrankedTensorType, LiteRtRankedTensorType(),
                unranked_tensor_type, &weights, kLiteRtQuantizationNone,
                LiteRtQuantizationPerTensor(), LiteRtQuantizationPerChannel(),
                &rewriter, nullptr, 0, &tensor),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtRewriterTest, BuildRankedTensor) {
  LiteRtRewriterT rewriter;
  LiteRtTensor tensor;
  std::string name = "GT2";
  LiteRtRankedTensorType ranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32,
      .layout = {.rank = 2, .dimensions = {3, 3}}};
  LITERT_ASSERT_OK(LiteRtRewriterBuildTensor(
      kLiteRtRankedTensorType, ranked_tensor_type, LiteRtUnrankedTensorType(),
      LiteRtWeights(), kLiteRtQuantizationNone, LiteRtQuantizationPerTensor(),
      LiteRtQuantizationPerChannel(), &rewriter, name.c_str(), name.size(),
      &tensor));

  EXPECT_EQ(tensor->Type().first, kLiteRtRankedTensorType);
  EXPECT_EQ(tensor->Type().second.ranked_tensor_type.element_type,
            kLiteRtElementTypeFloat32);
  EXPECT_EQ(tensor->Type().second.ranked_tensor_type.layout.rank, 2);
  EXPECT_EQ(tensor->Type().second.ranked_tensor_type.layout.dimensions[0], 3);
  EXPECT_EQ(tensor->Type().second.ranked_tensor_type.layout.dimensions[1], 3);
  EXPECT_EQ(tensor->Name(), name);
}

TEST(LiteRtRewriterTest, BuildRankedTensorWithWeights) {
  static constexpr absl::string_view kData = "Nurburgring";
  litert::internal::BufferManager manager;
  LiteRtWeightsT weights;
  {
    weights.SetBufferManager(&manager);
    litert::OwningBufferRef<uint8_t> buf(kData);
    SetWeightsFromOwnedBuffer(weights, std::move(buf));
  }

  LiteRtRewriterT rewriter;
  LiteRtTensor tensor;
  LiteRtRankedTensorType ranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32,
      .layout = {.rank = 2, .dimensions = {3, 3}}};

  LITERT_ASSERT_OK(LiteRtRewriterBuildTensor(
      kLiteRtRankedTensorType, ranked_tensor_type, LiteRtUnrankedTensorType(),
      &weights, kLiteRtQuantizationNone, LiteRtQuantizationPerTensor(),
      LiteRtQuantizationPerChannel(), &rewriter, nullptr, 0, &tensor));

  EXPECT_EQ(tensor->Weights().GetBufferId(), 1);
  EXPECT_EQ(tensor->Weights().GetBufferManager(), &manager);
  EXPECT_EQ(tensor->Weights().Buffer().Size(), kData.size());
  EXPECT_EQ(tensor->Weights().Buffer().StrView(), kData);
}

TEST(LiteRtRewriterTest, BuildTensorWithPerTensorQuantization) {
  LiteRtRewriterT rewriter;
  LiteRtTensor tensor;
  std::string name = "GT3";
  LiteRtRankedTensorType ranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32,
      .layout = {.rank = 2, .dimensions = {3, 3}}};
  LiteRtQuantizationPerTensor per_tensor_quantization = {.scale = 1.0,
                                                         .zero_point = 1};
  LITERT_ASSERT_OK(LiteRtRewriterBuildTensor(
      kLiteRtRankedTensorType, ranked_tensor_type, LiteRtUnrankedTensorType(),
      LiteRtWeights(), kLiteRtQuantizationPerTensor, per_tensor_quantization,
      LiteRtQuantizationPerChannel(), &rewriter, name.c_str(), name.size(),
      &tensor));
  EXPECT_EQ(tensor->Qparams().first, kLiteRtQuantizationPerTensor);
  EXPECT_EQ(tensor->Qparams().second.per_tensor.scale, 1.0);
  EXPECT_EQ(tensor->Qparams().second.per_tensor.zero_point, 1);
}

TEST(LiteRtRewriterTest, BuildTensorWithPerChannelQuantization) {
  constexpr auto kNumChannels = 2;
  constexpr auto kQuantizedDimension = 0;
  constexpr float kScales[kNumChannels] = {1.0, 2.0};
  constexpr int64_t kZeroPoints[kNumChannels] = {0, 0};

  LiteRtRewriterT rewriter;
  LiteRtTensor tensor;
  std::string name = "GT4";
  LiteRtTensorT per_channel_quantized_tensor;
  LiteRtRankedTensorType ranked_tensor_type = {
      .element_type = kLiteRtElementTypeFloat32,
      .layout = {.rank = 2, .dimensions = {3, 3}}};
  LiteRtQuantizationPerChannel per_channel_quantization =
      MakePerChannelQuantization(kScales, kZeroPoints, kQuantizedDimension,
                                 per_channel_quantized_tensor)
          .second.per_channel;

  LITERT_ASSERT_OK(LiteRtRewriterBuildTensor(
      kLiteRtRankedTensorType, ranked_tensor_type, LiteRtUnrankedTensorType(),
      LiteRtWeights(), kLiteRtQuantizationPerChannel,
      LiteRtQuantizationPerTensor(), per_channel_quantization, &rewriter,
      name.c_str(), name.size(), &tensor));

  auto built_per_channel_quantization = tensor->Qparams().second.per_channel;
  EXPECT_THAT(
      absl::MakeConstSpan(built_per_channel_quantization.scales, kNumChannels),
      ::testing::ElementsAreArray(kScales));
  EXPECT_THAT(absl::MakeConstSpan(built_per_channel_quantization.zero_points,
                                  kNumChannels),
              ::testing::ElementsAreArray(kZeroPoints));
  EXPECT_EQ(built_per_channel_quantization.num_channels, kNumChannels);
  EXPECT_EQ(built_per_channel_quantization.quantized_dimension,
            kQuantizedDimension);
}

TEST(LiteRtRewriterTest, BuildOpTest) {
  LiteRtRewriterT rewriter;
  LiteRtTensorT input_tensor_0;
  LiteRtTensorT input_tensor_1;
  LiteRtTensorT output_tensor_0;
  LiteRtOp op;

  std::vector<LiteRtTensor> input_tensors = {&input_tensor_0, &input_tensor_1};
  std::vector<LiteRtTensor> output_tensors = {&output_tensor_0};
  LITERT_ASSERT_OK(LiteRtRewriterBuildOp(
      kLiteRtOpCodeTflAdd, input_tensors.size(), input_tensors.data(),
      output_tensors.size(), output_tensors.data(), &rewriter, &op));
  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflAdd);
  EXPECT_EQ(op->Inputs().size(), 2);
  EXPECT_EQ(op->Inputs().at(0), &input_tensor_0);
  EXPECT_EQ(op->Inputs().at(1), &input_tensor_1);
  EXPECT_EQ(op->Outputs().size(), 1);
  EXPECT_EQ(op->Outputs().at(0), &output_tensor_0);
}
}  // namespace
