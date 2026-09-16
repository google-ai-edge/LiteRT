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

#include "litert/compiler/cc/litert_quantization.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"

namespace litert::compiler {
namespace {

using ::MakePerChannelQuantization;
using ::MakePerTensorQuantization;
using ::SetWeightsFromOwnedBuffer;
using ::litert::OwningBufferRef;
using ::litert::RankedTensorType;
using ::testing::ElementsAre;
using ::testing::FloatEq;

// A 2x3 tensor, which exercises per-channel quantization over a dimension
// that is not the innermost one.
constexpr int32_t kDimensions[] = {2, 3};
constexpr LiteRtLayout kLayout = BuildLayout(kDimensions);

LiteRtRankedTensorType MakeTensorType(LiteRtElementType element_type) {
  return LiteRtRankedTensorType{
      /*.element_type=*/element_type,
      /*.layout=*/kLayout,
  };
}

// Owns the scaffolding needed to hand a quantized constant tensor to the
// helpers under test.
class QuantizedTensorFixture {
 public:
  template <typename StorageT>
  Tensor Build(LiteRtElementType element_type,
               absl::Span<const StorageT> values,
               const Quantization& quantization) {
    weights_.SetBufferManager(&manager_);
    OwningBufferRef<uint8_t> buf(
        reinterpret_cast<const uint8_t*>(values.data()),
        values.size() * sizeof(StorageT));
    SetWeightsFromOwnedBuffer(weights_, std::move(buf));

    auto spec_builder =
        RankedTensorSpecBuilder(RankedTensorType(MakeTensorType(element_type)))
            .WithWeights(Weights(ctx_, &weights_));
    if (quantization.first == kLiteRtQuantizationPerTensor) {
      spec_builder =
          std::move(spec_builder)
              .WithPerTensorQuantization(quantization.second.per_tensor);
    } else if (quantization.first == kLiteRtQuantizationPerChannel) {
      spec_builder =
          std::move(spec_builder)
              .WithPerChannelQuantization(quantization.second.per_channel);
    }

    Builder cc_builder(ctx_, &builder_);
    auto tensor = cc_builder.BuildTensor(std::move(spec_builder).Build());
    EXPECT_TRUE(tensor.HasValue());
    return *tensor;
  }

  const LiteRtCompilerContext* ctx() const { return ctx_; }

 private:
  const LiteRtCompilerContext* ctx_ = LrtGetCompilerContext();
  LiteRtBuilderT builder_;
  ::litert::internal::BufferManager manager_;
  LiteRtWeightsT weights_;
};

TEST(DequantizeWeightsTest, PerTensorInt8) {
  QuantizedTensorFixture fixture;
  constexpr int8_t kValues[] = {0, 2, 4, -2, 6, 10};
  auto tensor = fixture.Build<int8_t>(kLiteRtElementTypeInt8,
                                      absl::MakeConstSpan(kValues),
                                      MakePerTensorQuantization(0.5f, 2));

  auto result = DequantizeWeights(tensor);
  ASSERT_TRUE(result.HasValue());
  // (value - 2) * 0.5
  EXPECT_THAT(*result,
              ElementsAre(FloatEq(-1.0f), FloatEq(0.0f), FloatEq(1.0f),
                          FloatEq(-2.0f), FloatEq(2.0f), FloatEq(4.0f)));
}

TEST(DequantizeWeightsTest, PerTensorUInt8) {
  QuantizedTensorFixture fixture;
  constexpr uint8_t kValues[] = {0, 1, 2, 3, 4, 5};
  auto tensor = fixture.Build<uint8_t>(kLiteRtElementTypeUInt8,
                                       absl::MakeConstSpan(kValues),
                                       MakePerTensorQuantization(2.0f, 1));

  auto result = DequantizeWeights(tensor);
  ASSERT_TRUE(result.HasValue());
  // (value - 1) * 2.0
  EXPECT_THAT(*result,
              ElementsAre(FloatEq(-2.0f), FloatEq(0.0f), FloatEq(2.0f),
                          FloatEq(4.0f), FloatEq(6.0f), FloatEq(8.0f)));
}

TEST(DequantizeWeightsTest, PerTensorInt16) {
  QuantizedTensorFixture fixture;
  constexpr int16_t kValues[] = {0, 100, 200, -100, 300, 400};
  auto tensor = fixture.Build<int16_t>(kLiteRtElementTypeInt16,
                                       absl::MakeConstSpan(kValues),
                                       MakePerTensorQuantization(0.01f, 0));

  auto result = DequantizeWeights(tensor);
  ASSERT_TRUE(result.HasValue());
  EXPECT_THAT(*result,
              ElementsAre(FloatEq(0.0f), FloatEq(1.0f), FloatEq(2.0f),
                          FloatEq(-1.0f), FloatEq(3.0f), FloatEq(4.0f)));
}

// The quantized dimension is the outermost one, so each channel spans three
// contiguous elements. This is the case the previous vendor-local helper
// could not express, as it assumed one channel per element.
TEST(DequantizeWeightsTest, PerChannelStridesOverQuantizedDimension) {
  QuantizedTensorFixture fixture;
  constexpr int kNumChannels = 2;
  constexpr float kScales[kNumChannels] = {1.0f, 2.0f};
  constexpr int64_t kZeroPoints[kNumChannels] = {0, 1};
  constexpr int8_t kValues[] = {1, 2, 3, 4, 5, 6};

  LiteRtTensorT scratch;
  auto tensor = fixture.Build<int8_t>(
      kLiteRtElementTypeInt8, absl::MakeConstSpan(kValues),
      MakePerChannelQuantization(kScales, kZeroPoints,
                                 /*quantized_dim=*/0, scratch));

  auto result = DequantizeWeights(tensor);
  ASSERT_TRUE(result.HasValue());
  // Channel 0 (scale 1, zp 0) covers {1,2,3}; channel 1 (scale 2, zp 1)
  // covers {4,5,6}.
  EXPECT_THAT(*result,
              ElementsAre(FloatEq(1.0f), FloatEq(2.0f), FloatEq(3.0f),
                          FloatEq(6.0f), FloatEq(8.0f), FloatEq(10.0f)));
}

TEST(DequantizeWeightsTest, PerChannelRejectsMismatchedChannelCount) {
  QuantizedTensorFixture fixture;
  // The tensor extent along dimension 0 is 2, not 3.
  constexpr int kNumChannels = 3;
  constexpr float kScales[kNumChannels] = {1.0f, 2.0f, 3.0f};
  constexpr int64_t kZeroPoints[kNumChannels] = {0, 0, 0};
  constexpr int8_t kValues[] = {1, 2, 3, 4, 5, 6};

  LiteRtTensorT scratch;
  auto tensor = fixture.Build<int8_t>(
      kLiteRtElementTypeInt8, absl::MakeConstSpan(kValues),
      MakePerChannelQuantization(kScales, kZeroPoints,
                                 /*quantized_dim=*/0, scratch));

  auto result = DequantizeWeights(tensor);
  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(DequantizeWeightsTest, RejectsUnquantizedTensor) {
  QuantizedTensorFixture fixture;
  constexpr int8_t kValues[] = {1, 2, 3, 4, 5, 6};
  auto tensor = fixture.Build<int8_t>(kLiteRtElementTypeInt8,
                                      absl::MakeConstSpan(kValues),
                                      MakeEmptyQuantization());

  auto result = DequantizeWeights(tensor);
  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorUnsupported);
}

TEST(DequantizeWeightsTest, RejectsSubByteStorageType) {
  QuantizedTensorFixture fixture;
  // Int4 weights are packed two-per-byte, so they cannot be strided over
  // directly and must be unpacked first.
  constexpr int8_t kValues[] = {1, 2, 3, 4, 5, 6};
  auto tensor = fixture.Build<int8_t>(kLiteRtElementTypeInt4,
                                      absl::MakeConstSpan(kValues),
                                      MakePerTensorQuantization(1.0f, 0));

  auto result = DequantizeWeights(tensor);
  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorUnsupported);
}

TEST(DequantizeWeightsTest, RejectsTensorWithoutWeights) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtBuilderT builder;
  Builder cc_builder(ctx, &builder);
  auto quantization = MakePerTensorQuantization(1.0f, 0);
  auto tensor = cc_builder.BuildTensor(
      RankedTensorSpecBuilder(
          RankedTensorType(MakeTensorType(kLiteRtElementTypeInt8)))
          .WithPerTensorQuantization(quantization.second.per_tensor)
          .Build());
  ASSERT_TRUE(tensor.HasValue());

  auto result = DequantizeWeights(*tensor);
  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(DequantizeWeightsIntoTest, WritesIntoCallerBuffer) {
  QuantizedTensorFixture fixture;
  constexpr int8_t kValues[] = {0, 2, 4, -2, 6, 10};
  auto tensor = fixture.Build<int8_t>(kLiteRtElementTypeInt8,
                                      absl::MakeConstSpan(kValues),
                                      MakePerTensorQuantization(0.5f, 2));

  std::vector<float> out(6);
  auto result = DequantizeWeightsInto(tensor, absl::MakeSpan(out));
  ASSERT_TRUE(result.HasValue());
  EXPECT_THAT(out, ElementsAre(FloatEq(-1.0f), FloatEq(0.0f), FloatEq(1.0f),
                               FloatEq(-2.0f), FloatEq(2.0f), FloatEq(4.0f)));
}

TEST(DequantizeWeightsIntoTest, RejectsUndersizedBuffer) {
  QuantizedTensorFixture fixture;
  constexpr int8_t kValues[] = {0, 2, 4, -2, 6, 10};
  auto tensor = fixture.Build<int8_t>(kLiteRtElementTypeInt8,
                                      absl::MakeConstSpan(kValues),
                                      MakePerTensorQuantization(0.5f, 2));

  std::vector<float> out(3);
  auto result = DequantizeWeightsInto(tensor, absl::MakeSpan(out));
  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert::compiler
