// Copyright 2024 Google LLC.
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

#include "litert/core/model/flatbuffer_to_litert.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_model_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAreArray;

TEST(FlatbufferToLiteRtTest, MapStaticTensorType) {
  static constexpr int32_t kDims[] = {2, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);

  auto t = MapTensorType(std::make_pair(TflElementType::TensorType_INT32,
                                        TflShapeInfo(kDimsSpan)));
  ASSERT_TRUE(t);

  ASSERT_EQ(t->first, kLiteRtRankedTensorType);
  auto& ranked = t->second.ranked_tensor_type;
  EXPECT_EQ(ranked.element_type, kLiteRtElementTypeInt32);
  EXPECT_EQ(absl::MakeSpan(ranked.layout.dimensions, ranked.layout.rank),
            kDimsSpan);
}

TEST(FlatbufferToLiteRtTest, MapStaticTensorInt4Type) {
  static constexpr int32_t kDims[] = {2, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);

  auto t = MapTensorType(
      std::make_pair(TflElementType::TensorType_INT4, TflShapeInfo(kDimsSpan)));
  ASSERT_TRUE(t);

  ASSERT_EQ(t->first, kLiteRtRankedTensorType);
  auto& ranked = t->second.ranked_tensor_type;
  EXPECT_EQ(ranked.element_type, kLiteRtElementTypeInt4);
  EXPECT_EQ(absl::MakeSpan(ranked.layout.dimensions, ranked.layout.rank),
            kDimsSpan);
}

TEST(FlatbufferToLiteRtTest, MapStaticTensorInt2Type) {
  static constexpr int32_t kDims[] = {2, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);

  auto t = MapTensorType(
      std::make_pair(TflElementType::TensorType_INT2, TflShapeInfo(kDimsSpan)));
  ASSERT_TRUE(t);

  ASSERT_EQ(t->first, kLiteRtRankedTensorType);
  auto& ranked = t->second.ranked_tensor_type;
  EXPECT_EQ(ranked.element_type, kLiteRtElementTypeInt2);
  EXPECT_EQ(absl::MakeSpan(ranked.layout.dimensions, ranked.layout.rank),
            kDimsSpan);
}

TEST(FlatbufferToLiteRtTest, MapDynamicTensorType) {
  static constexpr int32_t kDims[] = {-1, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);

  auto t = MapTensorType(std::make_pair(TflElementType::TensorType_INT32,
                                        TflShapeInfo(kDimsSpan)));
  ASSERT_TRUE(t);

  ASSERT_EQ(t->first, kLiteRtRankedTensorType);
  auto& ranked = t->second.ranked_tensor_type;
  EXPECT_EQ(ranked.element_type, kLiteRtElementTypeInt32);
  EXPECT_EQ(absl::MakeSpan(ranked.layout.dimensions, ranked.layout.rank),
            kDimsSpan);
}

TEST(FlatbufferToLiteRtTest, MapNoQuantization) {
  auto q = MapQuantization(nullptr);
  ASSERT_TRUE(q);
  ASSERT_EQ(q->first, kLiteRtQuantizationNone);
}

TEST(FlatbufferToLiteRtTest, MapPerTensorQuantization) {
  static constexpr float kScale = 1.0;
  static constexpr int64_t kZp = 2;

  flatbuffers::FlatBufferBuilder fbb;
  std::vector<float> scales = {kScale};
  std::vector<int64_t> zero_points = {kZp};
  auto scales_fb = fbb.CreateVector(scales);
  auto zero_points_fb = fbb.CreateVector(zero_points);
  tflite::QuantizationParametersBuilder qpb(fbb);
  qpb.add_scale(scales_fb);
  qpb.add_zero_point(zero_points_fb);
  auto q_offset = qpb.Finish();
  fbb.Finish(q_offset);
  auto* tfl_q = flatbuffers::GetRoot<tflite::QuantizationParameters>(
      fbb.GetBufferPointer());

  auto q = MapQuantization(tfl_q);
  ASSERT_TRUE(q);
  ASSERT_EQ(q->first, kLiteRtQuantizationPerTensor);
  EXPECT_EQ(q->second.per_tensor.scale, kScale);
  EXPECT_EQ(q->second.per_tensor.zero_point, kZp);
}

TEST(FlatbufferToLiteRtTest, MapPerChannelQuantization) {
  static constexpr size_t kRank = 2;
  static constexpr float kScales[kRank] = {1.0, 2.0};
  static constexpr int64_t kZps[kRank] = {2, 3};
  static constexpr int32_t kQDim = 1;

  flatbuffers::FlatBufferBuilder fbb;
  std::vector<float> scales = {kScales[0], kScales[1]};
  std::vector<int64_t> zero_points = {kZps[0], kZps[1]};
  auto scales_fb = fbb.CreateVector(scales);
  auto zero_points_fb = fbb.CreateVector(zero_points);
  tflite::QuantizationParametersBuilder qpb(fbb);
  qpb.add_scale(scales_fb);
  qpb.add_zero_point(zero_points_fb);
  qpb.add_quantized_dimension(kQDim);
  auto q_offset = qpb.Finish();
  fbb.Finish(q_offset);
  auto* tfl_q = flatbuffers::GetRoot<tflite::QuantizationParameters>(
      fbb.GetBufferPointer());

  auto q = MapQuantization(tfl_q);
  ASSERT_TRUE(q);
  ASSERT_EQ(q->first, kLiteRtQuantizationPerChannel);
  EXPECT_THAT(absl::MakeConstSpan(q->second.per_channel.scales, kRank),
              ElementsAreArray(kScales));

  EXPECT_THAT(absl::MakeConstSpan(q->second.per_channel.zero_points, kRank),
              ElementsAreArray(kZps));
  EXPECT_EQ(q->second.per_channel.quantized_dimension, kQDim);
  EXPECT_EQ(q->second.per_channel.num_channels, kRank);
}

}  // namespace
}  // namespace litert::internal
