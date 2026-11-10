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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAreArray;

struct ElementTypeMappingCase {
  std::string test_name;
  TflElementType tfl_type;
  LiteRtElementType expected_type;
};

using ElementTypeMappingTest = ::testing::TestWithParam<ElementTypeMappingCase>;

TEST_P(ElementTypeMappingTest, MapsElementAndStaticTensorType) {
  static constexpr int32_t kDims[] = {2, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);
  const auto& param = GetParam();

  EXPECT_EQ(MapElementType(param.tfl_type), param.expected_type);

  auto t =
      MapTensorType(std::make_pair(param.tfl_type, TflShapeInfo(kDimsSpan)));
  ASSERT_TRUE(t);
  ASSERT_EQ(t->first, kLiteRtRankedTensorType);
  const auto& ranked = t->second.ranked_tensor_type;
  EXPECT_EQ(ranked.element_type, param.expected_type);
  EXPECT_EQ(absl::MakeSpan(ranked.layout.dimensions, ranked.layout.rank),
            kDimsSpan);
}

INSTANTIATE_TEST_SUITE_P(
    AllElementTypes, ElementTypeMappingTest,
    ::testing::ValuesIn<ElementTypeMappingCase>({
        {"Float64", tflite::TensorType_FLOAT64, kLiteRtElementTypeFloat64},
        {"Float32", tflite::TensorType_FLOAT32, kLiteRtElementTypeFloat32},
        {"Float16", tflite::TensorType_FLOAT16, kLiteRtElementTypeFloat16},
        {"BFloat16", tflite::TensorType_BFLOAT16, kLiteRtElementTypeBFloat16},
        {"Float8E4M3FN", tflite::TensorType_FLOAT8_E4M3FN,
         kLiteRtElementTypeFloat8E4M3FN},
        {"Float8E5M2", tflite::TensorType_FLOAT8_E5M2,
         kLiteRtElementTypeFloat8E5M2},
        {"Complex64", tflite::TensorType_COMPLEX64,
         kLiteRtElementTypeComplex64},
        {"Complex128", tflite::TensorType_COMPLEX128,
         kLiteRtElementTypeComplex128},
        {"Int32", tflite::TensorType_INT32, kLiteRtElementTypeInt32},
        {"UInt32", tflite::TensorType_UINT32, kLiteRtElementTypeUInt32},
        {"Int64", tflite::TensorType_INT64, kLiteRtElementTypeInt64},
        {"UInt64", tflite::TensorType_UINT64, kLiteRtElementTypeUInt64},
        {"Bool", tflite::TensorType_BOOL, kLiteRtElementTypeBool},
        {"Int16", tflite::TensorType_INT16, kLiteRtElementTypeInt16},
        {"UInt16", tflite::TensorType_UINT16, kLiteRtElementTypeUInt16},
        {"Int8", tflite::TensorType_INT8, kLiteRtElementTypeInt8},
        {"UInt8", tflite::TensorType_UINT8, kLiteRtElementTypeUInt8},
        {"UInt4", tflite::TensorType_UINT4, kLiteRtElementTypeUInt4},
        {"Int4", tflite::TensorType_INT4, kLiteRtElementTypeInt4},
        {"Int2", tflite::TensorType_INT2, kLiteRtElementTypeInt2},
        {"Resource", tflite::TensorType_RESOURCE, kLiteRtElementTypeTfResource},
        {"String", tflite::TensorType_STRING, kLiteRtElementTypeTfString},
        {"Variant", tflite::TensorType_VARIANT, kLiteRtElementTypeTfVariant},
    }),
    [](const ::testing::TestParamInfo<ElementTypeMappingCase>& info) {
      return info.param.test_name;
    });

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

TEST(FlatbufferToLiteRtTest,
     MapQuantizationRejectsMissingOrMismatchedZeroPoint) {
  for (bool add_empty_zero_point : {false, true}) {
    flatbuffers::FlatBufferBuilder fbb;
    auto scales_fb = fbb.CreateVector(std::vector<float>{1.0f});
    auto zero_points_fb =
        add_empty_zero_point ? fbb.CreateVector(std::vector<int64_t>{}) : 0;
    tflite::QuantizationParametersBuilder qpb(fbb);
    qpb.add_scale(scales_fb);
    if (add_empty_zero_point) {
      qpb.add_zero_point(zero_points_fb);
    }
    fbb.Finish(qpb.Finish());
    const auto* tfl_q = flatbuffers::GetRoot<tflite::QuantizationParameters>(
        fbb.GetBufferPointer());
    EXPECT_FALSE(MapQuantization(tfl_q));
  }
}

TEST(FlatbufferToLiteRtTest,
     MapTensorTypeRejectsUnsupportedElementTypeAndExcessiveRank) {
  static constexpr int32_t kDims[] = {2, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);

  EXPECT_EQ(MapElementType(static_cast<TflElementType>(999)),
            kLiteRtElementTypeNone);
  EXPECT_FALSE(MapTensorType(std::make_pair(static_cast<TflElementType>(999),
                                            TflShapeInfo(kDimsSpan))));

  std::vector<int32_t> huge_rank_dims(LITERT_TENSOR_MAX_RANK + 1, 1);
  EXPECT_FALSE(MapTensorType(
      std::make_pair(tflite::TensorType_FLOAT32,
                     TflShapeInfo(absl::MakeConstSpan(huge_rank_dims)))));
}

TEST(FlatbufferToLiteRtTest, SupportPredicatesForOpBufferAndTensor) {
  tflite::OperatorT op;
  EXPECT_EQ(IsOpSupported(op), kLiteRtStatusOk);

  op.intermediates = {0};
  EXPECT_EQ(IsOpSupported(op), kLiteRtStatusErrorUnsupported);
  op.intermediates.clear();

  op.large_custom_options_size = 16;
  EXPECT_EQ(IsOpSupported(op), kLiteRtStatusErrorUnsupported);
  op.large_custom_options_size = 0;

  op.mutating_variable_inputs = {false, true};
  EXPECT_EQ(IsOpSupported(op), kLiteRtStatusErrorUnsupported);

  tflite::BufferT buffer;
  EXPECT_EQ(IsBufferSupported(buffer), kLiteRtStatusOk);
  buffer.offset = 64;
  EXPECT_EQ(IsBufferSupported(buffer), kLiteRtStatusErrorUnsupported);

  TflTensor tensor;
  EXPECT_EQ(IsTensorSupported(tensor), kLiteRtStatusOk);
  tensor.is_variable = true;
  EXPECT_EQ(IsTensorSupported(tensor), kLiteRtStatusErrorUnsupported);
  tensor.is_variable = false;

  tensor.variant_tensors.push_back(std::make_unique<tflite::VariantSubTypeT>());
  EXPECT_EQ(IsTensorSupported(tensor), kLiteRtStatusErrorUnsupported);
  tensor.variant_tensors.clear();

  tensor.sparsity = std::make_unique<tflite::SparsityParametersT>();
  EXPECT_EQ(IsTensorSupported(tensor), kLiteRtStatusErrorUnsupported);
}

}  // namespace
}  // namespace litert::internal
