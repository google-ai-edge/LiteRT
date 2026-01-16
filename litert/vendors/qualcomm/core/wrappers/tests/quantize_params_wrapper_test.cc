// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {

TEST(UndefinedQuantizeParamsWrapperTest, DefaultConstructorTest) {
  UndefinedQuantizeParamsWrapper wrapper;
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_UNDEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_UNDEFINED);
}

TEST(UndefinedQuantizeParamsWrapperTest, CopyConstructorTest) {
  UndefinedQuantizeParamsWrapper wrapper1;
  UndefinedQuantizeParamsWrapper wrapper2(wrapper1);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_UNDEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_UNDEFINED);
}

TEST(UndefinedQuantizeParamsWrapperTest, MoveConstructorTest) {
  UndefinedQuantizeParamsWrapper wrapper1;
  UndefinedQuantizeParamsWrapper wrapper2(std::move(wrapper1));
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_UNDEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_UNDEFINED);
}

TEST(UndefinedQuantizeParamsWrapperTest, EqualityOperatorTest) {
  UndefinedQuantizeParamsWrapper wrapper1;
  UndefinedQuantizeParamsWrapper wrapper2;

  EXPECT_TRUE(wrapper1 == wrapper2);
}

TEST(ScaleOffsetQuantizeParamsWrapperTest, ConstructorTest) {
  float scale = 1.5f;
  std::int32_t zero_point = 10;
  ScaleOffsetQuantizeParamsWrapper wrapper(scale, zero_point);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_FLOAT_EQ(dst.scaleOffsetEncoding.scale, scale);
  EXPECT_EQ(dst.scaleOffsetEncoding.offset, -zero_point);
}

TEST(ScaleOffsetQuantizeParamsWrapperTest, CopyConstructorTest) {
  float scale = 1.5f;
  std::int32_t zero_point = 10;
  ScaleOffsetQuantizeParamsWrapper wrapper1(scale, zero_point);
  ScaleOffsetQuantizeParamsWrapper wrapper2(wrapper1);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_FLOAT_EQ(dst.scaleOffsetEncoding.scale, scale);
  EXPECT_EQ(dst.scaleOffsetEncoding.offset, -zero_point);
}

TEST(ScaleOffsetQuantizeParamsWrapperTest, MoveConstructorTest) {
  float scale = 1.5f;
  std::int32_t zero_point = 10;
  ScaleOffsetQuantizeParamsWrapper wrapper1(scale, zero_point);
  ScaleOffsetQuantizeParamsWrapper wrapper2(std::move(wrapper1));
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_FLOAT_EQ(dst.scaleOffsetEncoding.scale, scale);
  EXPECT_EQ(dst.scaleOffsetEncoding.offset, -zero_point);
}

TEST(ScaleOffsetQuantizeParamsWrapperTest, GetterTest) {
  float scale = 1.5f;
  std::int32_t zero_point = 10;
  ScaleOffsetQuantizeParamsWrapper wrapper(scale, zero_point);
  EXPECT_FLOAT_EQ(wrapper.GetScale(), scale);
  EXPECT_EQ(wrapper.GetZeroPoint(), zero_point);
  EXPECT_EQ(wrapper.GetOffset(), -zero_point);
}

using ScaleOffsetParamsWithType = std::tuple<float, int32_t, float, int32_t>;
class ScaleOffsetEqualTestWithType
    : public ::testing::TestWithParam<ScaleOffsetParamsWithType> {};
TEST_P(ScaleOffsetEqualTestWithType, EqualityOperator) {
  auto [scale1, offset1, scale2, offset2] = GetParam();

  qnn::ScaleOffsetQuantizeParamsWrapper wrapper1(scale1, offset1);
  qnn::ScaleOffsetQuantizeParamsWrapper wrapper2(scale2, offset2);
  qnn::ScaleOffsetQuantizeParamsWrapper wrapper3(scale2, offset2);

  const bool expected_equal = (scale1 == scale2) && (offset1 == offset2);
  EXPECT_EQ(wrapper1 == wrapper2, expected_equal);
  EXPECT_EQ(wrapper1 == wrapper3, expected_equal);
  EXPECT_TRUE(wrapper2 == wrapper3);
}
// Data type, scale and offset
INSTANTIATE_TEST_SUITE_P(ScaleOffsetQuantizeParamsWrapperTest_Combine,
                         ScaleOffsetEqualTestWithType,
                         ::testing::Combine(::testing::Values(1.5f),
                                            ::testing::Values(10),
                                            ::testing::Values(1.5f, 1.6f),
                                            ::testing::Values(10, 12)));

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, ConstructorTest) {
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper(axis, scales, zero_points);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding,
            QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.axis, axis);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.numScaleOffsets, scales.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    EXPECT_FLOAT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].scale,
                    scales[i]);
    EXPECT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].offset,
              -zero_points[i]);
  }
}

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, CopyConstructorTest) {
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper1(axis, scales, zero_points);
  AxisScaleOffsetQuantizeParamsWrapper wrapper2(wrapper1);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding,
            QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.axis, axis);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.numScaleOffsets, scales.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    EXPECT_FLOAT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].scale,
                    scales[i]);
    EXPECT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].offset,
              -zero_points[i]);
  }
}

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, MoveConstructorTest) {
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper1(axis, scales, zero_points);
  AxisScaleOffsetQuantizeParamsWrapper wrapper2(std::move(wrapper1));
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  EXPECT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  EXPECT_EQ(dst.quantizationEncoding,
            QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.axis, axis);
  EXPECT_EQ(dst.axisScaleOffsetEncoding.numScaleOffsets, scales.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    EXPECT_FLOAT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].scale,
                    scales[i]);
    EXPECT_EQ(dst.axisScaleOffsetEncoding.scaleOffset[i].offset,
              -zero_points[i]);
  }
}

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, GetterTest) {
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper(axis, scales, zero_points);
  std::vector<float> scales_out;
  wrapper.GetScales(scales_out);
  EXPECT_EQ(scales, scales_out);
  std::vector<std::int32_t> zero_points_out;
  wrapper.GetZeroPoints(zero_points_out);
  EXPECT_EQ(zero_points, zero_points_out);
}

TEST(ScaleOffsetQuantizeParamsWrapperTest, QnnConstructorTest) {
  ScaleOffsetQuantizeParamsWrapper wrapper1(1.5f, 10);
  Qnn_QuantizeParams_t dst1 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst1);
  ScaleOffsetQuantizeParamsWrapper wrapper2(dst1.scaleOffsetEncoding);
  Qnn_QuantizeParams_t dst2 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst2);
  EXPECT_EQ(dst1.encodingDefinition, dst2.encodingDefinition);
  EXPECT_EQ(dst1.quantizationEncoding, dst2.quantizationEncoding);
  EXPECT_FLOAT_EQ(dst1.scaleOffsetEncoding.scale,
                  dst2.scaleOffsetEncoding.scale);
  EXPECT_EQ(dst1.scaleOffsetEncoding.offset, dst2.scaleOffsetEncoding.offset);
}

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, QnnConstructorTest) {
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper1(axis, scales, zero_points);
  Qnn_QuantizeParams_t dst1 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst1);
  AxisScaleOffsetQuantizeParamsWrapper wrapper2(dst1.axisScaleOffsetEncoding);
  Qnn_QuantizeParams_t dst2 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst2);
  EXPECT_EQ(dst1.encodingDefinition, dst2.encodingDefinition);
  EXPECT_EQ(dst1.quantizationEncoding, dst2.quantizationEncoding);
  EXPECT_EQ(dst1.axisScaleOffsetEncoding.numScaleOffsets,
            dst2.axisScaleOffsetEncoding.numScaleOffsets);
  for (size_t i = 0; i < dst1.axisScaleOffsetEncoding.numScaleOffsets; ++i) {
    EXPECT_EQ(dst1.axisScaleOffsetEncoding.scaleOffset[i].scale,
              dst2.axisScaleOffsetEncoding.scaleOffset[i].scale);
    EXPECT_EQ(dst1.axisScaleOffsetEncoding.scaleOffset[i].offset,
              dst2.axisScaleOffsetEncoding.scaleOffset[i].offset);
  }
}

TEST(BwScaleOffsetQuantizeParamsWrapperTest, CopyConstructorTest) {
  BwScaleOffsetQuantizeParamsWrapper wrapper1(4, 1.5f, 10);
  Qnn_QuantizeParams_t dst1 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst1);
  BwScaleOffsetQuantizeParamsWrapper wrapper2(wrapper1);
  Qnn_QuantizeParams_t dst2 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst2);
  ASSERT_EQ(dst1.encodingDefinition, dst2.encodingDefinition);
  ASSERT_EQ(dst1.quantizationEncoding, dst2.quantizationEncoding);
  ASSERT_EQ(dst1.bwScaleOffsetEncoding.bitwidth,
            dst2.bwScaleOffsetEncoding.bitwidth);
  ASSERT_FLOAT_EQ(dst1.bwScaleOffsetEncoding.scale,
                  dst2.bwScaleOffsetEncoding.scale);
  ASSERT_EQ(dst1.bwScaleOffsetEncoding.offset,
            dst2.bwScaleOffsetEncoding.offset);
}

TEST(BwScaleOffsetQuantizeParamsWrapperTest, MoveConstructorTest) {
  BwScaleOffsetQuantizeParamsWrapper wrapper1(4, 1.5f, 10);
  Qnn_QuantizeParams_t dst1 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst1);
  BwScaleOffsetQuantizeParamsWrapper wrapper2(std::move(wrapper1));
  Qnn_QuantizeParams_t dst2 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst2);
  ASSERT_EQ(dst1.encodingDefinition, dst2.encodingDefinition);
  ASSERT_EQ(dst1.quantizationEncoding, dst2.quantizationEncoding);
  ASSERT_EQ(dst1.bwScaleOffsetEncoding.bitwidth,
            dst2.bwScaleOffsetEncoding.bitwidth);
  ASSERT_FLOAT_EQ(dst1.bwScaleOffsetEncoding.scale,
                  dst2.bwScaleOffsetEncoding.scale);
  ASSERT_EQ(dst1.bwScaleOffsetEncoding.offset,
            dst2.bwScaleOffsetEncoding.offset);
}

TEST(BwScaleOffsetQuantizeParamsWrapperTest, GetBitwidthTest) {
  BwScaleOffsetQuantizeParamsWrapper wrapper(4, 1.5f, 10);
  ASSERT_EQ(wrapper.GetBitwidth(), 4);
}

TEST(BwAxisScaleOffsetQuantizeParamsWrapperTest, CopyConstructorTest) {
  std::uint32_t bw = 4;
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  BwAxisScaleOffsetQuantizeParamsWrapper wrapper1(bw, axis, scales,
                                                  zero_points);
  BwAxisScaleOffsetQuantizeParamsWrapper wrapper2(wrapper1);
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  ASSERT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  ASSERT_EQ(dst.quantizationEncoding,
            QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET);
  ASSERT_EQ(dst.bwAxisScaleOffsetEncoding.bitwidth, bw);
  ASSERT_EQ(dst.bwAxisScaleOffsetEncoding.axis, axis);
  ASSERT_EQ(dst.bwAxisScaleOffsetEncoding.numElements, scales.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    ASSERT_FLOAT_EQ(dst.bwAxisScaleOffsetEncoding.scales[i], scales[i]);
    ASSERT_EQ(dst.bwAxisScaleOffsetEncoding.offsets[i], -zero_points[i]);
  }
}

TEST(BwAxisScaleOffsetQuantizeParamsWrapperTest, MoveConstructorTest) {
  std::uint32_t bw = 4;
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  BwAxisScaleOffsetQuantizeParamsWrapper wrapper1(bw, axis, scales,
                                                  zero_points);
  BwAxisScaleOffsetQuantizeParamsWrapper wrapper2(std::move(wrapper1));
  Qnn_QuantizeParams_t dst = QNN_QUANTIZE_PARAMS_INIT;
  wrapper2.CloneTo(dst);
  ASSERT_EQ(dst.encodingDefinition, QNN_DEFINITION_DEFINED);
  ASSERT_EQ(dst.quantizationEncoding,
            QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET);
  ASSERT_EQ(dst.bwAxisScaleOffsetEncoding.bitwidth, bw);
  ASSERT_EQ(dst.bwAxisScaleOffsetEncoding.axis, axis);
  ASSERT_EQ(dst.bwAxisScaleOffsetEncoding.numElements, scales.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    ASSERT_FLOAT_EQ(dst.bwAxisScaleOffsetEncoding.scales[i], scales[i]);
    ASSERT_EQ(dst.bwAxisScaleOffsetEncoding.offsets[i], -zero_points[i]);
  }
}

TEST(BwAxisScaleOffsetQuantizeParamsWrapperTest, GetBitwidthTest) {
  std::uint32_t bw = 4;
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  BwAxisScaleOffsetQuantizeParamsWrapper wrapper(bw, axis, scales, zero_points);
  ASSERT_EQ(wrapper.GetBitwidth(), 4);
}
using AxisScaleOffsetParamsWithType =
    std::tuple<int32_t, std::vector<float>, std::vector<int32_t>, int32_t,
               std::vector<float>, std::vector<int32_t>>;
class AxisScaleOffsetEqualTestWithType
    : public ::testing::TestWithParam<AxisScaleOffsetParamsWithType> {};
TEST_P(AxisScaleOffsetEqualTestWithType, EqualityOperator) {
  const auto& [axis1, scale1, offsets1, axis2, scales2, offsets2] = GetParam();

  qnn::AxisScaleOffsetQuantizeParamsWrapper wrapper1(axis1, scale1, offsets1);
  qnn::AxisScaleOffsetQuantizeParamsWrapper wrapper2(axis2, scales2, offsets2);
  qnn::AxisScaleOffsetQuantizeParamsWrapper wrapper3(wrapper2);

  const bool expected_equal =
      (axis1 == axis2) && (scale1 == scales2) && (offsets1 == offsets2);
  EXPECT_EQ(wrapper1 == wrapper2, expected_equal);
  EXPECT_EQ(wrapper1 == wrapper3, expected_equal);
  EXPECT_TRUE(wrapper2 == wrapper3);
}
// Data type, axis, scales and zero_points
INSTANTIATE_TEST_SUITE_P(
    AxisScaleOffsetQuantizeParamsWrapperTest_Combine,
    AxisScaleOffsetEqualTestWithType,
    ::testing::Combine(::testing::Values(1),
                       ::testing::Values(std::vector<float>{1.5f, 2.5f}),
                       ::testing::Values(std::vector<int32_t>{10, 20}),
                       ::testing::Values(1, 0),
                       ::testing::Values(std::vector<float>{1.5f, 2.5f},
                                         std::vector<float>{1.5f, 2.6f}),
                       ::testing::Values(std::vector<int32_t>{10, 20},
                                         std::vector<int32_t>{10, 30})));

using BwScaleOffsetParams =
    std::tuple<uint32_t, float, int32_t, uint32_t, float, int32_t>;
class BwScaleOffsetEqualTest
    : public ::testing::TestWithParam<BwScaleOffsetParams> {};
TEST_P(BwScaleOffsetEqualTest, EqualityOperator) {
  const auto& [bitwidth1, scale1, offset1, bitwidth2, scale2, offset2] =
      GetParam();

  qnn::BwScaleOffsetQuantizeParamsWrapper wrapper1(bitwidth1, scale1, offset1);
  qnn::BwScaleOffsetQuantizeParamsWrapper wrapper2(bitwidth2, scale2, offset2);
  qnn::BwScaleOffsetQuantizeParamsWrapper wrapper3(wrapper2);

  const bool expected_equal =
      (bitwidth1 == bitwidth2) && (scale1 == scale2) && (offset1 == offset2);
  EXPECT_EQ(wrapper1 == wrapper2, expected_equal);
  EXPECT_EQ(wrapper1 == wrapper3, expected_equal);
  EXPECT_TRUE(wrapper2 == wrapper3);
}
// Bit width, scale and offset
INSTANTIATE_TEST_SUITE_P(
    BwScaleOffsetQuantizeParamsWrapperTest_Combine, BwScaleOffsetEqualTest,
    ::testing::Combine(::testing::Values(2u, 4u), ::testing::Values(1.5f),
                       ::testing::Values(10), ::testing::Values(2u, 4u),
                       ::testing::Values(1.5f, 1.6f),
                       ::testing::Values(10, 12)));

using BwAxisScaleOffsetParams =
    std::tuple<uint32_t, int32_t, std::vector<float>, std::vector<int32_t>,
               uint32_t, int32_t, std::vector<float>, std::vector<int32_t>>;
class BwAxisScaleOffsetEqualTest
    : public ::testing::TestWithParam<BwAxisScaleOffsetParams> {};
TEST_P(BwAxisScaleOffsetEqualTest, EqualityOperator) {
  const auto& [bitwidth1, axis1, scales1, offsets1, bitwidth2, axis2, scales2,
               offsets2] = GetParam();

  qnn::BwAxisScaleOffsetQuantizeParamsWrapper wrapper1(bitwidth1, axis1,
                                                       scales1, offsets1);
  qnn::BwAxisScaleOffsetQuantizeParamsWrapper wrapper2(bitwidth2, axis2,
                                                       scales2, offsets2);
  qnn::BwAxisScaleOffsetQuantizeParamsWrapper wrapper3(wrapper2);

  const bool expected_equal = (bitwidth1 == bitwidth2) && (axis1 == axis2) &&
                              (scales1 == scales2) && (offsets1 == offsets2);
  EXPECT_EQ(wrapper1 == wrapper2, expected_equal);
  EXPECT_EQ(wrapper1 == wrapper3, expected_equal);
  EXPECT_TRUE(wrapper2 == wrapper3);
}
// Bit width, axis, scales and offsets
INSTANTIATE_TEST_SUITE_P(
    BwAxisScaleOffsetQuantizeParamsWrapperTest_Combine,
    BwAxisScaleOffsetEqualTest,
    ::testing::Combine(::testing::Values(4u), ::testing::Values(1),
                       ::testing::Values(std::vector<float>{1.5f, 2.5f}),
                       ::testing::Values(std::vector<int32_t>{10, 20}),
                       ::testing::Values(4u, 2u), ::testing::Values(1, 0),
                       ::testing::Values(std::vector<float>{1.5f, 2.5f},
                                         std::vector<float>{1.5f, 2.6f}),
                       ::testing::Values(std::vector<int32_t>{10, 20},
                                         std::vector<int32_t>{10, 30})));

}  // namespace
}  // namespace qnn
