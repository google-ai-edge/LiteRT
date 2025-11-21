// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"

#include <cstddef>
#include <cstdint>
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
  UndefinedQuantizeParamsWrapper wrapper2(wrapper1);
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

TEST(ScaleOffsetQuantizeParamsWrapperTest, QnnConstructorTest) {
  ScaleOffsetQuantizeParamsWrapper wrapper1(1.5f, 10);
  Qnn_QuantizeParams_t dst1 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst1);
  ScaleOffsetQuantizeParamsWrapper wrapper2(dst1.scaleOffsetEncoding);
  Qnn_QuantizeParams_t dst2 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst2);
  EXPECT_EQ(dst1.encodingDefinition, dst2.encodingDefinition);
  EXPECT_EQ(dst1.quantizationEncoding, dst2.quantizationEncoding);
  EXPECT_FLOAT_EQ(dst1.scaleOffsetEncoding.scale,
                  dst2.scaleOffsetEncoding.scale);
  EXPECT_EQ(dst1.scaleOffsetEncoding.offset, dst2.scaleOffsetEncoding.offset);
}

TEST(ScaleOffsetQuantizeParamsWrapperTest, EqualityOperatorTest) {
  float scale1 = 1.5f;
  std::int32_t zero_point1 = 10;
  ScaleOffsetQuantizeParamsWrapper wrapper1(scale1, zero_point1);
  float scale2 = 1.6f;
  std::int32_t zero_point2 = 10;
  ScaleOffsetQuantizeParamsWrapper wrapper2(scale2, zero_point2);
  ScaleOffsetQuantizeParamsWrapper wrapper3(wrapper2);
  EXPECT_FALSE(wrapper1 == wrapper2);
  EXPECT_FALSE(wrapper1 == wrapper3);
  EXPECT_TRUE(wrapper2 == wrapper3);
}

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

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, QnnConstructorTest) {
  std::int32_t axis = 1;
  std::vector<float> scales = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper1(axis, scales, zero_points);
  Qnn_QuantizeParams_t dst1 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst1);
  AxisScaleOffsetQuantizeParamsWrapper wrapper2(dst1.axisScaleOffsetEncoding);
  Qnn_QuantizeParams_t dst2 = QNN_QUANTIZE_PARAMS_INIT;
  wrapper1.CloneTo(dst2);
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

TEST(AxisScaleOffsetQuantizeParamsWrapperTest, EqualityOperatorTest) {
  std::int32_t axis1 = 1;
  std::vector<float> scales1 = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points1 = {10, 20};
  AxisScaleOffsetQuantizeParamsWrapper wrapper1(axis1, scales1, zero_points1);
  std::int32_t axis2 = 1;
  std::vector<float> scales2 = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points2 = {10, 30};
  AxisScaleOffsetQuantizeParamsWrapper wrapper2(axis2, scales2, zero_points2);
  AxisScaleOffsetQuantizeParamsWrapper wrapper3(wrapper2);
  EXPECT_FALSE(wrapper1 == wrapper2);
  EXPECT_FALSE(wrapper1 == wrapper3);
  EXPECT_TRUE(wrapper2 == wrapper3);
}

TEST(BwScaleOffsetQuantizeParamsWrapperTest, EqualityOperatorTest) {
  std::uint32_t bitwidth1 = 32;
  float scale1 = 1.5f;
  std::int32_t zero_point1 = 10;
  BwScaleOffsetQuantizeParamsWrapper wrapper1(bitwidth1, scale1, zero_point1);
  std::uint32_t bitwidth2 = 64;
  float scale2 = 1.5f;
  std::int32_t zero_point2 = 10;
  BwScaleOffsetQuantizeParamsWrapper wrapper2(bitwidth2, scale2, zero_point2);
  BwScaleOffsetQuantizeParamsWrapper wrapper3(wrapper2);
  EXPECT_FALSE(wrapper1 == wrapper2);
  EXPECT_FALSE(wrapper1 == wrapper3);
  EXPECT_TRUE(wrapper2 == wrapper3);
}

TEST(BwAxisScaleOffsetQuantizeParamsWrapperTest, EqualityOperatorTest) {
  std::uint32_t bitwidth1 = 32;
  std::int32_t axis1 = 1;
  std::vector<float> scales1 = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points1 = {10, 20};
  BwAxisScaleOffsetQuantizeParamsWrapper wrapper1(bitwidth1, axis1, scales1,
                                                  zero_points1);
  std::uint32_t bitwidth2 = 32;
  std::int32_t axis2 = 1;
  std::vector<float> scales2 = {1.5f, 2.5f};
  std::vector<std::int32_t> zero_points2 = {10, 30};
  BwAxisScaleOffsetQuantizeParamsWrapper wrapper2(bitwidth2, axis2, scales2,
                                                  zero_points2);
  BwAxisScaleOffsetQuantizeParamsWrapper wrapper3(wrapper2);
  EXPECT_FALSE(wrapper1 == wrapper2);
  EXPECT_FALSE(wrapper1 == wrapper3);
  EXPECT_TRUE(wrapper2 == wrapper3);
}

}  // namespace
}  // namespace qnn
