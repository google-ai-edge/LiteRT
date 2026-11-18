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

#include "litert/core/model/ops/simple_unary.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/reference_numeric_utils.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/test/matchers.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::litert::IsError;

TEST(SimpleUnaryOpTest, AbsStaticShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferAbs(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleUnaryOpTest, AbsDynamicShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{-1, 128}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferAbs(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(-1, 128));
}

// Consolidating Cast tests
TEST(SimpleUnaryOpTest, CastStaticShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferCast(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleUnaryOpTest, CastDynamicShape) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{-1, 128}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferCast(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(-1, 128));
}

TEST(SimpleUnaryOpTest, L2Normalization) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 128}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferL2Normalization(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 128));
}

TEST(SimpleUnaryOpTest, ReverseV2) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {1}};  // Input, Axis
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferReverseV2(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(SimpleUnaryOpTest, ReferenceTanh) {
  std::vector<float> input = {0.0f, 1.0f, -1.0f};
  std::vector<float> output(3);

  ReferenceTanh(input.data(), input.size(), output.data());

  EXPECT_NEAR(output[0], 0.0f, 1e-5);
  EXPECT_NEAR(output[1], std::tanh(1.0f), 1e-5);
  EXPECT_NEAR(output[2], std::tanh(-1.0f), 1e-5);
}

TEST(SimpleUnaryOpTest, ReferenceRsqrt) {
  std::vector<float> input = {1.0f, 4.0f, 16.0f};
  std::vector<float> output(3);

  ReferenceRsqrt(input.data(), input.size(), output.data());

  EXPECT_NEAR(output[0], 1.0f, 1e-5);
  EXPECT_NEAR(output[1], 0.5f, 1e-5);
  EXPECT_NEAR(output[2], 0.25f, 1e-5);
}

TEST(SimpleUnaryOpTest, ReferenceSin) {
  std::vector<float> input = {0.0f, 1.0f, 2.0f};
  std::vector<float> output(3);

  ReferenceSin(input.data(), input.size(), output.data());

  EXPECT_NEAR(output[0], 0.0f, 1e-5);
  EXPECT_NEAR(output[1], std::sin(1.0f), 1e-5);
  EXPECT_NEAR(output[2], std::sin(2.0f), 1e-5);
}

TEST(SimpleUnaryOpTest, ReferenceCos) {
  std::vector<float> input = {0.0f, 1.0f, 2.0f};
  std::vector<float> output(3);

  ReferenceCos(input.data(), input.size(), output.data());

  EXPECT_NEAR(output[0], 1.0f, 1e-5);
  EXPECT_NEAR(output[1], std::cos(1.0f), 1e-5);
  EXPECT_NEAR(output[2], std::cos(2.0f), 1e-5);
}

TEST(SimpleUnaryOpTest, ReferenceCast) {
  std::vector<int32_t> in_i32 = {1, 2, -3};
  std::vector<float> out_f32(3);
  ReferenceCast(in_i32.data(), in_i32.size(), out_f32.data());
  EXPECT_THAT(out_f32, ElementsAre(1.0f, 2.0f, -3.0f));

  std::vector<float> in_f32 = {1.5f, 2.7f, -3.2f};
  std::vector<int32_t> out_i32(3);
  ReferenceCast(in_f32.data(), in_f32.size(), out_i32.data());
  EXPECT_THAT(out_i32, ElementsAre(1, 2, -3));
}

TEST(SimpleUnaryOpTest, ReferenceDequantize) {
  std::vector<int32_t> in = {-10, 0, 10, 20};
  std::vector<float> out(4);
  ReferenceDequantize(in.data(), in.size(), out.data(), /*scale=*/0.5f,
                      /*zero_point=*/10);
  EXPECT_THAT(out, ElementsAre(-10.0f, -5.0f, 0.0f, 5.0f));
}

TEST(SimpleUnaryOpTest, ReferenceDequantizePerChannel) {
  std::vector<int32_t> in = {-2, 2, -1, 3};
  std::vector<int32_t> dims = {2, 2};
  std::vector<float> scales = {0.5f, 2.0f};
  std::vector<int64_t> zero_points = {0, 1};
  std::vector<float> out(4);
  ReferenceDequantizePerChannel(in.data(), dims.data(), /*rank=*/2,
                                /*quantized_dimension=*/0, scales.data(),
                                zero_points.data(), out.data());
  EXPECT_THAT(out, ElementsAre(-1.0f, 1.0f, -4.0f, 4.0f));
}

TEST(SimpleUnaryOpTest, AbsEmptyInputsFailure) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {};
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferAbs(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SimpleUnaryOpTest,
     ReferenceLeakyReluQuantizedScalesNegativesAndRequantizes) {
  // input: scale=0.5, zp=0 -> vals = {-4.0, -1.0, 0.0, 2.0}
  // alpha = 0.25 -> {-1.0, -0.25, 0.0, 2.0}
  // output: scale=0.5, zp=1 -> {-2+1=-1, llround(-0.5)+1=-1+1=0, 1, 5}
  const std::vector<int8_t> in = {-8, -2, 0, 4};
  std::vector<int8_t> out(4, 0);

  ReferenceLeakyReluQuantized(in.data(), out.data(), in.size(), /*alpha=*/0.25f,
                              AffineQuantSpec{0.5f, 0},
                              AffineQuantSpec{0.5f, 1});

  EXPECT_THAT(out, ElementsAre(-1, 0, 1, 5));
}

TEST(SimpleUnaryOpTest, ReferenceLeakyReluFloatScalesNegativesByAlpha) {
  const std::vector<float> in = {-4.0f, -1.0f, 0.0f, 2.5f};
  std::vector<float> out(4, 0.0f);

  ReferenceLeakyRelu(in.data(), out.data(), in.size(), /*alpha=*/0.25f);

  EXPECT_THAT(out, ElementsAre(-1.0f, -0.25f, 0.0f, 2.5f));
}

TEST(SimpleUnaryOpTest, ReferenceLogicalNotInvertsEachBooleanElement) {
  const bool in[] = {true, false, true, false};
  bool out[4] = {};

  ReferenceLogicalNot(in, out, 4);

  EXPECT_THAT(out, ElementsAre(false, true, false, true));
}

TEST(SimpleUnaryOpTest, ReferenceQuantizeFloatToInt8Saturates) {
  const std::vector<float> in = {-100.0f, -0.75f, 0.0f, 0.75f, 100.0f};
  std::vector<int8_t> out(5, 0);

  ReferenceQuantize(in.data(), in.size(), /*input_scale=*/1.0f,
                    /*input_zero_point=*/0, /*output_scale=*/0.5f,
                    /*output_zero_point=*/10, out.data());

  EXPECT_THAT(out, ElementsAre(-128, 8, 10, 12, 127));
}

TEST(SimpleUnaryOpTest,
     ReferenceQuantizeInt8ToUint8AppliesDistinctScaleAndZeroPoint) {
  // real = (in - 2) * 0.5 -> {-1.0, 0.0, 1.0}; q_out = real / 0.25 + 10.
  const std::vector<int8_t> in = {0, 2, 4};
  std::vector<uint8_t> out(3, 0);

  ReferenceQuantize(in.data(), in.size(), /*input_scale=*/0.5f,
                    /*input_zero_point=*/2, /*output_scale=*/0.25f,
                    /*output_zero_point=*/10, out.data());

  EXPECT_THAT(out, ElementsAre(6, 10, 14));
}

struct ReferenceReverseV2AxisCase {
  std::string test_name;
  std::vector<int32_t> axes;
  std::vector<int32_t> expected;
};

using ReferenceReverseV2AxisTest =
    ::testing::TestWithParam<ReferenceReverseV2AxisCase>;

TEST_P(ReferenceReverseV2AxisTest, ReversesSpecifiedAxes) {
  const auto& tc = GetParam();
  const std::vector<int32_t> in = {1, 2, 3, 4, 5, 6};
  const int32_t dims[] = {2, 3};
  std::vector<int32_t> out(6, 0);

  ReferenceReverseV2(in.data(), dims, /*rank=*/2, tc.axes.data(),
                     static_cast<int>(tc.axes.size()), out.data());

  EXPECT_THAT(out, ElementsAreArray(tc.expected));
}

INSTANTIATE_TEST_SUITE_P(
    ReferenceReverseV2AxisTests, ReferenceReverseV2AxisTest,
    ::testing::Values(
        ReferenceReverseV2AxisCase{"AxisZero", {0}, {4, 5, 6, 1, 2, 3}},
        ReferenceReverseV2AxisCase{
            "NegativeLastAxis", {-1}, {3, 2, 1, 6, 5, 4}},
        ReferenceReverseV2AxisCase{"BothAxes", {0, 1}, {6, 5, 4, 3, 2, 1}},
        ReferenceReverseV2AxisCase{
            "EmptyAxesCopiesUnchanged", {}, {1, 2, 3, 4, 5, 6}}),
    [](const ::testing::TestParamInfo<ReferenceReverseV2AxisCase>& info) {
      return info.param.test_name;
    });

TEST(SimpleUnaryOpTest, ReferenceReverseV2RankZeroCopiesScalar) {
  const int32_t scalar = 19;
  int32_t out = 0;

  ReferenceReverseV2(&scalar, nullptr, /*rank=*/0, nullptr, 0, &out);

  EXPECT_EQ(out, 19);
}

struct ReferenceReverseV2InvalidShapeCase {
  std::string test_name;
  std::vector<int32_t> dims = {2, 3};
  int rank = 2;
};

using ReferenceReverseV2InvalidShapeTest =
    ::testing::TestWithParam<ReferenceReverseV2InvalidShapeCase>;

TEST_P(ReferenceReverseV2InvalidShapeTest, LeavesOutputUntouchedOnInvalidArgs) {
  const auto& tc = GetParam();
  const std::vector<int32_t> in = {1, 2, 3, 4, 5, 6};
  const int32_t axes[] = {0};
  std::vector<int32_t> out(6, -33);

  ReferenceReverseV2<int32_t>(in.data(), tc.dims.data(), tc.rank, axes,
                              /*num_axes=*/1, out.data());

  EXPECT_THAT(out, ElementsAre(-33, -33, -33, -33, -33, -33));
}

INSTANTIATE_TEST_SUITE_P(
    ReferenceReverseV2InvalidShapeTests, ReferenceReverseV2InvalidShapeTest,
    ::testing::Values(
        ReferenceReverseV2InvalidShapeCase{.test_name = "NegativeRank",
                                           .rank = -1},
        ReferenceReverseV2InvalidShapeCase{.test_name = "NegativeDim",
                                           .dims = {-1, 3}}),
    [](const ::testing::TestParamInfo<ReferenceReverseV2InvalidShapeCase>&
           info) { return info.param.test_name; });

TEST(SimpleUnaryOpTest, ReferenceReverseV2EmptyTensorIsNoOp) {
  const int32_t in[] = {1};
  const int32_t dims[] = {0, 3};
  const int32_t axes[] = {0};
  std::vector<int32_t> out(6, -33);

  ReferenceReverseV2<int32_t>(in, dims, /*rank=*/2, axes, /*num_axes=*/1,
                              out.data());

  EXPECT_THAT(out, ElementsAre(-33, -33, -33, -33, -33, -33));
}

TEST(SimpleUnaryOpTest,
     ReferenceReverseV2RankZeroNullInputLeavesOutputUntouched) {
  int32_t out = -33;

  ReferenceReverseV2<int32_t>(/*input_data=*/nullptr, /*dims=*/nullptr,
                              /*rank=*/0, /*axes=*/nullptr, /*num_axes=*/0,
                              &out);

  EXPECT_EQ(out, -33);
}

TEST(SimpleUnaryOpTest, ReferenceReverseV2NullInputDimsLeavesOutputUntouched) {
  const std::vector<int32_t> in = {1, 2, 3, 4, 5, 6};
  const int32_t axes[] = {0};
  std::vector<int32_t> out(6, -33);

  ReferenceReverseV2<int32_t>(in.data(), /*dims=*/nullptr, /*rank=*/2, axes,
                              /*num_axes=*/1, out.data());

  EXPECT_THAT(out, ElementsAre(-33, -33, -33, -33, -33, -33));
}

TEST(SimpleUnaryOpTest, ReferenceReverseV2NullAxesLeavesOutputUntouched) {
  const std::vector<int32_t> in = {1, 2, 3, 4, 5, 6};
  const int32_t dims[] = {2, 3};
  std::vector<int32_t> out(6, -33);

  ReferenceReverseV2<int32_t>(in.data(), dims, /*rank=*/2, /*axes=*/nullptr,
                              /*num_axes=*/1, out.data());

  EXPECT_THAT(out, ElementsAre(-33, -33, -33, -33, -33, -33));
}

TEST(SimpleUnaryOpTest, ReferenceReverseV2NullInputDataLeavesOutputUntouched) {
  const int32_t dims[] = {2, 3};
  const int32_t axes[] = {0};
  std::vector<int32_t> out(6, -33);

  ReferenceReverseV2<int32_t>(/*input_data=*/nullptr, dims, /*rank=*/2, axes,
                              /*num_axes=*/1, out.data());

  EXPECT_THAT(out, ElementsAre(-33, -33, -33, -33, -33, -33));
}

TEST(SimpleUnaryOpTest,
     ReferenceReverseV2NullOutputDataReturnsInvalidArgument) {
  const std::vector<int32_t> in = {1, 2, 3, 4, 5, 6};
  const int32_t dims[] = {2, 3};
  const int32_t axes[] = {0};

  EXPECT_THAT(ReferenceReverseV2<int32_t>(in.data(), dims, /*rank=*/2, axes,
                                          /*num_axes=*/1,
                                          /*output_data=*/nullptr),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(SimpleUnaryOpTest,
     ReferenceReverseV2OutOfRangeAxisReturnsInvalidArgument) {
  const std::vector<int32_t> in = {1, 2, 3, 4, 5, 6};
  const int32_t dims[] = {2, 3};
  const int32_t axes[] = {2};
  std::vector<int32_t> out(6, -33);

  EXPECT_THAT(ReferenceReverseV2<int32_t>(in.data(), dims, /*rank=*/2, axes,
                                          /*num_axes=*/1, out.data()),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out, ElementsAre(-33, -33, -33, -33, -33, -33));
}

TEST(SimpleUnaryOpTest,
     ReferenceDequantizePerChannelScalesNegativeAxisWithNullZeroPoints) {
  const std::vector<int32_t> in = {2, 4, -2, -4};
  const int32_t dims[] = {2, 2};
  const std::vector<float> scales = {0.5f, 2.0f};
  std::vector<float> out(4, 0.0f);

  ReferenceDequantizePerChannel(in.data(), dims, /*rank=*/2,
                                /*quantized_dimension=*/-1, scales.data(),
                                /*zero_points=*/nullptr, out.data());

  EXPECT_THAT(out, ElementsAre(1.0f, 8.0f, -1.0f, -8.0f));
}

struct DequantizePerChannelInvalidShapeCase {
  std::string test_name;
  std::vector<int32_t> dims = {2, 2};
  int rank = 2;
  int32_t quantized_dimension = 0;
};

using ReferenceDequantizePerChannelInvalidShapeTest =
    ::testing::TestWithParam<DequantizePerChannelInvalidShapeCase>;

TEST_P(ReferenceDequantizePerChannelInvalidShapeTest,
       LeavesOutputUntouchedOnInvalidArgs) {
  const auto& tc = GetParam();
  const std::vector<int32_t> in = {2, 4, -2, -4};
  const std::vector<float> scales = {0.5f, 2.0f};
  std::vector<float> out(4, -5.0f);

  ReferenceDequantizePerChannel<int32_t>(in.data(), tc.dims.data(), tc.rank,
                                         tc.quantized_dimension, scales.data(),
                                         /*zero_points=*/nullptr, out.data());

  EXPECT_THAT(out, ElementsAre(-5.0f, -5.0f, -5.0f, -5.0f));
}

INSTANTIATE_TEST_SUITE_P(
    ReferenceDequantizePerChannelInvalidShapeTests,
    ReferenceDequantizePerChannelInvalidShapeTest,
    ::testing::Values(
        DequantizePerChannelInvalidShapeCase{.test_name = "ZeroRank",
                                             .rank = 0},
        DequantizePerChannelInvalidShapeCase{
            .test_name = "OutOfRangeQuantizedDimension",
            .quantized_dimension = 5}),
    [](const ::testing::TestParamInfo<DequantizePerChannelInvalidShapeCase>&
           info) { return info.param.test_name; });

TEST(SimpleUnaryOpTest, ReferenceDequantizePerChannelEmptyTensorIsNoOp) {
  const int32_t in[] = {2};
  const int32_t dims[] = {0, 2};
  const std::vector<float> scales = {0.5f, 2.0f};
  std::vector<float> out(4, -5.0f);

  ReferenceDequantizePerChannel<int32_t>(
      in, dims, /*rank=*/2, /*quantized_dimension=*/1, scales.data(),
      /*zero_points=*/nullptr, out.data());

  EXPECT_THAT(out, ElementsAre(-5.0f, -5.0f, -5.0f, -5.0f));
}

TEST(SimpleUnaryOpTest,
     ReferenceDequantizePerChannelNullDimsLeavesOutputUntouched) {
  const std::vector<int32_t> in = {2, 4, -2, -4};
  const std::vector<float> scales = {0.5f, 2.0f};
  std::vector<float> out(4, -5.0f);

  ReferenceDequantizePerChannel<int32_t>(
      in.data(), /*dims=*/nullptr, /*rank=*/2, /*quantized_dimension=*/0,
      scales.data(), /*zero_points=*/nullptr, out.data());

  EXPECT_THAT(out, ElementsAre(-5.0f, -5.0f, -5.0f, -5.0f));
}

TEST(SimpleUnaryOpTest,
     ReferenceDequantizePerChannelNullInputDataLeavesOutputUntouched) {
  const int32_t dims[] = {2, 2};
  const std::vector<float> scales = {0.5f, 2.0f};
  std::vector<float> out(4, -5.0f);

  ReferenceDequantizePerChannel<int32_t>(
      /*input_data=*/nullptr, dims, /*rank=*/2, /*quantized_dimension=*/0,
      scales.data(), /*zero_points=*/nullptr, out.data());

  EXPECT_THAT(out, ElementsAre(-5.0f, -5.0f, -5.0f, -5.0f));
}

TEST(SimpleUnaryOpTest,
     ReferenceDequantizePerChannelNullScalesLeavesOutputUntouched) {
  const std::vector<int32_t> in = {2, 4, -2, -4};
  const int32_t dims[] = {2, 2};
  std::vector<float> out(4, -5.0f);

  ReferenceDequantizePerChannel<int32_t>(
      in.data(), dims, /*rank=*/2, /*quantized_dimension=*/0,
      /*scales=*/nullptr, /*zero_points=*/nullptr, out.data());

  EXPECT_THAT(out, ElementsAre(-5.0f, -5.0f, -5.0f, -5.0f));
}

TEST(SimpleUnaryOpTest, ReferenceDequantizePerChannelNullOutputDataIsNoOp) {
  const std::vector<int32_t> in = {2, 4, -2, -4};
  const int32_t dims[] = {2, 2};
  const std::vector<float> scales = {0.5f, 2.0f};

  ReferenceDequantizePerChannel<int32_t>(
      in.data(), dims, /*rank=*/2, /*quantized_dimension=*/0, scales.data(),
      /*zero_points=*/nullptr, /*output_data=*/nullptr);
}

}  // namespace
}  // namespace litert::internal
