/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/backends/xnnpack/dequantize.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"

namespace litert::tensor {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::FloatEq;
using ::testing::Pointwise;

// Dequantizes `raw`, the bytes of a constant tensor described by `info`.
template <class T>
absl::StatusOr<std::vector<float>> Dequantize(
    const graph::TensorInformation& info, const std::vector<T>& raw) {
  return DequantizeConstantTensor(
      info, absl::MakeSpan(reinterpret_cast<const std::byte*>(raw.data()),
                           raw.size() * sizeof(T)));
}

graph::TensorInformation PerChannelInfo(Type type, graph::Shape shape,
                                        std::vector<float> scales,
                                        std::vector<int64_t> zero_points,
                                        int quantized_dimension = 0) {
  return graph::TensorInformation{
      .type = type,
      .shape = std::move(shape),
      .quantization = std::make_shared<graph::PerChannelAffineQuantization>(
          std::move(scales), std::move(zero_points), quantized_dimension),
  };
}

graph::TensorInformation BlockwiseInfo(Type type, graph::Shape shape,
                                       std::vector<float> scales,
                                       std::vector<int64_t> zero_points,
                                       int block_size,
                                       int quantized_dimension) {
  return graph::TensorInformation{
      .type = type,
      .shape = std::move(shape),
      .quantization = std::make_shared<graph::BlockwiseQuantization>(
          std::move(scales), std::move(zero_points), block_size,
          quantized_dimension),
  };
}

TEST(XnnpackDequantizeTest,
     DequantizeConstantTensorRejectsMissingQuantization) {
  graph::TensorInformation info{
      .type = Type::kI8,
      .shape = {2, 2},
      .quantization = nullptr,
  };
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{1, 2, 3, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorRejectsFloats) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kFP32, {2}, {0.5f}, {});
  EXPECT_THAT(Dequantize(info, std::vector<float>{1.0f, 2.0f}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorRejectsUnsignedIntegers) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kU8, {2}, {0.5f}, {128});
  EXPECT_THAT(Dequantize(info, std::vector<uint8_t>{200, 10}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorRejectsPerTensor) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI8, {2, 2}, {0.5f}, {1});
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{3, 5, -1, 1}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorMultiChannelZeroPoints) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI8, {2, 2}, {0.5f, 2.0f}, {1, -2});
  // Channel 0: (3 - 1)*0.5 = 1.0, (5 - 1)*0.5 = 2.0
  // Channel 1: (0 - (-2))*2.0 = 4.0, (2 - (-2))*2.0 = 8.0
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{3, 5, 0, 2}),
              IsOkAndHolds(Pointwise(FloatEq(), {1.0f, 2.0f, 4.0f, 8.0f})));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorBroadcastsSingleZeroPoint) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI8, {2, 2}, {0.5f, 2.0f}, {1});
  // Channel 0: (3 - 1)*0.5 = 1.0, (5 - 1)*0.5 = 2.0
  // Channel 1: (2 - 1)*2.0 = 2.0, (4 - 1)*2.0 = 6.0
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{3, 5, 2, 4}),
              IsOkAndHolds(Pointwise(FloatEq(), {1.0f, 2.0f, 2.0f, 6.0f})));
}

TEST(XnnpackDequantizeTest,
     DequantizeConstantTensorEmptyZeroPointsDefaultsToZero) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI8, {2, 2}, {0.5f, 2.0f}, {});
  // Channel 0: (2 - 0)*0.5 = 1.0, (4 - 0)*0.5 = 2.0
  // Channel 1: (1 - 0)*2.0 = 2.0, (3 - 0)*2.0 = 6.0
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{2, 4, 1, 3}),
              IsOkAndHolds(Pointwise(FloatEq(), {1.0f, 2.0f, 2.0f, 6.0f})));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorSecondQuantizedDimension) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI8, {2, 3}, {0.5f, 0.25f, 2.0f}, {1, 1, 1},
                     /*quantized_dimension=*/1);
  // The channels are the columns.
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{1, 2, 3, 4, 5, 6}),
              IsOkAndHolds(Pointwise(FloatEq(),
                                     {0.0f, 0.25f, 4.0f, 1.5f, 1.0f, 10.0f})));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorRank3) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI8, {2, 2, 2}, {1.0f, 10.0f}, {},
                     /*quantized_dimension=*/1);
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8}),
              IsOkAndHolds(Pointwise(FloatEq(), {1.0f, 2.0f, 30.0f, 40.0f, 5.0f,
                                                 6.0f, 70.0f, 80.0f})));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorInt4) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI4, {2, 2}, {0.5f, 0.25f}, {1, -1});
  // Channel 0: (1 - 1)*0.5 = 0.0, (2 - 1)*0.5 = 0.5
  // Channel 1: (-8 + 1)*0.25 = -1.75, (-1 + 1)*0.25 = 0.0
  EXPECT_THAT(Dequantize(info, std::vector<int4_t>{{1, 2}, {-8, -1}}),
              IsOkAndHolds(Pointwise(FloatEq(), {0.0f, 0.5f, -1.75f, 0.0f})));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorInt4OddSize) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI4, {3}, {1.0f, 2.0f, 4.0f}, {});
  EXPECT_THAT(Dequantize(info, std::vector<int4_t>{{1, 2}, {3, 0}}),
              IsOkAndHolds(Pointwise(FloatEq(), {1.0f, 4.0f, 12.0f})));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorInt2) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI2, {4}, {0.5f, 0.5f, 0.5f, 0.5f}, {});
  EXPECT_THAT(Dequantize(info, std::vector<int2_t>{{0, 1, -2, -1}}),
              IsOkAndHolds(Pointwise(FloatEq(), {0.0f, 0.5f, -1.0f, -0.5f})));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorRejectsBlockwise) {
  const graph::TensorInformation info =
      BlockwiseInfo(Type::kI8, {2, 4}, {1.0f, 2.0f, 4.0f, 8.0f}, {1, 0, -1, 2},
                    /*block_size=*/2, /*quantized_dimension=*/0);
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(XnnpackDequantizeTest,
     DequantizeConstantTensorRejectsBlockSizeThatDoesntDivide) {
  const graph::TensorInformation info =
      BlockwiseInfo(Type::kI8, {2, 3}, {1.0f, 1.0f}, {}, /*block_size=*/2,
                    /*quantized_dimension=*/0);
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{1, 2, 3, 4, 5, 6}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorRejectsRank3Blockwise) {
  const graph::TensorInformation info =
      BlockwiseInfo(Type::kI8, {1, 2, 2}, {1.0f, 1.0f}, {}, /*block_size=*/2,
                    /*quantized_dimension=*/0);
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{1, 2, 3, 4}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorRejectsWrongScaleCount) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI8, {2, 2}, {0.5f, 0.25f, 0.1f}, {1});
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{1, 2, 3, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(XnnpackDequantizeTest,
     DequantizeConstantTensorRejectsWrongZeroPointCount) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI8, {3, 2}, {0.5f, 0.25f, 0.1f}, {1, 2});
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{1, 2, 3, 4, 5, 6}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(XnnpackDequantizeTest,
     DequantizeConstantTensorRejectsOutOfRangeQuantizedDimension) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI8, {2, 2}, {0.5f, 0.25f}, {},
                     /*quantized_dimension=*/2);
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{1, 2, 3, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorRejectsSmallRawData) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI8, {2, 2}, {0.5f, 0.25f}, {0, 0});
  // Expects 4 bytes.
  EXPECT_THAT(Dequantize(info, std::vector<int8_t>{1, 2, 3}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(XnnpackDequantizeTest, DequantizeConstantTensorRejectsSmallPackedRawData) {
  const graph::TensorInformation info =
      PerChannelInfo(Type::kI4, {5}, {1.0f}, {});
  // Expects 3 bytes.
  EXPECT_THAT(Dequantize(info, std::vector<int4_t>{{1, 2}, {3, 4}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::tensor
