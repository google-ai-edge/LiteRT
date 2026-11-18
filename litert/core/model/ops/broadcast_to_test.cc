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

#include "litert/core/model/ops/broadcast_to.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/tensor_layout_util.h"
#include "litert/test/matchers.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::litert::IsError;
using ::testing::litert::IsOk;

template <typename T>
const T* DataOrNull(const std::optional<std::vector<T>>& values) {
  return values.has_value() ? values->data() : nullptr;
}

int RankOrZero(const std::optional<std::vector<int32_t>>& dims) {
  return dims.has_value() ? static_cast<int>(dims->size()) : 0;
}

TEST(BroadcastToOpTest, SimpleTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&shape_tensor);

  std::vector<int32_t> shape_data = {2, 2};
  SetWeightsFromUnownedBuffer(
      shape_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(shape_data.data()),
                                 shape_data.size() * sizeof(int32_t)));
  shape_tensor.SetType(MakeRankedTensorType(
      kLiteRtElementTypeInt32, {static_cast<int32_t>(shape_data.size())}));

  ASSERT_EQ(InferBroadcastTo(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2));
}

TEST(BroadcastToOpTest, ScalarBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&shape_tensor);

  std::vector<int32_t> shape_data = {3, 1};
  SetWeightsFromUnownedBuffer(
      shape_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(shape_data.data()),
                                 shape_data.size() * sizeof(int32_t)));
  shape_tensor.SetType(MakeRankedTensorType(
      kLiteRtElementTypeInt32, {static_cast<int32_t>(shape_data.size())}));

  ASSERT_EQ(InferBroadcastTo(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(3, 1));
}

TEST(BroadcastToOpTest, InvalidBroadcast) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&shape_tensor);

  std::vector<int32_t> shape_data = {2, 3};
  SetWeightsFromUnownedBuffer(
      shape_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(shape_data.data()),
                                 shape_data.size() * sizeof(int32_t)));
  shape_tensor.SetType(MakeRankedTensorType(
      kLiteRtElementTypeInt32, {static_cast<int32_t>(shape_data.size())}));

  EXPECT_EQ(InferBroadcastTo(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

struct BroadcastToErrorCase {
  std::string test_name;
  bool omit_inputs;
  LiteRtElementType shape_type;
  std::vector<uint8_t> shape_bytes;
};

using BroadcastToErrorTest = ::testing::TestWithParam<BroadcastToErrorCase>;

TEST_P(BroadcastToErrorTest, RejectsMissingInputTensorOrInvalidTargetDims) {
  const auto& param = GetParam();
  std::vector<Dims> input_shapes = {{1, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtOpT op;
  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;
  if (!param.omit_inputs) {
    shape_tensor.SetType(MakeRankedTensorType(param.shape_type, {2}));
    SetWeightsFromOwnedBuffer(
        shape_tensor.Weights(),
        OwningBufferRef<uint8_t>(param.shape_bytes.data(),
                                 param.shape_bytes.size()));
    op.Inputs() = {&input_tensor, &shape_tensor};
  }

  EXPECT_EQ(InferBroadcastTo(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastToErrorTests, BroadcastToErrorTest,
    ::testing::Values(
        BroadcastToErrorCase{
            "MissingInputs", /*omit_inputs=*/true, kLiteRtElementTypeInt32, {}},
        BroadcastToErrorCase{"NegativeDimInInt32Shape", /*omit_inputs=*/false,
                             kLiteRtElementTypeInt32,
                             ToBytes<int32_t>({-2, 2})},
        BroadcastToErrorCase{
            "DimExceedingInt32MaxInInt64Shape",
            /*omit_inputs=*/false, kLiteRtElementTypeInt64,
            ToBytes<int64_t>(
                {2, static_cast<int64_t>(std::numeric_limits<int32_t>::max()) +
                        1})}),
    [](const ::testing::TestParamInfo<BroadcastToErrorCase>& info) {
      return info.param.test_name;
    });

struct ReferenceBroadcastToValidCase {
  std::string test_name;
  std::vector<int32_t> input_data;
  std::optional<std::vector<int32_t>> input_dims;
  std::optional<std::vector<int32_t>> output_dims;
  std::vector<int32_t> expected;
};

using ReferenceBroadcastToValidTest =
    ::testing::TestWithParam<ReferenceBroadcastToValidCase>;

TEST_P(ReferenceBroadcastToValidTest, BroadcastsInputToTargetShape) {
  const auto& tc = GetParam();
  std::vector<int32_t> out(tc.expected.size(), -1);
  const int input_rank = RankOrZero(tc.input_dims);
  const int output_rank = RankOrZero(tc.output_dims);

  const LiteRtStatus status = ReferenceBroadcastTo<int32_t>(
      tc.input_data.data(), DataOrNull(tc.input_dims), input_rank,
      DataOrNull(tc.output_dims), output_rank, out.data());

  EXPECT_THAT(status, IsOk());
  EXPECT_THAT(out, ElementsAreArray(tc.expected));
}

INSTANTIATE_TEST_SUITE_P(
    ReferenceBroadcastToValidTests, ReferenceBroadcastToValidTest,
    ::testing::Values(
        ReferenceBroadcastToValidCase{
            .test_name = "MiddleSingleton3D",
            .input_data = {1, 2, 3, 4, 5, 6},
            .input_dims = std::vector<int32_t>{2, 1, 3},
            .output_dims = std::vector<int32_t>{2, 2, 3},
            .expected = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6}},
        ReferenceBroadcastToValidCase{.test_name = "RankExpansion1DTo2D",
                                      .input_data = {7, 8, 9},
                                      .input_dims = std::vector<int32_t>{3},
                                      .output_dims = std::vector<int32_t>{2, 3},
                                      .expected = {7, 8, 9, 7, 8, 9}},
        ReferenceBroadcastToValidCase{.test_name = "ScalarTo2D",
                                      .input_data = {42},
                                      .input_dims = std::nullopt,
                                      .output_dims = std::vector<int32_t>{2, 2},
                                      .expected = {42, 42, 42, 42}},
        ReferenceBroadcastToValidCase{.test_name = "ScalarToScalar",
                                      .input_data = {42},
                                      .input_dims = std::nullopt,
                                      .output_dims = std::nullopt,
                                      .expected = {42}}),
    [](const ::testing::TestParamInfo<ReferenceBroadcastToValidCase>& info) {
      return info.param.test_name;
    });

TEST(BroadcastToOpTest, ReferenceBroadcastToEmptyOutputAllowsNullDataPointers) {
  const int32_t in_dims[] = {1, 3};
  const int32_t out_dims[] = {0, 3};

  const LiteRtStatus status = ReferenceBroadcastTo<int32_t>(
      /*input_data=*/nullptr, in_dims, /*input_rank=*/2, out_dims,
      /*output_rank=*/2, /*output_data=*/nullptr);

  EXPECT_THAT(status, IsOk());
}

struct ReferenceBroadcastToErrorCase {
  std::string test_name;
  std::optional<std::vector<int32_t>> input_dims = std::vector<int32_t>{1, 3};
  int input_rank = 2;
  std::optional<std::vector<int32_t>> output_dims = std::vector<int32_t>{2, 3};
  int output_rank = 2;
};

using ReferenceBroadcastToErrorTest =
    ::testing::TestWithParam<ReferenceBroadcastToErrorCase>;

TEST_P(ReferenceBroadcastToErrorTest, RejectsInvalidShapeOrRankArguments) {
  const auto& tc = GetParam();
  const std::vector<int32_t> input_data = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> out(16);

  const LiteRtStatus status = ReferenceBroadcastTo<int32_t>(
      input_data.data(), DataOrNull(tc.input_dims), tc.input_rank,
      DataOrNull(tc.output_dims), tc.output_rank, out.data());

  EXPECT_THAT(status, IsError(kLiteRtStatusErrorInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    ReferenceBroadcastToErrorTests, ReferenceBroadcastToErrorTest,
    ::testing::Values(
        ReferenceBroadcastToErrorCase{.test_name = "NegativeInputRank",
                                      .input_rank = -1},
        ReferenceBroadcastToErrorCase{.test_name = "NegativeOutputRank",
                                      .output_rank = -1},
        ReferenceBroadcastToErrorCase{.test_name = "NullInputDims",
                                      .input_dims = std::nullopt,
                                      .input_rank = 1},
        ReferenceBroadcastToErrorCase{.test_name = "NullOutputDims",
                                      .output_dims = std::nullopt,
                                      .output_rank = 2},
        ReferenceBroadcastToErrorCase{
            .test_name = "IncompatibleShape",
            .input_dims = std::vector<int32_t>{2, 1, 3},
            .input_rank = 3,
            .output_dims = std::vector<int32_t>{2, 2, 4},
            .output_rank = 3},
        ReferenceBroadcastToErrorCase{
            .test_name = "ByteCountOverflow",
            .input_dims = std::nullopt,
            .input_rank = 0,
            .output_dims = std::vector<int32_t>{1 << 30, 1 << 30, 8},
            .output_rank = 3}),
    [](const ::testing::TestParamInfo<ReferenceBroadcastToErrorCase>& info) {
      return info.param.test_name;
    });

TEST(BroadcastToOpTest, ReferenceBroadcastToRejectsNullInputData) {
  const int32_t out_dims[] = {2, 2};
  std::vector<int32_t> out(4);

  const LiteRtStatus status = ReferenceBroadcastTo<int32_t>(
      /*input_data=*/nullptr, /*input_dims=*/nullptr, /*input_rank=*/0,
      out_dims, /*output_rank=*/2, out.data());

  EXPECT_THAT(status, IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(BroadcastToOpTest, ReferenceBroadcastToRejectsNullOutputData) {
  const int32_t input = 42;
  const int32_t out_dims[] = {2, 2};

  const LiteRtStatus status = ReferenceBroadcastTo<int32_t>(
      &input, /*input_dims=*/nullptr, /*input_rank=*/0, out_dims,
      /*output_rank=*/2, /*output_data=*/nullptr);

  EXPECT_THAT(status, IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(BroadcastToOpTest, ReferenceBroadcastToTensorViewRejectsUndersizedSpans) {
  const int32_t in_dims[] = {2};
  const int32_t out_dims[] = {2, 2};
  const int32_t in_data[] = {10, 20};
  int32_t out_data[] = {-1, -1, -1, -1};

  EXPECT_THAT(ReferenceBroadcastTo<int32_t>(
                  ConstTensorView<int32_t>{
                      absl::MakeConstSpan(in_data).first(1), in_dims},
                  TensorView<int32_t>{absl::MakeSpan(out_data), out_dims}),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(
      ReferenceBroadcastTo<int32_t>(
          ConstTensorView<int32_t>{absl::MakeConstSpan(in_data), in_dims},
          TensorView<int32_t>{absl::MakeSpan(out_data).first(3), out_dims}),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(out_data, ElementsAre(-1, -1, -1, -1));
}

TEST(BroadcastToOpTest, ReferenceBroadcastToExpandsSingletonToHigherRank) {
  const int32_t in_dims[] = {1};
  const int32_t out_dims[] = {2, 3};
  const int32_t in_data[] = {7};
  std::vector<int32_t> out(6, 0);

  EXPECT_THAT(ReferenceBroadcastTo<int32_t>(
                  ConstTensorView<int32_t>{in_data, in_dims},
                  TensorView<int32_t>{absl::MakeSpan(out), out_dims}),
              IsOk());
  EXPECT_THAT(out, ElementsAre(7, 7, 7, 7, 7, 7));
}

}  // namespace
}  // namespace litert::internal
