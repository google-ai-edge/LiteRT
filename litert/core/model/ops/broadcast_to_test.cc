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

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

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

}  // namespace
}  // namespace litert::internal
