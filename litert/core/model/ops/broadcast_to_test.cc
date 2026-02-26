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

}  // namespace
}  // namespace litert::internal
