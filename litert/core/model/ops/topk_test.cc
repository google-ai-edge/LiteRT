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

#include "litert/core/model/ops/topk.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(TopKV2OpTest, SimpleTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 10}, {}};
  std::vector<Dims> output_shapes(2);

  LiteRtTensorT input_tensor;
  LiteRtTensorT k_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&k_tensor);

  std::vector<int32_t> k_data = {3};
  SetWeightsFromUnownedBuffer(
      k_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(k_data.data()),
                                 sizeof(int32_t)));

  ASSERT_EQ(InferTopKV2(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(2, 3));
  EXPECT_THAT(output_shapes[1], ElementsAre(2, 3));
}

TEST(TopKV2OpTest, MissingInputsTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 10}};  // Missing k tensor
  std::vector<Dims> output_shapes(2);

  EXPECT_EQ(InferTopKV2(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(TopKV2OpTest, DynamicKTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 10}, {}};
  std::vector<Dims> output_shapes(2);

  LiteRtTensorT input_tensor;
  LiteRtTensorT k_tensor;  // No buffer (dynamic size)
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&k_tensor);

  ASSERT_EQ(InferTopKV2(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(2, -1));
  EXPECT_THAT(output_shapes[1], ElementsAre(2, -1));
}

TEST(TopKV2OpTest, OnlyOneOutputTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 10}, {}};
  std::vector<Dims> output_shapes(1);  // Only 1 output

  LiteRtTensorT input_tensor;
  LiteRtTensorT k_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&k_tensor);

  std::vector<int32_t> k_data = {3};
  SetWeightsFromUnownedBuffer(
      k_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(k_data.data()),
                                 sizeof(int32_t)));

  ASSERT_EQ(InferTopKV2(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(2, 3));
}

TEST(TopKV2OpTest, ScalarInputTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{}, {}};  // Scalar input
  std::vector<Dims> output_shapes(2);

  LiteRtTensorT input_tensor;
  LiteRtTensorT k_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&k_tensor);

  std::vector<int32_t> k_data = {1};
  SetWeightsFromUnownedBuffer(
      k_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(k_data.data()),
                                 sizeof(int32_t)));

  EXPECT_EQ(InferTopKV2(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(TopKV2OpTest, InvalidKTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 10}, {}};
  std::vector<Dims> output_shapes(2);

  LiteRtTensorT input_tensor;
  LiteRtTensorT k_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&k_tensor);

  std::vector<int32_t> k_data = {15};  // k > internal dimension (10)
  SetWeightsFromUnownedBuffer(
      k_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(k_data.data()),
                                 sizeof(int32_t)));

  EXPECT_EQ(InferTopKV2(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(TopKV2OpTest, MultiDimensionalTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 4, 10}, {}};
  std::vector<Dims> output_shapes(2);

  LiteRtTensorT input_tensor;
  LiteRtTensorT k_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&k_tensor);

  std::vector<int32_t> k_data = {5};
  SetWeightsFromUnownedBuffer(
      k_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(k_data.data()),
                                 sizeof(int32_t)));

  ASSERT_EQ(InferTopKV2(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(2, 4, 5));
  EXPECT_THAT(output_shapes[1], ElementsAre(2, 4, 5));
}

}  // namespace
}  // namespace litert::internal
