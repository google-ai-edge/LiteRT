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

#include "litert/core/model/ops/reductions.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/simple_unary.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(ReductionsOpTest, Cumsum) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {1}};  // Input, Axis
  std::vector<Dims> output_shapes(1);

  // Cumsum is actually Unary in simple_unary.h as macro, but defined there.
  // It maintains shape.
  ASSERT_EQ(InferCumsum(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(ReductionsOpTest, MeanKeepDims) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 3},
                                    {2}};  // Input, Axis (const)
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT axis_tensor;
  op.Inputs().push_back(nullptr);       // Input 0 placeholder
  op.Inputs().push_back(&axis_tensor);  // Input 1 (Axis)

  int32_t axis_data[] = {1, 2};
  SetWeightsFromOwnedBuffer(
      axis_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(axis_data),
                               sizeof(axis_data)));

  auto options = std::make_unique<tflite::ReducerOptionsT>();
  options->keep_dims = true;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReducerOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferReduce(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 1, 1, 3));
}

TEST(ReductionsOpTest, ArgMax) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}, {1}};  // Input, Axis
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT axis_tensor;
  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&axis_tensor);

  int32_t axis_data[] = {3};  // Axis 3
  SetWeightsFromOwnedBuffer(
      axis_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(axis_data),
                               sizeof(axis_data)));

  ASSERT_EQ(InferArgMinMax(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 10, 10));  // Rank reduced
}

TEST(ReductionsOpTest, ArgMin) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}, {1}};  // Input, Axis
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT axis_tensor;
  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&axis_tensor);

  int32_t axis_data[] = {3};  // Axis 3
  SetWeightsFromOwnedBuffer(
      axis_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(axis_data),
                               sizeof(axis_data)));

  ASSERT_EQ(InferArgMinMax(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 10, 10));  // Rank reduced
}

TEST(ReductionsOpTest, SumReduce) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 112, 112, 32}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT axis_tensor;
  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&axis_tensor);

  int32_t axis_data[] = {1, 2};
  SetWeightsFromOwnedBuffer(
      axis_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(axis_data),
                               sizeof(axis_data)));

  auto options = std::make_unique<tflite::ReducerOptionsT>();
  options->keep_dims = false;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReducerOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferReduce(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 32));
}

}  // namespace
}  // namespace litert::internal
