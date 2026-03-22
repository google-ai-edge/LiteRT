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

#include "litert/core/model/ops/gather.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(GatherOpTest, SimpleGatherAxis0) {
  LiteRtOpT op;
  // Input [2, 2], Indices [2]
  std::vector<Dims> input_shapes = {{2, 2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = 0;
  options->batch_dims = 0;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferGather(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: [2] + [2] = [2, 2]
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2));
}

TEST(GatherOpTest, GatherAxis1) {
  LiteRtOpT op;
  // Input [1, 2, 3], Indices [2]
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = 1;
  options->batch_dims = 0;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferGather(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: Input[:1] + Indices + Input[1+1:] -> [1] + [2] + [3] = [1, 2, 3]
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 3));
}

TEST(GatherOpTest, GatherScalarIndex) {
  LiteRtOpT op;
  // Input [2, 2], Indices [] (scalar)
  std::vector<Dims> input_shapes = {{2, 2}, {}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = 0;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferGather(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: Input[:0] + [] + Input[1:] -> [] + [] + [2] = [2]
  EXPECT_THAT(output_shapes[0], ElementsAre(2));
}

TEST(GatherOpTest, GatherLastAxis) {
  LiteRtOpT op;
  // Input [1, 2, 3], Indices [2]
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = 2;  // -1 or 2

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferGather(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: Input[:2] + Indices + Input[3:] -> [1, 2] + [2] + [] = [1, 2, 2]
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2));
}

TEST(GatherOpTest, GatherNegativeAxis) {
  LiteRtOpT op;
  // Input [1, 2, 3], Indices [2]
  std::vector<Dims> input_shapes = {{1, 2, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::GatherOptionsT>();
  options->axis = -1;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_GatherOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferGather(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: [1, 2, 2] same as above
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 2, 2));
}

TEST(GatherOpTest, EmbeddingLookup) {
  LiteRtOpT op;
  // Ids [2], Params [10, 5] (Lookup 2 vectors of size 5)
  std::vector<Dims> input_shapes = {{2}, {10, 5}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(
      InferEmbeddingLookup(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 5));
}

TEST(GatherOpTest, GatherNd) {
  LiteRtOpT op;
  // Input [2, 2], Indices [2, 1]
  std::vector<Dims> input_shapes = {{2, 2}, {2, 1}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferGatherNd(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 2));
}

TEST(GatherOpTest, GatherNdSlice) {
  LiteRtOpT op;
  // Input [2, 2], Indices [1, 2]
  std::vector<Dims> input_shapes = {{2, 2}, {1, 2}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferGatherNd(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  // Output: [1] (indices batch dim) + [] (input suffix) = [1]
  EXPECT_THAT(output_shapes[0], ElementsAre(1));
}

}  // namespace
}  // namespace litert::internal
