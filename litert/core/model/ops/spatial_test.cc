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

#include "litert/core/model/ops/spatial.h"

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
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(SpatialOpTest, ResizeOpStaticSize) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input0;
  LiteRtTensorT size_tensor;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&size_tensor);

  int32_t size_data[] = {25, 25};
  SetWeightsFromOwnedBuffer(
      size_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(size_data),
                               sizeof(size_data)));

  ASSERT_EQ(InferResizeOp(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 25, 25, 3));
}

TEST(SpatialOpTest, ResizeOpDynamicSize) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input0;
  LiteRtTensorT size_tensor;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&size_tensor);

  ASSERT_EQ(InferResizeOp(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_EQ(output_shapes[0].size(), 4);
  EXPECT_EQ(output_shapes[0][0], 1);
  EXPECT_EQ(output_shapes[0][1], -1);
  EXPECT_EQ(output_shapes[0][2], -1);
  EXPECT_EQ(output_shapes[0][3], 3);
}

TEST(SpatialOpTest, ResizeOpInvalidRank) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input0;
  LiteRtTensorT size_tensor;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&size_tensor);

  EXPECT_EQ(InferResizeOp(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, ResizeBilinearStaticSize) {
  LiteRtOpT op;
  // Input 0: Image [1, 10, 10, 3]
  // Input 1: Size [2] (Height=20, Width=20) - Constant
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  // Setup constant size tensor
  LiteRtTensorT input0;
  LiteRtTensorT size_tensor;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&size_tensor);

  int32_t size_data[] = {20, 20};
  SetWeightsFromOwnedBuffer(
      size_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(size_data),
                               sizeof(size_data)));

  ASSERT_EQ(
      InferResizeBilinear(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 20, 20, 3));
}

TEST(SpatialOpTest, ResizeNearestNeighborStaticSize) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input0;
  LiteRtTensorT size_tensor;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&size_tensor);

  int32_t size_data[] = {30, 30};
  SetWeightsFromOwnedBuffer(
      size_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(size_data),
                               sizeof(size_data)));

  ASSERT_EQ(InferResizeNearestNeighbor(op, absl::MakeSpan(input_shapes),
                                       output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 30, 30, 3));
}

TEST(SpatialOpTest, ResizeBilinearDynamicSize) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  // Size tensor is dynamic (empty/no buffer)
  LiteRtTensorT input0;
  LiteRtTensorT size_tensor;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&size_tensor);

  ASSERT_EQ(
      InferResizeBilinear(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  // Check rank and known dimensions
  EXPECT_EQ(output_shapes[0].size(), 4);
  EXPECT_EQ(output_shapes[0][0], 1);
  EXPECT_EQ(output_shapes[0][1], -1);  // Height unknown
  EXPECT_EQ(output_shapes[0][2], -1);  // Width unknown
  EXPECT_EQ(output_shapes[0][3], 3);
}

TEST(SpatialOpTest, ResizeBilinearInvalidRank) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input0;
  LiteRtTensorT size_tensor;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&size_tensor);

  EXPECT_EQ(
      InferResizeBilinear(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, ResizeNearestNeighborDynamicSize) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  // Size tensor is dynamic (empty/no buffer)
  LiteRtTensorT input0;
  LiteRtTensorT size_tensor;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&size_tensor);

  ASSERT_EQ(InferResizeNearestNeighbor(op, absl::MakeSpan(input_shapes),
                                       output_shapes),
            kLiteRtStatusOk);

  // Check rank and known dimensions
  EXPECT_EQ(output_shapes[0].size(), 4);
  EXPECT_EQ(output_shapes[0][0], 1);
  EXPECT_EQ(output_shapes[0][1], -1);  // Height unknown
  EXPECT_EQ(output_shapes[0][2], -1);  // Width unknown
  EXPECT_EQ(output_shapes[0][3], 3);
}

TEST(SpatialOpTest, ResizeNearestNeighborInvalidRank) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input0;
  LiteRtTensorT size_tensor;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&size_tensor);

  EXPECT_EQ(InferResizeNearestNeighbor(op, absl::MakeSpan(input_shapes),
                                       output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, DepthToSpace) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 12}};  // Depth 12
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::DepthToSpaceOptionsT>();
  options->block_size = 2;  // 2x2 block -> Depth / 4 -> 3. Spatial * 2 -> 20x20
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthToSpaceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferDepthToSpace(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 20, 20, 3));
}

TEST(SpatialOpTest, DepthToSpaceInvalidRank) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10}};  // Rank 3
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::DepthToSpaceOptionsT>();
  options->block_size = 2;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthToSpaceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  EXPECT_EQ(InferDepthToSpace(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, DepthToSpaceInvalidDepth) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {
      {1, 10, 10, 10}};  // Depth 10 is not divisible by 2x2
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::DepthToSpaceOptionsT>();
  options->block_size = 2;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthToSpaceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  EXPECT_EQ(InferDepthToSpace(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToDepth) {
  LiteRtOpT op;
  // Input [1, 10, 10, 3] (Depth 3)
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}};
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::SpaceToDepthOptionsT>();
  options->block_size = 2;  // 2x2 block -> Depth * 4 -> 12. Spatial / 2 -> 5x5
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SpaceToDepthOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferSpaceToDepth(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 5, 5, 12));
}

TEST(SpatialOpTest, SpaceToDepthInvalidRank) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10}};  // Rank 3
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::SpaceToDepthOptionsT>();
  options->block_size = 2;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SpaceToDepthOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  EXPECT_EQ(InferSpaceToDepth(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToDepthInvalidSpatialDim) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {
      {1, 10, 11, 3}};  // Width 11 not divisible by 2
  std::vector<Dims> output_shapes(1);

  auto options = std::make_unique<tflite::SpaceToDepthOptionsT>();
  options->block_size = 2;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SpaceToDepthOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  EXPECT_EQ(InferSpaceToDepth(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

}  // namespace
}  // namespace litert::internal
