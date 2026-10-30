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
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

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

LiteRtOpT MakeDepthToSpaceOp(int32_t block_size) {
  LiteRtOpT op;
  auto options = std::make_unique<tflite::DepthToSpaceOptionsT>();
  options->block_size = block_size;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_DepthToSpaceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  return op;
}

LiteRtOpT MakeSpaceToDepthOp(int32_t block_size) {
  LiteRtOpT op;
  auto options = std::make_unique<tflite::SpaceToDepthOptionsT>();
  options->block_size = block_size;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SpaceToDepthOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  return op;
}

template <typename T>
std::vector<uint8_t> ToBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

TEST(SpatialOpTest, DepthToSpace) {
  LiteRtOpT op = MakeDepthToSpaceOp(2);
  std::vector<Dims> input_shapes = {{1, 10, 10, 12}};  // Depth 12
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferDepthToSpace(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 20, 20, 3));
}

TEST(SpatialOpTest, DepthToSpaceInvalidRank) {
  LiteRtOpT op = MakeDepthToSpaceOp(2);
  std::vector<Dims> input_shapes = {{1, 10, 10}};  // Rank 3
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferDepthToSpace(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, DepthToSpaceInvalidDepth) {
  LiteRtOpT op = MakeDepthToSpaceOp(2);
  std::vector<Dims> input_shapes = {
      {1, 10, 10, 10}};  // Depth 10 is not divisible by 2x2
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferDepthToSpace(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToDepth) {
  LiteRtOpT op = MakeSpaceToDepthOp(2);
  // Input [1, 10, 10, 3] (Depth 3)
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferSpaceToDepth(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 5, 5, 12));
}

TEST(SpatialOpTest, SpaceToDepthInvalidRank) {
  LiteRtOpT op = MakeSpaceToDepthOp(2);
  std::vector<Dims> input_shapes = {{1, 10, 10}};  // Rank 3
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferSpaceToDepth(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToDepthInvalidSpatialDim) {
  LiteRtOpT op = MakeSpaceToDepthOp(2);
  std::vector<Dims> input_shapes = {
      {1, 10, 11, 3}};  // Width 11 not divisible by 2
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferSpaceToDepth(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

class SpatialTestShapeInferenceContext : public ShapeInferenceContext {
 public:
  explicit SpatialTestShapeInferenceContext(
      LiteRtOpCode op_code, std::vector<Dims> input_shapes,
      std::vector<std::vector<uint8_t>> input_data = {})
      : op_code_(op_code),
        input_shapes_(std::move(input_shapes)),
        input_data_(std::move(input_data)) {}

  size_t GetNumInputs() const override { return input_shapes_.size(); }

  size_t GetNumOutputs() const override { return 1; }

  Dims GetInputShape(size_t index) const override {
    if (index >= input_shapes_.size()) return {};
    return input_shapes_[index];
  }

  absl::Span<const uint8_t> GetInputData(size_t index) const override {
    if (index >= input_data_.size()) return {};
    return absl::MakeConstSpan(input_data_[index]);
  }

  const TflOptions& GetOptions() const override { return options_; }

  LiteRtOpCode GetOpCode() const override { return op_code_; }

 private:
  LiteRtOpCode op_code_;
  std::vector<Dims> input_shapes_;
  TflOptions options_;
  std::vector<std::vector<uint8_t>> input_data_;
};

TEST(SpatialOpTest, SpaceToBatchNdStatic) {
  std::vector<int32_t> block_shape = {2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> paddings = {0, 0};
  std::vector<uint8_t> padding_bytes(2 * sizeof(int32_t));
  std::memcpy(padding_bytes.data(), paddings.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{1, 100, 32}, {1}, {1, 2}},
                                       {{}, block_bytes, padding_bytes});
  InferenceResult result;
  ASSERT_EQ(InferSpaceToBatchNd(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(2, 50, 32));
}

TEST(SpatialOpTest, SpaceToBatchNdStatic4D) {
  std::vector<int32_t> block_shape = {2, 2};
  std::vector<uint8_t> block_bytes(2 * sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), 2 * sizeof(int32_t));

  std::vector<int32_t> paddings = {1, 1, 0, 0};
  std::vector<uint8_t> padding_bytes(4 * sizeof(int32_t));
  std::memcpy(padding_bytes.data(), paddings.data(), 4 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{1, 10, 10, 3}, {2}, {2, 2}},
                                       {{}, block_bytes, padding_bytes});
  InferenceResult result;
  ASSERT_EQ(InferSpaceToBatchNd(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(4, 6, 5, 3));
}

TEST(SpatialOpTest, SpaceToBatchNdDynamicBatch) {
  std::vector<int32_t> block_shape = {2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> paddings = {0, 0};
  std::vector<uint8_t> padding_bytes(2 * sizeof(int32_t));
  std::memcpy(padding_bytes.data(), paddings.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{-1, 100, 32}, {1}, {1, 2}},
                                       {{}, block_bytes, padding_bytes});
  InferenceResult result;
  ASSERT_EQ(InferSpaceToBatchNd(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(-1, 50, 32));
}

TEST(SpatialOpTest, SpaceToBatchNdDynamicParameters) {
  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{1, 100, 32}, {1}, {1, 2}},
                                       {{}, {}, {}});
  InferenceResult result;
  ASSERT_EQ(InferSpaceToBatchNd(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(-1, -1, 32));
}

TEST(SpatialOpTest, SpaceToBatchNdInvalidRank) {
  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{100, 32}, {1}, {1, 2}}, {{}, {}, {}});
  InferenceResult result;
  EXPECT_EQ(InferSpaceToBatchNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToBatchNdNonDivisible) {
  std::vector<int32_t> block_shape = {3};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> paddings = {0, 0};
  std::vector<uint8_t> padding_bytes(2 * sizeof(int32_t));
  std::memcpy(padding_bytes.data(), paddings.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{1, 10, 3}, {1}, {1, 2}},
                                       {{}, block_bytes, padding_bytes});
  InferenceResult result;
  EXPECT_EQ(InferSpaceToBatchNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToBatchNdZeroBlockSize) {
  std::vector<int32_t> block_shape = {0};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> paddings = {0, 0};
  std::vector<uint8_t> padding_bytes(2 * sizeof(int32_t));
  std::memcpy(padding_bytes.data(), paddings.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{1, 10, 3}, {1}, {1, 2}},
                                       {{}, block_bytes, padding_bytes});
  InferenceResult result;
  EXPECT_EQ(InferSpaceToBatchNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToBatchNdNegativeBlockSize) {
  std::vector<int32_t> block_shape = {-2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> paddings = {0, 0};
  std::vector<uint8_t> padding_bytes(2 * sizeof(int32_t));
  std::memcpy(padding_bytes.data(), paddings.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{1, 10, 3}, {1}, {1, 2}},
                                       {{}, block_bytes, padding_bytes});
  InferenceResult result;
  EXPECT_EQ(InferSpaceToBatchNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToBatchNdNegativePadding) {
  std::vector<int32_t> block_shape = {2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> paddings = {-1, 0};
  std::vector<uint8_t> padding_bytes(2 * sizeof(int32_t));
  std::memcpy(padding_bytes.data(), paddings.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{1, 10, 3}, {1}, {1, 2}},
                                       {{}, block_bytes, padding_bytes});
  InferenceResult result;
  EXPECT_EQ(InferSpaceToBatchNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToBatchNdInvalidBlockShapeRank) {
  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{1, 10, 3}, {1, 2}, {1, 2}},
                                       {{}, {}, {}});
  InferenceResult result;
  EXPECT_EQ(InferSpaceToBatchNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToBatchNdInvalidPaddingsRank) {
  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{1, 10, 3}, {1}, {1, 3}}, {{}, {}, {}});
  InferenceResult result;
  EXPECT_EQ(InferSpaceToBatchNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, SpaceToBatchNdInvalidBufferBytes) {
  std::vector<uint8_t> invalid_bytes = {1, 2, 3};
  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflSpaceToBatchNd,
                                       {{1, 10, 3}, {1}, {1, 2}},
                                       {{}, invalid_bytes, {}});
  InferenceResult result;
  EXPECT_EQ(InferSpaceToBatchNd(ctx, result),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(SpatialOpTest, SpaceToBatchNdBatchOverflow) {
  std::vector<int32_t> block_shape = {2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> paddings = {0, 0};
  std::vector<uint8_t> padding_bytes(2 * sizeof(int32_t));
  std::memcpy(padding_bytes.data(), paddings.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(
      kLiteRtOpCodeTflSpaceToBatchNd,
      {{INT32_MAX / 2 + 10, 10, 3}, {1}, {1, 2}},
      {{}, block_bytes, padding_bytes});
  InferenceResult result;
  EXPECT_EQ(InferSpaceToBatchNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, BatchToSpaceNdStatic) {
  std::vector<int32_t> block_shape = {2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> crops = {0, 0};
  std::vector<uint8_t> crop_bytes(2 * sizeof(int32_t));
  std::memcpy(crop_bytes.data(), crops.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflBatchToSpaceNd,
                                       {{2, 50, 32}, {1}, {1, 2}},
                                       {{}, block_bytes, crop_bytes});
  InferenceResult result;
  ASSERT_EQ(InferBatchToSpaceNd(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(1, 100, 32));
}

TEST(SpatialOpTest, BatchToSpaceNdStatic4D) {
  std::vector<int32_t> block_shape = {2, 2};
  std::vector<uint8_t> block_bytes(2 * sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), 2 * sizeof(int32_t));

  std::vector<int32_t> crops = {1, 1, 0, 0};
  std::vector<uint8_t> crop_bytes(4 * sizeof(int32_t));
  std::memcpy(crop_bytes.data(), crops.data(), 4 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflBatchToSpaceNd,
                                       {{4, 6, 5, 3}, {2}, {2, 2}},
                                       {{}, block_bytes, crop_bytes});
  InferenceResult result;
  ASSERT_EQ(InferBatchToSpaceNd(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(1, 10, 10, 3));
}

TEST(SpatialOpTest, BatchToSpaceNdDynamicBatch) {
  std::vector<int32_t> block_shape = {2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> crops = {0, 0};
  std::vector<uint8_t> crop_bytes(2 * sizeof(int32_t));
  std::memcpy(crop_bytes.data(), crops.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflBatchToSpaceNd,
                                       {{-1, 50, 32}, {1}, {1, 2}},
                                       {{}, block_bytes, crop_bytes});
  InferenceResult result;
  ASSERT_EQ(InferBatchToSpaceNd(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(-1, 100, 32));
}

TEST(SpatialOpTest, BatchToSpaceNdDynamicParameters) {
  SpatialTestShapeInferenceContext ctx(
      kLiteRtOpCodeTflBatchToSpaceNd, {{2, 50, 32}, {1}, {1, 2}}, {{}, {}, {}});
  InferenceResult result;
  ASSERT_EQ(InferBatchToSpaceNd(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(-1, -1, 32));
}

TEST(SpatialOpTest, BatchToSpaceNdNonDivisibleBatch) {
  std::vector<int32_t> block_shape = {2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> crops = {0, 0};
  std::vector<uint8_t> crop_bytes(2 * sizeof(int32_t));
  std::memcpy(crop_bytes.data(), crops.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflBatchToSpaceNd,
                                       {{3, 50, 32}, {1}, {1, 2}},
                                       {{}, block_bytes, crop_bytes});
  InferenceResult result;
  EXPECT_EQ(InferBatchToSpaceNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, BatchToSpaceNdZeroBlockSize) {
  std::vector<int32_t> block_shape = {0};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> crops = {0, 0};
  std::vector<uint8_t> crop_bytes(2 * sizeof(int32_t));
  std::memcpy(crop_bytes.data(), crops.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflBatchToSpaceNd,
                                       {{2, 50, 32}, {1}, {1, 2}},
                                       {{}, block_bytes, crop_bytes});
  InferenceResult result;
  EXPECT_EQ(InferBatchToSpaceNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, BatchToSpaceNdNegativeBlockSize) {
  std::vector<int32_t> block_shape = {-2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> crops = {0, 0};
  std::vector<uint8_t> crop_bytes(2 * sizeof(int32_t));
  std::memcpy(crop_bytes.data(), crops.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflBatchToSpaceNd,
                                       {{2, 50, 32}, {1}, {1, 2}},
                                       {{}, block_bytes, crop_bytes});
  InferenceResult result;
  EXPECT_EQ(InferBatchToSpaceNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, BatchToSpaceNdNegativeCrop) {
  std::vector<int32_t> block_shape = {2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> crops = {-1, 0};
  std::vector<uint8_t> crop_bytes(2 * sizeof(int32_t));
  std::memcpy(crop_bytes.data(), crops.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflBatchToSpaceNd,
                                       {{2, 50, 32}, {1}, {1, 2}},
                                       {{}, block_bytes, crop_bytes});
  InferenceResult result;
  EXPECT_EQ(InferBatchToSpaceNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, BatchToSpaceNdNegativeOutputDim) {
  std::vector<int32_t> block_shape = {2};
  std::vector<uint8_t> block_bytes(sizeof(int32_t));
  std::memcpy(block_bytes.data(), block_shape.data(), sizeof(int32_t));

  std::vector<int32_t> crops = {6, 6};
  std::vector<uint8_t> crop_bytes(2 * sizeof(int32_t));
  std::memcpy(crop_bytes.data(), crops.data(), 2 * sizeof(int32_t));

  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflBatchToSpaceNd,
                                       {{2, 5, 32}, {1}, {1, 2}},
                                       {{}, block_bytes, crop_bytes});
  InferenceResult result;
  EXPECT_EQ(InferBatchToSpaceNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, BatchToSpaceNdInvalidBlockShapeRank) {
  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflBatchToSpaceNd,
                                       {{2, 50, 32}, {1, 2}, {1, 2}},
                                       {{}, {}, {}});
  InferenceResult result;
  EXPECT_EQ(InferBatchToSpaceNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, BatchToSpaceNdInvalidCropsRank) {
  SpatialTestShapeInferenceContext ctx(
      kLiteRtOpCodeTflBatchToSpaceNd, {{2, 50, 32}, {1}, {1, 3}}, {{}, {}, {}});
  InferenceResult result;
  EXPECT_EQ(InferBatchToSpaceNd(ctx, result),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(SpatialOpTest, BatchToSpaceNdInvalidBufferBytes) {
  std::vector<uint8_t> invalid_bytes = {1, 2, 3};
  SpatialTestShapeInferenceContext ctx(kLiteRtOpCodeTflBatchToSpaceNd,
                                       {{2, 50, 32}, {1}, {1, 2}},
                                       {{}, invalid_bytes, {}});
  InferenceResult result;
  EXPECT_EQ(InferBatchToSpaceNd(ctx, result),
            kLiteRtStatusErrorInvalidArgument);
}

struct DepthSpaceErrorCase {
  std::string test_name;
  Dims input_shape;
  int32_t block_size;
  size_t num_outputs = 1;
};

using DepthToSpaceErrorTest = ::testing::TestWithParam<DepthSpaceErrorCase>;

TEST_P(DepthToSpaceErrorTest, RejectsEmptyOutputsAndIntegerOverflows) {
  const auto& param = GetParam();
  LiteRtOpT op = MakeDepthToSpaceOp(param.block_size);
  std::vector<Dims> input_shapes = {param.input_shape};
  std::vector<Dims> output_shapes(param.num_outputs);
  EXPECT_EQ(InferDepthToSpace(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

INSTANTIATE_TEST_SUITE_P(
    DepthToSpaceErrorTests, DepthToSpaceErrorTest,
    ::testing::Values(
        DepthSpaceErrorCase{"EmptyOutputs",
                            {1, 10, 10, 12},
                            2,
                            /*num_outputs=*/0},
        DepthSpaceErrorCase{"BlockSizeSquareOverflow", {1, 10, 10, 12}, 225542},
        DepthSpaceErrorCase{
            "HeightProductOverflow", {1, INT32_MAX / 2 + 2, 4, 4}, 2},
        DepthSpaceErrorCase{
            "WidthProductOverflow", {1, 4, INT32_MAX / 2 + 2, 4}, 2},
        DepthSpaceErrorCase{"NegativeInputDimension", {-2, 4, 4, 4}, 2}),
    [](const ::testing::TestParamInfo<DepthSpaceErrorCase>& info) {
      return info.param.test_name;
    });

using SpaceToDepthErrorTest = ::testing::TestWithParam<DepthSpaceErrorCase>;

TEST_P(SpaceToDepthErrorTest, RejectsEmptyOutputsAndIntegerOverflows) {
  const auto& param = GetParam();
  LiteRtOpT op = MakeSpaceToDepthOp(param.block_size);
  std::vector<Dims> input_shapes = {param.input_shape};
  std::vector<Dims> output_shapes(param.num_outputs);
  EXPECT_EQ(InferSpaceToDepth(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

INSTANTIATE_TEST_SUITE_P(
    SpaceToDepthErrorTests, SpaceToDepthErrorTest,
    ::testing::Values(
        DepthSpaceErrorCase{"EmptyOutputs",
                            {1, 10, 10, 3},
                            2,
                            /*num_outputs=*/0},
        DepthSpaceErrorCase{"BlockSizeSquareOverflow", {1, 10, 10, 3}, 225542},
        DepthSpaceErrorCase{
            "DepthProductOverflow", {1, 4, 4, INT32_MAX / 4 + 2}, 2},
        DepthSpaceErrorCase{"NegativeInputDimension", {-2, 4, 4, 4}, 2}),
    [](const ::testing::TestParamInfo<DepthSpaceErrorCase>& info) {
      return info.param.test_name;
    });

struct SpaceBatchNdOverflowCase {
  std::string test_name;
  LiteRtOpCode op_code;
  Dims input_shape;
  std::vector<uint8_t> block_bytes;
  std::vector<uint8_t> padding_or_crop_bytes;
};

using SpaceBatchNdOverflowTest =
    ::testing::TestWithParam<SpaceBatchNdOverflowCase>;

TEST_P(SpaceBatchNdOverflowTest, RejectsNegativeDimsAndOverflows) {
  const auto& param = GetParam();
  SpatialTestShapeInferenceContext ctx(
      param.op_code, {param.input_shape, {1}, {1, 2}},
      {{}, param.block_bytes, param.padding_or_crop_bytes});
  InferenceResult result;
  LiteRtStatus status = param.op_code == kLiteRtOpCodeTflSpaceToBatchNd
                            ? InferSpaceToBatchNd(ctx, result)
                            : InferBatchToSpaceNd(ctx, result);
  EXPECT_EQ(status, kLiteRtStatusErrorShapeInferenceFailed);
}

INSTANTIATE_TEST_SUITE_P(
    SpaceBatchNdOverflowTests, SpaceBatchNdOverflowTest,
    ::testing::Values(
        SpaceBatchNdOverflowCase{
            "SpaceToBatchNdScaledDimExceedingInt32Max",
            kLiteRtOpCodeTflSpaceToBatchNd,
            {1, 1, 3},
            ToBytes<int32_t>({1}),
            ToBytes<int32_t>({INT32_MAX, 10}),
        },
        SpaceBatchNdOverflowCase{
            "SpaceToBatchNdNegativeInputDimension",
            kLiteRtOpCodeTflSpaceToBatchNd,
            {-2, 4, 3},
            ToBytes<int32_t>({1}),
            ToBytes<int32_t>({0, 0}),
        },
        SpaceBatchNdOverflowCase{
            "BatchToSpaceNdNegativeInputDimension",
            kLiteRtOpCodeTflBatchToSpaceNd,
            {-2, 4, 3},
            ToBytes<int32_t>({1}),
            ToBytes<int32_t>({0, 0}),
        },
        SpaceBatchNdOverflowCase{
            "SpaceToBatchNdOutputBatchSizeProductOverflow",
            kLiteRtOpCodeTflSpaceToBatchNd,
            {2, 4, 3},
            ToBytes<int64_t>({std::numeric_limits<int64_t>::max()}),
            ToBytes<int64_t>({0, 0}),
        }),
    [](const ::testing::TestParamInfo<SpaceBatchNdOverflowCase>& info) {
      return info.param.test_name;
    });

TEST(SpatialOpTest, ResizeOpRejectsMissingSizeTensorOrNonPositiveSize) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 10, 10, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  // Missing op.Inputs() tensors is rejected without dereferencing nullptr.
  EXPECT_EQ(InferResizeOp(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);

  LiteRtTensorT input0;
  LiteRtTensorT size_tensor;
  op.Inputs() = {&input0, &size_tensor};

  int32_t non_positive_size[] = {0, 10};
  SetWeightsFromOwnedBuffer(
      size_tensor.Weights(),
      OwningBufferRef<uint8_t>(
          reinterpret_cast<const uint8_t*>(non_positive_size),
          sizeof(non_positive_size)));
  EXPECT_EQ(InferResizeOp(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

}  // namespace
}  // namespace litert::internal
