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

#include "litert/core/model/ops/reshape.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

class TestShapeInferenceContext : public ShapeInferenceContext {
 public:
  explicit TestShapeInferenceContext(
      std::vector<Dims> input_shapes, TflOptions options = {},
      std::vector<std::vector<uint8_t>> input_data = {})
      : input_shapes_(std::move(input_shapes)),
        options_(std::move(options)),
        input_data_(std::move(input_data)) {}

  Dims GetInputShape(size_t index) const override {
    if (index >= input_shapes_.size()) return {};
    return input_shapes_[index];
  }

  absl::Span<const uint8_t> GetInputData(size_t index) const override {
    if (index >= input_data_.size()) return {};
    return absl::MakeConstSpan(input_data_[index]);
  }

  const TflOptions& GetOptions() const override { return options_; }

  LiteRtOpCode GetOpCode() const override { return kLiteRtOpCodeTflReshape; }

 private:
  std::vector<Dims> input_shapes_;
  TflOptions options_;
  std::vector<std::vector<uint8_t>> input_data_;
};

TEST(ReshapeOpTest, WithOptions) {
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, 4, 4, 3};
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();

  TestShapeInferenceContext ctx({{1, 48}}, std::move(tfl_options));
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);

  EXPECT_THAT(result.output_shapes[0], ElementsAre(1, 4, 4, 3));
}

TEST(ReshapeOpTest, WithShapeTensor) {
  std::vector<int32_t> shape_data = {1, 4, 4, 3};
  std::vector<uint8_t> shape_bytes(shape_data.size() * sizeof(int32_t));
  std::memcpy(shape_bytes.data(), shape_data.data(), shape_bytes.size());

  TestShapeInferenceContext ctx({{1, 48}, {4}}, {}, {{}, shape_bytes});
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);

  EXPECT_THAT(result.output_shapes[0], ElementsAre(1, 4, 4, 3));
}

TEST(ReshapeOpTest, DynamicShapeTensor) {
  TestShapeInferenceContext ctx({{1, 48}, {4}}, {}, {{}, {}});
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, InvalidShapeTensorBufferSize) {
  std::vector<uint8_t> shape_bytes = {1, 2, 3};

  TestShapeInferenceContext ctx({{1, 48}, {4}}, {}, {{}, shape_bytes});
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorInvalidArgument);
}

TEST(ReshapeOpTest, NoOptionsAndNoShapeTensor) {
  TestShapeInferenceContext ctx({{1, 48}}, {}, {{}});
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, MultipleMinusOne) {
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, -1, -1, 3};
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();

  TestShapeInferenceContext ctx({{1, 48}}, std::move(tfl_options));
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, MinusOneWithDynamicInput) {
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, -1, 3};
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();

  TestShapeInferenceContext ctx({{1, -1}}, std::move(tfl_options));
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(1, -1, 3));
}

TEST(ReshapeOpTest, IncompatibleVolumesWithMinusOne) {
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, -1, 5};
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();

  TestShapeInferenceContext ctx({{1, 48}}, std::move(tfl_options));
  InferenceResult result;

  EXPECT_EQ(InferReshape(ctx, result), kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, ResolveMinusOne) {
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, -1, 3};
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();

  TestShapeInferenceContext ctx({{1, 48}}, std::move(tfl_options));
  InferenceResult result;

  ASSERT_EQ(InferReshape(ctx, result), kLiteRtStatusOk);
  EXPECT_THAT(result.output_shapes[0], ElementsAre(1, 16, 3));
}

}  // namespace
}  // namespace litert::internal
