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

TEST(ReshapeOpTest, WithOptions) {
  LiteRtOpT op;
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, 4, 4, 3};
  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  std::vector<Dims> input_shapes = {{1, 48}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferReshape(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 4, 4, 3));
}

TEST(ReshapeOpTest, WithShapeTensor) {
  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;

  int32_t shape_data[] = {1, 4, 4, 3};
  SetWeightsFromOwnedBuffer(
      shape_tensor.Weights(),
      OwningBufferRef<uint8_t>(absl::MakeConstSpan(
          reinterpret_cast<const uint8_t*>(shape_data), sizeof(shape_data))));

  LiteRtOpT op;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&shape_tensor);

  std::vector<Dims> input_shapes = {{1, 48}, {4}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferReshape(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 4, 4, 3));
}

TEST(ReshapeOpTest, EmptyInputShapes) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes;
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(InferReshape(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(ReshapeOpTest, DynamicShapeTensor) {
  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;

  LiteRtOpT op;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&shape_tensor);

  std::vector<Dims> input_shapes = {{1, 48}, {4}};
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferReshape(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, InvalidShapeTensorBufferSize) {
  LiteRtTensorT input_tensor;
  LiteRtTensorT shape_tensor;

  uint8_t shape_data[] = {1, 2, 3};
  SetWeightsFromOwnedBuffer(
      shape_tensor.Weights(),
      OwningBufferRef<uint8_t>(absl::MakeConstSpan(shape_data, 3)));

  LiteRtOpT op;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&shape_tensor);

  std::vector<Dims> input_shapes = {{1, 48}, {4}};
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferReshape(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, NoOptionsAndNoShapeTensor) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 48}};
  std::vector<Dims> output_shapes(1);
  EXPECT_EQ(InferReshape(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, MultipleMinusOne) {
  LiteRtOpT op;
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, -1, -1, 3};
  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  std::vector<Dims> input_shapes = {{1, 48}};
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferReshape(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, MinusOneWithDynamicInput) {
  LiteRtOpT op;
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, -1, 3};
  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  std::vector<Dims> input_shapes = {{1, -1}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferReshape(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, -1, 3));
}

TEST(ReshapeOpTest, IncompatibleVolumesWithMinusOne) {
  LiteRtOpT op;
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, -1, 5};
  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  std::vector<Dims> input_shapes = {{1, 48}};
  std::vector<Dims> output_shapes(1);

  EXPECT_EQ(InferReshape(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

TEST(ReshapeOpTest, ResolveMinusOne) {
  LiteRtOpT op;
  auto options = std::make_unique<tflite::ReshapeOptionsT>();
  options->new_shape = {1, -1, 3};
  litert::internal::TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_ReshapeOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  std::vector<Dims> input_shapes = {{1, 48}};
  std::vector<Dims> output_shapes(1);

  ASSERT_EQ(InferReshape(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 16, 3));
}

}  // namespace
}  // namespace litert::internal
