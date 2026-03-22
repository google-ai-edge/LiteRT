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

#include "litert/core/model/ops/pack.h"

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

std::unique_ptr<tflite::PackOptionsT> CreatePackOptions(int axis) {
  auto options = std::make_unique<tflite::PackOptionsT>();
  options->axis = axis;
  return options;
}

TEST(PackOpTest, ThreeInputs) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2}, {2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePackOptions(0);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(3, 2));
}

TEST(PackOpTest, ThreeInputsDifferentAxis) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2}, {2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePackOptions(1);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 3));
}

TEST(PackOpTest, ThreeInputsNegativeAxis) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2}, {2}, {2}};
  std::vector<Dims> output_shapes(1);

  auto options = CreatePackOptions(-1);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_PackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferPack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(2, 3));
}

}  // namespace
}  // namespace litert::internal
