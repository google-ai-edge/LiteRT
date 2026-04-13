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

#include "litert/core/model/ops/unpack.h"

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

std::unique_ptr<tflite::UnpackOptionsT> CreateUnpackOptions(int axis, int num) {
  auto options = std::make_unique<tflite::UnpackOptionsT>();
  options->axis = axis;
  options->num = num;
  return options;
}

TEST(UnpackOpTest, ThreeOutputs) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{3, 2}};
  std::vector<Dims> output_shapes(3);

  auto options = CreateUnpackOptions(0, 3);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_UnpackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferUnpack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(2));
  }
}

TEST(UnpackOpTest, ThreeOutputsAxisOne) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{3, 2}};
  std::vector<Dims> output_shapes(2);

  auto options = CreateUnpackOptions(1, 2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_UnpackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferUnpack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(3));
  }
}

TEST(UnpackOpTest, ThreeOutputsNegativeAxis) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{3, 2}};
  std::vector<Dims> output_shapes(2);

  auto options = CreateUnpackOptions(-1, 2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_UnpackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferUnpack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(3));
  }
}

}  // namespace
}  // namespace litert::internal
