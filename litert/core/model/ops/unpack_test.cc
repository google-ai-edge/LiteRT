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

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

LiteRtOpT MakeUnpackOp(int axis, int num) {
  LiteRtOpT op;
  auto options = std::make_unique<tflite::UnpackOptionsT>();
  options->axis = axis;
  options->num = num;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_UnpackOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));
  return op;
}

TEST(UnpackOpTest, ThreeOutputs) {
  LiteRtOpT op = MakeUnpackOp(0, 3);
  std::vector<Dims> input_shapes = {{3, 2}};
  std::vector<Dims> output_shapes(3);

  ASSERT_EQ(InferUnpack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(2));
  }
}

TEST(UnpackOpTest, ThreeOutputsAxisOne) {
  LiteRtOpT op = MakeUnpackOp(1, 2);
  std::vector<Dims> input_shapes = {{3, 2}};
  std::vector<Dims> output_shapes(2);

  ASSERT_EQ(InferUnpack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(3));
  }
}

TEST(UnpackOpTest, ThreeOutputsNegativeAxis) {
  LiteRtOpT op = MakeUnpackOp(-1, 2);
  std::vector<Dims> input_shapes = {{3, 2}};
  std::vector<Dims> output_shapes(2);

  ASSERT_EQ(InferUnpack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(3));
  }
}

struct UnpackErrorCase {
  std::string test_name;
  int axis;
  int num;
};

using UnpackErrorTest = ::testing::TestWithParam<UnpackErrorCase>;

TEST_P(UnpackErrorTest, RejectsNonPositiveNumAndAxisUnderflow) {
  const auto& param = GetParam();
  LiteRtOpT op = MakeUnpackOp(param.axis, param.num);
  std::vector<Dims> input_shapes = {{3, 2}};
  std::vector<Dims> output_shapes(2);
  EXPECT_EQ(InferUnpack(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorShapeInferenceFailed);
}

INSTANTIATE_TEST_SUITE_P(
    UnpackErrorTests, UnpackErrorTest,
    ::testing::Values(UnpackErrorCase{"NonPositiveNum", /*axis=*/0, /*num=*/0},
                      UnpackErrorCase{"AxisUnderflow",
                                      std::numeric_limits<int32_t>::min(),
                                      /*num=*/2}),
    [](const ::testing::TestParamInfo<UnpackErrorCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace litert::internal
