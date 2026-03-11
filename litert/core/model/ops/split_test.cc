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

#include "litert/core/model/ops/split.h"

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

std::unique_ptr<tflite::SplitOptionsT> CreateSplitOptions(int num_splits) {
  auto options = std::make_unique<tflite::SplitOptionsT>();
  options->num_splits = num_splits;
  return options;
}

TEST(SplitOpTest, TwoSplits) {
  LiteRtOpT op;
  LiteRtTensorT axis_tensor;
  LiteRtTensorT input_tensor;
  op.Inputs().push_back(&axis_tensor);
  op.Inputs().push_back(&input_tensor);

  std::vector<int32_t> axis_data = {0};
  SetWeightsFromUnownedBuffer(
      axis_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(axis_data.data()),
                                 axis_data.size() * sizeof(int32_t)));

  std::vector<Dims> input_shapes = {{}, {4, 2}};  // Axis scalar, Input
  std::vector<Dims> output_shapes(2);

  auto options = CreateSplitOptions(2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SplitOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferSplit(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(2, 2));
  }
}

TEST(SplitOpTest, TwoSplitsAxisOne) {
  LiteRtOpT op;
  LiteRtTensorT axis_tensor;
  LiteRtTensorT input_tensor;
  op.Inputs().push_back(&axis_tensor);
  op.Inputs().push_back(&input_tensor);

  std::vector<int32_t> axis_data = {1};
  SetWeightsFromUnownedBuffer(
      axis_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(axis_data.data()),
                                 axis_data.size() * sizeof(int32_t)));

  std::vector<Dims> input_shapes = {{}, {4, 2}};
  std::vector<Dims> output_shapes(2);

  auto options = CreateSplitOptions(2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SplitOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferSplit(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(4, 1));
  }
}

TEST(SplitOpTest, NegativeAxis) {
  LiteRtOpT op;
  LiteRtTensorT axis_tensor;
  LiteRtTensorT input_tensor;
  op.Inputs().push_back(&axis_tensor);
  op.Inputs().push_back(&input_tensor);

  std::vector<int32_t> axis_data = {-1};
  SetWeightsFromUnownedBuffer(
      axis_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(axis_data.data()),
                                 axis_data.size() * sizeof(int32_t)));

  std::vector<Dims> input_shapes = {{}, {4, 2}};
  std::vector<Dims> output_shapes(2);

  auto options = CreateSplitOptions(2);
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_SplitOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  ASSERT_EQ(InferSplit(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  for (const auto& s : output_shapes) {
    EXPECT_THAT(s, ElementsAre(4, 1));
  }
}

}  // namespace
}  // namespace litert::internal
