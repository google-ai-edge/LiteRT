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

#include "litert/core/model/ops/slice.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(SliceOpTest, DynamicUpdateSlice) {
  LiteRtOpT op;
  // DynamicUpdateSlice takes 3 inputs: operand, update, start_indices.
  // Output shape is same as operand.
  std::vector<Dims> input_shapes = {
      {1, 10, 10, 3},  // operand
      {1, 2, 2, 3},    // update
      {4}              // start_indices
  };
  std::vector<Dims> output_shapes(1);

  // Setup inputs
  LiteRtTensorT operand;
  LiteRtTensorT update;
  LiteRtTensorT start_indices;

  op.Inputs().push_back(&operand);
  op.Inputs().push_back(&update);
  op.Inputs().push_back(&start_indices);

  ASSERT_EQ(
      InferDynamicUpdateSlice(op, absl::MakeSpan(input_shapes), output_shapes),
      kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 10, 10, 3));
}

TEST(SliceOpTest, SliceStatic) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {
      {1, 10, 10, 3}, {4}, {4}};  // Input, Begin, Size
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT size_tensor;
  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&size_tensor);

  int32_t size_data[] = {1, 5, 5, 3};
  SetWeightsFromOwnedBuffer(
      size_tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(size_data),
                               sizeof(size_data)));

  ASSERT_EQ(InferSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(1, 5, 5, 3));
}

TEST(SliceOpTest, StridedSliceSimple) {
  LiteRtOpT op;
  // Input [2, 3, 4]
  std::vector<Dims> input_shapes = {
      {2, 3, 4}, {3}, {3}, {3}};  // Input, Begin, End, Strides
  std::vector<Dims> output_shapes(1);

  // Setup inputs for begin/end/strides (dummy rank 1 tensors)
  LiteRtTensorT begin;
  LiteRtTensorT end;
  LiteRtTensorT strides;
  // Set shapes for begin/end/strides to match '3'
  begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));

  op.Inputs().push_back(nullptr);  // Input
  op.Inputs().push_back(&begin);
  op.Inputs().push_back(&end);
  op.Inputs().push_back(&strides);

  auto options = std::make_unique<tflite::StridedSliceOptionsT>();
  options->begin_mask = 0;
  options->end_mask = 0;
  options->ellipsis_mask = 0;
  options->new_axis_mask = 0;
  options->shrink_axis_mask = 0;

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_StridedSliceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  // The simplified logic sets output dims to -1 if not shrink/new_axis.
  ASSERT_EQ(InferStridedSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  // Expect [ -1, -1, -1 ]
  EXPECT_THAT(output_shapes[0], ElementsAre(-1, -1, -1));
}

TEST(SliceOpTest, StridedSliceShrinkAxis) {
  LiteRtOpT op;
  // Input [2, 3, 4]
  std::vector<Dims> input_shapes = {{2, 3, 4}, {3}, {3}, {3}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT begin;
  begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));

  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(&begin);
  op.Inputs().push_back(nullptr);
  op.Inputs().push_back(nullptr);

  auto options = std::make_unique<tflite::StridedSliceOptionsT>();
  options->shrink_axis_mask = 2;  // Shrink axis 1 (bit 1 set: 1<<1 = 2)

  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_StridedSliceOptions;
  tfl_options.value = options.release();
  SetTflOptions(op, std::move(tfl_options));

  // Axis 1 removed.
  // Expect [ -1, -1 ]
  ASSERT_EQ(InferStridedSlice(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);
  EXPECT_THAT(output_shapes[0], ElementsAre(-1, -1));
}

}  // namespace
}  // namespace litert::internal
