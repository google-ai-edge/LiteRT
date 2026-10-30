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

#include "litert/core/model/ops/pad.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(PadOpTest, Pad) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {3, 2}};
  std::vector<Dims> output_shapes(1);

  // Paddings tensor (input 1): [rank, 2] -> [3, 2]
  // Let's pad dim 1 by 1 on each side.

  op.Inputs().clear();
  LiteRtTensorT input0;
  LiteRtTensorT pads;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&pads);

  int32_t pads_data[] = {
      0, 0,  // dim 0
      1, 1,  // dim 1
      0, 0   // dim 2
  };
  SetWeightsFromOwnedBuffer(
      pads.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(pads_data),
                               sizeof(pads_data)));
  pads.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3, 2}));

  ASSERT_EQ(InferPad(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 4, 3));
}

TEST(PadOpTest, Padv2) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {
      {1, 2, 3}, {3, 2}, {}};  // Input, Paddings, Constant
  std::vector<Dims> output_shapes(1);

  op.Inputs().clear();
  LiteRtTensorT input0;
  LiteRtTensorT pads;
  LiteRtTensorT constant_values;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&pads);
  op.Inputs().push_back(&constant_values);  // Padv2 takes 3 inputs

  int32_t pads_data[] = {
      0, 0,  // dim 0
      1, 1,  // dim 1
      0, 0   // dim 2
  };
  SetWeightsFromOwnedBuffer(
      pads.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(pads_data),
                               sizeof(pads_data)));
  pads.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3, 2}));

  ASSERT_EQ(InferPadv2(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 4, 3));
}

TEST(PadOpTest, MirrorPad) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{1, 2, 3}, {3, 2}};
  std::vector<Dims> output_shapes(1);

  op.Inputs().clear();
  LiteRtTensorT input0;
  LiteRtTensorT pads;
  op.Inputs().push_back(&input0);
  op.Inputs().push_back(&pads);

  int32_t pads_data[] = {
      0, 0,  // dim 0
      1, 1,  // dim 1
      0, 0   // dim 2
  };
  SetWeightsFromOwnedBuffer(
      pads.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(pads_data),
                               sizeof(pads_data)));
  pads.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3, 2}));

  ASSERT_EQ(InferMirrorPad(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(1, 4, 3));
}

struct PadErrorCase {
  std::string test_name;
  bool omit_inputs;
  Dims input_shape;
  std::vector<int32_t> paddings;
};

using PadErrorTest = ::testing::TestWithParam<PadErrorCase>;

TEST_P(PadErrorTest, RejectsMissingInputsNegativePaddingsAndDimensionOverflow) {
  const auto& param = GetParam();
  std::vector<Dims> input_shapes = {param.input_shape, {2, 2}};
  std::vector<Dims> output_shapes(1);

  LiteRtOpT op;
  LiteRtTensorT input0;
  LiteRtTensorT pads;
  if (!param.omit_inputs) {
    pads.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {2, 2}));
    SetWeightsFromOwnedBuffer(
        pads.Weights(),
        OwningBufferRef<uint8_t>(
            reinterpret_cast<const uint8_t*>(param.paddings.data()),
            param.paddings.size() * sizeof(int32_t)));
    op.Inputs() = {&input0, &pads};
  }

  EXPECT_EQ(InferPad(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusErrorInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    PadErrorTests, PadErrorTest,
    ::testing::Values(
        PadErrorCase{"MissingInputs", /*omit_inputs=*/true, {1, 2}, {}},
        PadErrorCase{
            "NegativePadding", /*omit_inputs=*/false, {1, 2}, {0, 0, -1, 1}},
        PadErrorCase{"PaddedDimensionOverflow",
                     /*omit_inputs=*/false,
                     {1, std::numeric_limits<int32_t>::max() - 1},
                     {0, 0, 2, 2}}),
    [](const ::testing::TestParamInfo<PadErrorCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace litert::internal
