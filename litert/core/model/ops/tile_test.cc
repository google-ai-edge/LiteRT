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

#include "litert/core/model/ops/tile.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

TEST(TileOpTest, SimpleTest) {
  LiteRtOpT op;
  std::vector<Dims> input_shapes = {{2, 3}, {2}};
  std::vector<Dims> output_shapes(1);

  LiteRtTensorT input_tensor;
  LiteRtTensorT mults_tensor;
  op.Inputs().push_back(&input_tensor);
  op.Inputs().push_back(&mults_tensor);

  std::vector<int32_t> mults_data = {2, 2};
  SetWeightsFromUnownedBuffer(
      mults_tensor.Weights(),
      litert::BufferRef<uint8_t>(reinterpret_cast<uint8_t*>(mults_data.data()),
                                 mults_data.size() * sizeof(int32_t)));

  ASSERT_EQ(InferTile(op, absl::MakeSpan(input_shapes), output_shapes),
            kLiteRtStatusOk);

  EXPECT_THAT(output_shapes[0], ElementsAre(4, 6));  // [2*2, 3*2]
}

TEST(TileOpTest, ReferenceTile1D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  int32_t in_dims[] = {3};
  int32_t multiples[] = {2};
  std::vector<float> output(6);

  ReferenceTile(input.data(), in_dims, multiples, 1, output.data());

  EXPECT_THAT(output, ElementsAre(1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f));
}

TEST(TileOpTest, ReferenceTile2D) {
  // Input 2x3:
  // [[1, 2, 3],
  //  [4, 5, 6]]
  // Multiples: [2, 1] -> 4x3:
  // [[1, 2, 3],
  //  [4, 5, 6],
  //  [1, 2, 3],
  //  [4, 5, 6]]
  std::vector<int32_t> input = {1, 2, 3, 4, 5, 6};
  int32_t in_dims[] = {2, 3};
  int32_t multiples[] = {2, 1};
  std::vector<int32_t> output(12);

  ReferenceTile(input.data(), in_dims, multiples, 2, output.data());

  EXPECT_THAT(output, ElementsAre(1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6));
}

TEST(TileOpTest, ReferenceTile2DBothAxes) {
  // Input 2x2:
  // [[1, 2],
  //  [3, 4]]
  // Multiples: [2, 2] -> 4x4:
  // [[1, 2, 1, 2],
  //  [3, 4, 3, 4],
  //  [1, 2, 1, 2],
  //  [3, 4, 3, 4]]
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  int32_t in_dims[] = {2, 2};
  int32_t multiples[] = {2, 2};
  std::vector<float> output(16);

  ReferenceTile(input.data(), in_dims, multiples, 2, output.data());

  EXPECT_THAT(output, ElementsAre(1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f,
                                  4.0f, 1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                                  3.0f, 4.0f));
}

}  // namespace
}  // namespace litert::internal
