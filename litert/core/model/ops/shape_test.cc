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

#include "litert/core/model/ops/shape.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/ops/test_util.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {
namespace {

TEST(ShapeOpTest, ShapeOp) {
  MockShapeInferenceContext ctx(kLiteRtOpCodeTflShape,
                                {{{2, 3, 4}, /*data=*/{}}});
  InferenceResult result;

  ASSERT_EQ(InferShape(ctx, result), kLiteRtStatusOk);

  // Output shape should be [3] (rank of input)
  EXPECT_THAT(result.output_shapes[0], testing::ElementsAre(3));

  // Output transient data should be populated with [2, 3, 4]
  ASSERT_GT(result.propagated_data[0].size(), 0);
  const int32_t* dims =
      reinterpret_cast<const int32_t*>(result.propagated_data[0].data());
  EXPECT_EQ(dims[0], 2);
  EXPECT_EQ(dims[1], 3);
  EXPECT_EQ(dims[2], 4);
}

TEST(ShapeOpTest, RankOp) {
  MockShapeInferenceContext ctx(kLiteRtOpCodeTflRank,
                                {{{2, 3, 4}, /*data=*/{}}});
  InferenceResult result;

  ASSERT_EQ(InferRank(ctx, result), kLiteRtStatusOk);

  // Output shape should be 0-D (scalar).
  EXPECT_TRUE(result.output_shapes[0].empty());

  // Output transient data should be populated with [3] (rank).
  ASSERT_GT(result.propagated_data[0].size(), 0);
  const int32_t* val =
      reinterpret_cast<const int32_t*>(result.propagated_data[0].data());
  EXPECT_EQ(val[0], 3);
}

}  // namespace
}  // namespace litert::internal
