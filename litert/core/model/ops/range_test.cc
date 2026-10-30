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

#include "litert/core/model/ops/range.h"

#include <cstdint>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/ops/test_util.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {
namespace {

TEST(RangeOpTest, RangeInt32) {
  auto start = MakeTensorData({}, std::vector<int32_t>{0});
  auto limit = MakeTensorData({}, std::vector<int32_t>{10});
  auto delta = MakeTensorData({}, std::vector<int32_t>{2});
  MockShapeInferenceContext ctx(kLiteRtOpCodeTflRange, {start, limit, delta});
  InferenceResult result;

  ASSERT_EQ(InferRange(ctx, result), kLiteRtStatusOk);

  // Output shape should be [5]
  EXPECT_THAT(result.output_shapes[0], testing::ElementsAre(5));

  // Output transient data should be populated with [0, 2, 4, 6, 8]
  ASSERT_GT(result.propagated_data[0].size(), 0);
  const int32_t* data =
      reinterpret_cast<const int32_t*>(result.propagated_data[0].data());
  EXPECT_THAT(std::vector<int32_t>(data, data + 5),
              testing::ElementsAre(0, 2, 4, 6, 8));
}

TEST(RangeOpTest, RejectsShortInputBuffersAndElementCountOverflow) {
  auto valid = MakeTensorData({}, std::vector<int32_t>{1});
  MockShapeInferenceContext::TensorData short_buf = {{}, {1, 2}};

  MockShapeInferenceContext short_ctx(kLiteRtOpCodeTflRange,
                                      {short_buf, valid, valid});
  InferenceResult short_res;
  EXPECT_EQ(InferRange(short_ctx, short_res),
            kLiteRtStatusErrorInvalidArgument);

  // Range count exceeding INT32_MAX (INT32_MIN to INT32_MAX with delta 1) is
  // rejected without signed integer overflow.
  auto min_start = MakeTensorData(
      {}, std::vector<int32_t>{std::numeric_limits<int32_t>::min()});
  auto max_limit = MakeTensorData(
      {}, std::vector<int32_t>{std::numeric_limits<int32_t>::max()});
  auto delta_one = MakeTensorData({}, std::vector<int32_t>{1});
  MockShapeInferenceContext overflow_ctx(kLiteRtOpCodeTflRange,
                                         {min_start, max_limit, delta_one});
  InferenceResult overflow_res;
  EXPECT_EQ(InferRange(overflow_ctx, overflow_res),
            kLiteRtStatusErrorShapeInferenceFailed);

  // Range count exceeding kMaxPropagatedRangeElements (1 << 20) infers shape
  // without allocating huge propagated_data.
  constexpr int32_t kLargeCount = (1 << 20) + 10;
  auto zero_start = MakeTensorData({}, std::vector<int32_t>{0});
  auto large_limit = MakeTensorData({}, std::vector<int32_t>{kLargeCount});
  MockShapeInferenceContext large_ctx(kLiteRtOpCodeTflRange,
                                      {zero_start, large_limit, delta_one});
  InferenceResult large_res;
  ASSERT_EQ(InferRange(large_ctx, large_res), kLiteRtStatusOk);
  EXPECT_THAT(large_res.output_shapes[0], testing::ElementsAre(kLargeCount));
  EXPECT_TRUE(large_res.propagated_data.empty());
}

}  // namespace
}  // namespace litert::internal
