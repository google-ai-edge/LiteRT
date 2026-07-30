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

#include "litert/core/model/ops/broadcast_args.h"

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

TEST(BroadcastArgsOpTest, BasicBroadcast) {
  auto s1 = MakeTensorData({3}, std::vector<int32_t>{2, 1, 4});
  auto s2 = MakeTensorData({2}, std::vector<int32_t>{3, 4});
  MockShapeInferenceContext ctx(kLiteRtOpCodeTflBroadcastArgs, {s1, s2});
  InferenceResult result;

  ASSERT_EQ(InferBroadcastArgs(ctx, result), kLiteRtStatusOk);

  // Output shape should be [3] (rank of broadcast output)
  EXPECT_THAT(result.output_shapes[0], testing::ElementsAre(3));

  // Output transient data should be populated with [2, 3, 4]
  ASSERT_GT(result.propagated_data[0].size(), 0);
  const int32_t* data =
      reinterpret_cast<const int32_t*>(result.propagated_data[0].data());
  EXPECT_THAT(std::vector<int32_t>(data, data + 3),
              testing::ElementsAre(2, 3, 4));
}

}  // namespace
}  // namespace litert::internal
