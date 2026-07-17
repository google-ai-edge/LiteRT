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

#include "litert/c/litert_layout.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"

namespace {

TEST(LiteRtLayoutTest, GetNumLayoutElementsRejectsRankPastStorage) {
  LiteRtLayout layout = {};
  layout.rank = LITERT_TENSOR_MAX_RANK + 1;
  for (int i = 0; i < LITERT_TENSOR_MAX_RANK; ++i) {
    layout.dimensions[i] = 1;
  }

  size_t num_elements = 0;
  EXPECT_EQ(LiteRtGetNumLayoutElements(&layout, &num_elements),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtLayoutTest, GetNumLayoutElementsRejectsOverflow) {
  LiteRtLayout layout = {};
  layout.rank = 3;
  layout.dimensions[0] = std::numeric_limits<int32_t>::max();
  layout.dimensions[1] = std::numeric_limits<int32_t>::max();
  layout.dimensions[2] = std::numeric_limits<int32_t>::max();

  size_t num_elements = 0;
  EXPECT_EQ(LiteRtGetNumLayoutElements(&layout, &num_elements),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtLayoutTest, IsSameLayoutRejectsRankPastStorage) {
  LiteRtLayout layout = {};
  layout.rank = LITERT_TENSOR_MAX_RANK + 1;

  LiteRtLayout other = {};
  other.rank = 1;
  other.dimensions[0] = 1;

  bool result = false;
  EXPECT_EQ(LiteRtIsSameLayout(&layout, &other, &result),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtIsSameLayout(&other, &layout, &result),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtLayoutTest, GetNumLayoutElementsValidLayout) {
  LiteRtLayout layout = {};
  layout.rank = 3;
  layout.dimensions[0] = 2;
  layout.dimensions[1] = 3;
  layout.dimensions[2] = 5;

  size_t num_elements = 0;
  EXPECT_EQ(LiteRtGetNumLayoutElements(&layout, &num_elements),
            kLiteRtStatusOk);
  EXPECT_EQ(num_elements, 30);
}

}  // namespace
