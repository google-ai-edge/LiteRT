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

#include "litert/vendors/nvidia/dispatch/tensor_buffer_layout.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace litert::nvidia {
namespace {

TEST(TensorBufferLayoutTest, RequiresExplicitProducerContract) {
  LiteRtLayout layout{4, false, {1, 8, 256, 1152}, {}};
  EXPECT_TRUE(TensorBufferLayoutMatches(layout, false));
  EXPECT_FALSE(TensorBufferLayoutMatches(layout, true));

  ASSERT_TRUE(GetTransposedValueCacheStrides(layout, layout.strides));
  layout.has_strides = true;
  EXPECT_EQ(layout.strides[0], 8 * 1152 * 256);
  EXPECT_EQ(layout.strides[1], 1152 * 256);
  EXPECT_EQ(layout.strides[2], 1);
  EXPECT_EQ(layout.strides[3], 256);
  EXPECT_TRUE(TensorBufferLayoutMatches(layout, true));
  EXPECT_FALSE(TensorBufferLayoutMatches(layout, false));
}

TEST(TensorBufferLayoutTest, AcceptsExplicitDenseLayoutOnlyAsDense) {
  LiteRtLayout layout{
      4, true, {1, 8, 256, 1152}, {8 * 256 * 1152, 256 * 1152, 1152, 1}};
  EXPECT_TRUE(TensorBufferLayoutMatches(layout, false));
  EXPECT_FALSE(TensorBufferLayoutMatches(layout, true));
  ++layout.strides[1];
  EXPECT_FALSE(TensorBufferLayoutMatches(layout, false));
  EXPECT_FALSE(TensorBufferLayoutMatches(layout, true));
}

TEST(TensorBufferLayoutTest, RejectsInvalidCacheShapeAndOverflow) {
  LiteRtLayout layout{4, false, {1, 8, 256, 1152}, {}};
  uint32_t strides[4];
  layout.rank = 3;
  EXPECT_FALSE(GetTransposedValueCacheStrides(layout, strides));
  layout.rank = 4;
  layout.dimensions[0] = 2;
  EXPECT_FALSE(GetTransposedValueCacheStrides(layout, strides));
  layout.dimensions[0] = 1;
  layout.dimensions[3] = -1;
  EXPECT_FALSE(GetTransposedValueCacheStrides(layout, strides));
  layout.dimensions[3] = 0;
  EXPECT_FALSE(GetTransposedValueCacheStrides(layout, strides));
  layout.dimensions[3] = std::numeric_limits<int32_t>::max();
  EXPECT_FALSE(GetTransposedValueCacheStrides(layout, strides));
}

TEST(TensorBufferLayoutTest, MapsLogicalCoordinatesToSequenceMajorOffsets) {
  LiteRtLayout layout{4, false, {1, 2, 3, 5}, {}};
  ASSERT_TRUE(GetTransposedValueCacheStrides(layout, layout.strides));
  for (int h = 0; h < 2; ++h) {
    for (int d = 0; d < 3; ++d) {
      for (int s = 0; s < 5; ++s) {
        const int physical_index = (h * 5 + s) * 3 + d;
        EXPECT_EQ(h * layout.strides[1] + d * layout.strides[2] +
                      s * layout.strides[3],
                  physical_index);
      }
    }
  }
}

}  // namespace
}  // namespace litert::nvidia
