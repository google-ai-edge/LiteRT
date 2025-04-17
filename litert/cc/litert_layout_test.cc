// Copyright 2024 Google LLC.
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

#include "litert/cc/litert_layout.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert {
namespace {

using ::testing::ElementsAreArray;

const Dimensions kStaticDims = {2, 2};  // NOLINT
const Dimensions kDynDims = {-1, 2};    // NOLINT
const Strides kStrides = {1, 1};        // NOLINT

TEST(LayoutTest, BuildFromDims) {
  Layout layout(kStaticDims);
  EXPECT_EQ(layout.Rank(), 2);
  EXPECT_THAT(layout.Dimensions(), ElementsAreArray(kStaticDims));
  EXPECT_EQ(layout.HasStrides(), false);
}

TEST(LayoutTest, BuildFromDimsWithStrides) {
  Layout layout(kStaticDims, kStrides);
  EXPECT_EQ(layout.Rank(), 2);
  EXPECT_THAT(layout.Dimensions(), ElementsAreArray(kStaticDims));
  EXPECT_EQ(layout.HasStrides(), true);
  EXPECT_THAT(layout.Strides(), ElementsAreArray(kStrides));
}

TEST(LayoutTest, NumElementsStatic) {
  Layout layout(kStaticDims);
  auto num_elements = layout.NumElements();
  ASSERT_TRUE(num_elements);
  EXPECT_EQ(*num_elements, 4);
}

TEST(LayoutTest, NumElementsDynamic) {
  Layout layout(kDynDims);
  auto num_elements = layout.NumElements();
  ASSERT_FALSE(num_elements);
}

}  // namespace
}  // namespace litert
