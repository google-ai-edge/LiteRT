// Copyright 2025 Google LLC.
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

#include "litert/test/generators/reference.h"

#include <functional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert {
namespace testing {
namespace {

using ::testing::ElementsAreArray;

TEST(ElementWiseTest, NoBroadcastReference) {
  ElementWiseComputation comp;
  comp.InShape({2, 3}).InShape({2, 3}).OutShape({2, 3});
  const float lhs[4] = {1, 2, 3, 4};
  const float rhs[4] = {7, 8, 9, 10};
  float output[4];
  ASSERT_TRUE(comp.Compute(std::plus<float>(), output, lhs, rhs));
}

TEST(ElementWiseTest, LeftBroadcast) {
  ElementWiseComputation comp;
  comp.InShape({2, 1}).InShape({2, 2}).OutShape({2, 2});
  const float lhs[2] = {1, 2};
  const float rhs[4] = {7, 8, 9, 10};
  float output[4];
  ASSERT_TRUE(comp.Compute(std::plus<float>(), output, lhs, rhs));
  EXPECT_THAT(output, ElementsAreArray({8, 9, 11, 12}));
}

TEST(ElementWiseTest, RightBroadcast) {
  ElementWiseComputation comp;
  comp.InShape({2, 2}).InShape({1, 2}).OutShape({2, 2});
  const float lhs[4] = {1, 2, 3, 4};
  const float rhs[2] = {-1, -2};
  float output[4];
  ASSERT_TRUE(comp.Compute(std::plus<float>(), output, lhs, rhs));
  EXPECT_THAT(output, ElementsAreArray({0, 0, 2, 2}));
}

TEST(ElementWiseTest, BothBroadcast) {
  ElementWiseComputation comp;
  comp.InShape({2, 1}).InShape({1, 2}).OutShape({2, 2});
  const float lhs[2] = {1, 2};
  const float rhs[2] = {-1, -2};
  float output[4];
  ASSERT_TRUE(comp.Compute(std::plus<float>(), output, lhs, rhs));
  EXPECT_THAT(output, ElementsAreArray({0, -1, 1, 0}));
}

TEST(ElementWiseTest, BadShape) {
  ElementWiseComputation comp;
  comp.InShape({2, 1}).InShape({3, 2}).OutShape({2, 2});
  const float lhs[2] = {1, 2};
  const float rhs[2] = {-1, -2};
  float output[4];
  ASSERT_FALSE(comp.Compute(std::plus<float>(), output, lhs, rhs));
}

TEST(ElementWiseTest, Ternary) {
  ElementWiseComputation comp;
  comp.InShape({2, 1}).InShape({1, 2}).InShape({2, 2}).OutShape({2, 2});
  const float lhs[2] = {1, 2};
  const float rhs[2] = {-1, -2};
  const float rhs2[4] = {2, 3, 4, 5};
  float output[4];
  ASSERT_TRUE(comp.Compute(std::plus<float>(), output, lhs, rhs, rhs2));
  EXPECT_THAT(output, ElementsAreArray({2, 2, 5, 5}));
}

TEST(UnaryTest, Unary) {
  ElementWiseComputation comp;
  comp.InShape({2, 2}).OutShape({2, 2});
  const float lhs[4] = {1, 2, 3, 4};
  float output[4];
  ASSERT_TRUE(comp.Compute(std::negate<float>(), output, lhs));
  EXPECT_THAT(output, ElementsAreArray({-1, -2, -3, -4}));
}

}  // namespace
}  // namespace testing
}  // namespace litert
