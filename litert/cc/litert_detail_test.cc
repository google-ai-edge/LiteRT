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

#include "litert/cc/internal/litert_detail.h"

#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert {
namespace {

using ::testing::ElementsAre;

TEST(CTStringTest, ConstructFromStrLiteral) {
  EXPECT_EQ(CtStr("abc").Str(), "abc");
}

TEST(CTStringTest, ConstructFromStrData) {
  EXPECT_EQ(CtStr(CtStrData<3>{'a', 'b', 'c'}).Str(), "abc");
}

TEST(CTStringTest, ConcatSingle) { EXPECT_EQ(CtStrConcat("abc").Str(), "abc"); }

TEST(CTStringTest, ConcatMultipleSameLen) {
  EXPECT_EQ(CtStrConcat("abc", "def", "ghi").Str(), "abcdefghi");
}

TEST(CTStringTest, ConcatMultipleDifferentLen) {
  EXPECT_EQ(CtStrConcat("abc", "de", "g").Str(), "abcdeg");
}

TEST(CTStringTest, ConcatEmptyStr) {
  EXPECT_EQ(CtStrConcat("abc", "", "def").Str(), "abcdef");
}

template <typename L, typename R>
struct MyConstIntPair {
  static constexpr int kSum = L::value + R::value;
};

struct MyFunctor {
  template <typename T>
  void operator()() {
    captured.push_back(T::kSum);
  }

  std::vector<int> captured;
};

template <int V>
using CInt = std::integral_constant<int, V>;

TEST(ExpandProductTest, WithValues) {
  MyFunctor f;
  using LeftTypes = TypeList<CInt<1>, CInt<2>>;
  using RightTypes = TypeList<CInt<3>, CInt<4>>;
  ExpandProduct<MyConstIntPair, LeftTypes, RightTypes>(f);
  EXPECT_THAT(f.captured, ElementsAre(4, 5, 5, 6));
}

}  // namespace
}  // namespace litert
