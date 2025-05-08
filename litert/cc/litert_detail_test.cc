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

#include "litert/cc/litert_detail.h"

#include <gtest/gtest.h>

namespace litert {
namespace {

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

}  // namespace
}  // namespace litert
