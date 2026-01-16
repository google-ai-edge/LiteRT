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

#include "litert/tools/flags/flag_types.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::tools {
namespace {

using ::testing::ElementsAre;

TEST(IntListFlagTest, ParseEmpty) {
  std::string error;
  IntList value;

  EXPECT_TRUE(AbslParseFlag("", &value, &error));
  EXPECT_THAT(value.elements, ElementsAre());
  EXPECT_EQ("", AbslUnparseFlag(value));
}

TEST(IntListFlagTest, MultiInt) {
  std::string error;
  IntList value;

  EXPECT_TRUE(AbslParseFlag("1,2,3", &value, &error));
  EXPECT_THAT(value.elements, ElementsAre(1, 2, 3));
  EXPECT_EQ("1,2,3", AbslUnparseFlag(value));
}

TEST(IntListFlagTest, MultiUint32) {
  std::string error;
  IntList<std::uint32_t> value;

  EXPECT_TRUE(AbslParseFlag("1,2,3", &value, &error));
  EXPECT_THAT(value.elements, ElementsAre(1, 2, 3));
  EXPECT_EQ("1,2,3", AbslUnparseFlag(value));
}

TEST(IntListFlagTest, SingleInt) {
  std::string error;
  IntList value;

  EXPECT_TRUE(AbslParseFlag("1", &value, &error));
  EXPECT_THAT(value.elements, ElementsAre(1));
  EXPECT_EQ("1", AbslUnparseFlag(value));
}

TEST(IntListFlagTest, NotInt) {
  std::string error;
  IntList value;

  EXPECT_FALSE(AbslParseFlag("1,2,3,not_int", &value, &error));
}

TEST(IntListMapFlagTest, ParseEmpty) {
  std::string error;
  IntListMap value;

  EXPECT_TRUE(AbslParseFlag("", &value, &error));
  EXPECT_TRUE(value.elements.empty());
  EXPECT_EQ("", AbslUnparseFlag(value));
}

TEST(IntListMapFlagTest, ParseSimple) {
  std::string error;
  IntListMap value;

  EXPECT_TRUE(AbslParseFlag("0|1,2,3;1|4,5", &value, &error));
  ASSERT_EQ(value.elements.size(), 2);
  EXPECT_THAT(value.elements[0], ElementsAre(1, 2, 3));
  EXPECT_THAT(value.elements[1], ElementsAre(4, 5));
  EXPECT_EQ("0|1,2,3;1|4,5", AbslUnparseFlag(value));
}

TEST(IntListMapFlagTest, ParseWithRanges) {
  std::string error;
  IntListMap value;

  EXPECT_TRUE(AbslParseFlag("0|1-3,5;2|8", &value, &error));
  ASSERT_EQ(value.elements.size(), 2);
  EXPECT_THAT(value.elements[0], ElementsAre(1, 2, 3, 5));
  EXPECT_THAT(value.elements[2], ElementsAre(8));
  EXPECT_EQ("0|1,2,3,5;2|8", AbslUnparseFlag(value));
}

TEST(IntListMapFlagTest, ParseUnordered) {
  std::string error;
  IntListMap value;

  EXPECT_TRUE(AbslParseFlag("2|3;0|1", &value, &error));
  ASSERT_EQ(value.elements.size(), 2);
  EXPECT_THAT(value.elements[0], ElementsAre(1));
  EXPECT_THAT(value.elements[2], ElementsAre(3));
  // Unparse should order by key
  EXPECT_EQ("0|1;2|3", AbslUnparseFlag(value));
}

TEST(IntListMapFlagTest, ParseInvalidFormat) {
  std::string error;
  IntListMap value;

  EXPECT_FALSE(AbslParseFlag("0|1,2;invalid", &value, &error));
  EXPECT_EQ(error, "Invalid format: invalid");
}

}  // namespace
}  // namespace litert::tools
