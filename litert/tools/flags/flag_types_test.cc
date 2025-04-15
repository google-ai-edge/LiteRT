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

}  // namespace
}  // namespace litert::tools
