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

#include "litert/core/litert_toml_parser.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"

namespace litert {
namespace internal {
namespace {

using ::testing::litert::IsOk;
using ::testing::litert::IsOkAndHolds;

TEST(LiteRtTomlParserTest, ParseBool) {
  EXPECT_THAT(ParseTomlBool("true"), IsOkAndHolds(true));
  EXPECT_THAT(ParseTomlBool("false"), IsOkAndHolds(false));
  EXPECT_FALSE(ParseTomlBool("invalid").HasValue());
}

TEST(LiteRtTomlParserTest, ParseInt) {
  EXPECT_THAT(ParseTomlInt("123"), IsOkAndHolds(123));
  EXPECT_THAT(ParseTomlInt("-456"), IsOkAndHolds(-456));
  EXPECT_FALSE(ParseTomlInt("invalid").HasValue());
}

TEST(LiteRtTomlParserTest, ParseTomlKeyValues) {
  std::string toml_str = R"(
    # Comment
    key1 = value1
    key2 = value2
    key3  =  value3
  )";

  int call_count = 0;
  EXPECT_THAT(ParseToml(toml_str,
                        [&](absl::string_view key, absl::string_view value) {
                          call_count++;
                          if (key == "key1") {
                            EXPECT_EQ(value, "value1");
                          } else if (key == "key2") {
                            EXPECT_EQ(value, "value2");
                          } else if (key == "key3") {
                            EXPECT_EQ(value, "value3");
                          } else {
                            ADD_FAILURE() << "Unexpected key: " << key;
                          }
                          return kLiteRtStatusOk;
                        }),
              IsOk());
  EXPECT_EQ(call_count, 3);
}

TEST(LiteRtTomlParserTest, ParseTomlEmpty) {
  EXPECT_THAT(ParseToml("", [](auto, auto) { return kLiteRtStatusOk; }),
              IsOk());
  EXPECT_THAT(ParseToml("   \n  ", [](auto, auto) { return kLiteRtStatusOk; }),
              IsOk());
}

}  // namespace
}  // namespace internal
}  // namespace litert
