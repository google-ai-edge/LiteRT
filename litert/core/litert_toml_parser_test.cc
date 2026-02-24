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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"
#define TOML_IMPLEMENTATION  // Include the TOML implementation in this file.
#define TOML_EXCEPTIONS 0    // Don't use exceptions for TOML parsing.
#include "toml.hpp"  // from @tomlplusplus

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

TEST(LiteRtTomlParserTest, ParseString) {
  EXPECT_THAT(ParseTomlString("\"hello\""), IsOkAndHolds("hello"));
  EXPECT_THAT(ParseTomlString("'world'"), IsOkAndHolds("world"));
  EXPECT_THAT(ParseTomlString("\"\""), IsOkAndHolds(""));
  EXPECT_FALSE(ParseTomlString("invalid").HasValue());
  EXPECT_FALSE(ParseTomlString("\"invalid").HasValue());
  EXPECT_FALSE(ParseTomlString("'invalid").HasValue());
}

TEST(LiteRtTomlParserTest, ParseStringArray) {
  EXPECT_THAT(ParseTomlStringArray("[\"a\", \"b\", \"c\"]"),
              IsOkAndHolds(std::vector<std::string>{"a", "b", "c"}));
  EXPECT_THAT(ParseTomlStringArray("['single']"),
              IsOkAndHolds(std::vector<std::string>{"single"}));
  EXPECT_THAT(ParseTomlStringArray("[]"),
              IsOkAndHolds(std::vector<std::string>{}));
  EXPECT_THAT(ParseTomlStringArray("[  \"a\"  ,  \"str with space\"  ]"),
              IsOkAndHolds(std::vector<std::string>{"a", "str with space"}));
  EXPECT_THAT(ParseTomlStringArray("[\"1,2\", \"3\"]"),
              IsOkAndHolds(std::vector<std::string>{"1,2", "3"}));
  EXPECT_THAT(ParseTomlStringArray("['1,2', '3']"),
              IsOkAndHolds(std::vector<std::string>{"1,2", "3"}));
  // Optional trailing comma
  EXPECT_THAT(ParseTomlStringArray("[\"a\", ]"),
              IsOkAndHolds(std::vector<std::string>{"a"}));

  EXPECT_FALSE(ParseTomlStringArray("invalid").HasValue());
  EXPECT_FALSE(ParseTomlStringArray("[\"invalid\"").HasValue());
  EXPECT_FALSE(ParseTomlStringArray("[\"invalid\", noquotes]").HasValue());
}

TEST(LiteRtTomlParserTest, ParseTomlKeyValues) {
  std::string toml_str = R"(
    # Comment
    key1 = 1234
    key2 = true
    key3  =  "value3"
    key4 = "value4"
    key5 = ["item1" ,"item2", "item3"]
  )";

  auto tbl = toml::parse(toml_str);
  if (tbl.failed()) {
    ADD_FAILURE() << "tomlplusplus parse failed: " << tbl.error().description();
  }

  int call_count = 0;
  EXPECT_THAT(ParseToml(toml_str,
                        [&](absl::string_view key, absl::string_view value) {
                          call_count++;
                          if (key == "key1") {
                            EXPECT_THAT(ParseTomlInt(value),
                                        IsOkAndHolds(1234));
                          } else if (key == "key2") {
                            EXPECT_THAT(ParseTomlBool(value),
                                        IsOkAndHolds(true));
                          } else if (key == "key3") {
                            EXPECT_EQ(value, "value3");
                          } else if (key == "key4") {
                            EXPECT_EQ(value, "value4");
                          } else if (key == "key5") {
                            EXPECT_EQ(value,
                                      "[\"item1\" ,\"item2\", \"item3\"]");

                            auto parsed_array = ParseTomlStringArray(value);
                            EXPECT_TRUE(parsed_array.HasValue());
                            auto toml_array = tbl["key5"].as_array();
                            EXPECT_NE(toml_array, nullptr);
                            if (parsed_array.HasValue() &&
                                toml_array != nullptr) {
                              std::vector<std::string> expected_array;
                              for (const auto& item : *toml_array) {
                                expected_array.push_back(
                                    item.value<std::string>().value_or(""));
                              }
                              EXPECT_EQ(*parsed_array, expected_array);
                            }
                          } else {
                            ADD_FAILURE() << "Unexpected key: " << key;
                          }
                          return kLiteRtStatusOk;
                        }),
              IsOk());
  EXPECT_EQ(call_count, 5);
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
