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

#include "litert/core/litert_options_encoder.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/litert_toml_parser.h"
#include "litert/test/matchers.h"

namespace litert {
namespace internal {
namespace {

using ::testing::litert::IsOk;
using ::testing::litert::IsOkAndHolds;

TEST(LiteRtOptionsEncoderTest, EncodeScalarAndArrayTypes) {
  OptionsEncoder encoder;
  EXPECT_TRUE(encoder.Empty());

  EXPECT_THAT(encoder.EncodeBool("flag_true", true), IsOk());
  EXPECT_THAT(encoder.EncodeBool("flag_false", false), IsOk());
  EXPECT_THAT(encoder.EncodeInt("num_threads", 4), IsOk());
  EXPECT_THAT(encoder.EncodeInt("negative_val", -42), IsOk());
  EXPECT_THAT(encoder.EncodeString("cache_dir", "/data/user/0/cache"), IsOk());
  EXPECT_THAT(encoder.EncodeString("empty_str", ""), IsOk());
  const std::vector<std::string> patterns = {"kv_cache_", "mask,0"};
  EXPECT_THAT(encoder.EncodeStringArray("patterns", patterns), IsOk());
  const std::vector<absl::string_view> empty_arr = {};
  EXPECT_THAT(encoder.EncodeStringArray("empty_arr", empty_arr), IsOk());

  EXPECT_FALSE(encoder.Empty());
  EXPECT_EQ(encoder.Str(),
            "flag_true = true\n"
            "flag_false = false\n"
            "num_threads = 4\n"
            "negative_val = -42\n"
            "cache_dir = \"/data/user/0/cache\"\n"
            "empty_str = \"\"\n"
            "patterns = [\"kv_cache_\", \"mask,0\"]\n"
            "empty_arr = []\n");
}

TEST(LiteRtOptionsEncoderTest, RoundTripsWithOptionsDecoder) {
  OptionsEncoder encoder;
  ASSERT_THAT(encoder.EncodeBool("enable_sharing", true), IsOk());
  ASSERT_THAT(
      encoder.EncodeInt("int64_min", std::numeric_limits<int64_t>::min()),
      IsOk());
  ASSERT_THAT(
      encoder.EncodeInt("int64_max", std::numeric_limits<int64_t>::max()),
      IsOk());
  ASSERT_THAT(encoder.EncodeInt(
                  "uint32_max",
                  static_cast<int64_t>(std::numeric_limits<uint32_t>::max())),
              IsOk());
  ASSERT_THAT(encoder.EncodeString("path_with_eq_and_quote", "a=b'c d"),
              IsOk());
  ASSERT_THAT(encoder.EncodeString("win_path", "C:\\Users\\litert\\cache\\"),
              IsOk());
  const std::vector<absl::string_view> items = {"1", "2,3", "it's ok"};
  ASSERT_THAT(encoder.EncodeStringArray("items", items), IsOk());

  std::string encoded = encoder.Release();
  EXPECT_TRUE(encoder.Empty());

  int count = 0;
  EXPECT_THAT(
      ParseToml(
          encoded,
          [&](absl::string_view key, absl::string_view value) {
            ++count;
            if (key == "enable_sharing") {
              EXPECT_THAT(ParseTomlBool(value), IsOkAndHolds(true));
            } else if (key == "int64_min") {
              EXPECT_THAT(ParseTomlInt(value),
                          IsOkAndHolds(std::numeric_limits<int64_t>::min()));
            } else if (key == "int64_max") {
              EXPECT_THAT(ParseTomlInt(value),
                          IsOkAndHolds(std::numeric_limits<int64_t>::max()));
            } else if (key == "uint32_max") {
              EXPECT_THAT(ParseTomlInt(value),
                          IsOkAndHolds(static_cast<int64_t>(
                              std::numeric_limits<uint32_t>::max())));
            } else if (key == "path_with_eq_and_quote") {
              EXPECT_EQ(value, "a=b'c d");
            } else if (key == "win_path") {
              EXPECT_EQ(value, "C:\\Users\\litert\\cache\\");
            } else if (key == "items") {
              EXPECT_THAT(ParseTomlStringArray(value),
                          IsOkAndHolds(
                              std::vector<std::string>{"1", "2,3", "it's ok"}));
            } else {
              ADD_FAILURE() << "Unexpected key: " << key;
            }
            return kLiteRtStatusOk;
          }),
      IsOk());
  EXPECT_EQ(count, 7);
}

TEST(LiteRtOptionsEncoderTest, ReleaseAndClearAllowReuse) {
  OptionsEncoder encoder;
  ASSERT_THAT(encoder.EncodeInt("a", 1), IsOk());
  EXPECT_EQ(encoder.Release(), "a = 1\n");
  EXPECT_TRUE(encoder.Empty());

  ASSERT_THAT(encoder.EncodeBool("b", false), IsOk());
  encoder.Clear();
  EXPECT_TRUE(encoder.Empty());

  ASSERT_THAT(encoder.EncodeString("c", "ok"), IsOk());
  EXPECT_EQ(encoder.Str(), "c = \"ok\"\n");
}

TEST(LiteRtOptionsEncoderTest,
     RejectsInvalidKeysAndValuesWithoutMutatingBuffer) {
  OptionsEncoder encoder;
  ASSERT_THAT(encoder.EncodeInt("valid", 1), IsOk());
  const std::string expected_snapshot(encoder.Str());

  // Invalid keys.
  EXPECT_EQ(encoder.EncodeBool("", true), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(encoder.EncodeInt("bad=key", 1), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(encoder.EncodeInt("bad key", 1), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(encoder.EncodeInt("bad\tkey", 1),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(encoder.EncodeInt("bad\nkey", 1),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(encoder.EncodeInt("bad\rkey", 1),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(encoder.EncodeInt("#comment", 1),
            kLiteRtStatusErrorInvalidArgument);

  // Invalid string values.
  EXPECT_EQ(encoder.EncodeString("k", "has\"quote"),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(encoder.EncodeString("k", "has\nnewline"),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(encoder.EncodeString("k", "has\rcr"),
            kLiteRtStatusErrorInvalidArgument);

  // Invalid array element leaves buffer untouched even if first element is ok.
  const std::vector<std::string> bad_array = {"good", "bad\"quote"};
  EXPECT_EQ(encoder.EncodeStringArray("arr", bad_array),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(encoder.Str(), expected_snapshot);
}

}  // namespace
}  // namespace internal
}  // namespace litert
