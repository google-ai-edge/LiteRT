// Copyright 2026 Amlogic, Inc.
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

#include "litert/c/options/litert_amlogic_options.h"

#include <gtest/gtest.h>
#include "litert/test/matchers.h"

namespace litert::amlogic {
namespace {

void SerializeAndParse(LrtAmlogicOptions payload, LrtAmlogicOptions* parsed) {
  const char* identifier = nullptr;
  void* raw_payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueAmlogicOptionsData(payload, &identifier,
                                              &raw_payload, &payload_deleter));
  EXPECT_STREQ(identifier, "Amlogic");
  const char* toml_str = static_cast<const char*>(raw_payload);
  LITERT_ASSERT_OK(LrtCreateAmlogicOptionsFromToml(toml_str, parsed));
  payload_deleter(raw_payload);
}

TEST(LrtAmlogicOptionsTest, GetOpaqueDataEmpty) {
  LrtAmlogicOptions options = nullptr;
  LITERT_ASSERT_OK(LrtCreateAmlogicOptions(&options));

  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueAmlogicOptionsData(options, &identifier, &payload,
                                              &payload_deleter));
  EXPECT_STREQ(identifier, "Amlogic");
  EXPECT_STREQ(static_cast<const char*>(payload), "");

  payload_deleter(payload);
  LrtDestroyAmlogicOptions(options);
}

TEST(LrtAmlogicOptionsTest, GetUnsetReturnsInvalidArgument) {
  LrtAmlogicOptions options = nullptr;
  LITERT_ASSERT_OK(LrtCreateAmlogicOptions(&options));

  const char* model_path = nullptr;
  EXPECT_EQ(LrtAmlogicOptionsGetModelPath(options, &model_path),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LrtAmlogicOptionsGetModelName(options, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LrtAmlogicOptionsSetModelPath(nullptr, "/tmp"),
            kLiteRtStatusErrorInvalidArgument);

  LrtDestroyAmlogicOptions(options);
}

TEST(LrtAmlogicOptionsTest, ModelPathAndNameRoundTrip) {
  LrtAmlogicOptions options = nullptr;
  LITERT_ASSERT_OK(LrtCreateAmlogicOptions(&options));
  LITERT_ASSERT_OK(LrtAmlogicOptionsSetModelPath(options, "/data/vendor/nn/"));
  LITERT_ASSERT_OK(LrtAmlogicOptionsSetModelName(options, "mobilenet"));

  LrtAmlogicOptions parsed = nullptr;
  SerializeAndParse(options, &parsed);
  const char* model_path = nullptr;
  const char* model_name = nullptr;
  LITERT_ASSERT_OK(LrtAmlogicOptionsGetModelPath(parsed, &model_path));
  LITERT_ASSERT_OK(LrtAmlogicOptionsGetModelName(parsed, &model_name));
  EXPECT_STREQ(model_path, "/data/vendor/nn/");
  EXPECT_STREQ(model_name, "mobilenet");

  LrtDestroyAmlogicOptions(parsed);
  LrtDestroyAmlogicOptions(options);
}

TEST(LrtAmlogicOptionsTest, EscapesQuotesAndNewlines) {
  LrtAmlogicOptions options = nullptr;
  LITERT_ASSERT_OK(LrtCreateAmlogicOptions(&options));
  LITERT_ASSERT_OK(
      LrtAmlogicOptionsSetModelPath(options, "/tmp/a\nb\"c\\d"));
  LITERT_ASSERT_OK(LrtAmlogicOptionsSetModelName(options, "line\rname"));

  LrtAmlogicOptions parsed = nullptr;
  SerializeAndParse(options, &parsed);
  const char* model_path = nullptr;
  const char* model_name = nullptr;
  LITERT_ASSERT_OK(LrtAmlogicOptionsGetModelPath(parsed, &model_path));
  LITERT_ASSERT_OK(LrtAmlogicOptionsGetModelName(parsed, &model_name));
  EXPECT_STREQ(model_path, "/tmp/a\nb\"c\\d");
  EXPECT_STREQ(model_name, "line\rname");

  LrtDestroyAmlogicOptions(parsed);
  LrtDestroyAmlogicOptions(options);
}

TEST(LrtAmlogicOptionsTest, LogLevelRoundTrip) {
  LrtAmlogicOptions options = nullptr;
  LITERT_ASSERT_OK(LrtCreateAmlogicOptions(&options));
  LITERT_ASSERT_OK(
      LrtAmlogicOptionsSetLogLevel(options, kLrtAmlogicLogLevelDebug));

  LrtAmlogicOptions parsed = nullptr;
  SerializeAndParse(options, &parsed);
  LrtAmlogicOptionsLogLevel log_level;
  LITERT_ASSERT_OK(LrtAmlogicOptionsGetLogLevel(parsed, &log_level));
  EXPECT_EQ(log_level, kLrtAmlogicLogLevelDebug);

  LrtDestroyAmlogicOptions(parsed);
  LrtDestroyAmlogicOptions(options);
}

TEST(LrtAmlogicOptionsTest, RejectsLogLevelOutOfRange) {
  LrtAmlogicOptions options = nullptr;
  EXPECT_EQ(LrtCreateAmlogicOptionsFromToml("log_level = 9\n", &options),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(options, nullptr);
}

TEST(LrtAmlogicOptionsTest, CreateFromToml) {
  const char* toml =
      "model_path = \"/tmp/adla\"\n"
      "model_name = \"aml_model\"\n";
  LrtAmlogicOptions options = nullptr;
  LITERT_ASSERT_OK(LrtCreateAmlogicOptionsFromToml(toml, &options));

  const char* model_path = nullptr;
  const char* model_name = nullptr;
  LITERT_ASSERT_OK(LrtAmlogicOptionsGetModelPath(options, &model_path));
  LITERT_ASSERT_OK(LrtAmlogicOptionsGetModelName(options, &model_name));
  EXPECT_STREQ(model_path, "/tmp/adla");
  EXPECT_STREQ(model_name, "aml_model");

  LrtDestroyAmlogicOptions(options);
}

TEST(LrtAmlogicOptionsTest, RejectsBadEscape) {
  LrtAmlogicOptions options = nullptr;
  EXPECT_EQ(LrtCreateAmlogicOptionsFromToml("model_path = \"abc\\q\"\n",
                                            &options),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(options, nullptr);
}

}  // namespace
}  // namespace litert::amlogic
