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

#include "litert/c/options/litert_samsung_options.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/test/matchers.h"

namespace litert::samsung {
namespace {

void SerializeAndParse(LrtSamsungOptions payload,
                       LrtSamsungOptions* parsed) {
  const char* identifier;
  void* raw_payload = nullptr;
  void (*payload_deleter)(void*);
  LITERT_ASSERT_OK(LrtGetOpaqueSamsungOptionsData(
      payload, &identifier, &raw_payload, &payload_deleter));
  EXPECT_STREQ(identifier, "samsung");
  const char* toml_str = static_cast<const char*>(raw_payload);

  LITERT_ASSERT_OK(LrtCreateSamsungOptionsFromToml(toml_str, parsed));

  payload_deleter(raw_payload);
}

TEST(LrtSamsungOptionsTest, GetOpaqueDataEmpty) {
  LrtSamsungOptions options;
  LITERT_ASSERT_OK(LrtCreateSamsungOptions(&options));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*);
  LITERT_ASSERT_OK(LrtGetOpaqueSamsungOptionsData(options, &identifier,
                                                   &payload, &payload_deleter));

  EXPECT_STREQ(identifier, "samsung");
  const char* toml_str = static_cast<const char*>(payload);
  EXPECT_STREQ(toml_str, "");

  payload_deleter(payload);
  LrtDestroySamsungOptions(options);
}

TEST(LrtSamsungOptionsTest, EnableLargeModelSupport) {
  LrtSamsungOptions options;
  LITERT_ASSERT_OK(LrtCreateSamsungOptions(&options));

  bool enable_large_model_support;
  LITERT_ASSERT_OK(LrtSamsungOptionsGetEnableLargeModelSupport(
      options, &enable_large_model_support));
  EXPECT_FALSE(enable_large_model_support);

  LITERT_ASSERT_OK(
      LrtSamsungOptionsSetEnableLargeModelSupport(options, true));
  LITERT_ASSERT_OK(LrtSamsungOptionsGetEnableLargeModelSupport(
      options, &enable_large_model_support));
  EXPECT_TRUE(enable_large_model_support);

  LrtSamsungOptions parsed;
  SerializeAndParse(options, &parsed);
  bool parsed_val;
  LrtSamsungOptionsGetEnableLargeModelSupport(parsed, &parsed_val);
  EXPECT_TRUE(parsed_val);

  LrtDestroySamsungOptions(parsed);
  LrtDestroySamsungOptions(options);
}

TEST(LrtSamsungOptionsTest, CreateFromToml) {
  const char* toml_payload = "enable_large_model_support = true\n";
  LrtSamsungOptions options;
  LITERT_ASSERT_OK(LrtCreateSamsungOptionsFromToml(toml_payload, &options));

  bool enable_large_model_support;
  LITERT_ASSERT_OK(LrtSamsungOptionsGetEnableLargeModelSupport(
      options, &enable_large_model_support));
  EXPECT_TRUE(enable_large_model_support);

  LrtDestroySamsungOptions(options);
}

}  // namespace
}  // namespace litert::samsung
