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
#include "litert/c/options/litert_verisilicon_options.h"

#include <gtest/gtest.h>

#include <string>

#include "absl/strings/match.h"        // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"

namespace litert::verisilicon {
namespace {

void SerializeAndParse(LrtVerisiliconOptions payload,
                       LrtVerisiliconOptions* parsed) {
  const char* identifier;
  void* raw_payload = nullptr;
  void (*payload_deleter)(void*);
  LITERT_ASSERT_OK(LrtGetOpaqueVerisiliconOptionsData(
      payload, &identifier, &raw_payload, &payload_deleter));
  EXPECT_STREQ(identifier, "verisilicon");
  const char* toml_str = static_cast<const char*>(raw_payload);

  LITERT_ASSERT_OK(LrtCreateVerisiliconOptionsFromToml(toml_str, parsed));

  payload_deleter(raw_payload);
}

TEST(LrtVerisiliconOptionsTest, GetOpaqueDataEmpty) {
  LrtVerisiliconOptions options;
  LITERT_ASSERT_OK(LrtCreateVerisiliconOptions(&options));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*);
  LITERT_ASSERT_OK(LrtGetOpaqueVerisiliconOptionsData(
      options, &identifier, &payload, &payload_deleter));

  EXPECT_STREQ(identifier, "verisilicon");
  const char* toml_str = static_cast<const char*>(payload);
  EXPECT_STREQ(toml_str, "");

  payload_deleter(payload);
  LrtDestroyVerisiliconOptions(options);
}

TEST(LrtVersiliconOptionsTest, DeviceIndex) {
  LrtVerisiliconOptions options;
  LITERT_ASSERT_OK(LrtCreateVerisiliconOptions(&options));

  unsigned int device_index = 0;
  LITERT_ASSERT_OK(LrtVerisiliconOptionsGetDeviceIndex(options, &device_index));

  EXPECT_EQ(device_index, 0);
  LITERT_ASSERT_OK(LrtVerisiliconOptionsSetDeviceIndex(options, 1));

  LITERT_ASSERT_OK(LrtVerisiliconOptionsGetDeviceIndex(options, &device_index));
  ASSERT_EQ(device_index, 1);

  LrtVerisiliconOptions parsed;
  SerializeAndParse(options, &parsed);
  unsigned int parsed_idx;
  LrtVerisiliconOptionsGetDeviceIndex(parsed, &parsed_idx);
  EXPECT_EQ(parsed_idx, 1);

  LrtDestroyVerisiliconOptions(parsed);
  LrtDestroyVerisiliconOptions(options);
}

TEST(LiteRtVersiliconOptionsTest, CoreIndex) {
  LrtVerisiliconOptions options;
  LITERT_ASSERT_OK(LrtCreateVerisiliconOptions(&options));

  unsigned int core_index = 0;
  LITERT_ASSERT_OK(LrtVerisiliconOptionsGetCoreIndex(options, &core_index));

  EXPECT_EQ(core_index, 0);
  LITERT_ASSERT_OK(LrtVerisiliconOptionsSetCoreIndex(options, 1));

  LITERT_ASSERT_OK(LrtVerisiliconOptionsGetCoreIndex(options, &core_index));
  ASSERT_EQ(core_index, 1);

  LrtVerisiliconOptions parsed;
  SerializeAndParse(options, &parsed);
  unsigned int parsed_idx;
  LrtVerisiliconOptionsGetCoreIndex(parsed, &parsed_idx);
  EXPECT_EQ(parsed_idx, 1);

  LrtDestroyVerisiliconOptions(parsed);
  LrtDestroyVerisiliconOptions(options);
}

TEST(LiteRtVersiliconOptionsTest, TimeOut) {
  LrtVerisiliconOptions options;
  LITERT_ASSERT_OK(LrtCreateVerisiliconOptions(&options));

  unsigned int time_out = 840000;
  LITERT_ASSERT_OK(LrtVerisiliconOptionsGetTimeOut(options, &time_out));

  EXPECT_EQ(time_out, 0);
  // LITERT_ASSERT_OK(LrtVerisiliconOptionsSetTimeOut(options, 1000));

  LITERT_ASSERT_OK(LrtVerisiliconOptionsGetTimeOut(options, &time_out));
  ASSERT_EQ(time_out, 1000);

  LrtVerisiliconOptions parsed;
  SerializeAndParse(options, &parsed);
  unsigned int parsed_idx;
  LrtVerisiliconOptionsGetTimeOut(parsed, &parsed_idx);
  EXPECT_EQ(parsed_idx, 1000);

  LrtDestroyVerisiliconOptions(parsed);
  LrtDestroyVerisiliconOptions(options);
}

TEST(LiteRtVersiliconOptionsTest, ProfileLevel) {
  LrtVerisiliconOptions options;
  LITERT_ASSERT_OK(LrtCreateVerisiliconOptions(&options));

  unsigned int profile_level = 10;
  LITERT_ASSERT_OK(
      LrtVerisiliconOptionsGetProfileLevel(options, &profile_level));

  EXPECT_EQ(profile_level, 0);
  // LITERT_ASSERT_OK(LrtVerisiliconOptionsSetProfileLevel(options, 2));

  LITERT_ASSERT_OK(
      LrtVerisiliconOptionsGetProfileLevel(options, &profile_level));
  ASSERT_EQ(profile_level, 2);

  LrtVerisiliconOptions parsed;
  SerializeAndParse(options, &parsed);
  unsigned int parsed_idx;
  LrtVerisiliconOptionsGetProfileLevel(parsed, &parsed_idx);
  EXPECT_EQ(parsed_idx, 2);

  LrtDestroyVerisiliconOptions(parsed);
  LrtDestroyVerisiliconOptions(options);
}

TEST(LiteRtVersiliconOptionsTest, DumpNBG) {
  LrtVerisiliconOptions options;
  LITERT_ASSERT_OK(LrtCreateVerisiliconOptions(&options));

  bool enable = false;
  LITERT_ASSERT_OK(LrtVerisiliconOptionsGetDumpNBG(options, &enable));

  EXPECT_FALSE(enable);
  // LITERT_ASSERT_OK(LrtVerisiliconOptionsSetDumpNBG(options, true));

  LITERT_ASSERT_OK(LrtVerisiliconOptionsGetDumpNBG(options, &enable));
  EXPECT_TRUE(enable);

  LrtVerisiliconOptions parsed;
  SerializeAndParse(options, &parsed);
  bool parsed_value;
  LrtVerisiliconOptionsGetDumpNBG(parsed, &parsed_value);
  EXPECT_TRUE(parsed_value);

  LrtDestroyVerisiliconOptions(parsed);
  LrtDestroyVerisiliconOptions(options);
}

}  // namespace
}  // namespace litert::verisilicon
