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

#include "litert/c/options/litert_webnn_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"

namespace {

using ::testing::NotNull;

void SerializeAndParse(LrtWebNnOptions* payload,
                       LrtWebNnOptions** payload_from_toml) {
  const char* identifier = nullptr;
  void* opaque_payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;

  LITERT_ASSERT_OK(LrtGetOpaqueWebNnOptionsData(
      payload, &identifier, &opaque_payload, &payload_deleter));

  LITERT_ASSERT_OK(LrtCreateWebNnOptionsFromToml(
      static_cast<const char*>(opaque_payload), payload_from_toml));

  if (payload_deleter && opaque_payload) {
    payload_deleter(opaque_payload);
  }
}

TEST(LiteRtWebNnOptionsTest, CreateAndDestroy) {
  LrtWebNnOptions* options = nullptr;
  ASSERT_EQ(LrtCreateWebNnOptions(&options), kLiteRtStatusOk);
  ASSERT_THAT(options, NotNull());
  LrtDestroyWebNnOptions(options);
}

TEST(LiteRtWebNnOptionsTest, SetAndGetDevicePreference) {
  LrtWebNnOptions* options = nullptr;
  ASSERT_EQ(LrtCreateWebNnOptions(&options), kLiteRtStatusOk);

  constexpr LiteRtWebNnDeviceType kDevicePreference = kLiteRtWebNnDeviceTypeGpu;
  EXPECT_EQ(LrtSetWebNnOptionsDevicePreference(options, kDevicePreference),
            kLiteRtStatusOk);

  LiteRtWebNnDeviceType device_preference = kLiteRtWebNnDeviceTypeCpu;
  EXPECT_EQ(LrtGetWebNnOptionsDevicePreference(options, &device_preference),
            kLiteRtStatusOk);
  EXPECT_EQ(device_preference, kDevicePreference);

  LrtWebNnOptions* options_from_toml = nullptr;
  SerializeAndParse(options, &options_from_toml);

  LiteRtWebNnDeviceType device_preference_from_toml;
  EXPECT_EQ(LrtGetWebNnOptionsDevicePreference(options_from_toml,
                                               &device_preference_from_toml),
            kLiteRtStatusOk);
  EXPECT_EQ(device_preference_from_toml, kDevicePreference);

  LrtDestroyWebNnOptions(options_from_toml);
  LrtDestroyWebNnOptions(options);
}

TEST(LiteRtWebNnOptionsTest, SetAndGetPowerPreference) {
  LrtWebNnOptions* options = nullptr;
  ASSERT_EQ(LrtCreateWebNnOptions(&options), kLiteRtStatusOk);

  constexpr LiteRtWebNnPowerPreference kPowerPreference =
      kLiteRtWebNnPowerPreferenceLowPower;
  EXPECT_EQ(LrtSetWebNnOptionsPowerPreference(options, kPowerPreference),
            kLiteRtStatusOk);

  LiteRtWebNnPowerPreference power_preference =
      kLiteRtWebNnPowerPreferenceDefault;
  EXPECT_EQ(LrtGetWebNnOptionsPowerPreference(options, &power_preference),
            kLiteRtStatusOk);
  EXPECT_EQ(power_preference, kPowerPreference);

  LrtWebNnOptions* options_from_toml = nullptr;
  SerializeAndParse(options, &options_from_toml);

  LiteRtWebNnPowerPreference power_preference_from_toml;
  EXPECT_EQ(LrtGetWebNnOptionsPowerPreference(options_from_toml,
                                              &power_preference_from_toml),
            kLiteRtStatusOk);
  EXPECT_EQ(power_preference_from_toml, kPowerPreference);

  LrtDestroyWebNnOptions(options_from_toml);
  LrtDestroyWebNnOptions(options);
}

TEST(LiteRtWebNnOptionsTest, SetAndGetPrecision) {
  LrtWebNnOptions* options = nullptr;
  ASSERT_EQ(LrtCreateWebNnOptions(&options), kLiteRtStatusOk);

  constexpr LiteRtWebNnPrecision kPrecision = kLiteRtWebNnPrecisionFp16;
  EXPECT_EQ(LrtSetWebNnOptionsPrecision(options, kPrecision), kLiteRtStatusOk);

  LiteRtWebNnPrecision precision = kLiteRtWebNnPrecisionFp32;
  EXPECT_EQ(LrtGetWebNnOptionsPrecision(options, &precision), kLiteRtStatusOk);
  EXPECT_EQ(precision, kPrecision);

  LrtWebNnOptions* options_from_toml = nullptr;
  SerializeAndParse(options, &options_from_toml);

  LiteRtWebNnPrecision precision_from_toml;
  EXPECT_EQ(
      LrtGetWebNnOptionsPrecision(options_from_toml, &precision_from_toml),
      kLiteRtStatusOk);
  EXPECT_EQ(precision_from_toml, kPrecision);

  LrtDestroyWebNnOptions(options_from_toml);
  LrtDestroyWebNnOptions(options);
}

TEST(LiteRtWebNnOptionsTest, Serialization) {
  LrtWebNnOptions* options = nullptr;
  ASSERT_EQ(LrtCreateWebNnOptions(&options), kLiteRtStatusOk);

  EXPECT_EQ(
      LrtSetWebNnOptionsDevicePreference(options, kLiteRtWebNnDeviceTypeGpu),
      kLiteRtStatusOk);

  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;

  ASSERT_EQ(LrtGetOpaqueWebNnOptionsData(options, &identifier, &payload,
                                         &payload_deleter),
            kLiteRtStatusOk);

  EXPECT_STREQ(identifier, "webnn_options_string");
  EXPECT_THAT(static_cast<char*>(payload),
              testing::HasSubstr("device_type = 1"));

  payload_deleter(payload);
  LrtDestroyWebNnOptions(options);
}

}  // namespace
