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
#include "litert/c/litert_opaque_options.h"

namespace {

using ::testing::NotNull;

TEST(LiteRtWebNnOptionsTest, CreateAndDestroy) {
  LiteRtOpaqueOptions options = nullptr;
  ASSERT_EQ(LiteRtCreateWebNnOptions(&options), kLiteRtStatusOk);
  ASSERT_THAT(options, NotNull());
  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtWebNnOptionsTest, SetAndGetDevicePreference) {
  LiteRtOpaqueOptions options = nullptr;
  ASSERT_EQ(LiteRtCreateWebNnOptions(&options), kLiteRtStatusOk);
  ASSERT_THAT(options, NotNull());

  constexpr LiteRtWebNnDeviceType kDevicePreference =
      kLiteRtWebNnDeviceTypeGpu;
  EXPECT_EQ(LiteRtSetWebNnOptionsDevicePreference(options, kDevicePreference),
            kLiteRtStatusOk);

  LiteRtWebNnOptionsPayload payload = nullptr;
  const char* identifier = nullptr;
  ASSERT_EQ(LiteRtGetOpaqueOptionsIdentifier(options, &identifier),
            kLiteRtStatusOk);
  void* data = nullptr;
  ASSERT_EQ(LiteRtGetOpaqueOptionsData(options, &data), kLiteRtStatusOk);
  payload = reinterpret_cast<LiteRtWebNnOptionsPayload>(data);

  LiteRtWebNnDeviceType device_preference = kLiteRtWebNnDeviceTypeCpu;
  EXPECT_EQ(LiteRtGetWebNnOptionsDevicePreference(&device_preference, payload),
            kLiteRtStatusOk);
  EXPECT_EQ(device_preference, kDevicePreference);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtWebNnOptionsTest, SetAndGetPowerPreference) {
  LiteRtOpaqueOptions options = nullptr;
  ASSERT_EQ(LiteRtCreateWebNnOptions(&options), kLiteRtStatusOk);
  ASSERT_THAT(options, NotNull());

  constexpr LiteRtWebNnPowerPreference kPowerPreference =
      kLiteRtWebNnPowerPreferenceLowPower;
  EXPECT_EQ(LiteRtSetWebNnOptionsPowerPreference(options, kPowerPreference),
            kLiteRtStatusOk);

  LiteRtWebNnOptionsPayload payload = nullptr;
  const char* identifier = nullptr;
  ASSERT_EQ(LiteRtGetOpaqueOptionsIdentifier(options, &identifier),
            kLiteRtStatusOk);
  void* data = nullptr;
  ASSERT_EQ(LiteRtGetOpaqueOptionsData(options, &data), kLiteRtStatusOk);
  payload = reinterpret_cast<LiteRtWebNnOptionsPayload>(data);

  LiteRtWebNnPowerPreference power_preference =
      kLiteRtWebNnPowerPreferenceDefault;
  EXPECT_EQ(LiteRtGetWebNnOptionsPowerPreference(&power_preference, payload),
            kLiteRtStatusOk);
  EXPECT_EQ(power_preference, kPowerPreference);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtWebNnOptionsTest, SetAndGetPrecision) {
  LiteRtOpaqueOptions options = nullptr;
  ASSERT_EQ(LiteRtCreateWebNnOptions(&options), kLiteRtStatusOk);
  ASSERT_THAT(options, NotNull());

  constexpr LiteRtWebNnPrecision kPrecision = kLiteRtWebNnPrecisionFp16;
  EXPECT_EQ(LiteRtSetWebNnOptionsPrecision(options, kPrecision),
            kLiteRtStatusOk);

  LiteRtWebNnOptionsPayload payload = nullptr;
  const char* identifier = nullptr;
  ASSERT_EQ(LiteRtGetOpaqueOptionsIdentifier(options, &identifier),
            kLiteRtStatusOk);
  void* data = nullptr;
  ASSERT_EQ(LiteRtGetOpaqueOptionsData(options, &data), kLiteRtStatusOk);
  payload = reinterpret_cast<LiteRtWebNnOptionsPayload>(data);

  LiteRtWebNnPrecision precision = kLiteRtWebNnPrecisionFp32;
  EXPECT_EQ(LiteRtGetWebNnOptionsPrecision(&precision, payload),
            kLiteRtStatusOk);
  EXPECT_EQ(precision, kPrecision);

  LiteRtDestroyOpaqueOptions(options);
}

}  // namespace
