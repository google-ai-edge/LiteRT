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

#include "litert/cc/options/litert_webnn_options.h"

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"
#include "litert/c/options/litert_webnn_options.h"

namespace {

using ::litert::WebNnOptions;

TEST(LiteRtWebNnOptionsTest, SetAndGetDevicePreference) {
  LITERT_ASSERT_OK_AND_ASSIGN(WebNnOptions options, WebNnOptions::Create());

  constexpr LiteRtWebNnDeviceType kDevicePreference =
      LiteRtWebNnDeviceType::kLiteRtWebNnDeviceTypeGpu;
  EXPECT_EQ(options.SetDevicePreference(kDevicePreference), kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtWebNnOptionsPayload payload,
                              options.GetData<LiteRtWebNnOptionsPayloadT>());

  LiteRtWebNnDeviceType device_preference =
      LiteRtWebNnDeviceType::kLiteRtWebNnDeviceTypeCpu;
  EXPECT_EQ(LiteRtGetWebNnOptionsDevicePreference(&device_preference, payload),
            kLiteRtStatusOk);
  EXPECT_EQ(device_preference, kDevicePreference);
}

TEST(LiteRtWebNnOptionsTest, SetAndGetPowerPreference) {
  LITERT_ASSERT_OK_AND_ASSIGN(WebNnOptions options, WebNnOptions::Create());

  constexpr LiteRtWebNnPowerPreference kPowerPreference =
      LiteRtWebNnPowerPreference::kLiteRtWebNnPowerPreferenceLowPower;
  EXPECT_EQ(options.SetPowerPreference(kPowerPreference), kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtWebNnOptionsPayload payload,
                              options.GetData<LiteRtWebNnOptionsPayloadT>());

  LiteRtWebNnPowerPreference power_preference =
      LiteRtWebNnPowerPreference::kLiteRtWebNnPowerPreferenceDefault;
  EXPECT_EQ(LiteRtGetWebNnOptionsPowerPreference(&power_preference, payload),
            kLiteRtStatusOk);
  EXPECT_EQ(power_preference, kPowerPreference);
}

TEST(LiteRtWebNnOptionsTest, SetAndGetPrecision) {
  LITERT_ASSERT_OK_AND_ASSIGN(WebNnOptions options, WebNnOptions::Create());
  constexpr LiteRtWebNnPrecision kPrecision =
      LiteRtWebNnPrecision::kLiteRtWebNnPrecisionFp16;
  EXPECT_EQ(options.SetPrecision(kPrecision), kLiteRtStatusOk);

  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtWebNnOptionsPayload payload,
                              options.GetData<LiteRtWebNnOptionsPayloadT>());
  LiteRtWebNnPrecision precision =
      LiteRtWebNnPrecision::kLiteRtWebNnPrecisionFp32;
  EXPECT_EQ(LiteRtGetWebNnOptionsPrecision(&precision, payload),
            kLiteRtStatusOk);
  EXPECT_EQ(precision, kPrecision);
}

}  // namespace
