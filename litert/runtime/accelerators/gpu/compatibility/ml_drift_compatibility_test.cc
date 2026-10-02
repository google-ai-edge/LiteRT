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

#include "litert/runtime/accelerators/gpu/compatibility/ml_drift_compatibility.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/experimental/acceleration/compatibility/android_info.h"
#include "tflite/experimental/acceleration/compatibility/devicedb-sample.h"

namespace litert::ml_drift {

using ::tflite::acceleration::AndroidInfo;

// Friend of GpuCompatibilityChecker: checks a device against a given binary.
class GpuCompatibilityCheckerTest : public ::testing::Test {
 protected:
  // devicedb-sample.json compiled to a flatbuffer.
  static absl::Span<const unsigned char> SampleBinary() {
    return absl::MakeConstSpan(
        g_tflite_acceleration_devicedb_sample_binary,
        g_tflite_acceleration_devicedb_sample_binary_len);
  }

  static bool IsSupported(absl::Span<const unsigned char> compatibility_binary,
                          const AndroidInfo& android_info,
                          absl::string_view gl_renderer, int gles_major,
                          int gles_minor) {
    return GpuCompatibilityChecker(compatibility_binary)
        .IsSupported(android_info, gl_renderer, gles_major, gles_minor);
  }
};

namespace {

AndroidInfo MakeAndroidInfo(absl::string_view manufacturer,
                            absl::string_view model, absl::string_view device,
                            absl::string_view sdk) {
  AndroidInfo info;
  info.manufacturer = std::string(manufacturer);
  info.model = std::string(model);
  info.device = std::string(device);
  info.android_sdk_version = std::string(sdk);
  return info;
}

// devicedb-sample.json: m712c + GLES 3.1 + SDK >= 24 is SUPPORTED.
TEST_F(GpuCompatibilityCheckerTest, SupportedInDatabaseReturnsTrue) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "m712c", "sample_device", "24");

  EXPECT_TRUE(IsSupported(SampleBinary(), android_info, "Mali", 3, 1));
}

// devicedb-sample.json: sm_j810m + j8y18lte + SDK 24 is SUPPORTED. The raw
// values are passed as reported by Android and must be canonicalized.
TEST_F(GpuCompatibilityCheckerTest, SupportedWithRawAndroidValuesReturnsTrue) {
  AndroidInfo android_info =
      MakeAndroidInfo("samsung", "SM-J810M", "j8y18lte", "24");

  EXPECT_TRUE(
      IsSupported(SampleBinary(), android_info, "Adreno (TM) 308", 3, 0));
}

// devicedb-sample.json: sm_j810f + j8y18lte + SDK 24 is UNSUPPORTED.
TEST_F(GpuCompatibilityCheckerTest, UnsupportedInDatabaseReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("samsung", "SM-J810F", "j8y18lte", "24");

  EXPECT_FALSE(
      IsSupported(SampleBinary(), android_info, "Adreno (TM) 308", 3, 0));
}

// devicedb-sample.json: m712c is stored lowercase; the model reported by
// Android must be canonicalized before the lookup.
TEST_F(GpuCompatibilityCheckerTest, UppercaseModelReturnsTrue) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "M712C", "sample_device", "24");

  EXPECT_TRUE(IsSupported(SampleBinary(), android_info, "Mali", 3, 1));
}

// devicedb-sample.json: shiraz_ag_2011 + SDK <= 28 is UNSUPPORTED.
TEST_F(GpuCompatibilityCheckerTest,
       UnsupportedModelWithMaximumSdkReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "shiraz-ag-2011", "sample_device", "28");

  EXPECT_FALSE(IsSupported(SampleBinary(), android_info, "Mali", 3, 1));
}

// A device missing from the database has no status (UNKNOWN), which is
// treated as not supported.
TEST_F(GpuCompatibilityCheckerTest, DeviceNotInDatabaseReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "mag2016", "sample_device", "26");

  EXPECT_FALSE(IsSupported(SampleBinary(), android_info, "Mali", 3, 1));
}

// The sm_j810m entry also requires device name j8y18lte, so the device name
// must be looked up independently of the model.
TEST_F(GpuCompatibilityCheckerTest, MismatchedDeviceNameReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("samsung", "SM-J810M", "other_device", "24");

  EXPECT_FALSE(
      IsSupported(SampleBinary(), android_info, "Adreno (TM) 308", 3, 0));
}

// The m712c entry requires GLES 3.1.
TEST_F(GpuCompatibilityCheckerTest, UnlistedGlesVersionReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "m712c", "sample_device", "24");

  EXPECT_FALSE(IsSupported(SampleBinary(), android_info, "Mali", 3, 0));
}

TEST_F(GpuCompatibilityCheckerTest, EmptyBinaryReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "m712c", "sample_device", "24");

  EXPECT_FALSE(IsSupported({}, android_info, "Mali", 3, 1));
}

TEST_F(GpuCompatibilityCheckerTest, CorruptedBinaryReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "m712c", "sample_device", "24");
  const unsigned char kCorruptedBinary[] = {0xff, 0xff, 0xff, 0xff,
                                            0xff, 0xff, 0xff, 0xff};

  EXPECT_FALSE(IsSupported(kCorruptedBinary, android_info, "Mali", 3, 1));
}

#if !defined(__ANDROID__)
TEST_F(GpuCompatibilityCheckerTest, NonAndroidReturnsTrue) {
  EXPECT_TRUE(GpuCompatibilityChecker::Instance().IsSupportedOnThisDevice());
}
#endif

}  // namespace
}  // namespace litert::ml_drift
