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
#include "tflite/experimental/acceleration/compatibility/android_info.h"
#include "tflite/experimental/acceleration/compatibility/devicedb-sample.h"

namespace litert::ml_drift {
namespace {

using ::tflite::acceleration::AndroidInfo;

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
TEST(MlDriftCompatibilityTest, SupportedInDatabaseReturnsTrue) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "m712c", "sample_device", "24");

  EXPECT_TRUE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, "Mali", 3, 1));
}

// devicedb-sample.json: sm_j810m + j8y18lte + SDK 24 is SUPPORTED. The raw
// values are passed as reported by Android and must be canonicalized.
TEST(MlDriftCompatibilityTest, SupportedWithRawAndroidValuesReturnsTrue) {
  AndroidInfo android_info =
      MakeAndroidInfo("samsung", "SM-J810M", "j8y18lte", "24");

  EXPECT_TRUE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, "Adreno (TM) 308", 3, 0));
}

// devicedb-sample.json: sm_j810f + j8y18lte + SDK 24 is UNSUPPORTED.
TEST(MlDriftCompatibilityTest, UnsupportedInDatabaseReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("samsung", "SM-J810F", "j8y18lte", "24");

  EXPECT_FALSE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, "Adreno (TM) 308", 3, 0));
}

// devicedb-sample.json: m712c is stored lowercase; the model reported by
// Android must be canonicalized before the lookup.
TEST(MlDriftCompatibilityTest, UppercaseModelReturnsTrue) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "M712C", "sample_device", "24");

  EXPECT_TRUE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, "Mali", 3, 1));
}

// devicedb-sample.json: shiraz_ag_2011 + SDK <= 28 is UNSUPPORTED.
TEST(MlDriftCompatibilityTest, UnsupportedModelWithMaximumSdkReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "shiraz-ag-2011", "sample_device", "28");

  EXPECT_FALSE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, "Mali", 3, 1));
}

// A device missing from the database has no status (UNKNOWN), which is
// treated as not supported.
TEST(MlDriftCompatibilityTest, DeviceNotInDatabaseReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "mag2016", "sample_device", "26");

  EXPECT_FALSE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, "Mali", 3, 1));
}

// The sm_j810m entry also requires device name j8y18lte, so the device name
// must be looked up independently of the model.
TEST(MlDriftCompatibilityTest, MismatchedDeviceNameReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("samsung", "SM-J810M", "other_device", "24");

  EXPECT_FALSE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, "Adreno (TM) 308", 3, 0));
}

// The m712c entry requires GLES 3.1.
TEST(MlDriftCompatibilityTest, UnlistedGlesVersionReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "m712c", "sample_device", "24");

  EXPECT_FALSE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, "Mali", 3, 0));
}

TEST(MlDriftCompatibilityTest, NullBinaryReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "m712c", "sample_device", "24");

  EXPECT_FALSE(IsMlDriftGpuSupported(nullptr, 0, android_info, "Mali", 3, 1));
}

TEST(MlDriftCompatibilityTest, CorruptedBinaryReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("sample_mfr", "m712c", "sample_device", "24");
  const unsigned char kCorruptedBinary[] = {0xff, 0xff, 0xff, 0xff,
                                            0xff, 0xff, 0xff, 0xff};

  EXPECT_FALSE(IsMlDriftGpuSupported(kCorruptedBinary, sizeof(kCorruptedBinary),
                                     android_info, "Mali", 3, 1));
}

#if !defined(__ANDROID__)
TEST(MlDriftCompatibilityTest, NonAndroidReturnsTrue) {
  EXPECT_TRUE(IsMlDriftGpuSupportedOnThisDevice());
}
#endif

}  // namespace
}  // namespace litert::ml_drift
