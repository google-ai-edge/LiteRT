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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/tools/flags/vendors/intel_openvino_flags.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/marshalling.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/test/matchers.h"

namespace litert::intel_openvino {
namespace {

TEST(DeviceTypeFlagTest, Malformed) {
  std::string error;
  LiteRtIntelOpenVinoDeviceType value;

  EXPECT_FALSE(absl::ParseFlag("invalid", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("+", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("unknown", &value, &error));
}

TEST(DeviceTypeFlagTest, Parse) {
  std::string error;
  LiteRtIntelOpenVinoDeviceType value;

  {
    static constexpr absl::string_view kDevice = "cpu";
    static constexpr LiteRtIntelOpenVinoDeviceType kDeviceEnum =
        kLiteRtIntelOpenVinoDeviceTypeCPU;
    EXPECT_TRUE(absl::ParseFlag(kDevice, &value, &error));
    EXPECT_EQ(value, kDeviceEnum);
    EXPECT_EQ(kDevice, absl::UnparseFlag(value));
  }

  {
    static constexpr absl::string_view kDevice = "gpu";
    static constexpr LiteRtIntelOpenVinoDeviceType kDeviceEnum =
        kLiteRtIntelOpenVinoDeviceTypeGPU;
    EXPECT_TRUE(absl::ParseFlag(kDevice, &value, &error));
    EXPECT_EQ(value, kDeviceEnum);
    EXPECT_EQ(kDevice, absl::UnparseFlag(value));
  }

  {
    static constexpr absl::string_view kDevice = "npu";
    static constexpr LiteRtIntelOpenVinoDeviceType kDeviceEnum =
        kLiteRtIntelOpenVinoDeviceTypeNPU;
    EXPECT_TRUE(absl::ParseFlag(kDevice, &value, &error));
    EXPECT_EQ(value, kDeviceEnum);
    EXPECT_EQ(kDevice, absl::UnparseFlag(value));
  }

  {
    static constexpr absl::string_view kDevice = "auto";
    static constexpr LiteRtIntelOpenVinoDeviceType kDeviceEnum =
        kLiteRtIntelOpenVinoDeviceTypeAUTO;
    EXPECT_TRUE(absl::ParseFlag(kDevice, &value, &error));
    EXPECT_EQ(value, kDeviceEnum);
    EXPECT_EQ(kDevice, absl::UnparseFlag(value));
  }
}

TEST(PerformanceModeFlagTest, Parse) {
  std::string error;
  LiteRtIntelOpenVinoPerformanceMode value;

  {
    static constexpr absl::string_view kMode = "latency";
    static constexpr LiteRtIntelOpenVinoPerformanceMode kModeEnum =
        kLiteRtIntelOpenVinoPerformanceModeLatency;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "throughput";
    static constexpr LiteRtIntelOpenVinoPerformanceMode kModeEnum =
        kLiteRtIntelOpenVinoPerformanceModeThroughput;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "cumulative_throughput";
    static constexpr LiteRtIntelOpenVinoPerformanceMode kModeEnum =
        kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
}

TEST(UpdateIntelOpenVinoOptionsFromFlagsTest, DefaultValues) {
  LITERT_ASSERT_OK_AND_ASSIGN(IntelOpenVinoOptions options,
                             IntelOpenVinoOptions::Create());
  ASSERT_TRUE(UpdateIntelOpenVinoOptionsFromFlags(options).HasValue());
  EXPECT_EQ(options.GetDeviceType(), kLiteRtIntelOpenVinoDeviceTypeNPU);
  EXPECT_EQ(options.GetPerformanceMode(),
            kLiteRtIntelOpenVinoPerformanceModeLatency);
}

TEST(UpdateIntelOpenVinoOptionsFromFlagsTest, SetDeviceTypeToCPU) {
  absl::SetFlag(&FLAGS_intel_openvino_device_type,
                kLiteRtIntelOpenVinoDeviceTypeCPU);
  LITERT_ASSERT_OK_AND_ASSIGN(IntelOpenVinoOptions options,
                             IntelOpenVinoOptions::Create());

  ASSERT_TRUE(UpdateIntelOpenVinoOptionsFromFlags(options).HasValue());
  EXPECT_EQ(options.GetDeviceType(), kLiteRtIntelOpenVinoDeviceTypeCPU);

  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_intel_openvino_device_type,
                kLiteRtIntelOpenVinoDeviceTypeNPU);
}

TEST(UpdateIntelOpenVinoOptionsFromFlagsTest, SetPerformanceModeToThroughput) {
  absl::SetFlag(&FLAGS_intel_openvino_performance_mode,
                kLiteRtIntelOpenVinoPerformanceModeThroughput);
  LITERT_ASSERT_OK_AND_ASSIGN(IntelOpenVinoOptions options,
                             IntelOpenVinoOptions::Create());

  ASSERT_TRUE(UpdateIntelOpenVinoOptionsFromFlags(options).HasValue());
  EXPECT_EQ(options.GetPerformanceMode(),
            kLiteRtIntelOpenVinoPerformanceModeThroughput);

  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_intel_openvino_performance_mode,
                kLiteRtIntelOpenVinoPerformanceModeLatency);
}

TEST(UpdateIntelOpenVinoOptionsFromFlagsTest, ConfigsMapSingleOption) {
  absl::SetFlag(&FLAGS_intel_openvino_configs_map,
                "INFERENCE_PRECISION_HINT=f16");
  LITERT_ASSERT_OK_AND_ASSIGN(IntelOpenVinoOptions options,
                             IntelOpenVinoOptions::Create());

  ASSERT_TRUE(UpdateIntelOpenVinoOptionsFromFlags(options).HasValue());
  // The options should be created successfully with the config map set

  // Reset flag to default
  absl::SetFlag(&FLAGS_intel_openvino_configs_map, "");
}

TEST(UpdateIntelOpenVinoOptionsFromFlagsTest, ConfigsMapMultipleOptions) {
  absl::SetFlag(&FLAGS_intel_openvino_configs_map,
                "INFERENCE_PRECISION_HINT=f16,NPU_COMPILATION_MODE_PARAMS=test,"
                "CACHE_DIR=/tmp/cache");
  LITERT_ASSERT_OK_AND_ASSIGN(IntelOpenVinoOptions options,
                             IntelOpenVinoOptions::Create());

  ASSERT_TRUE(UpdateIntelOpenVinoOptionsFromFlags(options).HasValue());
  // The options should be created successfully with multiple config map entries

  // Reset flag to default
  absl::SetFlag(&FLAGS_intel_openvino_configs_map, "");
}

TEST(UpdateIntelOpenVinoOptionsFromFlagsTest, ConfigsMapEmptyValue) {
  absl::SetFlag(&FLAGS_intel_openvino_configs_map, "");
  LITERT_ASSERT_OK_AND_ASSIGN(IntelOpenVinoOptions options,
                             IntelOpenVinoOptions::Create());

  ASSERT_TRUE(UpdateIntelOpenVinoOptionsFromFlags(options).HasValue());
  // Empty configs_map should work fine
}

TEST(UpdateIntelOpenVinoOptionsFromFlagsTest, ConfigsMapWithSpaces) {
  // Test handling of values with spaces (though typically avoided)
  absl::SetFlag(&FLAGS_intel_openvino_configs_map, "KEY1=VALUE1,KEY2=VALUE2");
  LITERT_ASSERT_OK_AND_ASSIGN(IntelOpenVinoOptions options,
                             IntelOpenVinoOptions::Create());

  ASSERT_TRUE(UpdateIntelOpenVinoOptionsFromFlags(options).HasValue());

  // Reset flag to default
  absl::SetFlag(&FLAGS_intel_openvino_configs_map, "");
}

TEST(UpdateIntelOpenVinoOptionsFromFlagsTest, ConfigsMapMalformedPairs) {
  // Test handling of malformed config strings (missing '=' or extra '=')
  // Should still create options successfully, but malformed pairs are ignored
  // with warning
  absl::SetFlag(
      &FLAGS_intel_openvino_configs_map,
      "GOOD_KEY=GOOD_VALUE,BAD_KEY_NO_EQUALS,KEY_WITH=MULTIPLE=EQUALS");
  LITERT_ASSERT_OK_AND_ASSIGN(IntelOpenVinoOptions options,
                             IntelOpenVinoOptions::Create());

  ASSERT_TRUE(UpdateIntelOpenVinoOptionsFromFlags(options).HasValue());
  // Only the well-formed pair should be set (BAD_KEY_NO_EQUALS will be ignored)
  // KEY_WITH=MULTIPLE=EQUALS will be split as KEY_WITH = MULTIPLE=EQUALS (3
  // parts, ignored)

  // Reset flag to default
  absl::SetFlag(&FLAGS_intel_openvino_configs_map, "");
}

}  // namespace
}  // namespace litert::intel_openvino
