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

#include "litert/tools/flags/vendors/qualcomm_flags.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_qualcomm_options.h"

namespace litert::qualcomm {
namespace {

TEST(LogLevelFlagTest, Malformed) {
  std::string error;
  LiteRtQualcommOptionsLogLevel value;

  EXPECT_FALSE(AbslParseFlag("oogabooga", &value, &error));
}

TEST(LogLevelFlagTest, Parse) {
  std::string error;
  LiteRtQualcommOptionsLogLevel value;

  {
    static constexpr absl::string_view kLevel = "off";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogOff;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "error";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogLevelError;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "warn";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogLevelWarn;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "info";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogLevelInfo;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "verbose";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogLevelVerbose;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "debug";
    static constexpr LiteRtQualcommOptionsLogLevel kLevelEnum =
        kLiteRtQualcommLogLevelDebug;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }
}

TEST(HtpPerformanceModeTest, Malformed) {
  std::string error;
  LiteRtQualcommOptionsHtpPerformanceMode value;

  EXPECT_FALSE(AbslParseFlag("boogabooga", &value, &error));
}

TEST(HtpPerformanceModeTest, Parse) {
  std::string error;
  LiteRtQualcommOptionsHtpPerformanceMode value;

  {
    static constexpr absl::string_view kMode = "default";
    static constexpr LiteRtQualcommOptionsHtpPerformanceMode kModeEnum =
        kLiteRtQualcommHtpPerformanceModeDefault;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "sustained_high_performance";
    static constexpr LiteRtQualcommOptionsHtpPerformanceMode kModeEnum =
        kLiteRtQualcommHtpPerformanceModeSustainedHighPerformance;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "burst";
    static constexpr LiteRtQualcommOptionsHtpPerformanceMode kModeEnum =
        kLiteRtQualcommHtpPerformanceModeBurst;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "high_performance";
    static constexpr LiteRtQualcommOptionsHtpPerformanceMode kModeEnum =
        kLiteRtQualcommHtpPerformanceModeHighPerformance;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "power_saver";
    static constexpr LiteRtQualcommOptionsHtpPerformanceMode kModeEnum =
        kLiteRtQualcommHtpPerformanceModePowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "low_power_saver";
    static constexpr LiteRtQualcommOptionsHtpPerformanceMode kModeEnum =
        kLiteRtQualcommHtpPerformanceModeLowPowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "high_power_saver";
    static constexpr LiteRtQualcommOptionsHtpPerformanceMode kModeEnum =
        kLiteRtQualcommHtpPerformanceModeHighPowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "low_balanced";
    static constexpr LiteRtQualcommOptionsHtpPerformanceMode kModeEnum =
        kLiteRtQualcommHtpPerformanceModeLowBalanced;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "balanced";
    static constexpr LiteRtQualcommOptionsHtpPerformanceMode kModeEnum =
        kLiteRtQualcommHtpPerformanceModeBalanced;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "extreme_power_saver";
    static constexpr LiteRtQualcommOptionsHtpPerformanceMode kModeEnum =
        kLiteRtQualcommHtpPerformanceModeExtremePowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }
}

TEST(ProfilingTest, Malformed) {
  std::string error;
  LiteRtQualcommOptionsProfiling value;

  EXPECT_FALSE(AbslParseFlag("boogabooga", &value, &error));
}

TEST(ProfilingTest, Parse) {
  std::string error;
  LiteRtQualcommOptionsProfiling value;

  {
    static constexpr absl::string_view kProfiling = "off";
    static constexpr LiteRtQualcommOptionsProfiling kProfilingEnum =
        kLiteRtQualcommProfilingOff;
    EXPECT_TRUE(AbslParseFlag(kProfiling, &value, &error));
    EXPECT_EQ(value, kProfilingEnum);
    EXPECT_EQ(kProfiling, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kProfiling = "basic";
    static constexpr LiteRtQualcommOptionsProfiling kProfilingEnum =
        kLiteRtQualcommProfilingBasic;
    EXPECT_TRUE(AbslParseFlag(kProfiling, &value, &error));
    EXPECT_EQ(value, kProfilingEnum);
    EXPECT_EQ(kProfiling, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kProfiling = "detailed";
    static constexpr LiteRtQualcommOptionsProfiling kProfilingEnum =
        kLiteRtQualcommProfilingDetailed;
    EXPECT_TRUE(AbslParseFlag(kProfiling, &value, &error));
    EXPECT_EQ(value, kProfilingEnum);
    EXPECT_EQ(kProfiling, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kProfiling = "linting";
    static constexpr LiteRtQualcommOptionsProfiling kProfilingEnum =
        kLiteRtQualcommProfilingLinting;
    EXPECT_TRUE(AbslParseFlag(kProfiling, &value, &error));
    EXPECT_EQ(value, kProfilingEnum);
    EXPECT_EQ(kProfiling, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kProfiling = "optrace";
    static constexpr LiteRtQualcommOptionsProfiling kProfilingEnum =
        kLiteRtQualcommProfilingOptrace;
    EXPECT_TRUE(AbslParseFlag(kProfiling, &value, &error));
    EXPECT_EQ(value, kProfilingEnum);
    EXPECT_EQ(kProfiling, AbslUnparseFlag(value));
  }
}

TEST(OptimizationLevelTest, Parse) {
  std::string error;
  LiteRtQualcommOptionsOptimizationLevel value;

  {
    static constexpr absl::string_view kOptimizationLevel = "O1";
    static constexpr LiteRtQualcommOptionsOptimizationLevel
        kOptimizationLevelEnum = kHtpOptimizeForInference;
    EXPECT_TRUE(AbslParseFlag(kOptimizationLevel, &value, &error));
    EXPECT_EQ(value, kOptimizationLevelEnum);
    EXPECT_EQ(kOptimizationLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kOptimizationLevel = "O2";
    static constexpr LiteRtQualcommOptionsOptimizationLevel
        kOptimizationLevelEnum = kHtpOptimizeForPrepare;
    EXPECT_TRUE(AbslParseFlag(kOptimizationLevel, &value, &error));
    EXPECT_EQ(value, kOptimizationLevelEnum);
    EXPECT_EQ(kOptimizationLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kOptimizationLevel = "O3";
    static constexpr LiteRtQualcommOptionsOptimizationLevel
        kOptimizationLevelEnum = kHtpOptimizeForInferenceO3;
    EXPECT_TRUE(AbslParseFlag(kOptimizationLevel, &value, &error));
    EXPECT_EQ(value, kOptimizationLevelEnum);
    EXPECT_EQ(kOptimizationLevel, AbslUnparseFlag(value));
  }
}

TEST(QualcommOptionsFromFlagsTest, DefaultValue) {
  Expected<QualcommOptions> options = QualcommOptionsFromFlags();
  ASSERT_TRUE(options.HasValue());
  EXPECT_EQ(options.Value().GetLogLevel(), kLiteRtQualcommLogLevelInfo);
  EXPECT_EQ(options.Value().GetProfiling(), kLiteRtQualcommProfilingOff);
  EXPECT_FALSE(options.Value().GetUseHtpPreference());
  EXPECT_FALSE(options.Value().GetUseQint16AsQuint16());
  EXPECT_FALSE(options.Value().GetEnableWeightSharing());
  EXPECT_TRUE(options.Value().GetUseConvHMX());
  EXPECT_TRUE(options.Value().GetUseFoldReLU());
  EXPECT_EQ(options.Value().GetHtpPerformanceMode(),
            kLiteRtQualcommHtpPerformanceModeDefault);
  EXPECT_TRUE(options.Value().GetDumpTensorIds().empty());
  EXPECT_EQ(options.Value().GetVtcmSize(), 0);
  EXPECT_EQ(options.Value().GetNumHvxThreads(), 0);
  EXPECT_EQ(options.Value().GetOptimizationLevel(), kHtpOptimizeForInferenceO3);
}

}  // namespace
}  // namespace litert::qualcomm
