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
  QualcommOptions::LogLevel value;

  EXPECT_FALSE(AbslParseFlag("oogabooga", &value, &error));
}

TEST(LogLevelFlagTest, Parse) {
  std::string error;
  QualcommOptions::LogLevel value;

  {
    static constexpr absl::string_view kLevel = "off";
    static constexpr QualcommOptions::LogLevel kLevelEnum =
        QualcommOptions::LogLevel::kOff;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "error";
    static constexpr QualcommOptions::LogLevel kLevelEnum =
        QualcommOptions::LogLevel::kError;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "warn";
    static constexpr QualcommOptions::LogLevel kLevelEnum =
        QualcommOptions::LogLevel::kWarn;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "info";
    static constexpr QualcommOptions::LogLevel kLevelEnum =
        QualcommOptions::LogLevel::kInfo;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "verbose";
    static constexpr QualcommOptions::LogLevel kLevelEnum =
        QualcommOptions::LogLevel::kVerbose;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "debug";
    static constexpr QualcommOptions::LogLevel kLevelEnum =
        QualcommOptions::LogLevel::kDebug;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }
}

TEST(HtpPerformanceModeTest, Malformed) {
  std::string error;
  QualcommOptions::HtpPerformanceMode value;

  EXPECT_FALSE(AbslParseFlag("boogabooga", &value, &error));
}

TEST(HtpPerformanceModeTest, Parse) {
  std::string error;
  QualcommOptions::HtpPerformanceMode value;

  {
    static constexpr absl::string_view kMode = "default";
    static constexpr QualcommOptions::HtpPerformanceMode kModeEnum =
        QualcommOptions::HtpPerformanceMode::kDefault;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "sustained_high_performance";
    static constexpr QualcommOptions::HtpPerformanceMode kModeEnum =
        QualcommOptions::HtpPerformanceMode::kSustainedHighPerformance;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "burst";
    static constexpr QualcommOptions::HtpPerformanceMode kModeEnum =
        QualcommOptions::HtpPerformanceMode::kBurst;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "high_performance";
    static constexpr QualcommOptions::HtpPerformanceMode kModeEnum =
        QualcommOptions::HtpPerformanceMode::kHighPerformance;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "power_saver";
    static constexpr QualcommOptions::HtpPerformanceMode kModeEnum =
        QualcommOptions::HtpPerformanceMode::kPowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "low_power_saver";
    static constexpr QualcommOptions::HtpPerformanceMode kModeEnum =
        QualcommOptions::HtpPerformanceMode::kLowPowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "high_power_saver";
    static constexpr QualcommOptions::HtpPerformanceMode kModeEnum =
        QualcommOptions::HtpPerformanceMode::kHighPowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "low_balanced";
    static constexpr QualcommOptions::HtpPerformanceMode kModeEnum =
        QualcommOptions::HtpPerformanceMode::kLowBalanced;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "balanced";
    static constexpr QualcommOptions::HtpPerformanceMode kModeEnum =
        QualcommOptions::HtpPerformanceMode::kBalanced;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "extreme_power_saver";
    static constexpr QualcommOptions::HtpPerformanceMode kModeEnum =
        QualcommOptions::HtpPerformanceMode::kExtremePowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }
}

TEST(DspPerformanceModeTest, Malformed) {
  std::string error;
  QualcommOptions::DspPerformanceMode value;

  EXPECT_FALSE(AbslParseFlag("boogabooga", &value, &error));
}

TEST(DspPerformanceModeTest, Parse) {
  std::string error;
  QualcommOptions::DspPerformanceMode value;

  {
    static constexpr absl::string_view kMode = "default";
    static constexpr QualcommOptions::DspPerformanceMode kModeEnum =
        QualcommOptions::DspPerformanceMode::kDefault;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "sustained_high_performance";
    static constexpr QualcommOptions::DspPerformanceMode kModeEnum =
        QualcommOptions::DspPerformanceMode::kSustainedHighPerformance;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "burst";
    static constexpr QualcommOptions::DspPerformanceMode kModeEnum =
        QualcommOptions::DspPerformanceMode::kBurst;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "high_performance";
    static constexpr QualcommOptions::DspPerformanceMode kModeEnum =
        QualcommOptions::DspPerformanceMode::kHighPerformance;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "power_saver";
    static constexpr QualcommOptions::DspPerformanceMode kModeEnum =
        QualcommOptions::DspPerformanceMode::kPowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "low_power_saver";
    static constexpr QualcommOptions::DspPerformanceMode kModeEnum =
        QualcommOptions::DspPerformanceMode::kLowPowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "high_power_saver";
    static constexpr QualcommOptions::DspPerformanceMode kModeEnum =
        QualcommOptions::DspPerformanceMode::kHighPowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "low_balanced";
    static constexpr QualcommOptions::DspPerformanceMode kModeEnum =
        QualcommOptions::DspPerformanceMode::kLowBalanced;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "balanced";
    static constexpr QualcommOptions::DspPerformanceMode kModeEnum =
        QualcommOptions::DspPerformanceMode::kBalanced;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }
}

TEST(ProfilingTest, Malformed) {
  std::string error;
  QualcommOptions::Profiling value;

  EXPECT_FALSE(AbslParseFlag("boogabooga", &value, &error));
}

TEST(ProfilingTest, Parse) {
  std::string error;
  QualcommOptions::Profiling value;

  {
    static constexpr absl::string_view kProfiling = "off";
    static constexpr QualcommOptions::Profiling kProfilingEnum =
        QualcommOptions::Profiling::kOff;
    EXPECT_TRUE(AbslParseFlag(kProfiling, &value, &error));
    EXPECT_EQ(value, kProfilingEnum);
    EXPECT_EQ(kProfiling, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kProfiling = "basic";
    static constexpr QualcommOptions::Profiling kProfilingEnum =
        QualcommOptions::Profiling::kBasic;
    EXPECT_TRUE(AbslParseFlag(kProfiling, &value, &error));
    EXPECT_EQ(value, kProfilingEnum);
    EXPECT_EQ(kProfiling, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kProfiling = "detailed";
    static constexpr QualcommOptions::Profiling kProfilingEnum =
        QualcommOptions::Profiling::kDetailed;
    EXPECT_TRUE(AbslParseFlag(kProfiling, &value, &error));
    EXPECT_EQ(value, kProfilingEnum);
    EXPECT_EQ(kProfiling, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kProfiling = "linting";
    static constexpr QualcommOptions::Profiling kProfilingEnum =
        QualcommOptions::Profiling::kLinting;
    EXPECT_TRUE(AbslParseFlag(kProfiling, &value, &error));
    EXPECT_EQ(value, kProfilingEnum);
    EXPECT_EQ(kProfiling, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kProfiling = "optrace";
    static constexpr QualcommOptions::Profiling kProfilingEnum =
        QualcommOptions::Profiling::kOptrace;
    EXPECT_TRUE(AbslParseFlag(kProfiling, &value, &error));
    EXPECT_EQ(value, kProfilingEnum);
    EXPECT_EQ(kProfiling, AbslUnparseFlag(value));
  }
}

TEST(OptimizationLevelTest, Parse) {
  std::string error;
  QualcommOptions::OptimizationLevel value;

  {
    static constexpr absl::string_view kOptimizationLevel = "O1";
    static constexpr QualcommOptions::OptimizationLevel kOptimizationLevelEnum =
        QualcommOptions::OptimizationLevel::kOptimizeForInference;
    EXPECT_TRUE(AbslParseFlag(kOptimizationLevel, &value, &error));
    EXPECT_EQ(value, kOptimizationLevelEnum);
    EXPECT_EQ(kOptimizationLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kOptimizationLevel = "O2";
    static constexpr QualcommOptions::OptimizationLevel kOptimizationLevelEnum =
        QualcommOptions::OptimizationLevel::kOptimizeForPrepare;
    EXPECT_TRUE(AbslParseFlag(kOptimizationLevel, &value, &error));
    EXPECT_EQ(value, kOptimizationLevelEnum);
    EXPECT_EQ(kOptimizationLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kOptimizationLevel = "O3";
    static constexpr QualcommOptions::OptimizationLevel kOptimizationLevelEnum =
        QualcommOptions::OptimizationLevel::kOptimizeForInferenceO3;
    EXPECT_TRUE(AbslParseFlag(kOptimizationLevel, &value, &error));
    EXPECT_EQ(value, kOptimizationLevelEnum);
    EXPECT_EQ(kOptimizationLevel, AbslUnparseFlag(value));
  }
}

TEST(GraphPriorityTest, Parse) {
  std::string error;
  QualcommOptions::GraphPriority value;

  {
    static constexpr absl::string_view kGraphPriority = "default";
    static constexpr QualcommOptions::GraphPriority kGraphPriorityEnum =
        QualcommOptions::GraphPriority::kDefault;
    EXPECT_TRUE(AbslParseFlag(kGraphPriority, &value, &error));
    EXPECT_EQ(value, kGraphPriorityEnum);
    EXPECT_EQ(kGraphPriority, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kGraphPriority = "low";
    static constexpr QualcommOptions::GraphPriority kGraphPriorityEnum =
        QualcommOptions::GraphPriority::kLow;
    EXPECT_TRUE(AbslParseFlag(kGraphPriority, &value, &error));
    EXPECT_EQ(value, kGraphPriorityEnum);
    EXPECT_EQ(kGraphPriority, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kGraphPriority = "normal";
    static constexpr QualcommOptions::GraphPriority kGraphPriorityEnum =
        QualcommOptions::GraphPriority::kNormal;
    EXPECT_TRUE(AbslParseFlag(kGraphPriority, &value, &error));
    EXPECT_EQ(value, kGraphPriorityEnum);
    EXPECT_EQ(kGraphPriority, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kGraphPriority = "normal_high";
    static constexpr QualcommOptions::GraphPriority kGraphPriorityEnum =
        QualcommOptions::GraphPriority::kNormalHigh;
    EXPECT_TRUE(AbslParseFlag(kGraphPriority, &value, &error));
    EXPECT_EQ(value, kGraphPriorityEnum);
    EXPECT_EQ(kGraphPriority, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kGraphPriority = "high";
    static constexpr QualcommOptions::GraphPriority kGraphPriorityEnum =
        QualcommOptions::GraphPriority::kHigh;
    EXPECT_TRUE(AbslParseFlag(kGraphPriority, &value, &error));
    EXPECT_EQ(value, kGraphPriorityEnum);
    EXPECT_EQ(kGraphPriority, AbslUnparseFlag(value));
  }
}

TEST(BackendTest, Parse) {
  std::string error;
  QualcommOptions::Backend value;

  {
    static constexpr absl::string_view kBackend = "gpu";
    static constexpr QualcommOptions::Backend kBackendEnum =
        QualcommOptions::Backend::kGpu;
    EXPECT_TRUE(AbslParseFlag(kBackend, &value, &error));
    EXPECT_EQ(value, kBackendEnum);
    EXPECT_EQ(kBackend, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kBackend = "htp";
    static constexpr QualcommOptions::Backend kBackendEnum =
        QualcommOptions::Backend::kHtp;
    EXPECT_TRUE(AbslParseFlag(kBackend, &value, &error));
    EXPECT_EQ(value, kBackendEnum);
    EXPECT_EQ(kBackend, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kBackend = "dsp";
    static constexpr QualcommOptions::Backend kBackendEnum =
        QualcommOptions::Backend::kDsp;
    EXPECT_TRUE(AbslParseFlag(kBackend, &value, &error));
    EXPECT_EQ(value, kBackendEnum);
    EXPECT_EQ(kBackend, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kBackend = "ir";
    static constexpr QualcommOptions::Backend kBackendEnum =
        QualcommOptions::Backend::kIr;
    EXPECT_TRUE(AbslParseFlag(kBackend, &value, &error));
    EXPECT_EQ(value, kBackendEnum);
    EXPECT_EQ(kBackend, AbslUnparseFlag(value));
  }
}

TEST(QualcommOptionsFromFlagsTest, DefaultValue) {
  Expected<QualcommOptions> options = QualcommOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateQualcommOptionsFromFlags(options.Value()).HasValue());
  EXPECT_EQ(options.Value().GetLogLevel(), QualcommOptions::LogLevel::kInfo);
  EXPECT_EQ(options.Value().GetProfiling(), QualcommOptions::Profiling::kOff);
  EXPECT_FALSE(options.Value().GetUseHtpPreference());
  EXPECT_FALSE(options.Value().GetUseQint16AsQuint16());
  EXPECT_FALSE(options.Value().GetEnableWeightSharing());
  EXPECT_TRUE(options.Value().GetUseConvHMX());
  EXPECT_TRUE(options.Value().GetUseFoldReLU());
  EXPECT_EQ(options.Value().GetHtpPerformanceMode(),
            QualcommOptions::HtpPerformanceMode::kDefault);
  EXPECT_TRUE(options.Value().GetDumpTensorIds().empty());
  EXPECT_EQ(options.Value().GetVtcmSize(), 0);
  EXPECT_EQ(options.Value().GetNumHvxThreads(), 0);
  EXPECT_EQ(options.Value().GetOptimizationLevel(),
            QualcommOptions::OptimizationLevel::kOptimizeForInferenceO3);
  EXPECT_EQ(options.Value().GetGraphPriority(),
            QualcommOptions::GraphPriority::kDefault);
  EXPECT_EQ(options.Value().GetBackend(), QualcommOptions::Backend::kHtp);
}

}  // namespace
}  // namespace litert::qualcomm
