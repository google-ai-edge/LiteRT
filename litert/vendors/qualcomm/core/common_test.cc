// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/common.h"

#include <gtest/gtest.h>

namespace qnn {
namespace {

TEST(QnnOptionTest, LogLevel) {
  Options options;
  static constexpr LogLevel kLogOff = LogLevel::kOff;
  options.SetLogLevel(kLogOff);
  EXPECT_EQ(options.GetLogLevel(), kLogOff);

  static constexpr LogLevel kLogError = LogLevel::kError;
  options.SetLogLevel(kLogError);
  EXPECT_EQ(options.GetLogLevel(), kLogError);

  static constexpr LogLevel kLogWarn = LogLevel::kWarn;
  options.SetLogLevel(kLogWarn);
  EXPECT_EQ(options.GetLogLevel(), kLogWarn);

  static constexpr LogLevel kLogInfo = LogLevel::kInfo;
  options.SetLogLevel(kLogInfo);
  EXPECT_EQ(options.GetLogLevel(), kLogInfo);

  static constexpr LogLevel kLogVerbose = LogLevel::kVerbose;
  options.SetLogLevel(kLogVerbose);
  EXPECT_EQ(options.GetLogLevel(), kLogVerbose);

  static constexpr LogLevel kLogDebug = LogLevel::kDebug;
  options.SetLogLevel(kLogDebug);
  EXPECT_EQ(options.GetLogLevel(), kLogDebug);
}

TEST(QnnOptionTest, BackendType) {
  Options options;
  static constexpr BackendType kUndefined = BackendType::kUndefinedBackend;
  options.SetBackendType(kUndefined);
  EXPECT_EQ(options.GetBackendType(), kUndefined);

  static constexpr BackendType kHtp = BackendType::kHtpBackend;
  options.SetBackendType(kHtp);
  EXPECT_EQ(options.GetBackendType(), kHtp);

  static constexpr BackendType kIr = BackendType::kIrBackend;
  options.SetBackendType(kIr);
  EXPECT_EQ(options.GetBackendType(), kIr);

  static constexpr BackendType kDsp = BackendType::kDspBackend;
  options.SetBackendType(kDsp);
  EXPECT_EQ(options.GetBackendType(), kDsp);

  static constexpr BackendType kGpu = BackendType::kGpuBackend;
  options.SetBackendType(kGpu);
  EXPECT_EQ(options.GetBackendType(), kGpu);
}

TEST(QnnOptionTest, HtpPerformanceMode) {
  Options options;

  static constexpr HtpPerformanceMode kDefault = HtpPerformanceMode::kDefault;
  options.SetHtpPerformanceMode(kDefault);
  EXPECT_EQ(options.GetHtpPerformanceMode(), kDefault);

  static constexpr HtpPerformanceMode kSustainedHighPerformance =
      HtpPerformanceMode::kSustainedHighPerformance;
  options.SetHtpPerformanceMode(kSustainedHighPerformance);
  EXPECT_EQ(options.GetHtpPerformanceMode(), kSustainedHighPerformance);

  static constexpr HtpPerformanceMode kBurst = HtpPerformanceMode::kBurst;
  options.SetHtpPerformanceMode(kBurst);
  EXPECT_EQ(options.GetHtpPerformanceMode(), kBurst);

  static constexpr HtpPerformanceMode kHighPerformance =
      HtpPerformanceMode::kHighPerformance;
  options.SetHtpPerformanceMode(kHighPerformance);
  EXPECT_EQ(options.GetHtpPerformanceMode(), kHighPerformance);

  static constexpr HtpPerformanceMode kPowerSaver =
      HtpPerformanceMode::kPowerSaver;
  options.SetHtpPerformanceMode(kPowerSaver);
  EXPECT_EQ(options.GetHtpPerformanceMode(), kPowerSaver);

  static constexpr HtpPerformanceMode kLowPowerSaver =
      HtpPerformanceMode::kLowPowerSaver;
  options.SetHtpPerformanceMode(kLowPowerSaver);
  EXPECT_EQ(options.GetHtpPerformanceMode(), kLowPowerSaver);

  static constexpr HtpPerformanceMode kHighPowerSaver =
      HtpPerformanceMode::kHighPowerSaver;
  options.SetHtpPerformanceMode(kHighPowerSaver);
  EXPECT_EQ(options.GetHtpPerformanceMode(), kHighPowerSaver);

  static constexpr HtpPerformanceMode kLowBalanced =
      HtpPerformanceMode::kLowBalanced;
  options.SetHtpPerformanceMode(kLowBalanced);
  EXPECT_EQ(options.GetHtpPerformanceMode(), kLowBalanced);

  static constexpr HtpPerformanceMode kBalanced = HtpPerformanceMode::kBalanced;
  options.SetHtpPerformanceMode(kBalanced);
  EXPECT_EQ(options.GetHtpPerformanceMode(), kBalanced);

  static constexpr HtpPerformanceMode kExtremePowerSaver =
      HtpPerformanceMode::kExtremePowerSaver;
  options.SetHtpPerformanceMode(kExtremePowerSaver);
  EXPECT_EQ(options.GetHtpPerformanceMode(), kExtremePowerSaver);
}

TEST(QnnOptionTest, DspPerformanceMode) {
  Options options;

  static constexpr DspPerformanceMode kDefault = DspPerformanceMode::kDefault;
  options.SetDspPerformanceMode(kDefault);
  EXPECT_EQ(options.GetDspPerformanceMode(), kDefault);

  static constexpr DspPerformanceMode kSustainedHighPerformance =
      DspPerformanceMode::kSustainedHighPerformance;
  options.SetDspPerformanceMode(kSustainedHighPerformance);
  EXPECT_EQ(options.GetDspPerformanceMode(), kSustainedHighPerformance);

  static constexpr DspPerformanceMode kBurst = DspPerformanceMode::kBurst;
  options.SetDspPerformanceMode(kBurst);
  EXPECT_EQ(options.GetDspPerformanceMode(), kBurst);

  static constexpr DspPerformanceMode kHighPerformance =
      DspPerformanceMode::kHighPerformance;
  options.SetDspPerformanceMode(kHighPerformance);
  EXPECT_EQ(options.GetDspPerformanceMode(), kHighPerformance);

  static constexpr DspPerformanceMode kPowerSaver =
      DspPerformanceMode::kPowerSaver;
  options.SetDspPerformanceMode(kPowerSaver);
  EXPECT_EQ(options.GetDspPerformanceMode(), kPowerSaver);

  static constexpr DspPerformanceMode kLowPowerSaver =
      DspPerformanceMode::kLowPowerSaver;
  options.SetDspPerformanceMode(kLowPowerSaver);
  EXPECT_EQ(options.GetDspPerformanceMode(), kLowPowerSaver);

  static constexpr DspPerformanceMode kHighPowerSaver =
      DspPerformanceMode::kHighPowerSaver;
  options.SetDspPerformanceMode(kHighPowerSaver);
  EXPECT_EQ(options.GetDspPerformanceMode(), kHighPowerSaver);

  static constexpr DspPerformanceMode kLowBalanced =
      DspPerformanceMode::kLowBalanced;
  options.SetDspPerformanceMode(kLowBalanced);
  EXPECT_EQ(options.GetDspPerformanceMode(), kLowBalanced);

  static constexpr DspPerformanceMode kBalanced = DspPerformanceMode::kBalanced;
  options.SetDspPerformanceMode(kBalanced);
  EXPECT_EQ(options.GetDspPerformanceMode(), kBalanced);
}

TEST(QnnOptionTest, UseHtpPreference) {
  Options options;
  options.SetUseHtpPreference(true);
  EXPECT_EQ(options.GetUseHtpPreference(), true);
  options.SetUseHtpPreference(false);
  EXPECT_EQ(options.GetUseHtpPreference(), false);
}

TEST(QnnOptionTest, UseQint16AsQuint16) {
  Options options;
  options.SetUseQint16AsQuint16(true);
  EXPECT_EQ(options.GetUseQint16AsQuint16(), true);
  options.SetUseQint16AsQuint16(false);
  EXPECT_EQ(options.GetUseQint16AsQuint16(), false);
}

TEST(QnnOptionTest, EnableWeightSharing) {
  Options options;
  options.SetEnableWeightSharing(true);
  EXPECT_EQ(options.GetEnableWeightSharing(), true);
  options.SetEnableWeightSharing(false);
  EXPECT_EQ(options.GetEnableWeightSharing(), false);
}

TEST(QnnOptionTest, UseConvHMX) {
  Options options;
  options.SetUseConvHMX(true);
  EXPECT_EQ(options.GetUseConvHMX(), true);
  options.SetUseConvHMX(false);
  EXPECT_EQ(options.GetUseConvHMX(), false);
}

TEST(QnnOptionTest, UseFoldReLU) {
  Options options;
  options.SetUseFoldReLU(true);
  EXPECT_EQ(options.GetUseFoldReLU(), true);
  options.SetUseFoldReLU(false);
  EXPECT_EQ(options.GetUseFoldReLU(), false);
}

TEST(QnnOptionTest, SetIrJsonDir) {
  Options options;
  options.SetIrJsonDir("tmp/");
  EXPECT_FALSE(options.GetIrJsonDir().empty());
  EXPECT_EQ(options.GetIrJsonDir(), "tmp/");
  options.SetIrJsonDir("");
  EXPECT_TRUE(options.GetIrJsonDir().empty());
}

TEST(QnnOptionTest, SetDlcDir) {
  Options options;
  options.SetDlcDir("tmp/");
  EXPECT_FALSE(options.GetDlcDir().empty());
  EXPECT_EQ(options.GetDlcDir(), "tmp/");
  options.SetDlcDir("");
  EXPECT_TRUE(options.GetDlcDir().empty());
}

TEST(QnnOptionTest, SetVtcmSize) {
  Options options;
  options.SetVtcmSize(4);
  EXPECT_NE(options.GetVtcmSize(), 0);
  EXPECT_EQ(options.GetVtcmSize(), 4);
  options.SetVtcmSize(0);
  EXPECT_EQ(options.GetVtcmSize(), 0);
}

TEST(QnnOptionTest, SetHvxThread) {
  Options options;
  options.SetNumHvxThreads(4);
  EXPECT_NE(options.GetNumHvxThreads(), 0);
  EXPECT_EQ(options.GetNumHvxThreads(), 4);
  options.SetNumHvxThreads(0);
  EXPECT_EQ(options.GetNumHvxThreads(), 0);
}

TEST(QnnOptionTest, SetOptimizationLevel) {
  Options options;
  options.SetOptimizationLevel(OptimizationLevel::kHtpOptimizeForPrepare);
  EXPECT_NE(options.GetOptimizationLevel(),
            OptimizationLevel::kHtpOptimizeForInferenceO3);
  EXPECT_EQ(options.GetOptimizationLevel(),
            OptimizationLevel::kHtpOptimizeForPrepare);
  options.SetOptimizationLevel(OptimizationLevel::kHtpOptimizeForInferenceO3);
  EXPECT_EQ(options.GetOptimizationLevel(),
            OptimizationLevel::kHtpOptimizeForInferenceO3);
}

TEST(QnnOptionTest, SetGraphPriority) {
  Options options;
  options.SetGraphPriority(GraphPriority::kHigh);
  EXPECT_NE(options.GetGraphPriority(), GraphPriority::kDefault);
  EXPECT_EQ(options.GetGraphPriority(), GraphPriority::kHigh);
  options.SetGraphPriority(GraphPriority::kDefault);
  EXPECT_EQ(options.GetGraphPriority(), GraphPriority::kDefault);
}

TEST(QnnOptionTest, Default) {
  Options options;
  EXPECT_EQ(options.GetLogLevel(), LogLevel::kInfo);
  EXPECT_EQ(options.GetBackendType(), BackendType::kHtpBackend);
  EXPECT_EQ(options.GetProfiling(), Profiling::kOff);
  EXPECT_FALSE(options.GetUseHtpPreference());
  EXPECT_FALSE(options.GetUseQint16AsQuint16());
  EXPECT_FALSE(options.GetEnableWeightSharing());
  EXPECT_TRUE(options.GetUseConvHMX());
  EXPECT_TRUE(options.GetUseFoldReLU());
  EXPECT_EQ(options.GetHtpPerformanceMode(), HtpPerformanceMode::kDefault);
  EXPECT_EQ(options.GetDspPerformanceMode(), DspPerformanceMode::kDefault);
  EXPECT_TRUE(options.GetIrJsonDir().empty());
  EXPECT_TRUE(options.GetDlcDir().empty());
  EXPECT_EQ(options.GetVtcmSize(), 0);
  EXPECT_EQ(options.GetNumHvxThreads(), 0);
  EXPECT_EQ(options.GetOptimizationLevel(),
            OptimizationLevel::kHtpOptimizeForInferenceO3);
  EXPECT_EQ(options.GetGraphPriority(), GraphPriority::kDefault);
}

}  // namespace
}  // namespace qnn
