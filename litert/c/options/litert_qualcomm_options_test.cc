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

#include "litert/c/options/litert_qualcomm_options.h"

#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/test/matchers.h"

namespace litert::qualcomm {
namespace {

TEST(LiteRtQualcommOptionsTest, CreateAndGet) {
  EXPECT_NE(LiteRtQualcommOptionsCreate(nullptr), kLiteRtStatusOk);

  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  const char* id;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsIdentifier(options, &id));
  EXPECT_STREQ(id, "qualcomm");

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, LogLevel) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(LiteRtQualcommOptionsSetLogLevel(
      qualcomm_options, kLiteRtQualcommLogLevelWarn));

  LiteRtQualcommOptionsLogLevel log_level;
  LITERT_ASSERT_OK(
      LiteRtQualcommOptionsGetLogLevel(qualcomm_options, &log_level));
  EXPECT_EQ(log_level, kLiteRtQualcommLogLevelWarn);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, UseHtpPreference) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(
      LiteRtQualcommOptionsSetUseHtpPreference(qualcomm_options, true));

  bool use_htp_preference;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGetUseHtpPreference(
      qualcomm_options, &use_htp_preference));
  EXPECT_TRUE(use_htp_preference);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, UseQint16AsQuint16) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(
      LiteRtQualcommOptionsSetUseQint16AsQuint16(qualcomm_options, false));

  bool use_qint16_as_quint16;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGetUseQint16AsQuint16(
      qualcomm_options, &use_qint16_as_quint16));
  EXPECT_FALSE(use_qint16_as_quint16);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, EnableWeightSharing) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(
      LiteRtQualcommOptionsSetEnableWeightSharing(qualcomm_options, false));

  bool weight_sharing;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGetEnableWeightSharing(
      qualcomm_options, &weight_sharing));
  EXPECT_FALSE(weight_sharing);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, HtpPerformanceMode) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(LiteRtQualcommOptionsSetHtpPerformanceMode(
      qualcomm_options, kLiteRtQualcommHtpPerformanceModeBurst));

  LiteRtQualcommOptionsHtpPerformanceMode htp_performance_mode;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGetHtpPerformanceMode(
      qualcomm_options, &htp_performance_mode));
  EXPECT_EQ(htp_performance_mode, kLiteRtQualcommHtpPerformanceModeBurst);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, Profiling) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(LiteRtQualcommOptionsSetProfiling(
      qualcomm_options, kLiteRtQualcommProfilingDetailed));

  LiteRtQualcommOptionsProfiling profiling;
  LITERT_ASSERT_OK(
      LiteRtQualcommOptionsGetProfiling(qualcomm_options, &profiling));
  EXPECT_EQ(profiling, kLiteRtQualcommProfilingDetailed);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, IrJsonDir) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(LiteRtQualcommOptionsSetIrJsonDir(qualcomm_options, "tmp/"));

  const char* ir_json_dir;
  LITERT_ASSERT_OK(
      LiteRtQualcommOptionsGetIrJsonDir(qualcomm_options, &ir_json_dir));
  EXPECT_STREQ(ir_json_dir, "tmp/");

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, VtcmSize) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(LiteRtQualcommOptionsSetVtcmSize(qualcomm_options, 4));

  uint32_t vtcm_size;
  LITERT_ASSERT_OK(
      LiteRtQualcommOptionsGetVtcmSize(qualcomm_options, &vtcm_size));
  EXPECT_EQ(vtcm_size, 4);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, HvxThread) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(LiteRtQualcommOptionsSetNumHvxThreads(qualcomm_options, 4));

  uint32_t num_hvx_threads;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGetNumHvxThreads(qualcomm_options,
                                                         &num_hvx_threads));
  EXPECT_EQ(num_hvx_threads, 4);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, OptimizationLevel) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(LiteRtQualcommOptionsSetOptimizationLevel(
      qualcomm_options, kHtpOptimizeForPrepare));

  LiteRtQualcommOptionsOptimizationLevel optimization_level;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGetOptimizationLevel(
      qualcomm_options, &optimization_level));
  EXPECT_EQ(optimization_level, kHtpOptimizeForPrepare);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtQualcommOptionsTest, DumpTensorIds) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  const std::vector<std::int32_t> kDumpTensorIds{1, 2, 3};
  LITERT_ASSERT_OK(LiteRtQualcommOptionsSetDumpTensorIds(
      qualcomm_options, kDumpTensorIds.data(), kDumpTensorIds.size()));

  std::int32_t* ids;
  std::uint32_t number_of_ids;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGetDumpTensorIds(qualcomm_options, &ids,
                                                         &number_of_ids));
  EXPECT_EQ(number_of_ids, kDumpTensorIds.size());
  for (size_t i = 0; i < kDumpTensorIds.size(); i++) {
    EXPECT_EQ(kDumpTensorIds[i], ids[i]);
  }

  LiteRtDestroyOpaqueOptions(options);
}

TEST(QualcommOptionsTest, CppApi) {
  auto options = QualcommOptions::Create();
  ASSERT_TRUE(options);

  EXPECT_EQ(options->GetLogLevel(), kLiteRtQualcommLogLevelInfo);
  options->SetLogLevel(kLiteRtQualcommLogLevelWarn);
  EXPECT_EQ(options->GetLogLevel(), kLiteRtQualcommLogLevelWarn);

  EXPECT_FALSE(options->GetEnableWeightSharing());
  options->SetEnableWeightSharing(true);
  EXPECT_TRUE(options->GetEnableWeightSharing());

  EXPECT_FALSE(options->GetUseHtpPreference());
  options->SetUseHtpPreference(true);
  EXPECT_TRUE(options->GetUseHtpPreference());

  EXPECT_FALSE(options->GetUseQint16AsQuint16());
  options->SetUseQint16AsQuint16(true);
  EXPECT_TRUE(options->GetUseQint16AsQuint16());

  EXPECT_EQ(options->GetHtpPerformanceMode(),
            kLiteRtQualcommHtpPerformanceModeDefault);
  options->SetHtpPerformanceMode(kLiteRtQualcommHtpPerformanceModeBurst);
  EXPECT_EQ(options->GetHtpPerformanceMode(),
            kLiteRtQualcommHtpPerformanceModeBurst);

  EXPECT_EQ(options->GetProfiling(), kLiteRtQualcommProfilingOff);
  options->SetProfiling(kLiteRtQualcommProfilingDetailed);
  EXPECT_EQ(options->GetProfiling(), kLiteRtQualcommProfilingDetailed);

  const std::vector<std::int32_t> kDumpTensorIds{1, 2, 3};
  EXPECT_TRUE(options->GetDumpTensorIds().empty());
  options->SetDumpTensorIds(kDumpTensorIds);
  auto ids = options->GetDumpTensorIds();
  for (size_t i = 0; i < kDumpTensorIds.size(); i++) {
    EXPECT_EQ(ids[i], kDumpTensorIds[i]);
  }

  EXPECT_EQ(options->GetIrJsonDir(), "");
  options->SetIrJsonDir("tmp");
  EXPECT_EQ(options->GetIrJsonDir(), "tmp");

  EXPECT_EQ(options->GetVtcmSize(), 0);
  options->SetVtcmSize(4);
  EXPECT_EQ(options->GetVtcmSize(), 4);

  EXPECT_EQ(options->GetNumHvxThreads(), 0);
  options->SetNumHvxThreads(4);
  EXPECT_EQ(options->GetNumHvxThreads(), 4);

  EXPECT_EQ(options->GetOptimizationLevel(), kHtpOptimizeForInferenceO3);
  options->SetOptimizationLevel(kHtpOptimizeForPrepare);
  EXPECT_EQ(options->GetOptimizationLevel(), kHtpOptimizeForPrepare);
}

TEST(QualcommOptionsTest, FindFromChain) {
  void* payload = malloc(8);
  auto options =
      OpaqueOptions::Create("not-qualcomm", payload, [](void* d) { free(d); });
  ASSERT_TRUE(options);
  auto qnn_options = QualcommOptions::Create();
  ASSERT_TRUE(qnn_options);
  options->Append(std::move(*qnn_options));

  auto qnn_opts_d = FindOpaqueOptions<QualcommOptions>(*options);
  EXPECT_TRUE(qnn_opts_d);
}

}  // namespace
}  // namespace litert::qualcomm
