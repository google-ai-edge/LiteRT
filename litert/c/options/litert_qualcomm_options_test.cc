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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/test/matchers.h"

namespace litert::qualcomm {
namespace {

QualcommOptions SerializeAndParse(LrtQualcommOptions options) {
  const char* identifier;
  void* payload;
  void (*payload_deleter)(void*);
  EXPECT_EQ(LrtGetOpaqueQualcommOptionsData(options, &identifier, &payload,
                                            &payload_deleter),
            kLiteRtStatusOk);

  LiteRtOpaqueOptions opaque_options;
  EXPECT_EQ(LiteRtCreateOpaqueOptions(identifier, payload, payload_deleter,
                                      &opaque_options),
            kLiteRtStatusOk);

  void* opaque_payload;
  EXPECT_EQ(LiteRtGetOpaqueOptionsData(opaque_options, &opaque_payload),
            kLiteRtStatusOk);
  absl::string_view toml(static_cast<const char*>(opaque_payload));
  LrtQualcommOptions options_handle = nullptr;
  EXPECT_EQ(LrtCreateQualcommOptionsFromToml(toml.data(), &options_handle),
            kLiteRtStatusOk);
  auto result = litert::qualcomm::QualcommOptions(options_handle);

  LiteRtDestroyOpaqueOptions(opaque_options);
  return result;
}

TEST(LiteRtQualcommOptionsTest, CreateAndGet) {
  EXPECT_NE(LrtCreateQualcommOptions(nullptr), kLiteRtStatusOk);

  LrtQualcommOptions options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&options));

  LiteRtOpaqueOptions opaque_options;
  const char* id1;
  void* pl1;
  void (*pl_del1)(void*);
  LITERT_ASSERT_OK(
      LrtGetOpaqueQualcommOptionsData(options, &id1, &pl1, &pl_del1));
  LITERT_ASSERT_OK(
      LiteRtCreateOpaqueOptions(id1, pl1, pl_del1, &opaque_options));

  const char* id;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsIdentifier(opaque_options, &id));
  EXPECT_STREQ(id, "qualcomm");

  LiteRtDestroyOpaqueOptions(opaque_options);
  LrtDestroyQualcommOptions(options);
}

TEST(LiteRtQualcommOptionsTest, LogLevel) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetLogLevel(qualcomm_options,
                                                 kLiteRtQualcommLogLevelWarn));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetLogLevel(), QualcommOptions::LogLevel::kWarn);

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, UseInt64BiasAsInt32) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(
      LrtQualcommOptionsSetUseInt64BiasAsInt32(qualcomm_options, false));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_FALSE(parsed.GetUseInt64BiasAsInt32());

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, Backend) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetBackend(qualcomm_options,
                                                kLiteRtQualcommBackendDsp));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetBackend(), QualcommOptions::Backend::kDsp);

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, EnableWeightSharing) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(
      LrtQualcommOptionsSetEnableWeightSharing(qualcomm_options, false));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_FALSE(parsed.GetEnableWeightSharing());

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, UseConvHMX) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetUseConvHMX(qualcomm_options, false));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_FALSE(parsed.GetUseConvHMX());

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, UseFoldReLU) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetUseFoldReLU(qualcomm_options, false));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_FALSE(parsed.GetUseFoldReLU());

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, HtpPerformanceMode) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetHtpPerformanceMode(
      qualcomm_options, kLiteRtQualcommHtpPerformanceModeBurst));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetHtpPerformanceMode(),
            QualcommOptions::HtpPerformanceMode::kBurst);

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, DspPerformanceMode) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetDspPerformanceMode(
      qualcomm_options, kLiteRtQualcommDspPerformanceModeBurst));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetDspPerformanceMode(),
            QualcommOptions::DspPerformanceMode::kBurst);

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, IrJsonDir) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(
      LrtQualcommOptionsSetIrJsonDir(qualcomm_options, "test_dir"));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetIrJsonDir(), "test_dir");

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, DlcDir) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetDlcDir(qualcomm_options, "test_dir"));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetDlcDir(), "test_dir");

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, VtcmSize) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetVtcmSize(qualcomm_options, 4));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetVtcmSize(), 4);

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, NumHvxThreads) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetNumHvxThreads(qualcomm_options, 4));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetNumHvxThreads(), 4);

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, OptimizationLevel) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetOptimizationLevel(
      qualcomm_options, kHtpOptimizeForPrepare));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetOptimizationLevel(),
            QualcommOptions::OptimizationLevel::kOptimizeForPrepare);

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, GraphPriority) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetGraphPriority(
      qualcomm_options, kLiteRTQualcommGraphPriorityHigh));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetGraphPriority(), QualcommOptions::GraphPriority::kHigh);

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, SaverOutputDir) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(
      LrtQualcommOptionsSetSaverOutputDir(qualcomm_options, "test_dir"));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetSaverOutputDir(), "test_dir");

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, Profiling) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  LITERT_ASSERT_OK(LrtQualcommOptionsSetProfiling(
      qualcomm_options, kLiteRtQualcommProfilingDetailed));

  auto parsed = SerializeAndParse(qualcomm_options);
  EXPECT_EQ(parsed.GetProfiling(), QualcommOptions::Profiling::kDetailed);

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(LiteRtQualcommOptionsTest, DumpTensorIds) {
  LrtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LrtCreateQualcommOptions(&qualcomm_options));

  const std::int32_t test_ids[] = {1, 2, 3};
  LITERT_ASSERT_OK(
      LrtQualcommOptionsSetDumpTensorIds(qualcomm_options, test_ids, 3));

  auto parsed = SerializeAndParse(qualcomm_options);
  for (size_t i = 0; i < 3; i++) {
    EXPECT_EQ(parsed.GetDumpTensorIds()[i], test_ids[i]);
  }

  LrtDestroyQualcommOptions(qualcomm_options);
}

TEST(QualcommOptionsTest, CppWrapper) {
  auto options = QualcommOptions::Create();
  ASSERT_TRUE(options);

  options->SetLogLevel(QualcommOptions::LogLevel::kWarn);
  EXPECT_EQ(options->GetLogLevel(), QualcommOptions::LogLevel::kWarn);

  EXPECT_FALSE(options->GetUseHtpPreference());
  options->SetUseHtpPreference(true);
  EXPECT_FALSE(options->GetUseHtpPreference());

  EXPECT_FALSE(options->GetUseQint16AsQuint16());
  options->SetUseQint16AsQuint16(true);
  EXPECT_FALSE(options->GetUseQint16AsQuint16());

  EXPECT_TRUE(options->GetUseInt64BiasAsInt32());
  options->SetUseInt64BiasAsInt32(false);
  EXPECT_FALSE(options->GetUseInt64BiasAsInt32());

  EXPECT_EQ(options->GetHtpPerformanceMode(),
            QualcommOptions::HtpPerformanceMode::kDefault);
  options->SetHtpPerformanceMode(QualcommOptions::HtpPerformanceMode::kBurst);
  EXPECT_EQ(options->GetHtpPerformanceMode(),
            QualcommOptions::HtpPerformanceMode::kBurst);

  EXPECT_EQ(options->GetDspPerformanceMode(),
            QualcommOptions::DspPerformanceMode::kDefault);
  options->SetDspPerformanceMode(QualcommOptions::DspPerformanceMode::kBurst);
  EXPECT_EQ(options->GetDspPerformanceMode(),
            QualcommOptions::DspPerformanceMode::kBurst);

  EXPECT_EQ(options->GetProfiling(), QualcommOptions::Profiling::kOff);
  options->SetProfiling(QualcommOptions::Profiling::kDetailed);
  EXPECT_EQ(options->GetProfiling(), QualcommOptions::Profiling::kDetailed);

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

  EXPECT_EQ(options->GetDlcDir(), "");
  options->SetDlcDir("tmp");
  EXPECT_EQ(options->GetDlcDir(), "tmp");

  EXPECT_EQ(options->GetVtcmSize(), 0);
  options->SetVtcmSize(4);
  EXPECT_EQ(options->GetVtcmSize(), 4);

  EXPECT_EQ(options->GetNumHvxThreads(), 0);
  options->SetNumHvxThreads(4);
  EXPECT_EQ(options->GetNumHvxThreads(), 4);

  EXPECT_EQ(options->GetOptimizationLevel(),
            QualcommOptions::OptimizationLevel::kOptimizeForInferenceO3);
  options->SetOptimizationLevel(
      QualcommOptions::OptimizationLevel::kOptimizeForPrepare);
  EXPECT_EQ(options->GetOptimizationLevel(),
            QualcommOptions::OptimizationLevel::kOptimizeForPrepare);

  EXPECT_EQ(options->GetGraphPriority(),
            QualcommOptions::GraphPriority::kDefault);
  options->SetGraphPriority(QualcommOptions::GraphPriority::kHigh);
  EXPECT_EQ(options->GetGraphPriority(), QualcommOptions::GraphPriority::kHigh);

  EXPECT_TRUE(options->GetUseConvHMX());
  options->SetUseConvHMX(false);
  EXPECT_FALSE(options->GetUseConvHMX());

  EXPECT_TRUE(options->GetUseFoldReLU());
  options->SetUseFoldReLU(false);
  EXPECT_FALSE(options->GetUseFoldReLU());

  EXPECT_EQ(options->GetBackend(), QualcommOptions::Backend::kHtp);
  options->SetBackend(QualcommOptions::Backend::kDsp);
  EXPECT_EQ(options->GetBackend(), QualcommOptions::Backend::kDsp);

  EXPECT_EQ(options->GetSaverOutputDir(), "");
  options->SetSaverOutputDir("tmp");
  EXPECT_EQ(options->GetSaverOutputDir(), "tmp");
}

}  // namespace
}  // namespace litert::qualcomm
