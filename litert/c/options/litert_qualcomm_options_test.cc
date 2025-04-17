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

#include <cstdlib>
#include <utility>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_opaque_options.h"
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

TEST(LiteRtQualcommOptionsTest, WeightSharing) {
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

TEST(LiteRtQualcommOptionsTest, PowerMode) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsCreate(&options));

  LiteRtQualcommOptions qualcomm_options;
  LITERT_ASSERT_OK(LiteRtQualcommOptionsGet(options, &qualcomm_options));

  LITERT_ASSERT_OK(LiteRtQualcommOptionsSetPowerMode(
      qualcomm_options, kLiteRtQualcommPowerModePowerSaver));

  LiteRtQualcommOptionsPowerMode power_mode;
  LITERT_ASSERT_OK(
      LiteRtQualcommOptionsGetPowerMode(qualcomm_options, &power_mode));
  EXPECT_EQ(power_mode, kLiteRtQualcommPowerModePowerSaver);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(QualcommOptionsTest, CppApi) {
  auto options = QualcommOptions::Create();
  ASSERT_TRUE(options);

  EXPECT_EQ(options->GetLogLevel(), kLiteRtQualcommLogLevelInfo);
  options->SetLogLevel(kLiteRtQualcommLogLevelWarn);
  EXPECT_EQ(options->GetLogLevel(), kLiteRtQualcommLogLevelWarn);

  EXPECT_TRUE(options->GetEnableWeightSharing());
  options->SetEnableWeightSharing(false);
  EXPECT_FALSE(options->GetEnableWeightSharing());

  EXPECT_EQ(options->GetPowerMode(), kLiteRtQualcommPowerModePerformance);
  options->SetPowerMode(kLiteRtQualcommPowerModePowerSaver);
  EXPECT_EQ(options->GetPowerMode(), kLiteRtQualcommPowerModePowerSaver);
}

TEST(QualcommOptionsTest, FindFromChain) {
  void* payload = malloc(8);
  auto options =
      OpaqueOptions::Create("not-qualcomm", payload, [](void* d) { free(d); });
  ASSERT_TRUE(options);
  auto qnn_options = QualcommOptions::Create();
  ASSERT_TRUE(qnn_options);
  options->Append(std::move(*qnn_options));

  auto qnn_opts_d = Find<QualcommOptions>(*options);
  EXPECT_TRUE(qnn_opts_d);
}

}  // namespace
}  // namespace litert::qualcomm
