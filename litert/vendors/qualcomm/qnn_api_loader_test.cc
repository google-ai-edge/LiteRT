// Copyright 2024 Google LLC.
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

#include "litert/vendors/qualcomm/qnn_api_loader.h"

#include <cstdlib>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/utils/test_utils.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "litert/vendors/qualcomm/qnn_sdk_version.h"
#include "litert/vendors/qualcomm/tools/dump.h"

namespace {

using ::litert::qnn::QnnApiLoader;
using ::litert::qnn::QnnManager;
using ::litert::qnn::QnnManagerMode;
using ::litert::qnn::SdkVersion;
using ::litert::qnn::internal::Dump;
using ::testing::HasSubstr;

// NOTE: This tests that all of the dynamic loading works properly and
// the QNN SDK instance can be properly initialized and destroyed.
auto CreateLoader(const ::qnn::Options& options) {
  return QnnApiLoader::Create(options);
}

// Helper to get options based on target
std::optional<::qnn::Options> GetOptionsForTarget() {
  auto options = ::qnn::Options();
  if (::qnn::IsTestHtpBackend()) {
    options.SetBackendType(::qnn::BackendType::kHtpBackend);
    return options;
  }
  if (::qnn::IsTestDspBackend()) {
    options.SetBackendType(::qnn::BackendType::kDspBackend);
    return options;
  }
  if (::qnn::IsTestGpuBackend()) {
    options.SetBackendType(::qnn::BackendType::kGpuBackend);
    return options;
  }
  return std::nullopt;
}

TEST(QnnApiLoaderTest, LoadsLibraries) {
  auto options = GetOptionsForTarget();
  if (!options) {
    GTEST_SKIP() << "Skipping test because targeted backend is not supported";
  }

  auto qnn = CreateLoader(*options);
  ASSERT_TRUE(qnn);
}

TEST(QnnApiLoaderTest, Dump) {
  auto options = GetOptionsForTarget();
  if (!options) {
    GTEST_SKIP() << "Skipping test because targeted backend is not supported";
  }

  auto qnn = CreateLoader(*options);
  ASSERT_TRUE(qnn);

  auto dump = Dump(**qnn);

  EXPECT_THAT(dump, HasSubstr("< QnnInterface_t >"));
  EXPECT_THAT(dump, HasSubstr("< QnnSystemInterface_t >"));
}

TEST(QnnApiLoaderTest, GetOptions) {
  auto options = GetOptionsForTarget();
  if (!options) {
    GTEST_SKIP() << "Skipping test because targeted backend is not supported";
  }

  auto qnn = CreateLoader(*options);
  ASSERT_TRUE(qnn);

  const auto& options_ref = (*qnn)->GetOptions();
  EXPECT_EQ(options->GetLogLevel(), options_ref.GetLogLevel());
  EXPECT_EQ(options->GetProfiling(), options_ref.GetProfiling());
  EXPECT_EQ(options->GetEnableWeightSharing(),
            options_ref.GetEnableWeightSharing());
  EXPECT_EQ(options->GetHtpPerformanceMode(),
            options_ref.GetHtpPerformanceMode());
  EXPECT_EQ(options->GetDspPerformanceMode(),
            options_ref.GetDspPerformanceMode());
  EXPECT_EQ(options->GetDumpTensorIds(), options_ref.GetDumpTensorIds());
  EXPECT_EQ(options->GetIrJsonDir(), options_ref.GetIrJsonDir());
  EXPECT_EQ(options->GetDlcDir(), options_ref.GetDlcDir());
}

TEST(QnnApiLoaderTest, GetSdkVersion) {
  auto options = GetOptionsForTarget();
  if (!options) {
    GTEST_SKIP() << "Skipping test because targeted backend is not supported";
  }

  auto qnn = CreateLoader(*options);
  ASSERT_TRUE(qnn);
  const auto sdk_version = qnn.Value().get()->GetSdkVersion();
  static constexpr SdkVersion kInitSdkVersion{0, 0, 0};
  EXPECT_NE(sdk_version, kInitSdkVersion);
}

// QnnManager::Create() returns a ready QnnManager, and binding a second,
// different SoC off the same loader also returns a ready QnnManager (correct
// SoC) with no library reload.
TEST(QnnApiLoaderTest, QnnManagerRebindsAcrossSocs) {
  auto options = GetOptionsForTarget();
  if (!options || !::qnn::IsTestHtpBackend()) {
    GTEST_SKIP() << "QnnManager SoC-rebind test requires the HTP backend";
  }

  // Library-only manager (no SoC bound yet).
  auto qnn = QnnApiLoader::Create(*options);
  ASSERT_TRUE(qnn);

  const auto soc_a = ::qnn::FindSocModel("SM8650");
  const auto soc_b = ::qnn::FindSocModel("SM8750");
  ASSERT_TRUE(soc_a.has_value());
  ASSERT_TRUE(soc_b.has_value());

  auto qnn_manager_a =
      QnnManager::Create(**qnn, soc_a, QnnManagerMode::kCompile);
  ASSERT_TRUE(qnn_manager_a);
  EXPECT_NE(qnn_manager_a->BackendHandle(), nullptr);
  EXPECT_EQ(qnn_manager_a->GetSocInfo().soc_model, soc_a->soc_model);

  auto qnn_manager_b =
      QnnManager::Create(**qnn, soc_b, QnnManagerMode::kCompile);
  ASSERT_TRUE(qnn_manager_b);
  EXPECT_NE(qnn_manager_b->BackendHandle(), nullptr);
  EXPECT_EQ(qnn_manager_b->GetSocInfo().soc_model, soc_b->soc_model);
}

// On device (non-x86), QnnManager::Create() can auto-detect the SoC from the
// running platform, so callers need not pass a SoC. Binding with std::nullopt
// must still return a fully-ready QnnManager (valid handle, resolved SoC). On
// x86 hosts the target device cannot be queried, so a SoC is mandatory there
// and this scenario is skipped.
#if defined(__x86_64__) || defined(_M_X64)
TEST(QnnApiLoaderTest, QnnManagerAutoDetectRequiresDeviceSkippedOnHost) {
  GTEST_SKIP() << "Host build cannot auto-detect a SoC; a SoC is required.";
}
#else
TEST(QnnApiLoaderTest, QnnManagerAutoDetectsSocOnDevice) {
  auto options = GetOptionsForTarget();
  if (!options || !::qnn::IsTestHtpBackend()) {
    GTEST_SKIP() << "QnnManager SoC auto-detect test requires the HTP "
                    "backend";
  }

  // Library-only loader (no SoC bound yet).
  auto qnn = QnnApiLoader::Create(*options);
  ASSERT_TRUE(qnn);

  // No SoC passed -- the QnnManager is expected to resolve it from the device.
  auto qnn_manager =
      QnnManager::Create(**qnn, std::nullopt, QnnManagerMode::kCompile);
  ASSERT_TRUE(qnn_manager);
  EXPECT_NE(qnn_manager->BackendHandle(), nullptr);
  // A successfully-created HTP QnnManager always has a resolved DSP
  // architecture.
  EXPECT_NE(qnn_manager->GetSocInfo().dsp_arch, ::qnn::DspArch::NONE);
}
#endif

struct SdkVersionTest : public ::testing::Test {
  const SdkVersion v1_0_0{1, 0, 0};
  const SdkVersion v1_0_1{1, 0, 1};
  const SdkVersion v1_1_0{1, 1, 0};
  const SdkVersion v2_0_0{2, 0, 0};
};

TEST_F(SdkVersionTest, HandlesEquality) {
  SdkVersion v1_0_0_copy = v1_0_0;
  EXPECT_EQ(v1_0_0, v1_0_0_copy);
  EXPECT_NE(v1_0_0, v1_0_1);
  EXPECT_NE(v1_0_0, v1_1_0);
  EXPECT_NE(v1_0_0, v2_0_0);

  EXPECT_TRUE(v1_0_0 == v1_0_0_copy);
  EXPECT_FALSE(v1_0_0 == v1_0_1);

  EXPECT_TRUE(v1_0_0 != v1_0_1);
  EXPECT_FALSE(v1_0_0 != v1_0_0_copy);
}

TEST_F(SdkVersionTest, HandlesLessThan) {
  EXPECT_LT(v1_0_0, v1_0_1);
  EXPECT_LT(v1_0_1, v1_1_0);
  EXPECT_LT(v1_1_0, v2_0_0);
  EXPECT_FALSE(v1_0_0 < v1_0_0);
  EXPECT_FALSE(v1_0_1 < v1_0_0);
}

TEST_F(SdkVersionTest, HandlesGreaterThan) {
  EXPECT_GT(v1_0_1, v1_0_0);
  EXPECT_GT(v1_1_0, v1_0_1);
  EXPECT_GT(v2_0_0, v1_1_0);
  EXPECT_FALSE(v1_0_0 > v1_0_0);
  EXPECT_FALSE(v1_0_0 > v1_0_1);
}

TEST_F(SdkVersionTest, HandlesLessThanOrEqual) {
  SdkVersion v1_0_0_copy = v1_0_0;
  EXPECT_LE(v1_0_0, v1_0_0_copy);
  EXPECT_LE(v1_0_0, v1_0_1);
  EXPECT_LE(v1_0_1, v1_1_0);
  EXPECT_LE(v1_1_0, v2_0_0);
  EXPECT_FALSE(v1_0_1 <= v1_0_0);
}

TEST_F(SdkVersionTest, HandlesGreaterThanOrEqual) {
  SdkVersion v1_0_0_copy = v1_0_0;
  EXPECT_GE(v1_0_0, v1_0_0_copy);
  EXPECT_GE(v1_0_1, v1_0_0);
  EXPECT_GE(v1_1_0, v1_0_1);
  EXPECT_GE(v2_0_0, v1_1_0);
  EXPECT_FALSE(v1_0_0 >= v1_0_1);
}

TEST(QnnApiLoaderTest, AdspLibraryPathNoDuplicate) {
  static constexpr char kAdsp[] = "ADSP_LIBRARY_PATH";
  const char* original_adsp_ptr = getenv(kAdsp);
  std::optional<std::string> original_adsp;
  if (original_adsp_ptr) {
    original_adsp = original_adsp_ptr;
  }

  // Set a known value
  setenv(kAdsp, "/my/path", /*overwrite=*/1);

  auto options = ::qnn::Options();

  // This will fail to load libraries but should update env var.
  auto qnn = QnnApiLoader::Create(options, "/my/path");

  // Verify that it didn't duplicate
  const char* new_adsp = getenv(kAdsp);
  EXPECT_STREQ(new_adsp, "/my/path");

  // Now try to add a new one
  auto qnn2 = QnnApiLoader::Create(options, "/another/path");
  new_adsp = getenv(kAdsp);
  EXPECT_STREQ(new_adsp, "/another/path;/my/path");

  // Restore original value
  if (original_adsp) {
    setenv(kAdsp, original_adsp->c_str(), /*overwrite=*/1);
  } else {
    unsetenv(kAdsp);
  }
}

}  // namespace
