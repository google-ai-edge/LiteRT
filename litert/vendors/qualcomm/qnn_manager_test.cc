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

#include "litert/vendors/qualcomm/qnn_manager.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/tools/dump.h"

namespace {

using ::litert::qnn::QnnManager;
using ::litert::qnn::internal::Dump;
using ::testing::HasSubstr;

// NOTE: This tests that all of the dynamic loading works properly and
// the QNN SDK instance can be properly initialized and destroyed.
auto CreateQnnManager(const ::qnn::Options& options) {
#if defined(__x86_64__) || defined(_M_X64)
  return QnnManager::Create(options, {}, ::qnn::kSocInfos[8]);
#else
  return QnnManager::Create(options);
#endif
}

TEST(QnnManagerTest, SetupQnnManager) {
  auto qnn = CreateQnnManager(::qnn::Options());
  ASSERT_TRUE(qnn);
}

TEST(QnnManagerTest, Dump) {
  auto qnn = CreateQnnManager(::qnn::Options());
  ASSERT_TRUE(qnn);

  auto dump = Dump(**qnn);

  EXPECT_THAT(dump, HasSubstr("< QnnInterface_t >"));
  EXPECT_THAT(dump, HasSubstr("< QnnSystemInterface_t >"));
}

TEST(QnnManagerTest, GetOptions) {
  auto options = ::qnn::Options();
  auto qnn = CreateQnnManager(options);
  ASSERT_TRUE(qnn);

  const auto& options_ref = (*qnn)->GetOptions();
  EXPECT_EQ(options.GetLogLevel(), options_ref.GetLogLevel());
  EXPECT_EQ(options.GetProfiling(), options_ref.GetProfiling());
  EXPECT_EQ(options.GetUseHtpPreference(), options_ref.GetUseHtpPreference());
  EXPECT_EQ(options.GetUseQint16AsQuint16(),
            options_ref.GetUseQint16AsQuint16());
  EXPECT_EQ(options.GetEnableWeightSharing(),
            options_ref.GetEnableWeightSharing());
  EXPECT_EQ(options.GetHtpPerformanceMode(),
            options_ref.GetHtpPerformanceMode());
  EXPECT_EQ(options.GetDspPerformanceMode(),
            options_ref.GetDspPerformanceMode());
  EXPECT_EQ(options.GetDumpTensorIds(), options_ref.GetDumpTensorIds());
  EXPECT_EQ(options.GetIrJsonDir(), options_ref.GetIrJsonDir());
  EXPECT_EQ(options.GetDlcDir(), options_ref.GetDlcDir());
}

}  // namespace
