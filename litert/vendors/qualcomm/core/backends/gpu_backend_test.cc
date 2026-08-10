// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/gpu_backend.h"

#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
namespace {

constexpr auto kDefaultSocInfo = FindSocInfo("SM8750");
static_assert(kDefaultSocInfo.has_value());

class GpuBackendTest : public testing::Test {
 public:
  void SetUp() override {
    handle_ = CreateDLHandle(GpuBackend::GetLibraryName());
    if (!handle_) GTEST_SKIP();

    auto* qnn_api =
        ResolveQnnApi(handle_.get(), GpuBackend::GetExpectedBackendVersion());
    ASSERT_TRUE(qnn_api);
    backend_ = std::make_unique<GpuBackend>(qnn_api);
  }

  DLHandle handle_;
  std::unique_ptr<GpuBackend> backend_;
};

TEST_F(GpuBackendTest, DISABLED_InitializeWithLogLevelOffTest) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);

  ASSERT_TRUE(backend_->Init(options, kDefaultSocInfo));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_FALSE(backend_->GetLogHandle());
  EXPECT_EQ(backend_->GetSocInfo().soc_model, kDefaultSocInfo->soc_model);
}

TEST_F(GpuBackendTest, DISABLED_InitializeWithLogLevelVerboseTest) {
  Options options;
  options.SetLogLevel(LogLevel::kVerbose);

  ASSERT_TRUE(backend_->Init(options, kDefaultSocInfo));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_TRUE(backend_->GetLogHandle());
  EXPECT_EQ(backend_->GetSocInfo().soc_model, kDefaultSocInfo->soc_model);
}

}  // namespace
}  // namespace qnn
