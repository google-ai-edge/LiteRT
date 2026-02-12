// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/gpu_backend.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include "QnnCommon.h"
#include "QnnInterface.h"
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/backends/backend_utils.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "QnnDevice.h"  // from @qairt

namespace qnn {
namespace {
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

  ASSERT_TRUE(backend_->Init(options, std::nullopt));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_FALSE(backend_->GetLogHandle());
}

TEST_F(GpuBackendTest, DISABLED_InitializeWithLogLevelVerboseTest) {
  Options options;
  options.SetLogLevel(LogLevel::kVerbose);

  ASSERT_TRUE(backend_->Init(options, std::nullopt));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_TRUE(backend_->GetLogHandle());
}

}  // namespace
}  // namespace qnn
