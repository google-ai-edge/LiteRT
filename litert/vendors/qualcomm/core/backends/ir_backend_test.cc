// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/ir_backend.h"

#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
namespace {

class IrBackendTest : public testing::Test {
 public:
  void SetUp() override {
    // empty string for lib path means use default
    handle_ = ::qnn::CreateDLHandle(::qnn::IrBackend::GetLibraryName());
    backend_ = std::make_unique<::qnn::IrBackend>(::qnn::ResolveQnnApi(
        handle_.get(), ::qnn::IrBackend::GetExpectedBackendVersion()));
  }

  DLHandle handle_;
  std::unique_ptr<IrBackend> backend_;
};

TEST_F(IrBackendTest, DISABLED_InitializeWithLogLevelOffTest) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);

  options.SetBackendType(BackendType::kIrBackend);
  ASSERT_TRUE(backend_->Init(options, std::nullopt));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_FALSE(backend_->GetLogHandle());
}

TEST_F(IrBackendTest, DISABLED_InitializeWithLogLevelVerboseTest) {
  Options options;
  options.SetLogLevel(LogLevel::kVerbose);

  options.SetBackendType(BackendType::kIrBackend);
  ASSERT_TRUE(backend_->Init(options, std::nullopt));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_TRUE(backend_->GetLogHandle());
}

}  // namespace
}  // namespace qnn
