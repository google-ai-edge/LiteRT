// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
namespace {

class QnnBackendTest : public testing::Test {
 public:
  void SetUp() override {
    // empty string for lib path means use default
    handle_ = ::qnn::CreateDLHandle(::qnn::IrBackend::GetLibraryName());
    backend_ = std::make_unique<::qnn::IrBackend>(::qnn::ResolveQnnApi(
        handle_.get(), ::qnn::IrBackend::GetExpectedBackendVersion()));
  }

  DLHandle handle_;
  std::unique_ptr<QnnBackend> backend_;
};

TEST_F(QnnBackendTest, DISABLED_InitializeWithLogLevelOffTest) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);

  options.SetBackendType(BackendType::kIrBackend);
  ASSERT_TRUE(backend_->Init(options, std::nullopt));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_NE(backend_->GetBackendHandle(), nullptr);
  ASSERT_EQ(backend_->GetLogHandle(), nullptr);
}

TEST_F(QnnBackendTest, DISABLED_InitializeWithLogLevelVerboseTest) {
  Options options;
  options.SetLogLevel(LogLevel::kVerbose);

  options.SetBackendType(BackendType::kIrBackend);
  ASSERT_TRUE(backend_->Init(options, std::nullopt));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_NE(backend_->GetBackendHandle(), nullptr);
  ASSERT_NE(backend_->GetLogHandle(), nullptr);
}

}  // namespace
}  // namespace qnn
