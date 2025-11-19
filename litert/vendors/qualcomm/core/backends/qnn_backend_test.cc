// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"

#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
namespace {

class QnnBackendTest : public testing::TestWithParam<BackendType> {
 public:
  void SetUp() override {
    // empty string for lib path means use default
    switch (GetParam()) {
      case BackendType::kHtpBackend:
        handle_ = ::qnn::CreateDLHandle(::qnn::HtpBackend::GetLibraryName());
        backend_ = std::make_unique<::qnn::HtpBackend>(::qnn::ResolveQnnApi(
            handle_.get(), ::qnn::HtpBackend::GetExpectedBackendVersion()));

        break;
      case BackendType::kIrBackend:
        handle_ = ::qnn::CreateDLHandle(::qnn::IrBackend::GetLibraryName());
        backend_ = std::make_unique<::qnn::IrBackend>(::qnn::ResolveQnnApi(
            handle_.get(), ::qnn::IrBackend::GetExpectedBackendVersion()));

        break;
      default:
        break;
    }
  }

  DLHandle handle_;
  std::unique_ptr<QnnBackend> backend_;
};

INSTANTIATE_TEST_SUITE_P(QnnBackendTest, QnnBackendTest,
                         testing::Values(BackendType::kHtpBackend,
                                         BackendType::kIrBackend));

TEST_P(QnnBackendTest, DISABLED_InitializeWithLogLevelOffTest) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);

  switch (GetParam()) {
    case BackendType::kHtpBackend:
      // Use V75 to test with HTP backend
      ASSERT_TRUE(backend_->Init(options, ::qnn::kSocInfos[7]));
      ASSERT_NE(backend_->GetDeviceHandle(), nullptr);
      break;
    case BackendType::kIrBackend:
      options.SetBackendType(BackendType::kIrBackend);
      ASSERT_TRUE(backend_->Init(options, std::nullopt));
      ASSERT_EQ(backend_->GetDeviceHandle(), nullptr);
      break;
    default:
      break;
  }

  ASSERT_NE(backend_->GetBackendHandle(), nullptr);
  ASSERT_EQ(backend_->GetLogHandle(), nullptr);
}

TEST_P(QnnBackendTest, DISABLED_InitializeWithLogLevelVerboseTest) {
  Options options;
  options.SetLogLevel(LogLevel::kVerbose);

  switch (GetParam()) {
    case BackendType::kHtpBackend:
      // Use V75 to test with HTP backend
      ASSERT_TRUE(backend_->Init(options, ::qnn::kSocInfos[7]));
      ASSERT_NE(backend_->GetDeviceHandle(), nullptr);
      break;
    case BackendType::kIrBackend:
      options.SetBackendType(BackendType::kIrBackend);
      ASSERT_TRUE(backend_->Init(options, std::nullopt));
      ASSERT_EQ(backend_->GetDeviceHandle(), nullptr);
      break;
    default:
      break;
  }

  ASSERT_NE(backend_->GetBackendHandle(), nullptr);
  ASSERT_NE(backend_->GetLogHandle(), nullptr);
}

}  // namespace
}  // namespace qnn
