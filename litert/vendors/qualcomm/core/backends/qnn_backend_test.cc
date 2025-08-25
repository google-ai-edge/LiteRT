// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/htp_perf_control.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {

class QnnBackendTest : public testing::TestWithParam<BackendType> {
 public:
  void SetUp() {
    // empty string for lib path means use default
    switch (GetParam()) {
      case BackendType::kHtpBackend:
        handle = ::qnn::CreateDLHandle(::qnn::HtpBackend::GetLibraryName());
        backend = std::make_unique<::qnn::HtpBackend>(::qnn::ResolveQnnApi(
            handle.get(), ::qnn::HtpBackend::GetExpectedBackendVersion()));

        break;
      case BackendType::kIrBackend:
        handle = ::qnn::CreateDLHandle(::qnn::IrBackend::GetLibraryName());
        backend = std::make_unique<::qnn::IrBackend>(::qnn::ResolveQnnApi(
            handle.get(), ::qnn::IrBackend::GetExpectedBackendVersion()));

        break;
      default:
        break;
    }
  }

  DLHandle handle;
  std::unique_ptr<QnnBackend> backend;
};

INSTANTIATE_TEST_SUITE_P(QnnBackendTest, QnnBackendTest,
                         testing::Values(BackendType::kHtpBackend,
                                         BackendType::kIrBackend));

// TODO: Enable dlopen related tests after libs added to open source env
TEST_P(QnnBackendTest, DISABLED_InitializeWithLogLevelOffTest) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);

  switch (GetParam()) {
    case BackendType::kHtpBackend:
      // Use V75 to test with HTP backend
      ASSERT_TRUE(backend->Init(options, ::qnn::kSocInfos[7]));
      ASSERT_NE(backend->GetDeviceHandle(), nullptr);
      break;
    case BackendType::kIrBackend:
      options.SetBackendType(BackendType::kIrBackend);
      ASSERT_TRUE(backend->Init(options, std::nullopt));
      ASSERT_EQ(backend->GetDeviceHandle(), nullptr);
      break;
    default:
      break;
  }

  ASSERT_NE(backend->GetBackendHandle(), nullptr);
  ASSERT_EQ(backend->GetLogHandle(), nullptr);
}

TEST_P(QnnBackendTest, DISABLED_InitializeWithLogLevelVerboseTest) {
  Options options;
  options.SetLogLevel(LogLevel::kVerbose);

  switch (GetParam()) {
    case BackendType::kHtpBackend:
      // Use V75 to test with HTP backend
      ASSERT_TRUE(backend->Init(options, ::qnn::kSocInfos[7]));
      ASSERT_NE(backend->GetDeviceHandle(), nullptr);
      break;
    case BackendType::kIrBackend:
      options.SetBackendType(BackendType::kIrBackend);
      ASSERT_TRUE(backend->Init(options, std::nullopt));
      ASSERT_EQ(backend->GetDeviceHandle(), nullptr);
      break;
    default:
      break;
  }

  ASSERT_NE(backend->GetBackendHandle(), nullptr);
  ASSERT_NE(backend->GetLogHandle(), nullptr);
}

}  // namespace
}  // namespace qnn
