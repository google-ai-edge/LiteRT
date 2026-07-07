// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/backend_factory.h"

#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"
#include "litert/vendors/qualcomm/core/backends/gpu_backend.h"
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "QnnCommon.h"  // from @qairt

namespace qnn {
namespace {

const SocInfo kDefaultSocInfo = FindSocModel("SM8750").value_or(kSocInfos[0]);

struct RegisterCall {
  Qnn_BackendHandle_t backend = nullptr;
  std::string package_path;
  std::string interface_provider;
  std::string target;
};

RegisterCall& LastRegisterCall() {
  static RegisterCall call;
  return call;
}

Qnn_ErrorHandle_t MockRegisterOpPackage(Qnn_BackendHandle_t backend,
                                        const char* package_path,
                                        const char* interface_provider,
                                        const char* target) {
  auto& call = LastRegisterCall();
  call.backend = backend;
  call.package_path = package_path ? package_path : "";
  call.interface_provider = interface_provider ? interface_provider : "";
  call.target = target ? target : "";
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockRegisterOpPackageFail(Qnn_BackendHandle_t /*backend*/,
                                            const char* /*package_path*/,
                                            const char* /*interface_provider*/,
                                            const char* /*target*/) {
  return QNN_COMMON_ERROR_NOT_SUPPORTED;
}

template <typename BackendT>
void TestCreateBackend(BackendType backend_type,
                       std::optional<SocInfo> soc_info = kDefaultSocInfo) {
  auto handle = CreateDLHandle(BackendT::GetLibraryName());
  if (!handle) GTEST_SKIP();
  const auto* real =
      ResolveQnnApi(handle.get(), BackendT::GetExpectedBackendVersion());
  ASSERT_TRUE(real);
  auto api = *real;
  api.backendRegisterOpPackage = MockRegisterOpPackage;

  const bool is_custom_op_supported =
        backend_type == BackendType::kHtpBackend;

  // Base create + empty custom-op name skips register.
  {
    LastRegisterCall() = {};
    Options options;
    options.SetBackendType(backend_type);
    auto backend = CreateBackend(&api, options, soc_info,
                                 /*is_compiler=*/true);
    EXPECT_NE(backend.get(), nullptr);
    EXPECT_EQ(LastRegisterCall().backend, nullptr);
  }

  // Shared setup for the custom-op scenarios below.
  Options options;
  options.SetBackendType(backend_type);
  options.SetCustomOpPackage("MyPackage", "MyProvider",
                              "/tmp/compile_package.so",
                              "/tmp/dispatch_package.so", "HTP");

  // Compile path overrides target to CPU.
  {
    LastRegisterCall() = {};
    auto backend = CreateBackend(&api, options, soc_info,
                                 /*is_compiler=*/true);
    ASSERT_NE(backend.get(), nullptr);
    const auto& call = LastRegisterCall();
    if (is_custom_op_supported) {
      EXPECT_NE(call.backend, nullptr);
      EXPECT_EQ(call.package_path, "/tmp/compile_package.so");
      EXPECT_EQ(call.interface_provider, "MyProvider");
      EXPECT_EQ(call.target, "CPU");
    } else {
      EXPECT_EQ(call.backend, nullptr);
    }
  }

  // Dispatch path uses options target.
  {
    LastRegisterCall() = {};
    auto backend = CreateBackend(&api, options, soc_info,
                                 /*is_compiler=*/false);
    ASSERT_NE(backend.get(), nullptr);
    const auto& call = LastRegisterCall();
    if (is_custom_op_supported) {
      EXPECT_NE(call.backend, nullptr);
      EXPECT_EQ(call.package_path, "/tmp/dispatch_package.so");
      EXPECT_EQ(call.interface_provider, "MyProvider");
      EXPECT_EQ(call.target, "HTP");
    } else {
      EXPECT_EQ(call.backend, nullptr);
    }
  }

  // Register failure returns null.
  if (is_custom_op_supported) {
    LastRegisterCall() = {};
    api.backendRegisterOpPackage = MockRegisterOpPackageFail;
    auto backend = CreateBackend(&api, options, soc_info,
                                 /*is_compiler=*/true);
    EXPECT_EQ(backend.get(), nullptr);
    EXPECT_EQ(LastRegisterCall().backend, nullptr);
  }
}

TEST(CreateBackendTest, CreateReturnsNullForUnsupportedBackend) {
  Options options;
  options.SetBackendType(BackendType::kUndefinedBackend);

  auto backend = CreateBackend(nullptr, options, kDefaultSocInfo,
                               /*is_compiler=*/true);
  EXPECT_EQ(backend.get(), nullptr);
}

TEST(CreateBackendTest, DISABLED_CreateGpuBackend) {
  TestCreateBackend<GpuBackend>(BackendType::kGpuBackend);
}

TEST(CreateBackendTest, DISABLED_CreateHtpBackend) {
#if defined(__x86_64__) || defined(_M_X64)
  TestCreateBackend<HtpBackend>(BackendType::kHtpBackend);
#else
  TestCreateBackend<HtpBackend>(BackendType::kHtpBackend, std::nullopt);
#endif
}

TEST(CreateBackendTest, DISABLED_CreateIrBackend) {
  TestCreateBackend<IrBackend>(BackendType::kIrBackend);
}

TEST(CreateBackendTest, DISABLED_CreateDspBackend) {
  TestCreateBackend<DspBackend>(BackendType::kDspBackend);
}

}  // namespace
}  // namespace qnn
