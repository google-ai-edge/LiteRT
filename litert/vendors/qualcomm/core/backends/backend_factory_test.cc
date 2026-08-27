// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/backend_factory.h"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "HTP/QnnHtpDeviceConfigShared.h"  // from @qairt
#include "QnnBackend.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnLog.h"  // from @qairt
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"
#include "litert/vendors/qualcomm/core/backends/gpu_backend.h"
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
namespace {

constexpr auto kDefaultSocInfo = FindSocInfo("SM8750");
static_assert(kDefaultSocInfo.has_value());

struct RegisterCall {
  Qnn_BackendHandle_t backend = nullptr;
  std::string package_path;
  std::string interface_provider;
  std::string target;
};

RegisterCall& LastRegisterCall() {
  static absl::NoDestructor<RegisterCall> call;
  return *call;
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

  const bool is_custom_op_supported = backend_type == BackendType::kHtpBackend;

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

// SIGNED PROCESS DOMAIN (SignedPD) TESTS //////////////////////////////////////

struct MockDeviceCreateCall {
  std::vector<QnnHtpDevice_CustomConfig_t> custom_configs;
};

struct MockQnnState {
  int device_create_call_count = 0;
  int fail_device_create_on_call =
      0;  // 0: never, 1: fail 1st call only, -1: fail all
  std::vector<MockDeviceCreateCall> device_create_calls;
};

static MockQnnState* g_mock_qnn_state = nullptr;

static char kDummyBackendObj = 0;
static char kDummyDeviceObj = 0;

Qnn_ErrorHandle_t MockQnnLogCreate(QnnLog_Callback_t, QnnLog_Level_t,
                                   Qnn_LogHandle_t* log) {
  *log = reinterpret_cast<Qnn_LogHandle_t>(&kDummyDeviceObj);
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQnnLogFree(Qnn_LogHandle_t) { return QNN_SUCCESS; }

Qnn_ErrorHandle_t MockQnnBackendCreate(Qnn_LogHandle_t,
                                       const QnnBackend_Config_t**,
                                       Qnn_BackendHandle_t* backend) {
  *backend = reinterpret_cast<Qnn_BackendHandle_t>(&kDummyBackendObj);
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQnnBackendFree(Qnn_BackendHandle_t) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQnnDeviceFree(Qnn_DeviceHandle_t) { return QNN_SUCCESS; }

Qnn_ErrorHandle_t MockQnnDeviceCreate(Qnn_LogHandle_t,
                                      const QnnDevice_Config_t** configs,
                                      Qnn_DeviceHandle_t* device) {
  if (!g_mock_qnn_state) {
    return QNN_COMMON_ERROR_GENERAL;
  }
  g_mock_qnn_state->device_create_call_count++;

  MockDeviceCreateCall call_record;
  if (configs) {
    for (size_t i = 0; configs[i] != nullptr; ++i) {
      if (configs[i]->option == QNN_DEVICE_CONFIG_OPTION_CUSTOM &&
          configs[i]->customConfig != nullptr) {
        const auto* htp_cfg = static_cast<const QnnHtpDevice_CustomConfig_t*>(
            configs[i]->customConfig);
        call_record.custom_configs.push_back(*htp_cfg);
      }
    }
  }
  g_mock_qnn_state->device_create_calls.push_back(std::move(call_record));

  if (g_mock_qnn_state->fail_device_create_on_call == -1 ||
      g_mock_qnn_state->fail_device_create_on_call ==
          g_mock_qnn_state->device_create_call_count) {
    return QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE;
  }

  *device = reinterpret_cast<Qnn_DeviceHandle_t>(&kDummyDeviceObj);
  return QNN_SUCCESS;
}

class HtpBackendSignedPdTest : public testing::Test {
 protected:
  void SetUp() override {
    unsetenv("LITERT_QUALCOMM_USE_SIGNED_PD");
    mock_state_ = std::make_unique<MockQnnState>();
    g_mock_qnn_state = mock_state_.get();

    api_ = {};
    api_.logCreate = MockQnnLogCreate;
    api_.logFree = MockQnnLogFree;
    api_.backendCreate = MockQnnBackendCreate;
    api_.backendFree = MockQnnBackendFree;
    api_.deviceCreate = MockQnnDeviceCreate;
    api_.deviceFree = MockQnnDeviceFree;
  }

  void TearDown() override {
    unsetenv("LITERT_QUALCOMM_USE_SIGNED_PD");
    g_mock_qnn_state = nullptr;
    mock_state_.reset();
  }

  QNN_INTERFACE_VER_TYPE api_{};
  std::unique_ptr<MockQnnState> mock_state_;
};

TEST_F(HtpBackendSignedPdTest, Init_DefaultSucceedsWithoutSignedPd) {
  HtpBackend backend(&api_);
  Options options;
  EXPECT_TRUE(backend.Init(options, kDefaultSocInfo));

  ASSERT_EQ(mock_state_->device_create_call_count, 1);
  ASSERT_EQ(mock_state_->device_create_calls.size(), 1);

  const auto& configs = mock_state_->device_create_calls[0].custom_configs;
  ASSERT_EQ(configs.size(), 1);
  EXPECT_EQ(configs[0].option, QNN_HTP_DEVICE_CONFIG_OPTION_SOC);
  EXPECT_EQ(configs[0].socModel, kDefaultSocInfo->soc_model);
}

TEST_F(HtpBackendSignedPdTest, Init_FallbackToSignedPdWhenDefaultFails) {
  mock_state_->fail_device_create_on_call = 1;  // fail 1st attempt
  HtpBackend backend(&api_);
  Options options;
  EXPECT_TRUE(backend.Init(options, kDefaultSocInfo));

  ASSERT_EQ(mock_state_->device_create_call_count, 2);
  ASSERT_EQ(mock_state_->device_create_calls.size(), 2);

  // First call: SOC only
  const auto& first_configs =
      mock_state_->device_create_calls[0].custom_configs;
  ASSERT_EQ(first_configs.size(), 1);
  EXPECT_EQ(first_configs[0].option, QNN_HTP_DEVICE_CONFIG_OPTION_SOC);

  // Second call: SOC + SignedPD
  const auto& second_configs =
      mock_state_->device_create_calls[1].custom_configs;
  ASSERT_EQ(second_configs.size(), 2);
  EXPECT_EQ(second_configs[0].option, QNN_HTP_DEVICE_CONFIG_OPTION_SOC);
  EXPECT_EQ(second_configs[1].option, QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD);
  EXPECT_EQ(second_configs[1].useSignedProcessDomain.deviceId, 0);
  EXPECT_TRUE(second_configs[1].useSignedProcessDomain.useSignedProcessDomain);
}

TEST_F(HtpBackendSignedPdTest, Init_ForceSignedPdViaEnvVarNumeric) {
  setenv("LITERT_QUALCOMM_USE_SIGNED_PD", "1", 1);
  HtpBackend backend(&api_);
  Options options;
  EXPECT_TRUE(backend.Init(options, kDefaultSocInfo));

  ASSERT_EQ(mock_state_->device_create_call_count, 1);
  ASSERT_EQ(mock_state_->device_create_calls.size(), 1);

  const auto& configs = mock_state_->device_create_calls[0].custom_configs;
  ASSERT_EQ(configs.size(), 2);
  EXPECT_EQ(configs[0].option, QNN_HTP_DEVICE_CONFIG_OPTION_SOC);
  EXPECT_EQ(configs[1].option, QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD);
  EXPECT_TRUE(configs[1].useSignedProcessDomain.useSignedProcessDomain);
}

TEST_F(HtpBackendSignedPdTest, Init_ForceSignedPdViaEnvVarString) {
  setenv("LITERT_QUALCOMM_USE_SIGNED_PD", "true", 1);
  HtpBackend backend(&api_);
  Options options;
  EXPECT_TRUE(backend.Init(options, kDefaultSocInfo));

  ASSERT_EQ(mock_state_->device_create_call_count, 1);
  ASSERT_EQ(mock_state_->device_create_calls.size(), 1);

  const auto& configs = mock_state_->device_create_calls[0].custom_configs;
  ASSERT_EQ(configs.size(), 2);
  EXPECT_EQ(configs[0].option, QNN_HTP_DEVICE_CONFIG_OPTION_SOC);
  EXPECT_EQ(configs[1].option, QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD);
  EXPECT_TRUE(configs[1].useSignedProcessDomain.useSignedProcessDomain);
}

TEST_F(HtpBackendSignedPdTest, Init_FailsWhenBothAttemptsFail) {
  mock_state_->fail_device_create_on_call = -1;  // fail all attempts
  HtpBackend backend(&api_);
  Options options;
  EXPECT_FALSE(backend.Init(options, kDefaultSocInfo));

  EXPECT_EQ(mock_state_->device_create_call_count, 2);
}

}  // namespace
}  // namespace qnn
