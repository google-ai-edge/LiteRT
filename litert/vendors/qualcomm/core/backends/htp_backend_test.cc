// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/htp_backend.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnCommon.h"
#include "QnnInterface.h"
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/backend_utils.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "QnnDevice.h"  // from @qairt

namespace qnn {
namespace {

// HTP PERF CONTROL /////////////////////////////////////////////////////////
QnnDevice_GetInfrastructureFn_t real_device_get_infrastructure = nullptr;
QnnDevice_GetPlatformInfoFn_t real_device_get_platform_info = nullptr;
QnnDevice_FreeFn_t real_device_free = nullptr;
absl::NoDestructor<
    std::vector<std::vector<QnnHtpPerfInfrastructure_PowerConfig_t>>>
    captured_configs;

// Test Platform Info structs
QnnHtpDevice_DeviceInfoExtension_t test_htp_device_info_extension = {
    QNN_HTP_DEVICE_TYPE_ON_CHIP,
    {{0 /*vtcmSize*/, 87 /*socModel*/, false /*signedPdSupport*/,
      false /*dlbcSupport*/, QNN_HTP_DEVICE_ARCH_V81 /*arch*/}}};

QnnDevice_CoreInfo_t test_device_core_info = {
    QNN_DEVICE_CORE_INFO_VERSION_1, {{0 /*coreId*/, 0 /*coreType*/, nullptr}}};

QnnDevice_HardwareDeviceInfo_t test_device_hardware_device_info = {
    QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1,
    {{0 /*deviceId*/, 0 /*deviceType*/, 1 /*numCores*/, &test_device_core_info,
      (QnnDevice_DeviceInfoExtension_t)&test_htp_device_info_extension}}};

constexpr QnnDevice_PlatformInfo_t kTestDevicePlatformInfo = {
    QNN_DEVICE_PLATFORM_INFO_VERSION_1,
    {{1 /*numHwDevices*/, &test_device_hardware_device_info}}};

// Mock Functions
Qnn_ErrorHandle_t MockSetPowerConfig(
    uint32_t power_config_id,
    const QnnHtpPerfInfrastructure_PowerConfig_t** power_configs) {
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> local_configs;
  if (power_configs) {
    for (size_t i = 0; power_configs[i] != nullptr; ++i) {
      local_configs.emplace_back(*power_configs[i]);
    }
  }
  captured_configs->emplace_back(std::move(local_configs));
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockDeviceGetInfrastructure(
    const QnnDevice_Infrastructure_t* infra) {
  if (!real_device_get_infrastructure) {
    return QNN_COMMON_ERROR_GENERAL;
  }

  auto res = real_device_get_infrastructure(infra);
  if (res != QNN_SUCCESS) {
    return res;
  }

  if ((*infra)->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
    (*infra)->perfInfra.setPowerConfig = MockSetPowerConfig;
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockDeviceGetPlatformInfo(
    Qnn_LogHandle_t logger, const QnnDevice_PlatformInfo_t** platformInfo) {
  *platformInfo = &kTestDevicePlatformInfo;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockDeviceFree(void* handle) {
  if (handle == &kTestDevicePlatformInfo) {
    return QNN_SUCCESS;
  }
  return QNN_COMMON_ERROR_GENERAL;
}

struct HtpPerfParams {
  HtpPerformanceMode mode;
  uint32_t sleep_latency;
  uint32_t dcvs_enable;
  QnnHtpPerfInfrastructure_PowerMode_t power_mode;
  QnnHtpPerfInfrastructure_VoltageCorner_t voltage_corner;
};

class HtpBackendPerfBaseTest : public testing::TestWithParam<HtpPerfParams> {
 public:
  void SetUp() override {
    // Clean the previous captured configs.
    captured_configs->clear();

    handle_ = CreateDLHandle(HtpBackend::GetLibraryName());
    if (!handle_) GTEST_SKIP();

    auto* qnn_api =
        ResolveQnnApi(handle_.get(), HtpBackend::GetExpectedBackendVersion());
    ASSERT_TRUE(qnn_api);

    qnn_api_copy_ = *qnn_api;

    real_device_get_infrastructure = qnn_api_copy_.deviceGetInfrastructure;
    qnn_api_copy_.deviceGetInfrastructure = MockDeviceGetInfrastructure;

    real_device_get_platform_info = qnn_api_copy_.deviceGetPlatformInfo;
    qnn_api_copy_.deviceGetPlatformInfo = MockDeviceGetPlatformInfo;

    real_device_free = qnn_api_copy_.deviceFree;
    qnn_api_copy_.deviceFree = MockDeviceFree;
  }

  void TearDown() override {
    real_device_get_infrastructure = nullptr;
    real_device_get_platform_info = nullptr;
    real_device_free = nullptr;
  }

 protected:
  DLHandle handle_;
  QNN_INTERFACE_VER_TYPE qnn_api_copy_{};
};

class HtpBackendRPCPollingPerfParamTest : public HtpBackendPerfBaseTest {};

TEST_P(HtpBackendRPCPollingPerfParamTest, InitWithPerfMode) {
  const auto& params = GetParam();

  Options options;
  options.SetHtpPerformanceMode(params.mode);
  HtpBackend backend(&qnn_api_copy_);

#if defined(__x86_64__) || defined(_M_X64)
  EXPECT_TRUE(backend.Init(options, kSocInfos[8]));
#else
  EXPECT_TRUE(backend.Init(options, std::nullopt));
#endif

  const auto& configs = *captured_configs;
  ASSERT_EQ(configs.size(), 2);

  const auto& rpc_configs = configs[0];
  ASSERT_EQ(rpc_configs.size(), 2);
  EXPECT_EQ(rpc_configs[0].option,
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME);
  EXPECT_EQ(rpc_configs[0].rpcPollingTimeConfig,
            PowerConfig::kRpcPollingTimeHighPower);

  EXPECT_EQ(rpc_configs[1].option,
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY);
  EXPECT_EQ(rpc_configs[1].rpcControlLatencyConfig, 0);

  const auto& dcvs_configs = configs[1];
  ASSERT_EQ(dcvs_configs.size(), 1);
  EXPECT_EQ(dcvs_configs[0].option,
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3);

  const auto& dcvs = dcvs_configs[0].dcvsV3Config;
  EXPECT_EQ(dcvs.sleepLatency, params.sleep_latency);
  EXPECT_EQ(dcvs.dcvsEnable, params.dcvs_enable);
  EXPECT_EQ(dcvs.powerMode, params.power_mode);
  EXPECT_EQ(dcvs.busVoltageCornerMin, params.voltage_corner);
  EXPECT_EQ(dcvs.busVoltageCornerTarget, params.voltage_corner);
  EXPECT_EQ(dcvs.busVoltageCornerMax, params.voltage_corner);
  EXPECT_EQ(dcvs.coreVoltageCornerMin, params.voltage_corner);
  EXPECT_EQ(dcvs.coreVoltageCornerTarget, params.voltage_corner);
  EXPECT_EQ(dcvs.coreVoltageCornerMax, params.voltage_corner);
}

INSTANTIATE_TEST_SUITE_P(
    HtpBackendRPCPollingTests, HtpBackendRPCPollingPerfParamTest,
    testing::Values(
        HtpPerfParams{HtpPerformanceMode::kBurst, PowerConfig::kSleepMinLatency,
                      PowerConfig::kDcvsDisable,
                      QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
                      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER},
        HtpPerfParams{HtpPerformanceMode::kSustainedHighPerformance,
                      PowerConfig::kSleepLowLatency, PowerConfig::kDcvsDisable,
                      QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
                      DCVS_VOLTAGE_VCORNER_TURBO},
        HtpPerfParams{HtpPerformanceMode::kHighPerformance,
                      PowerConfig::kSleepLowLatency, PowerConfig::kDcvsDisable,
                      QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
                      DCVS_VOLTAGE_VCORNER_TURBO}));

class HtpBackendRPCControlPerfParamTest : public HtpBackendPerfBaseTest {};

TEST_P(HtpBackendRPCControlPerfParamTest, InitWithPerfMode) {
  const auto& params = GetParam();

  Options options;
  options.SetHtpPerformanceMode(params.mode);
  HtpBackend backend(&qnn_api_copy_);

#if defined(__x86_64__) || defined(_M_X64)
  EXPECT_TRUE(backend.Init(options, kSocInfos[8]));
#else
  EXPECT_TRUE(backend.Init(options, std::nullopt));
#endif

  const auto& configs = *captured_configs;
  ASSERT_EQ(configs.size(), 2);

  const auto& rpc_configs = configs[0];
  ASSERT_EQ(rpc_configs.size(), 1);
  EXPECT_EQ(rpc_configs[0].option,
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY);
  EXPECT_EQ(rpc_configs[0].rpcControlLatencyConfig,
            PowerConfig::kRpcControlLatency);

  const auto& dcvs_configs = configs[1];
  ASSERT_EQ(dcvs_configs.size(), 1);
  EXPECT_EQ(dcvs_configs[0].option,
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3);

  const auto& dcvs = dcvs_configs[0].dcvsV3Config;
  EXPECT_EQ(dcvs.sleepLatency, params.sleep_latency);
  EXPECT_EQ(dcvs.dcvsEnable, params.dcvs_enable);
  EXPECT_EQ(dcvs.powerMode, params.power_mode);
  EXPECT_EQ(dcvs.busVoltageCornerMin, params.voltage_corner);
  EXPECT_EQ(dcvs.busVoltageCornerTarget, params.voltage_corner);
  EXPECT_EQ(dcvs.busVoltageCornerMax, params.voltage_corner);
  EXPECT_EQ(dcvs.coreVoltageCornerMin, params.voltage_corner);
  EXPECT_EQ(dcvs.coreVoltageCornerTarget, params.voltage_corner);
  EXPECT_EQ(dcvs.coreVoltageCornerMax, params.voltage_corner);
}

INSTANTIATE_TEST_SUITE_P(
    HtpBackendRPCControlTests, HtpBackendRPCControlPerfParamTest,
    testing::Values(
        HtpPerfParams{HtpPerformanceMode::kPowerSaver,
                      PowerConfig::kSleepMediumLatency,
                      PowerConfig::kDcvsEnable,
                      QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
                      DCVS_VOLTAGE_VCORNER_SVS},
        HtpPerfParams{HtpPerformanceMode::kLowPowerSaver,
                      PowerConfig::kSleepMediumLatency,
                      PowerConfig::kDcvsEnable,
                      QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
                      DCVS_VOLTAGE_VCORNER_SVS2},
        HtpPerfParams{HtpPerformanceMode::kHighPowerSaver,
                      PowerConfig::kSleepMediumLatency,
                      PowerConfig::kDcvsEnable,
                      QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
                      DCVS_VOLTAGE_VCORNER_SVS_PLUS},
        HtpPerfParams{HtpPerformanceMode::kLowBalanced,
                      PowerConfig::kSleepMediumLatency,
                      PowerConfig::kDcvsEnable,
                      QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
                      DCVS_VOLTAGE_VCORNER_NOM},
        HtpPerfParams{HtpPerformanceMode::kBalanced,
                      PowerConfig::kSleepMediumLatency,
                      PowerConfig::kDcvsEnable,
                      QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
                      DCVS_VOLTAGE_VCORNER_NOM_PLUS},
        HtpPerfParams{HtpPerformanceMode::kExtremePowerSaver,
                      PowerConfig::kSleepMediumLatency,
                      PowerConfig::kDcvsEnable,
                      QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
                      DCVS_VOLTAGE_CORNER_DISABLE}));

// HTP BACKEND /////////////////////////////////////////////////////////
class HtpBackendTest : public testing::Test {
 public:
  void SetUp() override {
    handle_ = CreateDLHandle(HtpBackend::GetLibraryName());
    if (!handle_) GTEST_SKIP();

    auto* qnn_api =
        ResolveQnnApi(handle_.get(), HtpBackend::GetExpectedBackendVersion());
    ASSERT_TRUE(qnn_api);
    backend_ = std::make_unique<HtpBackend>(qnn_api);
  }

  DLHandle handle_;
  std::unique_ptr<HtpBackend> backend_;
};

TEST_F(HtpBackendTest, DISABLED_InitializeWithLogLevelOffTest) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);

#if defined(__x86_64__) || defined(_M_X64)
  ASSERT_TRUE(backend_->Init(options, kSocInfos[8]));
#else
  ASSERT_TRUE(backend_->Init(options, std::nullopt));
#endif
  ASSERT_TRUE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_FALSE(backend_->GetLogHandle());
}

TEST_F(HtpBackendTest, DISABLED_InitializeWithLogLevelVerboseTest) {
  Options options;
  options.SetLogLevel(LogLevel::kVerbose);

  // Use V75 to test with HTP backend.
  ASSERT_TRUE(backend_->Init(options, kSocInfos[8]));
  ASSERT_TRUE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_TRUE(backend_->GetLogHandle());
}

}  // namespace
}  // namespace qnn
