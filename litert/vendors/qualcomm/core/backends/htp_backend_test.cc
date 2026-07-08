// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/htp_backend.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "HTP/QnnHtpDevice.h"  // from @qairt
#include "HTP/QnnHtpGraph.h"  // from @qairt
#include "HTP/QnnHtpPerfInfrastructure.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/backend_utils.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
namespace {

const SocInfo kFp16SocInfo = FindSocModel("SM8750").value();
const SocInfo kLegacySocInfo = FindSocModel("SA8295").value();

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
  EXPECT_TRUE(backend.Init(options, kFp16SocInfo));
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
  EXPECT_TRUE(backend.Init(options, kFp16SocInfo));
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
  ASSERT_TRUE(backend_->Init(options, kFp16SocInfo));
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
  ASSERT_TRUE(backend_->Init(options, kFp16SocInfo));
  ASSERT_TRUE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_TRUE(backend_->GetLogHandle());
}

// SETPERFORMANCEMODE /////////////////////////////////////////////////////////
TEST_P(HtpBackendRPCPollingPerfParamTest, ManualSameModeSkipsRevote) {
  const auto& params = GetParam();
  Options options;
  options.SetHtpPerformanceMode(params.mode);
  options.SetHtpPerfCtrlMode(HtpPerfCtrlMode::kManual);
  HtpBackend backend(&qnn_api_copy_);

#if defined(__x86_64__) || defined(_M_X64)
  ASSERT_TRUE(backend.Init(options, kSocInfos[8]));
#else
  ASSERT_TRUE(backend.Init(options, std::nullopt));
#endif

  const size_t configs_after_init = captured_configs->size();

  EXPECT_TRUE(backend.SetPerformanceMode(options));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(captured_configs->size(), configs_after_init);
}

TEST_P(HtpBackendRPCPollingPerfParamTest, AutoSameModeRevotes) {
  const auto& params = GetParam();
  Options options;
  options.SetHtpPerformanceMode(params.mode);
  options.SetHtpPerfCtrlMode(HtpPerfCtrlMode::kAuto);
  HtpBackend backend(&qnn_api_copy_);

#if defined(__x86_64__) || defined(_M_X64)
  ASSERT_TRUE(backend.Init(options, kSocInfos[8]));
#else
  ASSERT_TRUE(backend.Init(options, std::nullopt));
#endif

  const size_t configs_after_init = captured_configs->size();
  EXPECT_TRUE(backend.SetPerformanceMode(options));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(captured_configs->size(), configs_after_init + 1);
}

TEST_F(HtpBackendPerfBaseTest, ManualModeChangeRevotes) {
  Options options;
  options.SetHtpPerformanceMode(HtpPerformanceMode::kPowerSaver);
  options.SetHtpPerfCtrlMode(HtpPerfCtrlMode::kManual);
  HtpBackend backend(&qnn_api_copy_);

#if defined(__x86_64__) || defined(_M_X64)
  ASSERT_TRUE(backend.Init(options, kSocInfos[8]));
#else
  ASSERT_TRUE(backend.Init(options, std::nullopt));
#endif
  const size_t configs_after_init = captured_configs->size();

  options.SetHtpPerformanceMode(HtpPerformanceMode::kBurst);
  EXPECT_TRUE(backend.SetPerformanceMode(options));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_GT(captured_configs->size(), configs_after_init);
}

TEST_F(HtpBackendPerfBaseTest, DefaultModeSchedulesDownvote) {
  Options options;
  options.SetHtpPerformanceMode(HtpPerformanceMode::kBurst);
  options.SetHtpPerfCtrlMode(HtpPerfCtrlMode::kManual);
  HtpBackend backend(&qnn_api_copy_);

#if defined(__x86_64__) || defined(_M_X64)
  ASSERT_TRUE(backend.Init(options, kSocInfos[8]));
#else
  ASSERT_TRUE(backend.Init(options, std::nullopt));
#endif
  const size_t configs_after_init = captured_configs->size();

  Options default_options;  // kDefault
  EXPECT_TRUE(backend.SetPerformanceMode(default_options));
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  EXPECT_GT(captured_configs->size(), configs_after_init);
}

TEST_F(HtpBackendPerfBaseTest, AutoModeChangesAcrossExecutes) {
  Options options;
  options.SetHtpPerformanceMode(HtpPerformanceMode::kPowerSaver);
  options.SetHtpPerfCtrlMode(HtpPerfCtrlMode::kAuto);
  HtpBackend backend(&qnn_api_copy_);

#if defined(__x86_64__) || defined(_M_X64)
  ASSERT_TRUE(backend.Init(options, kSocInfos[8]));
#else
  ASSERT_TRUE(backend.Init(options, std::nullopt));
#endif
  const size_t configs_after_init = captured_configs->size();

  EXPECT_TRUE(backend.SetPerformanceMode(options));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const size_t configs_after_same = captured_configs->size();
  EXPECT_GT(configs_after_same, configs_after_init);

  Options default_options;
  EXPECT_TRUE(backend.SetPerformanceMode(default_options));

  Options burst_options;
  burst_options.SetHtpPerformanceMode(HtpPerformanceMode::kBurst);
  burst_options.SetHtpPerfCtrlMode(HtpPerfCtrlMode::kAuto);
  EXPECT_TRUE(backend.SetPerformanceMode(burst_options));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_GT(captured_configs->size(), configs_after_same);
}

// GRAPH CONFIG CHARACTERIZATION ///////////////////////////////////////////////

struct ExtractedConfigs {
  // - N × {QNN_GRAPH_CONFIG_OPTION_CUSTOM → QnnGraph_Config_t*}
  // - 1 × {QNN_GRAPH_CONFIG_OPTION_PRIORITY → QnnGraph_Config_t*}
  std::multimap<QnnGraph_ConfigOption_t, const QnnGraph_Config_t*> graph_configs;

  std::multimap<QnnHtpGraph_ConfigOption_t, const QnnHtpGraph_CustomConfig_t*>
      custom_configs;

  size_t size() const { return graph_configs.size(); }
};

ExtractedConfigs ExtractConfigs(
    const std::vector<const QnnGraph_Config_t*>& configs) {
  ExtractedConfigs result;
  for (const QnnGraph_Config_t* c : configs) {
    if (c == nullptr) break;
    result.graph_configs.emplace(c->option, c);
    if (c->option == QNN_GRAPH_CONFIG_OPTION_CUSTOM && c->customConfig) {
      const auto* cc =
          static_cast<const QnnHtpGraph_CustomConfig_t*>(c->customConfig);
      result.custom_configs.emplace(cc->option, cc);
    }
  }
  return result;
}

const QnnHtpGraph_CustomConfig_t* FindOptimization(
    const ExtractedConfigs& ext, QnnHtpGraph_OptimizationType_t type) {
  auto [b, e] =
      ext.custom_configs.equal_range(QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION);
  for (auto it = b; it != e; ++it) {
    if (it->second->optimizationOption.type == type) return it->second;
  }
  return nullptr;
}

const QnnHtpGraph_CustomConfig_t* FindCustom(const ExtractedConfigs& ext,
                                             QnnHtpGraph_ConfigOption_t opt) {
  auto it = ext.custom_configs.find(opt);
  return it == ext.custom_configs.end() ? nullptr : it->second;
}

const QnnGraph_Config_t* FindPriority(const ExtractedConfigs& ext) {
  auto it = ext.graph_configs.find(QNN_GRAPH_CONFIG_OPTION_PRIORITY);
  return it == ext.graph_configs.end() ? nullptr : it->second;
}

TEST(HtpBackendGraphConfigTest, DefaultConfigsIncludePrecisionAndPriority) {
  QNN_INTERFACE_VER_TYPE api{};
  HtpBackend backend(&api);

  Options options;

  auto config_builder = backend.BuildGraphConfigs(options, "graph");
  auto configs = config_builder.GetNullTerminatedConfigs();

  ASSERT_GE(configs.size(), 2u);
  ASSERT_EQ(configs.back(), nullptr);

  auto ext = ExtractConfigs(configs);
  const auto* precision =
      FindCustom(ext, QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION);
  ASSERT_NE(precision, nullptr);
  EXPECT_EQ(precision->precision, QNN_PRECISION_FLOAT16);

  const auto* priority = FindPriority(ext);
  ASSERT_NE(priority, nullptr);
}

// --- Default path (fp16 supported) ---
class HtpBackendDefaultGraphConfigTest : public testing::Test {
 protected:
  QNN_INTERFACE_VER_TYPE api_{};
  HtpBackend backend_{&api_};
};

TEST_F(HtpBackendDefaultGraphConfigTest, DefaultOptionsProduceExpectedKinds) {
  Options options;
  auto config_builder = backend_.BuildGraphConfigs(options, "graph");
  auto configs = config_builder.GetNullTerminatedConfigs();

  ASSERT_EQ(configs.back(), nullptr);

  auto ext = ExtractConfigs(configs);
  ASSERT_EQ(ext.size(), 6u);
  EXPECT_TRUE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION));
  EXPECT_TRUE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION));
  EXPECT_TRUE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE));
  EXPECT_TRUE(ext.custom_configs.count(
      QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF));
  EXPECT_TRUE(ext.custom_configs.count(
      QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF));
  EXPECT_TRUE(ext.graph_configs.count(QNN_GRAPH_CONFIG_OPTION_PRIORITY));
}

TEST_F(HtpBackendDefaultGraphConfigTest, PPointAndHvxInsertedWhenSet) {
  Options options;
  options.SetHtpPPoint(3);
  options.SetNumHvxThreads(4);
  auto config_builder = backend_.BuildGraphConfigs(options, "graph");
  auto configs = config_builder.GetNullTerminatedConfigs();

  ASSERT_EQ(configs.back(), nullptr);

  auto ext = ExtractConfigs(configs);
  ASSERT_EQ(ext.size(), 8u);
  EXPECT_TRUE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG));
  EXPECT_TRUE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS));
  EXPECT_TRUE(ext.graph_configs.count(QNN_GRAPH_CONFIG_OPTION_PRIORITY));

  // P-point value and key.
  const auto* ppoint_cc =
      FindCustom(ext, QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG);
  ASSERT_NE(ppoint_cc, nullptr);
  ASSERT_STREQ(ppoint_cc->finalizeConfig.key, "P");
  EXPECT_EQ(ppoint_cc->finalizeConfig.value.dataType, QNN_DATATYPE_INT_32);
  EXPECT_EQ(ppoint_cc->finalizeConfig.value.int32Value, 3);

  // HVX thread count.
  const auto* hvx_cc =
      FindCustom(ext, QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS);
  ASSERT_NE(hvx_cc, nullptr);
  EXPECT_EQ(hvx_cc->numHvxThreads, 4u);
}

TEST_F(HtpBackendDefaultGraphConfigTest, DlbcOptionsAppendOptimizationConfigs) {
  Options options;
  options.SetHtpDlbc(true);
  options.SetHtpDlbcWeights(true);  // weight sharing off by default, so kept.
  auto config_builder = backend_.BuildGraphConfigs(options, "graph");
  auto configs = config_builder.GetNullTerminatedConfigs();

  ASSERT_EQ(configs.back(), nullptr);

  auto ext = ExtractConfigs(configs);
  ASSERT_EQ(ext.size(), 8u);
  EXPECT_EQ(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION), 3u);
  EXPECT_TRUE(ext.graph_configs.count(QNN_GRAPH_CONFIG_OPTION_PRIORITY));

  const auto* dlbc_cc =
      FindOptimization(ext, QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC);
  ASSERT_NE(dlbc_cc, nullptr);
  EXPECT_FLOAT_EQ(dlbc_cc->optimizationOption.floatValue, 1.0f);

  const auto* dlbc_w_cc = FindOptimization(
      ext, QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC_WEIGHTS);
  ASSERT_NE(dlbc_w_cc, nullptr);
  EXPECT_FLOAT_EQ(dlbc_w_cc->optimizationOption.floatValue, 1.0f);
}

TEST_F(HtpBackendDefaultGraphConfigTest, NegativePPointSkipped) {
  Options options;
  options.SetHtpPPoint(-1);
  auto config_builder = backend_.BuildGraphConfigs(options, "graph");
  auto configs = config_builder.GetNullTerminatedConfigs();

  auto ext = ExtractConfigs(configs);
  EXPECT_FALSE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG))
      << "P-point config should be absent for negative HtpPPoint";
}

TEST_F(HtpBackendDefaultGraphConfigTest, ValuesReflectOptions) {
  Options options;
  options.SetVtcmSize(8);
  options.SetUseFoldReLU(true);
  options.SetUseConvHMX(false);
  options.SetOptimizationLevel(OptimizationLevel::kHtpOptimizeForPrepare);
  options.SetGraphPriority(GraphPriority::kHigh);
  auto config_builder = backend_.BuildGraphConfigs(options, "graph");
  auto configs = config_builder.GetNullTerminatedConfigs();

  ASSERT_EQ(configs.back(), nullptr);
  auto ext = ExtractConfigs(configs);
  ASSERT_EQ(ext.size(), 6u);

  const auto* vtcm_cc = FindCustom(ext, QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE);
  ASSERT_NE(vtcm_cc, nullptr);
  EXPECT_EQ(vtcm_cc->vtcmSizeInMB, 8u);

  const auto* relu_cc = FindCustom(
      ext, QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF);
  ASSERT_NE(relu_cc, nullptr);
  EXPECT_FALSE(relu_cc->foldReluActivationIntoConvOff);

  const auto* conv_cc =
      FindCustom(ext, QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF);
  ASSERT_NE(conv_cc, nullptr);
  EXPECT_TRUE(conv_cc->shortDepthConvOnHmxOff);

  const auto* opt_cc =
      FindCustom(ext, QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION);
  ASSERT_NE(opt_cc, nullptr);
  EXPECT_FLOAT_EQ(opt_cc->optimizationOption.floatValue, 1.0f);

  const QnnGraph_Config_t* prio = FindPriority(ext);
  ASSERT_NE(prio, nullptr);
  EXPECT_EQ(prio->priority, QNN_PRIORITY_HIGH);
}

// --- Legacy path (V68 / SAR2230P — fp16 NOT supported) ---
class HtpBackendLegacyGraphConfigTest : public testing::Test {
 public:
  void SetUp() override {
    handle_ = CreateDLHandle(HtpBackend::GetLibraryName());
    if (!handle_) GTEST_SKIP();
    auto* qnn_api =
        ResolveQnnApi(handle_.get(), HtpBackend::GetExpectedBackendVersion());
    ASSERT_TRUE(qnn_api);
    qnn_api_copy_ = *qnn_api;
    qnn_api_copy_.deviceGetPlatformInfo = MockDeviceGetPlatformInfo;
    qnn_api_copy_.deviceFree = MockDeviceFree;
    backend_ = std::make_unique<HtpBackend>(&qnn_api_copy_);
#if defined(__x86_64__) || defined(_M_X64)
    ASSERT_TRUE(backend_->Init(Options{}, kLegacySocInfo));
#else
    ASSERT_TRUE(backend_->Init(Options{}, std::nullopt));
    if (IsFp16Supported(backend_->GetSocInfo())) {
      GTEST_SKIP() << "On-device SoC is not legacy (fp16-capable); "
                      "skipping legacy path test.";
    }
#endif
  }

 protected:
  DLHandle handle_;
  QNN_INTERFACE_VER_TYPE qnn_api_copy_{};
  std::unique_ptr<HtpBackend> backend_;
};

TEST_F(HtpBackendLegacyGraphConfigTest,
       DISABLED_ExactSequenceNoPrecisionNoPPoint) {
  Options options;
  options.SetHtpPPoint(3);  // P-point set, but must be suppressed on legacy.
  auto config_builder = backend_->BuildGraphConfigs(options, "graph");
  auto configs = config_builder.GetNullTerminatedConfigs();

  ASSERT_EQ(configs.back(), nullptr);

  auto ext = ExtractConfigs(configs);
  ASSERT_EQ(ext.size(), 5u);
  EXPECT_TRUE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION));
  EXPECT_TRUE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE));
  EXPECT_TRUE(ext.custom_configs.count(
      QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF));
  EXPECT_TRUE(ext.custom_configs.count(
      QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF));
  EXPECT_TRUE(ext.graph_configs.count(QNN_GRAPH_CONFIG_OPTION_PRIORITY));

  EXPECT_FALSE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION))
      << "Precision config must be absent on legacy path";
  EXPECT_FALSE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG))
      << "P-point config must be absent on legacy path";
}

TEST_F(HtpBackendLegacyGraphConfigTest, DISABLED_HvxAppearsWhenSet) {
  Options options;
  options.SetNumHvxThreads(2);
  auto config_builder = backend_->BuildGraphConfigs(options, "graph");
  auto configs = config_builder.GetNullTerminatedConfigs();

  ASSERT_EQ(configs.back(), nullptr);

  auto ext = ExtractConfigs(configs);
  ASSERT_EQ(ext.size(), 6u);
  EXPECT_TRUE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS));
  EXPECT_TRUE(ext.graph_configs.count(QNN_GRAPH_CONFIG_OPTION_PRIORITY));

  // Still no precision or P-point.
  EXPECT_FALSE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION));
  EXPECT_FALSE(ext.custom_configs.count(QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG));
}

}  // namespace
}  // namespace qnn
