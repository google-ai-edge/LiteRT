// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/htp_backend.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "HTP/QnnHtpDevice.h"  // from @qairt
#include "HTP/QnnHtpGraph.h"  // from @qairt
#include "HTP/QnnHtpPerfInfrastructure.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
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

// BuildGraphConfigs only reads soc_info_ / options; it never touches the QNN
// API, so a default-constructed fake API is enough and no backend library is
// needed.
TEST(HtpBackendGraphConfigTest, DefaultConfigsIncludePrecisionAndPriority) {
  QNN_INTERFACE_VER_TYPE api{};
  HtpBackend backend(&api);

  Options options;
  // Default soc_info_ (kSocInfos[0], DspArch::NONE) is fp16-supported, so the
  // default (non-legacy) config set is returned.
  auto mgr = backend.BuildGraphConfigs(options, "graph");
  auto configs = mgr.Configs();

  // Null-terminated span; the first custom config is the relax-precision config
  // and the final non-null config carries the graph priority.
  ASSERT_GE(configs.size(), 2u);
  ASSERT_EQ(configs.back(), nullptr);
  ASSERT_NE(configs[0], nullptr);
  EXPECT_EQ(configs[0]->option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  auto* precision =
      static_cast<QnnHtpGraph_CustomConfig_t*>(configs[0]->customConfig);
  ASSERT_NE(precision, nullptr);
  EXPECT_EQ(precision->option, QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION);
  EXPECT_EQ(precision->precision, QNN_PRECISION_FLOAT16);

  const QnnGraph_Config_t* last = configs[configs.size() - 2];
  ASSERT_NE(last, nullptr);
  EXPECT_EQ(last->option, QNN_GRAPH_CONFIG_OPTION_PRIORITY);
}

// GRAPH CONFIG CHARACTERIZATION ///////////////////////////////////////////////

// Extracts the ordered (graph option, htp custom option) pairs from a
// null-terminated BuildGraphConfigs span, skipping the null terminator.
struct ConfigKind {
  QnnGraph_ConfigOption_t option;
  // Only meaningful when option == QNN_GRAPH_CONFIG_OPTION_CUSTOM.
  QnnHtpGraph_ConfigOption_t custom_option;
};

std::vector<ConfigKind> ExtractConfigKinds(
    absl::Span<const QnnGraph_Config_t*> configs) {
  std::vector<ConfigKind> result;
  for (const QnnGraph_Config_t* c : configs) {
    if (c == nullptr) break;
    ConfigKind k;
    k.option = c->option;
    k.custom_option =
        (c->option == QNN_GRAPH_CONFIG_OPTION_CUSTOM && c->customConfig)
            ? static_cast<const QnnHtpGraph_CustomConfig_t*>(c->customConfig)
                  ->option
            : QNN_HTP_GRAPH_CONFIG_OPTION_UNKNOWN;
    result.push_back(k);
  }
  return result;
}

// --- Default path (fp16 supported) ---
// These tests do NOT call Init() and do NOT require libQnnHtp.so.
// Default soc_info_ = kSocInfos[0] (UNKNOWN_SDM, DspArch::NONE) is fp16-
// supported (neither V68 nor SAR2230P).

class HtpBackendDefaultGraphConfigTest : public testing::Test {
 protected:
  QNN_INTERFACE_VER_TYPE api_{};
  HtpBackend backend_{&api_};
};

TEST_F(HtpBackendDefaultGraphConfigTest, ExactSequenceWithDefaultOptions) {
  Options options;
  auto mgr = backend_.BuildGraphConfigs(options, "graph");
  auto configs = mgr.Configs();

  ASSERT_EQ(configs.back(), nullptr);

  auto kinds = ExtractConfigKinds(configs);
  // Expected: precision, opt, vtcm, fold-relu, conv-hmx, priority (6 entries)
  // P-point omitted (GetHtpPPoint()==0), HVX omitted (GetNumHvxThreads()==0).
  ASSERT_EQ(kinds.size(), 6u);
  EXPECT_EQ(kinds[0].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[0].custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION);
  EXPECT_EQ(kinds[1].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[1].custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION);
  EXPECT_EQ(kinds[2].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[2].custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE);
  EXPECT_EQ(kinds[3].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[3].custom_option,
            QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF);
  EXPECT_EQ(kinds[4].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[4].custom_option,
            QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF);
  EXPECT_EQ(kinds[5].option, QNN_GRAPH_CONFIG_OPTION_PRIORITY);
}

TEST_F(HtpBackendDefaultGraphConfigTest, PPointAndHvxInsertedWhenSet) {
  Options options;
  options.SetHtpPPoint(3);
  options.SetNumHvxThreads(4);
  auto mgr = backend_.BuildGraphConfigs(options, "graph");
  auto configs = mgr.Configs();

  ASSERT_EQ(configs.back(), nullptr);

  auto kinds = ExtractConfigKinds(configs);
  // Expected: precision, opt, vtcm, fold-relu, conv-hmx, p-point, hvx, priority
  ASSERT_EQ(kinds.size(), 8u);
  EXPECT_EQ(kinds[5].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[5].custom_option,
            QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG);
  EXPECT_EQ(kinds[6].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[6].custom_option,
            QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS);
  EXPECT_EQ(kinds[7].option, QNN_GRAPH_CONFIG_OPTION_PRIORITY);

  // P-point value and key
  const QnnGraph_Config_t* ppoint_cfg = configs[5];
  ASSERT_NE(ppoint_cfg, nullptr);
  const auto* cc =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(ppoint_cfg->customConfig);
  ASSERT_NE(cc, nullptr);
  ASSERT_STREQ(cc->finalizeConfig.key, "P");
  EXPECT_EQ(cc->finalizeConfig.value.dataType, QNN_DATATYPE_INT_32);
  EXPECT_EQ(cc->finalizeConfig.value.int32Value, 3);

  // HVX thread count
  const QnnGraph_Config_t* hvx_cfg = configs[6];
  ASSERT_NE(hvx_cfg, nullptr);
  const auto* hvx_cc =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(hvx_cfg->customConfig);
  ASSERT_NE(hvx_cc, nullptr);
  EXPECT_EQ(hvx_cc->numHvxThreads, 4u);
}

TEST_F(HtpBackendDefaultGraphConfigTest, DlbcInsertedAfterHvxBeforePriority) {
  Options options;
  options.SetHtpDlbc(true);
  options.SetHtpDlbcWeights(true);  // weight sharing off by default, so kept.
  auto mgr = backend_.BuildGraphConfigs(options, "graph");
  auto configs = mgr.Configs();

  ASSERT_EQ(configs.back(), nullptr);

  auto kinds = ExtractConfigKinds(configs);
  // Expected: precision, opt, vtcm, fold-relu, conv-hmx, dlbc, dlbc-weights,
  // priority (8 entries). P-point and HVX omitted (unset).
  ASSERT_EQ(kinds.size(), 8u);
  EXPECT_EQ(kinds[5].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[5].custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION);
  EXPECT_EQ(kinds[6].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[6].custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION);
  EXPECT_EQ(kinds[7].option, QNN_GRAPH_CONFIG_OPTION_PRIORITY);

  // DLBC (activations) optimization.
  const auto* dlbc_cc =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(configs[5]->customConfig);
  ASSERT_NE(dlbc_cc, nullptr);
  EXPECT_EQ(dlbc_cc->optimizationOption.type,
            QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC);
  EXPECT_FLOAT_EQ(dlbc_cc->optimizationOption.floatValue, 1.0f);

  // DLBC weights optimization.
  const auto* dlbc_w_cc =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(configs[6]->customConfig);
  ASSERT_NE(dlbc_w_cc, nullptr);
  EXPECT_EQ(dlbc_w_cc->optimizationOption.type,
            QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC_WEIGHTS);
  EXPECT_FLOAT_EQ(dlbc_w_cc->optimizationOption.floatValue, 1.0f);
}

TEST_F(HtpBackendDefaultGraphConfigTest, NegativePPointSkipped) {
  Options options;
  options.SetHtpPPoint(-1);
  auto mgr = backend_.BuildGraphConfigs(options, "graph");
  auto configs = mgr.Configs();

  auto kinds = ExtractConfigKinds(configs);
  for (const auto& k : kinds) {
    if (k.option == QNN_GRAPH_CONFIG_OPTION_CUSTOM) {
      EXPECT_NE(k.custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG)
          << "P-point config should be absent for negative HtpPPoint";
    }
  }
}

TEST_F(HtpBackendDefaultGraphConfigTest, ValuesReflectOptions) {
  Options options;
  options.SetVtcmSize(8);
  options.SetUseFoldReLU(true);
  options.SetUseConvHMX(false);
  options.SetOptimizationLevel(OptimizationLevel::kHtpOptimizeForPrepare);
  options.SetGraphPriority(GraphPriority::kHigh);
  auto mgr = backend_.BuildGraphConfigs(options, "graph");
  auto configs = mgr.Configs();

  ASSERT_EQ(configs.back(), nullptr);
  auto kinds = ExtractConfigKinds(configs);
  ASSERT_EQ(kinds.size(), 6u);

  const auto* vtcm_cc =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(configs[2]->customConfig);
  EXPECT_EQ(vtcm_cc->vtcmSizeInMB, 8u);

  const auto* relu_cc =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(configs[3]->customConfig);
  EXPECT_FALSE(relu_cc->foldReluActivationIntoConvOff);

  const auto* conv_cc =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(configs[4]->customConfig);
  EXPECT_TRUE(conv_cc->shortDepthConvOnHmxOff);

  const auto* opt_cc =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(configs[1]->customConfig);
  EXPECT_FLOAT_EQ(opt_cc->optimizationOption.floatValue, 1.0f);

  const QnnGraph_Config_t* prio = configs[5];
  EXPECT_EQ(prio->option, QNN_GRAPH_CONFIG_OPTION_PRIORITY);
  EXPECT_EQ(prio->priority, QNN_PRIORITY_HIGH);
}

// --- Legacy path (V68 / SAR2230P — fp16 NOT supported) ---
// Requires Init() and libQnnHtp.so; follows the perf-test pattern.
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
    // kSocInfos[4] = SA8295, DspArch::V68 — legacy path.
#if defined(__x86_64__) || defined(_M_X64)
    ASSERT_TRUE(backend_->Init(Options{}, kSocInfos[4]));
#else
    ASSERT_TRUE(backend_->Init(Options{}, std::nullopt));
    if (backend_->GetSocInfo().dsp_arch != DspArch::V68 &&
        backend_->GetSocInfo().soc_model != SnapdragonModel::SAR2230P) {
      GTEST_SKIP() << "On-device SoC is not legacy (V68/SAR2230P); "
                      "skipping legacy path test.";
    }
#endif
  }

 protected:
  DLHandle handle_;
  QNN_INTERFACE_VER_TYPE qnn_api_copy_{};
  std::unique_ptr<HtpBackend> backend_;
};

TEST_F(HtpBackendLegacyGraphConfigTest, ExactSequenceNoPrecisionNoPPoint) {
  Options options;
  options.SetHtpPPoint(3);  // P-point set, but must be suppressed on legacy.
  auto mgr = backend_->BuildGraphConfigs(options, "graph");
  auto configs = mgr.Configs();

  ASSERT_EQ(configs.back(), nullptr);

  auto kinds = ExtractConfigKinds(configs);
  // Expected: opt, vtcm, fold-relu, conv-hmx, priority (5 entries).
  // NO precision config, NO P-point config.
  ASSERT_EQ(kinds.size(), 5u);
  EXPECT_EQ(kinds[0].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[0].custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION);
  EXPECT_EQ(kinds[1].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[1].custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE);
  EXPECT_EQ(kinds[2].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[2].custom_option,
            QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF);
  EXPECT_EQ(kinds[3].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[3].custom_option,
            QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF);
  EXPECT_EQ(kinds[4].option, QNN_GRAPH_CONFIG_OPTION_PRIORITY);

  for (const auto& k : kinds) {
    if (k.option == QNN_GRAPH_CONFIG_OPTION_CUSTOM) {
      EXPECT_NE(k.custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION)
          << "Precision config must be absent on legacy path";
      EXPECT_NE(k.custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG)
          << "P-point config must be absent on legacy path";
    }
  }
}

TEST_F(HtpBackendLegacyGraphConfigTest, HvxAppearsWhenSet) {
  Options options;
  options.SetNumHvxThreads(2);
  auto mgr = backend_->BuildGraphConfigs(options, "graph");
  auto configs = mgr.Configs();

  ASSERT_EQ(configs.back(), nullptr);

  auto kinds = ExtractConfigKinds(configs);
  // Expected: opt, vtcm, fold-relu, conv-hmx, hvx, priority (6 entries).
  ASSERT_EQ(kinds.size(), 6u);
  EXPECT_EQ(kinds[4].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(kinds[4].custom_option,
            QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS);
  EXPECT_EQ(kinds[5].option, QNN_GRAPH_CONFIG_OPTION_PRIORITY);

  // Still no precision or P-point.
  for (const auto& k : kinds) {
    if (k.option == QNN_GRAPH_CONFIG_OPTION_CUSTOM) {
      EXPECT_NE(k.custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION);
      EXPECT_NE(k.custom_option, QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG);
    }
  }
}

}  // namespace
}  // namespace qnn
