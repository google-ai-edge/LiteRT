// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "DSP/QnnDspDevice.h"  // from @qairt
#include "DSP/QnnDspPerfInfrastructure.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/backend_utils.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "QnnDevice.h"  // from @qairt

namespace qnn {
namespace {
// DSP PERF CONTROL /////////////////////////////////////////////////////////
QnnDevice_GetInfrastructureFn_t real_device_get_infrastructure = nullptr;
absl::NoDestructor<std::vector<QnnDspPerfInfrastructure_PowerConfig_t>>
    captured_configs;
std::atomic<int> set_power_config_call_count{0};

// Mock Functions
Qnn_ErrorHandle_t MockSetPowerConfig(
    uint32_t power_config_id,
    const QnnDspPerfInfrastructure_PowerConfig_t** power_configs) {
  if (power_configs) {
    captured_configs->clear();
    for (size_t i = 0; power_configs[i] != nullptr; ++i) {
      captured_configs->emplace_back(*power_configs[i]);
    }
  }
  set_power_config_call_count.fetch_add(1);
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

  (*infra)->setPowerConfig = MockSetPowerConfig;

  return QNN_SUCCESS;
}

struct DspPerfParams {
  DspPerformanceMode mode;
  uint32_t expected_sleep;
  QnnDspPerfInfrastructure_VoltageCorner_t expected_min_voltage;
  QnnDspPerfInfrastructure_VoltageCorner_t expected_target_voltage;
  QnnDspPerfInfrastructure_VoltageCorner_t expected_max_voltage;
};

class DspBackendPerfParamTest : public testing::TestWithParam<DspPerfParams> {
 public:
  void SetUp() override {
    captured_configs->clear();
    set_power_config_call_count.store(0);
    handle_ = CreateDLHandle(DspBackend::GetLibraryName());
    if (!handle_) GTEST_SKIP();

    auto* qnn_api =
        ResolveQnnApi(handle_.get(), DspBackend::GetExpectedBackendVersion());
    ASSERT_TRUE(qnn_api);

    qnn_api_copy_ = *qnn_api;

    real_device_get_infrastructure = qnn_api_copy_.deviceGetInfrastructure;
    qnn_api_copy_.deviceGetInfrastructure = MockDeviceGetInfrastructure;
  }

  void TearDown() override { real_device_get_infrastructure = nullptr; }

 protected:
  DLHandle handle_;
  QNN_INTERFACE_VER_TYPE qnn_api_copy_{};
};

TEST_P(DspBackendPerfParamTest, DISABLED_InitWithPerfMode) {
  const auto& params = GetParam();

  Options options;
  options.SetDspPerformanceMode(params.mode);
  DspBackend backend(&qnn_api_copy_);

  EXPECT_TRUE(backend.Init(options, std::nullopt));

  const auto& configs = *captured_configs;
  ASSERT_FALSE(configs.empty());
  ASSERT_EQ(configs.size(), 6);

  EXPECT_EQ(configs[0].config,
            QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_LATENCY);
  EXPECT_EQ(configs[0].sleepLatencyConfig, params.expected_sleep);
  EXPECT_EQ(configs[1].config,
            QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER);
  EXPECT_EQ(configs[1].dcvsVoltageCornerMinConfig, params.expected_min_voltage);
  EXPECT_EQ(configs[2].config,
            QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER);
  EXPECT_EQ(configs[2].dcvsVoltageCornerTargetConfig,
            params.expected_target_voltage);
  EXPECT_EQ(configs[3].config,
            QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER);
  EXPECT_EQ(configs[3].dcvsVoltageCornerMaxConfig, params.expected_max_voltage);
  EXPECT_EQ(configs[4].config,
            QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE);
  EXPECT_EQ(configs[4].dcvsEnableConfig, PowerConfig::kDcvsDisable);
  EXPECT_EQ(configs[5].config,
            QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE);
  EXPECT_EQ(configs[5].dcvsPowerModeConfig,
            QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE);
}

INSTANTIATE_TEST_SUITE_P(
    DspBackendPerfTests, DspBackendPerfParamTest,
    testing::Values(
        DspPerfParams{DspPerformanceMode::kBurst, PowerConfig::kSleepMinLatency,
                      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER},
        DspPerfParams{DspPerformanceMode::kSustainedHighPerformance,
                      PowerConfig::kSleepLowLatency, DCVS_VOLTAGE_VCORNER_TURBO,
                      DCVS_VOLTAGE_VCORNER_TURBO,
                      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER},
        DspPerfParams{DspPerformanceMode::kHighPerformance,
                      PowerConfig::kSleepLowLatency, DCVS_VOLTAGE_VCORNER_TURBO,
                      DCVS_VOLTAGE_VCORNER_TURBO,
                      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER},
        DspPerfParams{DspPerformanceMode::kPowerSaver,
                      PowerConfig::kSleepMediumLatency,
                      DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
                      DCVS_VOLTAGE_VCORNER_SVS},
        DspPerfParams{DspPerformanceMode::kLowPowerSaver,
                      PowerConfig::kSleepMediumLatency,
                      DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2,
                      DCVS_VOLTAGE_VCORNER_SVS2},
        DspPerfParams{
            DspPerformanceMode::kHighPowerSaver,
            PowerConfig::kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
            DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_SVS_PLUS},
        DspPerfParams{DspPerformanceMode::kLowBalanced,
                      PowerConfig::kSleepMediumLatency,
                      DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM,
                      DCVS_VOLTAGE_VCORNER_NOM},
        DspPerfParams{
            DspPerformanceMode::kBalanced, PowerConfig::kSleepMediumLatency,
            DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
            DCVS_VOLTAGE_VCORNER_NOM_PLUS}));

// DSP BACKEND /////////////////////////////////////////////////////////
class DspBackendTest : public testing::Test {
 public:
  void SetUp() override {
    handle_ = CreateDLHandle(DspBackend::GetLibraryName());
    if (!handle_) GTEST_SKIP();

    auto* qnn_api =
        ResolveQnnApi(handle_.get(), DspBackend::GetExpectedBackendVersion());
    ASSERT_TRUE(qnn_api);
    backend_ = std::make_unique<DspBackend>(qnn_api);
  }

  DLHandle handle_;
  std::unique_ptr<DspBackend> backend_;
};

TEST_F(DspBackendTest, DISABLED_InitializeWithLogLevelOffTest) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);

  ASSERT_TRUE(backend_->Init(options, std::nullopt));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_FALSE(backend_->GetLogHandle());
}

TEST_F(DspBackendTest, DISABLED_InitializeWithLogLevelVerboseTest) {
  Options options;
  options.SetLogLevel(LogLevel::kVerbose);

  ASSERT_TRUE(backend_->Init(options, std::nullopt));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_TRUE(backend_->GetLogHandle());
}

// SETPERFORMANCEMODE /////////////////////////////////////////////////////////
TEST_P(DspBackendPerfParamTest, ManualSameModeSkipsRevote) {
  const auto& params = GetParam();
  Options options;
  options.SetDspPerformanceMode(params.mode);
  options.SetDspPerfCtrlMode(DspPerfCtrlMode::kManual);
  DspBackend backend(&qnn_api_copy_);

  ASSERT_TRUE(backend.Init(options, std::nullopt));
  const int calls_after_init = set_power_config_call_count.load();

  EXPECT_TRUE(backend.SetPerformanceMode(options));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(set_power_config_call_count.load(), calls_after_init);
}

TEST_P(DspBackendPerfParamTest, AutoSameModeRevotes) {
  const auto& params = GetParam();
  Options options;
  options.SetDspPerformanceMode(params.mode);
  options.SetDspPerfCtrlMode(DspPerfCtrlMode::kAuto);
  DspBackend backend(&qnn_api_copy_);

  ASSERT_TRUE(backend.Init(options, std::nullopt));
  const int calls_after_init = set_power_config_call_count.load();

  EXPECT_TRUE(backend.SetPerformanceMode(options));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(set_power_config_call_count.load(), calls_after_init + 1);
}

class DspBackendPerfTest : public testing::Test {
 public:
  void SetUp() override {
    captured_configs->clear();
    set_power_config_call_count.store(0);
    handle_ = CreateDLHandle(DspBackend::GetLibraryName());
    if (!handle_) GTEST_SKIP();

    auto* qnn_api =
        ResolveQnnApi(handle_.get(), DspBackend::GetExpectedBackendVersion());
    ASSERT_TRUE(qnn_api);

    qnn_api_copy_ = *qnn_api;
    real_device_get_infrastructure = qnn_api_copy_.deviceGetInfrastructure;
    qnn_api_copy_.deviceGetInfrastructure = MockDeviceGetInfrastructure;
  }

  void TearDown() override { real_device_get_infrastructure = nullptr; }

 protected:
  DLHandle handle_;
  QNN_INTERFACE_VER_TYPE qnn_api_copy_{};
};

TEST_F(DspBackendPerfTest, ManualModeChangeRevotes) {
  Options options;
  options.SetDspPerformanceMode(DspPerformanceMode::kPowerSaver);
  options.SetDspPerfCtrlMode(DspPerfCtrlMode::kManual);
  DspBackend backend(&qnn_api_copy_);

  ASSERT_TRUE(backend.Init(options, std::nullopt));
  const int calls_after_init = set_power_config_call_count.load();

  options.SetDspPerformanceMode(DspPerformanceMode::kBurst);
  EXPECT_TRUE(backend.SetPerformanceMode(options));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_GT(set_power_config_call_count.load(), calls_after_init);
}

TEST_F(DspBackendPerfTest, DefaultModeSchedulesDownvote) {
  Options options;
  options.SetDspPerformanceMode(DspPerformanceMode::kBurst);
  options.SetDspPerfCtrlMode(DspPerfCtrlMode::kManual);
  DspBackend backend(&qnn_api_copy_);

  ASSERT_TRUE(backend.Init(options, std::nullopt));
  const int calls_after_init = set_power_config_call_count.load();

  Options default_options;  // kDefault
  EXPECT_TRUE(backend.SetPerformanceMode(default_options));
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  EXPECT_GT(set_power_config_call_count.load(), calls_after_init);
}

TEST_F(DspBackendPerfTest, AutoModeChangesAcrossExecutes) {
  Options options;
  options.SetDspPerformanceMode(DspPerformanceMode::kPowerSaver);
  options.SetDspPerfCtrlMode(DspPerfCtrlMode::kAuto);
  DspBackend backend(&qnn_api_copy_);

  ASSERT_TRUE(backend.Init(options, std::nullopt));
  const int calls_after_init = set_power_config_call_count.load();

  EXPECT_TRUE(backend.SetPerformanceMode(options));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const int calls_after_same = set_power_config_call_count.load();
  EXPECT_GT(calls_after_same, calls_after_init);

  Options default_options;
  EXPECT_TRUE(backend.SetPerformanceMode(default_options));

  Options burst_options;
  burst_options.SetDspPerformanceMode(DspPerformanceMode::kBurst);
  burst_options.SetDspPerfCtrlMode(DspPerfCtrlMode::kAuto);
  EXPECT_TRUE(backend.SetPerformanceMode(burst_options));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_GT(set_power_config_call_count.load(), calls_after_same);
}

}  // namespace
}  // namespace qnn
