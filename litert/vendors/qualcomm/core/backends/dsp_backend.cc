// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "DSP/QnnDspDevice.h"              // from @qairt
#include "DSP/QnnDspPerfInfrastructure.h"  // from @qairt
#include "QnnBackend.h"                    // from @qairt
#include "QnnInterface.h"                  // from @qairt
#include "absl/types/span.h"               // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
namespace {
std::vector<QnnDspPerfInfrastructure_PowerConfig_t> CreatePowerConfigs(
    ::qnn::DspPerformanceMode perf_mode, PerformanceModeVoteType vote_type) {
  // 0: sleep_latency, 1: dcvs_vcorner_min, 2: bus_vcorner_target,
  // 3: bus_vcorner_max, 4: dcvs_enable, 5: dcvs_power_mode
  std::vector<QnnDspPerfInfrastructure_PowerConfig_t> power_configs = {
      {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_LATENCY},
      {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER},
      {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER},
      {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER},
      {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE},
      {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE}};

  auto set_configs = [&power_configs](auto sleep_latency, auto min, auto target,
                                      auto max) {
    power_configs[0].sleepLatencyConfig = sleep_latency;
    power_configs[1].dcvsVoltageCornerMinConfig = min;
    power_configs[2].busVoltageCornerTargetConfig = target;
    power_configs[3].busVoltageCornerMaxConfig = max;
  };

  if (vote_type == PerformanceModeVoteType::kDownVote) {
    power_configs[4].dcvsEnableConfig = ::qnn::PowerConfig::kDcvsEnable;
    power_configs[5].dcvsPowerModeConfig =
        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
    set_configs(::qnn::PowerConfig::kSleepHighLatency,
                DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS,
                DCVS_VOLTAGE_VCORNER_SVS);
  } else {
    power_configs[4].dcvsEnableConfig = ::qnn::PowerConfig::kDcvsDisable;
    power_configs[5].dcvsPowerModeConfig =
        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;

    switch (perf_mode) {
      case ::qnn::DspPerformanceMode::kBurst:
        set_configs(::qnn::PowerConfig::kSleepMinLatency,
                    DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                    DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                    DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER);
        break;
      case ::qnn::DspPerformanceMode::kSustainedHighPerformance:
      case ::qnn::DspPerformanceMode::kHighPerformance:
        set_configs(::qnn::PowerConfig::kSleepLowLatency,
                    DCVS_VOLTAGE_VCORNER_TURBO, DCVS_VOLTAGE_VCORNER_TURBO,
                    DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER);
        break;
      case ::qnn::DspPerformanceMode::kPowerSaver:
        set_configs(::qnn::PowerConfig::kSleepMediumLatency,
                    DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
                    DCVS_VOLTAGE_VCORNER_SVS);
        break;
      case ::qnn::DspPerformanceMode::kLowPowerSaver:
        set_configs(::qnn::PowerConfig::kSleepMediumLatency,
                    DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2,
                    DCVS_VOLTAGE_VCORNER_SVS2);
        break;
      case ::qnn::DspPerformanceMode::kHighPowerSaver:
        set_configs(::qnn::PowerConfig::kSleepMediumLatency,
                    DCVS_VOLTAGE_VCORNER_SVS_PLUS,
                    DCVS_VOLTAGE_VCORNER_SVS_PLUS,
                    DCVS_VOLTAGE_VCORNER_SVS_PLUS);
        break;
      case ::qnn::DspPerformanceMode::kLowBalanced:
        set_configs(::qnn::PowerConfig::kSleepMediumLatency,
                    DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM,
                    DCVS_VOLTAGE_VCORNER_NOM);
        break;
      case ::qnn::DspPerformanceMode::kBalanced:
        set_configs(::qnn::PowerConfig::kSleepMediumLatency,
                    DCVS_VOLTAGE_VCORNER_NOM_PLUS,
                    DCVS_VOLTAGE_VCORNER_NOM_PLUS,
                    DCVS_VOLTAGE_VCORNER_NOM_PLUS);
        break;
      default:
        QNN_LOG_INFO("performance profile %d don't need to set power configs",
                     perf_mode);
        break;
    }
    return power_configs;
  }
}
}  // namespace

DspBackend::DspBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : QnnBackend(qnn_api) {}

DspBackend::~DspBackend() {}

bool DspBackend::Init(const Options& options,
                      std::optional<::qnn::SocInfo> soc_info) {
  // Log Handle
  auto local_log_handle = CreateLogHandle(options.GetLogLevel());
  if (!local_log_handle && options.GetLogLevel() != ::qnn::LogLevel::kOff) {
    QNN_LOG_ERROR("Failed to create log handle!");
    return false;
  }

  // Backend Handle
  std::vector<const QnnBackend_Config_t*> backend_configs;
  backend_configs.emplace_back(nullptr);

  auto local_backend_handle = CreateBackendHandle(
      local_log_handle.get(), absl::MakeSpan(backend_configs));
  if (!local_backend_handle) {
    QNN_LOG_ERROR("Failed to create backend handle!");
    return false;
  }

  // DSP Performance Settings
  performance_mode_ = options.GetDspPerformanceMode();
  if (IsPerfModeEnabled()) {
    dsp_perf_infra_ = CreateDspPerfInfra();
    if (!dsp_perf_infra_) {
      return false;
    }
    // vote immediately, which only take effects in manual mode.
    PerformanceVote();
  }
  // Follow RAII pattern to manage handles
  log_handle_ = std::move(local_log_handle);
  backend_handle_ = std::move(local_backend_handle);

  return true;
}

DspBackend::DspPerfInfra DspBackend::CreateDspPerfInfra() {
  auto* infra =
      reinterpret_cast<QnnDspDevice_Infrastructure_t*>(GetPerfInfra());
  if (!infra) {
    return {nullptr, DspPerfInfraDeleter{}};
  }

  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  error = infra->createPowerConfigId(&powerconfig_client_id_);
  if (error != QNN_SUCCESS) {
    QNN_LOG_ERROR("DSP backend unable to create power config. Error %d",
                  QNN_GET_ERROR_CODE(error));
  }

  // Initialize power configurations.
  // We need to prepare both:
  // 1. UpVote config: for entering performance mode (used in PerformanceVote).
  // 2. DownVote config: for resetting/cleanup (used in DspPerfInfraDeleter).
  perf_power_configs_ =
      CreatePowerConfigs(performance_mode_, PerformanceModeVoteType::kUpVote);
  perf_power_configs_ptr_ = ObtainNullTermPtrVector(perf_power_configs_);

  down_vote_power_configs_ = CreatePowerConfigs(
      ::qnn::DspPerformanceMode::kDefault, PerformanceModeVoteType::kDownVote);
  down_vote_power_configs_ptr_ =
      ObtainNullTermPtrVector(down_vote_power_configs_);

  return DspPerfInfra(infra,
                      DspPerfInfraDeleter{powerconfig_client_id_,
                                          down_vote_power_configs_ptr_.data()});
}

void DspBackend::PerformanceVote() {
  if (IsPerfModeEnabled() && manual_voting_type_ != kUpVote) {
    dsp_perf_infra_->setPowerConfig(powerconfig_client_id_,
                                    perf_power_configs_ptr_.data());
    manual_voting_type_ = kUpVote;
  }
};

}  // namespace qnn
