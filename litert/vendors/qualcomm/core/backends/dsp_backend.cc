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

namespace qnn {

namespace {
std::vector<QnnDspPerfInfrastructure_PowerConfig_t> SetPowerConfig(
    ::qnn::DspPerformanceMode perf_mode, PerformanceModeVoteType vote_type) {
  constexpr const int kNumConfigs = 6;
  std::vector<QnnDspPerfInfrastructure_PowerConfig_t> power_configs(
      kNumConfigs);
  QnnDspPerfInfrastructure_PowerConfig_t& dcvs_enable = power_configs[0];
  QnnDspPerfInfrastructure_PowerConfig_t& sleep = power_configs[1];
  QnnDspPerfInfrastructure_PowerConfig_t& dcvs_power_mode = power_configs[2];
  QnnDspPerfInfrastructure_PowerConfig_t& dcvs_vcorner_min = power_configs[3];
  QnnDspPerfInfrastructure_PowerConfig_t& dcvs_vcorner_target =
      power_configs[4];
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
  QnnDspPerfInfrastructure_PowerConfig_t& dcvs_vcorner_max = power_configs[5];

  // configs
  dcvs_enable.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE;
  sleep.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_LATENCY;
  dcvs_power_mode.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE;
  dcvs_vcorner_min.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER;
  dcvs_vcorner_target.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER;
  dcvs_vcorner_max.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER;

  if (vote_type == PerformanceModeVoteType::kDownVote) {
    dcvs_enable.dcvsEnableConfig = ::qnn::PowerConfig::kDcvsEnable;
    sleep.sleepLatencyConfig = ::qnn::PowerConfig::kSleepHighLatency;
    dcvs_power_mode.dcvsPowerModeConfig =
        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
    dcvs_vcorner_min.dcvsVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_SVS2;
    dcvs_vcorner_target.busVoltageCornerTargetConfig = DCVS_VOLTAGE_VCORNER_SVS;
    dcvs_vcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_SVS;
    return power_configs;
  }

  // UpVote
  dcvs_power_mode.dcvsPowerModeConfig =
      QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvs_enable.dcvsEnableConfig = ::qnn::PowerConfig::kDcvsDisable;
  // choose performance mode
  switch (perf_mode) {
    case ::qnn::DspPerformanceMode::kBurst:
      sleep.sleepLatencyConfig = ::qnn::PowerConfig::kSleepMinLatency;
      dcvs_vcorner_min.dcvsVoltageCornerMinConfig =
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_vcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_vcorner_max.busVoltageCornerMaxConfig =
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      break;
    case ::qnn::DspPerformanceMode::kSustainedHighPerformance:
    case ::qnn::DspPerformanceMode::kHighPerformance:
      sleep.sleepLatencyConfig = ::qnn::PowerConfig::kSleepLowLatency;
      dcvs_vcorner_min.dcvsVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_vcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_vcorner_max.busVoltageCornerMaxConfig =
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      break;
    case ::qnn::DspPerformanceMode::kPowerSaver:
      sleep.sleepLatencyConfig = ::qnn::PowerConfig::kSleepMediumLatency;
      dcvs_vcorner_min.dcvsVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_vcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_vcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_SVS;
      break;
    case ::qnn::DspPerformanceMode::kLowPowerSaver:
      sleep.sleepLatencyConfig = ::qnn::PowerConfig::kSleepMediumLatency;
      dcvs_vcorner_min.dcvsVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_vcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_vcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_SVS2;
      break;
    case ::qnn::DspPerformanceMode::kHighPowerSaver:
      sleep.sleepLatencyConfig = ::qnn::PowerConfig::kSleepMediumLatency;
      dcvs_vcorner_min.dcvsVoltageCornerMinConfig =
          DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_vcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_vcorner_max.busVoltageCornerMaxConfig =
          DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      break;
    case ::qnn::DspPerformanceMode::kLowBalanced:
      sleep.sleepLatencyConfig = ::qnn::PowerConfig::kSleepMediumLatency;
      dcvs_vcorner_min.dcvsVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_vcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_vcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_NOM;
      break;
    case ::qnn::DspPerformanceMode::kBalanced:
      sleep.sleepLatencyConfig = ::qnn::PowerConfig::kSleepMediumLatency;
      dcvs_vcorner_min.dcvsVoltageCornerMinConfig =
          DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_vcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_vcorner_max.busVoltageCornerMaxConfig =
          DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      break;
    default:
      QNN_LOG_INFO("performance profile %d don't need to set power configs",
                   perf_mode);
      break;
  }
  return power_configs;
}
}  // namespace

struct DspBackend::BackendConfig {
  QnnDspDevice_Infrastructure_t* dsp_perf_infra_{nullptr};
  std::vector<QnnDspPerfInfrastructure_PowerConfig_t> perf_power_configs_;
  std::vector<QnnDspPerfInfrastructure_PowerConfig_t> down_vote_power_configs_;
  std::vector<const QnnDspPerfInfrastructure_PowerConfig_t*>
      perf_power_configs_ptr_;
  std::vector<const QnnDspPerfInfrastructure_PowerConfig_t*>
      down_vote_power_configs_ptr_;
};

DspBackend::DspBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : QnnBackend(qnn_api) {
  backend_config_ = std::make_unique<BackendConfig>();
}

DspBackend::~DspBackend() {
  if (IsPerfModeEnabled()) {
    // Downvote here after voting thread is destroyed after reset()
    // downvoting or not, and add a systrace msg
    backend_config_->dsp_perf_infra_->setPowerConfig(
        powerconfig_client_id_,
        backend_config_->down_vote_power_configs_ptr_.data());
    backend_config_->dsp_perf_infra_->destroyPowerConfigId(
        powerconfig_client_id_);
    manual_voting_type_ = kNoVote;
  }
}

QnnDspDevice_Infrastructure_t* GetPerfInfra(const QNN_INTERFACE_VER_TYPE* api) {
  QnnDevice_Infrastructure_t device_infra = nullptr;
  Qnn_ErrorHandle_t error = api->deviceGetInfrastructure(&device_infra);
  if (error != QNN_SUCCESS) {
    QNN_LOG_ERROR("DSP backend performance_mode creation failed. Error %d",
                  QNN_GET_ERROR_CODE(error));
    return nullptr;
  }

  return static_cast<QnnDspDevice_Infrastructure_t*>(device_infra);
}

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
    backend_config_->dsp_perf_infra_ = GetPerfInfra(QnnApi());
    if (!backend_config_->dsp_perf_infra_) {
      return false;
    }

    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    error = backend_config_->dsp_perf_infra_->createPowerConfigId(
        &powerconfig_client_id_);
    if (error != QNN_SUCCESS) {
      QNN_LOG_ERROR("DSP backend unable to create power config. Error %d",
                    QNN_GET_ERROR_CODE(error));
    }

    // Set vector of PowerConfigs and map it to a vector of pointers.
    if (auto status =
            CreatePerfPowerConfigPtr(PerformanceModeVoteType::kUpVote);
        !status) {
      return false;
    }
    if (auto status =
            CreatePerfPowerConfigPtr(PerformanceModeVoteType::kDownVote);
        !status) {
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

void DspBackend::PerformanceVote() {
  if (IsPerfModeEnabled() && manual_voting_type_ != kUpVote) {
    backend_config_->dsp_perf_infra_->setPowerConfig(
        powerconfig_client_id_,
        backend_config_->perf_power_configs_ptr_.data());
    manual_voting_type_ = kUpVote;
  }
};

bool DspBackend::CreatePerfPowerConfigPtr(
    const PerformanceModeVoteType vote_type) {
  // Set vector of PowerConfigs and map it to a vector of pointers.
  if (vote_type == PerformanceModeVoteType::kUpVote) {
    backend_config_->perf_power_configs_ =
        SetPowerConfig(performance_mode_, PerformanceModeVoteType::kUpVote);
    for (auto& config : backend_config_->perf_power_configs_) {
      backend_config_->perf_power_configs_ptr_.emplace_back(&config);
    }
    backend_config_->perf_power_configs_ptr_.push_back(nullptr);
  } else if (vote_type == PerformanceModeVoteType::kDownVote) {
    backend_config_->down_vote_power_configs_ =
        SetPowerConfig(::qnn::DspPerformanceMode::kDefault,
                       PerformanceModeVoteType::kDownVote);
    for (auto& config : backend_config_->down_vote_power_configs_) {
      backend_config_->down_vote_power_configs_ptr_.emplace_back(&config);
    }
    backend_config_->down_vote_power_configs_ptr_.push_back(nullptr);
  } else {
    QNN_LOG_ERROR(
        "Something wrong when creating perf power config pointer "
        "in mode %d during vote type %d",
        performance_mode_, vote_type);
    return false;
  }

  return true;
}

}  // namespace qnn
