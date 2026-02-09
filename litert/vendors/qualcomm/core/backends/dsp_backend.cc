// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/backend_utils.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "DSP/QnnDspDevice.h"  // from @qairt
#include "DSP/QnnDspPerfInfrastructure.h"  // from @qairt
#include "QnnBackend.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace qnn {

// DSP PERF CONTROL /////////////////////////////////////////////////////////
class DspBackend::DspPerfControl {
 public:
  explicit DspPerfControl(const QNN_INTERFACE_VER_TYPE* api) : api_(api) {}

  ~DspPerfControl() {
    DownVote();
    if (dsp_perf_infra_) {
      dsp_perf_infra_->destroyPowerConfigId(power_config_id_);
    }
  }

  bool Init(DspPerformanceMode performance_mode) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    if (error = api_->deviceGetInfrastructure(&dsp_perf_infra_);
        error != QNN_SUCCESS) {
      QNN_LOG_ERROR(
          "DSP backend unable to create device infrastructure. Error %d",
          QNN_GET_ERROR_CODE(error));
      return false;
    }

    if (error = dsp_perf_infra_->createPowerConfigId(&power_config_id_);
        error != QNN_SUCCESS) {
      QNN_LOG_ERROR("DSP backend unable to create power config. Error %d",
                    QNN_GET_ERROR_CODE(error));
      return false;
    }

    // Initialize power configurations.
    // We need to prepare both:
    // 1. UpVote config: for entering performance mode.
    // 2. DownVote config: for resetting/cleanup.
    InitUpVotePowerConfigs(performance_mode);
    InitDownVotePowerConfigs();

    return true;
  }

  void UpVote() {
    if (dsp_perf_infra_) {
      dsp_perf_infra_->setPowerConfig(power_config_id_,
                                      up_vote_power_configs_ptr_.data());
    }
  }

  void DownVote() {
    if (dsp_perf_infra_) {
      dsp_perf_infra_->setPowerConfig(power_config_id_,
                                      down_vote_power_configs_ptr_.data());
    }
  }

 private:
  static constexpr size_t kNumPowerConfigs = 6;
  void SetPowerConfigs(std::array<QnnDspPerfInfrastructure_PowerConfig_t,
                                  kNumPowerConfigs>& power_configs,
                       uint32_t sleep_latency,
                       QnnDspPerfInfrastructure_VoltageCorner_t voltage_min,
                       QnnDspPerfInfrastructure_VoltageCorner_t voltage_target,
                       QnnDspPerfInfrastructure_VoltageCorner_t voltage_max,
                       QnnDspPerfInfrastructure_DcvsEnable_t dcvs_enable,
                       QnnDspPerfInfrastructure_PowerMode_t power_mode) {
    constexpr size_t kSleepLatency = 0;
    constexpr size_t kDcvsVcornerMin = 1;
    constexpr size_t kDcvsVcornerTarget = 2;
    constexpr size_t kDcvsVcornerMax = 3;
    constexpr size_t kDcvsEnable = 4;
    constexpr size_t kDcvsPowerMode = 5;

    power_configs[kSleepLatency].config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_LATENCY;
    power_configs[kSleepLatency].sleepLatencyConfig = sleep_latency;

    power_configs[kDcvsVcornerMin].config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER;
    power_configs[kDcvsVcornerMin].dcvsVoltageCornerMinConfig = voltage_min;

    power_configs[kDcvsVcornerTarget].config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER;
    power_configs[kDcvsVcornerTarget].dcvsVoltageCornerTargetConfig =
        voltage_target;

    power_configs[kDcvsVcornerMax].config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER;
    power_configs[kDcvsVcornerMax].dcvsVoltageCornerMaxConfig = voltage_max;

    power_configs[kDcvsEnable].config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE;
    power_configs[kDcvsEnable].dcvsEnableConfig = dcvs_enable;

    power_configs[kDcvsPowerMode].config =
        QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE;
    power_configs[kDcvsPowerMode].dcvsPowerModeConfig = power_mode;
  }

  void InitUpVotePowerConfigs(DspPerformanceMode perf_mode) {
    switch (perf_mode) {
      case DspPerformanceMode::kBurst:
        SetPowerConfigs(up_vote_power_configs_, PowerConfig::kSleepMinLatency,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        PowerConfig::kDcvsDisable,
                        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE);
        break;
      case DspPerformanceMode::kSustainedHighPerformance:
      case DspPerformanceMode::kHighPerformance:
        SetPowerConfigs(up_vote_power_configs_, PowerConfig::kSleepLowLatency,
                        DCVS_VOLTAGE_VCORNER_TURBO, DCVS_VOLTAGE_VCORNER_TURBO,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        PowerConfig::kDcvsDisable,
                        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE);
        break;
      case DspPerformanceMode::kPowerSaver:
        SetPowerConfigs(up_vote_power_configs_,
                        PowerConfig::kSleepMediumLatency,
                        DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
                        DCVS_VOLTAGE_VCORNER_SVS, PowerConfig::kDcvsDisable,
                        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE);
        break;
      case DspPerformanceMode::kLowPowerSaver:
        SetPowerConfigs(up_vote_power_configs_,
                        PowerConfig::kSleepMediumLatency,
                        DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2,
                        DCVS_VOLTAGE_VCORNER_SVS2, PowerConfig::kDcvsDisable,
                        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE);
        break;
      case DspPerformanceMode::kHighPowerSaver:
        SetPowerConfigs(
            up_vote_power_configs_, PowerConfig::kSleepMediumLatency,
            DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
            DCVS_VOLTAGE_VCORNER_SVS_PLUS, PowerConfig::kDcvsDisable,
            QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE);
        break;
      case DspPerformanceMode::kLowBalanced:
        SetPowerConfigs(up_vote_power_configs_,
                        PowerConfig::kSleepMediumLatency,
                        DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM,
                        DCVS_VOLTAGE_VCORNER_NOM, PowerConfig::kDcvsDisable,
                        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE);
        break;
      case DspPerformanceMode::kBalanced:
        SetPowerConfigs(
            up_vote_power_configs_, PowerConfig::kSleepMediumLatency,
            DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
            DCVS_VOLTAGE_VCORNER_NOM_PLUS, PowerConfig::kDcvsDisable,
            QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE);
        break;
      default:
        QNN_LOG_INFO("Unknown perf mode %d, use default.", perf_mode);
        break;
    }
    SetNullTermPtrArray(absl::MakeConstSpan(up_vote_power_configs_),
                        up_vote_power_configs_ptr_);
  }

  void InitDownVotePowerConfigs() {
    SetPowerConfigs(down_vote_power_configs_, PowerConfig::kSleepHighLatency,
                    DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS,
                    DCVS_VOLTAGE_VCORNER_SVS, PowerConfig::kDcvsEnable,
                    QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE);
    SetNullTermPtrArray(absl::MakeConstSpan(down_vote_power_configs_),
                        down_vote_power_configs_ptr_);
  }

  const QNN_INTERFACE_VER_TYPE* api_{nullptr};
  std::uint32_t power_config_id_{0};
  QnnDevice_Infrastructure_t dsp_perf_infra_{nullptr};
  std::array<QnnDspPerfInfrastructure_PowerConfig_t, kNumPowerConfigs>
      up_vote_power_configs_;
  std::array<QnnDspPerfInfrastructure_PowerConfig_t, kNumPowerConfigs>
      down_vote_power_configs_;
  std::array<const QnnDspPerfInfrastructure_PowerConfig_t*,
             kNumPowerConfigs + 1>
      up_vote_power_configs_ptr_;
  std::array<const QnnDspPerfInfrastructure_PowerConfig_t*,
             kNumPowerConfigs + 1>
      down_vote_power_configs_ptr_;
};

// DSP BACKEND /////////////////////////////////////////////////////////
DspBackend::DspBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : QnnBackend(qnn_api) {}

DspBackend::~DspBackend() = default;

bool DspBackend::Init(const Options& options, std::optional<SocInfo> soc_info) {
  // Log Handle
  auto local_log_handle = CreateLogHandle(options.GetLogLevel());
  if (!local_log_handle && options.GetLogLevel() != LogLevel::kOff) {
    QNN_LOG_ERROR("Failed to create log handle.");
    return false;
  }

  // Backend Handle
  std::vector<const QnnBackend_Config_t*> backend_configs;
  backend_configs.emplace_back(nullptr);

  auto local_backend_handle = CreateBackendHandle(
      local_log_handle.get(), absl::MakeSpan(backend_configs));
  if (!local_backend_handle) {
    QNN_LOG_ERROR("Failed to create backend handle.");
    return false;
  }

  // DSP Performance Settings
  DspPerformanceMode performance_mode = options.GetDspPerformanceMode();
  if (performance_mode != DspPerformanceMode::kDefault) {
    dsp_perf_control_ = std::make_unique<DspPerfControl>(QnnApi());
    if (!dsp_perf_control_->Init(performance_mode)) {
      QNN_LOG_ERROR("Failed to initialize DSP performance Control.");
      return false;
    }
    dsp_perf_control_->UpVote();
  }

  // Follow RAII pattern to manage handles.
  log_handle_ = std::move(local_log_handle);
  backend_handle_ = std::move(local_backend_handle);

  return true;
}

}  // namespace qnn
