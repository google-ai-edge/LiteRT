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

// DSP PERF CONTROL /////////////////////////////////////////////////////////
class DspBackend::DspPerfControl {
 public:
  static constexpr int kNumPowerConfigs = 6;
  explicit DspPerfControl(const QNN_INTERFACE_VER_TYPE* api) : api_(api) {}

  ~DspPerfControl() {
    DownVote();
    if (dsp_perf_infra_) {
      dsp_perf_infra_->destroyPowerConfigId(power_config_id_);
    }
  }

  bool Init(DspPerformanceMode performance_mode) {
    QnnDevice_Infrastructure_t local_dsp_perf_infra = nullptr;
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    if (error = api_->deviceGetInfrastructure(&local_dsp_perf_infra);
        error != QNN_SUCCESS) {
      QNN_LOG_ERROR(
          "DSP backend unable to create device infrastructure. Error %d",
          QNN_GET_ERROR_CODE(error));
      return false;
    }

    if (error = local_dsp_perf_infra->createPowerConfigId(&power_config_id_);
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
    perf_power_configs_ptr_ = ObtainNullTermPtrArray(perf_power_configs_);

    InitDownVotePowerConfigs();
    down_vote_power_configs_ptr_ =
        ObtainNullTermPtrArray(down_vote_power_configs_);

    dsp_perf_infra_ = std::move(local_dsp_perf_infra);

    return true;
  }

  void UpVote() {
    if (dsp_perf_infra_) {
      dsp_perf_infra_->setPowerConfig(power_config_id_,
                                      perf_power_configs_ptr_.data());
    }
  }

  void DownVote() {
    if (dsp_perf_infra_) {
      dsp_perf_infra_->setPowerConfig(power_config_id_,
                                      down_vote_power_configs_ptr_.data());
    }
  }

 private:
  static void SetPowerConfigs(
      std::array<QnnDspPerfInfrastructure_PowerConfig_t, kNumPowerConfigs>&
          power_configs,
      uint32_t sleep_latency, QnnDspPerfInfrastructure_VoltageCorner_t min,
      QnnDspPerfInfrastructure_VoltageCorner_t target,
      QnnDspPerfInfrastructure_VoltageCorner_t max) {
    power_configs[0].sleepLatencyConfig = sleep_latency;
    power_configs[1].dcvsVoltageCornerMinConfig = min;
    power_configs[2].busVoltageCornerTargetConfig = target;
    power_configs[3].busVoltageCornerMaxConfig = max;
  }

  void InitUpVotePowerConfigs(DspPerformanceMode perf_mode) {
    // 0: sleep_latency, 1: dcvs_vcorner_min, 2: bus_vcorner_target,
    // 3: bus_vcorner_max, 4: dcvs_enable, 5: dcvs_power_mode
    perf_power_configs_ = {
        {{QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_LATENCY},
         {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER},
         {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER},
         {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER},
         {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE},
         {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE}}};

    perf_power_configs_[4].dcvsEnableConfig = ::qnn::PowerConfig::kDcvsDisable;
    perf_power_configs_[5].dcvsPowerModeConfig =
        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;

    switch (perf_mode) {
      case ::qnn::DspPerformanceMode::kBurst:
        SetPowerConfigs(perf_power_configs_,
                        ::qnn::PowerConfig::kSleepMinLatency,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER);
        break;
      case ::qnn::DspPerformanceMode::kSustainedHighPerformance:
      case ::qnn::DspPerformanceMode::kHighPerformance:
        SetPowerConfigs(perf_power_configs_,
                        ::qnn::PowerConfig::kSleepLowLatency,
                        DCVS_VOLTAGE_VCORNER_TURBO, DCVS_VOLTAGE_VCORNER_TURBO,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER);
        break;
      case ::qnn::DspPerformanceMode::kPowerSaver:
        SetPowerConfigs(perf_power_configs_,
                        ::qnn::PowerConfig::kSleepMediumLatency,
                        DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
                        DCVS_VOLTAGE_VCORNER_SVS);
        break;
      case ::qnn::DspPerformanceMode::kLowPowerSaver:
        SetPowerConfigs(perf_power_configs_,
                        ::qnn::PowerConfig::kSleepMediumLatency,
                        DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2,
                        DCVS_VOLTAGE_VCORNER_SVS2);
        break;
      case ::qnn::DspPerformanceMode::kHighPowerSaver:
        SetPowerConfigs(
            perf_power_configs_, ::qnn::PowerConfig::kSleepMediumLatency,
            DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
            DCVS_VOLTAGE_VCORNER_SVS_PLUS);
        break;
      case ::qnn::DspPerformanceMode::kLowBalanced:
        SetPowerConfigs(perf_power_configs_,
                        ::qnn::PowerConfig::kSleepMediumLatency,
                        DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM,
                        DCVS_VOLTAGE_VCORNER_NOM);
        break;
      case ::qnn::DspPerformanceMode::kBalanced:
        SetPowerConfigs(
            perf_power_configs_, ::qnn::PowerConfig::kSleepMediumLatency,
            DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
            DCVS_VOLTAGE_VCORNER_NOM_PLUS);
        break;
      default:
        QNN_LOG_INFO("performance profile %d don't need to set power configs",
                     perf_mode);
        break;
    }
  }

  void InitDownVotePowerConfigs() {
    // 0: sleep_latency, 1: dcvs_vcorner_min, 2: bus_vcorner_target,
    // 3: bus_vcorner_max, 4: dcvs_enable, 5: dcvs_power_mode
    down_vote_power_configs_ = {
        {{QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_LATENCY},
         {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER},
         {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER},
         {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_VOLTAGE_CORNER},
         {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE},
         {QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE}}};

    down_vote_power_configs_[4].dcvsEnableConfig =
        ::qnn::PowerConfig::kDcvsEnable;
    down_vote_power_configs_[5].dcvsPowerModeConfig =
        QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
    SetPowerConfigs(down_vote_power_configs_,
                    ::qnn::PowerConfig::kSleepHighLatency,
                    DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS,
                    DCVS_VOLTAGE_VCORNER_SVS);
  }

  const QNN_INTERFACE_VER_TYPE* api_{nullptr};
  std::uint32_t power_config_id_{0};
  QnnDevice_Infrastructure_t dsp_perf_infra_{nullptr};
  std::array<QnnDspPerfInfrastructure_PowerConfig_t, kNumPowerConfigs>
      perf_power_configs_;
  std::array<QnnDspPerfInfrastructure_PowerConfig_t, kNumPowerConfigs>
      down_vote_power_configs_;
  std::array<const QnnDspPerfInfrastructure_PowerConfig_t*,
             kNumPowerConfigs + 1>
      perf_power_configs_ptr_;
  std::array<const QnnDspPerfInfrastructure_PowerConfig_t*,
             kNumPowerConfigs + 1>
      down_vote_power_configs_ptr_;
};

// DSP BACKEND /////////////////////////////////////////////////////////
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
  DspPerformanceMode performance_mode = options.GetDspPerformanceMode();
  if (performance_mode != DspPerformanceMode::kDefault) {
    dsp_perf_control_ = std::make_unique<DspPerfControl>(QnnApi());
    if (!dsp_perf_control_->Init(performance_mode)) {
      QNN_LOG_ERROR("Failed to initialize DSP performance Control!");
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
