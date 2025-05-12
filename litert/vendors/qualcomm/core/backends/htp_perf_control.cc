// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/htp_perf_control.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "HTP/QnnHtpDevice.h"  // from @qairt
#include "HTP/QnnHtpPerfInfrastructure.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace {
// constexpr config values
constexpr int kSleepMinLatency = 40;
constexpr int kSleepLowLatency = 100;
constexpr int kSleepMediumLatency = 1000;
constexpr int kSleepMaxLatency = 65535;
constexpr int kDcvsDisable = 0;
constexpr int kDcvsEnable = 1;

// default rpc control latency - 0 us
constexpr int kRpcControlLatency = 0;
// default rpc polling time for high power modes - 9999 us
constexpr int kRpcPollingTimeHighPower = 9999;
}  // namespace

bool GetPerfInfra(const QNN_INTERFACE_VER_TYPE* api,
                  QnnHtpDevice_PerfInfrastructure_t* p_out) {
  QnnDevice_Infrastructure_t device_infra = nullptr;
  Qnn_ErrorHandle_t error = api->deviceGetInfrastructure(&device_infra);

  if (error != QNN_SUCCESS) {
    QNN_LOG_ERROR("HTP backend perf_infrastructure creation failed. Error %d",
                  QNN_GET_ERROR_CODE(error));
    return false;
  }

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(device_infra);
  if (htp_infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
    QNN_LOG_ERROR("HTP infra type = %d, which is not perf infra type.",
                  htp_infra->infraType);
    return false;
  }

  *p_out = htp_infra->perfInfra;
  return true;
}

bool SetQnnHtpPerfInfrastructure(
    QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3,
    const QnnHtpPerfInfrastructure_PowerMode_t& power_mode,
    const int& sleep_latency,
    const QnnHtpPerfInfrastructure_VoltageCorner_t& bus_voltage_corner_min,
    const QnnHtpPerfInfrastructure_VoltageCorner_t& bus_voltage_corner_max,
    const QnnHtpPerfInfrastructure_VoltageCorner_t& bus_voltage_corner_target,
    const QnnHtpPerfInfrastructure_VoltageCorner_t& core_voltage_corner_min,
    const QnnHtpPerfInfrastructure_VoltageCorner_t& core_voltage_corner_max,
    const QnnHtpPerfInfrastructure_VoltageCorner_t&
        core_voltage_corner_target) {
  if (power_mode == QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_UNKNOWN ||
      bus_voltage_corner_min == DCVS_VOLTAGE_VCORNER_UNKNOWN ||
      bus_voltage_corner_max == DCVS_VOLTAGE_VCORNER_UNKNOWN ||
      bus_voltage_corner_target == DCVS_VOLTAGE_VCORNER_UNKNOWN ||
      core_voltage_corner_min == DCVS_VOLTAGE_VCORNER_UNKNOWN ||
      core_voltage_corner_max == DCVS_VOLTAGE_VCORNER_UNKNOWN ||
      core_voltage_corner_target == DCVS_VOLTAGE_VCORNER_UNKNOWN) {
    QNN_LOG_ERROR("Invalid QnnHtpPerfInfrastructure setting.");
    return false;
  }
  dcvs_v3.powerMode = power_mode;
  dcvs_v3.sleepLatency = sleep_latency;

  dcvs_v3.busVoltageCornerMin = bus_voltage_corner_min;
  dcvs_v3.busVoltageCornerTarget = bus_voltage_corner_max;
  dcvs_v3.busVoltageCornerMax = bus_voltage_corner_target;

  dcvs_v3.coreVoltageCornerMin = core_voltage_corner_min;
  dcvs_v3.coreVoltageCornerTarget = core_voltage_corner_max;
  dcvs_v3.coreVoltageCornerMax = core_voltage_corner_target;
  return true;
}

bool HandleDownvoteConfig(const ::qnn::HtpPerformanceMode perf_mode,
                          QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3) {
  bool status = true;
  dcvs_v3.dcvsEnable = kDcvsEnable;

  switch (perf_mode) {
    case ::qnn::HtpPerformanceMode::kBurst:
    case ::qnn::HtpPerformanceMode::kSustainedHighPerformance:
    case ::qnn::HtpPerformanceMode::kHighPerformance:
    case ::qnn::HtpPerformanceMode::kBalanced:
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
          kSleepMaxLatency, DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS2,
          DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS);
      break;
    case ::qnn::HtpPerformanceMode::kPowerSaver:
    case ::qnn::HtpPerformanceMode::kLowPowerSaver:
    case ::qnn::HtpPerformanceMode::kHighPowerSaver:
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE,
          kSleepMaxLatency, DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER);
      break;
    case ::qnn::HtpPerformanceMode::kLowBalanced:
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE,
          kSleepMaxLatency, DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS2,
          DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS);
      break;
    case ::qnn::HtpPerformanceMode::kExtremePowerSaver:
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE,
          kSleepMaxLatency, DCVS_VOLTAGE_CORNER_DISABLE,
          DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE,
          DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE,
          DCVS_VOLTAGE_CORNER_DISABLE);
      break;
    default:
      status = false;
      break;
  }
  return status;
}

bool HandleUpvoteConfig(const ::qnn::HtpPerformanceMode perf_mode,
                        QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3) {
  bool status = true;
  switch (perf_mode) {
    case ::qnn::HtpPerformanceMode::kBurst:
      dcvs_v3.dcvsEnable = kDcvsDisable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
          kSleepMinLatency, DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER);
      break;
    case ::qnn::HtpPerformanceMode::kSustainedHighPerformance:
    case ::qnn::HtpPerformanceMode::kHighPerformance:
      dcvs_v3.dcvsEnable = kDcvsDisable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
          kSleepLowLatency, DCVS_VOLTAGE_VCORNER_TURBO,
          DCVS_VOLTAGE_VCORNER_TURBO, DCVS_VOLTAGE_VCORNER_TURBO,
          DCVS_VOLTAGE_VCORNER_TURBO, DCVS_VOLTAGE_VCORNER_TURBO,
          DCVS_VOLTAGE_VCORNER_TURBO);
      break;
    case ::qnn::HtpPerformanceMode::kPowerSaver:
      dcvs_v3.dcvsEnable = kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS);
      break;
    case ::qnn::HtpPerformanceMode::kLowPowerSaver:
      dcvs_v3.dcvsEnable = kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_SVS2,
          DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2,
          DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2,
          DCVS_VOLTAGE_VCORNER_SVS2);
      break;
    case ::qnn::HtpPerformanceMode::kHighPowerSaver:
      dcvs_v3.dcvsEnable = kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
          DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
          DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
          DCVS_VOLTAGE_VCORNER_SVS_PLUS);
      break;
    case ::qnn::HtpPerformanceMode::kLowBalanced:
      dcvs_v3.dcvsEnable = kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_NOM,
          DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM,
          DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM,
          DCVS_VOLTAGE_VCORNER_NOM);
      break;
    case ::qnn::HtpPerformanceMode::kBalanced:
      dcvs_v3.dcvsEnable = kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
          DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
          DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
          DCVS_VOLTAGE_VCORNER_NOM_PLUS);
      break;
    case ::qnn::HtpPerformanceMode::kExtremePowerSaver:
      dcvs_v3.dcvsEnable = kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          kSleepMediumLatency, DCVS_VOLTAGE_CORNER_DISABLE,
          DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE,
          DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE,
          DCVS_VOLTAGE_CORNER_DISABLE);
      break;
    default:
      status = false;
      break;
  }

  return status;
}

std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> SetVotePowerConfig(
    const std::uint32_t power_config_id,
    const ::qnn::HtpPerformanceMode perf_mode,
    const PerformanceModeVoteType vote_type) {
  constexpr const int kNumConfigs = 1;
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> power_configs(
      kNumConfigs);

  QnnHtpPerfInfrastructure_PowerConfig_t& dcvs_config = power_configs[0];

  dcvs_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = dcvs_config.dcvsV3Config;
  dcvs_v3.contextId = power_config_id;

  dcvs_v3.setSleepDisable = 0;  // false
  dcvs_v3.sleepDisable = 0;

  dcvs_v3.setDcvsEnable = 1;  // true

  dcvs_v3.setSleepLatency = 1;  // true

  dcvs_v3.setBusParams = 1;   // true
  dcvs_v3.setCoreParams = 1;  // true

  // Check DownVote before performance mode
  if (vote_type == PerformanceModeVoteType::kDownVote) {
    if (auto status = HandleDownvoteConfig(perf_mode, dcvs_v3); !status) {
      QNN_LOG_ERROR(
          "Invalid performance profile %d to set power configs "
          "during downvote.",
          perf_mode);
    }
    return power_configs;
  }

  // UpVote
  if (auto status = HandleUpvoteConfig(perf_mode, dcvs_v3); !status) {
    QNN_LOG_ERROR(
        "Invalid performance profile %d to set power configs during upvote.",
        perf_mode);
  }
  return power_configs;
}
struct PerfControl::BackendConfig {
  QnnHtpDevice_PerfInfrastructure_t owned_htp_perf_infra_ =
      QNN_HTP_DEVICE_PERF_INFRASTRUCTURE_INIT;
  QnnHtpDevice_PerfInfrastructure_t* htp_perf_infra_{nullptr};
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> perf_power_configs_;
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> down_vote_power_configs_;
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> rpc_power_configs_;
  std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*>
      rpc_power_configs_ptr_;
  std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*>
      perf_power_configs_ptr_;
  std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*>
      down_vote_power_configs_ptr_;
};

PerfControl::PerfControl(const QNN_INTERFACE_VER_TYPE* api,
                         const ::qnn::HtpPerformanceMode htp_performance_mode)
    : api_(api), performance_mode_(htp_performance_mode) {
  backend_config_ = std::make_unique<BackendConfig>();
}

PerfControl::~PerfControl() = default;

bool PerfControl::CreatePerfPowerConfigPtr(
    const std::uint32_t power_config_id,
    const ::qnn::HtpPerformanceMode perf_mode,
    const PerformanceModeVoteType vote_type) {
  if (vote_type == PerformanceModeVoteType::kUpVote) {
    backend_config_->perf_power_configs_ =
        SetVotePowerConfig(power_config_id, perf_mode, vote_type);
    backend_config_->perf_power_configs_ptr_ =
        ObtainNullTermPtrVector(backend_config_->perf_power_configs_);
  } else if (vote_type == PerformanceModeVoteType::kDownVote) {
    // Downvote
    backend_config_->down_vote_power_configs_ =
        SetVotePowerConfig(power_config_id, perf_mode, vote_type);
    backend_config_->down_vote_power_configs_ptr_ =
        ObtainNullTermPtrVector(backend_config_->down_vote_power_configs_);
  } else {
    QNN_LOG_ERROR(
        "Something wrong when creating perf power config pointer "
        "in mode %d during vote type %d",
        perf_mode, vote_type);
    return false;
  }
  return true;
}

void PerfControl::PerformanceVote() {
  if (IsPerfModeEnabled() && manual_voting_type_ != kUpVote) {
    backend_config_->htp_perf_infra_->setPowerConfig(
        powerconfig_client_id_,
        backend_config_->perf_power_configs_ptr_.data());
    manual_voting_type_ = kUpVote;
  }
};

std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> SetRpcPollingPowerConfig(
    ::qnn::HtpPerformanceMode perf_mode) {
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> power_configs;

  QnnHtpPerfInfrastructure_PowerConfig_t rpc_control_latency;
  rpc_control_latency.option =
      QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN;
  QnnHtpPerfInfrastructure_PowerConfig_t rpc_polling_time;
  rpc_polling_time.option =
      QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN;

  switch (perf_mode) {
    case ::qnn::HtpPerformanceMode::kBurst:
    case ::qnn::HtpPerformanceMode::kSustainedHighPerformance:
    case ::qnn::HtpPerformanceMode::kHighPerformance:
      rpc_polling_time.option =
          QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
      rpc_polling_time.rpcPollingTimeConfig = kRpcPollingTimeHighPower;
      power_configs.emplace_back(rpc_polling_time);
      ABSL_FALLTHROUGH_INTENDED;
      // intentionally no break here.
    case ::qnn::HtpPerformanceMode::kPowerSaver:
    case ::qnn::HtpPerformanceMode::kLowPowerSaver:
    case ::qnn::HtpPerformanceMode::kHighPowerSaver:
    case ::qnn::HtpPerformanceMode::kLowBalanced:
    case ::qnn::HtpPerformanceMode::kBalanced:
    case ::qnn::HtpPerformanceMode::kDefault:
    case ::qnn::HtpPerformanceMode::kExtremePowerSaver:
      rpc_control_latency.option =
          QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
      rpc_control_latency.rpcControlLatencyConfig = kRpcControlLatency;
      power_configs.emplace_back(rpc_control_latency);
      break;
    default:
      QNN_LOG_ERROR("Invalid performance profile %d to set power configs",
                    perf_mode);
      break;
  }
  return power_configs;
}

bool PerfControl::Init(const QnnHtpDevice_Arch_t& arch) {
  // Get htp_perf_infra
  backend_config_->htp_perf_infra_ = &backend_config_->owned_htp_perf_infra_;
  if (auto status = GetPerfInfra(api_, backend_config_->htp_perf_infra_);
      !status) {
    return false;
  }
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  error = backend_config_->htp_perf_infra_->createPowerConfigId(
      device_id_, /*core_id=*/0, &powerconfig_client_id_);
  if (error != QNN_SUCCESS) {
    QNN_LOG_ERROR("HTP backend unable to create power config. Error %d",
                  QNN_GET_ERROR_CODE(error));
    return false;
  }

  if (IsPerfModeEnabled()) {
    // Set vector of PowerConfigs and map it to a vector of pointers.
    if (auto status =
            CreatePerfPowerConfigPtr(powerconfig_client_id_, performance_mode_,
                                     PerformanceModeVoteType::kUpVote);
        !status) {
      return false;
    }

    if (auto status =
            CreatePerfPowerConfigPtr(powerconfig_client_id_, performance_mode_,
                                     PerformanceModeVoteType::kDownVote);
        !status) {
      return false;
    }

    // vote immediately, which only take effects in manual mode.
    PerformanceVote();

    // Set Rpc polling mode
    if (arch >= QNN_HTP_DEVICE_ARCH_V69) {
      backend_config_->rpc_power_configs_ =
          SetRpcPollingPowerConfig(performance_mode_);
      backend_config_->rpc_power_configs_ptr_ =
          ObtainNullTermPtrVector(backend_config_->rpc_power_configs_);

      backend_config_->htp_perf_infra_->setPowerConfig(
          powerconfig_client_id_,
          backend_config_->rpc_power_configs_ptr_.data());
    }
  }
  return true;
}

bool PerfControl::Terminate() {
  manual_voting_type_ = kNoVote;
  if (backend_config_->htp_perf_infra_ != nullptr &&
      powerconfig_client_id_ != 0 &&
      !backend_config_->down_vote_power_configs_ptr_.empty()) {
    backend_config_->htp_perf_infra_->setPowerConfig(
        powerconfig_client_id_,
        backend_config_->down_vote_power_configs_ptr_.data());
    backend_config_->htp_perf_infra_->destroyPowerConfigId(
        powerconfig_client_id_);
  } else if (backend_config_->htp_perf_infra_ != nullptr &&
             powerconfig_client_id_ != 0) {
    backend_config_->htp_perf_infra_->destroyPowerConfigId(
        powerconfig_client_id_);
  }

  return true;
}
