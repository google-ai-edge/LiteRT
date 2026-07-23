// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/htp_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "HTP/QnnHtpDevice.h"  // from @qairt
#include "HTP/QnnHtpDeviceConfigShared.h"  // from @qairt
#include "HTP/QnnHtpGraph.h"  // from @qairt
#include "HTP/QnnHtpPerfInfrastructure.h"  // from @qairt
#include "QnnBackend.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/backend_utils.h"
#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"

namespace qnn {

namespace {

float GetOptimizationValue(OptimizationLevel level) {
  // Default optimization level value is 2
  switch (level) {
    case OptimizationLevel::kHtpOptimizeForInference:
      return 2.0f;
    case OptimizationLevel::kHtpOptimizeForPrepare:
      return 1.0f;
    case OptimizationLevel::kHtpOptimizeForInferenceO3:
      return 3.0f;
    default:
      return 2.0f;
  }
}

Qnn_Priority_t GetGraphPriorityValue(GraphPriority graph_priority) {
  // Default priority is NORMAL
  switch (graph_priority) {
    case GraphPriority::kDefault:
      return QNN_PRIORITY_DEFAULT;
    case GraphPriority::kLow:
      return QNN_PRIORITY_LOW;
    case GraphPriority::kNormal:
      return QNN_PRIORITY_NORMAL;
    case GraphPriority::kNormalHigh:
      return QNN_PRIORITY_NORMAL_HIGH;
    case GraphPriority::kHigh:
      return QNN_PRIORITY_HIGH;
    default:
      return QNN_PRIORITY_UNDEFINED;
  }
}

}  // namespace

// HTP PERF CONTROL /////////////////////////////////////////////////////////
class HtpBackend::HtpPerfControl {
 public:
  explicit HtpPerfControl(const QNN_INTERFACE_VER_TYPE* api) : api_(api) {}

  ~HtpPerfControl() {
    DownVote();
    if (htp_perf_infra_ != nullptr && power_config_id_ != 0) {
      htp_perf_infra_->perfInfra.destroyPowerConfigId(power_config_id_);
    }
  }

  bool Init(HtpPerformanceMode performance_mode) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    // Acquire infra & power-config id once, then reuse across mode changes.
    if (htp_perf_infra_ == nullptr) {
      if (error = api_->deviceGetInfrastructure(&htp_perf_infra_);
          error != QNN_SUCCESS) {
        QNN_LOG_ERROR(
            "HTP backend unable to create device infrastructure. Error %d",
            QNN_GET_ERROR_CODE(error));
        return false;
      }

      if (htp_perf_infra_ == nullptr) {
        QNN_LOG_ERROR(
            "HTP backend failed to create device infrastructure but reported "
            "no error.");
        return false;
      }

      if (htp_perf_infra_->infraType !=
          QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
        QNN_LOG_ERROR("HTP infra type = %d, which is not perf infra type.",
                      htp_perf_infra_->infraType);
        return false;
      }
    }

    if (power_config_id_ == 0) {
      if (error = htp_perf_infra_->perfInfra.createPowerConfigId(
              /*device_id=*/0, /*core_id=*/0, &power_config_id_);
          error != QNN_SUCCESS) {
        QNN_LOG_ERROR("HTP backend unable to create power config. Error %d",
                      QNN_GET_ERROR_CODE(error));
        return false;
      }
    }

    // Rebuild the power configurations for the requested mode. Both are needed:
    // 1. UpVote config: for entering performance mode.
    // 2. DownVote config: for resetting/cleanup.
    InitUpVotePowerConfigs(performance_mode);
    InitDownVotePowerConfigs(performance_mode);
    current_mode_ = performance_mode;

    return true;
  }

  bool SetRpcPolling(HtpPerformanceMode performance_mode) {
    if (!InitRpcPollingPowerConfig(performance_mode)) {
      QNN_LOG_ERROR("Failed to init RPC polling power config.");
      return false;
    }

    if (htp_perf_infra_) {
      htp_perf_infra_->perfInfra.setPowerConfig(power_config_id_,
                                                rpc_power_configs_ptr_.data());
    }

    return true;
  }

  void UpVote() {
    if (htp_perf_infra_) {
      htp_perf_infra_->perfInfra.setPowerConfig(
          power_config_id_, up_vote_power_configs_ptr_.data());
    }
  }

  void DownVote() {
    if (htp_perf_infra_) {
      htp_perf_infra_->perfInfra.setPowerConfig(
          power_config_id_, down_vote_power_configs_ptr_.data());
    }
  }

  void ScheduleUpVote() {
    EnsureVotingThread();
    voting_thread_->Enqueue(VotingThread::VoteType::kUpVote);
  }

  // Debounce the downvote only for burst/sustained modes to avoid thrashing
  // the high-perf vote between back-to-back inferences.
  void ScheduleDownVote() {
    EnsureVotingThread();
    const bool debounce =
        current_mode_ == HtpPerformanceMode::kBurst ||
        current_mode_ == HtpPerformanceMode::kSustainedHighPerformance;
    voting_thread_->Enqueue(VotingThread::VoteType::kDownVote, debounce);
  }

  bool ReinitIfNeeded(HtpPerformanceMode new_mode) {
    const bool needs_init =
        htp_perf_infra_ == nullptr || new_mode != current_mode_;
    if (needs_init && !Init(new_mode)) {
      QNN_LOG_ERROR("HTP backend failed to re-init for performance mode %d.",
                    new_mode);
      return false;
    }
    ScheduleUpVote();
    return true;
  }

  // Applies new_mode for one inference. Manual skips a same-mode re-vote
  // when init already upvoted, auto always re-votes.
  bool ApplyPerfMode(HtpPerformanceMode new_mode, HtpPerfCtrlMode ctrl_mode,
                     bool supports_rpc_polling) {
    const bool mode_changed = new_mode != current_mode_;
    if (!mode_changed && ctrl_mode == HtpPerfCtrlMode::kManual) {
      return true;
    }
    if (!ReinitIfNeeded(new_mode)) {
      return false;
    }
    if (mode_changed && supports_rpc_polling && !SetRpcPolling(new_mode)) {
      QNN_LOG_ERROR("Failed to set RPC Polling in ApplyPerfMode.");
      return false;
    }
    return true;
  }

 private:
  void EnsureVotingThread() {
    if (!voting_thread_) {
      voting_thread_ =
          std::make_unique<VotingThread>([this](VotingThread::VoteType v) {
            v == VotingThread::VoteType::kUpVote ? UpVote() : DownVote();
          });
    }
  }

  static constexpr size_t kNumPowerConfigs = 1;
  static constexpr size_t kNumRpcPollingPowerConfigs = 2;

  void SetPowerConfigs(
      std::array<QnnHtpPerfInfrastructure_PowerConfig_t, kNumPowerConfigs>&
          power_configs,
      uint32_t context_id, uint32_t dcvs_enable,
      QnnHtpPerfInfrastructure_PowerMode_t power_mode, uint32_t sleep_latency,
      QnnHtpPerfInfrastructure_VoltageCorner_t bus_voltage_corner_min,
      QnnHtpPerfInfrastructure_VoltageCorner_t bus_voltage_corner_max,
      QnnHtpPerfInfrastructure_VoltageCorner_t bus_voltage_corner_target,
      QnnHtpPerfInfrastructure_VoltageCorner_t core_voltage_corner_min,
      QnnHtpPerfInfrastructure_VoltageCorner_t core_voltage_corner_max,
      QnnHtpPerfInfrastructure_VoltageCorner_t core_voltage_corner_target) {
    QnnHtpPerfInfrastructure_PowerConfig_t& dcvs_config = power_configs[0];
    dcvs_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = dcvs_config.dcvsV3Config;

    dcvs_v3.contextId = context_id;
    dcvs_v3.powerMode = power_mode;

    dcvs_v3.setDcvsEnable = 1;
    dcvs_v3.dcvsEnable = dcvs_enable;

    dcvs_v3.setSleepDisable = 0;
    dcvs_v3.sleepDisable = 0;
    dcvs_v3.setSleepLatency = 1;
    dcvs_v3.sleepLatency = sleep_latency;

    dcvs_v3.setBusParams = 1;
    dcvs_v3.busVoltageCornerMin = bus_voltage_corner_min;
    dcvs_v3.busVoltageCornerTarget = bus_voltage_corner_target;
    dcvs_v3.busVoltageCornerMax = bus_voltage_corner_max;

    dcvs_v3.setCoreParams = 1;
    dcvs_v3.coreVoltageCornerMin = core_voltage_corner_min;
    dcvs_v3.coreVoltageCornerTarget = core_voltage_corner_target;
    dcvs_v3.coreVoltageCornerMax = core_voltage_corner_max;
  }
  void InitUpVotePowerConfigs(HtpPerformanceMode performance_mode) {
    switch (performance_mode) {
      case HtpPerformanceMode::kBurst:
        SetPowerConfigs(up_vote_power_configs_, power_config_id_,
                        PowerConfig::kDcvsDisable,
                        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
                        PowerConfig::kSleepMinLatency,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER);
        break;
      case HtpPerformanceMode::kSustainedHighPerformance:
      case HtpPerformanceMode::kHighPerformance:
        SetPowerConfigs(up_vote_power_configs_, power_config_id_,
                        PowerConfig::kDcvsDisable,
                        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
                        PowerConfig::kSleepLowLatency,
                        DCVS_VOLTAGE_VCORNER_TURBO, DCVS_VOLTAGE_VCORNER_TURBO,
                        DCVS_VOLTAGE_VCORNER_TURBO, DCVS_VOLTAGE_VCORNER_TURBO,
                        DCVS_VOLTAGE_VCORNER_TURBO, DCVS_VOLTAGE_VCORNER_TURBO);
        break;
      case HtpPerformanceMode::kPowerSaver:
        SetPowerConfigs(up_vote_power_configs_, power_config_id_,
                        PowerConfig::kDcvsEnable,
                        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
                        PowerConfig::kSleepMediumLatency,
                        DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
                        DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
                        DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS);
        break;
      case HtpPerformanceMode::kLowPowerSaver:
        SetPowerConfigs(up_vote_power_configs_, power_config_id_,
                        PowerConfig::kDcvsEnable,
                        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
                        PowerConfig::kSleepMediumLatency,
                        DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2,
                        DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2,
                        DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2);
        break;
      case HtpPerformanceMode::kHighPowerSaver:
        SetPowerConfigs(
            up_vote_power_configs_, power_config_id_, PowerConfig::kDcvsEnable,
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
            PowerConfig::kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
            DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
            DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
            DCVS_VOLTAGE_VCORNER_SVS_PLUS);
        break;
      case HtpPerformanceMode::kLowBalanced:
        SetPowerConfigs(up_vote_power_configs_, power_config_id_,
                        PowerConfig::kDcvsEnable,
                        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
                        PowerConfig::kSleepMediumLatency,
                        DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM,
                        DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM,
                        DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM);
        break;
      case HtpPerformanceMode::kBalanced:
        SetPowerConfigs(
            up_vote_power_configs_, power_config_id_, PowerConfig::kDcvsEnable,
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
            PowerConfig::kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
            DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
            DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
            DCVS_VOLTAGE_VCORNER_NOM_PLUS);
        break;
      case HtpPerformanceMode::kExtremePowerSaver:
        SetPowerConfigs(
            up_vote_power_configs_, power_config_id_, PowerConfig::kDcvsEnable,
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
            PowerConfig::kSleepMediumLatency, DCVS_VOLTAGE_CORNER_DISABLE,
            DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE,
            DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE,
            DCVS_VOLTAGE_CORNER_DISABLE);
        break;
      default:
        QNN_LOG_ERROR(
            "Invalid performance profile %d to set power configs during "
            "upvote.",
            performance_mode);
        break;
    }
    SetNullTermPtrArray(absl::MakeConstSpan(up_vote_power_configs_),
                        up_vote_power_configs_ptr_);
  }

  void InitDownVotePowerConfigs(HtpPerformanceMode performance_mode) {
    switch (performance_mode) {
      case HtpPerformanceMode::kBurst:
      case HtpPerformanceMode::kSustainedHighPerformance:
      case HtpPerformanceMode::kHighPerformance:
        SetPowerConfigs(down_vote_power_configs_, power_config_id_,
                        PowerConfig::kDcvsEnable,
                        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
                        PowerConfig::kSleepHighLatency,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER);
        break;
      case HtpPerformanceMode::kBalanced:
        SetPowerConfigs(down_vote_power_configs_, power_config_id_,
                        PowerConfig::kDcvsEnable,
                        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE,
                        PowerConfig::kSleepHighLatency,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER);
        break;
      case HtpPerformanceMode::kPowerSaver:
      case HtpPerformanceMode::kLowPowerSaver:
      case HtpPerformanceMode::kHighPowerSaver:
      case HtpPerformanceMode::kLowBalanced:
      case HtpPerformanceMode::kExtremePowerSaver:
        SetPowerConfigs(down_vote_power_configs_, power_config_id_,
                        PowerConfig::kDcvsEnable,
                        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE,
                        PowerConfig::kSleepMaxLatency,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
                        DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER);
        break;
      default:
        QNN_LOG_ERROR(
            "Invalid performance profile %d to set power configs "
            "during downvote.",
            performance_mode);
        break;
    }
    SetNullTermPtrArray(absl::MakeConstSpan(down_vote_power_configs_),
                        down_vote_power_configs_ptr_);
  }

  bool InitRpcPollingPowerConfig(HtpPerformanceMode performance_mode) {
    rpc_power_configs_ = {
        {{QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN},
         {QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN}}};
    size_t config_count = 0;

    QnnHtpPerfInfrastructure_PowerConfig_t rpc_control_latency;
    rpc_control_latency.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN;
    QnnHtpPerfInfrastructure_PowerConfig_t rpc_polling_time;
    rpc_polling_time.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN;

    switch (performance_mode) {
      case HtpPerformanceMode::kBurst:
      case HtpPerformanceMode::kSustainedHighPerformance:
      case HtpPerformanceMode::kHighPerformance:
        rpc_polling_time.option =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        rpc_polling_time.rpcPollingTimeConfig =
            PowerConfig::kRpcPollingTimeHighPower;
        rpc_power_configs_[config_count++] = rpc_polling_time;
        [[fallthrough]];
      case HtpPerformanceMode::kPowerSaver:
      case HtpPerformanceMode::kLowPowerSaver:
      case HtpPerformanceMode::kHighPowerSaver:
      case HtpPerformanceMode::kLowBalanced:
      case HtpPerformanceMode::kBalanced:
      case HtpPerformanceMode::kDefault:
      case HtpPerformanceMode::kExtremePowerSaver:
        rpc_control_latency.option =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
        rpc_control_latency.rpcControlLatencyConfig =
            PowerConfig::kRpcControlLatency;
        rpc_power_configs_[config_count++] = rpc_control_latency;
        break;
      default:
        QNN_LOG_ERROR("Invalid performance profile %d to set power configs",
                      performance_mode);
        return false;
    }
    SetNullTermPtrArray(
        absl::MakeConstSpan(rpc_power_configs_.data(), config_count),
        rpc_power_configs_ptr_);

    return true;
  }

  // Performance control
  const QNN_INTERFACE_VER_TYPE* api_{nullptr};
  std::uint32_t power_config_id_{0};
  QnnDevice_Infrastructure_t htp_perf_infra_{nullptr};
  // Last successfully-applied mode, used to skip a redundant re-vote.
  HtpPerformanceMode current_mode_{HtpPerformanceMode::kDefault};
  std::array<QnnHtpPerfInfrastructure_PowerConfig_t, kNumPowerConfigs>
      up_vote_power_configs_;
  std::array<QnnHtpPerfInfrastructure_PowerConfig_t, kNumPowerConfigs>
      down_vote_power_configs_;
  std::array<QnnHtpPerfInfrastructure_PowerConfig_t, kNumRpcPollingPowerConfigs>
      rpc_power_configs_;
  std::array<const QnnHtpPerfInfrastructure_PowerConfig_t*,
             kNumRpcPollingPowerConfigs + 1>
      rpc_power_configs_ptr_;
  std::array<const QnnHtpPerfInfrastructure_PowerConfig_t*,
             kNumPowerConfigs + 1>
      up_vote_power_configs_ptr_;
  std::array<const QnnHtpPerfInfrastructure_PowerConfig_t*,
             kNumPowerConfigs + 1>
      down_vote_power_configs_ptr_;
  // Declared last — destroyed first, before power-config arrays are freed.
  std::unique_ptr<VotingThread> voting_thread_;
};

// HTP BACKEND /////////////////////////////////////////////////////////
HtpBackend::HtpBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : QnnBackend(qnn_api) {}

HtpBackend::~HtpBackend() = default;

HtpBackend::QnnDevicePlatformInfo HtpBackend::CreateDevicePlatformInfo() {
  const QnnDevice_PlatformInfo_t* local_qnn_device_platform_info = nullptr;
  auto error =
      QnnApi()->deviceGetPlatformInfo(nullptr, &local_qnn_device_platform_info);
  if (error != QNN_SUCCESS) {
    QNN_LOG_ERROR("failed to get device platform info, %d", error);
    return QnnDevicePlatformInfo{nullptr, PlatformInfoDeleter{QnnApi()}};
  }
  return QnnDevicePlatformInfo{local_qnn_device_platform_info,
                               PlatformInfoDeleter{QnnApi()}};
}

bool HtpBackend::Init(const Options& options, std::optional<SocInfo> soc_info) {
  // Log Handle
  auto local_log_handle = CreateLogHandle(options.GetLogLevel());
  if (!local_log_handle && options.GetLogLevel() != LogLevel::kOff) {
    QNN_LOG_ERROR("Failed to create log handle.");
    return false;
  }

  // Backend Handle
  std::array<const QnnBackend_Config_t*, 1> backend_configs = {nullptr};

  auto local_backend_handle = CreateBackendHandle(
      local_log_handle.get(), absl::MakeSpan(backend_configs));
  if (!local_backend_handle) {
    QNN_LOG_ERROR("Failed to create backend handle.");
    return false;
  }

  // Starting from QAIRT 2.39, platform information will be available even when
  // this API is called during offline preparation. However, it will always
  // return the default SoC info (SM8350). If user specifies a SoC, we will
  // override the default.
  if (soc_info.has_value()) {
    QNN_LOG_INFO("Using provided SoC info. SoC name: %s.", soc_info->soc_name);
    soc_info_ = *soc_info;
  } else {
#if defined(__x86_64__) || defined(_M_X64)
    // Offline compilation on desktop hosts cannot query the target device.
#else
    if (auto device_platform_info = CreateDevicePlatformInfo();
        device_platform_info) {
      auto soc_model = device_platform_info->v1.hwDevices->v1
                           .deviceInfoExtension->onChipDevice.socModel;
      auto soc_info_online =
          FindSocInfo(static_cast<SnapdragonModel>(soc_model));
      soc_info_ = soc_info_online.value_or(kSocInfos[0]);
    }
#if defined(_WIN32) && defined(_M_ARM64)
    if (soc_info_.dsp_arch == DspArch::NONE) {
      QNN_LOG_WARNING(
          "Unable to map Windows ARM64 QNN platform info; using SC8380XP "
          "fallback for Snapdragon X Elite.");
      soc_info_ = FindSocInfo(SnapdragonModel::SC8380XP).value_or(kSocInfos[0]);
    }
#endif
#endif
  }
  if (soc_info_.dsp_arch == DspArch::NONE) {
    QNN_LOG_ERROR("SoC info was not configured successfully.")
    return false;
  }
  QNN_LOG_INFO("Initializing QNN backend for SoC model: %s",
               soc_info_.soc_name);

  // Device Handle
  std::vector<QnnDevice_CustomConfig_t> device_custom_configs;
  QnnHtpDevice_CustomConfig_t* htp_device_custom_config =
      &AllocateHtpDeviceConfig();
  htp_device_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
  htp_device_custom_config->socModel =
      static_cast<uint32_t>(soc_info_.soc_model);
  device_custom_configs.emplace_back(
      static_cast<QnnDevice_CustomConfig_t>(htp_device_custom_config));

#if defined(__x86_64__) || defined(_M_X64)
  std::vector<QnnDevice_PlatformInfo_t*> device_platform_infos;

  QnnDevice_PlatformInfo_t* device_platform_info =
      &AllocateDevicePlatformInfo();
  device_platform_info->version = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
  device_platform_info->v1.numHwDevices = 1;

  QnnDevice_HardwareDeviceInfo_t* hardware_device_info =
      &AllocateDeviceHardwareInfo();
  hardware_device_info->version = QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1;
  hardware_device_info->v1.deviceId = 0;
  hardware_device_info->v1.deviceType = 0;
  hardware_device_info->v1.numCores = 1;

  QnnHtpDevice_DeviceInfoExtension_t* htp_device_info_extension =
      &AllocHtpDeviceInfoExtension();
  htp_device_info_extension->devType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
  htp_device_info_extension->onChipDevice.vtcmSize = soc_info_.vtcm_size_in_mb;
  // TODO(jiunkaiy): Given by user, default value is unsigned pd
  htp_device_info_extension->onChipDevice.signedPdSupport = false;
  htp_device_info_extension->onChipDevice.socModel =
      static_cast<uint32_t>(soc_info_.soc_model);
  htp_device_info_extension->onChipDevice.arch =
      static_cast<QnnHtpDevice_Arch_t>(soc_info_.dsp_arch);
  // TODO(jiunkaiy): For Htp, dlbcSupport is true
  htp_device_info_extension->onChipDevice.dlbcSupport = true;
  hardware_device_info->v1.deviceInfoExtension = htp_device_info_extension;

  QnnDevice_CoreInfo_t* device_core_info = &AllocateDeviceCoreInfo();
  device_core_info->version = QNN_DEVICE_CORE_INFO_VERSION_1;
  device_core_info->v1.coreId = 0;
  device_core_info->v1.coreType = 0;
  device_core_info->v1.coreInfoExtension = nullptr;
  hardware_device_info->v1.cores = device_core_info;

  device_platform_info->v1.hwDevices = hardware_device_info;
  device_platform_infos.emplace_back(device_platform_info);
#else
  std::vector<QnnDevice_PlatformInfo_t*> device_platform_infos = {};
#endif

  std::vector<const QnnDevice_Config_t*> device_configs;
  uint32_t num_custom_configs =
      device_platform_infos.size() + device_custom_configs.size();
  // +1 for null terminated
  device_configs.reserve(num_custom_configs + 1);
  for (std::size_t i = 0; i < device_custom_configs.size(); ++i) {
    QnnDevice_Config_t* device_custom_config = &AllocateDeviceConfig();
    device_custom_config->option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    device_custom_config->customConfig = device_custom_configs[i];
    device_configs.emplace_back(device_custom_config);
  }
  for (std::size_t i = 0; i < device_platform_infos.size(); ++i) {
    QnnDevice_Config_t* device_custom_config = &AllocateDeviceConfig();
    device_custom_config->option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
    device_custom_config->hardwareInfo = device_platform_infos[i];
    device_configs.emplace_back(device_custom_config);
  }
  // null terminated
  device_configs.emplace_back(nullptr);

  auto local_device_handle = CreateDeviceHandle(local_log_handle.get(),
                                                absl::MakeSpan(device_configs));
  if (!local_device_handle) {
    QNN_LOG_ERROR("Failed to create device handle.");
    return false;
  }

  // HTP Performance Settings
  HtpPerformanceMode performance_mode = options.GetHtpPerformanceMode();
  if (performance_mode != HtpPerformanceMode::kDefault) {
    QNN_LOG_INFO("Set HTP performance mode: %d", performance_mode);

    htp_perf_control_ = std::make_unique<HtpPerfControl>(QnnApi());
    if (!htp_perf_control_->Init(performance_mode)) {
      QNN_LOG_ERROR(
          "Failed to initialize HTP performance Control, using default "
          "performance mode.");
      return false;
    } else {
      if (soc_info_.dsp_arch >= DspArch::V69) {
        if (!htp_perf_control_->SetRpcPolling(performance_mode)) {
          QNN_LOG_ERROR("Failed to initialize HTP RPC polling.");
          return false;
        }
      }

      // Manual mode upvotes at init. Auto defers the upvote to Execute().
      if (options.GetHtpPerfCtrlMode() == HtpPerfCtrlMode::kManual) {
        htp_perf_control_->UpVote();
      }
    }
  }

  // Follow RAII pattern to manage handles.
  log_handle_ = std::move(local_log_handle);
  backend_handle_ = std::move(local_backend_handle);
  device_handle_ = std::move(local_device_handle);

  return true;
}

bool HtpBackend::SetPerformanceMode(const Options& options) {
  HtpPerformanceMode performance_mode = options.GetHtpPerformanceMode();

  if (performance_mode == HtpPerformanceMode::kDefault) {
    if (htp_perf_control_) {
      htp_perf_control_->ScheduleDownVote();
    }
    return true;
  }

  if (!htp_perf_control_) {
    QNN_LOG_ERROR(
        "HTP performance control is not initialized in SetPerformanceMode.");
    return false;
  }

  const bool supports_rpc_polling = soc_info_.dsp_arch >= DspArch::V69;
  if (!htp_perf_control_->ApplyPerfMode(performance_mode,
                                        options.GetHtpPerfCtrlMode(),
                                        supports_rpc_polling)) {
    QNN_LOG_ERROR("Failed to set HTP performance mode in SetPerformanceMode");
    return false;
  }

  return true;
}

GraphConfigBuilder HtpBackend::BuildGraphConfigs(
    const Options& options, absl::string_view /*qnn_graph_name*/) {
  const bool fp16_supported = IsFp16Supported(soc_info_);

  GraphConfigBuilder config_builder;

  if (fp16_supported) {
    // QNN suggest always enable relax precision.
    QnnHtpGraph_CustomConfig_t precision = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    precision.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    precision.precision = QNN_PRECISION_FLOAT16;
    config_builder.AddCustomConfig(precision);
  }

  // Default use O3 for now.
  QnnHtpGraph_CustomConfig_t optimization = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  optimization.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  optimization.optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  optimization.optimizationOption.floatValue =
      GetOptimizationValue(options.GetOptimizationLevel());
  config_builder.AddCustomConfig(optimization);

  // VTCM — default value is 0 which means the MAX value.
  QnnHtpGraph_CustomConfig_t vtcm = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  vtcm.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  vtcm.vtcmSizeInMB = options.GetVtcmSize();
  config_builder.AddCustomConfig(vtcm);

  // FoldRelu Off
  QnnHtpGraph_CustomConfig_t fold_relu = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  fold_relu.option =
      QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF;
  fold_relu.foldReluActivationIntoConvOff = !options.GetUseFoldReLU();
  config_builder.AddCustomConfig(fold_relu);

  // ConvHMX Off
  QnnHtpGraph_CustomConfig_t conv_hmx = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  conv_hmx.option = QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF;
  conv_hmx.shortDepthConvOnHmxOff = !options.GetUseConvHMX();
  config_builder.AddCustomConfig(conv_hmx);

  if (fp16_supported) {
    // TODO: Need to verify if legacy SoCs support P point as well.
    const std::int32_t htp_p_point = options.GetHtpPPoint();
    if (htp_p_point > 0) {
      QnnHtpGraph_CustomConfig_t p_point = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
      p_point.option = QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG;
      p_point.finalizeConfig.key = "P";
      p_point.finalizeConfig.value = {QNN_DATATYPE_INT_32,
                                      {.int32Value = htp_p_point}};
      config_builder.AddCustomConfig(p_point);
    } else if (htp_p_point < 0) {
      QNN_LOG_WARNING(
          "Invalid P point (%d): negative values not supported, skipping "
          "P point config.",
          htp_p_point);
    }
  }

  // Hvx Thread
  if (const std::uint32_t num_hvx_threads = options.GetNumHvxThreads();
      num_hvx_threads > 0) {
    QnnHtpGraph_CustomConfig_t hvx_threads = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    hvx_threads.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
    hvx_threads.numHvxThreads = num_hvx_threads;
    config_builder.AddCustomConfig(hvx_threads);
  }

  // DLBC (activations / inputs). Offline-prep only.
  if (options.GetHtpDlbc()) {
    QnnHtpGraph_CustomConfig_t dlbc = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    dlbc.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    dlbc.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
    dlbc.optimizationOption.floatValue = 1.0f;
    config_builder.AddCustomConfig(dlbc);
  }

  // DLBC weights. Offline-prep only.
  if (options.GetHtpDlbcWeights()) {
    QnnHtpGraph_CustomConfig_t dlbc_weights = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    dlbc_weights.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    dlbc_weights.optimizationOption.type =
        QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC_WEIGHTS;
    dlbc_weights.optimizationOption.floatValue = 1.0f;
    config_builder.AddCustomConfig(dlbc_weights);
  }

  // Graph Priority
  QnnGraph_Config_t priority = QNN_GRAPH_CONFIG_INIT;
  priority.option = QNN_GRAPH_CONFIG_OPTION_PRIORITY;
  priority.priority = GetGraphPriorityValue(options.GetGraphPriority());
  config_builder.AddGraphConfig(priority);

  return config_builder;
}

}  // namespace qnn
