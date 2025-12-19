// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/htp_backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "HTP/QnnHtpDevice.h"  // from @qairt
#include "QnnBackend.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace qnn {
namespace {
std::optional<::qnn::SocInfo> FindSocInfo(
    const ::qnn::SnapdragonModel& soc_model) {
  for (auto i = 0; i < ::qnn::kNumSocInfos; ++i) {
    if (soc_model == ::qnn::kSocInfos[i].soc_model) {
      return ::qnn::kSocInfos[i];
    }
  }
  QNN_LOG_ERROR("Failed to find available SoC!");
  return std::nullopt;
}

template <typename T>
std::vector<std::add_pointer_t<std::add_const_t<T>>> ObtainNullTermPtrVector(
    const std::vector<T>& vec) {
  std::vector<std::add_pointer_t<std::add_const_t<T>>> ret(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    ret[i] = &(vec[i]);
  }
  ret.emplace_back(nullptr);
  return ret;
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
  dcvs_v3.dcvsEnable = ::qnn::PowerConfig::kDcvsEnable;

  switch (perf_mode) {
    case ::qnn::HtpPerformanceMode::kBurst:
    case ::qnn::HtpPerformanceMode::kSustainedHighPerformance:
    case ::qnn::HtpPerformanceMode::kHighPerformance:
    case ::qnn::HtpPerformanceMode::kBalanced:
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
          ::qnn::PowerConfig::kSleepMaxLatency, DCVS_VOLTAGE_VCORNER_SVS2,
          DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS);
      break;
    case ::qnn::HtpPerformanceMode::kPowerSaver:
    case ::qnn::HtpPerformanceMode::kLowPowerSaver:
    case ::qnn::HtpPerformanceMode::kHighPowerSaver:
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE,
          ::qnn::PowerConfig::kSleepMaxLatency,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER);
      break;
    case ::qnn::HtpPerformanceMode::kLowBalanced:
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE,
          ::qnn::PowerConfig::kSleepMaxLatency, DCVS_VOLTAGE_VCORNER_SVS2,
          DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS);
      break;
    case ::qnn::HtpPerformanceMode::kExtremePowerSaver:
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE,
          ::qnn::PowerConfig::kSleepMaxLatency, DCVS_VOLTAGE_CORNER_DISABLE,
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
      dcvs_v3.dcvsEnable = ::qnn::PowerConfig::kDcvsDisable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
          ::qnn::PowerConfig::kSleepMinLatency,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER);
      break;
    case ::qnn::HtpPerformanceMode::kSustainedHighPerformance:
    case ::qnn::HtpPerformanceMode::kHighPerformance:
      dcvs_v3.dcvsEnable = ::qnn::PowerConfig::kDcvsDisable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
          ::qnn::PowerConfig::kSleepLowLatency, DCVS_VOLTAGE_VCORNER_TURBO,
          DCVS_VOLTAGE_VCORNER_TURBO, DCVS_VOLTAGE_VCORNER_TURBO,
          DCVS_VOLTAGE_VCORNER_TURBO, DCVS_VOLTAGE_VCORNER_TURBO,
          DCVS_VOLTAGE_VCORNER_TURBO);
      break;
    case ::qnn::HtpPerformanceMode::kPowerSaver:
      dcvs_v3.dcvsEnable = ::qnn::PowerConfig::kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          ::qnn::PowerConfig::kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS,
          DCVS_VOLTAGE_VCORNER_SVS);
      break;
    case ::qnn::HtpPerformanceMode::kLowPowerSaver:
      dcvs_v3.dcvsEnable = ::qnn::PowerConfig::kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          ::qnn::PowerConfig::kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_SVS2,
          DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2,
          DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS2,
          DCVS_VOLTAGE_VCORNER_SVS2);
      break;
    case ::qnn::HtpPerformanceMode::kHighPowerSaver:
      dcvs_v3.dcvsEnable = ::qnn::PowerConfig::kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          ::qnn::PowerConfig::kSleepMediumLatency,
          DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
          DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_SVS_PLUS,
          DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_SVS_PLUS);
      break;
    case ::qnn::HtpPerformanceMode::kLowBalanced:
      dcvs_v3.dcvsEnable = ::qnn::PowerConfig::kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          ::qnn::PowerConfig::kSleepMediumLatency, DCVS_VOLTAGE_VCORNER_NOM,
          DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM,
          DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM,
          DCVS_VOLTAGE_VCORNER_NOM);
      break;
    case ::qnn::HtpPerformanceMode::kBalanced:
      dcvs_v3.dcvsEnable = ::qnn::PowerConfig::kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          ::qnn::PowerConfig::kSleepMediumLatency,
          DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
          DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_NOM_PLUS,
          DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_NOM_PLUS);
      break;
    case ::qnn::HtpPerformanceMode::kExtremePowerSaver:
      dcvs_v3.dcvsEnable = ::qnn::PowerConfig::kDcvsEnable;
      status = SetQnnHtpPerfInfrastructure(
          dcvs_v3, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
          ::qnn::PowerConfig::kSleepMediumLatency, DCVS_VOLTAGE_CORNER_DISABLE,
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
      rpc_polling_time.rpcPollingTimeConfig =
          ::qnn::PowerConfig::kRpcPollingTimeHighPower;
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
      rpc_control_latency.rpcControlLatencyConfig =
          ::qnn::PowerConfig::kRpcControlLatency;
      power_configs.emplace_back(rpc_control_latency);
      break;
    default:
      QNN_LOG_ERROR("Invalid performance profile %d to set power configs",
                    perf_mode);
      break;
  }
  return power_configs;
}

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
}  // namespace

struct HtpBackend::BackendConfig {
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

HtpBackend::HtpBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : QnnBackend(qnn_api),
      qnn_device_platform_info_(nullptr, PlatformInfoDeleter{QnnApi()}) {
  backend_config_ = std::make_unique<BackendConfig>();
}

HtpBackend::~HtpBackend() {
  if (IsPerfModeEnabled()) {
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
  }
}

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

bool HtpBackend::Init(const Options& options,
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

  // Starting from QAIRT 2.39, platform information will be available even when
  // this API is called during offline preparation. However, it will always
  // return the default SoC info (SM8350). If user specifies a SoC, we will
  // override the default.
  std::optional<::qnn::SocInfo> soc_info_online;
  auto local_qnn_device_platform_info = CreateDevicePlatformInfo();
  if (local_qnn_device_platform_info) {
    auto online_soc_model = local_qnn_device_platform_info->v1.hwDevices->v1
                                .deviceInfoExtension->onChipDevice.socModel;
    soc_info_online =
        FindSocInfo(static_cast<::qnn::SnapdragonModel>(online_soc_model));
    QNN_LOG_INFO(
        "Succssfully get platform info. SoC model: %d. SoC name: %s.",
        online_soc_model,
        soc_info_online.has_value() ? soc_info_online->soc_name : "NotFound");
  }

#if defined(__ANDROID__)
  if (soc_info_online.has_value()) {
    QNN_LOG_INFO("Using online SoC info. SoC name: %s.",
                 soc_info_online->soc_name);
    soc_info_ = *soc_info_online;
  } else if (soc_info.has_value()) {
    QNN_LOG_INFO("Using provided SoC info. SoC name: %s.", soc_info->soc_name);
    soc_info_ = *soc_info;
  } else {
    QNN_LOG_WARNING("Fail to get SoC info, using default.");
  }
#else
  if (soc_info.has_value()) {
    QNN_LOG_INFO("Using provided SoC info. SoC name: %s.", soc_info->soc_name);
    soc_info_ = *soc_info;
  } else if (soc_info_online.has_value()) {
    QNN_LOG_INFO("Using online SoC info. SoC name: %s.",
                 soc_info_online->soc_name);
    soc_info_ = *soc_info_online;
  } else {
    QNN_LOG_WARNING("Fail to get SoC info, using default.");
  }
#endif

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

#ifdef __ANDROID__
  std::vector<QnnDevice_PlatformInfo_t*> device_platform_infos = {};
#else
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
    QNN_LOG_ERROR("Failed to create device handle!");
    return false;
  }

  // HTP Performance Settings
  performance_mode_ = options.GetHtpPerformanceMode();
  if (IsPerfModeEnabled()) {
    QNN_LOG_INFO("Set HTP performance mode: %d", performance_mode_);

    if (!local_qnn_device_platform_info) {
      QNN_LOG_WARNING(
          "The platforminfo is not available, using default performance mode.");
    } else {
      QnnHtpDevice_Arch_t local_arch =
          local_qnn_device_platform_info->v1.hwDevices->v1.deviceInfoExtension
              ->onChipDevice.arch;

      // Get htp_perf_infra
      backend_config_->htp_perf_infra_ =
          &backend_config_->owned_htp_perf_infra_;
      if (auto status =
              GetPerfInfra(QnnApi(), backend_config_->htp_perf_infra_);
          !status) {
        QNN_LOG_ERROR(
            "Failed to init perf control, using default performance mode.");
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

      // Set vector of PowerConfigs and map it to a vector of pointers.
      if (auto status =
              CreatePerfPowerConfigPtr(PerformanceModeVoteType::kUpVote);
          !status) {
        QNN_LOG_ERROR(
            "Failed to init perf control, using default performance mode.");
        return false;
      }

      if (auto status =
              CreatePerfPowerConfigPtr(PerformanceModeVoteType::kDownVote);
          !status) {
        QNN_LOG_ERROR(
            "Failed to init perf control, using default performance mode.");
        return false;
      }

      // vote immediately, which only take effects in manual mode.
      PerformanceVote();

      // Set Rpc polling mode
      if (local_arch >= QNN_HTP_DEVICE_ARCH_V69) {
        backend_config_->rpc_power_configs_ =
            SetRpcPollingPowerConfig(performance_mode_);
        backend_config_->rpc_power_configs_ptr_ =
            ObtainNullTermPtrVector(backend_config_->rpc_power_configs_);

        backend_config_->htp_perf_infra_->setPowerConfig(
            powerconfig_client_id_,
            backend_config_->rpc_power_configs_ptr_.data());
      }
    }
  }

  // Follow RAII pattern to manage handles
  qnn_device_platform_info_ = std::move(local_qnn_device_platform_info);
  log_handle_ = std::move(local_log_handle);
  backend_handle_ = std::move(local_backend_handle);
  device_handle_ = std::move(local_device_handle);

  return true;
}

void HtpBackend::PerformanceVote() {
  if (manual_voting_type_ != kUpVote) {
    backend_config_->htp_perf_infra_->setPowerConfig(
        powerconfig_client_id_,
        backend_config_->perf_power_configs_ptr_.data());
    manual_voting_type_ = kUpVote;
  }
};

bool HtpBackend::CreatePerfPowerConfigPtr(
    const PerformanceModeVoteType vote_type) {
  if (vote_type == PerformanceModeVoteType::kUpVote) {
    backend_config_->perf_power_configs_ = SetVotePowerConfig(
        powerconfig_client_id_, performance_mode_, vote_type);
    backend_config_->perf_power_configs_ptr_ =
        ObtainNullTermPtrVector(backend_config_->perf_power_configs_);
  } else if (vote_type == PerformanceModeVoteType::kDownVote) {
    // Downvote
    backend_config_->down_vote_power_configs_ = SetVotePowerConfig(
        powerconfig_client_id_, performance_mode_, vote_type);
    backend_config_->down_vote_power_configs_ptr_ =
        ObtainNullTermPtrVector(backend_config_->down_vote_power_configs_);
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
