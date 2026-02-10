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

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/backend_utils.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "HTP/QnnHtpDevice.h"  // from @qairt
#include "HTP/QnnHtpPerfInfrastructure.h"  // from @qairt
#include "QnnBackend.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace qnn {

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

    if (error = api_->deviceGetInfrastructure(&htp_perf_infra_);
        error != QNN_SUCCESS) {
      QNN_LOG_ERROR(
          "DSP backend unable to create device infrastructure. Error %d",
          QNN_GET_ERROR_CODE(error));
      return false;
    }

    if (htp_perf_infra_->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
      QNN_LOG_ERROR("HTP infra type = %d, which is not perf infra type.",
                    htp_perf_infra_->infraType);
      return false;
    }

    if (error = htp_perf_infra_->perfInfra.createPowerConfigId(
            /*device_id=*/0, /*core_id=*/0, &power_config_id_);
        error != QNN_SUCCESS) {
      QNN_LOG_ERROR("HTP backend unable to create power config. Error %d",
                    QNN_GET_ERROR_CODE(error));
      return false;
    }

    // Initialize power configurations.
    // We need to prepare both:
    // 1. UpVote config: for entering performance mode.
    // 2. DownVote config: for resetting/cleanup.
    InitUpVotePowerConfigs(performance_mode);
    InitDownVotePowerConfigs(performance_mode);

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

 private:
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
      case HtpPerformanceMode::kBalanced:
        SetPowerConfigs(down_vote_power_configs_, power_config_id_,
                        PowerConfig::kDcvsEnable,
                        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
                        PowerConfig::kSleepMaxLatency,
                        DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS,
                        DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS2,
                        DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS);
        break;
      case HtpPerformanceMode::kPowerSaver:
      case HtpPerformanceMode::kLowPowerSaver:
      case HtpPerformanceMode::kHighPowerSaver:
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
      case HtpPerformanceMode::kLowBalanced:
        SetPowerConfigs(down_vote_power_configs_, power_config_id_,
                        PowerConfig::kDcvsEnable,
                        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE,
                        PowerConfig::kSleepMaxLatency,
                        DCVS_VOLTAGE_VCORNER_SVS2, DCVS_VOLTAGE_VCORNER_SVS,
                        DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS2,
                        DCVS_VOLTAGE_VCORNER_SVS, DCVS_VOLTAGE_VCORNER_SVS);
        break;
      case HtpPerformanceMode::kExtremePowerSaver:
        SetPowerConfigs(
            down_vote_power_configs_, power_config_id_,
            PowerConfig::kDcvsEnable,
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE,
            PowerConfig::kSleepMaxLatency, DCVS_VOLTAGE_CORNER_DISABLE,
            DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE,
            DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE,
            DCVS_VOLTAGE_CORNER_DISABLE);
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
  std::vector<const QnnBackend_Config_t*> backend_configs;
  backend_configs.emplace_back(nullptr);

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
#if defined(__x86_64__) || defined(_M_X64)
  if (soc_info.has_value()) {
    QNN_LOG_INFO("Using provided SoC info. SoC name: %s.", soc_info->soc_name);
    soc_info_ = *soc_info;
  }
#else
  if (auto device_platform_info = CreateDevicePlatformInfo();
      device_platform_info) {
    auto soc_model = device_platform_info->v1.hwDevices->v1.deviceInfoExtension
                         ->onChipDevice.socModel;
    auto soc_info_online = FindSocInfo(static_cast<SnapdragonModel>(soc_model));
    soc_info_ = soc_info_online.value_or(kSocInfos[0]);
  }
#endif
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

      htp_perf_control_->UpVote();
    }
  }

  // Follow RAII pattern to manage handles.
  log_handle_ = std::move(local_log_handle);
  backend_handle_ = std::move(local_backend_handle);
  device_handle_ = std::move(local_device_handle);

  return true;
}

}  // namespace qnn
