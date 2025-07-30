// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/htp_backend.h"

namespace qnn {

namespace {
std::optional<::qnn::SocInfo> FindSocInfo(
    const ::qnn::SnapdragonModel &soc_model) {
  for (auto i = 0; i < ::qnn::kNumSocInfos; ++i) {
    if (soc_model == ::qnn::kSocInfos[i].soc_model) {
      return ::qnn::kSocInfos[i];
    }
  }
  QNN_LOG_ERROR("Failed to find available SoC!");
  return std::nullopt;
}
}  // namespace

HtpBackend::HtpBackend(const QNN_INTERFACE_VER_TYPE *qnn_api)
    : QnnBackend(qnn_api) {}

HtpBackend::~HtpBackend() {}

bool HtpBackend::Init(const Options &options,
                      std::optional<::qnn::SocInfo> soc_info) {
  // Log Handle
  auto local_log_handle = CreateLogHandle(options.GetLogLevel());
  if (!local_log_handle) {
    QNN_LOG_ERROR("Failed to create log handle!");
    return false;
  }

  // Backend Handle
  std::vector<const QnnBackend_Config_t *> backend_configs;
  backend_configs.emplace_back(nullptr);

  auto local_backend_handle = CreateBackendHandle(
      local_log_handle.get(), absl::MakeSpan(backend_configs));
  if (!local_backend_handle) {
    QNN_LOG_ERROR("Failed to create backend handle!");
    return false;
  }

  // Soc Info
  auto soc_info_ = ::qnn::kSocInfos[7];  // V75
  const QnnDevice_PlatformInfo_t *local_device_platform_info = nullptr;

  if (soc_info.has_value()) {
    QNN_LOG_INFO("Using provided SoC info.");
    soc_info_ = *soc_info;
  } else {
    QNN_LOG_INFO("Apply deviceGetPlatformInfo for SoC info.");
    if (auto status = QnnApi()->deviceGetPlatformInfo(
            nullptr, &local_device_platform_info);
        status == QNN_SUCCESS) {
      auto soc_info_online = FindSocInfo(static_cast<::qnn::SnapdragonModel>(
          local_device_platform_info->v1.hwDevices->v1.deviceInfoExtension
              ->onChipDevice.socModel));

      if (soc_info_online.has_value()) {
        soc_info_ = *soc_info_online;
      }

    } else {
      QNN_LOG_WARNING("Fail to get platforminfo: %d, using default.", status);
    }
  }

  QNN_LOG_INFO("Initializing QNN backend for SoC model: %s",
               soc_info_.soc_name);

  // Device Handle
  std::vector<QnnDevice_CustomConfig_t> device_custom_configs;
  QnnHtpDevice_CustomConfig_t *htp_device_custom_config =
      &AllocateHtpDeviceConfig();
  htp_device_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
  htp_device_custom_config->socModel =
      static_cast<uint32_t>(soc_info_.soc_model);
  device_custom_configs.emplace_back(
      static_cast<QnnDevice_CustomConfig_t>(htp_device_custom_config));

#ifdef __ANDROID__
  std::vector<QnnDevice_PlatformInfo_t *> device_platform_infos = {};
#else
  std::vector<QnnDevice_PlatformInfo_t *> device_platform_infos;

  QnnDevice_PlatformInfo_t *device_platform_info =
      &AllocateDevicePlatformInfo();
  device_platform_info->version = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
  device_platform_info->v1.numHwDevices = 1;

  QnnDevice_HardwareDeviceInfo_t *hardware_device_info =
      &AllocateDeviceHardwareInfo();
  hardware_device_info->version = QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1;
  hardware_device_info->v1.deviceId = 0;
  hardware_device_info->v1.deviceType = 0;
  hardware_device_info->v1.numCores = 1;

  QnnHtpDevice_DeviceInfoExtension_t *htp_device_info_extension =
      &AllocHtpDeviceInfoExtension();
  htp_device_info_extension->devType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
  htp_device_info_extension->onChipDevice.vtcmSize = soc_info->vtcm_size_in_mb;
  // TODO(jiunkaiy): Given by user, default value is unsigned pd
  htp_device_info_extension->onChipDevice.signedPdSupport = 0;
  htp_device_info_extension->onChipDevice.socModel =
      static_cast<uint32_t>(soc_info->soc_model);
  htp_device_info_extension->onChipDevice.arch =
      static_cast<QnnHtpDevice_Arch_t>(soc_info->dsp_arch);
  // TODO(jiunkaiy): For Htp, dlbcSupport is true
  htp_device_info_extension->onChipDevice.dlbcSupport = true;
  hardware_device_info->v1.deviceInfoExtension = htp_device_info_extension;

  QnnDevice_CoreInfo_t *device_core_info = &AllocateDeviceCoreInfo();
  device_core_info->version = QNN_DEVICE_CORE_INFO_VERSION_1;
  device_core_info->v1.coreId = 0;
  device_core_info->v1.coreType = 0;
  device_core_info->v1.coreInfoExtension = nullptr;
  hardware_device_info->v1.cores = device_core_info;

  device_platform_info->v1.hwDevices = hardware_device_info;
  device_platform_infos.emplace_back(device_platform_info);
#endif

  std::vector<const QnnDevice_Config_t *> device_configs;
  uint32_t num_custom_configs =
      device_platform_infos.size() + device_custom_configs.size();
  // +1 for null terminated
  device_configs.reserve(num_custom_configs + 1);
  for (std::size_t i = 0; i < device_custom_configs.size(); ++i) {
    QnnDevice_Config_t *device_custom_config = &AllocateDeviceConfig();
    device_custom_config->option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    device_custom_config->customConfig = device_custom_configs[i];
    device_configs.emplace_back(device_custom_config);
  }
  for (std::size_t i = 0; i < device_platform_infos.size(); ++i) {
    QnnDevice_Config_t *device_custom_config = &AllocateDeviceConfig();
    device_custom_config->option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
    device_custom_config->hardwareInfo = device_platform_infos[i];
    device_configs.emplace_back(device_custom_config);
  }
  // null terminatedD
  device_configs.emplace_back(nullptr);

  auto local_device_handle = CreateDeviceHandle(local_log_handle.get(),
                                                absl::MakeSpan(device_configs));
  if (!local_device_handle) {
    QNN_LOG_ERROR("Failed to create device handle!");
    return false;
  }

  // HTP Performance Settings
  if (options.GetHtpPerformanceMode() != ::qnn::HtpPerformanceMode::kDefault) {
    QNN_LOG_INFO("Set HTP performance mode: %d",
                 options.GetHtpPerformanceMode());
    QnnHtpDevice_Arch_t local_arch =
        local_device_platform_info->v1.hwDevices->v1.deviceInfoExtension
            ->onChipDevice.arch;
    if (auto status = perf_control_.Init(local_arch); !status) {
      return false;
    }
  }

  if (local_device_platform_info != nullptr) {
    if (auto status = QnnApi()->deviceFreePlatformInfo(
            nullptr, local_device_platform_info);
        status != QNN_SUCCESS) {
      QNN_LOG_ERROR("Failed to free HTP backend platform info: %d", status);
    }
  }

  // Follow RAII pattern to manage handles
  log_handle_ = std::move(local_log_handle);
  backend_handle_ = std::move(local_backend_handle);
  device_handle_ = std::move(local_device_handle);

  return true;
}

}  // namespace qnn
