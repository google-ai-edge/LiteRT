// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/htp_backend.h"

#include <optional>

#include "HTP/QnnHtpCommon.h"  // from @qairt
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/log.h"

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
}  // namespace
HtpBackend::HtpBackend(const QNN_INTERFACE_VER_TYPE* api,
                       const ::qnn::LogLevel log_level,
                       std::optional<::qnn::SocInfo> soc_info_input)
    : api_(api), log_level_(log_level), soc_info_input_(soc_info_input) {
  htp_backend_config_ = DefaultBackendConfigs();
}

HtpBackend::~HtpBackend() = default;

bool HtpBackend::Init() {
  // Log Handle
  if (log_level_ != ::qnn::LogLevel::kOff) {
    if (auto status = api_->logCreate(GetDefaultStdOutLogger(),
                                      static_cast<QnnLog_Level_t>(log_level_),
                                      &log_handle_);
        status != QNN_SUCCESS) {
      QNN_LOG_ERROR("Failed to create QNN logger: %d", status);
      return false;
    }
  }

  // Backend Handle
  std::vector<const QnnBackend_Config_t*> backend_configs;
  backend_configs.reserve(htp_backend_config_.size());
  for (const auto& config:htp_backend_config_){
    backend_configs.emplace_back(config.get());
  }

  if (auto status = api_->backendCreate(log_handle_, backend_configs.data(),
                                        &backend_handle_);
      status != QNN_SUCCESS) {
    QNN_LOG_ERROR("Failed to create QNN backend: %d", status);
    return false;
  }

  // Device Handle
  if (soc_info_input_.has_value()) {
    QNN_LOG_INFO("Using provided SoC info.");
    soc_info_ = *soc_info_input_;
  } else {
    QNN_LOG_INFO("Apply deviceGetPlatformInfo for SoC info.");
    if (auto status =
            api_->deviceGetPlatformInfo(nullptr, &device_platform_info_);
        status == QNN_SUCCESS) {
      auto soc_info_online = FindSocInfo(static_cast<::qnn::SnapdragonModel>(
          device_platform_info_->v1.hwDevices->v1.deviceInfoExtension
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

  std::vector<const QnnDevice_Config_t*> device_configs;
  const std::vector<QnnDevice_CustomConfig_t> device_custom_config =
      CreateDeviceCustomConfig(&soc_info_);
  const std::vector<QnnDevice_PlatformInfo_t*> device_platform_info =
      CreateDevicePlatformInfo(&soc_info_);

  uint32_t num_custom_configs =
      device_platform_info.size() + device_custom_config.size();
  device_configs_.resize(num_custom_configs);
  // +1 for null terminated
  device_configs.reserve(num_custom_configs + 1);
  for (std::size_t i = 0; i < device_custom_config.size(); ++i) {
    device_configs_[i].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    device_configs_[i].customConfig = device_custom_config[i];
    device_configs.emplace_back(&device_configs_[i]);
  }
  for (std::size_t i = 0; i < device_platform_info.size(); ++i) {
    device_configs_[device_custom_config.size() + i].option =
        QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
    device_configs_[device_custom_config.size() + i].hardwareInfo =
        device_platform_info[i];
    device_configs.emplace_back(
        &device_configs_[device_custom_config.size() + i]);
  }
  // null terminatedD
  device_configs.emplace_back(nullptr);
  if (auto status = api_->deviceCreate(log_handle_, device_configs.data(),
                                       &device_handle_);
      status != QNN_SUCCESS) {
    QNN_LOG_ERROR("Failed to create QNN device: %d", status);
    return false;
  }

  return true;
}

const QnnInterface_t* HtpBackend::GetValidProvider(
    absl::Span<const QnnInterface_t*> providers) {
  auto qnn_version = providers[0]->apiVersion;
  if (qnn_version.coreApiVersion.major != QNN_API_VERSION_MAJOR) {
    QNN_LOG_ERROR(
        "Qnn library version %u.%u.%u is not supported. "
        "The minimum supported version is %u.%u.%u. Please make "
        "sure you have the correct library version.",
        qnn_version.coreApiVersion.major, qnn_version.coreApiVersion.minor,
        qnn_version.coreApiVersion.patch, QNN_API_VERSION_MAJOR,
        QNN_API_VERSION_MINOR, QNN_API_VERSION_PATCH);
    return nullptr;
  }

  if ((qnn_version.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
       qnn_version.coreApiVersion.minor < QNN_API_VERSION_MINOR)) {
    QNN_LOG_ERROR(
        "Qnn library version %u.%u.%u is mismatched. "
        "The minimum supported version is %u.%u.%u. Please make "
        "sure you have the correct library version.",
        qnn_version.coreApiVersion.major, qnn_version.coreApiVersion.minor,
        qnn_version.coreApiVersion.patch, QNN_API_VERSION_MAJOR,
        QNN_API_VERSION_MINOR, QNN_API_VERSION_PATCH);
    return nullptr;
  }

  if (qnn_version.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
      qnn_version.coreApiVersion.minor > QNN_API_VERSION_MINOR) {
    QNN_LOG_WARNING(
        "Qnn library version %u.%u.%u is used. "
        "The version LiteRT using is %u.%u.%u.",
        qnn_version.coreApiVersion.major, qnn_version.coreApiVersion.minor,
        qnn_version.coreApiVersion.patch, QNN_API_VERSION_MAJOR,
        QNN_API_VERSION_MINOR, QNN_API_VERSION_PATCH);
  }

  // TODO (chunhsue-qti) more backend version
  if (qnn_version.backendApiVersion.major != QNN_HTP_API_VERSION_MAJOR) {
    QNN_LOG_ERROR(
        "Qnn backend library version %u.%u.%u is not supported. "
        "The minimum supported version is %u.%u.%u. Please make "
        "sure you have the correct library version.",
        qnn_version.backendApiVersion.major,
        qnn_version.backendApiVersion.minor,
        qnn_version.backendApiVersion.patch, QNN_HTP_API_VERSION_MAJOR,
        QNN_HTP_API_VERSION_MINOR, QNN_HTP_API_VERSION_PATCH);
    return nullptr;
  }

  if ((qnn_version.backendApiVersion.major == QNN_HTP_API_VERSION_MAJOR &&
       qnn_version.backendApiVersion.minor < QNN_HTP_API_VERSION_MINOR)) {
    QNN_LOG_ERROR(
        "Qnn backend library version %u.%u.%u is mismatched. "
        "The minimum supported version is %u.%u.%u. Please make "
        "sure you have the correct library version.",
        qnn_version.backendApiVersion.major,
        qnn_version.backendApiVersion.minor,
        qnn_version.backendApiVersion.patch, QNN_HTP_API_VERSION_MAJOR,
        QNN_HTP_API_VERSION_MINOR, QNN_HTP_API_VERSION_PATCH);
    return nullptr;
  }

  if (qnn_version.backendApiVersion.major == QNN_HTP_API_VERSION_MAJOR &&
      qnn_version.backendApiVersion.minor > QNN_HTP_API_VERSION_MINOR) {
    QNN_LOG_WARNING(
        "Qnn backend library version %u.%u.%u is used. "
        "The version LiteRT using is %u.%u.%u.",
        qnn_version.backendApiVersion.major,
        qnn_version.backendApiVersion.minor,
        qnn_version.backendApiVersion.patch, QNN_HTP_API_VERSION_MAJOR,
        QNN_HTP_API_VERSION_MINOR, QNN_HTP_API_VERSION_PATCH);
  }
  return providers[0];
}

std::vector<QnnDevice_CustomConfig_t> HtpBackend::CreateDeviceCustomConfig(
    const ::qnn::SocInfo* soc_info) {
  std::vector<QnnDevice_CustomConfig_t> ret;
  QnnHtpDevice_CustomConfig_t* p_custom_config = nullptr;

  p_custom_config = AllocDeviceCustomConfig();
  p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
  p_custom_config->socModel = static_cast<uint32_t>(soc_info->soc_model);
  ret.emplace_back(static_cast<QnnDevice_CustomConfig_t>(p_custom_config));

  return ret;
}

std::vector<QnnDevice_PlatformInfo_t*> HtpBackend::CreateDevicePlatformInfo(
    const ::qnn::SocInfo* soc_info) {
#ifdef __ANDROID__
  return {};
#else
  std::vector<QnnDevice_PlatformInfo_t*> ret;
  QnnDevice_PlatformInfo_t* p_platform_info = nullptr;
  QnnDevice_HardwareDeviceInfo_t* p_hw_device_info = nullptr;
  QnnHtpDevice_DeviceInfoExtension_t* p_device_info_extension = nullptr;
  QnnDevice_CoreInfo_t* p_core_info = nullptr;

  p_platform_info = AllocDevicePlatformInfo();
  p_platform_info->version = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
  p_platform_info->v1.numHwDevices = 1;

  p_hw_device_info = AllocHwDeviceInfo();
  p_hw_device_info->version = QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1;
  p_hw_device_info->v1.deviceId = 0;
  p_hw_device_info->v1.deviceType = 0;
  p_hw_device_info->v1.numCores = 1;

  p_device_info_extension = AllocDeviceInfoExtension();
  p_device_info_extension->devType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
  p_device_info_extension->onChipDevice.vtcmSize = soc_info->vtcm_size_in_mb;
  // TODO(jiunkaiy): Given by user, default value is unsigned pd
  p_device_info_extension->onChipDevice.signedPdSupport = 0;
  p_device_info_extension->onChipDevice.socModel =
      static_cast<uint32_t>(soc_info->soc_model);
  p_device_info_extension->onChipDevice.arch =
      static_cast<QnnHtpDevice_Arch_t>(soc_info->dsp_arch);
  // TODO(jiunkaiy): For Htp, dlbcSupport is true
  p_device_info_extension->onChipDevice.dlbcSupport = true;
  p_hw_device_info->v1.deviceInfoExtension = p_device_info_extension;

  p_core_info = AllocCoreInfo();
  p_core_info->version = QNN_DEVICE_CORE_INFO_VERSION_1;
  p_core_info->v1.coreId = 0;
  p_core_info->v1.coreType = 0;
  p_core_info->v1.coreInfoExtension = nullptr;
  p_hw_device_info->v1.cores = p_core_info;

  p_platform_info->v1.hwDevices = p_hw_device_info;
  ret.emplace_back(p_platform_info);

  return ret;
#endif
}

};  // namespace qnn