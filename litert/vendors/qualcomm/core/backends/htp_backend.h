// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_BACKEND_H_

#include <optional>
#include <string>
#include <vector>

#include "HTP/QnnHtpDevice.h"  // from @qairt
#include "QnnBackend.h"        // from @qairt
#include "QnnDevice.h"         // from @qairt
#include "QnnInterface.h"      // from @qairt
#include "absl/types/span.h"   // from @com_google_absl
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"

namespace qnn {
class HtpBackend {
 public:
  explicit HtpBackend(const QNN_INTERFACE_VER_TYPE* api,
                      const ::qnn::LogLevel log_level,
                      std::optional<::qnn::SocInfo> soc_info);

  HtpBackend(const HtpBackend&) = delete;
  HtpBackend(HtpBackend&&) = delete;
  HtpBackend& operator=(const HtpBackend&) = delete;
  HtpBackend& operator=(HtpBackend&&) = delete;
  ~HtpBackend();

  bool Init();

  static const QnnInterface_t* GetValidProvider(
      absl::Span<const QnnInterface_t*> providers);

  bool Terminate();

  Qnn_BackendHandle_t& BackendHandle() { return backend_handle_; }

  Qnn_DeviceHandle_t& DeviceHandle() { return device_handle_; }

  Qnn_LogHandle_t& LogHandle() { return log_handle_; }

  const QnnDevice_PlatformInfo_t* DevicePlatforminfo() {
    return device_platform_info_;
  }

  const ::qnn::SocInfo SocInfo() { return soc_info_; }
  static constexpr const char* library_name_ = "libQnnHtp.so";

 private:
 std::vector<std::unique_ptr<QnnBackend_Config_t>> DefaultBackendConfigs(){
   std::vector<std::unique_ptr<QnnBackend_Config_t>> configs;
   configs.emplace_back(nullptr);

   return configs;
 }
  // std::vector<const QnnBackend_Config_t*> DefaultBackendConfigs() {
  //   backend_configs_.emplace_back(nullptr);
  //   return backend_configs_;
  // }

  std::vector<QnnDevice_CustomConfig_t> CreateDeviceCustomConfig(
      const ::qnn::SocInfo* soc_info);

  std::vector<QnnDevice_PlatformInfo_t*> CreateDevicePlatformInfo(
      const ::qnn::SocInfo* soc_info);

  QnnHtpDevice_CustomConfig_t* AllocDeviceCustomConfig() {
    htp_device_config_.emplace_back(
        std::make_unique<QnnHtpDevice_CustomConfig_t>());
    htp_device_config_.back()->option = QNN_HTP_DEVICE_CONFIG_OPTION_UNKNOWN;
    return htp_device_config_.back().get();
  }

  QnnDevice_PlatformInfo_t* AllocDevicePlatformInfo() {
    htp_platform_info_.emplace_back(
        std::make_unique<QnnDevice_PlatformInfo_t>());
    htp_platform_info_.back()->version =
        QNN_DEVICE_PLATFORM_INFO_VERSION_UNDEFINED;
    return htp_platform_info_.back().get();
  }

  QnnDevice_HardwareDeviceInfo_t* AllocHwDeviceInfo() {
    htp_hw_device_info_.emplace_back(
        std::make_unique<QnnDevice_HardwareDeviceInfo_t>());
    htp_hw_device_info_.back()->version =
        QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_UNDEFINED;
    return htp_hw_device_info_.back().get();
  }

  QnnDevice_CoreInfo_t* AllocCoreInfo() {
    htp_core_info_.emplace_back(std::make_unique<QnnDevice_CoreInfo_t>());
    htp_core_info_.back()->version = QNN_DEVICE_CORE_INFO_VERSION_UNDEFINED;
    return htp_core_info_.back().get();
  }

  QnnHtpDevice_DeviceInfoExtension_t* AllocDeviceInfoExtension() {
    htp_device_info_extension_.emplace_back(
        std::make_unique<QnnHtpDevice_DeviceInfoExtension_t>());
    htp_device_info_extension_.back()->devType = QNN_HTP_DEVICE_TYPE_UNKNOWN;
    return htp_device_info_extension_.back().get();
  }

  // std::vector<const QnnBackend_Config_t*> backend_configs_;
  std::vector<QnnDevice_Config_t> device_configs_;
  const QnnDevice_PlatformInfo_t* device_platform_info_ = nullptr;
  ::qnn::SocInfo soc_info_ = ::qnn::kSocInfos[6];  // V75
  std::optional<::qnn::SocInfo> soc_info_input_;

  std::vector<std::unique_ptr<QnnBackend_Config_t>> htp_backend_config_;

  std::vector<std::unique_ptr<QnnHtpDevice_CustomConfig_t>> htp_device_config_;
  std::vector<std::unique_ptr<QnnDevice_PlatformInfo_t>> htp_platform_info_;
  std::vector<std::unique_ptr<QnnDevice_HardwareDeviceInfo_t>>
      htp_hw_device_info_;
  std::vector<std::unique_ptr<QnnDevice_CoreInfo_t>> htp_core_info_;
  std::vector<std::unique_ptr<QnnHtpDevice_DeviceInfoExtension_t>>
      htp_device_info_extension_;

  Qnn_LogHandle_t log_handle_{nullptr};
  Qnn_BackendHandle_t backend_handle_{nullptr};
  Qnn_DeviceHandle_t device_handle_{nullptr};

  const QNN_INTERFACE_VER_TYPE* api_{nullptr};
  ::qnn::LogLevel log_level_{::qnn::LogLevel::kInfo};
};
}  // namespace qnn
#endif  // # ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_BACKEND_H_