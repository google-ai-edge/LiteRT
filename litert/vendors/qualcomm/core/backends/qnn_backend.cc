// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "QnnBackend.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnLog.h"  // from @qairt

namespace qnn {

QnnBackend::QnnBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : qnn_api_(qnn_api),
      log_handle_(nullptr, QnnApi()->logFree),
      backend_handle_(nullptr, QnnApi()->backendFree),
      device_handle_(nullptr, QnnApi()->deviceFree) {}

const QNN_INTERFACE_VER_TYPE* QnnBackend::QnnApi() { return qnn_api_; }

QnnBackend_Config_t& QnnBackend::AllocateBackendConfig() {
  auto& back = backend_configs_.emplace_back();
  back = QNN_BACKEND_CONFIG_INIT;
  return back;
}

QnnDevice_Config_t& QnnBackend::AllocateDeviceConfig() {
  auto& back = device_configs_.emplace_back();
  back = QNN_DEVICE_CONFIG_INIT;
  return back;
}

QnnDevice_PlatformInfo_t& QnnBackend::AllocateDevicePlatformInfo() {
  auto& back = device_platform_infos_.emplace_back();
  back = QNN_DEVICE_PLATFORM_INFO_INIT;
  return back;
}

QnnDevice_HardwareDeviceInfo_t& QnnBackend::AllocateDeviceHardwareInfo() {
  auto& back = device_hardware_infos_.emplace_back();
  back = QNN_DEVICE_HARDWARE_DEVICE_INFO_INIT;
  return back;
}

QnnDevice_CoreInfo_t& QnnBackend::AllocateDeviceCoreInfo() {
  auto& back = device_core_infos_.emplace_back();
  back = QNN_DEVICE_CORE_INFO_INIT;
  return back;
}
QnnBackend::QnnLogHandle QnnBackend::CreateLogHandle(
    ::qnn::LogLevel log_level) {
  if (log_level != ::qnn::LogLevel::kOff) {
    Qnn_LogHandle_t local_log_handle = nullptr;
    if (auto status = QnnApi()->logCreate(
            GetDefaultStdOutLogger(), static_cast<QnnLog_Level_t>(log_level),
            &local_log_handle);
        status != QNN_SUCCESS) {
      QNN_LOG_ERROR("Failed to create QNN logger: %d", status);
      return QnnLogHandle{nullptr, QnnApi()->logFree};
    }
    return QnnLogHandle{local_log_handle, QnnApi()->logFree};
  }

  return QnnLogHandle{nullptr, QnnApi()->logFree};
}

QnnBackend::QnnBackendHandle QnnBackend::CreateBackendHandle(
    Qnn_LogHandle_t log_handle,
    absl::Span<const QnnBackend_Config_t*> configs) {
  Qnn_BackendHandle_t local_backend_handle = nullptr;
  auto error = QnnApi()->backendCreate(
      log_handle, configs.size() <= 1 ? nullptr : configs.data(),
      &local_backend_handle);
  if (error != QNN_SUCCESS) {
    QNN_LOG_ERROR("failed to call backend create, %d", error);
    return QnnBackendHandle{nullptr, QnnApi()->backendFree};
  }

  return QnnBackendHandle{local_backend_handle, QnnApi()->backendFree};
}

QnnBackend::QnnDeviceHandle QnnBackend::CreateDeviceHandle(
    Qnn_LogHandle_t log_handle, absl::Span<const QnnDevice_Config_t*> configs) {
  Qnn_BackendHandle_t local_device_handle = nullptr;
  auto error = QnnApi()->deviceCreate(
      log_handle, configs.size() <= 1 ? nullptr : configs.data(),
      &local_device_handle);
  if (error != QNN_SUCCESS) {
    QNN_LOG_ERROR("failed to call device create, %d", error);
    return QnnDeviceHandle{nullptr, QnnApi()->deviceFree};
  }

  return QnnDeviceHandle{local_device_handle, QnnApi()->deviceFree};
}

Qnn_BackendHandle_t QnnBackend::GetBackendHandle() {
  return backend_handle_.get();
}

Qnn_DeviceHandle_t QnnBackend::GetDeviceHandle() {
  return device_handle_.get();
}

Qnn_LogHandle_t QnnBackend::GetLogHandle() { return log_handle_.get(); }

}  // namespace qnn
