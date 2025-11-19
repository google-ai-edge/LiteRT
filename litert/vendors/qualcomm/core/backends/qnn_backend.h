// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_QNN_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_QNN_BACKEND_H_

#include <list>
#include <memory>
#include <optional>
#include <type_traits>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "QnnBackend.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace qnn {

class QnnBackend {
 public:
  using QnnLogHandle =
      std::unique_ptr<std::remove_pointer<Qnn_LogHandle_t>::type,
                      QnnLog_FreeFn_t>;
  using QnnBackendHandle =
      std::unique_ptr<std::remove_pointer<Qnn_BackendHandle_t>::type,
                      QnnBackend_FreeFn_t>;
  using QnnDeviceHandle =
      std::unique_ptr<std::remove_pointer<Qnn_DeviceHandle_t>::type,
                      QnnDevice_FreeFn_t>;

  explicit QnnBackend(const QNN_INTERFACE_VER_TYPE* qnn_api);

  virtual ~QnnBackend() = default;

  virtual bool Init(const Options& options,
                    std::optional<::qnn::SocInfo> soc_info) = 0;

  Qnn_BackendHandle_t GetBackendHandle();

  Qnn_DeviceHandle_t GetDeviceHandle();

  Qnn_LogHandle_t GetLogHandle();

 private:
  const QNN_INTERFACE_VER_TYPE* qnn_api_ = nullptr;
  std::list<QnnBackend_Config_t> backend_configs_;
  std::list<QnnDevice_Config_t> device_configs_;
  std::list<QnnDevice_PlatformInfo_t> device_platform_infos_;
  std::list<QnnDevice_HardwareDeviceInfo_t> device_hardware_infos_;
  std::list<QnnDevice_CoreInfo_t> device_core_infos_;

 protected:
  const QNN_INTERFACE_VER_TYPE* QnnApi();

  QnnBackend_Config_t& AllocateBackendConfig();

  QnnDevice_Config_t& AllocateDeviceConfig();

  QnnDevice_PlatformInfo_t& AllocateDevicePlatformInfo();

  QnnDevice_HardwareDeviceInfo_t& AllocateDeviceHardwareInfo();

  QnnDevice_CoreInfo_t& AllocateDeviceCoreInfo();

  QnnLogHandle CreateLogHandle(::qnn::LogLevel log_level);

  QnnBackendHandle CreateBackendHandle(
      Qnn_LogHandle_t log_handle,
      absl::Span<const QnnBackend_Config_t*> configs);

  QnnDeviceHandle CreateDeviceHandle(
      Qnn_LogHandle_t log_handle,
      absl::Span<const QnnDevice_Config_t*> configs);

  QnnLogHandle log_handle_;
  QnnBackendHandle backend_handle_;
  QnnDeviceHandle device_handle_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_QNN_BACKEND_H_
