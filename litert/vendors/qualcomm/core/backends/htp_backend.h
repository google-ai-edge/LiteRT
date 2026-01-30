// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_BACKEND_H_

#include <list>
#include <memory>
#include <optional>

#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "HTP/QnnHtpCommon.h"  // from @qairt
#include "HTP/QnnHtpDevice.h"  // from @qairt
#include "QnnDevice.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

class HtpBackend : public QnnBackend {
 public:
  struct PlatformInfoDeleter {
    const QNN_INTERFACE_VER_TYPE* api_;
    void operator()(const QnnDevice_PlatformInfo_t* ptr) const {
      if (ptr) api_->deviceFreePlatformInfo(nullptr, ptr);
    }
  };

  using QnnDevicePlatformInfo =
      std::unique_ptr<const QnnDevice_PlatformInfo_t, PlatformInfoDeleter>;

  static const char* GetLibraryName() { return "libQnnHtp.so"; }

  static Qnn_Version_t GetExpectedBackendVersion() {
    Qnn_Version_t backend_version;
    backend_version.major = QNN_HTP_API_VERSION_MAJOR;
    backend_version.minor = QNN_HTP_API_VERSION_MINOR;
    backend_version.patch = QNN_HTP_API_VERSION_PATCH;
    return backend_version;
  }

  explicit HtpBackend(const QNN_INTERFACE_VER_TYPE* qnn_api);

  ~HtpBackend();

  HtpBackend(const HtpBackend&) = delete;
  HtpBackend& operator=(const HtpBackend&) = delete;
  HtpBackend(HtpBackend&&) = delete;
  HtpBackend& operator=(HtpBackend&&) = delete;

  bool Init(const Options& options, std::optional<SocInfo> soc_info) override;

  SocInfo GetSocInfo() { return soc_info_; }

 private:
  QnnHtpDevice_CustomConfig_t& AllocateHtpDeviceConfig() {
    auto& back = htp_device_configs_.emplace_back();
    back.option = QNN_HTP_DEVICE_CONFIG_OPTION_UNKNOWN;
    return back;
  }

  QnnHtpDevice_DeviceInfoExtension_t& AllocHtpDeviceInfoExtension() {
    auto& back = htp_device_info_extensions_.emplace_back();
    back.devType = QNN_HTP_DEVICE_TYPE_UNKNOWN;
    return back;
  }

  QnnDevicePlatformInfo CreateDevicePlatformInfo();

  SocInfo soc_info_ = kSocInfos[0];
  // The qnn_device_platform_info_ is referenced by device configurations
  // managed in the lists below. It must be destructed after the configs to
  // ensure valid references during destruction.
  QnnDevicePlatformInfo qnn_device_platform_info_;
  std::list<QnnHtpDevice_CustomConfig_t> htp_device_configs_;
  std::list<QnnHtpDevice_DeviceInfoExtension_t> htp_device_info_extensions_;
  class HtpPerfControl;
  std::unique_ptr<HtpPerfControl> htp_perf_control_{nullptr};
};
}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_BACKEND_H_
