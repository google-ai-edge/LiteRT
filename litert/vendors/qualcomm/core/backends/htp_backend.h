// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_BACKEND_H_

#include "HTP/QnnHtpCommon.h"  // from @qairt
#include "HTP/QnnHtpDevice.h"  // from @qairt
#include "QnnDevice.h"         // from @qairt
#include "QnnInterface.h"      // from @qairt
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"

namespace qnn {

class HtpBackend : public QnnBackend {
 public:
  static const char *GetLibraryName() { return "libQnnHtp.so"; }

  static const Qnn_Version_t GetExpectedBackendVersion() {
    Qnn_Version_t backend_version;
    backend_version.major = QNN_HTP_API_VERSION_MAJOR;
    backend_version.minor = QNN_HTP_API_VERSION_MINOR;
    backend_version.patch = QNN_HTP_API_VERSION_PATCH;
    return backend_version;
  }

  HtpBackend(const QNN_INTERFACE_VER_TYPE *qnn_api);

  ~HtpBackend();

  bool Init(const Options &options,
            std::optional<::qnn::SocInfo> soc_info) override;

  ::qnn::SocInfo GetSocInfo() { return soc_info_; }

 private:
  QnnHtpDevice_CustomConfig_t &AllocateHtpDeviceConfig() {
    auto &back = htp_device_config_.emplace_back();
    back.option = QNN_HTP_DEVICE_CONFIG_OPTION_UNKNOWN;
    return back;
  }

  QnnHtpDevice_DeviceInfoExtension_t &AllocHtpDeviceInfoExtension() {
    auto &back = htp_device_info_extension_.emplace_back();
    back.devType = QNN_HTP_DEVICE_TYPE_UNKNOWN;
    return back;
  }

  ::qnn::SocInfo soc_info_ = ::qnn::kSocInfos[7];  // V75
  std::list<QnnHtpDevice_CustomConfig_t> htp_device_config_;
  std::list<QnnHtpDevice_DeviceInfoExtension_t> htp_device_info_extension_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_BACKEND_H_
