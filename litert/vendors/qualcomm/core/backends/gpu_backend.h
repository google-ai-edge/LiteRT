// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_GPU_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_GPU_BACKEND_H_

#include <memory>
#include <optional>

#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "GPU/QnnGpuCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

class GpuBackend : public QnnBackend {
 public:
  static const char* GetLibraryName() { return "libQnnGpu.so"; }

  static Qnn_Version_t GetExpectedBackendVersion() {
    Qnn_Version_t backend_version;
    backend_version.major = QNN_GPU_API_VERSION_MAJOR;
    backend_version.minor = QNN_GPU_API_VERSION_MINOR;
    backend_version.patch = QNN_GPU_API_VERSION_PATCH;
    return backend_version;
  }

  explicit GpuBackend(const QNN_INTERFACE_VER_TYPE* qnn_api);

  ~GpuBackend() = default;

  GpuBackend(const GpuBackend&) = delete;
  GpuBackend& operator=(const GpuBackend&) = delete;
  GpuBackend(GpuBackend&&) = delete;
  GpuBackend& operator=(GpuBackend&&) = delete;

  bool Init(const Options& options, std::optional<SocInfo> soc_info) override;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_GPU_BACKEND_H_
