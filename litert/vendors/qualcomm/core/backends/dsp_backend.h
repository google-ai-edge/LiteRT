// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_DSP_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_DSP_BACKEND_H_

#include <memory>
#include <optional>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "DSP/QnnDspCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

class DspBackend : public QnnBackend {
 public:
  static const char* GetLibraryName() {
#if defined(_WIN32)
    return "QnnDsp.dll";
#else
    return "libQnnDsp.so";
#endif
  }

  static Qnn_Version_t GetExpectedBackendVersion() {
    Qnn_Version_t backend_version;
    backend_version.major = QNN_DSP_API_VERSION_MAJOR;
    backend_version.minor = QNN_DSP_API_VERSION_MINOR;
    backend_version.patch = QNN_DSP_API_VERSION_PATCH;
    return backend_version;
  }

  explicit DspBackend(const QNN_INTERFACE_VER_TYPE* qnn_api);

  // Declared here, defined in .cc so the unique_ptr to the incomplete
  // DspPerfControl (pimpl) is destroyed where its type is complete.
  ~DspBackend();

  DspBackend(const DspBackend&) = delete;
  DspBackend& operator=(const DspBackend&) = delete;
  DspBackend(DspBackend&&) = delete;
  DspBackend& operator=(DspBackend&&) = delete;

  bool Init(const Options& options, std::optional<SocInfo> soc_info) override;
  bool SetPerformanceMode(const Options& options) override;

  // TODO: DSP does not build any graph configs yet; returns an empty builder.
  GraphConfigBuilder BuildGraphConfigs(
      const Options& options, absl::string_view qnn_graph_name) override;

 private:
  class DspPerfControl;
  std::unique_ptr<DspPerfControl> dsp_perf_control_{nullptr};
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_DSP_BACKEND_H_
