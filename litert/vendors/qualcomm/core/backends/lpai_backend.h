// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_LPAI_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_LPAI_BACKEND_H_

#include <list>
#include <optional>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "LPAI/QnnLpaiBackend.h"  // from @qairt
#include "LPAI/QnnLpaiCommon.h"  // from @qairt
#include "LPAI/QnnLpaiGraph.h"  // from @qairt
#include "LPAI/QnnLpaiGraphPrepare.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

// QNN backend for the LPAI (Low Power AI) subsystem. Builds an HW-info custom
// config at Init, supplies the prepare (core-selection) config at graph-create
// time, and pushes per-graph perf / core-affinity configs after the graph is
// retrieved.
class LpaiBackend : public QnnBackend {
 public:
  static const char* GetLibraryName() {
#if defined(_WIN32)
    return "QnnLpai.dll";
#else
    return "libQnnLpai.so";
#endif
  }

  static Qnn_Version_t GetExpectedBackendVersion() {
    Qnn_Version_t backend_version;
    backend_version.major = QNN_LPAI_API_VERSION_MAJOR;
    backend_version.minor = QNN_LPAI_API_VERSION_MINOR;
    backend_version.patch = QNN_LPAI_API_VERSION_PATCH;
    return backend_version;
  }

  explicit LpaiBackend(const QNN_INTERFACE_VER_TYPE* qnn_api);

  bool Init(const Options& options, std::optional<SocInfo> soc_info) override;

  GraphConfigBuilder BuildGraphConfigs(
      const Options& options, absl::string_view qnn_graph_name) override;

  bool ConfigureGraphAfterRetrieve(const GraphConfigContext& ctx,
                                   const Options& options) override;

 private:
  // HW-info config storage. Built once at Init and referenced by the backend
  // handle, so it is owned by the backend (not per-graph).
  std::list<QnnLpaiBackend_CustomConfig_t> lpai_backend_custom_configs_;
  std::list<QnnLpaiBackend_CustomConfigHwInfo_t> lpai_hw_infos_;

  SocInfo soc_info_ = kSocInfos[0];
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_LPAI_BACKEND_H_
