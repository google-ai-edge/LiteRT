// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0


#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/backends/saver_backend.h"

namespace qnn::SaverBackend {

absl::Span<const QnnSaver_Config_t*> GetDefaultSaverConfigs(
    absl::string_view saver_output_dir) {
  static std::array<QnnSaver_Config_t, 1> saver_configs;
  saver_configs[0].option = QNN_SAVER_CONFIG_OPTION_OUTPUT_DIRECTORY;
  saver_configs[0].outputDirectory = saver_output_dir.data();

  static std::array<const QnnSaver_Config_t*, 2> result = {&saver_configs[0],
                                                           nullptr};
  return absl::MakeSpan(result.data(), result.size());
}

bool Init(QnnSaverInitFn_t saver_initialize, absl::string_view saver_output_dir) {
  // saver_config must be set before backend initialization
  if (Qnn_ErrorHandle_t error =
          saver_initialize(GetDefaultSaverConfigs(saver_output_dir).data());
      error != QNN_SUCCESS) {
    QNN_LOG_ERROR(
               "Qnn saver backend failed to initialize. Error code: %d.",
               QNN_GET_ERROR_CODE(error));
    return false;
  }
  return true;
}

Qnn_Version_t GetExpectedBackendVersion() {
  Qnn_Version_t backend_version;
  backend_version.major = QNN_SAVER_API_VERSION_MAJOR;
  backend_version.minor = QNN_SAVER_API_VERSION_MINOR;
  backend_version.patch = QNN_SAVER_API_VERSION_PATCH;
  return backend_version;
}
}  // namespace qnn
