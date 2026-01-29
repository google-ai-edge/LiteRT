// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/qnn_saver_utils.h"

#include <array>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_macros.h"
#include "QnnCommon.h"  // from @qairt
#include "Saver/QnnSaver.h"  // from @qairt

namespace litert::qnn {

typedef Qnn_ErrorHandle_t (*QnnSaverInitFn_t)(const QnnSaver_Config_t**);

constexpr char kLibQnnSaverSymbol[] = "QnnSaver_initialize";

LiteRtStatus InitSaver(const SharedLibrary& lib,
                       absl::string_view saver_output_dir) {
  // saver_config must be set before backend initialization
  std::string output_dir_str(saver_output_dir);
  std::array<QnnSaver_Config_t, 1> saver_configs;
  saver_configs[0].option = QNN_SAVER_CONFIG_OPTION_OUTPUT_DIRECTORY;
  saver_configs[0].outputDirectory = output_dir_str.c_str();

  const QnnSaver_Config_t* config_ptrs[] = {&saver_configs[0], nullptr};

  LITERT_ASSIGN_OR_RETURN(
      QnnSaverInitFn_t saver_initialize,
      lib.LookupSymbol<QnnSaverInitFn_t>(kLibQnnSaverSymbol));
  if (Qnn_ErrorHandle_t error = saver_initialize(config_ptrs);
      error != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR,
               "Qnn saver backend failed to initialize. Error code: %d.",
               QNN_GET_ERROR_CODE(error));
    return kLiteRtStatusErrorDynamicLoading;
  }
  return kLiteRtStatusOk;
}
}  // namespace litert::qnn
