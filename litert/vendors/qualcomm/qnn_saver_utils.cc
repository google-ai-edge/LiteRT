// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/qnn_saver_utils.h"

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "Saver/QnnSaver.h"  // from @qairt

namespace litert::qnn {

typedef Qnn_ErrorHandle_t (*QnnSaverInitFn_t)(const QnnSaver_Config_t**);

constexpr char kLibQnnSaverSymbol[] = "QnnSaver_initialize";

absl::Span<const QnnSaver_Config_t*> GetDefaultSaverConfigs(
    absl::string_view saver_output_dir) {
  static std::array<QnnSaver_Config_t, 1> saver_configs;
  saver_configs[0].option = QNN_SAVER_CONFIG_OPTION_OUTPUT_DIRECTORY;
  saver_configs[0].outputDirectory = saver_output_dir.data();

  static std::array<const QnnSaver_Config_t*, 2> result = {&saver_configs[0],
                                                           nullptr};
  return absl::MakeSpan(result.data(), result.size());
}

LiteRtStatus InitSaver(const SharedLibrary& lib,
                       absl::string_view saver_output_dir) {
  // saver_config must be set before backend initialization
  LITERT_ASSIGN_OR_RETURN(
      QnnSaverInitFn_t saver_initialize,
      lib.LookupSymbol<QnnSaverInitFn_t>(kLibQnnSaverSymbol));
  if (Qnn_ErrorHandle_t error =
          saver_initialize(GetDefaultSaverConfigs(saver_output_dir).data());
      error != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR,
               "Qnn saver backend failed to initialize. Error code: %d.",
               QNN_GET_ERROR_CODE(error));
    return kLiteRtStatusErrorDynamicLoading;
  }
  return kLiteRtStatusOk;
}
}  // namespace litert::qnn
