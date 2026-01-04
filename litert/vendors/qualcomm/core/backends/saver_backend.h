// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_SAVER_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_SAVER_BACKEND_H_

#include <optional>

#include "QnnInterface.h"          // from @qairt
#include "QnnTypes.h"              // from @qairt
#include "Saver/QnnSaver.h"        // from @qairt
#include "Saver/QnnSaverCommon.h"  // from @qairt
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"

namespace qnn::SaverBackend {

static const char* GetLibraryName() { return "libQnnSaver.so"; }

typedef Qnn_ErrorHandle_t (*QnnSaverInitFn_t)(const QnnSaver_Config_t**);

constexpr char kLibQnnSaverSymbol[] = "QnnSaver_initialize";

Qnn_Version_t GetExpectedBackendVersion();

bool Init(QnnSaverInitFn_t saver_initialize,
          absl::string_view saver_output_dir);

}  // namespace qnn::SaverBackend

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_SAVER_BACKEND_H_
