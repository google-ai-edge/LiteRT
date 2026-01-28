// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_SAVER_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_SAVER_UTILS_H_

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "QnnTypes.h"  // from @qairt
#include "Saver/QnnSaverCommon.h"  // from @qairt

namespace litert::qnn {

constexpr char kSaverLibraryName[] = "libQnnSaver.so";

inline Qnn_Version_t GetExpectedSaverVersion() {
  Qnn_Version_t backend_version;
  backend_version.major = QNN_SAVER_API_VERSION_MAJOR;
  backend_version.minor = QNN_SAVER_API_VERSION_MINOR;
  backend_version.patch = QNN_SAVER_API_VERSION_PATCH;
  return backend_version;
}

LiteRtStatus InitSaver(const SharedLibrary& lib,
                       absl::string_view saver_output_dir);

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_SAVER_UTILS_H_
