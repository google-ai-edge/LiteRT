// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_SAVER_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_SAVER_UTILS_H_

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "QnnTypes.h"  // from @qairt

namespace litert::qnn {

constexpr char kSaverLibraryName[] = "libQnnSaver.so";

Qnn_Version_t GetExpectedSaverVersion();

LiteRtStatus InitSaver(const SharedLibrary& lib, absl::string_view saver_output_dir);

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_QNN_SAVER_UTILS_H_
