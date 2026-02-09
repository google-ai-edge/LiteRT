// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_TEST_UTILS_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_TEST_UTILS_H_

#include "absl/strings/string_view.h"

namespace qnn {

// Checks if the target backend for testing is HTP.
// This relies on the global variable set by the test main.
// Default is true (HTP).
bool IsTestTargetHtp();

// Sets the target backend for testing.
// This should be called by the test main.
void SetTestBackend(absl::string_view backend_name);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_UTILS_TEST_UTILS_H_
