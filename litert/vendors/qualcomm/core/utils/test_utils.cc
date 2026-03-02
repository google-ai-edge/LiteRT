// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/utils/test_utils.h"

#include <string>

#include "absl/strings/string_view.h"

namespace qnn {

namespace {
std::string& GetBackendStorage() {
  static std::string backend = "htp";
  return backend;
}
}  // namespace

bool IsTestTargetHtp() { return GetBackendStorage() == "htp"; }

void SetTestBackend(absl::string_view backend_name) {
  GetBackendStorage() = std::string(backend_name);
}

}  // namespace qnn
