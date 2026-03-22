// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/utils/test_utils.h"

#include "litert/vendors/qualcomm/core/common.h"

namespace qnn {

namespace {
BackendType& GetTestBackend() {
  static BackendType test_backend = BackendType::kHtpBackend;

  return test_backend;
}
}  // namespace

bool IsTestHtpBackend() { return GetTestBackend() == BackendType::kHtpBackend; }

bool IsTestDspBackend() { return GetTestBackend() == BackendType::kDspBackend; }

void SetTestBackend(BackendType backend_type) {
  GetTestBackend() = backend_type;
}

}  // namespace qnn
