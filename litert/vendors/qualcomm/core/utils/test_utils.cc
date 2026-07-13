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

bool IsTestGpuBackend() { return GetTestBackend() == BackendType::kGpuBackend; }

void SetTestBackend(BackendType backend_type) {
  GetTestBackend() = backend_type;
}

namespace {
std::string& GetTestDispatchLibraryDirMutable() {
  static std::string dir = "/data/local/tmp";
  return dir;
}
}  // namespace

const std::string& GetTestDispatchLibraryDir() {
  return GetTestDispatchLibraryDirMutable();
}

void SetTestDispatchLibraryDir(const std::string& dir) {
  GetTestDispatchLibraryDirMutable() = dir;
}

}  // namespace qnn
