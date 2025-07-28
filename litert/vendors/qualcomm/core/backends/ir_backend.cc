// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/ir_backend.h"

namespace qnn {

IrBackend::IrBackend(const QNN_INTERFACE_VER_TYPE *qnn_api)
    : QnnBackend(qnn_api) {}

IrBackend::~IrBackend() {}

bool IrBackend::Init(Qnn_LogHandle_t log_handle, const Options &options) {
  std::vector<const QnnBackend_Config_t *> backend_configs;
  backend_configs.emplace_back(nullptr);
  auto local_backend_handle =
      CreateBackendHandle(log_handle, absl::MakeSpan(backend_configs));
  if (!local_backend_handle) {
    QNN_LOG_ERROR("failed to create backend");
    return false;
  }

  backend_handle_ = std::move(local_backend_handle);
  return true;
}
}  // namespace qnn
