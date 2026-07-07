// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/backend_factory.h"

#include <memory>
#include <optional>

#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"
#include "litert/vendors/qualcomm/core/backends/gpu_backend.h"
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"

namespace {

constexpr char kCustomOpPackageCompileTarget[] = "CPU";

}  // namespace

namespace qnn {

std::unique_ptr<QnnBackend> CreateBackend(const QnnApi* api,
                                          const Options& options,
                                          std::optional<SocInfo> soc_info,
                                          bool is_compiler) {
  std::unique_ptr<QnnBackend> backend;
  switch (options.GetBackendType()) {
    case BackendType::kGpuBackend: {
      backend = std::make_unique<GpuBackend>(api);
      break;
    }
    case BackendType::kHtpBackend: {
      backend = std::make_unique<HtpBackend>(api);
      break;
    }
    case BackendType::kIrBackend: {
      backend = std::make_unique<IrBackend>(api);
      break;
    }
    case BackendType::kDspBackend: {
      backend = std::make_unique<DspBackend>(api);
      break;
    }
    default:
      return nullptr;
  }
  if (!backend->Init(options, soc_info)) {
    return nullptr;
  }

  const auto& custom_op_package = options.GetCustomOpPackage();
  if (custom_op_package.name.empty()) {
    return backend;
  }
  if (options.GetBackendType() != BackendType::kHtpBackend) {
    QNN_LOG_INFO(
        "Custom op package is only supported on HtpBackend. Ignore.");
    return backend;
  }
  const auto& package_path = is_compiler
                                 ? custom_op_package.compile_package_path
                                 : custom_op_package.dispatch_package_path;
  const auto& target =
      is_compiler ? kCustomOpPackageCompileTarget : custom_op_package.target;
  if (auto status = api->backendRegisterOpPackage(
          backend->GetBackendHandle(), package_path.c_str(),
          custom_op_package.interface_provider.c_str(), target.c_str());
      status != QNN_SUCCESS) {
    QNN_LOG_ERROR("Failed to register op package. Error code: %d", status);
    return nullptr;
  }

  return backend;
}

}  // namespace qnn
