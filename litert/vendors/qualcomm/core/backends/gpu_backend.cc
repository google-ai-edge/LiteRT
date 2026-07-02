// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/gpu_backend.h"

#include <array>
#include <memory>
#include <optional>
#include <utility>

#include "GPU/QnnGpuGraph.h"  // from @qairt
#include "QnnBackend.h"  // from @qairt    // from @qairt
#include "QnnInterface.h"  // from @qairt  // from @qairt
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"

namespace qnn {
namespace {

QnnGpu_Precision_t GetGpuPrecisionValue(GpuPrecision precision) {
  switch (precision) {
    case GpuPrecision::kUserProvided:
      return QNN_GPU_PRECISION_USER_PROVIDED;
    case GpuPrecision::kFp32:
      return QNN_GPU_PRECISION_FP32;
    case GpuPrecision::kFp16:
      return QNN_GPU_PRECISION_FP16;
    case GpuPrecision::kHybrid:
      return QNN_GPU_PRECISION_HYBRID;
    default:
      return QNN_GPU_PRECISION_FP16;
  }
}

}  // namespace

GpuBackend::GpuBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : QnnBackend(qnn_api) {}

bool GpuBackend::Init(const Options& options, std::optional<SocInfo> soc_info) {
  // Log Handle
  auto local_log_handle = CreateLogHandle(options.GetLogLevel());
  if (!local_log_handle && options.GetLogLevel() != LogLevel::kOff) {
    QNN_LOG_ERROR("Failed to create log handle.");
    return false;
  }

  // Backend Handle
  std::array<const QnnBackend_Config_t*, 1> backend_configs = {nullptr};

  auto local_backend_handle = CreateBackendHandle(
      local_log_handle.get(), absl::MakeSpan(backend_configs));
  if (!local_backend_handle) {
    QNN_LOG_ERROR("Failed to create backend handle.");
    return false;
  }

  // Follow RAII pattern to manage handles.
  log_handle_ = std::move(local_log_handle);
  backend_handle_ = std::move(local_backend_handle);

  return true;
}

GraphConfigBuilder GpuBackend::BuildGraphConfigs(
    const Options& options, absl::string_view /*qnn_graph_name*/) {
  GraphConfigBuilder mgr;

  // Precision
  auto& custom_config = mgr.AddCustom<QnnGpuGraph_CustomConfig_t>(
      QNN_GPU_GRAPH_CUSTOM_CONFIG_INIT);
  custom_config.precision = GetGpuPrecisionValue(options.GetGpuPrecision());

  return mgr;
}

}  // namespace qnn
