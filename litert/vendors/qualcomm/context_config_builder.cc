// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/context_config_builder.h"

#include <cstdint>
#include <utility>

#include "litert/c/internal/litert_logging.h"
#include "litert/vendors/qualcomm/core/backends/backend_utils.h"

namespace qnn {
namespace {

QnnGpuContext_PerfHint_t GetGpuPerformanceHint(
    GpuPerformanceMode performance_mode) {
  switch (performance_mode) {
    case GpuPerformanceMode::kHigh:
      return QNN_GPU_CONTEXT_PERF_HINT_HIGH;
    case GpuPerformanceMode::kNormal:
      return QNN_GPU_CONTEXT_PERF_HINT_NORMAL;
    case GpuPerformanceMode::kLow:
      return QNN_GPU_CONTEXT_PERF_HINT_LOW;
    case GpuPerformanceMode::kDefault:
      return QNN_GPU_CONTEXT_PERF_HINT_HIGH;
  }
}

}  // namespace

QnnContextConfigs BuildCompileContextConfigs(
    const Options& options, bool enable_effective_weight_sharing) {
  QnnContextConfigsBuilder builder;
  if (enable_effective_weight_sharing) {
    builder.EmplaceCustomConfig<HtpWeightSharingConfig>(true);
    LITERT_LOG(LITERT_INFO, "Enable weight sharing feature");
  } else if (const auto performance_mode = options.GetGpuPerformanceMode();
             options.GetBackendType() == BackendType::kGpuBackend &&
             performance_mode != GpuPerformanceMode::kDefault) {
    builder.EmplaceCustomConfig<GpuPerformanceConfig>(
        GetGpuPerformanceHint(performance_mode));
  }
  return std::move(builder).Build();
}

QnnContextConfigs BuildDispatchContextConfigs(const Options& options) {
  QnnContextConfigsBuilder builder;
  if (options.GetBackendType() != BackendType::kHtpBackend) {
    return std::move(builder).Build();
  }
  if (options.GetGraphPriority() != GraphPriority::kDefault) {
    QnnContext_Config_t priority = QNN_CONTEXT_CONFIG_INIT;
    priority.option = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
    priority.priority = GetGraphPriorityValue(options.GetGraphPriority());
    builder.AddContextConfig(priority);
  }
  if (const uint32_t budget_mb = options.GetHtpFileReadMemoryBudgetMB();
      budget_mb > 0) {
    builder.EmplaceCustomConfig<HtpFileReadMemoryBudgetConfig>(budget_mb);
  }
  if (options.GetHtpContextInitAcceleration()) {
    builder.EmplaceCustomConfig<HtpInitAccelerationConfig>(true);
  }
  return std::move(builder).Build();
}

}  // namespace qnn
