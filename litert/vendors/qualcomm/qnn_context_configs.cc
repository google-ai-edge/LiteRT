// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/qnn_context_configs.h"

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/common.h"
#include "GPU/QnnGpuContext.h"  // from @qairt
#include "HTP/QnnHtpContext.h"  // from @qairt
#include "QnnContext.h"  // from @qairt

namespace litert::qnn {

absl::Span<const QnnContext_Config_t*> DefaultContextConfigs() {
  static const QnnContext_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
}

absl::Span<const QnnContext_Config_t*> WeightSharingContextConfigs() {
  static QnnHtpContext_CustomConfig_t customConfig =
      QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
  customConfig.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
  customConfig.weightSharingEnabled = true;
  static QnnContext_Config_t contextConfig = QNN_CONTEXT_CONFIG_INIT;
  contextConfig.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  contextConfig.customConfig = &customConfig;
  static const QnnContext_Config_t* configs[2] = {&contextConfig, nullptr};
  return absl::MakeSpan(configs);
}

absl::Span<const QnnContext_Config_t*> GpuPerformanceContextConfigs(
    ::qnn::GpuPerformanceMode performance_mode) {
  static QnnGpuContext_CustomConfig_t customConfig =
      QNN_GPU_CONTEXT_CUSTOM_CONFIG_INIT;
  customConfig.option = QNN_GPU_CONTEXT_CONFIG_OPTION_PERF_HINT;
  switch (performance_mode) {
    case ::qnn::GpuPerformanceMode::kHigh:
      customConfig.perfHint = QNN_GPU_CONTEXT_PERF_HINT_HIGH;
      break;
    case ::qnn::GpuPerformanceMode::kNormal:
      customConfig.perfHint = QNN_GPU_CONTEXT_PERF_HINT_NORMAL;
      break;
    case ::qnn::GpuPerformanceMode::kLow:
      customConfig.perfHint = QNN_GPU_CONTEXT_PERF_HINT_LOW;
      break;
    case ::qnn::GpuPerformanceMode::kDefault:
    default:
      return DefaultContextConfigs();
  }

  static QnnContext_Config_t contextConfig = QNN_CONTEXT_CONFIG_INIT;
  contextConfig.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  contextConfig.customConfig = &customConfig;
  static const QnnContext_Config_t* configs[2] = {&contextConfig, nullptr};
  return absl::MakeSpan(configs);
}

}  // namespace litert::qnn
