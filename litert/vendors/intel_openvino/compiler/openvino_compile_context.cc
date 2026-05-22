// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/intel_openvino/compiler/openvino_compile_context.h"

#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/runtime/properties.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/vendors/intel_openvino/compiler/npu_optimizer.h"
#include "litert/vendors/intel_openvino/compiler/openvino_soc_config.h"

namespace litert {
namespace openvino {

OpenVinoCompileContext::OpenVinoCompileContext() {
  configs_map_[ov::hint::performance_mode.name()] =
      ov::hint::PerformanceMode::LATENCY;
}

::litert::Expected<OpenVinoCompileContext> OpenVinoCompileContext::Create(
    const ::litert::Expected< ::litert::intel_openvino::IntelOpenVinoOptions>&
        opts) {
  OpenVinoCompileContext context;
  if (!opts.HasValue()) {
    LITERT_LOG(LITERT_INFO, "Using default configuration (LATENCY mode)");
    return context;
  }
  const auto& options = opts.Value();

  // Configure device type.
  auto device_type = options.GetDeviceType();
  switch (device_type) {
    case kLiteRtIntelOpenVinoDeviceTypeCPU:
      context.device_ = "CPU";
      break;
    case kLiteRtIntelOpenVinoDeviceTypeGPU:
      context.device_ = "GPU";
      break;
    case kLiteRtIntelOpenVinoDeviceTypeNPU:
      context.device_ = "NPU";
      break;
    case kLiteRtIntelOpenVinoDeviceTypeAUTO:
      context.device_ = "AUTO";
      break;
  }

  LITERT_LOG(LITERT_INFO, "Using Intel OpenVINO device: %s",
             context.device_.c_str());

  auto performance_mode = options.GetPerformanceMode();

  // Add custom configuration options.
  int num_custom_options = options.GetNumConfigsMapOptions();
  for (int i = 0; i < num_custom_options; ++i) {
    auto [key, value] = options.GetConfigsMapOption(i);
    if (!key.empty()) {
      if (key == "optimize_fq_after_matmul") {
        LITERT_LOG(LITERT_INFO, "Custom config: optimize_fq_after_matmul = %s",
                   value.c_str());
        context.eliminate_fq_after_matmul_ = (value == "true");
        continue;
      }
      context.configs_map_[key] = value;
      LITERT_LOG(LITERT_INFO, "Custom config: %s = %s", key.c_str(),
                 value.c_str());
    }
  }

  // Configure performance mode (can be overridden by custom options).
  switch (performance_mode) {
    case kLiteRtIntelOpenVinoPerformanceModeLatency:
      if (context.configs_map_.find(ov::hint::performance_mode.name()) ==
          context.configs_map_.end()) {
        context.configs_map_[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::LATENCY;
        LITERT_LOG(LITERT_INFO, "Performance mode: LATENCY");
      }
      break;
    case kLiteRtIntelOpenVinoPerformanceModeThroughput:
      if (context.configs_map_.find(ov::hint::performance_mode.name()) ==
          context.configs_map_.end()) {
        context.configs_map_[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::THROUGHPUT;
        LITERT_LOG(LITERT_INFO, "Performance mode: THROUGHPUT");
      }
      break;
    case kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput:
      if (context.configs_map_.find(ov::hint::performance_mode.name()) ==
          context.configs_map_.end()) {
        context.configs_map_[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
        LITERT_LOG(LITERT_INFO, "Performance mode: CUMULATIVE_THROUGHPUT");
      }
      break;
  }

  return context;
}

LiteRtStatus OpenVinoCompileContext::ConfigureForSoc(const char* soc_model) {
  if (device_ == "NPU") {
    return litert::openvino::ConfigureCompilationParams(soc_model,
                                                        configs_map_);
  }
  return kLiteRtStatusOk;
}

void OpenVinoCompileContext::OptimizeModel(
    const std::shared_ptr<ov::Model>& model) const {
  if (device_ == "NPU") {
    NpuOptimizer()
        .SetEliminateMatMulFakeQuantize(eliminate_fq_after_matmul_)
        .Run(model);
  }
}

}  // namespace openvino
}  // namespace litert
