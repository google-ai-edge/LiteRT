// Copyright 2025 Google LLC.
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

#include "litert/cc/internal/litert_compiled_model_next.h"

#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_metrics.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"

namespace litert {

Expected<CompiledModelNext> CompiledModelNext::Create(
    litert::Environment& env, const litert::Model& model,
    Options& compilation_options) {
  LITERT_RETURN_IF_ERROR(compilation_options.Build());
  LiteRtModel litert_model = model.Get();
  LiteRtCompiledModel compiled_model;
  LITERT_RETURN_IF_ERROR(LiteRtCreateCompiledModel(
      env.Get(), litert_model, compilation_options.Get(), &compiled_model));
  return CompiledModelNext(litert_model, compiled_model, OwnHandle::kYes);
}

Expected<CompiledModelNext> CompiledModelNext::Create(
    litert::Environment& env, const litert::Model& model,
    const Options& compilation_options) {
  LiteRtModel litert_model = model.Get();
  LiteRtCompiledModel compiled_model;
  LITERT_RETURN_IF_ERROR(LiteRtCreateCompiledModel(
      env.Get(), litert_model, compilation_options.Get(), &compiled_model));
  return CompiledModelNext(litert_model, compiled_model, OwnHandle::kYes);
}

Expected<CompiledModelNext> CompiledModelNext::Create(
    litert::Environment& env, const litert::Model& model,
    litert::HwAccelerators hardware_accelerators) {
  LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
  compilation_options.SetHardwareAccelerators(hardware_accelerators);
  LiteRtModel litert_model = model.Get();
  LiteRtCompiledModel compiled_model;
  LITERT_RETURN_IF_ERROR(LiteRtCreateCompiledModel(
      env.Get(), litert_model, compilation_options.Get(), &compiled_model));
  return CompiledModelNext(litert_model, compiled_model, OwnHandle::kYes);
}

Expected<void> CompiledModelNext::StartMetricsCollection(int detail_level) {
  if (auto status =
          LiteRtCompiledModelStartMetricsCollection(Get(), detail_level);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to start metrics collection");
  }
  return {};
}

Expected<CompiledModelNext::Metrics>
CompiledModelNext::StopMetricsCollection() {
  LiteRtMetrics metrics = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtCreateMetrics(&metrics));
  absl::Cleanup metrics_cleanup = [&metrics] { LiteRtDestroyMetrics(metrics); };
  LITERT_RETURN_IF_ERROR(
      LiteRtCompiledModelStopMetricsCollection(Get(), metrics));
  int num_metrics;
  LITERT_RETURN_IF_ERROR(LiteRtGetNumMetrics(metrics, &num_metrics));

  std::vector<Metrics::Metric> compiled_model_metrics;
  compiled_model_metrics.reserve(num_metrics);
  for (int i = 0; i < num_metrics; ++i) {
    LiteRtMetric metric;
    LITERT_RETURN_IF_ERROR(LiteRtGetMetric(metrics, i, &metric));
    compiled_model_metrics.push_back({metric.name, metric.value});
  }
  return CompiledModelNext::Metrics{.metrics =
                                        std::move(compiled_model_metrics)};
}

}  // namespace litert
