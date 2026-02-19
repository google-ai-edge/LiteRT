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

#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
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
  auto env_holder = env.GetHolder();
  LITERT_RETURN_IF_ERROR(env_holder.runtime->CreateCompiledModel(
      env_holder.handle, litert_model, compilation_options.Get(),
      &compiled_model));
  return CompiledModelNext(env_holder, litert_model, compiled_model,
                           OwnHandle::kYes);
}

Expected<CompiledModelNext> CompiledModelNext::Create(
    litert::Environment& env, const litert::Model& model,
    const Options& compilation_options) {
  LiteRtModel litert_model = model.Get();
  LiteRtCompiledModel compiled_model;
  auto env_holder = env.GetHolder();
  LITERT_RETURN_IF_ERROR(env_holder.runtime->CreateCompiledModel(
      env_holder.handle, litert_model, compilation_options.Get(),
      &compiled_model));
  return CompiledModelNext(env_holder, litert_model, compiled_model,
                           OwnHandle::kYes);
}

Expected<CompiledModelNext> CompiledModelNext::Create(
    litert::Environment& env, const std::string& model_filename,
    Options& compilation_options) {
  LITERT_RETURN_IF_ERROR(compilation_options.Build());
  LiteRtModel litert_model;
  auto env_holder = env.GetHolder();
  if (auto status = env_holder.runtime->CreateModelFromFile(
          model_filename.c_str(), &litert_model);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to load model from file");
  }
  LiteRtCompiledModel compiled_model;
  if (auto status = env_holder.runtime->CreateCompiledModel(
          env_holder.handle, litert_model, compilation_options.Get(),
          &compiled_model);
      status != kLiteRtStatusOk) {
    env_holder.runtime->DestroyModel(litert_model);
    return Unexpected(status, "Failed to compile model");
  }
  return CompiledModelNext(env_holder, litert_model,
                           /*model_owned=*/OwnHandle::kYes, compiled_model,
                           OwnHandle::kYes);
}

Expected<CompiledModelNext> CompiledModelNext::Create(
    litert::Environment& env, const litert::Model& model,
    litert::HwAccelerators hardware_accelerators) {
  LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
  compilation_options.SetHardwareAccelerators(hardware_accelerators);
  LiteRtModel litert_model = model.Get();
  LiteRtCompiledModel compiled_model;
  auto env_holder = env.GetHolder();
  LITERT_RETURN_IF_ERROR(env_holder.runtime->CreateCompiledModel(
      env_holder.handle, litert_model, compilation_options.Get(),
      &compiled_model));
  return CompiledModelNext(env_holder, litert_model, compiled_model,
                           OwnHandle::kYes);
}

Expected<CompiledModelNext> CompiledModelNext::Create(
    litert::Environment& env, const std::string& model_filename,
    litert::HwAccelerators hardware_accelerators) {
  LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
  compilation_options.SetHardwareAccelerators(hardware_accelerators);
  return Create(env, model_filename, compilation_options);
}

Expected<void> CompiledModelNext::StartMetricsCollection(int detail_level) {
  if (auto status = env_.runtime->CompiledModelStartMetricsCollection(
          Get(), detail_level);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to start metrics collection");
  }
  return {};
}

Expected<CompiledModelNext::Metrics>
CompiledModelNext::StopMetricsCollection() {
  LiteRtMetrics metrics = nullptr;
  LITERT_RETURN_IF_ERROR(env_.runtime->CreateMetrics(&metrics));
  absl::Cleanup metrics_cleanup = [&metrics, runtime = env_.runtime] {
    runtime->DestroyMetrics(metrics);
  };
  LITERT_RETURN_IF_ERROR(
      env_.runtime->CompiledModelStopMetricsCollection(Get(), metrics));
  int num_metrics;
  LITERT_RETURN_IF_ERROR(env_.runtime->GetNumMetrics(metrics, &num_metrics));

  std::vector<Metrics::Metric> compiled_model_metrics;
  compiled_model_metrics.reserve(num_metrics);
  for (int i = 0; i < num_metrics; ++i) {
    LiteRtMetric metric;
    LITERT_RETURN_IF_ERROR(env_.runtime->GetMetric(metrics, i, &metric));
    compiled_model_metrics.push_back({metric.name, metric.value});
  }
  return CompiledModelNext::Metrics{.metrics =
                                        std::move(compiled_model_metrics)};
}

Expected<void> CompiledModelNext::SetSchedulingInfo(
    const LiteRtSchedulingInfo& scheduling_info) const {
  auto status =
      env_.runtime->CompiledModelSetSchedulingInfo(Get(), &scheduling_info);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to set scheduling info");
  }
  return {};
}

Expected<void> CompiledModelNext::ClearSchedulingInfo() const {
  auto status = env_.runtime->CompiledModelSetSchedulingInfo(Get(), nullptr);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to clear scheduling info");
  }
  return {};
}

}  // namespace litert
