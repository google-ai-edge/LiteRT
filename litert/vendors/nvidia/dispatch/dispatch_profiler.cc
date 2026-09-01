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

#include "litert/vendors/nvidia/dispatch/dispatch_profiler.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::nvidia {
namespace {

bool EnvironmentFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

litert::Expected<void> CudaOk(cudaError_t error, const char* what) {
  if (error == cudaSuccess) {
    return {};
  }
  return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                       std::string(what) + ": " + cudaGetErrorString(error));
}

}  // namespace

bool DispatchProfilingEnabled() {
  return EnvironmentFlagEnabled("LITERT_NVIDIA_DISPATCH_PROFILE");
}

bool DispatchLayerProfilingEnabled() {
  return EnvironmentFlagEnabled("LITERT_NVIDIA_DISPATCH_LAYER_PROFILE");
}

DispatchInvocationProfiler::~DispatchInvocationProfiler() { DestroyEvents(); }

litert::Expected<void> DispatchInvocationProfiler::Begin(cudaStream_t stream) {
  cpu_start_ = absl::Now();
  LITERT_RETURN_IF_ERROR(EnsureEvents());
  LITERT_RETURN_IF_ERROR(
      CudaOk(cudaEventRecord(event_start_, stream), "cudaEventRecord"));
  return {};
}

litert::Expected<void> DispatchInvocationProfiler::RecordInputsReady(
    cudaStream_t stream) {
  return CudaOk(cudaEventRecord(event_after_h2d_, stream), "cudaEventRecord");
}

litert::Expected<void> DispatchInvocationProfiler::RecordEnqueued(
    cudaStream_t stream) {
  return CudaOk(cudaEventRecord(event_after_enqueue_, stream),
                "cudaEventRecord");
}

litert::Expected<void> DispatchInvocationProfiler::RecordOutputsReady(
    cudaStream_t stream) {
  return CudaOk(cudaEventRecord(event_after_d2h_, stream), "cudaEventRecord");
}

litert::Expected<void> DispatchInvocationProfiler::Finish(
    const char* function_name, const DispatchProfileMetrics& metrics) {
  float h2d_ms = 0.0f;
  float enqueue_ms = 0.0f;
  float d2h_ms = 0.0f;
  LITERT_RETURN_IF_ERROR(
      CudaOk(cudaEventElapsedTime(&h2d_ms, event_start_, event_after_h2d_),
             "cudaEventElapsedTime H2D"));
  LITERT_RETURN_IF_ERROR(CudaOk(
      cudaEventElapsedTime(&enqueue_ms, event_after_h2d_, event_after_enqueue_),
      "cudaEventElapsedTime enqueue"));
  LITERT_RETURN_IF_ERROR(CudaOk(
      cudaEventElapsedTime(&d2h_ms, event_after_enqueue_, event_after_d2h_),
      "cudaEventElapsedTime D2H"));
  const double cpu_total_ms =
      absl::ToDoubleMilliseconds(absl::Now() - cpu_start_);
  LITERT_LOG(LITERT_INFO,
             "NVIDIA dispatch profile function=%s host_inputs=%d "
             "direct_inputs=%d host_outputs=%d direct_outputs=%d "
             "h2d_bytes=%zu d2h_bytes=%zu stream_h2d_ms=%.3f "
             "stream_enqueue_ms=%.3f stream_d2h_ms=%.3f "
             "cpu_input_setup_ms=%.3f cpu_output_setup_ms=%.3f "
             "cpu_enqueue_call_ms=%.3f cpu_output_copy_setup_ms=%.3f "
             "cpu_sync_ms=%.3f cpu_unlock_ms=%.3f "
             "set_address_calls=%d set_address_skips=%d "
             "cpu_total_ms=%.3f",
             function_name, metrics.host_inputs, metrics.direct_inputs,
             metrics.host_outputs, metrics.direct_outputs, metrics.h2d_bytes,
             metrics.d2h_bytes, h2d_ms, enqueue_ms, d2h_ms,
             metrics.cpu_input_setup_ms, metrics.cpu_output_setup_ms,
             metrics.cpu_enqueue_call_ms, metrics.cpu_output_copy_setup_ms,
             metrics.cpu_sync_ms, metrics.cpu_unlock_ms,
             metrics.set_address_calls, metrics.set_address_skips,
             cpu_total_ms);
  return {};
}

litert::Expected<void> DispatchInvocationProfiler::EnsureEvents() {
  if (event_start_ != nullptr && event_after_h2d_ != nullptr &&
      event_after_enqueue_ != nullptr && event_after_d2h_ != nullptr) {
    return {};
  }
  DestroyEvents();
  LITERT_RETURN_IF_ERROR(
      CudaOk(cudaEventCreateWithFlags(&event_start_, cudaEventDefault),
             "cudaEventCreateWithFlags"));
  LITERT_RETURN_IF_ERROR(
      CudaOk(cudaEventCreateWithFlags(&event_after_h2d_, cudaEventDefault),
             "cudaEventCreateWithFlags"));
  LITERT_RETURN_IF_ERROR(
      CudaOk(cudaEventCreateWithFlags(&event_after_enqueue_, cudaEventDefault),
             "cudaEventCreateWithFlags"));
  LITERT_RETURN_IF_ERROR(
      CudaOk(cudaEventCreateWithFlags(&event_after_d2h_, cudaEventDefault),
             "cudaEventCreateWithFlags"));
  return {};
}

void DispatchInvocationProfiler::DestroyEvents() {
  if (event_start_ != nullptr) {
    cudaEventDestroy(event_start_);
    event_start_ = nullptr;
  }
  if (event_after_h2d_ != nullptr) {
    cudaEventDestroy(event_after_h2d_);
    event_after_h2d_ = nullptr;
  }
  if (event_after_enqueue_ != nullptr) {
    cudaEventDestroy(event_after_enqueue_);
    event_after_enqueue_ = nullptr;
  }
  if (event_after_d2h_ != nullptr) {
    cudaEventDestroy(event_after_d2h_);
    event_after_d2h_ = nullptr;
  }
}

void TensorRtLayerProfiler::reportLayerTime(const char* layer_name,
                                            float ms) noexcept {
  auto& entry = accumulated_[layer_name];
  entry.first += ms;
  ++entry.second;
}

void TensorRtLayerProfiler::Dump(const std::string& function_name) const {
  if (accumulated_.empty()) {
    return;
  }
  std::vector<std::pair<std::string, std::pair<double, int>>> rows(
      accumulated_.begin(), accumulated_.end());
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
    return a.second.first > b.second.first;
  });
  double total = 0.0;
  for (const auto& row : rows) {
    total += row.second.first;
  }
  LITERT_LOG(LITERT_INFO, "NVIDIA layer profile %s: layers=%zu total_ms=%.1f",
             function_name.c_str(), rows.size(), total);
  size_t top_limit = 40;
  if (const char* value =
          std::getenv("LITERT_NVIDIA_DISPATCH_LAYER_PROFILE_TOP");
      value != nullptr && value[0] != '\0') {
    size_t parsed = 0;
    if (absl::SimpleAtoi(value, &parsed) && parsed > 0) {
      top_limit = parsed;
    }
  }
  const size_t top = std::min<size_t>(rows.size(), top_limit);
  for (size_t i = 0; i < top; ++i) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA layer profile %s: total=%.1fms calls=%d avg=%.3fms "
               "share=%.1f%% name=%.200s",
               function_name.c_str(), rows[i].second.first,
               rows[i].second.second,
               rows[i].second.first / rows[i].second.second,
               100.0 * rows[i].second.first / total, rows[i].first.c_str());
  }
}

}  // namespace litert::nvidia
