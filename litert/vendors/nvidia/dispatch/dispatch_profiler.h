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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_DISPATCH_PROFILER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_DISPATCH_PROFILER_H_

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "cuda_runtime_api.h"
#include "litert/cc/litert_expected.h"
#include "NvInfer.h"

namespace litert::nvidia {

bool DispatchProfilingEnabled();
bool DispatchLayerProfilingEnabled();

// Reads the clock only when dispatch profiling is enabled, keeping the default
// invocation path free of clock calls.
class DispatchCpuTimer {
 public:
  explicit DispatchCpuTimer(bool enabled) : enabled_(enabled) {
    if (enabled_) {
      start_ = absl::Now();
    }
  }

  double ElapsedMs() const {
    if (!enabled_) {
      return 0.0;
    }
    return absl::ToDoubleMilliseconds(absl::Now() - start_);
  }

 private:
  bool enabled_;
  absl::Time start_;
};

struct DispatchProfileMetrics {
  int host_inputs = 0;
  int direct_inputs = 0;
  int host_outputs = 0;
  int direct_outputs = 0;
  size_t h2d_bytes = 0;
  size_t d2h_bytes = 0;
  double cpu_input_setup_ms = 0.0;
  double cpu_output_setup_ms = 0.0;
  double cpu_enqueue_call_ms = 0.0;
  double cpu_output_copy_setup_ms = 0.0;
  double cpu_sync_ms = 0.0;
  double cpu_unlock_ms = 0.0;
  int set_address_calls = 0;
  int set_address_skips = 0;
};

// Owns the optional CUDA events and emits one complete invocation profile.
// Callers retain the stage boundaries so instrumentation cannot reorder work.
class DispatchInvocationProfiler {
 public:
  DispatchInvocationProfiler() = default;
  ~DispatchInvocationProfiler();

  DispatchInvocationProfiler(const DispatchInvocationProfiler&) = delete;
  DispatchInvocationProfiler& operator=(const DispatchInvocationProfiler&) =
      delete;

  litert::Expected<void> Begin(cudaStream_t stream);
  litert::Expected<void> RecordInputsReady(cudaStream_t stream);
  litert::Expected<void> RecordEnqueued(cudaStream_t stream);
  litert::Expected<void> RecordOutputsReady(cudaStream_t stream);
  litert::Expected<void> Finish(const char* function_name,
                                const DispatchProfileMetrics& metrics);

 private:
  litert::Expected<void> EnsureEvents();
  void DestroyEvents();

  absl::Time cpu_start_;
  cudaEvent_t event_start_ = nullptr;
  cudaEvent_t event_after_h2d_ = nullptr;
  cudaEvent_t event_after_enqueue_ = nullptr;
  cudaEvent_t event_after_d2h_ = nullptr;
};

// Accumulates TensorRT per-layer GPU times across invocations and reports them
// when the owning dispatch context is destroyed.
class TensorRtLayerProfiler final : public nvinfer1::IProfiler {
 public:
  void reportLayerTime(const char* layer_name, float ms) noexcept override;
  void Dump(const std::string& function_name) const;

 private:
  std::unordered_map<std::string, std::pair<double, int>> accumulated_;
};

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_DISPATCH_PROFILER_H_
