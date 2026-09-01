// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_MEMORY_PROFILE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_MEMORY_PROFILE_H_

#include <cstddef>

namespace litert::nvidia {

struct MemoryProfileSnapshot {
  size_t cpu_rss_bytes = 0;
  size_t cpu_peak_rss_bytes = 0;
  size_t cuda_device_free_bytes = 0;
  size_t cuda_device_total_bytes = 0;
  bool cpu_available = false;
  bool cuda_available = false;
};

bool MemoryProfilingEnabled();

MemoryProfileSnapshot CaptureMemoryProfileSnapshot();

// cuda_device_used_bytes is device-wide usage (total - free), not exact
// per-process ownership. Its delta is useful when the profiling run has
// exclusive access to the device. CPU RSS and peak RSS are process-local.
void LogMemoryProfile(const char* component, const char* phase,
                      const char* context = nullptr);

// Binds a component name once so compiler and dispatch code only declares the
// profiling phase and context at each instrumentation point.
class MemoryProfiler {
 public:
  explicit MemoryProfiler(const char* component) : component_(component) {}

  bool enabled() const { return MemoryProfilingEnabled(); }
  void Log(const char* phase, const char* context = nullptr) const {
    LogMemoryProfile(component_, phase, context);
  }

 private:
  const char* component_;
};

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_MEMORY_PROFILE_H_
