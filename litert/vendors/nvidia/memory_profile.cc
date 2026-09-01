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

#include "litert/vendors/nvidia/memory_profile.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_runtime_api.h"
#include "litert/c/internal/litert_logging.h"

namespace litert::nvidia {

bool MemoryProfilingEnabled() {
  const char* value = std::getenv("LITERT_NVIDIA_MEMORY_PROFILE");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

MemoryProfileSnapshot CaptureMemoryProfileSnapshot() {
  MemoryProfileSnapshot snapshot;

  if (std::FILE* status = std::fopen("/proc/self/status", "r");
      status != nullptr) {
    char line[256];
    size_t rss_kb = 0;
    size_t peak_rss_kb = 0;
    while (std::fgets(line, sizeof(line), status) != nullptr) {
      if (std::sscanf(line, "VmRSS: %zu kB", &rss_kb) == 1) {
        continue;
      }
      std::sscanf(line, "VmHWM: %zu kB", &peak_rss_kb);
    }
    std::fclose(status);
    snapshot.cpu_rss_bytes = rss_kb << 10;
    snapshot.cpu_peak_rss_bytes = peak_rss_kb << 10;
    snapshot.cpu_available = rss_kb != 0 || peak_rss_kb != 0;
  }

  snapshot.cuda_available =
      cudaMemGetInfo(&snapshot.cuda_device_free_bytes,
                     &snapshot.cuda_device_total_bytes) == cudaSuccess;
  return snapshot;
}

void LogMemoryProfile(const char* component, const char* phase,
                      const char* context) {
  if (!MemoryProfilingEnabled()) {
    return;
  }
  const MemoryProfileSnapshot snapshot = CaptureMemoryProfileSnapshot();
  const uint64_t monotonic_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  const size_t cuda_device_used_bytes =
      snapshot.cuda_device_total_bytes >= snapshot.cuda_device_free_bytes
          ? snapshot.cuda_device_total_bytes - snapshot.cuda_device_free_bytes
          : 0;
  LITERT_LOG(LITERT_INFO,
             "NVIDIA memory profile component=%s phase=%s context=%s "
             "monotonic_ns=%llu "
             "cpu_available=%d cpu_rss_bytes=%zu cpu_peak_rss_bytes=%zu "
             "cuda_available=%d cuda_device_used_bytes=%zu "
             "cuda_device_free_bytes=%zu cuda_device_total_bytes=%zu",
             component, phase, context != nullptr ? context : "-",
             static_cast<unsigned long long>(monotonic_ns),
             snapshot.cpu_available, snapshot.cpu_rss_bytes,
             snapshot.cpu_peak_rss_bytes, snapshot.cuda_available,
             cuda_device_used_bytes, snapshot.cuda_device_free_bytes,
             snapshot.cuda_device_total_bytes);
}

}  // namespace litert::nvidia
