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

#include "litert/cc/options/litert_cpu_options.h"

#include <cstdint>

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

Expected<CpuOptions> CpuOptions::Create() {
  LrtCpuOptions* options = nullptr;
  LITERT_RETURN_IF_ERROR(LrtCreateCpuOptions(&options));
  return CpuOptions(options);
}

CpuOptions::CpuOptions(LrtCpuOptions* options) : options_(options) {}

Expected<void> CpuOptions::SetNumThreads(int num_threads) {
  LITERT_RETURN_IF_ERROR(
      LrtSetCpuOptionsNumThread(options_.get(), num_threads));
  return {};
}

Expected<int> CpuOptions::GetNumThreads() const {
  int num_threads;
  auto s = LrtGetCpuOptionsNumThread(options_.get(), &num_threads);
  if (s == kLiteRtStatusErrorNotFound) {
    return 0;
  }
  LITERT_RETURN_IF_ERROR(s);
  return num_threads;
}

Expected<void> CpuOptions::SetXNNPackFlags(uint32_t flags) {
  LITERT_RETURN_IF_ERROR(LrtSetCpuOptionsXNNPackFlags(options_.get(), flags));
  return {};
}

Expected<uint32_t> CpuOptions::GetXNNPackFlags() const {
  uint32_t flags;
  auto s = LrtGetCpuOptionsXNNPackFlags(options_.get(), &flags);
  if (s == kLiteRtStatusErrorNotFound) {
    return 0;
  }
  LITERT_RETURN_IF_ERROR(s);
  return flags;
}

Expected<void> CpuOptions::SetXNNPackWeightCachePath(const char* path) {
  LITERT_RETURN_IF_ERROR(
      LrtSetCpuOptionsXnnPackWeightCachePath(options_.get(), path));
  return {};
}

Expected<absl::string_view> CpuOptions::GetXNNPackWeightCachePath() const {
  const char* path;
  auto s = LrtGetCpuOptionsXnnPackWeightCachePath(options_.get(), &path);
  if (s == kLiteRtStatusErrorNotFound) {
    return absl::string_view();
  }
  LITERT_RETURN_IF_ERROR(s);
  return absl::NullSafeStringView(path);
}

Expected<void> CpuOptions::SetXNNPackWeightCacheFileDescriptor(int fd) {
  LITERT_RETURN_IF_ERROR(
      LrtSetCpuOptionsXnnPackWeightCacheFileDescriptor(options_.get(), fd));
  return {};
}

Expected<int> CpuOptions::GetXNNPackWeightCacheFileDescriptor() const {
  int fd;
  auto s =
      LrtGetCpuOptionsXnnPackWeightCacheFileDescriptor(options_.get(), &fd);
  if (s == kLiteRtStatusErrorNotFound) {
    return -1;
  }
  LITERT_RETURN_IF_ERROR(s);
  return fd;
}

}  // namespace litert
