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

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert {

absl::string_view CpuOptions::Identifier() {
  return LiteRtGetCpuOptionsIdentifier();
}

Expected<CpuOptions> CpuOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtCreateCpuOptions(&options));
  return CpuOptions(options, OwnHandle::kYes);
}

Expected<CpuOptions> CpuOptions::Create(OpaqueOptions& original_options) {
  LITERT_ASSIGN_OR_RETURN(absl::string_view original_identifier,
                          original_options.GetIdentifier());
  LITERT_RETURN_IF_ERROR(original_identifier == Identifier(),
                         ErrorStatusBuilder::InvalidArgument())
      << "Cannot create CPU options from an opaque options object that doesn't "
         "already hold CPU options.";
  LiteRtOpaqueOptions options = original_options.Get();
  return CpuOptions(options, OwnHandle::kNo);
}

Expected<void> CpuOptions::SetNumThreads(int num_threads) {
  LiteRtCpuOptions cpu_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCpuOptions(Get(), &cpu_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetCpuOptionsNumThread(cpu_options, num_threads));
  return {};
}

Expected<int> CpuOptions::GetNumThreads() const {
  LiteRtCpuOptions cpu_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCpuOptions(Get(), &cpu_options));
  int num_threads;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetCpuOptionsNumThread(cpu_options, &num_threads));
  return num_threads;
}

Expected<void> CpuOptions::SetXNNPackFlags(uint32_t flags) {
  LiteRtCpuOptions cpu_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCpuOptions(Get(), &cpu_options));
  LITERT_RETURN_IF_ERROR(LiteRtSetCpuOptionsXNNPackFlags(cpu_options, flags));
  return {};
}

Expected<uint32_t> CpuOptions::GetXNNPackFlags() const {
  LiteRtCpuOptions cpu_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCpuOptions(Get(), &cpu_options));
  uint32_t flags;
  LITERT_RETURN_IF_ERROR(LiteRtGetCpuOptionsXNNPackFlags(cpu_options, &flags));
  return flags;
}

Expected<void> CpuOptions::SetXNNPackWeightCachePath(const char* path) {
  LiteRtCpuOptions cpu_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCpuOptions(Get(), &cpu_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetCpuOptionsXnnPackWeightCachePath(cpu_options, path));
  return {};
}

Expected<absl::string_view> CpuOptions::GetXNNPackWeightCachePath() const {
  LiteRtCpuOptions cpu_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCpuOptions(Get(), &cpu_options));
  const char* path;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetCpuOptionsXnnPackWeightCachePath(cpu_options, &path));
  return absl::NullSafeStringView(path);
}

Expected<void> CpuOptions::SetXNNPackWeightCacheFileDescriptor(int fd) {
  LiteRtCpuOptions cpu_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCpuOptions(Get(), &cpu_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetCpuOptionsXnnPackWeightCacheFileDescriptor(cpu_options, fd));
  return {};
}

Expected<int> CpuOptions::GetXNNPackWeightCacheFileDescriptor() const {
  LiteRtCpuOptions cpu_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindCpuOptions(Get(), &cpu_options));
  int fd;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetCpuOptionsXnnPackWeightCacheFileDescriptor(cpu_options, &fd));
  return fd;
}

}  // namespace litert
