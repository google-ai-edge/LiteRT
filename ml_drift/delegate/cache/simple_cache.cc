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

#include "third_party/odml/litert/ml_drift/delegate/cache/simple_cache.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#include <cstdint>
#include <filesystem>  // NOLINT for path manipulation.
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/status.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/file_util.h"
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/mmap_handle.h"

namespace litert::ml_drift {
namespace {

// Suffix for the compiled shader programs cache file.
constexpr absl::string_view kCompiledCacheSuffix =
    "_mldrift_compiled_cache.bin";

std::string GetCompiledCacheFilePath(absl::string_view serialization_dir,
                                     absl::string_view model_token) {
  return (std::filesystem::path(std::string(serialization_dir)) /
          absl::StrCat(model_token, kCompiledCacheSuffix))
      .string();
}

absl::Status WriteToFd(const ::ml_drift::FileDescriptor& fd,
                       absl::Span<const uint8_t> data) {
  return fd.IsValid() && fd.Write(data.data(), data.size())
             ? absl::OkStatus()
             : absl::InternalError("Failed to write cache file.");
}

}  // namespace

SimpleCache::SimpleCache(::ml_drift::FileDescriptor fd) : fd_(std::move(fd)) {}

SimpleCache::SimpleCache(const std::string& cache_file_path)
    : cache_file_path_(cache_file_path) {}

SimpleCache::SimpleCache(const std::string& serialization_dir,
                         absl::string_view model_token)
    : SimpleCache(GetCompiledCacheFilePath(serialization_dir, model_token)) {}

absl::Status SimpleCache::Load(
    absl::AnyInvocable<absl::Status(absl::Span<const uint8_t>,
                                    ::ml_drift::MMapHandle&) &&>
        callback) {
  ::ml_drift::MMapHandle mmap_handle;
  if (fd_.IsValid()) {
    RETURN_IF_ERROR(mmap_handle.Map(fd_));
  } else {
    RETURN_IF_ERROR(mmap_handle.Map(cache_file_path_.c_str()));
  }
  return std::move(callback)(
      absl::MakeConstSpan(mmap_handle.data(), mmap_handle.size()), mmap_handle);
}

absl::Status SimpleCache::Store(absl::Span<const uint8_t> data) {
  if (fd_.IsValid()) {
    if (!fd_.Truncate(0)) {
      return absl::InternalError("Failed to truncate cache file.");
    }
    return WriteToFd(fd_, data);
  }

  ::ml_drift::FileDescriptor fd = ::ml_drift::FileDescriptor::Open(
      cache_file_path_.c_str(),
      O_WRONLY | O_CREAT | O_TRUNC,  // NOLINT: b/332641196
      0644);
  return WriteToFd(fd, data);
}

std::string SimpleCache::ToString() const {
  if (fd_.IsValid()) {
    return absl::StrCat("<fd=", fd_.Value(), ">");
  }
  return cache_file_path_.empty() ? "<invalid>" : cache_file_path_;
}

}  // namespace litert::ml_drift
