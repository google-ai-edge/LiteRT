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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_CACHE_SIMPLE_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_CACHE_SIMPLE_CACHE_H_

#include <cstdint>
#include <string>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/file_util.h"
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/mmap_handle.h"

namespace litert::ml_drift {

// Persistent cache file which loads and stores the entire content at once.
// Subclasses may have its own structure serialized in the content.
class SimpleCache {
 public:
  SimpleCache() = default;
  explicit SimpleCache(::ml_drift::FileDescriptor fd);
  explicit SimpleCache(const std::string& cache_file_path);
  SimpleCache(const std::string& serialization_dir,
              absl::string_view model_token);

  virtual ~SimpleCache() = default;

  // Movable only.
  SimpleCache(SimpleCache&&) = default;
  SimpleCache& operator=(SimpleCache&&) = default;
  SimpleCache(const SimpleCache&) = delete;
  SimpleCache& operator=(const SimpleCache&) = delete;

  // Loads the cache file content and call the callback.
  // Caller must take the ownership of mmap_handle if data must not be freed
  // after the callback returns.
  absl::Status Load(
      absl::AnyInvocable<absl::Status(absl::Span<const uint8_t> data,
                                      ::ml_drift::MMapHandle& mmap_handle) &&>
          callback);

  // Stores the data in the cache file. The file has only one entry.
  absl::Status Store(absl::Span<const uint8_t> data);

  // Wether the underlying cache file is valid.
  bool IsValid() const { return fd_.IsValid() || !cache_file_path_.empty(); }

  std::string ToString() const;

 private:
  ::ml_drift::FileDescriptor fd_;
  std::string cache_file_path_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_CACHE_SIMPLE_CACHE_H_
