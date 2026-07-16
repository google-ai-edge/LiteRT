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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_PROGRAM_CACHE_SERIALIZATION_PROGRAM_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_PROGRAM_CACHE_SERIALIZATION_PROGRAM_CACHE_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"

namespace ml_drift {

// A cache that stores arbitrary binary blobs on disk, designed for program
// caching. It stores metadata in a FlatBuffer at the beginning of the file
// (first 64KB) and appends blob data in 64KB aligned chunks.
class SerializationProgramCache {
 public:
  explicit SerializationProgramCache(int fd);

  // Opens the file at the given path. The file is created if it does not exist.
  explicit SerializationProgramCache(absl::string_view file_path);

  // Opens the file at the given directory with the name
  // "{model_token}_mldrift_program_cache.bin". The file is created if it does
  // not exist.
  SerializationProgramCache(absl::string_view directory_path,
                            absl::string_view model_token);

  // Writes a key-value pair to the cache file.
  // The file descriptor must be open for reading and writing.
  // If the key already exists, its value is updated (appended to the file,
  // old value space is not reclaimed).
  absl::Status Insert(uint64_t key, absl::string_view value);

  // Reads a value for a given key from the cache file.
  // The file descriptor must be open for reading.
  // Returns NOT_FOUND if the key is not present.
  absl::StatusOr<std::string> LookUp(uint64_t key);

 private:
  FileDescriptor fd_;
};

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_PROGRAM_CACHE_SERIALIZATION_PROGRAM_CACHE_H_
