/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_PERSISTENT_WEIGHT_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_PERSISTENT_WEIGHT_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift

namespace ml_drift {
class SerializationWeightCache;
}  // namespace ml_drift

namespace litert::tensor {

// Options of `PersistentWeightCache::Open` (at namespace scope so that `Open`
// can default them).
struct PersistentWeightCacheOptions {
  // Memory-maps the weights (`ReleaseMappedWeights`) instead of reading them
  // into descriptors (`ReleaseDescriptors`).
  bool map_data = false;
  // Checks the CRC32C of all bytes of every weight. Otherwise only the first
  // and last `PersistentWeightCache::kSampleBytes` of each (reading every
  // byte would page in the whole file when mapping); the index and the file
  // size are always checked.
  bool verify_all_bytes = true;
};

// A file of packed constant weight descriptors (the GPU-layout weights of
// `CompiledMlDriftModelRunner::SharedWeightsCache`, keyed by weight name), so
// that later runs skip packing (or converting) the weights. Uses the file
// format of the ML Drift delegate's `SerializationWeightCache` (which also
// invalidates files written by a different ML Drift weights layout version),
// with an index entry mapping weight names to entries, their file offsets and
// CRC32Cs. The data of every weight starts at a multiple of `kEntryAlignment`
// in the file, so that memory-mapped weights can back GPU buffers directly.
//
// The `identifier` passed to `Open` must change whenever the packed bytes
// could: model weights, GPU and driver, and every option that selects weight
// layouts (see `WeightCacheIdentifier`).
class PersistentWeightCache {
 public:
  // Bump when the packed layouts of the Tensor API runners or this file's
  // index change (layouts computed by ML Drift are covered by
  // `SerializationWeightCache`'s build identifier).
  static constexpr uint32_t kFormatVersion = 2;
  // File offset alignment of every weight's data.
  static constexpr size_t kEntryAlignment = 4096;
  // Bytes at each end of a weight checked when not checking all bytes.
  static constexpr size_t kSampleBytes = 4096;

  using Options = PersistentWeightCacheOptions;

  // A memory-mapped weight of a loaded cache.
  struct MappedWeight {
    // Without data.
    ::ml_drift::TensorDescriptor desc;
    absl::Span<const uint8_t> data;
    // Keeps `data` mapped; the last reference unmaps it.
    std::shared_ptr<const void> mapping;
  };

  // Loads `path` if it holds a valid cache for `identifier` (`loaded()`: every
  // entry read or mapped, and checked). Otherwise (missing, corrupt or stale
  // file) starts writing a new cache to `path` (`building()`), replacing the
  // file when `Finish()` succeeds. Errors only if neither works.
  static absl::StatusOr<std::unique_ptr<PersistentWeightCache>> Open(
      absl::string_view path, uint64_t identifier, const Options& options = {});

  ~PersistentWeightCache();

  // Loaded without `Options::map_data`: the descriptors (with data) by weight
  // name; moves them out (the cache doesn't keep them).
  absl::flat_hash_map<std::string, ::ml_drift::TensorDescriptor>
  ReleaseDescriptors();

  // Loaded with `Options::map_data`: the mapped weights by name; moves them
  // out (the cache doesn't keep them). The mappings stay valid after this
  // cache is destroyed.
  absl::flat_hash_map<std::string, MappedWeight> ReleaseMappedWeights();

  // Building: writes `desc` (with data) under `name`. Ignored if `name` is
  // already inserted.
  absl::Status Insert(const std::string& name,
                      const ::ml_drift::TensorDescriptor& desc);

  // Building: writes the index and replaces `path` with the new cache. No
  // inserts afterwards.
  absl::Status Finish();

  // Appends `extra_entries` (skipping names already in `path`) to an existing
  // valid cache at `path` for `identifier` via a temporary file renamed over
  // `path`. On Apple APFS when `!verify_all_bytes`, clones `path` with
  // copy-on-write (`clonefile`) and appends only the new entries plus a delta
  // index; falls back to rewriting the cache on other filesystems or when
  // `verify_all_bytes` is true. Returns OK without touching (or checking)
  // `path` when `extra_entries` is empty; otherwise fails if `path` cannot be
  // opened and verified for `identifier`. Static: it works on the file at
  // `path`, not on an opened cache.
  static absl::Status Append(
      absl::string_view path, uint64_t identifier,
      absl::Span<const std::pair<std::string, ::ml_drift::TensorDescriptor>>
          extra_entries,
      bool verify_all_bytes = true);

  // Tests: exercises the incremental delta-index append path on any filesystem
  // by copying `path` to a temporary file before appending.
  static absl::Status AppendIncrementalForTesting(
      absl::string_view path, uint64_t identifier,
      absl::Span<const std::pair<std::string, ::ml_drift::TensorDescriptor>>
          extra_entries);

  // Internal: an index entry, the location and CRC32Cs of one weight in the
  // file. Public only so that the index (de)serialization in the .cc file can
  // use it.
  struct IndexValue {
    // Entry id in the `SerializationWeightCache` file.
    uint32_t id = 0;
    // CRC32C of all bytes, and of the first and last `kSampleBytes`.
    uint32_t crc = 0;
    uint32_t sample_crc = 0;
    // File offset and size of the data.
    uint64_t offset = 0;
    uint64_t size = 0;
  };

  bool loaded() const { return loaded_; }
  bool building() const { return building_; }
  const std::string& path() const { return path_; }
  size_t num_entries() const { return num_entries_; }
  size_t data_bytes() const { return data_bytes_; }

 private:
  PersistentWeightCache();

  absl::Status Load(uint64_t identifier, const Options& options);
  absl::Status StartBuild(uint64_t identifier);
  absl::Status InsertRaw(const std::string& name,
                         const ::ml_drift::TensorDescriptor& desc,
                         absl::Span<const uint8_t> data,
                         std::optional<uint32_t> expected_crc = std::nullopt);
  // Building: pads the file so that the next entry's data starts at a multiple
  // of `kEntryAlignment`; returns that offset.
  absl::StatusOr<uint64_t> AlignNextEntry();

  std::string path_;
  std::string tmp_path_;
  std::unique_ptr<::ml_drift::SerializationWeightCache> cache_;
  bool loaded_ = false;
  bool building_ = false;
  absl::flat_hash_map<std::string, ::ml_drift::TensorDescriptor> descriptors_;
  absl::flat_hash_map<std::string, MappedWeight> mapped_;
  // Building: the file being written (owned by `cache_`), and the entry id,
  // CRC32Cs and location by name.
  int build_fd_ = -1;
  uint32_t num_padding_entries_ = 0;
  absl::flat_hash_map<std::string, IndexValue> index_;
  size_t num_entries_ = 0;
  size_t data_bytes_ = 0;
};

// An identifier for `PersistentWeightCache::Open` from a caller-provided model
// token (e.g. a fingerprint of the checkpoint files) and every other input of
// the packed weights (`gpu_key`: GPU name, vendor and driver; `options_key`:
// precision, storage type, hints, ...).
uint64_t WeightCacheIdentifier(absl::string_view model_token,
                               absl::string_view gpu_key,
                               absl::string_view options_key);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_PERSISTENT_WEIGHT_CACHE_H_
