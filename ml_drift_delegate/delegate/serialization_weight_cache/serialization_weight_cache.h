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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_WEIGHT_CACHE_SERIALIZATION_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SERIALIZATION_WEIGHT_CACHE_SERIALIZATION_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/precision.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/cache_builder.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/mmap_handle.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_schema_generated.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "tflite/c/common.h"

namespace ml_drift {

using ::litert::ml_drift::ReleaseDataCallback;
using ::litert::ml_drift::UnownedDataTensorDescriptor;

absl::StatusOr<ml_drift::cache::schema::PackingAlgorithm> ToPackingAlgorithm(
    Layout layout);
absl::StatusOr<ml_drift::cache::schema::PackingAlgorithm> ToPackingAlgorithm(
    WeightsLayout layout);
// The key to identify a cache entry. It is not enough to use the global tensor
// id to identify a cache entry because quantization tensors have a different
// global tensor id space.
struct CacheKey {
  CacheKey(uint32_t global_tensor_id, bool is_quantization_param_tensor)
      : global_tensor_id(global_tensor_id),
        is_quantization_param_tensor(is_quantization_param_tensor) {}

  // The global tensor id for the tensor.
  uint32_t global_tensor_id;
  // If the tensor is a quantization parameter tensor, it will have a different
  // global tensor id space. These two fields must be used together to identify
  // an entry.
  bool is_quantization_param_tensor;

  bool operator==(const CacheKey& other) const {
    return global_tensor_id == other.global_tensor_id &&
           is_quantization_param_tensor == other.is_quantization_param_tensor;
  }

  struct Hash {
    size_t operator()(const CacheKey& key) const {
      std::hash<uint64_t> hasher;
      return hasher(key.global_tensor_id) ^
             hasher(key.is_quantization_param_tensor);
    }
  };
};

// The location and descriptor of a buffer in the cache. This information is
// used to construct the flatbuffer.
struct CacheEntry {
  CacheEntry(BufferLocation location, TensorDescriptor tensor_desc,
             ml_drift::cache::schema::PackingAlgorithm packing_algorithm)
      : location(std::move(location)),
        tensor_desc(std::move(tensor_desc)),
        packing_algorithm(packing_algorithm) {}

  BufferLocation location;
  TensorDescriptor tensor_desc;
  ml_drift::cache::schema::PackingAlgorithm packing_algorithm;
};

// Allows MLDrift to directly load packed weights from disk instead of having to
// repack them every time.
class SerializationWeightCache {
 public:
  SerializationWeightCache() = default;

  // Non-copyable.
  SerializationWeightCache(const SerializationWeightCache&) = delete;
  SerializationWeightCache& operator=(const SerializationWeightCache&) = delete;

  // Non-moveable.
  SerializationWeightCache(SerializationWeightCache&&) = delete;
  SerializationWeightCache& operator=(SerializationWeightCache&&) = delete;

  // Makes the cache provider ready for new entries to be inserted via Insert().
  // When insertion is done, StopBuild() should be called to write the data to
  // disk.
  absl::Status StartBuild(absl::string_view directory,
                          absl::string_view model_token,
                          uint64_t unique_model_identifier);
  // Starts building the cache using the given file descriptor. The file
  // descriptor is owned by the cache.
  absl::Status StartBuild(int fd, uint64_t unique_model_identifier);

  // Writes the new cache entries from Insert() to disk at the file path
  // specified in StartBuild().
  absl::Status StopBuild();

  // Loads the cache from the given path.
  absl::Status Load(absl::string_view directory, absl::string_view model_token,
                    uint64_t unique_model_identifier);
  // Loads the cache from the given file descriptor. The file descriptor is
  // owned by the cache.
  absl::Status Load(int fd, uint64_t unique_model_identifier);

  // Returns the TensorDescriptor for the given global_tensor_id. Will error if
  // the global_tensor_id is not found or if the packing_algorithm doesn't
  // match.
  absl::Status LookUp(
      uint32_t global_tensor_id, bool is_quantization_param_tensor,
      ml_drift::cache::schema::PackingAlgorithm packing_algorithm,
      TensorDescriptor& tensor_desc);

  // Similar to the above, but returns an unowned data tensor descriptor which
  // is backed by mmap'd memory. This is used to avoid unnecessary copies when
  // creating tensors from the cache. This approach is useful on Apple platforms
  // where the CPU/GPU memory is unified.
  //
  // Returns:
  //   - unowned_data_tensor_desc: The unowned data tensor descriptor that has a
  //       pointer to the data.
  //   - page_adjusted_offset: The offset of the data in the mmaped memory.
  //   - release_data_callback: A callback to release the data when it is no
  //       longer needed. The user is responsible for calling this callback
  //       when they are done using the data.
  absl::Status LookUp(
      uint32_t global_tensor_id, bool is_quantization_param_tensor,
      ml_drift::cache::schema::PackingAlgorithm packing_algorithm,
      UnownedDataTensorDescriptor& unowned_data_tensor_desc,
      size_t& page_adjusted_offset, ReleaseDataCallback& release_data_callback);

  // Returns true if the cache is ready for Insert() to be called.
  bool IsReadyForInsert() const { return IsBuilding(); }

  // Inserts a new TensorDescriptor into the cache using the global_tensor_id as
  // the key.
  absl::Status Insert(
      uint32_t global_tensor_id, bool is_quantization_param_tensor,
      ml_drift::cache::schema::PackingAlgorithm packing_algorithm,
      const TensorDescriptor& tensor_desc);

  // Releases the cache's memory.
  void Release();

  // Returns the current size of the cache.
  size_t GetCurrentSize() const {
    return global_tensor_id_to_cache_entry_.size();
  }

  // A general helper function for generating a unique model + runtime options
  // for the cache. Users may choose to use this function to generate a unique
  // identifier for their cache or they may choose to use their own custom
  // logic.
  static uint64_t GenerateUniqueModelIdentifier(
      absl::string_view model_token, TfLiteContext* context,
      const TfLiteDelegateParams* delegate_params,
      absl::string_view serialization_prefix,
      const MlDriftDelegatePrecision& precision, bool prefer_texture_weights,
      bool allow_src_quantized_fc_conv_ops, bool prepare_weights_in_batches,
      bool serialize_external_tensors, bool ordered_by_size);

 private:
  friend class SerializationWeightCacheTestPeer;

  absl::Status LoadInternal(const FileDescriptor& fd,
                            uint64_t unique_model_identifier);

  absl::Status StartBuildInternal(uint64_t unique_model_identifier);

  absl::Status SetFilePath(absl::string_view directory,
                           absl::string_view model_token);

  // Returns true if any weights have been added to the underlying builder.
  bool IsBuilding() const { return is_building_; };

  // Path to the cache file.
  std::string file_path_;

  // Unique model identifier for the cache file.
  uint64_t unique_model_identifier_;

  // Mapping of the global tensor id to the cache entry.
  std::unordered_multimap<CacheKey, CacheEntry, CacheKey::Hash>
      global_tensor_id_to_cache_entry_;

  // MMap allocation handler.
  std::vector<MMapHandle> mmap_handles_;

  // The offset to the first buffer data in the MMap allocation.
  size_t mmap_buffer_base_offset_;

  // Can hold a file descriptor when building a temporary cache to prevent it
  // from being deleted.
  FileDescriptor temporary_file_descriptor_;

  // Used to build the cache.
  CacheBuilder builder_;

  // This is used to check whether the builder is active, which means that
  // some of the buffers are not available/can't be retrieved.
  bool is_building_ = false;

  // Stores the loaded buffer addresses corresponding to the given offset in
  // the cache file.
  std::map<size_t, void*> offset_to_addr_;

  // True if this is the first build in the current instance. Used to determine
  // whether to truncate or append to the file.
  bool is_first_build_ = true;
};

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_SERIALIZATION_WEIGHT_CACHE_SERIALIZATION_CACHE_H_
