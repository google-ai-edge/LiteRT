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

#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#include <memory>

#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift_delegate/delegate/precision.h"
#include "tflite/c/common.h"
#include "util/hash/farmhash_fingerprint.h"

#if defined(_WIN32)
#include <io.h>
#define F_OK 0
#else  // defined(_WIN32)
#include <unistd.h>
#endif  // defined(_WIN32)

#include <cerrno>  // IWYU pragma: keep
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/serialization_base.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/serialization_weight_cache/build_identifier.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/cache_builder.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/mmap_handle.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_schema_generated.h"
#include "tflite/delegates/serialization.h"

namespace ml_drift {

namespace {

// Returns true if the given path exists.
[[nodiscard]]
bool FileExists(const char* path) {
  return access(path, F_OK) != -1;
}

[[nodiscard]]
std::string JoinPath(absl::string_view path1, absl::string_view path2) {
#if defined(_WIN32)
  char slash = '\\';
#else   // defined(_WIN32)
  char slash = '/';
#endif  // defined(_WIN32)
  return (path1.empty() || path1.back() == slash)
             ? absl::StrCat(path1, path2)
             : absl::StrCat(path1, std::string(1, slash), path2);
}

}  // namespace

absl::Status SerializationWeightCache::StartBuild(
    absl::string_view directory, absl::string_view model_token,
    uint64_t unique_model_identifier) {
  if (IsBuilding()) {
    return absl::InvalidArgumentError(
        "Cannot start building when the cache is already building.");
  }
  RETURN_IF_ERROR(SetFilePath(directory, model_token));
  temporary_file_descriptor_.Close();
  return StartBuildInternal(unique_model_identifier);
}

absl::Status SerializationWeightCache::StartBuildInternal(
    uint64_t unique_model_identifier) {
  unique_model_identifier_ = unique_model_identifier;
  if (!builder_.IsStarted()) {
    absl::Status status;
    if (temporary_file_descriptor_.IsValid()) {
      status = builder_.Start(std::move(temporary_file_descriptor_),
                              unique_model_identifier_,
                              /*append=*/!is_first_build_);
    } else {
      status = builder_.Start(file_path_, unique_model_identifier_,
                              /*append=*/!is_first_build_);
    }
    if (!status.ok()) {
      return status;
    }
  }
  is_first_build_ = false;
  RETURN_IF_ERROR(builder_.StartBuildStep(unique_model_identifier_));
  // Duplicate the file descriptor to avoid losing the temporary file when
  // the builder is reset. The file descriptor is a RAII object. It will be
  // cleaned up when the builder_ is destroyed.
  temporary_file_descriptor_ = builder_.GetFileDescriptor().Duplicate();
  is_building_ = true;

  return absl::OkStatus();
}

absl::Status SerializationWeightCache::StartBuild(
    int fd, uint64_t unique_model_identifier) {
  if (IsBuilding()) {
    // Close the fd since we took ownership of it.
    if (fd >= 0) {
      close(fd);
    }
    return absl::InvalidArgumentError(
        "Cannot start building when the cache is already building.");
  }
  temporary_file_descriptor_.Reset(fd);
  file_path_ = absl::StrCat("FD:", fd);
  return StartBuildInternal(unique_model_identifier);
}

absl::Status SerializationWeightCache::StopBuild() {
  // If we never started building, then we can exit early.
  if (!IsBuilding()) {
    return absl::OkStatus();
  }
  RETURN_IF_ERROR(builder_.StopBuildStep());
  is_building_ = false;
  return absl::OkStatus();
}

absl::Status SerializationWeightCache::Load(absl::string_view directory,
                                            absl::string_view model_token,
                                            uint64_t unique_model_identifier) {
  RETURN_IF_ERROR(SetFilePath(directory, model_token));

  bool opened_internally = false;
  if (!temporary_file_descriptor_.IsValid()) {
    if (!FileExists(file_path_.c_str())) {
      return absl::NotFoundError(absl::StrCat(
          "MLDrift shared memory serialization cache: could not load ",
          file_path_, ": ", strerror(errno)));
    }
    temporary_file_descriptor_ = FileDescriptor::Open(
        file_path_.c_str(), O_RDONLY);  // NOLINT: b/332641196
    opened_internally = true;
  }

  auto status =
      LoadInternal(temporary_file_descriptor_, unique_model_identifier);
  if (!status.ok() && opened_internally) {
    temporary_file_descriptor_.Close();
  }
  return status;
}

absl::Status SerializationWeightCache::LoadInternal(
    const FileDescriptor& fd, uint64_t unique_model_identifier) {
  mmap_buffer_base_offset_ = 0;
  global_tensor_id_to_cache_entry_.clear();
  mmap_handles_.resize(2);
  ScopeGuard unmap_on_fail([this] { mmap_handles_.clear(); });

  // Mmap just the header for now.
  MMapHandle& header_handle = mmap_handles_.at(0);
  RETURN_IF_ERROR(header_handle.Map(fd,
                                    /*offset=*/0, sizeof(MLDriftCacheHeader),
                                    file_path_.c_str()));

  if (header_handle.size() < sizeof(MLDriftCacheHeader)) {
    return absl::InternalError("Invalid cache file size");
  }

  const MLDriftCacheHeader header = [&header_handle] {
    MLDriftCacheHeader header;
    memcpy(&header, header_handle.data(), sizeof(header));
    return header;
  }();

  if (header.version != MLDriftCacheHeader::kVersion) {
    return absl::InternalError(absl::StrCat(
        "Incompatible header version. Got ", header.version, ", expected ",
        MLDriftCacheHeader::kVersion, ". Cache needs to be built again."));
  }

  if (!ml_drift::CheckBuildIdentifier(
          absl::MakeSpan(header.mldrift_build_identifier,
                         sizeof(header.mldrift_build_identifier)))) {
    return absl::InternalError(
        "Incompatible MLDrift version. Cache needs to be built again.");
  }

  unique_model_identifier_ = unique_model_identifier;

  // Mmap just the model cache.
  MMapHandle& buffer_list_handle = mmap_handles_.at(1);
  RETURN_IF_ERROR(buffer_list_handle.Map(fd, header.buffer_list_offset,
                                         header.buffer_list_size,
                                         file_path_.c_str()));

  if (buffer_list_handle.size() < header.buffer_list_size) {
    return absl::InternalError("Invalid buffer list size");
  }

  // Verify the flatbuffer part of the file.
  flatbuffers::Verifier verifier(buffer_list_handle.data(),
                                 header.buffer_list_size);
  if (!ml_drift::cache::schema::VerifyModelCacheBuffer(verifier)) {
    return absl::InternalError("Model cache validation failed.");
  }
  is_first_build_ = false;

  // Load flatbuffer.
  const ml_drift::cache::schema::ModelCache* model_cache =
      ml_drift::cache::schema::GetModelCache(buffer_list_handle.data());
  if (!model_cache) {
    return absl::InternalError("Could not load model cache from cache file.");
  }

  const ml_drift::cache::schema::SubgraphBufferList* current_subgraph = nullptr;
  if (model_cache->subgraphs()) {
    if (model_cache->subgraphs()->size() > kMaxSupportedSubgraphs) {
      return absl::InternalError("Corrupted cache: Too many subgraphs.");
    }
    for (const auto* subgraph : *model_cache->subgraphs()) {
      if (subgraph->unique_model_identifier() == unique_model_identifier_) {
        current_subgraph = subgraph;
        break;
      }
    }
  }

  if (!current_subgraph) {
    return absl::NotFoundError(absl::StrCat(
        "Subgraph ", unique_model_identifier_, " not found in cache."));
  }

  mmap_buffer_base_offset_ = current_subgraph->base_offset();
  if (const auto buffers = current_subgraph->buffers(); buffers) {
    for (auto* buffer : *buffers) {
      if (!buffer) {
        return absl::InternalError("Invalid buffer address in buffer list.");
      }

      // Tensor will be decoded without its data. Its data will be read
      // separately on demand.
      TensorDescriptor tensor_desc;
      RETURN_IF_ERROR(Decode(buffer->tensor_descriptor(), &tensor_desc));

      global_tensor_id_to_cache_entry_.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(buffer->global_tensor_id(),
                                buffer->is_quantization_param_tensor()),
          std::forward_as_tuple(BufferLocation{.offset = buffer->offset(),
                                               .size = buffer->size()},
                                std::move(tensor_desc)));
    }
  }

  unmap_on_fail.Deactivate();
  return absl::OkStatus();
}

absl::Status SerializationWeightCache::Load(int fd,
                                            uint64_t unique_model_identifier) {
  temporary_file_descriptor_.Reset(fd);
  file_path_ = absl::StrCat("FD:", fd);

  return LoadInternal(temporary_file_descriptor_, unique_model_identifier);
}

absl::Status SerializationWeightCache::LookUp(
    uint32_t global_tensor_id, bool is_quantization_param_tensor,
    UnownedDataTensorDescriptor& unowned_data_tensor_desc,
    size_t& page_adjusted_offset, ReleaseDataCallback& release_data_callback) {
  if (IsBuilding()) {
    return absl::InvalidArgumentError(
        "Cannot look up a buffer in a cache that is building.");
  }
  if (!temporary_file_descriptor_.IsValid()) {
    return absl::InvalidArgumentError(
        "Cannot look up a buffer in a cache that is not loaded.");
  }
  auto offset_it = global_tensor_id_to_cache_entry_.find(
      CacheKey{global_tensor_id, is_quantization_param_tensor});
  if (offset_it == global_tensor_id_to_cache_entry_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to look up ", global_tensor_id, " from cache."));
  }
  const auto& cache_entry = offset_it->second;
  TensorDescriptor tensor_desc_no_data = cache_entry.tensor_desc;

  if (cache_entry.location.offset >
      std::numeric_limits<size_t>::max() - mmap_buffer_base_offset_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cache entry offset integer overflow: offset=",
                     cache_entry.location.offset,
                     ", base_offset=", mmap_buffer_base_offset_));
  }
  size_t offset = mmap_buffer_base_offset_ + cache_entry.location.offset;

  auto mmap_handle = std::make_unique<MMapHandle>();
  RETURN_IF_ERROR(mmap_handle->Map(temporary_file_descriptor_, offset,
                                   cache_entry.location.size,
                                   file_path_.c_str()));
  page_adjusted_offset = mmap_handle->offset_page_adjustment();

  unowned_data_tensor_desc = UnownedDataTensorDescriptor(
      tensor_desc_no_data,
      absl::MakeSpan(mmap_handle->data(), cache_entry.location.size));

  // Take ownership of the mmaped memory so the caller can release it after
  // it has been uploaded to the GPU.
  release_data_callback = mmap_handle->TakeOwnership();

  return absl::OkStatus();
}

absl::Status SerializationWeightCache::LookUp(uint32_t global_tensor_id,
                                              bool is_quantization_param_tensor,
                                              TensorDescriptor& tensor_desc) {
  if (IsBuilding()) {
    return absl::InvalidArgumentError(
        "Cannot look up a buffer in a cache that is building.");
  }
  if (!temporary_file_descriptor_.IsValid()) {
    return absl::InvalidArgumentError(
        "Cannot look up a buffer in a cache that is not loaded.");
  }
  auto offset_it = global_tensor_id_to_cache_entry_.find(
      CacheKey{global_tensor_id, is_quantization_param_tensor});
  if (offset_it == global_tensor_id_to_cache_entry_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to look up ", global_tensor_id, " from cache."));
  }
  const auto& cache_entry = offset_it->second;
  tensor_desc = cache_entry.tensor_desc;

  auto file_size = temporary_file_descriptor_.SetPosFromEnd(0);
  if (file_size < 0) {
    return absl::InternalError("Failed to get file size.");
  }

  if (cache_entry.location.offset >
      std::numeric_limits<size_t>::max() - mmap_buffer_base_offset_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cache entry offset integer overflow: offset=",
                     cache_entry.location.offset,
                     ", base_offset=", mmap_buffer_base_offset_));
  }
  size_t offset = mmap_buffer_base_offset_ + cache_entry.location.offset;
  size_t size = cache_entry.location.size;

  if (size > file_size || offset > file_size - size) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cache entry location out of bounds: offset=", offset,
                     ", size=", size, ", file_size=", file_size));
  }

  temporary_file_descriptor_.SetPos(offset);

  std::vector<uint8_t> data(size);

  if (!temporary_file_descriptor_.Read(data.data(), size)) {
    return absl::InternalError("Failed to lookup serialized tensor from file.");
  }

  tensor_desc.SetData(std::move(data));

  return absl::OkStatus();
}

absl::Status SerializationWeightCache::Insert(
    uint32_t global_tensor_id, bool is_quantization_param_tensor,
    const TensorDescriptor& tensor_desc) {
  if (!IsBuilding()) {
    return absl::InvalidArgumentError(
        "Cannot insert a buffer in a cache that is not building.");
  }

  if (auto offset_it = global_tensor_id_to_cache_entry_.find(
          CacheKey{global_tensor_id, is_quantization_param_tensor});
      offset_it != global_tensor_id_to_cache_entry_.end()) {
    return absl::InvalidArgumentError("Tensor already exists in cache.");
  }

  size_t size = tensor_desc.GetData().size();
  void* ptr = builder_.Reserve(size);
  if (ptr == nullptr && size > 0) {
    return absl::ResourceExhaustedError(
        "Failed to allocate memory for cache staging buffer.");
  }
  if (size > 0) {
    std::memcpy(ptr, tensor_desc.GetData().data(), size);
  }

  TensorDescriptor tensor_desc_without_data;
  tensor_desc.CopyWithoutData(&tensor_desc_without_data);

  BufferLocation location;
  RETURN_IF_ERROR(
      builder_.Append(global_tensor_id, is_quantization_param_tensor,
                      tensor_desc_without_data, ptr, size, location));

  global_tensor_id_to_cache_entry_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(global_tensor_id, is_quantization_param_tensor),
      std::forward_as_tuple(std::move(location),
                            std::move(tensor_desc_without_data)));
  return absl::OkStatus();
}

void SerializationWeightCache::Release() {
  global_tensor_id_to_cache_entry_.clear();
  mmap_handles_.clear();
  mmap_buffer_base_offset_ = 0;
  builder_ = CacheBuilder();
}

absl::Status SerializationWeightCache::SetFilePath(
    absl::string_view directory, absl::string_view model_token) {
  if (absl::StrContains(model_token, "/") ||
      absl::StrContains(model_token, "\\") ||
      absl::StrContains(model_token, "..")) {
    return absl::InvalidArgumentError(
        "Invalid model_token: contains path traversal characters.");
  }
  std::string file_name =
      absl::StrCat(model_token, "_mldrift_weight_cache.bin");
  std::string path = JoinPath(directory, file_name);
  if (IsBuilding()) {
    return absl::InvalidArgumentError(
        "Cannot change the path of a cache that has already been loaded.");
  }
  // We try to keep file_path_'s data as stable as possible. Don't overwrite
  // if the path hasn't changed.
  if (file_path_ != path) {
    file_path_ = path;
  }
  return absl::OkStatus();
}

uint64_t SerializationWeightCache::GenerateUniqueModelIdentifier(
    absl::string_view model_token, TfLiteContext* context,
    const TfLiteDelegateParams* delegate_params,
    absl::string_view serialization_prefix,
    const MlDriftDelegatePrecision& precision, bool prefer_texture_weights,
    bool allow_src_quantized_fc_conv_ops, bool prepare_weights_in_batches,
    bool serialize_external_tensors, bool ordered_by_size) {
  // Generate fingerprints for the relevant delegate data options.
  uint64_t precision_fingerprint = farmhash::Fingerprint64(
      reinterpret_cast<const char*>(&precision), sizeof(precision));
  uint64_t prefer_texture_weights_fingerprint = farmhash::Fingerprint64(
      reinterpret_cast<const char*>(&prefer_texture_weights),
      sizeof(prefer_texture_weights));
  uint64_t allow_src_quantized_fc_conv_ops_fingerprint =
      farmhash::Fingerprint64(
          reinterpret_cast<const char*>(&allow_src_quantized_fc_conv_ops),
          sizeof(allow_src_quantized_fc_conv_ops));
  uint64_t ordered_by_size_fingerprint = farmhash::Fingerprint64(
      reinterpret_cast<const char*>(&ordered_by_size), sizeof(ordered_by_size));
  uint64_t alignment_fingerprint = farmhash::Fingerprint64(
      reinterpret_cast<const char*>(&kMinAlignment), sizeof(kMinAlignment));
  uint64_t prepare_weights_in_batches_fingerprint = farmhash::Fingerprint64(
      reinterpret_cast<const char*>(&prepare_weights_in_batches),
      sizeof(prepare_weights_in_batches));
  uint64_t serialize_external_tensors_fingerprint = farmhash::Fingerprint64(
      reinterpret_cast<const char*>(&serialize_external_tensors),
      sizeof(serialize_external_tensors));
  // Combine the fingerprints of the relevant delegate data options into a
  // single fingerprint.
  // copybara:uncomment_begin(google-only)
  // uint64_t options_fingerprint = farmhash::Fingerprint(
      // precision_fingerprint, prefer_texture_weights_fingerprint);
  // options_fingerprint = farmhash::Fingerprint(
      // options_fingerprint, allow_src_quantized_fc_conv_ops_fingerprint);
  // options_fingerprint =
      // farmhash::Fingerprint(options_fingerprint, ordered_by_size_fingerprint);
  // options_fingerprint =
      // farmhash::Fingerprint(options_fingerprint, alignment_fingerprint);
  // options_fingerprint = farmhash::Fingerprint(
      // options_fingerprint, prepare_weights_in_batches_fingerprint);
  // options_fingerprint = farmhash::Fingerprint(
      // options_fingerprint, serialize_external_tensors_fingerprint);
  // copybara:uncomment_end_and_comment_begin
  uint64_t options_fingerprint = util::Fingerprint(
  std::make_pair(precision_fingerprint, prefer_texture_weights_fingerprint));
  options_fingerprint = util::Fingerprint(
  std::make_pair(options_fingerprint,
  allow_src_quantized_fc_conv_ops_fingerprint));
  options_fingerprint = util::Fingerprint(
  std::make_pair(options_fingerprint, ordered_by_size_fingerprint));
  options_fingerprint = util::Fingerprint(
  std::make_pair(options_fingerprint, alignment_fingerprint));
  options_fingerprint = util::Fingerprint(
  std::make_pair(options_fingerprint,
  prepare_weights_in_batches_fingerprint));
  options_fingerprint = util::Fingerprint(
  std::make_pair(options_fingerprint,
  serialize_external_tensors_fingerprint));
  // copybara:comment_end

  // Add "_external_tensors" to prevent collision with the non-external
  // tensors serialization.
  std::string custom_key = absl::StrCat(
      serialization_prefix, options_fingerprint, "_external_tensors");
  // Generate a unique fingerprint for the model and runtime options.
  return tflite::delegates::Serialization::GetFingerprint(
      model_token.data(), custom_key, context, delegate_params);
}

}  // namespace ml_drift
