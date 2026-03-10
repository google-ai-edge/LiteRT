// Copyright 2024 Google LLC.
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

#include "litert/core/util/flatbuffer_tools.h"

#include <algorithm>
#include <filesystem>  // NOLINT
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_macros.h"
#include "tflite/converter/allocation.h"
#include "tflite/converter/core/model_builder_base.h"

#ifndef NDEBUG
// Make flatbuffers verifier `assert` in debug mode.
#define FLATBUFFERS_DEBUG_VERIFICATION_FAILURE

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers  // IWYU pragma: keep
#endif

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/stderr_reporter.h"

namespace litert::internal {

using ::flatbuffers::Verifier;
using ::tflite::VerifyModelBuffer;

namespace {

Expected<uint32_t> FindMetadataInd(const TflModel& model,
                                   absl::string_view key) {
  tflite::MetadataT* fb_metadata = nullptr;
  for (auto& m : model.metadata) {
    if (m->name == key) {
      fb_metadata = m.get();
      break;
    }
  }
  if (fb_metadata == nullptr) {
    return Error(kLiteRtStatusErrorNotFound);
  }
  return fb_metadata->buffer;
}

std::filesystem::path MakeStdPath(absl::string_view path) {
  return std::filesystem::path(std::string(path.begin(), path.end()));
}

Expected<OwningBufferRef<uint8_t>> ReadFilePrefix(absl::string_view path,
                                                  size_t size) {
  const auto std_path = MakeStdPath(path);
  std::error_code ec;
  const auto raw_file_size = std::filesystem::file_size(std_path, ec);
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO, "Failed to stat file");
  }
  if (raw_file_size > std::numeric_limits<size_t>::max()) {
    return Error(kLiteRtStatusErrorInvalidArgument, "File is too large");
  }
  const size_t file_size = static_cast<size_t>(raw_file_size);
  if (size > file_size) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Read size exceeds file size");
  }

  std::ifstream ifs(std_path, std::ios::binary);
  if (!ifs) {
    return Error(kLiteRtStatusErrorFileIO, "Failed to open file");
  }

  OwningBufferRef<uint8_t> out(size);
  if (size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Prefix read size too large");
  }
  ifs.read(reinterpret_cast<char*>(out.Data()),
           static_cast<std::streamsize>(size));
  if (!ifs) {
    return Error(kLiteRtStatusErrorFileIO, "Failed to read file prefix");
  }
  return out;
}

Expected<size_t> FindFlatbufferRootSizeByVerification(absl::string_view path) {
  const auto std_path = MakeStdPath(path);
  std::error_code ec;
  const auto raw_file_size = std::filesystem::file_size(std_path, ec);
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO, "Failed to stat file");
  }
  if (raw_file_size > std::numeric_limits<size_t>::max()) {
    return Error(kLiteRtStatusErrorInvalidArgument, "File is too large");
  }
  const size_t file_size = static_cast<size_t>(raw_file_size);
  if (file_size == 0) {
    return Error(kLiteRtStatusErrorInvalidArgument, "Empty model file");
  }

  size_t lower_invalid = 0;
  size_t upper_valid = 0;
  size_t probe = std::min<size_t>(file_size, 4096);
  for (;;) {
    LITERT_ASSIGN_OR_RETURN(auto prefix, ReadFilePrefix(path, probe));
    if (VerifyFlatbuffer(prefix.Data(), prefix.Size())) {
      upper_valid = probe;
      break;
    }
    if (probe == file_size) {
      return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                   "Could not verify model flatbuffer");
    }
    lower_invalid = probe;
    probe = std::min<size_t>(file_size, probe * 2);
  }

  while (lower_invalid + 1 < upper_valid) {
    const size_t mid = lower_invalid + (upper_valid - lower_invalid) / 2;
    LITERT_ASSIGN_OR_RETURN(auto prefix, ReadFilePrefix(path, mid));
    if (VerifyFlatbuffer(prefix.Data(), prefix.Size())) {
      upper_valid = mid;
    } else {
      lower_invalid = mid;
    }
  }

  return upper_valid;
}

Expected<OwningBufferRef<uint8_t>> CopyMetadataFromPackedModel(
    const tflite::Model* model, BufferRef<uint8_t> model_buffer,
    absl::string_view key) {
  if (model == nullptr || model->metadata() == nullptr ||
      model->buffers() == nullptr) {
    return Error(kLiteRtStatusErrorNotFound, "Model metadata not found");
  }

  const tflite::Metadata* metadata_entry = nullptr;
  for (size_t i = 0; i < model->metadata()->size(); ++i) {
    const auto* metadata = model->metadata()->Get(i);
    if (metadata != nullptr && metadata->name() != nullptr &&
        metadata->name()->str() == key) {
      metadata_entry = metadata;
      break;
    }
  }
  if (metadata_entry == nullptr) {
    return Error(kLiteRtStatusErrorNotFound,
                 "Requested metadata key not found");
  }
  if (metadata_entry->buffer() >= model->buffers()->size()) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                 "Metadata references invalid buffer");
  }

  const auto* metadata_buffer = model->buffers()->Get(metadata_entry->buffer());
  if (metadata_buffer == nullptr) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                 "Metadata buffer is missing");
  }

  OwningBufferRef<uint8_t> copied;
  if (metadata_buffer->data() != nullptr) {
    copied.Assign(metadata_buffer->data()->data(),
                  metadata_buffer->data()->size());
    return copied;
  }

  const size_t offset = metadata_buffer->offset();
  const size_t size = metadata_buffer->size();
  if (offset > model_buffer.Size() || size > model_buffer.Size() - offset) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                 "Metadata buffer points outside flatbuffer root");
  }
  copied.Assign(model_buffer.Data() + offset, size);
  return copied;
}

bool HasExternalTensorBufferRanges(const tflite::Model* model) {
  if (model == nullptr || model->subgraphs() == nullptr ||
      model->buffers() == nullptr) {
    return false;
  }
  for (size_t subgraph_index = 0; subgraph_index < model->subgraphs()->size();
       ++subgraph_index) {
    const auto* subgraph = model->subgraphs()->Get(subgraph_index);
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
      continue;
    }
    for (size_t tensor_index = 0; tensor_index < subgraph->tensors()->size();
         ++tensor_index) {
      const auto* tensor = subgraph->tensors()->Get(tensor_index);
      if (tensor == nullptr) {
        continue;
      }
      const auto buffer_index = tensor->buffer();
      if (buffer_index == 0 || buffer_index >= model->buffers()->size()) {
        continue;
      }
      const auto* buffer = model->buffers()->Get(buffer_index);
      if (buffer != nullptr && buffer->offset() != 0 && buffer->size() != 0) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

absl::string_view FbBufToStr(const uint8_t* fb_data, size_t size) {
  auto fb_buf_raw = reinterpret_cast<const char*>(fb_data);
  return absl::string_view(fb_buf_raw, size);
}

absl::string_view FbBufToStr(absl::Span<const uint8_t> fb_buf) {
  auto fb_buf_raw = reinterpret_cast<const char*>(fb_buf.data());
  const size_t fb_buf_size = fb_buf.size();
  return absl::string_view(fb_buf_raw, fb_buf_size);
}

absl::Span<char> FbBufToStr(absl::Span<uint8_t> fb_buf) {
  return absl::MakeSpan(reinterpret_cast<char*>(fb_buf.data()), fb_buf.size());
}

absl::Span<char> FbBufToStr(uint8_t* fb_data, size_t size) {
  return absl::MakeSpan(reinterpret_cast<char*>(fb_data), size);
}

bool VerifyFlatbuffer(absl::Span<const uint8_t> buf) {
  return VerifyFlatbuffer(buf.data(), buf.size());
}

bool VerifyFlatbuffer(const uint8_t* buf, size_t buf_size) {
  flatbuffers::Verifier::Options options;
#ifndef NDEBUG
  options.assert = true;
#endif
  flatbuffers::Verifier verifier(buf, buf_size, options);
  return VerifyModelBuffer(verifier);
}

Expected<size_t> GetFlatbufferRootSizeFromFile(absl::string_view path) {
  return FindFlatbufferRootSizeByVerification(path);
}

Expected<OwningBufferRef<uint8_t>> CopyModelMetadataFromBuffer(
    BufferRef<uint8_t> model_buffer, absl::string_view key) {
  if (!VerifyFlatbuffer(model_buffer.Data(), model_buffer.Size())) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer, "Invalid flatbuffer");
  }
  const auto* model = tflite::GetModel(model_buffer.Data());
  return CopyMetadataFromPackedModel(model, model_buffer, key);
}

Expected<OwningBufferRef<uint8_t>> CopyModelMetadataFromFile(
    absl::string_view path, absl::string_view key) {
  LITERT_ASSIGN_OR_RETURN(size_t root_size,
                          FindFlatbufferRootSizeByVerification(path));
  LITERT_ASSIGN_OR_RETURN(auto root_prefix, ReadFilePrefix(path, root_size));
  return CopyModelMetadataFromBuffer(root_prefix, key);
}

Expected<MutableBufferRef<uint8_t>> GetMetadata(absl::string_view key,
                                                TflModel& model) {
  auto buffer_ind = FindMetadataInd(model, key);
  if (!buffer_ind) {
    // Metadata key already has value.
    return buffer_ind.Error();
  }
  auto& fb_vec = model.buffers.at(*buffer_ind)->data;
  return MutableBufferRef<uint8_t>(fb_vec.data(), fb_vec.size());
}

Expected<BufferRef<uint8_t>> GetMetadata(absl::string_view key,
                                         const TflModel& model) {
  auto metadata = GetMetadata(key, const_cast<TflModel&>(model));
  if (!metadata) {
    return metadata.Error();
  }
  return *metadata;
}

LiteRtStatus PushMetadata(absl::string_view key, TflModel& model,
                          BufferRef<uint8_t> metadata) {
  auto buffer_ind = FindMetadataInd(model, key);
  if (buffer_ind) {
    // Metadata key already has value.
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto& new_metadata =
      model.metadata.emplace_back(std::make_unique<tflite::MetadataT>());
  new_metadata->name.assign(key.data(), key.size());

  const auto new_m_buffer_ind = model.buffers.size();
  new_metadata->buffer = new_m_buffer_ind;

  auto& new_buffer = model.buffers.emplace_back(std::make_unique<TflBuffer>());
  new_buffer->data.assign(metadata.Data(), metadata.Data() + metadata.Size());

  return kLiteRtStatusOk;
}

Expected<MutableBufferRef<uint8_t>> GetTflBuffer(TflModel& tfl_model,
                                                 uint32_t buffer_ind) {
  if (buffer_ind >= tfl_model.buffers.size()) {
    return Error(kLiteRtStatusErrorIndexOOB);
  }
  auto& tfl_data = tfl_model.buffers.at(buffer_ind)->data;
  return MutableBufferRef<uint8_t>(tfl_data.data(), tfl_data.size());
}

Expected<BufferRef<uint8_t>> GetTflBuffer(const TflModel& tfl_model,
                                          uint32_t buffer_ind) {
  auto buffer = GetTflBuffer(const_cast<TflModel&>(tfl_model), buffer_ind);
  if (!buffer) {
    return buffer.Error();
  }
  return *buffer;
}

Expected<const TflBuffer*> GetBuffer(const TflModel& tfl_model,
                                     uint32_t buffer_ind) {
  if (buffer_ind >= tfl_model.buffers.size()) {
    return Error(kLiteRtStatusErrorIndexOOB);
  }
  return tfl_model.buffers.at(buffer_ind).get();
}

Expected<TflBufferPtr> TakeBuffer(TflModel& tfl_model, uint32_t buffer_ind) {
  if (buffer_ind >= tfl_model.buffers.size()) {
    return Error(kLiteRtStatusErrorIndexOOB);
  }
  return std::move(tfl_model.buffers.at(buffer_ind));
}

Expected<uint32_t> PushTflBuffer(TflModel& tfl_model,
                                 BufferRef<uint8_t> buffer) {
  tfl_model.buffers.emplace_back(std::make_unique<::tflite::BufferT>())
      ->data.assign(buffer.Data(), buffer.Data() + buffer.Size());
  return tfl_model.buffers.size() - 1;
}

Expected<TflOpCodeEnum> GetTflOpCode(const TflModel& tfl_model,
                                     uint32_t op_code_ind) {
  if (op_code_ind >= tfl_model.operator_codes.size()) {
    return Error(kLiteRtStatusErrorIndexOOB);
  }
  return std::move(tfl_model.operator_codes.at(op_code_ind)->builtin_code);
}

bool IsRankedTensorType(const TflShapeInfo& tfl_shape) {
  return tfl_shape.has_rank;
}

bool IsStaticTensorType(const TflShapeInfo& tfl_shape) {
  return !IsRankedTensorType(tfl_shape) ||
         std::none_of(tfl_shape.shape_signature.begin(),
                      tfl_shape.shape_signature.end(),
                      [](auto d) { return d < 0; });
}

Expected<absl::Span<const int32_t>> AsStaticShape(
    const TflShapeInfo& tfl_shape) {
  if (!IsStaticTensorType(tfl_shape)) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return absl::MakeConstSpan(tfl_shape.shape.data(), tfl_shape.shape.size());
}

Expected<absl::Span<const int32_t>> AsDynamicShape(
    const TflShapeInfo& tfl_shape) {
  auto static_shape = AsStaticShape(tfl_shape);
  if (static_shape) {
    return static_shape;
  }
  if (!IsRankedTensorType(tfl_shape)) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return absl::MakeConstSpan(tfl_shape.shape_signature.data(),
                             tfl_shape.shape_signature.size());
}

bool IsQuantized(const TflQuantization* tfl_quantization) {
  return tfl_quantization &&
         (!tfl_quantization->scale.empty() ||
          tfl_quantization->details.type != tflite::QuantizationDetails_NONE);
}

bool IsPerChannelQuantized(const TflQuantization* tfl_quantization) {
  return tfl_quantization && tfl_quantization->scale.size() > 1;
}

bool IsPerTensorQuantized(const TflQuantization* tfl_quantization) {
  return tfl_quantization && tfl_quantization->scale.size() == 1;
}

bool IsBlockwiseQuantized(const TflQuantization* tfl_quantization) {
  return tfl_quantization &&
         tfl_quantization->details.type ==
             tflite::QuantizationDetails_BlockwiseQuantization;
}

bool IsCustomQuantized(const TflQuantization* tfl_quantization) {
  return tfl_quantization && tfl_quantization->details.type ==
                                 tflite::QuantizationDetails_CustomQuantization;
}

Expected<TflPerTensorQParams> AsPerTensorQparams(
    const TflQuantization* tfl_quantization) {
  if (!IsPerTensorQuantized(tfl_quantization)) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return std::make_pair(tfl_quantization->zero_point.front(),
                        tfl_quantization->scale.front());
}

Expected<TflPerChannelQParams> AsPerChannelQparams(
    const TflQuantization* tfl_quantization) {
  if (!IsPerChannelQuantized(tfl_quantization)) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return TflPerChannelQParams(tfl_quantization->quantized_dimension,
                              tfl_quantization->zero_point.size(),
                              tfl_quantization->zero_point,
                              tfl_quantization->scale);
}

::tflite::Allocation::Ptr MakeAllocation(BufferRef<uint8_t> buf) {
  return std::make_unique<::tflite::MemoryAllocation>(
      buf.Data(), buf.Size(), ::tflite::DefaultErrorReporter());
}

Expected<FlatbufferWrapper::Ptr> FlatbufferWrapper::CreateFromBuffer(
    BufferRef<uint8_t> buffer) {
  static constexpr size_t k2GiB = 2e+9;
  if (buffer.Size() < k2GiB &&
      !VerifyFlatbuffer(buffer.Data(), buffer.Size())) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer, "Invalid flatbuffer");
  }
  auto alloc = MakeAllocation(buffer);
  LITERT_ASSIGN_OR_ABORT(auto wrapper,
                           (CreateFromAllocation(std::move(alloc))));
  return wrapper;
}

Expected<FlatbufferWrapper::Ptr> FlatbufferWrapper::CreateFromAllocation(
    ::tflite::Allocation::Ptr alloc) {
  if (alloc == nullptr) {
    return Error(kLiteRtStatusErrorFileIO, "Invalid allocation");
  }

  auto fb_model =
      ::tflite::FlatBufferModel::BuildFromAllocation(std::move(alloc));
  if (fb_model == nullptr) {
    return Error(kLiteRtStatusErrorFileIO, "Failed to build flatbuffer model");
  }

  return FlatbufferWrapper::Ptr(new FlatbufferWrapper(
      std::move(fb_model)));
}

Expected<FlatbufferWrapper::Ptr> FlatbufferWrapper::CreateFromBuffer(
    OwningBufferRef<uint8_t>&& buffer) {
  LITERT_ASSIGN_OR_ABORT(auto wrapper, (CreateFromBuffer(buffer)));
  // Keep the buffer alive for the lifetime of the wrapper.
  wrapper->model_buf_ = std::move(buffer);
  return wrapper;
}

Expected<FlatbufferWrapper::Ptr> FlatbufferWrapper::CreateFromTflFile(
    absl::string_view path, FileLoadOptions options) {
  if (options.load_mode == FileLoadMode::kMetadataOnlyForFileCopy) {
    LITERT_ASSIGN_OR_RETURN(auto root_size,
                            FindFlatbufferRootSizeByVerification(path));
    LITERT_ASSIGN_OR_RETURN(auto root_prefix, ReadFilePrefix(path, root_size));

    const auto* packed_model = tflite::GetModel(root_prefix.Data());
    if (packed_model == nullptr ||
        !VerifyFlatbuffer(root_prefix.Data(), root_prefix.Size())) {
      return Error(kLiteRtStatusErrorInvalidFlatbuffer,
                   "Failed to verify flatbuffer root for metadata-only load");
    }
    if (HasExternalTensorBufferRanges(packed_model)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Metadata-only model load mode requires full model payload "
                   "because tensor buffers reference external ranges");
    }

    auto metadata_only_wrapper =
        FlatbufferWrapper::CreateFromBuffer(std::move(root_prefix));
    if (!metadata_only_wrapper) {
      return metadata_only_wrapper.Error();
    }
    return std::move(*metadata_only_wrapper);
  }

  auto error_reporter = tflite::DefaultErrorReporter();
  auto allocation = tflite::GetAllocationFromFile(path.data(), error_reporter,
                                                  options.allow_modifications);
  if (allocation == nullptr) {
    return Error(kLiteRtStatusErrorFileIO, "Failed to allocate model file");
  }

  return FlatbufferWrapper::CreateFromAllocation(std::move(allocation));
}

Expected<FlatbufferWrapper::Ptr> FlatbufferWrapper::CreateFromTflFile(
    absl::string_view path, bool allow_modifications) {
  FileLoadOptions options;
  options.allow_modifications = allow_modifications;
  return CreateFromTflFile(path, options);
}

OwningBufferRef<uint8_t> SerializeFlatbuffer(const TflModel& tfl_model) {
  flatbuffers::FlatBufferBuilder b;
  auto model_offset = tflite::Model::Pack(b, &tfl_model);
  tflite::FinishModelBuffer(b, model_offset);

  OwningBufferRef<uint8_t> buffer;
  auto [new_buf, new_size, new_offset] = buffer.GetWeak();
  new_buf = b.ReleaseRaw(new_size, new_offset);

  return buffer;
}

OwningBufferRef<uint8_t> SerializeFlatbuffer(
    const FlatbufferWrapper& flatbuffer) {
  auto tfl_model = flatbuffer.Unpack();
  return SerializeFlatbuffer(*tfl_model);
}

}  // namespace litert::internal
