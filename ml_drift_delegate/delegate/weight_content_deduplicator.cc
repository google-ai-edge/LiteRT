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

#include "ml_drift_delegate/delegate/weight_content_deduplicator.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/numeric/int128.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "tflite/core/c/common.h"
#include "tflite/kernels/kernel_util.h"
#include "farmhash.h"

namespace litert::ml_drift {
namespace {

template <typename T>
void AppendPod(std::string& buf, const T& val) {
  buf.append(reinterpret_cast<const char*>(&val), sizeof(T));
}

void AppendIntArray(std::string& buf, const TfLiteIntArray* arr) {
  if (arr == nullptr) {
    AppendPod<int32_t>(buf, -1);
    return;
  }
  AppendPod<int32_t>(buf, static_cast<int32_t>(arr->size));
  if (arr->size > 0) {
    buf.append(reinterpret_cast<const char*>(arr->data),
               static_cast<size_t>(arr->size) * sizeof(int));
  }
}

void AppendFloatArray(std::string& buf, const TfLiteFloatArray* arr) {
  if (arr == nullptr) {
    AppendPod<int32_t>(buf, -1);
    return;
  }
  AppendPod<int32_t>(buf, static_cast<int32_t>(arr->size));
  if (arr->size > 0) {
    buf.append(reinterpret_cast<const char*>(arr->data),
               static_cast<size_t>(arr->size) * sizeof(float));
  }
}

// Appends a 128-bit fingerprint of a referenced auxiliary tensor (such as a
// blockwise scale or zero-point tensor) to `buf`. Returns false if the tensor
// index is valid but its host data is unavailable.
bool AppendReferencedTensorFingerprint(const TfLiteContext& context,
                                       int tensor_index, std::string& buf) {
  AppendPod<int32_t>(buf, tensor_index >= 0 ? 1 : 0);
  if (tensor_index < 0) {
    return true;
  }
  if (static_cast<size_t>(tensor_index) >= context.tensors_size) {
    return false;
  }
  const TfLiteTensor& ref = context.tensors[tensor_index];
  if (ref.data.raw_const == nullptr || ref.bytes == 0) {
    return false;
  }
  AppendPod<int32_t>(buf, static_cast<int32_t>(ref.type));
  AppendIntArray(buf, ref.dims);
  AppendPod<uint64_t>(buf, static_cast<uint64_t>(ref.bytes));
  const absl::uint128 fp =
      util::Fingerprint128(ref.data.raw_const, ref.bytes);
  AppendPod<uint64_t>(buf, absl::Uint128Low64(fp));
  AppendPod<uint64_t>(buf, absl::Uint128High64(fp));
  return true;
}

}  // namespace

std::optional<WeightContentDeduplicator::ContentKey>
WeightContentDeduplicator::ComputeContentKey(const TfLiteContext& context,
                                             const TfLiteTensor& tensor) {
  if (!tflite::IsConstantTensor(&tensor) || tensor.data.raw_const == nullptr ||
      tensor.bytes == 0 || tensor.dims == nullptr) {
    return std::nullopt;
  }

  std::string meta;
  meta.reserve(128);
  AppendPod<int32_t>(meta, static_cast<int32_t>(tensor.type));
  AppendIntArray(meta, tensor.dims);
  AppendPod<int32_t>(meta, static_cast<int32_t>(tensor.quantization.type));

  switch (tensor.quantization.type) {
    case kTfLiteNoQuantization:
      break;
    case kTfLiteAffineQuantization: {
      const auto* qparams = static_cast<const TfLiteAffineQuantization*>(
          tensor.quantization.params);
      if (qparams == nullptr) {
        return std::nullopt;
      }
      AppendPod<int32_t>(meta, qparams->quantized_dimension);
      AppendFloatArray(meta, qparams->scale);
      AppendIntArray(meta, qparams->zero_point);
      break;
    }
    case kTfLiteBlockwiseQuantization: {
      const auto* qparams = static_cast<const TfLiteBlockwiseQuantization*>(
          tensor.quantization.params);
      if (qparams == nullptr) {
        return std::nullopt;
      }
      AppendPod<int32_t>(meta, qparams->blocksize);
      AppendPod<int32_t>(meta, qparams->quantized_dimension);
      if (!AppendReferencedTensorFingerprint(context, qparams->scale, meta) ||
          !AppendReferencedTensorFingerprint(context, qparams->zero_point,
                                             meta)) {
        return std::nullopt;
      }
      break;
    }
    case kTfLiteBlockwiseQuantizationV2: {
      const auto* qparams = static_cast<const TfLiteBlockwiseQuantizationV2*>(
          tensor.quantization.params);
      if (qparams == nullptr) {
        return std::nullopt;
      }
      AppendPod<int32_t>(meta, qparams->blocksize);
      AppendPod<int32_t>(meta, qparams->quantized_dimension);
      AppendIntArray(meta, qparams->block_shape);
      if (!AppendReferencedTensorFingerprint(context, qparams->scale, meta) ||
          !AppendReferencedTensorFingerprint(context, qparams->zero_point,
                                             meta)) {
        return std::nullopt;
      }
      break;
    }
    default:
      // Unknown quantization scheme: do not merge to stay conservatively safe.
      return std::nullopt;
  }

  const absl::uint128 data_fp =
      util::Fingerprint128(tensor.data.raw_const, tensor.bytes);
  const absl::uint128 meta_fp =
      util::Fingerprint128(meta.data(), meta.size());

  ContentKey key;
  key.type = static_cast<int32_t>(tensor.type);
  key.bytes = static_cast<uint64_t>(tensor.bytes);
  key.data_lo = absl::Uint128Low64(data_fp);
  key.data_hi = absl::Uint128High64(data_fp);
  key.meta_lo = absl::Uint128Low64(meta_fp);
  key.meta_hi = absl::Uint128High64(meta_fp);
  return key;
}

WeightContentDeduplicator::BufferIdMap
WeightContentDeduplicator::CanonicalizeBufferIdMap(
    const TfLiteContext& context, const BufferIdMap& tensor_to_buffer_id_map) {
  BufferIdMap canonical_map;
  canonical_map.reserve(tensor_to_buffer_id_map.size());

  absl::MutexLock lock(mutex_);
  for (const auto& [tensor_id, orig_buffer_id] : tensor_to_buffer_id_map) {
    if (auto it = buffer_id_to_canonical_.find(orig_buffer_id);
        it != buffer_id_to_canonical_.end()) {
      canonical_map.emplace(tensor_id, it->second);
      continue;
    }

    size_t canonical_id = orig_buffer_id;
    if (tensor_id < context.tensors_size) {
      const std::optional<ContentKey> key =
          ComputeContentKey(context, context.tensors[tensor_id]);
      if (key.has_value()) {
        auto [content_it, inserted] =
            content_to_canonical_.try_emplace(*key, orig_buffer_id);
        canonical_id = content_it->second;
        if (!inserted && canonical_id != orig_buffer_id) {
          ++num_deduplicated_buffers_;
        }
        buffer_id_to_canonical_.emplace(orig_buffer_id, canonical_id);
      }
    }
    canonical_map.emplace(tensor_id, canonical_id);
  }
  return canonical_map;
}

size_t WeightContentDeduplicator::NumDeduplicatedBuffers() const {
  absl::MutexLock lock(mutex_);
  return num_deduplicated_buffers_;
}

}  // namespace litert::ml_drift
