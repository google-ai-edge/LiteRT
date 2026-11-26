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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_TENSOR_HASH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_TENSOR_HASH_H_

// Canonical FNV-1a mixing and deterministic constant-tensor fingerprinting
// shared across backend lowering (`backends/xnnpack/conversion.cc`) and model
// weight cache validation (`models/common/`).

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/internal/graph.h"

namespace litert::tensor::graph::hash_internal {

inline constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
inline constexpr uint64_t kFnvPrime = 1099511628211ULL;

inline void FnvMixBytes(uint64_t* hash_state, const void* data, size_t len) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; ++i) {
    *hash_state ^= static_cast<uint64_t>(bytes[i]);
    *hash_state *= kFnvPrime;
  }
}

template <typename T>
inline void FnvMixScalar(uint64_t* hash_state, const T& val) {
  FnvMixBytes(hash_state, &val, sizeof(T));
}

inline void FnvMixString(uint64_t* hash_state, absl::string_view str) {
  const uint64_t len = static_cast<uint64_t>(str.size());
  FnvMixScalar(hash_state, len);
  if (!str.empty()) {
    FnvMixBytes(hash_state, str.data(), str.size());
  }
}

inline void FnvMixSampledBytes(uint64_t* hash_state, const std::byte* data,
                               size_t size) {
  FnvMixScalar(hash_state, static_cast<uint64_t>(size));
  if (data == nullptr || size == 0) return;
  if (size <= 4096) {
    FnvMixBytes(hash_state, data, size);
    return;
  }
  FnvMixBytes(hash_state, data, 128);
  FnvMixBytes(hash_state, data + size / 2 - 64, 128);
  FnvMixBytes(hash_state, data + size - 128, 128);
  const size_t step = size / 32;
  for (size_t i = 0; i < 32; ++i) {
    const size_t offset = std::min(i * step, size - sizeof(uint64_t));
    FnvMixBytes(hash_state, data + offset, sizeof(uint64_t));
  }
}

inline void MixTensorQuantization(const Quantization* quantization,
                                  uint64_t* hash_state) {
  if (quantization == nullptr) return;
  if (auto pc = quantization->As<PerChannelAffineQuantization>(); pc.ok()) {
    FnvMixScalar(hash_state, int32_t{1});
    FnvMixScalar(hash_state, pc->quantized_dimension);
    FnvMixSampledBytes(hash_state,
                       reinterpret_cast<const std::byte*>(pc->scales.data()),
                       pc->scales.size() * sizeof(float));
    if (auto zps = pc->zero_points.Data<int64_t>(); zps.ok() && !zps->empty()) {
      FnvMixSampledBytes(hash_state,
                         reinterpret_cast<const std::byte*>(zps->data()),
                         zps->size() * sizeof(int64_t));
    } else if (auto fzps = pc->zero_points.Data<float>();
               fzps.ok() && !fzps->empty()) {
      FnvMixSampledBytes(hash_state,
                         reinterpret_cast<const std::byte*>(fzps->data()),
                         fzps->size() * sizeof(float));
    }
  } else if (auto bw = quantization->As<BlockwiseQuantization>(); bw.ok()) {
    FnvMixScalar(hash_state, int32_t{2});
    FnvMixScalar(hash_state, bw->block_size);
    FnvMixScalar(hash_state, bw->quantized_dimension);
    FnvMixSampledBytes(hash_state,
                       reinterpret_cast<const std::byte*>(bw->scales.data()),
                       bw->scales.size() * sizeof(float));
    if (auto zps = bw->zero_points.Data<int64_t>(); zps.ok() && !zps->empty()) {
      FnvMixSampledBytes(hash_state,
                         reinterpret_cast<const std::byte*>(zps->data()),
                         zps->size() * sizeof(int64_t));
    } else if (auto fzps = bw->zero_points.Data<float>();
               fzps.ok() && !fzps->empty()) {
      FnvMixSampledBytes(hash_state,
                         reinterpret_cast<const std::byte*>(fzps->data()),
                         fzps->size() * sizeof(float));
    }
  }
}

inline void HashConstantTensorInfo(const TensorInformation& info,
                                   const std::byte* raw_data, size_t raw_size,
                                   bool include_name, uint64_t* hash_state) {
  if (include_name) {
    FnvMixString(hash_state, info.name);
  }
  FnvMixScalar(hash_state, static_cast<int32_t>(info.type));
  FnvMixScalar(hash_state, static_cast<uint64_t>(info.shape.size()));
  for (int32_t dim : info.shape) {
    FnvMixScalar(hash_state, dim);
  }
  if (raw_data == nullptr && info.buffer != nullptr) {
    LockedBufferSpan<const std::byte> locked = info.buffer->Lock();
    FnvMixSampledBytes(hash_state, locked.data(), locked.size());
  } else {
    FnvMixSampledBytes(hash_state, raw_data, raw_size);
  }
  MixTensorQuantization(info.quantization.get(), hash_state);
}

inline uint64_t ComputeConstantTensorHash(const TensorInformation& info,
                                          const std::byte* raw_data,
                                          size_t raw_size) {
  uint64_t hash_state = kFnvOffsetBasis;
  HashConstantTensorInfo(info, raw_data, raw_size, /*include_name=*/true,
                         &hash_state);
  return hash_state == 0 ? 1ULL : hash_state;
}

}  // namespace litert::tensor::graph::hash_internal

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_TENSOR_HASH_H_
