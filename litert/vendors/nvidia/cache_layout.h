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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_CACHE_LAYOUT_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_CACHE_LAYOUT_H_

#include <stdint.h>

#include "litert/c/litert_layout.h"
#include "litert/c/litert_tensor_buffer_types.h"

#define LITERT_NVIDIA_READ_ONLY_VALUE_CACHE_OPTIONS_ID \
  "nvidia.read_only_value_cache.v1"

// Opaque compilation-option payload. Each name is an underlying TFLite input
// tensor name, not a signature alias. The caller has established that these
// logical [1,H,D,S] tensors share physical [1,H,S,D] buffers with another
// compiled model. The compiler copies the names when the plugin is created.
typedef struct {
  uint32_t num_inputs;
  const char* const* input_names;
} LiteRtNvidiaReadOnlyValueCacheOptions;

#ifdef __cplusplus
#include <limits>
#include <string_view>

namespace litert::nvidia {

inline constexpr LiteRtTensorBufferType kNvidiaCudaTensorBufferType =
    static_cast<LiteRtTensorBufferType>(
        kLiteRtTensorBufferTypeUserCustomBuffer + 1);

// Engine IO names are already persisted in every bytecode format. This suffix
// makes the private physical layout explicit to dispatch, including AOT hits.
inline constexpr char kTransposedValueCacheSuffix[] = "__litert_value_cache_sd";

inline bool IsTransposedValueCacheTensor(std::string_view name) {
  constexpr std::string_view suffix = kTransposedValueCacheSuffix;
  return name.size() >= suffix.size() &&
         name.substr(name.size() - suffix.size()) == suffix;
}

// Strides are in elements and refer to the model's logical [1,H,D,S] shape.
inline bool GetTransposedValueCacheStrides(const LiteRtLayout& logical_layout,
                                           uint32_t strides[4]) {
  if (strides == nullptr || logical_layout.rank != 4 ||
      logical_layout.dimensions[0] != 1) {
    return false;
  }
  uint64_t product = 1;
  for (int i = 1; i < 4; ++i) {
    const int32_t dimension = logical_layout.dimensions[i];
    if (dimension <= 0 || product > std::numeric_limits<uint32_t>::max() /
                                        static_cast<uint32_t>(dimension)) {
      return false;
    }
    product *= static_cast<uint32_t>(dimension);
  }
  strides[0] = static_cast<uint32_t>(product);
  strides[1] = strides[0] / logical_layout.dimensions[1];
  strides[2] = 1;
  strides[3] = static_cast<uint32_t>(logical_layout.dimensions[2]);
  return true;
}

// Use as OpaqueOptions::SetHash() so LiteRT's outer compilation cache also
// distinguishes the contract. Sort names for order-independent cache reuse;
// an unsorted list is still correct but can cause an extra cache miss.
inline uint64_t HashReadOnlyValueCacheOptions(const void* payload) {
  const auto* options =
      static_cast<const LiteRtNvidiaReadOnlyValueCacheOptions*>(payload);
  if (options == nullptr ||
      (options->num_inputs != 0 && options->input_names == nullptr)) {
    return 0;
  }
  uint64_t hash = 14695981039346656037ULL;
  const auto append = [&hash](uint8_t byte) {
    hash = (hash ^ byte) * 1099511628211ULL;
  };
  for (uint32_t i = 0; i < options->num_inputs; ++i) {
    const char* name = options->input_names[i];
    if (name == nullptr || name[0] == '\0') {
      return 0;
    }
    for (; *name != '\0'; ++name) {
      append(static_cast<uint8_t>(*name));
    }
    append(0);  // Names cannot contain embedded NUL, so boundaries are unique.
  }
  return hash;
}

}  // namespace litert::nvidia
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_CACHE_LAYOUT_H_
