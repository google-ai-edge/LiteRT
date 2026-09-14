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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TENSOR_BUFFER_LAYOUT_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TENSOR_BUFFER_LAYOUT_H_

#include <algorithm>
#include <cstdint>
#include <limits>

#include "litert/c/litert_layout.h"
#include "litert/vendors/nvidia/cache_layout.h"

namespace litert::nvidia {

// A sequence-major value cache must carry explicit strides. An unannotated
// logical [1,H,D,S] buffer cannot safely be assumed to contain [1,H,S,D] data.
inline bool TensorBufferLayoutMatches(const LiteRtLayout& layout,
                                      bool transposed_value_cache) {
  if (layout.rank > LITERT_TENSOR_MAX_RANK) {
    return false;
  }
  if (transposed_value_cache) {
    uint32_t expected[4];
    return layout.has_strides &&
           GetTransposedValueCacheStrides(layout, expected) &&
           std::equal(expected, expected + 4, layout.strides);
  }
  if (!layout.has_strides) {
    return true;
  }
  uint64_t stride = 1;
  for (int i = static_cast<int>(layout.rank) - 1; i >= 0; --i) {
    if (layout.dimensions[i] <= 0 ||
        stride > std::numeric_limits<uint32_t>::max() ||
        layout.strides[i] != stride) {
      return false;
    }
    stride *= static_cast<uint32_t>(layout.dimensions[i]);
  }
  return true;
}

}  // namespace litert::nvidia

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TENSOR_BUFFER_LAYOUT_H_
