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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TENSOR_BUFFER_VIEW_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TENSOR_BUFFER_VIEW_H_

#include <cstddef>
#include <cstdint>
#include <limits>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::nvidia {

inline Expected<void*> ResolveCudaTensorBufferView(void* allocation,
                                                   size_t allocation_size,
                                                   size_t offset,
                                                   size_t packed_size) {
  if (allocation == nullptr) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Null CUDA tensor buffer allocation");
  }
  if (offset > allocation_size || packed_size > allocation_size - offset) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "CUDA tensor buffer view exceeds its allocation");
  }

  const uintptr_t allocation_address = reinterpret_cast<uintptr_t>(allocation);
  if (offset > std::numeric_limits<uintptr_t>::max() - allocation_address) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "CUDA tensor buffer view address overflows");
  }
  return reinterpret_cast<void*>(allocation_address + offset);
}

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_TENSOR_BUFFER_VIEW_H_
