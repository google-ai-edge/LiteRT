// Copyright 2025 Google LLC.
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

// Stub implementation for SerializeModel when mutable schema is not available

#include <cstddef>
#include <cstdint>
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"

namespace litert::internal {

// Stub implementation that just returns an error for now
Expected<OwningBufferRef<uint8_t>> SerializeModel(LiteRtModelT&& model,
                                                  size_t bytecode_alignment) {
  return Unexpected(kLiteRtStatusErrorNotFound,
               "SerializeModel is not implemented in CMake build yet. "
               "Mutable schema generation is required.");
}

}  // namespace litert::internal
