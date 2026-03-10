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

#ifndef ODML_LITERT_LITERT_CORE_DISPATCH_BYTECODE_MANIFEST_H_
#define ODML_LITERT_LITERT_CORE_DISPATCH_BYTECODE_MANIFEST_H_

#include <cstddef>
#include <string>
#include <vector>

#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"

namespace litert::internal {

// Model metadata key for dispatch bytecode location metadata.
extern const char kLiteRtDispatchBytecodeManifestKey[];

struct DispatchBytecodeManifestEntry {
  size_t subgraph_index = 0;
  size_t op_index = 0;
  std::string function_name;
  size_t bytecode_offset = 0;
  size_t bytecode_size = 0;
};

OwningBufferRef<uint8_t> MakeDispatchBytecodeManifest(
    const std::vector<DispatchBytecodeManifestEntry>& entries);

bool UpdateDispatchBytecodeManifestEntryInPlace(
    size_t manifest_entry_index, size_t bytecode_offset, size_t bytecode_size,
    MutableBufferRef<uint8_t> manifest_buffer);

Expected<std::vector<DispatchBytecodeManifestEntry>>
ParseDispatchBytecodeManifest(BufferRef<uint8_t> manifest_buffer);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_DISPATCH_BYTECODE_MANIFEST_H_
