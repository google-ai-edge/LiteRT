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

#include "litert/core/dispatch_bytecode_manifest.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/litert_common.h"

namespace litert::internal {
namespace {

constexpr char kManifestEntriesKey[] = "entries";
constexpr char kSubgraphIndexKey[] = "subgraph_index";
constexpr char kOpIndexKey[] = "op_index";
constexpr char kFunctionNameKey[] = "function_name";
constexpr char kBytecodeOffsetKey[] = "bytecode_offset";
constexpr char kBytecodeSizeKey[] = "bytecode_size";

}  // namespace

const char kLiteRtDispatchBytecodeManifestKey[] =
    "LiteRtDispatchBytecodeManifestV1";

OwningBufferRef<uint8_t> MakeDispatchBytecodeManifest(
    const std::vector<DispatchBytecodeManifestEntry>& entries) {
  flexbuffers::Builder fbb;

  // Use fixed-width integer scalars so offsets/sizes can be patched in place.
  fbb.ForceMinimumBitWidth(flexbuffers::BIT_WIDTH_64);

  const auto root_start = fbb.StartMap();
  const auto entries_start = fbb.StartVector(kManifestEntriesKey);
  for (const auto& entry : entries) {
    const auto entry_start = fbb.StartMap();
    fbb.UInt(kSubgraphIndexKey, entry.subgraph_index);
    fbb.UInt(kOpIndexKey, entry.op_index);
    fbb.String(kFunctionNameKey, entry.function_name);
    fbb.UInt(kBytecodeOffsetKey, entry.bytecode_offset);
    fbb.UInt(kBytecodeSizeKey, entry.bytecode_size);
    fbb.EndMap(entry_start);
  }
  fbb.EndVector(entries_start, /*typed=*/false, /*fixed=*/false);
  fbb.EndMap(root_start);
  fbb.Finish();

  const auto vec = fbb.GetBuffer();
  OwningBufferRef<uint8_t> out;
  out.Assign(vec.data(), vec.size());
  return out;
}

bool UpdateDispatchBytecodeManifestEntryInPlace(
    size_t manifest_entry_index, size_t bytecode_offset, size_t bytecode_size,
    MutableBufferRef<uint8_t> manifest_buffer) {
  const auto manifest_root =
      flexbuffers::GetRoot(manifest_buffer.Data(), manifest_buffer.Size())
          .AsMap();
  const auto entries = manifest_root[kManifestEntriesKey].AsVector();
  if (manifest_entry_index >= entries.size()) {
    return false;
  }
  const auto entry = entries[manifest_entry_index].AsMap();
  const bool offset_ok = entry[kBytecodeOffsetKey].MutateUInt(bytecode_offset);
  const bool size_ok = entry[kBytecodeSizeKey].MutateUInt(bytecode_size);
  return offset_ok && size_ok;
}

Expected<std::vector<DispatchBytecodeManifestEntry>>
ParseDispatchBytecodeManifest(BufferRef<uint8_t> manifest_buffer) {
  const auto manifest_root =
      flexbuffers::GetRoot(manifest_buffer.Data(), manifest_buffer.Size())
          .AsMap();
  const auto entries = manifest_root[kManifestEntriesKey].AsVector();
  if (entries.size() == 0) {
    return Error(kLiteRtStatusErrorNotFound,
                 "Dispatch bytecode manifest entries not found");
  }

  std::vector<DispatchBytecodeManifestEntry> out;
  out.reserve(entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto entry_map = entries[i].AsMap();
    DispatchBytecodeManifestEntry entry;
    entry.subgraph_index = entry_map[kSubgraphIndexKey].AsUInt64();
    entry.op_index = entry_map[kOpIndexKey].AsUInt64();
    entry.function_name = entry_map[kFunctionNameKey].AsString().str();
    entry.bytecode_offset = entry_map[kBytecodeOffsetKey].AsUInt64();
    entry.bytecode_size = entry_map[kBytecodeSizeKey].AsUInt64();
    out.push_back(std::move(entry));
  }
  return out;
}

}  // namespace litert::internal
