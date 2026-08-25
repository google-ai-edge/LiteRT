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

#include "litert/vendors/intel_openvino/compiler/global_graph.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::openvino {
namespace {

constexpr char kMagic[8] = {'O', 'V', 'G', 'L', 'O', 'B', 'A', 'L'};

void PutU16(std::string& s, uint16_t v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof(v));
}
void PutU32(std::string& s, uint32_t v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof(v));
}
void PutU64(std::string& s, uint64_t v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

// Bounds-checked little-endian readers over a [data, data+size) cursor.
struct Reader {
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;
  bool Bytes(void* out, size_t n) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return false;
    }
    std::memcpy(out, p, n);
    p += n;
    return true;
  }
  uint16_t U16() {
    uint16_t v = 0;
    Bytes(&v, sizeof(v));
    return v;
  }
  uint32_t U32() {
    uint32_t v = 0;
    Bytes(&v, sizeof(v));
    return v;
  }
  uint64_t U64() {
    uint64_t v = 0;
    Bytes(&v, sizeof(v));
    return v;
  }
  bool Str(std::string& out, size_t n) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return false;
    }
    out.assign(reinterpret_cast<const char*>(p), n);
    p += n;
    return true;
  }
};

}  // namespace

bool OpenVinoGlobalGraph::HasMagic(const uint8_t* data, size_t size) {
  return data != nullptr && size >= sizeof(kMagic) &&
         std::memcmp(data, kMagic, sizeof(kMagic)) == 0;
}

size_t OpenVinoGlobalGraph::BankBytes() const {
  size_t total = 0;
  for (const auto& entry : buffers) total += entry.bytes.size();
  return total;
}

const OpenVinoGlobalGraph::BufferEntry* OpenVinoGlobalGraph::FindBuffer(
    uint32_t id) const {
  for (const auto& entry : buffers) {
    if (entry.id == id) return &entry;
  }
  return nullptr;
}

std::string OpenVinoGlobalGraph::Serialize() const {
  std::string out;
  out.append(kMagic, sizeof(kMagic));
  PutU16(out, kVersion);
  // Shared buffer pool, stored contiguously so the NPU weightless path can copy
  // it to a temp file byte-for-byte. First the directory, then the contiguous
  // pool bytes. |buffers| MUST be in strictly-ascending buffer_id order with
  // pool_offset == the running byte sum: that is the invariant that makes a
  // Constant's WeightlessCacheAttribute bin_offset (== pool_offset) resolve to
  // mmap->data() + bin_offset in the staged temp file.
  PutU32(out, static_cast<uint32_t>(buffers.size()));
  uint64_t pool_size = 0;
  bool have_prev = false;
  uint32_t prev_id = 0;
  for (const auto& entry : buffers) {
    // Enforce the ordering/offset invariant rather than silently misserialize.
    ABSL_DCHECK(!have_prev || entry.id > prev_id)
        << "OpenVinoGlobalGraph: buffers must be strictly ascending by id";
    ABSL_DCHECK_EQ(entry.pool_offset, pool_size)
        << "OpenVinoGlobalGraph: pool_offset must equal the running byte sum";
    PutU32(out, entry.id);
    PutU64(out, pool_size);  // pool_offset (== WLCA bin_offset at dispatch)
    PutU64(out, entry.bytes.size());
    pool_size += entry.bytes.size();
    have_prev = true;
    prev_id = entry.id;
  }
  PutU64(out, pool_size);
  for (const auto& entry : buffers) {
    out.append(reinterpret_cast<const char*>(entry.bytes.data()),
               entry.bytes.size());  // contiguous, ascending buffer_id
  }
  // subgraphs
  PutU32(out, static_cast<uint32_t>(subgraphs.size()));
  for (const auto& [name, subgraph] : subgraphs) {
    PutU32(out, static_cast<uint32_t>(subgraph.name.size()));
    out.append(subgraph.name);
    out.push_back(static_cast<char>(subgraph.device));
    PutU32(out, static_cast<uint32_t>(subgraph.const_map.size()));
    for (const auto& [const_name, buffer_id] : subgraph.const_map) {
      PutU32(out, static_cast<uint32_t>(const_name.size()));
      out.append(const_name);
      PutU32(out, buffer_id);
    }
    PutU64(out, subgraph.payload.size());
    out.append(reinterpret_cast<const char*>(subgraph.payload.data()),
               subgraph.payload.size());
  }
  return out;
}

litert::Expected<OpenVinoGlobalGraph> OpenVinoGlobalGraph::Parse(
    const uint8_t* data, size_t size) {
  if (!HasMagic(data, size)) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: bad magic");
  }
  Reader reader{data + sizeof(kMagic), data + size};
  OpenVinoGlobalGraph graph;

  const uint16_t version = reader.U16();
  if (!reader.ok || version != kVersion) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: unsupported container version");
  }

  // Read the pool directory {id, pool_offset, size}, then locate the contiguous
  // pool that follows. Zero-copy: every buffer's bytes and each subgraph's
  // payload alias |data| (which must outlive the returned graph). Serves both
  // dispatch arms -- the GPU bank resolves buffers by id, the NPU path stages
  // the whole pool span to a temp file.
  const uint32_t num_buffers = reader.U32();
  std::vector<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>>
      dir;  // id->{off,sz}
  dir.reserve(num_buffers);
  for (uint32_t i = 0; i < num_buffers && reader.ok; ++i) {
    const uint32_t id = reader.U32();
    const uint64_t off = reader.U64();
    const uint64_t sz = reader.U64();
    dir.push_back({id, {off, sz}});
  }
  const uint64_t pool_size = reader.U64();
  if (!reader.ok) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: truncated directory");
  }
  // The contiguous pool begins right after pool_size. Its offset from the
  // container start is what the NPU temp-file writer copies from.
  const size_t pool_data_offset = static_cast<size_t>(reader.p - data);
  const uint8_t* pool_base = reader.p;
  if (static_cast<size_t>(reader.end - reader.p) < pool_size) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: truncated pool");
  }
  graph.pool_data_offset = pool_data_offset;
  graph.pool = absl::MakeConstSpan(pool_base, static_cast<size_t>(pool_size));
  reader.p += pool_size;
  // Every {offset, size} must lie within the pool; this is what guarantees a
  // WLCA bin_offset resolves inside the staged temp file (bin_offset + size
  // <= pool_size) and that a GPU buffer view stays in-bounds.
  graph.buffers.reserve(dir.size());
  for (const auto& [id, off_sz] : dir) {
    const uint64_t off = off_sz.first;
    const uint64_t sz = off_sz.second;
    if (off > pool_size || sz > pool_size - off) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "OpenVinoGlobalGraph: buffer out of pool bounds");
    }
    graph.buffers.push_back(
        {id, static_cast<size_t>(off),
         absl::MakeConstSpan(pool_base + off, static_cast<size_t>(sz))});
  }

  const uint32_t num_subgraphs = reader.U32();
  for (uint32_t i = 0; i < num_subgraphs && reader.ok; ++i) {
    Subgraph subgraph;
    const uint32_t name_len = reader.U32();
    reader.Str(subgraph.name, name_len);
    uint8_t dev = 0;
    reader.Bytes(&dev, 1);
    subgraph.device = dev;
    const uint32_t cm_len = reader.U32();
    for (uint32_t j = 0; j < cm_len && reader.ok; ++j) {
      const uint32_t const_name_len = reader.U32();
      std::string const_name;
      reader.Str(const_name, const_name_len);
      const uint32_t bid = reader.U32();
      subgraph.const_map.emplace(const_name, bid);
    }
    const uint64_t payload_len = reader.U64();
    // Alias the payload bytes in place (zero-copy) instead of copying.
    const uint8_t* payload_ptr = reader.p;
    if (!reader.ok ||
        static_cast<size_t>(reader.end - reader.p) < payload_len) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "OpenVinoGlobalGraph: truncated subgraph payload");
    }
    subgraph.payload =
        absl::MakeConstSpan(payload_ptr, static_cast<size_t>(payload_len));
    reader.p += payload_len;
    if (reader.ok) {
      const std::string name = subgraph.name;
      graph.subgraphs.emplace(name, std::move(subgraph));
    }
  }

  if (!reader.ok) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: truncated/corrupt container");
  }
  return graph;
}

}  // namespace litert::openvino
