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

#include <cstdint>
#include <cstring>
#include <string>

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
  for (const auto& [id, bytes] : buffers) total += bytes.size();
  return total;
}

std::string OpenVinoGlobalGraph::Serialize() const {
  std::string out;
  out.append(kMagic, sizeof(kMagic));
  PutU16(out, kVersion);
  // shared buffer pool
  PutU32(out, static_cast<uint32_t>(buffers.size()));
  for (const auto& [id, bytes] : buffers) {
    PutU32(out, id);
    PutU64(out, bytes.size());
    out.append(bytes);
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
    out.append(subgraph.payload);
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
    return litert::Error(
        kLiteRtStatusErrorRuntimeFailure,
        "OpenVinoGlobalGraph: unsupported container version");
  }

  const uint32_t num_buffers = reader.U32();
  for (uint32_t i = 0; i < num_buffers && reader.ok; ++i) {
    const uint32_t id = reader.U32();
    const uint64_t sz = reader.U64();
    std::string bytes;
    reader.Str(bytes, static_cast<size_t>(sz));
    if (reader.ok) graph.buffers.emplace(id, std::move(bytes));
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
    reader.Str(subgraph.payload, static_cast<size_t>(payload_len));
    if (reader.ok) graph.subgraphs.emplace(subgraph.name, std::move(subgraph));
  }

  if (!reader.ok) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: truncated/corrupt container");
  }
  return graph;
}

}  // namespace litert::openvino
