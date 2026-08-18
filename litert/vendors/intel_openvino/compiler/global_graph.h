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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"

namespace litert::openvino {

// Container for cross-partition weight sharing: all partitions are aggregated
// into one blob holding a shared buffer pool (deduplicated weight bytes) plus a
// per-partition subgraph, and the SAME blob is returned for every partition.
// The dispatcher parses it, selects its subgraph, and resolves that subgraph's
// weights against the shared pool.
//
// Each subgraph's const_map records how its OV payload references the pool: it
// maps a weight-Parameter's friendly_name to the pool buffer_id it is bound to
// at dispatch (matched by name so binding is robust to input reordering across
// import_model).
//
// Serialized layout (single blob, little-endian):
//   magic  "OVGLOBAL"                       (8 bytes)
//   uint16 version                          (format version, see kVersion)
//   uint32 num_buffers
//     directory, repeat num_buffers times (ascending buffer_id):
//       uint32 buffer_id, uint64 pool_offset, uint64 size
//   uint64 pool_size
//   [pool_size bytes]                       (CONTIGUOUS deduplicated pool)
//   uint32 num_subgraphs
//     repeat: uint32 name_len, [name], uint8 device_enum,
//             uint32 const_map_len,
//               repeat: uint32 name_len, [name], uint32 buffer_id  (const_map)
//             uint64 payload_len, [payload bytes]             (OV exported
//             blob)
//
// v2 over v1: the pool is stored CONTIGUOUSLY (directory of {id, pool_offset,
// size} then all bytes back-to-back) rather than interleaved {id, size, bytes},
// so the NPU weightless path can stage it as a byte-for-byte copy of the
// [pool_data_offset, +pool_size) span and a Constant's WLCA bin_offset
// (== pool_offset) resolves to mmap->data() + bin_offset.
//
// ONE type at BOTH compile and dispatch; every byte range is a borrowed
// absl::Span<const uint8_t>:
//   - Compile: pool spans borrow WeightBank's views into the model's mmapped
//     weights; payload spans borrow bytes in |payload_store_| (owned here).
//   - Dispatch: Parse() spans alias the caller-owned blob (must outlive this
//     object -- see Parse); |payload_store_| is empty.
// The object is MOVE-ONLY: a shallow copy would leave payload spans dangling
// into a moved-from payload_store_.
class OpenVinoGlobalGraph {
 public:
  // Container format version, written right after the magic. Bump on any
  // layout change so Parse() can reject blobs it does not understand rather
  // than misparsing. v2 = contiguous pool + directory (see layout above).
  static constexpr uint16_t kVersion = 2;

  // One deduplicated weight buffer in the contiguous pool. |bytes| aliases the
  // pool source (the container blob at dispatch, or WeightBank at compile).
  struct BufferEntry {
    uint32_t id = 0;
    size_t pool_offset = 0;  // byte offset within the contiguous pool
    absl::Span<const uint8_t> bytes;
  };

  // One compiled partition entry. |payload| aliases the export bytes (in
  // payload_store_ at compile, or the container blob at dispatch).
  struct Subgraph {
    std::string name;    // e.g. "Partition_0"
    uint8_t device = 0;  // LiteRtIntelOpenVinoGraphBackend
    std::map<std::string, uint32_t> const_map;  // friendly_name -> buffer_id
    absl::Span<const uint8_t> payload;          // OV exported blob (aliased)
  };

  OpenVinoGlobalGraph() = default;
  // Move-only: copying would leave payload spans dangling into a moved-from
  // payload_store_.
  OpenVinoGlobalGraph(OpenVinoGlobalGraph&&) = default;
  OpenVinoGlobalGraph& operator=(OpenVinoGlobalGraph&&) = default;
  OpenVinoGlobalGraph(const OpenVinoGlobalGraph&) = delete;
  OpenVinoGlobalGraph& operator=(const OpenVinoGlobalGraph&) = delete;

  // Shared buffer pool.
  std::vector<BufferEntry> buffers;
  // Partition topologies, keyed by graph name (selected at dispatch by
  // function_name / graph order).
  std::map<std::string, Subgraph> subgraphs;

  // Set by Parse(): the whole contiguous pool and its byte offset from the
  // container start. Used by the NPU weightless path to stage the pool to a
  // temp file and to cross-check the pool span against the bytecode buffer.
  size_t pool_data_offset = 0;
  absl::Span<const uint8_t> pool;

  // Owns the export payloads at compile so Subgraph::payload spans stay valid.
  // A deque (not vector) so emplace_back never invalidates earlier payloads'
  // addresses. Empty on the Parse() path (payloads alias the blob there).
  std::deque<std::string> payload_store;

  // Serialize the whole container to one blob (see layout above). Requires
  // |buffers| to be in strictly-ascending buffer_id order with pool_offset
  // equal to the running byte sum (DCHECK'd) -- the invariant that makes a
  // Constant's WLCA bin_offset resolve at the staged temp file.
  std::string Serialize() const;

  // Zero-copy parse: returns a graph whose spans (pool, each buffer's bytes,
  // each subgraph's payload) alias |data|, which MUST outlive the returned
  // graph. Validates magic, version, the directory (every {pool_offset, size}
  // lies within pool_size), and all bounds. Serves both dispatch arms (GPU
  // bank fill by buffer_id, NPU pool staging). Returns an error on bad
  // magic / bounds.
  static litert::Expected<OpenVinoGlobalGraph> Parse(const uint8_t* data,
                                                     size_t size);

  // Fast check: does |data| begin with the OVGLOBAL magic?
  static bool HasMagic(const uint8_t* data, size_t size);

  // The BufferEntry with |id|, or nullptr if absent. Used by the GPU bank to
  // resolve a const_map buffer_id to its pool bytes.
  const BufferEntry* FindBuffer(uint32_t id) const;

  // Total bytes across the shared buffer pool (the deduplicated weight size).
  size_t BankBytes() const;
};

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_
