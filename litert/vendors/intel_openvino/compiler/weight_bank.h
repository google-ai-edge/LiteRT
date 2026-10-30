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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHT_BANK_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHT_BANK_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/compiler/cc/litert_model.h"

namespace litert::openvino {

// Backs cross-partition weight sharing: collects the distinct constant-weight
// buffers used across a model's partitions into one deduplicated pool, so the
// shared weights are stored once (referenced by every partition via the
// GlobalGraph const_map / shared buffer pool) instead of duplicated per
// partition.
//
// Buffers are keyed by LiteRt Weights::BufferId(): LiteRt's buffer manager
// assigns the same BufferId to tensors that share storage, so a buffer used by
// both the prefill and decode partitions is recorded exactly once. The bank
// also maps each weight tensor's name to its BufferId so the converted
// OpenVINO weights (whose friendly_name is the LiteRt tensor name) can be
// resolved back to the shared buffer they belong to.
class WeightBank {
 public:
  WeightBank() = default;
  // Pool spans may point into |converted_buffers_|, so a copy would dangle.
  WeightBank(const WeightBank&) = delete;
  WeightBank& operator=(const WeightBank&) = delete;
  WeightBank(WeightBank&&) = default;
  WeightBank& operator=(WeightBank&&) = default;

  // Records every constant-weight buffer referenced by |subgraph|. Repeated
  // BufferIds (within or across subgraphs) collapse to a single entry. Safe to
  // call once per partition to accumulate the union of all weight buffers.
  //
  // |convert_i2_to_u2| (NPU shared path): store quantized int2 buffers in their
  // converted u2 form (see i2_u2_convert.h) instead of the raw i2 bytes. The
  // frontend rewrites those weights to u2 before OpenVINO sees them, so a pool
  // holding raw i2 bytes could never match the compiled u2 Constants -- and,
  // being tagged weightless, NPUW would reload the wrong bytes from it at
  // runtime. Storing u2 keeps int2 weights shareable and correct. The converted
  // copies are owned by the bank; all other buffers stay zero-copy views of the
  // model's mmapped weights. Must mirror the condition under which
  // GraphIteratorDelegate performs the same conversion.
  void AddSubgraph(const litert::compiler::Subgraph& subgraph,
                   bool convert_i2_to_u2 = false);

  // Number of int2 buffers stored converted (u2) in the pool.
  size_t NumConvertedBuffers() const { return converted_ids_.size(); }

  // Number of distinct weight buffers recorded so far.
  size_t NumBuffers() const { return buffer_bytes_.size(); }

  // Total bytes across all distinct buffers (i.e. the deduplicated weight
  // size).
  size_t TotalBytes() const;

  // BufferId of the weight tensor named |tensor_name|, or nullopt if unknown.
  // Used to build the GlobalGraph const_map (OV weight -> shared buffer id).
  // Real weights resolve as soon as AddSubgraph() has run; derived/generated
  // ones (registered under their source_key, which HarvestSharedConstants
  // requires to equal the constant's friendly_name) only resolve after
  // FinalizeDerivedBuffers().
  std::optional<int32_t> BufferIdOfName(std::string_view tensor_name) const;

  // The deduplicated buffer pool as (BufferId -> bytes view), for populating
  // the GlobalGraph shared buffer pool. Available after AddSubgraph() calls.
  const std::unordered_map<int32_t, absl::Span<const uint8_t>>& Buffers()
      const {
    return buffer_bytes_;
  }

  // Registers a generated/derived constant's bytes under |key|; the first
  // registration wins. Its BufferId exists only after
  // FinalizeDerivedBuffers(); registering after that is an error and drops.
  void RegisterGeneratedConstant(std::string_view key,
                                 std::vector<uint8_t> bytes);

  // Assigns BufferIds (past the highest real id, in ascending key order so the
  // pool layout is reproducible) to every RegisterGeneratedConstant key. Call
  // exactly once, after every partition's AddSubgraph/HarvestSharedConstants.
  void FinalizeDerivedBuffers();

 private:
  // BufferId -> the buffer's bytes: a view into the model's mmapped weights, or
  // into |converted_buffers_| for int2 buffers stored as u2.
  std::unordered_map<int32_t, absl::Span<const uint8_t>> buffer_bytes_;
  // Owned u2 copies of int2 buffers. A deque keeps element addresses stable as
  // it grows, so the spans in |buffer_bytes_| stay valid.
  std::deque<std::vector<uint8_t>> converted_buffers_;
  std::unordered_set<int32_t> converted_ids_;
  // Weight tensor name -> its BufferId. Many names may map to one BufferId
  // (tensors that share storage), which is how shared weights resolve to a
  // single pool buffer. Also holds derived/generated keys once
  // FinalizeDerivedBuffers() assigns their BufferIds (key == friendly_name,
  // enforced by HarvestSharedConstants).
  std::unordered_map<std::string, int32_t> name_to_buffer_id_;
  // Generated-constant key -> its bytes, which it keeps owning after
  // FinalizeDerivedBuffers() spans into them. std::map, not unordered_map:
  // ordered iteration (deterministic ids) and node-based (stable addresses).
  std::map<std::string, std::vector<uint8_t>> derived_bytes_;
  bool derived_finalized_ = false;
};

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHT_BANK_H_
