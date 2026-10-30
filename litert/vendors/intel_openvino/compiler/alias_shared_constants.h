// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ALIAS_SHARED_CONSTANTS_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ALIAS_SHARED_CONSTANTS_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"
#include "openvino/core/model.hpp"

namespace litert::openvino {

// One buffer's placement in the assembled cross-partition pool.
struct PoolEntry {
  int32_t buffer_id = 0;
  size_t pool_offset = 0;  // byte offset within the contiguous pool
  absl::Span<const uint8_t> bytes;
};

// The assembled pool, in ascending-BufferId order (the layout
// OpenVinoGlobalGraph::Serialize() lays down).
struct PoolLayout {
  std::vector<PoolEntry> buffers;

  // BufferId -> pool offset, for AliasAndTagSharedConstants. Derived from
  // |buffers| on demand, so an entry's offset has one source of truth.
  std::map<int32_t, size_t> OffsetMap() const;
};

// Lays out |weight_bank|'s buffers in ascending-BufferId order, calling
// WeightBank::FinalizeDerivedBuffers() first so derived ids exist. Call
// exactly once. Buffers whose name is absent from |live_names| are dropped;
// |live_names| is ignored unless |prune_dead|.
PoolLayout BuildPool(const std::unordered_set<std::string>& live_names,
                     WeightBank& weight_bank, bool prune_dead);

// In one pass over |ov_model|: registers each Constant tagged with
// LiteRtGeneratedConstantAttribute into |weight_bank|, and inserts every
// Constant's friendly_name into |live_names|. A tagged Constant's
// friendly_name MUST equal its source_key; violations are logged and skipped.
// The two share a traversal so |live_names| always matches what was harvested.
void HarvestSharedConstants(const std::shared_ptr<ov::Model>& ov_model,
                            WeightBank& weight_bank,
                            std::unordered_set<std::string>& live_names);

// NPU cross-partition weight-sharing transform (counterpart to the GPU
// ConvertWeightsToParameters).
//
// Rebuilds each large bank-backed weight Constant in |ov_model| so its data
// pointer ALIASES the deduplicated pool bytes for its BufferId (one stable host
// pointer shared across every partition), then stamps its
// WeightlessCacheAttribute(bin_offset).
//
// Aliasing is the load-bearing step for NPU weight sharing: NPUW dedups weights
// by their Constant data pointer, so pointing every partition's Constant at the
// one pool buffer is what makes NPUW collapse them to a single allocation.
// bin_offset tells NPUW where to mmap the weight at runtime.
//
// A Constant is aliased ONLY when its current bytes are byte-identical to the
// pool bytes for its BufferId. A mismatch means a content-altering frontend
// transform rewrote this weight, so aliasing would feed the graph the wrong
// data; such weights are left baked/per-partition. Comparing bytes is the
// robust guard -- no need to enumerate which transforms alter content.
//
// |pool_offset_of| maps BufferId -> byte offset in the contiguous shared pool
// (the same ascending-id layout Serialize() uses). |partition_idx| is used only
// for logging.
//
// Returns the number of Constants aliased (i.e. actually shareable). Constants
// with no BufferId or below the element threshold stay baked.
size_t AliasAndTagSharedConstants(
    const std::shared_ptr<ov::Model>& ov_model, const WeightBank& weight_bank,
    const std::map<int32_t, size_t>& pool_offset_of, int partition_idx);

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ALIAS_SHARED_CONSTANTS_H_
