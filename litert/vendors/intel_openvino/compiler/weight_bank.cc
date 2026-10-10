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

#include "litert/vendors/intel_openvino/compiler/weight_bank.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/compiler/cc/litert_model.h"

namespace litert::openvino {

void WeightBank::AddSubgraph(const litert::compiler::Subgraph& subgraph) {
  for (const auto& op : subgraph.Ops()) {
    for (const auto& input : op.Inputs()) {
      if (!input.HasWeights()) {
        continue;
      }
      const auto weights = input.Weights();
      // Keyed by BufferId, so a buffer shared by multiple ops/partitions is
      // recorded once. The bytes are identical for a given id, so re-assignment
      // is harmless.
      const int32_t buffer_id = weights.BufferId();
      // Defensive: BufferId() returns -1 only when the id lookup fails (missing
      // callback / bad handle). A real weight always has a valid id, so skip
      // rather than pollute the pool with a sentinel key.
      if (buffer_id < 0) {
        continue;
      }
      buffer_bytes_[buffer_id] = weights.Bytes();
      // Record this tensor's name so the matching OpenVINO weight (which takes
      // the tensor name as its friendly_name) can be resolved back to its
      // buffer. Distinct names sharing a buffer all point at the same id.
      name_to_buffer_id_[std::string(input.Name())] = buffer_id;
    }
  }
}

size_t WeightBank::TotalBytes() const {
  size_t total = 0;
  for (const auto& [buffer_id, bytes] : buffer_bytes_) {
    total += bytes.size();
  }
  return total;
}

void WeightBank::RegisterGeneratedConstant(std::string_view key,
                                           std::vector<uint8_t> bytes) {
  const std::string key_str(key);
  if (derived_finalized_) {
    // Ids are assigned and the pool laid out, so this key could never resolve.
    // Drop it to keep the bank consistent with the pool already built.
    LITERT_LOG(LITERT_ERROR,
               "WeightBank: generated constant key '%s' registered after "
               "FinalizeDerivedBuffers(); dropping it. Harvest every "
               "partition before finalizing",
               key_str.c_str());
    return;
  }
  auto it = derived_bytes_.find(key_str);
  if (it == derived_bytes_.end()) {
    derived_bytes_.emplace(key_str, std::move(bytes));
    return;
  }

  // The first registration wins either way, but a disagreement means two
  // partitions synthesized different data under one identity -- the pool then
  // can't match one of them.
  if (it->second.size() != bytes.size()) {
    LITERT_LOG(LITERT_ERROR,
               "WeightBank: generated constant key '%s' re-registered with a "
               "different size (%zu vs %zu bytes); keeping the first "
               "registration",
               key_str.c_str(), it->second.size(), bytes.size());
  } else if (!bytes.empty() &&
             std::memcmp(it->second.data(), bytes.data(), bytes.size()) != 0) {
    LITERT_LOG(LITERT_ERROR,
               "WeightBank: generated constant key '%s' re-registered with "
               "different contents (same size, %zu bytes); keeping the first "
               "registration",
               key_str.c_str(), bytes.size());
  }
}

void WeightBank::FinalizeDerivedBuffers() {
  if (derived_finalized_) {
    LITERT_LOG(LITERT_ERROR,
               "WeightBank::FinalizeDerivedBuffers called more than once; "
               "ignoring the extra call");
    return;
  }
  derived_finalized_ = true;

  // Safe only because every real BufferId (from AddSubgraph) is already
  // known: this must run after every partition's AddSubgraph has completed.
  int32_t next_id = 0;
  for (const auto& [id, bytes] : buffer_bytes_) {
    next_id = std::max(next_id, id + 1);
  }
  // Ascending key order (std::map), so the id -> pool offset -> WLCA
  // bin_offset chain is identical on every run. derived_bytes_ keeps owning
  // the bytes; being node-based, the spans below stay valid.
  const size_t num_derived = derived_bytes_.size();
  for (const auto& [key, bytes] : derived_bytes_) {
    const int32_t buffer_id = next_id++;
    buffer_bytes_[buffer_id] = absl::MakeConstSpan(bytes);
    name_to_buffer_id_[key] = buffer_id;
  }
  LITERT_LOG(LITERT_INFO,
             "WeightBank::FinalizeDerivedBuffers: assigned %zu derived "
             "buffer id(s)",
             num_derived);
}

std::optional<int32_t> WeightBank::BufferIdOfName(
    std::string_view tensor_name) const {
  auto name_it = name_to_buffer_id_.find(std::string(tensor_name));
  if (name_it == name_to_buffer_id_.end()) {
    return std::nullopt;
  }
  return name_it->second;
}

}  // namespace litert::openvino
