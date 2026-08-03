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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

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

std::optional<int32_t> WeightBank::BufferIdOfName(
    std::string_view tensor_name) const {
  auto name_it = name_to_buffer_id_.find(std::string(tensor_name));
  if (name_it == name_to_buffer_id_.end()) {
    return std::nullopt;
  }
  return name_it->second;
}

}  // namespace litert::openvino
