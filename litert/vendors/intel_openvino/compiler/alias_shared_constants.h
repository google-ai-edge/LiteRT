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

#include "openvino/core/model.hpp"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"

namespace litert::openvino {

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
