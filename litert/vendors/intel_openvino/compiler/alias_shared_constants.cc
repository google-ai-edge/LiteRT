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

#include "litert/vendors/intel_openvino/compiler/alias_shared_constants.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"
#include "litert/vendors/intel_openvino/compiler/weightless_caching_attributes.hpp"

namespace litert::openvino {

size_t AliasAndTagSharedConstants(
    const std::shared_ptr<ov::Model>& ov_model, const WeightBank& weight_bank,
    const std::map<int32_t, size_t>& pool_offset_of, int partition_idx) {
  // Collect before mutating: replacing nodes while iterating get_ordered_ops()
  // is unsafe.
  std::vector<std::shared_ptr<ov::op::v0::Constant>> candidates;
  for (const auto& node : ov_model->get_ordered_ops()) {
    auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!cnst) continue;
    const size_t elem_size = cnst->get_element_type().size();
    // Skip tiny/shape/scalar constants (fewer than 16 elements) and any type
    // with unknown element size.
    if (elem_size == 0 || cnst->get_byte_size() / elem_size < 16) continue;
    candidates.push_back(cnst);
  }

  const auto& buffers = weight_bank.Buffers();
  size_t aliased = 0;
  size_t tagged = 0;
  size_t mismatched = 0;
  for (const auto& cnst : candidates) {
    const auto bid = weight_bank.BufferIdOfName(cnst->get_friendly_name());
    if (!bid) continue;  // OV-synthesized const: not backed by a shared buffer
    const auto off_it = pool_offset_of.find(*bid);
    if (off_it == pool_offset_of.end()) continue;  // not in the shared pool
    const auto buf_it = buffers.find(*bid);
    if (buf_it == buffers.end()) continue;  // defensive: id present in map only
    const absl::Span<const uint8_t> pool_bytes = buf_it->second;

    const void* frontend_ptr = cnst->get_data_ptr();
    const size_t cnst_bytes = cnst->get_byte_size();

    // Alias only if the Constant's bytes match the pool bytes (see header
    // comment). Compare size first, then contents.
    const bool bytes_match =
        cnst_bytes == pool_bytes.size() &&
        std::memcmp(frontend_ptr, pool_bytes.data(), cnst_bytes) == 0;

    std::shared_ptr<ov::op::v0::Constant> tag_target = cnst;
    if (bytes_match) {
      // Non-owning Constant over the shared pool bytes (no copy). The pool span
      // views the ONE LiteRt weight mmap, which outlives this compile, so a null
      // keep-alive is safe. get_data_ptr() now returns the pool pointer, so
      // every partition's Constant for this BufferId shares one address -> NPUW
      // dedups.
      auto aliased_cnst = std::make_shared<ov::op::v0::Constant>(
          cnst->get_element_type(), cnst->get_shape(), pool_bytes.data(),
          std::shared_ptr<void>{});
      aliased_cnst->set_friendly_name(cnst->get_friendly_name());
      aliased_cnst->get_output_tensor(0).set_names(
          cnst->get_output_tensor(0).get_names());
      ov::replace_node(cnst, aliased_cnst);
      tag_target = std::move(aliased_cnst);
      ++aliased;
    } else {
      ++mismatched;
    }

    // Stamp the WLCA (bin_offset) on whichever Constant now lives in the graph.
    auto& rt = tag_target->get_rt_info();
    if (!rt.count(ov::WeightlessCacheAttribute::get_type_info_static())) {
      rt[ov::WeightlessCacheAttribute::get_type_info_static()] =
          ov::WeightlessCacheAttribute(tag_target->get_byte_size(),
                                       off_it->second,
                                       tag_target->get_element_type());
      ++tagged;
    }
  }

  LITERT_LOG(LITERT_INFO,
             "Weight sharing (NPU) p%d: aliased %zu constants to the shared "
             "pool, tagged %zu with WeightlessCacheAttribute, left %zu unshared "
             "(bytes differ from pool -- content-altered, e.g. i2->u2)",
             partition_idx, aliased, tagged, mismatched);
  return aliased;
}

}  // namespace litert::openvino
