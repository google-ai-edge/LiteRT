// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_GRAPH_TO_GRAPH_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_GRAPH_TO_GRAPH_H_

#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

enum class G2GConfig : std::uint32_t {
  // Disable G2G. Selected when graph_transform is empty.
  kOff = 0,
  // GQA-to-SHA optimization. Selected when "gqa" appears in graph_transform.
  kGqa = 1u << 0,
  // Simplify masking pattern. Selected when "masking" appears in
  // graph_transform.
  kMasking = 1u << 1,
  // Tile large MatMul ops along the K or N axis. Selected when "matmul_tiling"
  // appears in graph_transform.
  kMatMulTiling = 1u << 2,
  // Experimental MHA decode optimization. Not selectable from the
  // graph_transform flag; used by tests only.
  kExperimental = 1u << 3,
};

// Each enumerator above must occupy a distinct bit so the values can be OR'd
// into a flag set. Update this static_assert when adding new flags.
static_assert(static_cast<std::uint32_t>(G2GConfig::kGqa) == 0b0001,
              "G2GConfig::kGqa must occupy bit 0 (value 0b0001) so it can be "
              "OR'd into a flag set without colliding with other flags.");
static_assert(static_cast<std::uint32_t>(G2GConfig::kMasking) == 0b0010,
              "G2GConfig::kMasking must occupy bit 1 (value 0b0010) so it can "
              "be OR'd into a flag set without colliding with other flags.");
static_assert(static_cast<std::uint32_t>(G2GConfig::kMatMulTiling) == 0b0100,
              "G2GConfig::kMatMulTiling must occupy bit 2 (value 0b0100) so it "
              "can be OR'd into a flag set without colliding with other flags.");
static_assert(static_cast<std::uint32_t>(G2GConfig::kExperimental) == 0b1000,
              "G2GConfig::kExperimental must occupy bit 3 (value 0b1000) so it "
              "can be OR'd into a flag set without colliding with other flags.");

// Bitwise operators so G2GConfig can be combined as a flag set
// (e.g. kGqa | kMasking).
constexpr G2GConfig operator|(G2GConfig a, G2GConfig b) {
  using U = std::underlying_type_t<G2GConfig>;
  return static_cast<G2GConfig>(static_cast<U>(a) | static_cast<U>(b));
}
constexpr G2GConfig operator&(G2GConfig a, G2GConfig b) {
  using U = std::underlying_type_t<G2GConfig>;
  return static_cast<G2GConfig>(static_cast<U>(a) & static_cast<U>(b));
}
constexpr G2GConfig& operator|=(G2GConfig& a, G2GConfig b) {
  a = a | b;
  return a;
}

// Returns true if `flag` is set in `cfg`.
constexpr bool HasG2GFlag(G2GConfig cfg, G2GConfig flag) {
  return (cfg & flag) == flag;
}

void GraphToGraphTransform(G2GConfig g2g_option, std::vector<OpWrapper>& ops,
                           TensorPool& tensor_pool,
                           std::function<bool(OpWrapper&)> validate_op_config);
}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_GRAPH_TO_GRAPH_H_
