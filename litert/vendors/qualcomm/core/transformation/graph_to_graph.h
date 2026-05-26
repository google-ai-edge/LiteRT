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
  // Experimental MHA decode optimization. Not selectable from the
  // graph_transform flag; used by tests only.
  kExperimental = 1u << 2,
};

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
