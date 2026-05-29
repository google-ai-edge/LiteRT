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
  // Disable G2G.
  kOff = 0b0000, // if graph_transform = ""
  // Enable G2G GQA-to-SHA optimization.
  kGqa = 0b0001, // if "gqa" in graph_transform
  // Simplify masking pattern.
  kMasking = 0b0010,  // if "masking" in graph_transform
  //
  kExperimental = 0b0100,  // if "masking" in graph_transform
};

// Bitwise operators so G2GConfig can be combined as a flag set
// (e.g. kGqa | kMasking == 0b0011).
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
