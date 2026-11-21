// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MHROPE_TO_SHROPE_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MHROPE_TO_SHROPE_H_

#include <cstddef>
#include <functional>
#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

// Transform Multi-Head RoPE to Single-Head RoPE, eliminating the final
// Transpose by splitting the computation per head.
//
// Original pattern (input shape [B,S,H,D]):
//   StridedSlice → StridedSlice → Concat → Mul(cos) → Mul(sin) →
//   Add → Convert → Transpose([0,2,1,3])
//
// Replacement (output shape [B,H,S,D]):
//   Unpack(axis=2) → H × [StridedSlice → StridedSlice → Concat →
//   Mul(cos) → Mul(sin) → Add] → Pack(axis=1) → Convert
size_t MHRoPEToSHRoPE(std::function<bool(OpWrapper&)> validate_op_config,
                      std::vector<OpWrapper>& ops, size_t start_index,
                      TensorPool& tensor_pool, size_t pattern_size);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MHROPE_TO_SHROPE_H_
