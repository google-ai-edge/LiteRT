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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_HFA_HISTORY_H_
#define LITERT_VENDORS_INTEL_OPENVINO_HFA_HISTORY_H_

#include <cstdint>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/cc/litert_expected.h"

namespace litert {
namespace openvino {

// Host Flash Attention (HFA) in NPUW skips the past KV tiles of a prefill
// chunk that hold no history. On a graph without a "position_ids" input it
// takes the history (the number of tokens already in the KV cache) from the
// infer request's "npuw_history_size" variable state, which the host sets
// before every inference. The LiteRT prefill graphs carry no position input,
// so the history is read from their attention mask instead:
//   - the compiler (enable_hfa) adds kHfaHistoryMaskTensorName to the tensor
//     names of the graph input holding the mask of the SDPAs it decomposed;
//   - dispatch reads that input with HistorySizeFromMask() before each
//     inference and writes the result to the state.
inline constexpr char kHfaHistoryMaskTensorName[] = "litert_hfa_history_mask";
inline constexpr char kNpuwHistorySizeStateName[] = "npuw_history_size";

// Whether a mask of this type and shape can be read by HistorySizeFromMask():
// f32, f16 or bf16, [1, 1, Q, past + Q] with Q > 0 and past > 0.
Expected<void> CheckHistoryMask(const ov::element::Type& type,
                                const ov::Shape& shape);

// Reads the history from an additive attention mask [1, 1, Q, past + Q] whose
// first |past| columns are the KV cache and whose last Q are the chunk itself.
// Returns one past the last cache column any query row can see, so every
// visible cache entry is covered even if the visible ones are not a prefix.
// Within a row, the masked value is taken to be the row's minimum. A row with
// a single value is fully masked if that value is negative (e.g. a padding
// row) and fully visible otherwise.
Expected<int64_t> HistorySizeFromMask(const ov::Tensor& mask);

}  // namespace openvino
}  // namespace litert

#endif  // LITERT_VENDORS_INTEL_OPENVINO_HFA_HISTORY_H_
