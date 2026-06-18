/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_UTILS_H_

#include "absl/random/random.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/internal/graph.h"


namespace litert {
namespace tensor {

inline bool IsComparisonOp(const graph::Operation& op) {
  return op.GetName() == "Equal" || op.GetName() == "NotEqual" ||
         op.GetName() == "Greater" || op.GetName() == "GreaterEqual" ||
         op.GetName() == "Less" || op.GetName() == "LessEqual";
}

template <typename T>
void FillRandom(absl::Span<T> data, int seed = 0) {
  absl::SeedSeq seq = {seed};
  absl::BitGen bitgen(seq);
  for (auto& value : data) {
    value = static_cast<T>(absl::Uniform<T>(bitgen, 0, 100));
  }
}

template <typename T>
void FillRandom(absl::Span<T> data, int seed, T min_val, T max_val) {
  absl::SeedSeq seq = {seed};
  absl::BitGen bitgen(seq);
  for (auto& value : data) {
    value = absl::Uniform<T>(absl::IntervalClosed, bitgen, min_val, max_val);
  }
}


}  // namespace tensor
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_UTILS_H_
