/* Copyright 2026 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_ROPE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_ROPE_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Controls whether split-half RoPE uses fixed slice extents (`kStatic`) or
// slice-to-end `-1` extents (`kDynamic`) on leading dimensions so resized
// prefill inputs do not retain the authoring-time query row count.
enum class RopeLeadingDims {
  kStatic,
  kDynamic,
};

// Applies split-half Rotary Position Embedding (RoPE) to `input` using
// precomputed `cosine` and `sine` tables. Named `SplitHalfRoPE` rather than
// `RotaryEmbedding` to avoid shadowing the fused
// `litert::tensor::RotaryEmbedding` op in `transformer_ops.h`. Returns an error
// tensor if any input has a non-OK status, `input` is scalar, or the last
// dimension (`head_dim`) is not positive and even.
template <class... Mixins>
Tensor<Mixins...> SplitHalfRoPE(const Tensor<Mixins...>& input,
                                const Tensor<Mixins...>& cosine,
                                const Tensor<Mixins...>& sine,
                                RopeLeadingDims leading_dims) {
  if (leading_dims == RopeLeadingDims::kStatic) {
    return RoPE(input, cosine, sine);
  }
  LRT_TENSOR_RETURN_IF_ERROR(input);
  LRT_TENSOR_RETURN_IF_ERROR(cosine);
  LRT_TENSOR_RETURN_IF_ERROR(sine);
  const Shape& shape = input.GetShape();
  if (shape.empty()) {
    return absl::InvalidArgumentError(
        "RoPE expects input with rank >= 1, got scalar.");
  }
  if (shape.back() <= 0 || shape.back() % 2 != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("RoPE expects an even last dimension (head_dim), got ",
                     shape.back(), "."));
  }
  graph::ScopeGuard scope("rope");
  scope.Bind("input", input.GetRaw())
      .Bind("cos", cosine.GetRaw())
      .Bind("sin", sine.GetRaw());
  const int axis = static_cast<int>(shape.size()) - 1;
  const int half = shape.back() / 2;
  Shape sizes(shape.size(), -1);
  sizes.back() = half;
  Shape offsets(shape.size(), 0);
  Tensor<Mixins...> first = Slice(input, offsets, sizes);
  offsets.back() = half;
  Tensor<Mixins...> second = Slice(input, offsets, sizes);
  Tensor<Mixins...> neg_second = Neg(second);
  Tensor<Mixins...> rotated = Concatenation({neg_second, first}, axis);
  Tensor<Mixins...> input_cosine = Mul(input, cosine);
  Tensor<Mixins...> rotated_sine = Mul(rotated, sine);
  Tensor<Mixins...> output = Add(input_cosine, rotated_sine);
  scope.Bind("output", output.GetRaw());
  return output;
}

}  // namespace litert::tensor::examples::experimental::gemma4::cpu

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_ROPE_H_
