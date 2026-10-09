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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_ROPE_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_ROPE_H_

#include <algorithm>

#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/examples/ops/transformer/transformer_ops.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4::cpu {

// Split-half RoPE with slices that follow live leading dimensions. Fixed
// execution uses the shared helper; dynamic execution keeps Slice extents at
// -1 so a shorter prefill does not retain the authoring-time query count.
template <class... Mixins>
Tensor<Mixins...> RotaryEmbedding(const Tensor<Mixins...>& input,
                                  const Tensor<Mixins...>& cosine,
                                  const Tensor<Mixins...>& sine,
                                  bool dynamic_leading_dims) {
  if (!dynamic_leading_dims) {
    return RoPE(input, cosine, sine);
  }
  const Shape& shape = input.GetShape();
  const int axis = static_cast<int>(shape.size()) - 1;
  const int half = shape.back() / 2;
  Shape sizes = shape;
  std::fill(sizes.begin(), sizes.end() - 1, -1);
  sizes.back() = half;
  Shape offsets(shape.size(), 0);
  Tensor<Mixins...> first = Slice(input, offsets, sizes);
  offsets.back() = half;
  Tensor<Mixins...> second = Slice(input, offsets, sizes);
  Tensor<Mixins...> neg_second = Neg(second);
  Tensor<Mixins...> rotated = Concatenation({neg_second, first}, axis);
  Tensor<Mixins...> input_cosine = Mul(input, cosine);
  Tensor<Mixins...> rotated_sine = Mul(rotated, sine);
  return Add(input_cosine, rotated_sine);
}

}  // namespace litert::tensor::examples::gemma4::cpu

#endif  // LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_ROPE_H_
