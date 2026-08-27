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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_ROPE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_ROPE_H_

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4 {

// Calculates RoPE cos and sin frequency table values into caller-provided
// spans.
inline void RopeCosSin(const int start, const int seq_len, const int head_dim,
                       const float rope_base, const float rope_proportion,
                       const absl::Span<float> cos,
                       const absl::Span<float> sin) {
  ABSL_DCHECK_GE(cos.size(), static_cast<size_t>(seq_len) * head_dim);
  ABSL_DCHECK_GE(sin.size(), static_cast<size_t>(seq_len) * head_dim);

  const int half_dim = head_dim / 2;
  const int rope_angles =
      static_cast<int>(rope_proportion * static_cast<float>(half_dim));
  static thread_local std::vector<float> inv_freq;
  inv_freq.assign(half_dim, 0.0f);
  for (int i = 0; i < rope_angles; ++i) {
    inv_freq[i] = 1.0f / std::pow(rope_base, 2.0f * i / head_dim);
  }
  for (int s = 0, position = start; s < seq_len; ++s, ++position) {
    const size_t embedding_offset = s * head_dim;
    for (int i = 0; i < half_dim; ++i) {
      const float angle = static_cast<float>(position) * inv_freq[i];
      const float cos_val = std::cos(angle);
      const float sin_val = std::sin(angle);
      cos[embedding_offset + i] = cos_val;
      cos[embedding_offset + half_dim + i] = cos_val;
      sin[embedding_offset + i] = sin_val;
      sin[embedding_offset + half_dim + i] = sin_val;
    }
  }
}

// Allocates and populates RoPE cos and sin tensors for a sequence.
template <class... Mixins>
std::pair<Tensor<Mixins...>, Tensor<Mixins...>> RopeCosSin(
    const int seq_len, const int head_dim, const float rope_base,
    const float rope_proportion) {
  const Shape shape({1, 1, seq_len, head_dim});
  auto cos_data = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
  auto sin_data = OwningCpuBuffer::Allocate<Type::kFP32>(shape);
  Tensor<Mixins...> cos(
      {.name = "cos", .type = Type::kFP32, .shape = shape, .buffer = cos_data});
  Tensor<Mixins...> sin(
      {.name = "sin", .type = Type::kFP32, .shape = shape, .buffer = sin_data});
  RopeCosSin(/*start=*/0, seq_len, head_dim, rope_base, rope_proportion,
             cos_data->Span<float>(), sin_data->Span<float>());
  return {cos, sin};
}

// Applies the split-half variant of Rotary Position Embedding (RoPE).
//
// In this variant, the last dimension (head_dim) is split into two halves:
// x = [x1, x2]. The rotated tensor is defined as: rotated = [-x2, x1].
// The output is computed as: x * cos + rotated * sin.
template <class... Mixins>
Tensor<Mixins...> RoPE(const Tensor<Mixins...>& x, const Tensor<Mixins...>& cos,
                       const Tensor<Mixins...>& sin) {
  // We assume a shape of [batch, ..., head_dim].
  const Shape& x_shape = x.GetShape();
  const int half_dim = x_shape[3] / 2;
  const Shape slice_size = [&] {
    Shape s = x_shape;
    s.back() = half_dim;
    return s;
  }();

  // Split x in half along head_dim.
  std::vector<int> slice_begin(x_shape.size(), 0);
  Tensor x1 = Slice(x, slice_begin, slice_size);
  slice_begin.back() = half_dim;
  Tensor x2 = Slice(x, slice_begin, slice_size);

  Tensor neg_one = Tensor<Mixins...>(
      {.type = Type::kFP32,
       .shape = {1},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({-1.0f})});
  Tensor neg_x2 = Mul(x2, neg_one);
  Tensor rotated = Concatenation({neg_x2, x1}, /*axis=*/3);

  Tensor x_cos = Mul(x, cos);
  Tensor rotated_sin = Mul(rotated, sin);
  return Add(x_cos, rotated_sin);
}

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_ROPE_H_
