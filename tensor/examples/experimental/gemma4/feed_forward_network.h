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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_FEED_FORWARD_NETWORK_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_FEED_FORWARD_NETWORK_H_

#include <string>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/examples/experimental/gemma4/mobile_fully_connected.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Builds a GeGLU feed-forward network block using `MobileFullyConnected` so
// optional static INT8 activation scales from `weights` are applied when
// present.
//
// The expected input shapes are:
// - input: [batch, seq_len, embed_dim]
// - gate_proj: [hidden_dim, embed_dim]
// - up_proj: [hidden_dim, embed_dim]
// - down_proj: [embed_dim, hidden_dim]
template <class... Mixins>
Tensor<Mixins...> FeedForwardNetwork(
    const Tensor<Mixins...>& input, const Tensor<Mixins...>& gate_proj,
    const Tensor<Mixins...>& up_proj, const Tensor<Mixins...>& down_proj,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>* absl_nullable
        weights = nullptr) {
  LRT_TENSOR_RETURN_IF_ERROR(input);
  LRT_TENSOR_RETURN_IF_ERROR(gate_proj);
  LRT_TENSOR_RETURN_IF_ERROR(up_proj);
  LRT_TENSOR_RETURN_IF_ERROR(down_proj);
  Tensor<Mixins...> up = MobileFullyConnected(input, up_proj, weights);
  LRT_TENSOR_RETURN_IF_ERROR(up);
  Tensor<Mixins...> gate_proj_tensor =
      MobileFullyConnected(input, gate_proj, weights);
  LRT_TENSOR_RETURN_IF_ERROR(gate_proj_tensor);
  Tensor<Mixins...> gate = Gelu(gate_proj_tensor, /*approximate=*/true);
  Tensor<Mixins...> mul_out = Mul(up, gate);
  return MobileFullyConnected(mul_out, down_proj, weights);
}

}  // namespace litert::tensor::examples::experimental::gemma4::cpu

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_FEED_FORWARD_NETWORK_H_
