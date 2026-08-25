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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_FEED_FORWARD_NETWORK_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_FEED_FORWARD_NETWORK_H_

#include "tensor/arithmetic.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4 {

// Builds a feed forward network.
//
// The expected input shapes are:
//
// - input: [batch, seq_len, embed_dim]
// - gate_proj: [hidden_dim, embed_dim]
// - up_proj: [hidden_dim, embed_dim]
// - down_proj: [embed_dim, hidden_dim]
template <class... Mixins>
Tensor<Mixins...> FeedForwardNetwork(const Tensor<Mixins...>& input,
                                     const Tensor<Mixins...>& gate_proj,
                                     const Tensor<Mixins...>& up_proj,
                                     const Tensor<Mixins...>& down_proj) {
  Tensor up = FullyConnected(input, up_proj);
  Tensor gate_proj_tensor = FullyConnected(input, gate_proj);
  Tensor gate = Gelu(gate_proj_tensor, /*approximate=*/true);
  Tensor mul_out = Mul(up, gate);
  return FullyConnected(mul_out, down_proj);
}

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_HELPERS_FEED_FORWARD_NETWORK_H_
