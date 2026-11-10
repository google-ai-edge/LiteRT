// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_BIDIRECTIONAL_LSTM_BIDIRECTIONAL_LSTM_IMPL_H_
#define ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_BIDIRECTIONAL_LSTM_BIDIRECTIONAL_LSTM_IMPL_H_

#include <cstdint>

namespace litert {
namespace custom_ops {

// Runs one unidirectional LSTM over a single padded sequence.
//
// Gates are packed in PyTorch's `nn.LSTM` order (input, forget, cell, output),
// and `bias` is the sum of PyTorch's `bias_ih` and `bias_hh` -- only the sum is
// observable. The initial hidden and cell states are zero.
//
// Shapes, all row-major and densely packed:
//   x    [T, D]      w_ih [4H, D]    w_hh [4H, H]
//   bias [4H]        out  [T, H]
void RunUnidirectionalLstm(const float* x, const float* w_ih,
                           const float* w_hh, const float* bias, float* out,
                           int t_steps, int input_size, int hidden_size);

// Runs a bidirectional LSTM layer with length-aware reverse gathering and
// explicit padding masks.
//
// The backward direction is not a plain reverse scan. The input is gathered
// through `idx[t] = clamp(seq_lengths[b] - 1 - t, 0, T - 1)` so that the scan
// starts at the last real token rather than at the last padded one, then the
// output is gathered back through the same indices. Padding is zeroed
// arithmetically (multiply by `1 - mask`) on both the gathered input and the
// gathered output; the forward direction is deliberately left unmasked.
//
// Shapes, all row-major and densely packed:
//   x           [B, T, D]     seq_lengths [B]      mask [B, T]
//   w_ih_*      [4H, D]       w_hh_*      [4H, H]  b_*  [4H]
//   out         [B, T, 2H]    (forward in [0, H), backward in [H, 2H))
//
// `scratch` must hold at least `t_steps * (input_size + 2 * hidden_size)`
// floats. It is caller-provided so that a repeated Run() does not reallocate.
void ComputeBidirectionalLstm(const float* x, const int32_t* seq_lengths,
                              const float* mask, const float* w_ih_fwd,
                              const float* w_hh_fwd, const float* b_fwd,
                              const float* w_ih_bwd, const float* w_hh_bwd,
                              const float* b_bwd, float* out, float* scratch,
                              int batch, int t_steps, int input_size,
                              int hidden_size);

}  // namespace custom_ops
}  // namespace litert

#endif  // ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_BIDIRECTIONAL_LSTM_BIDIRECTIONAL_LSTM_IMPL_H_
