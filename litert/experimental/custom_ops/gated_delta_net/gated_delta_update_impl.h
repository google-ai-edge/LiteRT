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

#ifndef ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_GATED_DELTA_NET_GATED_DELTA_UPDATE_IMPL_H_
#define ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_GATED_DELTA_NET_GATED_DELTA_UPDATE_IMPL_H_

namespace litert {
namespace gated_delta_net {

// Computes exact forward substitution for strictly lower triangular intra-chunk
// matrix inverse: (I + A)^{-1}.
void ComputeTrilInv(const float* in_data, float* out_data, int total_elements,
                    int C);

// Recurrent implementation of gated delta update (optimized with Eigen).
void ComputeGatedDeltaUpdateRecurrent(const float* q_t, const float* k_t,
                                      const float* v_t, const float* beta_t,
                                      const float* g_t, const float* rec_state,
                                      float* core_out, float* new_rec, int B,
                                      int H, int N, int D_k, int D_v,
                                      int H_k = -1,
                                      const int* valid_len = nullptr);

// Chunked implementation of gated delta update (using Eigen).
void ComputeGatedDeltaUpdateChunked(const float* q_t, const float* k_t,
                                    const float* v_t, const float* beta_t,
                                    const float* g_t, const float* rec_state,
                                    float* core_out, float* new_rec, int B,
                                    int H, int N, int D_k, int D_v,
                                    int H_k = -1);

// Parameters of the block form of `gated_delta_update`. The custom op uses the
// block form when `num_k_heads` is set in its options; otherwise it is the
// recurrence-only op above.
struct GatedDeltaBlockParams {
  int num_k_heads = 0;     // H_k: number of q / k heads.
  int num_v_heads = 0;     // H_v: number of value heads, a multiple of H_k.
  int head_k_dim = 0;      // D_k.
  int head_v_dim = 0;      // D_v.
  float eps = 1e-6f;       // Epsilon of the q / k L2 norms.
  float norm_eps = 1e-6f;  // Epsilon of the gated RMSNorm.
};

// CPU implementation of the block form of `gated_delta_update`: everything in
// a Qwen3.5 GatedDeltaNet (linear attention) layer between the short conv and
// `out_proj`. Without the block form, the exporter emits the same math as
// builtin ops around the recurrence-only op.
//
// With K_dim = H_k * D_k and V_dim = H_v * D_v, for each token:
//   [q | k | v] = conv_out               (K_dim | K_dim | V_dim channels)
//   [z | b | a] = zba                    (V_dim | H_v | H_v channels)
//   q = l2norm(q) / sqrt(D_k), k = l2norm(k)      (per q / k head)
//   beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
//   S = exp(g) * S                               (per value head)
//   S = S + k * (beta * (v - S^T * k))^T
//   y = S^T * q
//   output = rmsnorm(y) * norm_weight * silu(z)  (per value head)
// where l2norm(x) = x / sqrt(sum(x^2) + eps) and
// rmsnorm(y) = y / sqrt(mean(y^2) + norm_eps). Value head h reads q / k head
// h / (H_v / H_k), and S, its D_k x D_v recurrent state, starts from
// `rec_state`. Tokens whose `valid_mask` is 0 leave S unchanged and get a zero
// output row.
//
// Shapes: conv_out [B, T, 2 * K_dim + V_dim], zba [B, T, V_dim + 2 * H_v],
// a_log and dt_bias [H_v], norm_weight [D_v], rec_state and new_rec
// [B, H_v, D_k, D_v], valid_mask [B, T] or nullptr (all tokens valid), and
// output [B, T, V_dim]. `new_rec` receives S after the last token.
//
// A positive `zba_stride` is the number of channels per token in `zba` when
// z/b/a are the last V_dim + 2 * H_v channels of a wider row, e.g. a fused
// [qkv | z | b | a] projection.
void ComputeGatedDeltaUpdateBlock(
    const float* conv_out, const float* zba, const float* a_log,
    const float* dt_bias, const float* norm_weight, const float* rec_state,
    const float* valid_mask, float* output, float* new_rec, int B, int T,
    const GatedDeltaBlockParams& params, int zba_stride = 0);

}  // namespace gated_delta_net
}  // namespace litert

#endif  // ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_GATED_DELTA_NET_GATED_DELTA_UPDATE_IMPL_H_
