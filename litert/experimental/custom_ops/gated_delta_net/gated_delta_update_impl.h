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
                                      int H, int N, int D_k, int D_v);

// Chunked implementation of gated delta update (using Eigen).
void ComputeGatedDeltaUpdateChunked(const float* q_t, const float* k_t,
                                    const float* v_t, const float* beta_t,
                                    const float* g_t, const float* rec_state,
                                    float* core_out, float* new_rec, int B,
                                    int H, int N, int D_k, int D_v);

}  // namespace gated_delta_net
}  // namespace litert

#endif  // ODML_LITERT_LITERT_EXPERIMENTAL_CUSTOM_OPS_GATED_DELTA_NET_GATED_DELTA_UPDATE_IMPL_H_
