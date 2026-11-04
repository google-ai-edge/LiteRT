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

#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive

namespace litert {
namespace gated_delta_net {

void ComputeTrilInv(const float* in_data, float* out_data, int total_elements,
                    int C) {
  const int num_matrices = total_elements / (C * C);
  std::vector<float> row(C);
  for (int m = 0; m < num_matrices; ++m) {
    const float* src = in_data + m * C * C;
    float* dst = out_data + m * C * C;
    for (int i = 0; i < C * C; ++i) {
      dst[i] = src[i];
    }
    for (int i = 1; i < C; ++i) {
      for (int k = 0; k < i; ++k) {
        row[k] = dst[i * C + k];
      }
      for (int j = 0; j < i; ++j) {
        float acc = 0.0f;
        for (int k = j + 1; k < i; ++k) {
          acc += row[k] * dst[k * C + j];
        }
        dst[i * C + j] += acc;
      }
    }
    for (int i = 0; i < C; ++i) {
      dst[i * C + i] += 1.0f;
    }
  }
}

void ComputeGatedDeltaUpdateRecurrent(const float* q_t, const float* k_t,
                                      const float* v_t, const float* beta_t,
                                      const float* g_t, const float* rec_state,
                                      float* core_out, float* new_rec, int B,
                                      int H, int N, int D_k, int D_v, int H_k,
                                      const int* valid_len) {
  using Matrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

  const int actual_H_k = (H_k > 0) ? H_k : H;
  const int gqa_ratio = H / actual_H_k;

  for (int bh = 0; bh < B * H; ++bh) {
    const int b = bh / H;
    const int h = bh % H;
    const int bh_k = b * actual_H_k + (h / gqa_ratio);

    Eigen::Map<const Matrix> S_in(rec_state + bh * D_k * D_v, D_k, D_v);
    Eigen::Map<Matrix> S_out(new_rec + bh * D_k * D_v, D_k, D_v);
    S_out = S_in;

    const float* q_bh = q_t + bh_k * N * D_k;
    const float* k_bh = k_t + bh_k * N * D_k;
    const float* v_bh = v_t + bh * N * D_v;
    const float* beta_bh = beta_t + bh * N;
    const float* g_bh = g_t + bh * N;
    float* out_bh = core_out + bh * N * D_v;

    const int active_len =
        (valid_len != nullptr) ? std::clamp(valid_len[b], 0, N) : N;
    for (int t = 0; t < active_len; ++t) {
      // Valid g_t <= 0 and beta_t > 0; right-padded positions use g_t > 0 and
      // beta_t <= 0 sentinels so state neither decays nor updates on padding.
      const float g_val = (g_bh[t] <= 0.0f) ? std::exp(g_bh[t]) : 1.0f;
      const float beta_val = (beta_bh[t] > 0.0f) ? beta_bh[t] : 0.0f;
      Eigen::Map<const Vector> k_val(k_bh + t * D_k, D_k);
      Eigen::Map<const Vector> v_val(v_bh + t * D_v, D_v);
      Eigen::Map<const Vector> q_val(q_bh + t * D_k, D_k);
      Eigen::Map<Vector> out_val(out_bh + t * D_v, D_v);

      S_out *= g_val;
      Vector kv_mem = S_out.transpose() * k_val;
      Vector delta = (v_val - kv_mem) * beta_val;
      S_out.noalias() += k_val * delta.transpose();
      out_val = S_out.transpose() * q_val;
    }
    if (active_len < N) {
      std::memset(out_bh + active_len * D_v, 0,
                  (N - active_len) * D_v * sizeof(float));
    }
  }
}

void ComputeGatedDeltaUpdateChunked(const float* q_t, const float* k_t,
                                    const float* v_t, const float* beta_t,
                                    const float* g_t, const float* rec_state,
                                    float* core_out, float* new_rec, int B,
                                    int H, int N, int D_k, int D_v, int H_k) {
  using Matrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

  const int chunk_size = 64;
  const int actual_H_k = (H_k > 0) ? H_k : H;
  const int gqa_ratio = H / actual_H_k;

  for (int bh = 0; bh < B * H; ++bh) {
    const int b = bh / H;
    const int h = bh % H;
    const int bh_k = b * actual_H_k + (h / gqa_ratio);

    Eigen::Map<const Matrix> S_in(rec_state + bh * D_k * D_v, D_k, D_v);
    Eigen::Map<Matrix> S_out(new_rec + bh * D_k * D_v, D_k, D_v);
    S_out = S_in;

    const float* q_bh = q_t + bh_k * N * D_k;
    const float* k_bh = k_t + bh_k * N * D_k;
    const float* v_bh = v_t + bh * N * D_v;
    const float* beta_bh = beta_t + bh * N;
    const float* g_bh = g_t + bh * N;
    float* out_bh = core_out + bh * N * D_v;

    int t = 0;
    for (; t <= N - chunk_size; t += chunk_size) {
      Eigen::Map<const Matrix> Q_c(q_bh + t * D_k, chunk_size, D_k);
      Eigen::Map<const Matrix> K_c(k_bh + t * D_k, chunk_size, D_k);
      Eigen::Map<const Matrix> V_c(v_bh + t * D_v, chunk_size, D_v);
      Eigen::Map<const Vector> beta_c(beta_bh + t, chunk_size);
      Eigen::Map<const Vector> g_c(g_bh + t, chunk_size);
      Eigen::Map<Matrix> Out_c(out_bh + t * D_v, chunk_size, D_v);

      Vector g_cumsum(chunk_size);
      float sum = 0.0f;
      for (int i = 0; i < chunk_size; ++i) {
        sum += (g_c[i] <= 0.0f) ? g_c[i] : 0.0f;
        g_cumsum[i] = sum;
      }

      Vector beta_clean = beta_c.cwiseMax(0.0f);
      Matrix K_beta = K_c.array().colwise() * beta_clean.array();
      Matrix V_beta = V_c.array().colwise() * beta_clean.array();

      Matrix A_in = Matrix::Zero(chunk_size, chunk_size);
      Matrix K_KT = K_beta * K_c.transpose();
      for (int i = 1; i < chunk_size; ++i) {
        for (int j = 0; j < i; ++j) {
          float decay = std::exp(g_cumsum[i] - g_cumsum[j]);
          A_in(i, j) = -K_KT(i, j) * decay;
        }
      }

      Matrix A = Matrix::Zero(chunk_size, chunk_size);
      ComputeTrilInv(A_in.data(), A.data(), chunk_size * chunk_size,
                     chunk_size);

      Matrix v_local = A * V_beta;

      Vector exp_g = g_cumsum.array().exp();
      Matrix K_beta_exp = K_beta.array().colwise() * exp_g.array();
      Matrix k_cumdecay = A * K_beta_exp;

      Matrix Q_exp = Q_c.array().colwise() * exp_g.array();
      Matrix attn_inter = Q_exp * S_out;

      Matrix v_prime = k_cumdecay * S_out;

      Matrix v_new = v_local - v_prime;

      Matrix A_chunk = Matrix::Zero(chunk_size, chunk_size);
      Matrix Q_KT = Q_c * K_c.transpose();
      for (int i = 0; i < chunk_size; ++i) {
        for (int j = 0; j <= i; ++j) {
          float decay = std::exp(g_cumsum[i] - g_cumsum[j]);
          A_chunk(i, j) = Q_KT(i, j) * decay;
        }
      }

      Out_c = attn_inter + A_chunk * v_new;

      float s_last = g_cumsum[chunk_size - 1];
      Vector exp_last_minus_s = (s_last - g_cumsum.array()).exp();
      Matrix K_scaled = K_c.array().colwise() * exp_last_minus_s.array();
      S_out = S_out * std::exp(s_last) + K_scaled.transpose() * v_new;
    }

    for (; t < N; ++t) {
      // Clamp right-padding sentinel values (g_t > 0, beta_t <= 0).
      const float g_val = (g_bh[t] <= 0.0f) ? std::exp(g_bh[t]) : 1.0f;
      const float beta_val = (beta_bh[t] > 0.0f) ? beta_bh[t] : 0.0f;
      Eigen::Map<const Vector> k_val(k_bh + t * D_k, D_k);
      Eigen::Map<const Vector> v_val(v_bh + t * D_v, D_v);
      Eigen::Map<const Vector> q_val(q_bh + t * D_k, D_k);
      Eigen::Map<Vector> out_val(out_bh + t * D_v, D_v);

      S_out *= g_val;
      Vector kv_mem = S_out.transpose() * k_val;
      Vector delta = (v_val - kv_mem) * beta_val;
      S_out.noalias() += k_val * delta.transpose();
      out_val = S_out.transpose() * q_val;
    }
  }
}

// Fused GatedDeltaNet block (math in the header), in three passes:
//  1. Per token: split `conv_out` and `zba`, L2-normalize q and k, and compute
//     beta and g into the [B, H, T, D] layout of the recurrence-only op.
//     Masked tokens get q = beta = g = 0.
//  2. Run the shared recurrence, ComputeGatedDeltaUpdateRecurrent, up to the
//     last valid token.
//  3. Gated RMSNorm: rmsnorm(y) * norm_weight * silu(z).
void ComputeGatedDeltaUpdateBlock(
    const float* conv_out, const float* zba, const float* a_log,
    const float* dt_bias, const float* norm_weight, const float* rec_state,
    const float* valid_mask, float* output, float* new_rec, int B, int T,
    const GatedDeltaBlockParams& params, int zba_stride) {
  const int H_k = params.num_k_heads;
  const int H_v = params.num_v_heads;
  const int D_k = params.head_k_dim;
  const int D_v = params.head_v_dim;
  const int K_dim = H_k * D_k;
  const int V_dim = H_v * D_v;
  const int conv_channels = 2 * K_dim + V_dim;
  const int zba_channels = V_dim + 2 * H_v;
  const int actual_zba_stride = (zba_stride > 0) ? zba_stride : zba_channels;
  const int zba_offset = actual_zba_stride - zba_channels;
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(D_k));
  const float inv_head_v_dim = 1.0f / static_cast<float>(D_v);

  std::vector<float> neg_a(H_v);
  for (int h = 0; h < H_v; ++h) {
    neg_a[h] = -std::exp(a_log[h]);
  }

  std::vector<float> q_t(B * H_k * T * D_k);
  std::vector<float> k_t(B * H_k * T * D_k);
  std::vector<float> v_t(B * H_v * T * D_v);
  std::vector<float> beta_t(B * H_v * T);
  std::vector<float> g_t(B * H_v * T);
  std::vector<int> valid_len(B, T);

  for (int b = 0; b < B; ++b) {
    int last_valid = 0;
    for (int t = 0; t < T; ++t) {
      const float m = (valid_mask != nullptr) ? valid_mask[b * T + t] : 1.0f;
      if (m != 0.0f) {
        last_valid = t + 1;
      }
      const float* conv_bt = conv_out + (b * T + t) * conv_channels;
      const float* zba_bt = zba + (b * T + t) * actual_zba_stride + zba_offset;

      for (int h_k = 0; h_k < H_k; ++h_k) {
        const float* q_in = conv_bt + h_k * D_k;
        const float* k_in = conv_bt + K_dim + h_k * D_k;
        float* q_dst = q_t.data() + ((b * H_k + h_k) * T + t) * D_k;
        float* k_dst = k_t.data() + ((b * H_k + h_k) * T + t) * D_k;

        if (m == 0.0f) {
          std::memset(q_dst, 0, D_k * sizeof(float));
        } else {
          float q_sumsq = 0.0f;
          for (int d = 0; d < D_k; ++d) {
            const float qv = q_in[d] * m;
            q_sumsq += qv * qv;
          }
          const float q_inv = q_scale / std::sqrt(q_sumsq + params.eps);
          for (int d = 0; d < D_k; ++d) {
            q_dst[d] = q_in[d] * m * q_inv;
          }
        }

        float k_sumsq = 0.0f;
        for (int d = 0; d < D_k; ++d) {
          k_sumsq += k_in[d] * k_in[d];
        }
        const float k_inv = 1.0f / std::sqrt(k_sumsq + params.eps);
        for (int d = 0; d < D_k; ++d) {
          k_dst[d] = k_in[d] * k_inv;
        }
      }

      for (int h = 0; h < H_v; ++h) {
        const float* v_in = conv_bt + 2 * K_dim + h * D_v;
        float* v_dst = v_t.data() + ((b * H_v + h) * T + t) * D_v;
        std::memcpy(v_dst, v_in, D_v * sizeof(float));

        const int bt_idx = (b * H_v + h) * T + t;
        if (m == 0.0f) {
          beta_t[bt_idx] = 0.0f;
          g_t[bt_idx] = 0.0f;
        } else {
          const float b_raw = zba_bt[V_dim + h];
          const float a_raw = zba_bt[V_dim + H_v + h];
          beta_t[bt_idx] = m / (1.0f + std::exp(-b_raw));
          const float a_in = a_raw + dt_bias[h];
          const float softplus_a =
              std::max(a_in, 0.0f) + std::log1pf(std::exp(-std::abs(a_in)));
          g_t[bt_idx] = neg_a[h] * softplus_a * m;
        }
      }
    }
    if (valid_mask != nullptr) {
      valid_len[b] = last_valid;
    }
  }

  std::vector<float> core_out(B * H_v * T * D_v);
  ComputeGatedDeltaUpdateRecurrent(
      q_t.data(), k_t.data(), v_t.data(), beta_t.data(), g_t.data(), rec_state,
      core_out.data(), new_rec, B, H_v, T, D_k, D_v, H_k,
      (valid_mask != nullptr) ? valid_len.data() : nullptr);

  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      const float m = (valid_mask != nullptr) ? valid_mask[b * T + t] : 1.0f;
      const float* zba_bt = zba + (b * T + t) * actual_zba_stride + zba_offset;
      float* out_bt = output + (b * T + t) * V_dim;
      if (m == 0.0f) {
        std::memset(out_bt, 0, V_dim * sizeof(float));
        continue;
      }
      for (int h = 0; h < H_v; ++h) {
        const float* y_in = core_out.data() + ((b * H_v + h) * T + t) * D_v;
        const float* z_in = zba_bt + h * D_v;
        float* out_ptr = out_bt + h * D_v;
        float sumsq = 0.0f;
        for (int d = 0; d < D_v; ++d) {
          sumsq += y_in[d] * y_in[d];
        }
        const float inv_rms =
            1.0f / std::sqrt(sumsq * inv_head_v_dim + params.norm_eps);
        for (int d = 0; d < D_v; ++d) {
          const float z_val = z_in[d];
          const float silu_z = z_val / (1.0f + std::exp(-z_val));
          out_ptr[d] = y_in[d] * inv_rms * norm_weight[d] * silu_z;
        }
      }
    }
  }
}

}  // namespace gated_delta_net
}  // namespace litert
