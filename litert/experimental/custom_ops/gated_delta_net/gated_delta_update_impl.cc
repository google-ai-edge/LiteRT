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

#include <cmath>
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
                                      int H, int N, int D_k, int D_v) {
  using Matrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

  for (int bh = 0; bh < B * H; ++bh) {
    Eigen::Map<const Matrix> S_in(rec_state + bh * D_k * D_v, D_k, D_v);
    Eigen::Map<Matrix> S_out(new_rec + bh * D_k * D_v, D_k, D_v);
    S_out = S_in;

    const float* q_bh = q_t + bh * N * D_k;
    const float* k_bh = k_t + bh * N * D_k;
    const float* v_bh = v_t + bh * N * D_v;
    const float* beta_bh = beta_t + bh * N;
    const float* g_bh = g_t + bh * N;
    float* out_bh = core_out + bh * N * D_v;

    for (int t = 0; t < N; ++t) {
      const float g_val = std::exp(g_bh[t]);
      const float beta_val = beta_bh[t];
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

void ComputeGatedDeltaUpdateChunked(const float* q_t, const float* k_t,
                                    const float* v_t, const float* beta_t,
                                    const float* g_t, const float* rec_state,
                                    float* core_out, float* new_rec, int B,
                                    int H, int N, int D_k, int D_v) {
  using Matrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

  const int chunk_size = 64;

  for (int bh = 0; bh < B * H; ++bh) {
    Eigen::Map<const Matrix> S_in(rec_state + bh * D_k * D_v, D_k, D_v);
    Eigen::Map<Matrix> S_out(new_rec + bh * D_k * D_v, D_k, D_v);
    S_out = S_in;

    const float* q_bh = q_t + bh * N * D_k;
    const float* k_bh = k_t + bh * N * D_k;
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
        sum += g_c[i];
        g_cumsum[i] = sum;
      }

      Matrix K_beta = K_c.array().colwise() * beta_c.array();
      Matrix V_beta = V_c.array().colwise() * beta_c.array();

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
      const float g_val = std::exp(g_bh[t]);
      const float beta_val = beta_bh[t];
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

}  // namespace gated_delta_net
}  // namespace litert
