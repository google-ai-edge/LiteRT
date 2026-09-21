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

#include "litert/experimental/custom_ops/bidirectional_lstm/bidirectional_lstm_impl.h"

#include <cstdint>

#include "Eigen/Core"  // from @eigen_archive

namespace litert {
namespace custom_ops {
namespace {

using RowMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Clamps `seq_lengths[b] - 1 - t` into a valid time index, reproducing the
// exporter's `(seq_lengths - 1 - arange).clamp(min=0)`. The upper clamp is
// defensive: a sequence length larger than the bucket would otherwise index
// out of bounds.
int ReverseIndex(int length, int t, int t_steps) {
  const int idx = length - 1 - t;
  if (idx < 0) return 0;
  if (idx > t_steps - 1) return t_steps - 1;
  return idx;
}

}  // namespace

void RunUnidirectionalLstm(const float* x, const float* w_ih,
                           const float* w_hh, const float* bias, float* out,
                           int t_steps, int input_size, int hidden_size) {
  const int h = hidden_size;
  Eigen::Map<const RowMatrix> x_map(x, t_steps, input_size);
  Eigen::Map<const RowMatrix> w_ih_map(w_ih, 4 * h, input_size);
  Eigen::Map<const RowMatrix> w_hh_map(w_hh, 4 * h, h);
  Eigen::Map<const Eigen::RowVectorXf> bias_map(bias, 4 * h);
  Eigen::Map<RowMatrix> out_map(out, t_steps, h);

  // The input projection does not depend on the recurrent state, so the whole
  // sequence is one GEMM hoisted out of the time loop. This is where nearly all
  // of the arithmetic lives; the loop below is only T small GEMVs.
  RowMatrix gates_all = x_map * w_ih_map.transpose();  // [T, 4H]
  gates_all.rowwise() += bias_map;

  Eigen::VectorXf state = Eigen::VectorXf::Zero(h);
  Eigen::VectorXf cell = Eigen::VectorXf::Zero(h);
  Eigen::VectorXf gates(4 * h);
  Eigen::ArrayXf input_gate(h);
  Eigen::ArrayXf forget_gate(h);
  Eigen::ArrayXf cell_gate(h);
  Eigen::ArrayXf output_gate(h);

  for (int t = 0; t < t_steps; ++t) {
    gates.noalias() = w_hh_map * state;
    gates += gates_all.row(t).transpose();

    input_gate = (1.0f + (-gates.segment(0, h).array()).exp()).inverse();
    forget_gate = (1.0f + (-gates.segment(h, h).array()).exp()).inverse();
    cell_gate = gates.segment(2 * h, h).array().tanh();
    output_gate = (1.0f + (-gates.segment(3 * h, h).array()).exp()).inverse();

    cell.array() = forget_gate * cell.array() + input_gate * cell_gate;
    state.array() = output_gate * cell.array().tanh();
    out_map.row(t) = state.transpose();
  }
}

void ComputeBidirectionalLstm(const float* x, const int32_t* seq_lengths,
                              const float* mask, const float* w_ih_fwd,
                              const float* w_hh_fwd, const float* b_fwd,
                              const float* w_ih_bwd, const float* w_hh_bwd,
                              const float* b_bwd, float* out, float* scratch,
                              int batch, int t_steps, int input_size,
                              int hidden_size) {
  const int h = hidden_size;
  const int d = input_size;
  const int out_stride = 2 * h;

  float* fwd_out = scratch;                       // [T, H]
  float* reversed_in = fwd_out + t_steps * h;     // [T, D]
  float* reversed_out = reversed_in + t_steps * d;  // [T, H]

  for (int b = 0; b < batch; ++b) {
    const float* x_b = x + static_cast<int64_t>(b) * t_steps * d;
    const float* mask_b = mask + static_cast<int64_t>(b) * t_steps;
    float* out_b = out + static_cast<int64_t>(b) * t_steps * out_stride;
    const int length = seq_lengths[b];

    RunUnidirectionalLstm(x_b, w_ih_fwd, w_hh_fwd, b_fwd, fwd_out, t_steps, d,
                          h);

    // Gather the input into "last real token first" order and zero padding.
    for (int t = 0; t < t_steps; ++t) {
      const float keep = 1.0f - mask_b[t];
      const float* src = x_b + ReverseIndex(length, t, t_steps) * d;
      float* dst = reversed_in + t * d;
      for (int i = 0; i < d; ++i) {
        dst[i] = src[i] * keep;
      }
    }

    RunUnidirectionalLstm(reversed_in, w_ih_bwd, w_hh_bwd, b_bwd, reversed_out,
                          t_steps, d, h);

    // Gather back into forward time order, mask, and concatenate. The forward
    // half is intentionally not masked, matching the exporter.
    for (int t = 0; t < t_steps; ++t) {
      const float keep = 1.0f - mask_b[t];
      const float* fwd = fwd_out + t * h;
      const float* bwd = reversed_out + ReverseIndex(length, t, t_steps) * h;
      float* dst = out_b + t * out_stride;
      for (int i = 0; i < h; ++i) {
        dst[i] = fwd[i];
      }
      for (int i = 0; i < h; ++i) {
        dst[h + i] = bwd[i] * keep;
      }
    }
  }
}

}  // namespace custom_ops
}  // namespace litert
