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

#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_litert_custom_op.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/experimental/custom_ops/gated_delta_net/gated_delta_update_impl.h"
#include "litert/test/matchers.h"

using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace gated_delta_net {
namespace {

// ============================================================================
// Reference Implementation for Golden Value Derivations
// ============================================================================
// Mathematical specification for single recurrent step t:
//   1. S'_t = S_{t-1} * exp(g_t)
//   2. kv_mem = (S'_t)^T * k_t
//   3. delta = (v_t - kv_mem) * beta_t
//   4. S_t = S'_t + k_t * delta^T
//   5. y_t = (S_t)^T * q_t
void ComputeGoldenRecurrentGatedDelta(
    const std::vector<float>& q, const std::vector<float>& k,
    const std::vector<float>& v, const std::vector<float>& beta,
    const std::vector<float>& g, const std::vector<float>& initial_state,
    std::vector<float>& golden_out, std::vector<float>& golden_final_state,
    int B, int H, int N, int D_k, int D_v, int H_k = -1) {
  golden_out.resize(B * H * N * D_v);
  golden_final_state = initial_state;

  const int actual_H_k = (H_k > 0) ? H_k : H;
  const int gqa_ratio = H / actual_H_k;

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const int bh = b * H + h;
      const int bh_k = b * actual_H_k + (h / gqa_ratio);
      float* S = golden_final_state.data() + bh * D_k * D_v;

      for (int t = 0; t < N; ++t) {
        const float* q_t = q.data() + (bh_k * N + t) * D_k;
        const float* k_t = k.data() + (bh_k * N + t) * D_k;
        const float* v_t = v.data() + (bh * N + t) * D_v;
        const float beta_raw = beta[bh * N + t];
        const float beta_t = (beta_raw > 0.0f) ? beta_raw : 0.0f;
        const float g_raw = g[bh * N + t];
        const float g_decay = (g_raw <= 0.0f) ? std::exp(g_raw) : 1.0f;
        float* out_t = golden_out.data() + (bh * N + t) * D_v;

        // 1. Decay state S = S * exp(g)
        for (int i = 0; i < D_k * D_v; ++i) {
          S[i] *= g_decay;
        }

        // 2. kv_mem[j] = sum_i (S[i, j] * k_t[i])
        std::vector<float> kv_mem(D_v, 0.0f);
        for (int j = 0; j < D_v; ++j) {
          for (int i = 0; i < D_k; ++i) {
            kv_mem[j] += S[i * D_v + j] * k_t[i];
          }
        }

        // 3. delta[j] = (v_t[j] - kv_mem[j]) * beta_t
        std::vector<float> delta(D_v, 0.0f);
        for (int j = 0; j < D_v; ++j) {
          delta[j] = (v_t[j] - kv_mem[j]) * beta_t;
        }

        // 4. Update state: S[i, j] += k_t[i] * delta[j]
        for (int i = 0; i < D_k; ++i) {
          for (int j = 0; j < D_v; ++j) {
            S[i * D_v + j] += k_t[i] * delta[j];
          }
        }

        // 5. Read out: y_t[j] = sum_i (S[i, j] * q_t[i])
        for (int j = 0; j < D_v; ++j) {
          float sum = 0.0f;
          for (int i = 0; i < D_k; ++i) {
            sum += S[i * D_v + j] * q_t[i];
          }
          out_t[j] = sum;
        }
      }
    }
  }
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest, TrilInvKernelNameAndVersion) {
  TrilInvCustomOpKernel kernel;
  EXPECT_EQ(kernel.OpName(), "gdn_tril_inv");
  EXPECT_EQ(kernel.OpVersion(), 1);
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest, GatedDeltaUpdateKernelNameAndVersion) {
  GatedDeltaUpdateCustomOpKernel kernel;
  EXPECT_EQ(kernel.OpName(), "gated_delta_update");
  EXPECT_EQ(kernel.OpVersion(), 1);
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest, TrilInvKernelComputation2x2) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  TrilInvCustomOpKernel kernel;

  auto tensor_type = MakeRankedTensorType<float>({1, 2, 2});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  tensor_type, sizeof(float) * 4));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  tensor_type, sizeof(float) * 4));

  // Lower triangular matrix A:
  // [0, 0]
  // [2, 0]
  // Inversion (I + A)^(-1) is [1, 0; 2, 1] for strictly lower triangular
  // forward substitution
  const std::vector<float> input_data = {0.0f, 0.0f, 2.0f, 0.0f};
  ASSERT_TRUE(input_buffer.Write<float>(absl::MakeConstSpan(input_data)));

  std::vector<Layout> in_layouts = {Layout(Dimensions({1, 2, 2}))};
  std::vector<Layout> out_layouts = {Layout(Dimensions({1, 2, 2}))};
  EXPECT_TRUE(kernel.GetOutputLayouts(in_layouts, out_layouts));

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(input_buffer));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(output_buffer));

  EXPECT_TRUE(kernel.Run(inputs, outputs));

  LITERT_ASSERT_OK_AND_ASSIGN(auto lock_and_addr,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[0], TensorBuffer::LockMode::kRead));
  auto output_span = absl::MakeSpan(lock_and_addr.second, 4);

  const std::vector<float> expected_data = {1.0f, 0.0f, 2.0f, 1.0f};
  EXPECT_THAT(output_span, Pointwise(FloatNear(1e-5f), expected_data));
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest, TrilInvKernelComputation4x4) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  TrilInvCustomOpKernel kernel;

  auto tensor_type = MakeRankedTensorType<float>({1, 1, 4, 4});
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  tensor_type, sizeof(float) * 16));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto output_buffer,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  tensor_type, sizeof(float) * 16));

  const std::vector<float> input_data = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f,
      0.2f, 0.3f, 0.0f, 0.0f, 0.1f, 0.4f, 0.6f, 0.0f,
  };
  ASSERT_TRUE(input_buffer.Write<float>(absl::MakeConstSpan(input_data)));

  std::vector<Layout> in_layouts = {Layout(Dimensions({1, 1, 4, 4}))};
  std::vector<Layout> out_layouts = {Layout(Dimensions({1, 1, 4, 4}))};
  EXPECT_TRUE(kernel.GetOutputLayouts(in_layouts, out_layouts));

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(input_buffer));
  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(output_buffer));

  EXPECT_TRUE(kernel.Run(inputs, outputs));

  LITERT_ASSERT_OK_AND_ASSIGN(auto lock_and_addr,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[0], TensorBuffer::LockMode::kRead));
  auto output_span = absl::MakeSpan(lock_and_addr.second, 16);

  // Reference forward substitution:
  // row 0: [1, 0, 0, 0]
  // row 1: [0.5, 1, 0, 0]
  // row 2: [0.2 + 0.3*0.5 = 0.35, 0.3, 1, 0]
  // row 3: [0.1 + 0.4*0.5 + 0.6*0.35 = 0.51, 0.4 + 0.6*0.3 = 0.58, 0.6, 1]
  const std::vector<float> expected_data = {
      1.0f,  0.0f, 0.0f, 0.0f, 0.5f,  1.0f,  0.0f, 0.0f,
      0.35f, 0.3f, 1.0f, 0.0f, 0.51f, 0.58f, 0.6f, 1.0f,
  };
  EXPECT_THAT(output_span, Pointwise(FloatNear(1e-5f), expected_data));
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest, SingleTokenDecodeExactGoldenMath) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  GatedDeltaUpdateCustomOpKernel kernel;

  // B=1, H=1, N=1, D_k=2, D_v=2
  auto q_type = MakeRankedTensorType<float>({1, 1, 1, 2});
  auto v_type = MakeRankedTensorType<float>({1, 1, 1, 2});
  auto beta_type = MakeRankedTensorType<float>({1, 1, 1});
  auto rec_type = MakeRankedTensorType<float>({1, 1, 2, 2});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, q_type,
                                  sizeof(float) * 2));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto k_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, q_type,
                                  sizeof(float) * 2));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto v_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, v_type,
                                  sizeof(float) * 2));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto beta_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, beta_type,
                                  sizeof(float) * 1));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto g_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, beta_type,
                                  sizeof(float) * 1));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto rec_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, rec_type,
                                  sizeof(float) * 4));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, v_type,
                                  sizeof(float) * 2));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto new_rec_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, rec_type,
                                  sizeof(float) * 4));

  // Hand-derived values:
  // S_prev = [1, 2; 3, 4]
  // g = 0 (decay = 1.0) -> S' = [1, 2; 3, 4]
  // k = [1, 0]^T
  // kv_mem = S'^T * k = [1, 2]^T
  // v = [3, 4]^T, beta = 0.5
  // delta = (v - kv_mem) * beta = ([3, 4] - [1, 2]) * 0.5 = [1, 1]^T
  // S_new = S' + k * delta^T = [1, 2; 3, 4] + [1, 1; 0, 0] = [2, 3; 3, 4]
  // q = [0, 1]^T
  // y = S_new^T * q = [3, 4]^T
  const std::vector<float> q_data = {0.0f, 1.0f};
  const std::vector<float> k_data = {1.0f, 0.0f};
  const std::vector<float> v_data = {3.0f, 4.0f};
  const std::vector<float> beta_data = {0.5f};
  const std::vector<float> g_data = {0.0f};
  const std::vector<float> rec_data = {1.0f, 2.0f, 3.0f, 4.0f};

  ASSERT_TRUE(q_buf.Write<float>(absl::MakeConstSpan(q_data)));
  ASSERT_TRUE(k_buf.Write<float>(absl::MakeConstSpan(k_data)));
  ASSERT_TRUE(v_buf.Write<float>(absl::MakeConstSpan(v_data)));
  ASSERT_TRUE(beta_buf.Write<float>(absl::MakeConstSpan(beta_data)));
  ASSERT_TRUE(g_buf.Write<float>(absl::MakeConstSpan(g_data)));
  ASSERT_TRUE(rec_buf.Write<float>(absl::MakeConstSpan(rec_data)));

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(q_buf));
  inputs.push_back(std::move(k_buf));
  inputs.push_back(std::move(v_buf));
  inputs.push_back(std::move(beta_buf));
  inputs.push_back(std::move(g_buf));
  inputs.push_back(std::move(rec_buf));

  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(out_buf));
  outputs.push_back(std::move(new_rec_buf));

  std::vector<Layout> in_layouts = {
      Layout(Dimensions({1, 1, 1, 2})), Layout(Dimensions({1, 1, 1, 2})),
      Layout(Dimensions({1, 1, 1, 2})), Layout(Dimensions({1, 1, 1})),
      Layout(Dimensions({1, 1, 1})),    Layout(Dimensions({1, 1, 2, 2}))};
  std::vector<Layout> out_layouts = {Layout(Dimensions({1, 1, 1, 2})),
                                     Layout(Dimensions({1, 1, 2, 2}))};
  EXPECT_TRUE(kernel.GetOutputLayouts(in_layouts, out_layouts));
  EXPECT_TRUE(kernel.Run(inputs, outputs));

  LITERT_ASSERT_OK_AND_ASSIGN(auto out_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[0], TensorBuffer::LockMode::kRead));
  LITERT_ASSERT_OK_AND_ASSIGN(auto state_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[1], TensorBuffer::LockMode::kRead));

  auto out_span = absl::MakeSpan(out_lock.second, 2);
  auto state_span = absl::MakeSpan(state_lock.second, 4);

  const std::vector<float> expected_out = {3.0f, 4.0f};
  const std::vector<float> expected_state = {2.0f, 3.0f, 3.0f, 4.0f};

  EXPECT_THAT(out_span, Pointwise(FloatNear(1e-5f), expected_out));
  EXPECT_THAT(state_span, Pointwise(FloatNear(1e-5f), expected_state));
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest,
     MultiBatchMultiHeadMultiStepRecurrent) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  GatedDeltaUpdateCustomOpKernel kernel;

  const int B = 2;
  const int H = 2;
  const int N = 8;
  const int D_k = 4;
  const int D_v = 4;

  auto q_type = MakeRankedTensorType<float>({B, H, N, D_k});
  auto v_type = MakeRankedTensorType<float>({B, H, N, D_v});
  auto beta_type = MakeRankedTensorType<float>({B, H, N});
  auto rec_type = MakeRankedTensorType<float>({B, H, D_k, D_v});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, q_type,
                                  sizeof(float) * B * H * N * D_k));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto k_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, q_type,
                                  sizeof(float) * B * H * N * D_k));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto v_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, v_type,
                                  sizeof(float) * B * H * N * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto beta_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, beta_type,
                                  sizeof(float) * B * H * N));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto g_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, beta_type,
                                  sizeof(float) * B * H * N));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto rec_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, rec_type,
                                  sizeof(float) * B * H * D_k * D_v));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, v_type,
                                  sizeof(float) * B * H * N * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto new_rec_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, rec_type,
                                  sizeof(float) * B * H * D_k * D_v));

  std::vector<float> q_data(B * H * N * D_k);
  std::vector<float> k_data(B * H * N * D_k);
  std::vector<float> v_data(B * H * N * D_v);
  std::vector<float> beta_data(B * H * N);
  std::vector<float> g_data(B * H * N);
  std::vector<float> rec_data(B * H * D_k * D_v);

  for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = std::sin(i * 0.1f);
  for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = std::cos(i * 0.2f);
  for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = std::sin(i * 0.3f);
  for (size_t i = 0; i < beta_data.size(); ++i)
    beta_data[i] = 0.5f + 0.4f * std::sin(i * 0.4f);
  for (size_t i = 0; i < g_data.size(); ++i)
    g_data[i] = -0.1f * std::abs(std::cos(i * 0.5f));
  for (size_t i = 0; i < rec_data.size(); ++i)
    rec_data[i] = 0.1f * std::sin(i * 0.6f);

  ASSERT_TRUE(q_buf.Write<float>(absl::MakeConstSpan(q_data)));
  ASSERT_TRUE(k_buf.Write<float>(absl::MakeConstSpan(k_data)));
  ASSERT_TRUE(v_buf.Write<float>(absl::MakeConstSpan(v_data)));
  ASSERT_TRUE(beta_buf.Write<float>(absl::MakeConstSpan(beta_data)));
  ASSERT_TRUE(g_buf.Write<float>(absl::MakeConstSpan(g_data)));
  ASSERT_TRUE(rec_buf.Write<float>(absl::MakeConstSpan(rec_data)));

  std::vector<float> golden_out;
  std::vector<float> golden_final_state;
  ComputeGoldenRecurrentGatedDelta(q_data, k_data, v_data, beta_data, g_data,
                                   rec_data, golden_out, golden_final_state, B,
                                   H, N, D_k, D_v);

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(q_buf));
  inputs.push_back(std::move(k_buf));
  inputs.push_back(std::move(v_buf));
  inputs.push_back(std::move(beta_buf));
  inputs.push_back(std::move(g_buf));
  inputs.push_back(std::move(rec_buf));

  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(out_buf));
  outputs.push_back(std::move(new_rec_buf));

  std::vector<Layout> in_layouts = {
      Layout(Dimensions({B, H, N, D_k})), Layout(Dimensions({B, H, N, D_k})),
      Layout(Dimensions({B, H, N, D_v})), Layout(Dimensions({B, H, N})),
      Layout(Dimensions({B, H, N})),      Layout(Dimensions({B, H, D_k, D_v}))};
  std::vector<Layout> out_layouts = {Layout(Dimensions({B, H, N, D_v})),
                                     Layout(Dimensions({B, H, D_k, D_v}))};
  EXPECT_TRUE(kernel.GetOutputLayouts(in_layouts, out_layouts));
  EXPECT_TRUE(kernel.Run(inputs, outputs));

  LITERT_ASSERT_OK_AND_ASSIGN(auto out_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[0], TensorBuffer::LockMode::kRead));
  LITERT_ASSERT_OK_AND_ASSIGN(auto state_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[1], TensorBuffer::LockMode::kRead));

  auto out_span = absl::MakeSpan(out_lock.second, golden_out.size());
  auto state_span =
      absl::MakeSpan(state_lock.second, golden_final_state.size());

  EXPECT_THAT(out_span, Pointwise(FloatNear(1e-4f), golden_out));
  EXPECT_THAT(state_span, Pointwise(FloatNear(1e-4f), golden_final_state));
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest,
     GqaRatio2MultiBatchMultiStepRecurrent) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  GatedDeltaUpdateCustomOpKernel kernel;

  const int B = 2;
  const int H = 4;    // value heads
  const int H_k = 2;  // key/query heads (GQA ratio = 2)
  const int N = 8;
  const int D_k = 4;
  const int D_v = 4;

  auto q_type = MakeRankedTensorType<float>({B, H_k, N, D_k});
  auto v_type = MakeRankedTensorType<float>({B, H, N, D_v});
  auto beta_type = MakeRankedTensorType<float>({B, H, N});
  auto rec_type = MakeRankedTensorType<float>({B, H, D_k, D_v});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, q_type,
                                  sizeof(float) * B * H_k * N * D_k));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto k_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, q_type,
                                  sizeof(float) * B * H_k * N * D_k));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto v_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, v_type,
                                  sizeof(float) * B * H * N * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto beta_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, beta_type,
                                  sizeof(float) * B * H * N));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto g_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, beta_type,
                                  sizeof(float) * B * H * N));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto rec_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, rec_type,
                                  sizeof(float) * B * H * D_k * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, v_type,
                                  sizeof(float) * B * H * N * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto new_rec_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, rec_type,
                                  sizeof(float) * B * H * D_k * D_v));

  std::vector<float> q_data(B * H_k * N * D_k);
  std::vector<float> k_data(B * H_k * N * D_k);
  std::vector<float> v_data(B * H * N * D_v);
  std::vector<float> beta_data(B * H * N);
  std::vector<float> g_data(B * H * N);
  std::vector<float> rec_data(B * H * D_k * D_v);

  for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = std::sin(i * 0.13f);
  for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = std::cos(i * 0.17f);
  for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = std::sin(i * 0.23f);
  for (size_t i = 0; i < beta_data.size(); ++i)
    beta_data[i] = 0.5f + 0.4f * std::sin(i * 0.31f);
  for (size_t i = 0; i < g_data.size(); ++i)
    g_data[i] = -0.1f * std::abs(std::cos(i * 0.41f));
  for (size_t i = 0; i < rec_data.size(); ++i)
    rec_data[i] = 0.1f * std::sin(i * 0.53f);

  ASSERT_TRUE(q_buf.Write<float>(absl::MakeConstSpan(q_data)));
  ASSERT_TRUE(k_buf.Write<float>(absl::MakeConstSpan(k_data)));
  ASSERT_TRUE(v_buf.Write<float>(absl::MakeConstSpan(v_data)));
  ASSERT_TRUE(beta_buf.Write<float>(absl::MakeConstSpan(beta_data)));
  ASSERT_TRUE(g_buf.Write<float>(absl::MakeConstSpan(g_data)));
  ASSERT_TRUE(rec_buf.Write<float>(absl::MakeConstSpan(rec_data)));

  std::vector<float> golden_out;
  std::vector<float> golden_final_state;
  ComputeGoldenRecurrentGatedDelta(q_data, k_data, v_data, beta_data, g_data,
                                   rec_data, golden_out, golden_final_state, B,
                                   H, N, D_k, D_v, H_k);

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(q_buf));
  inputs.push_back(std::move(k_buf));
  inputs.push_back(std::move(v_buf));
  inputs.push_back(std::move(beta_buf));
  inputs.push_back(std::move(g_buf));
  inputs.push_back(std::move(rec_buf));

  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(out_buf));
  outputs.push_back(std::move(new_rec_buf));

  std::vector<Layout> in_layouts = {Layout(Dimensions({B, H_k, N, D_k})),
                                    Layout(Dimensions({B, H_k, N, D_k})),
                                    Layout(Dimensions({B, H, N, D_v})),
                                    Layout(Dimensions({B, H, N})),
                                    Layout(Dimensions({B, H, N})),
                                    Layout(Dimensions({B, H, D_k, D_v}))};
  std::vector<Layout> out_layouts(2);
  EXPECT_TRUE(kernel.GetOutputLayouts(in_layouts, out_layouts));
  EXPECT_EQ(out_layouts[0].Dimensions(), Dimensions({B, H, N, D_v}));
  EXPECT_EQ(out_layouts[1].Dimensions(), Dimensions({B, H, D_k, D_v}));
  EXPECT_TRUE(kernel.Run(inputs, outputs));

  LITERT_ASSERT_OK_AND_ASSIGN(auto out_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[0], TensorBuffer::LockMode::kRead));
  LITERT_ASSERT_OK_AND_ASSIGN(auto state_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[1], TensorBuffer::LockMode::kRead));

  auto out_span = absl::MakeSpan(out_lock.second, golden_out.size());
  auto state_span =
      absl::MakeSpan(state_lock.second, golden_final_state.size());

  EXPECT_THAT(out_span, Pointwise(FloatNear(1e-4f), golden_out));
  EXPECT_THAT(state_span, Pointwise(FloatNear(1e-4f), golden_final_state));
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest,
     GqaRatio3RecurrentAndChunkedParityAndClamping) {
  // B=1, H=6, H_k=2 (GQA ratio = 3, matching Qwen 27B 48/16), N=72 (> 64 chunk)
  const int B = 1;
  const int H = 6;
  const int H_k = 2;
  const int N = 72;
  const int D_k = 4;
  const int D_v = 4;

  std::vector<float> q_data(B * H_k * N * D_k);
  std::vector<float> k_data(B * H_k * N * D_k);
  std::vector<float> v_data(B * H * N * D_v);
  std::vector<float> beta_data(B * H * N);
  std::vector<float> g_data(B * H * N);
  std::vector<float> rec_data(B * H * D_k * D_v);

  for (size_t i = 0; i < q_data.size(); ++i)
    q_data[i] = 0.2f * std::sin(i * 0.1f);
  for (size_t i = 0; i < k_data.size(); ++i)
    k_data[i] = 0.2f * std::cos(i * 0.2f);
  for (size_t i = 0; i < v_data.size(); ++i)
    v_data[i] = 0.2f * std::sin(i * 0.3f);
  for (size_t i = 0; i < beta_data.size(); ++i) {
    // Include some negative values to verify beta clamping to 0.0
    beta_data[i] = (i % 7 == 0) ? -0.25f : (0.3f + 0.2f * std::sin(i * 0.4f));
  }
  for (size_t i = 0; i < g_data.size(); ++i) {
    // Include some positive values to verify g clamping to 0.0 (decay = 1.0)
    g_data[i] = (i % 11 == 0) ? 0.5f : (-0.05f * std::abs(std::cos(i * 0.5f)));
  }
  for (size_t i = 0; i < rec_data.size(); ++i)
    rec_data[i] = 0.1f * std::sin(i * 0.6f);

  std::vector<float> golden_out;
  std::vector<float> golden_state;
  ComputeGoldenRecurrentGatedDelta(q_data, k_data, v_data, beta_data, g_data,
                                   rec_data, golden_out, golden_state, B, H, N,
                                   D_k, D_v, H_k);

  std::vector<float> rec_out(B * H * N * D_v);
  std::vector<float> rec_state(B * H * D_k * D_v);
  ComputeGatedDeltaUpdateRecurrent(
      q_data.data(), k_data.data(), v_data.data(), beta_data.data(),
      g_data.data(), rec_data.data(), rec_out.data(), rec_state.data(), B, H, N,
      D_k, D_v, H_k);

  std::vector<float> chunked_out(B * H * N * D_v);
  std::vector<float> chunked_state(B * H * D_k * D_v);
  ComputeGatedDeltaUpdateChunked(
      q_data.data(), k_data.data(), v_data.data(), beta_data.data(),
      g_data.data(), rec_data.data(), chunked_out.data(), chunked_state.data(),
      B, H, N, D_k, D_v, H_k);

  EXPECT_THAT(rec_out, Pointwise(FloatNear(1e-4f), golden_out));
  EXPECT_THAT(rec_state, Pointwise(FloatNear(1e-4f), golden_state));
  EXPECT_THAT(chunked_out, Pointwise(FloatNear(1e-4f), golden_out));
  EXPECT_THAT(chunked_state, Pointwise(FloatNear(1e-4f), golden_state));
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest, ValidLenSkipsRightPaddedTokens) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  GatedDeltaUpdateCustomOpKernel kernel;

  const int B = 1;
  const int H = 1;
  const int N = 4;
  const int D_k = 2;
  const int D_v = 2;
  const int valid_steps = 2;

  auto q_type = MakeRankedTensorType<float>({B, H, N, D_k});
  auto v_type = MakeRankedTensorType<float>({B, H, N, D_v});
  auto beta_type = MakeRankedTensorType<float>({B, H, N});
  auto rec_type = MakeRankedTensorType<float>({B, H, D_k, D_v});
  auto valid_len_type = MakeRankedTensorType<int32_t>({B});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, q_type,
                                  sizeof(float) * B * H * N * D_k));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto k_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, q_type,
                                  sizeof(float) * B * H * N * D_k));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto v_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, v_type,
                                  sizeof(float) * B * H * N * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto beta_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, beta_type,
                                  sizeof(float) * B * H * N));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto g_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, beta_type,
                                  sizeof(float) * B * H * N));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto rec_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, rec_type,
                                  sizeof(float) * B * H * D_k * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto valid_len_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  valid_len_type, sizeof(int32_t) * B));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, v_type,
                                  sizeof(float) * B * H * N * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto new_rec_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, rec_type,
                                  sizeof(float) * B * H * D_k * D_v));

  std::vector<float> q_data(B * H * N * D_k);
  std::vector<float> k_data(B * H * N * D_k);
  std::vector<float> v_data(B * H * N * D_v);
  std::vector<float> beta_data(B * H * N, 0.5f);
  std::vector<float> g_data(B * H * N, -0.1f);
  std::vector<float> rec_data = {1.0f, 0.5f, -0.5f, 1.0f};
  std::vector<int32_t> valid_len_data = {valid_steps};

  for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = 0.3f * (i + 1);
  for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = 0.2f * (i + 1);
  for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = 0.4f * (i + 1);

  ASSERT_TRUE(q_buf.Write<float>(absl::MakeConstSpan(q_data)));
  ASSERT_TRUE(k_buf.Write<float>(absl::MakeConstSpan(k_data)));
  ASSERT_TRUE(v_buf.Write<float>(absl::MakeConstSpan(v_data)));
  ASSERT_TRUE(beta_buf.Write<float>(absl::MakeConstSpan(beta_data)));
  ASSERT_TRUE(g_buf.Write<float>(absl::MakeConstSpan(g_data)));
  ASSERT_TRUE(rec_buf.Write<float>(absl::MakeConstSpan(rec_data)));
  ASSERT_TRUE(
      valid_len_buf.Write<int32_t>(absl::MakeConstSpan(valid_len_data)));

  // Golden over only the first `valid_steps` tokens
  std::vector<float> q_valid(q_data.begin(),
                             q_data.begin() + valid_steps * D_k);
  std::vector<float> k_valid(k_data.begin(),
                             k_data.begin() + valid_steps * D_k);
  std::vector<float> v_valid(v_data.begin(),
                             v_data.begin() + valid_steps * D_v);
  std::vector<float> beta_valid(beta_data.begin(),
                                beta_data.begin() + valid_steps);
  std::vector<float> g_valid(g_data.begin(), g_data.begin() + valid_steps);
  std::vector<float> golden_valid_out;
  std::vector<float> golden_final_state;
  ComputeGoldenRecurrentGatedDelta(
      q_valid, k_valid, v_valid, beta_valid, g_valid, rec_data,
      golden_valid_out, golden_final_state, B, H, valid_steps, D_k, D_v);

  std::vector<float> expected_out = golden_valid_out;
  expected_out.resize(B * H * N * D_v, 0.0f);  // right-padded tokens are zeroed

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(q_buf));
  inputs.push_back(std::move(k_buf));
  inputs.push_back(std::move(v_buf));
  inputs.push_back(std::move(beta_buf));
  inputs.push_back(std::move(g_buf));
  inputs.push_back(std::move(rec_buf));
  inputs.push_back(std::move(valid_len_buf));

  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(out_buf));
  outputs.push_back(std::move(new_rec_buf));

  std::vector<Layout> in_layouts = {Layout(Dimensions({B, H, N, D_k})),
                                    Layout(Dimensions({B, H, N, D_k})),
                                    Layout(Dimensions({B, H, N, D_v})),
                                    Layout(Dimensions({B, H, N})),
                                    Layout(Dimensions({B, H, N})),
                                    Layout(Dimensions({B, H, D_k, D_v})),
                                    Layout(Dimensions({B}))};
  std::vector<Layout> out_layouts(2);
  EXPECT_TRUE(kernel.GetOutputLayouts(in_layouts, out_layouts));
  EXPECT_TRUE(kernel.Run(inputs, outputs));

  LITERT_ASSERT_OK_AND_ASSIGN(auto out_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[0], TensorBuffer::LockMode::kRead));
  LITERT_ASSERT_OK_AND_ASSIGN(auto state_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[1], TensorBuffer::LockMode::kRead));

  auto out_span = absl::MakeSpan(out_lock.second, expected_out.size());
  auto state_span =
      absl::MakeSpan(state_lock.second, golden_final_state.size());

  EXPECT_THAT(out_span, Pointwise(FloatNear(1e-5f), expected_out));
  EXPECT_THAT(state_span, Pointwise(FloatNear(1e-5f), golden_final_state));
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest, ValidLenMismatchedBatchSizeRejected) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  GatedDeltaUpdateCustomOpKernel kernel;

  const int B = 2;
  const int H = 1;
  const int N = 4;
  const int D_k = 2;
  const int D_v = 2;

  auto q_type = MakeRankedTensorType<float>({B, H, N, D_k});
  auto v_type = MakeRankedTensorType<float>({B, H, N, D_v});
  auto beta_type = MakeRankedTensorType<float>({B, H, N});
  auto rec_type = MakeRankedTensorType<float>({B, H, D_k, D_v});
  auto valid_len_type = MakeRankedTensorType<int32_t>({1});

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto q_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, q_type,
                                  sizeof(float) * B * H * N * D_k));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto k_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, q_type,
                                  sizeof(float) * B * H * N * D_k));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto v_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, v_type,
                                  sizeof(float) * B * H * N * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto beta_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, beta_type,
                                  sizeof(float) * B * H * N));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto g_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, beta_type,
                                  sizeof(float) * B * H * N));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto rec_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, rec_type,
                                  sizeof(float) * B * H * D_k * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto valid_len_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  valid_len_type, sizeof(int32_t) * 1));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto out_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, v_type,
                                  sizeof(float) * B * H * N * D_v));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto new_rec_buf,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, rec_type,
                                  sizeof(float) * B * H * D_k * D_v));

  std::vector<TensorBuffer> inputs;
  inputs.push_back(std::move(q_buf));
  inputs.push_back(std::move(k_buf));
  inputs.push_back(std::move(v_buf));
  inputs.push_back(std::move(beta_buf));
  inputs.push_back(std::move(g_buf));
  inputs.push_back(std::move(rec_buf));
  inputs.push_back(std::move(valid_len_buf));

  std::vector<TensorBuffer> outputs;
  outputs.push_back(std::move(out_buf));
  outputs.push_back(std::move(new_rec_buf));

  EXPECT_FALSE(kernel.Run(inputs, outputs));
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest,
     RegisterGatedDeltaNetCustomOpsSucceeds) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  EXPECT_TRUE(RegisterGatedDeltaNetCustomOps(options));
}

std::vector<uint8_t> BuildBlockInitData(int H_k, int H_v, int D_k, int D_v,
                                        float eps = 1e-6f,
                                        float norm_eps = 1e-5f) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("mode", 0);
    fbb.Int("num_k_heads", H_k);
    fbb.Int("num_v_heads", H_v);
    fbb.Int("head_k_dim", D_k);
    fbb.Int("head_v_dim", D_v);
    fbb.Float("eps", eps);
    fbb.Float("norm_eps", norm_eps);
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

void ComputeGoldenGdnBlock(
    const std::vector<float>& conv_out, const std::vector<float>& zba,
    const std::vector<float>& a_log, const std::vector<float>& dt_bias,
    const std::vector<float>& norm_weight,
    const std::vector<float>& initial_state,
    const std::vector<float>* valid_mask, std::vector<float>& golden_out,
    std::vector<float>& golden_state, int B, int T, int H_k, int H_v, int D_k,
    int D_v, float eps, float norm_eps) {
  const int K_dim = H_k * D_k;
  const int V_dim = H_v * D_v;
  const int conv_ch = 2 * K_dim + V_dim;
  const int zba_ch = V_dim + 2 * H_v;
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(D_k));

  std::vector<float> q_t(B * H_k * T * D_k, 0.0f);
  std::vector<float> k_t(B * H_k * T * D_k, 0.0f);
  std::vector<float> v_t(B * H_v * T * D_v, 0.0f);
  std::vector<float> beta_t(B * H_v * T, 0.0f);
  std::vector<float> g_t(B * H_v * T, 0.0f);

  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      const float m = (valid_mask != nullptr) ? (*valid_mask)[b * T + t] : 1.0f;
      const float* conv_bt = conv_out.data() + (b * T + t) * conv_ch;
      const float* zba_bt = zba.data() + (b * T + t) * zba_ch;
      for (int hk = 0; hk < H_k; ++hk) {
        float q_ss = 0.0f;
        float k_ss = 0.0f;
        for (int d = 0; d < D_k; ++d) {
          float qv = conv_bt[hk * D_k + d] * m;
          float kv = conv_bt[K_dim + hk * D_k + d];
          q_ss += qv * qv;
          k_ss += kv * kv;
        }
        float q_inv = q_scale / std::sqrt(q_ss + eps);
        float k_inv = 1.0f / std::sqrt(k_ss + eps);
        for (int d = 0; d < D_k; ++d) {
          q_t[((b * H_k + hk) * T + t) * D_k + d] =
              (m == 0.0f) ? 0.0f : (conv_bt[hk * D_k + d] * m * q_inv);
          k_t[((b * H_k + hk) * T + t) * D_k + d] =
              conv_bt[K_dim + hk * D_k + d] * k_inv;
        }
      }
      for (int h = 0; h < H_v; ++h) {
        for (int d = 0; d < D_v; ++d) {
          v_t[((b * H_v + h) * T + t) * D_v + d] =
              conv_bt[2 * K_dim + h * D_v + d];
        }
        const float b_raw = zba_bt[V_dim + h];
        const float a_raw = zba_bt[V_dim + H_v + h];
        const float a_in = a_raw + dt_bias[h];
        const float sp =
            std::max(a_in, 0.0f) + std::log1pf(std::exp(-std::abs(a_in)));
        beta_t[(b * H_v + h) * T + t] =
            (m == 0.0f) ? 0.0f : (m / (1.0f + std::exp(-b_raw)));
        g_t[(b * H_v + h) * T + t] =
            (m == 0.0f) ? 0.0f : (-std::exp(a_log[h]) * sp * m);
      }
    }
  }

  std::vector<float> core_out;
  ComputeGoldenRecurrentGatedDelta(q_t, k_t, v_t, beta_t, g_t, initial_state,
                                   core_out, golden_state, B, H_v, T, D_k, D_v,
                                   H_k);

  golden_out.assign(B * T * V_dim, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      const float m = (valid_mask != nullptr) ? (*valid_mask)[b * T + t] : 1.0f;
      if (m == 0.0f) continue;
      const float* zba_bt = zba.data() + (b * T + t) * zba_ch;
      for (int h = 0; h < H_v; ++h) {
        const float* y = core_out.data() + ((b * H_v + h) * T + t) * D_v;
        float ss = 0.0f;
        for (int d = 0; d < D_v; ++d) ss += y[d] * y[d];
        float inv_rms =
            1.0f / std::sqrt(ss / static_cast<float>(D_v) + norm_eps);
        for (int d = 0; d < D_v; ++d) {
          float z = zba_bt[h * D_v + d];
          float silu_z = z / (1.0f + std::exp(-z));
          golden_out[(b * T + t) * V_dim + h * D_v + d] =
              y[d] * inv_rms * norm_weight[d] * silu_z;
        }
      }
    }
  }
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest, BlockModeGqaPrefillAndPaddingParity) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));
  GatedDeltaUpdateCustomOpKernel kernel;

  const int B = 1, T = 5, H_k = 2, H_v = 4, D_k = 8, D_v = 8;
  const int K_dim = H_k * D_k;
  const int V_dim = H_v * D_v;
  const float eps = 1e-6f;
  const float norm_eps = 1e-5f;
  auto init_buf = BuildBlockInitData(H_k, H_v, D_k, D_v, eps, norm_eps);
  ASSERT_TRUE(kernel.Init(init_buf.data(), init_buf.size()));

  std::vector<float> conv_data(B * T * (2 * K_dim + V_dim));
  std::vector<float> zba_data(B * T * (V_dim + 2 * H_v));
  std::vector<float> a_log_data = {-1.2f, -0.7f, -0.3f, -2.0f};
  std::vector<float> dt_bias_data = {0.1f, -0.2f, 0.3f, 0.05f};
  std::vector<float> norm_w_data(D_v);
  std::vector<float> rec_data(B * H_v * D_k * D_v);
  std::vector<float> mask_data = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f};

  for (size_t i = 0; i < conv_data.size(); ++i)
    conv_data[i] = 0.4f * std::sin(i * 0.17f + 0.1f);
  for (size_t i = 0; i < zba_data.size(); ++i)
    zba_data[i] = 0.5f * std::cos(i * 0.23f - 0.2f);
  for (size_t i = 0; i < norm_w_data.size(); ++i)
    norm_w_data[i] = 0.8f + 0.1f * static_cast<float>(i);
  for (size_t i = 0; i < rec_data.size(); ++i)
    rec_data[i] = 0.1f * std::sin(i * 0.31f);

  std::vector<float> golden_out;
  std::vector<float> golden_state;
  ComputeGoldenGdnBlock(conv_data, zba_data, a_log_data, dt_bias_data,
                        norm_w_data, rec_data, &mask_data, golden_out,
                        golden_state, B, T, H_k, H_v, D_k, D_v, eps, norm_eps);

  auto make_f32_buf = [&](Dimensions dims, const std::vector<float>& vals) {
    auto ttype = MakeRankedTensorType<float>(dims);
    auto buf = TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                           ttype, sizeof(float) * vals.size());
    if (!vals.empty()) {
      (void)buf->Write<float>(absl::MakeConstSpan(vals));
    }
    return std::move(*buf);
  };

  std::vector<TensorBuffer> inputs;
  inputs.push_back(make_f32_buf({B, T, 2 * K_dim + V_dim}, conv_data));
  inputs.push_back(make_f32_buf({B, T, V_dim + 2 * H_v}, zba_data));
  inputs.push_back(make_f32_buf({H_v}, a_log_data));
  inputs.push_back(make_f32_buf({H_v}, dt_bias_data));
  inputs.push_back(make_f32_buf({D_v}, norm_w_data));
  inputs.push_back(make_f32_buf({B, H_v, D_k, D_v}, rec_data));
  inputs.push_back(make_f32_buf({B, T}, mask_data));

  std::vector<float> zeros_out(B * T * V_dim, 0.0f);
  std::vector<float> zeros_state(B * H_v * D_k * D_v, 0.0f);
  std::vector<TensorBuffer> outputs;
  outputs.push_back(make_f32_buf({B, T, V_dim}, zeros_out));
  outputs.push_back(make_f32_buf({B, H_v, D_k, D_v}, zeros_state));

  std::vector<Layout> in_layouts = {
      Layout(Dimensions({B, T, 2 * K_dim + V_dim})),
      Layout(Dimensions({B, T, V_dim + 2 * H_v})),
      Layout(Dimensions({H_v})),
      Layout(Dimensions({H_v})),
      Layout(Dimensions({D_v})),
      Layout(Dimensions({B, H_v, D_k, D_v})),
      Layout(Dimensions({B, T}))};
  std::vector<Layout> out_layouts(2);
  ASSERT_TRUE(kernel.GetOutputLayouts(in_layouts, out_layouts));
  EXPECT_EQ(out_layouts[0].Dimensions(), Dimensions({B, T, V_dim}));
  EXPECT_EQ(out_layouts[1].Dimensions(), Dimensions({B, H_v, D_k, D_v}));

  ASSERT_TRUE(kernel.Run(inputs, outputs));

  LITERT_ASSERT_OK_AND_ASSIGN(auto out_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[0], TensorBuffer::LockMode::kRead));
  LITERT_ASSERT_OK_AND_ASSIGN(auto state_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[1], TensorBuffer::LockMode::kRead));
  EXPECT_THAT(absl::MakeSpan(out_lock.second, golden_out.size()),
              Pointwise(FloatNear(1e-5f), golden_out));
  EXPECT_THAT(absl::MakeSpan(state_lock.second, golden_state.size()),
              Pointwise(FloatNear(1e-5f), golden_state));
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest, BlockModeMaskedDecodePreservesState) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));
  GatedDeltaUpdateCustomOpKernel kernel;

  const int B = 1, T = 1, H_k = 2, H_v = 2, D_k = 4, D_v = 4;
  const int K_dim = H_k * D_k;
  const int V_dim = H_v * D_v;
  auto init_buf = BuildBlockInitData(H_k, H_v, D_k, D_v);
  ASSERT_TRUE(kernel.Init(init_buf.data(), init_buf.size()));

  std::vector<float> conv_data(B * T * (2 * K_dim + V_dim), 0.7f);
  std::vector<float> zba_data(B * T * (V_dim + 2 * H_v), 0.5f);
  std::vector<float> a_log_data(H_v, -0.5f);
  std::vector<float> dt_bias_data(H_v, 0.1f);
  std::vector<float> norm_w_data(D_v, 1.0f);
  std::vector<float> rec_data(B * H_v * D_k * D_v);
  for (size_t i = 0; i < rec_data.size(); ++i) rec_data[i] = 0.2f * (i + 1);
  std::vector<float> mask_data = {0.0f};

  auto make_f32_buf = [&](Dimensions dims, const std::vector<float>& vals) {
    auto ttype = MakeRankedTensorType<float>(dims);
    auto buf = TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                           ttype, sizeof(float) * vals.size());
    (void)buf->Write<float>(absl::MakeConstSpan(vals));
    return std::move(*buf);
  };

  std::vector<TensorBuffer> inputs;
  inputs.push_back(make_f32_buf({B, T, 2 * K_dim + V_dim}, conv_data));
  inputs.push_back(make_f32_buf({B, T, V_dim + 2 * H_v}, zba_data));
  inputs.push_back(make_f32_buf({H_v}, a_log_data));
  inputs.push_back(make_f32_buf({H_v}, dt_bias_data));
  inputs.push_back(make_f32_buf({D_v}, norm_w_data));
  inputs.push_back(make_f32_buf({B, H_v, D_k, D_v}, rec_data));
  inputs.push_back(make_f32_buf({B, T}, mask_data));

  std::vector<float> init_out(B * T * V_dim, 99.0f);
  std::vector<float> init_state(B * H_v * D_k * D_v, 0.0f);
  std::vector<TensorBuffer> outputs;
  outputs.push_back(make_f32_buf({B, T, V_dim}, init_out));
  outputs.push_back(make_f32_buf({B, H_v, D_k, D_v}, init_state));

  ASSERT_TRUE(kernel.Run(inputs, outputs));

  LITERT_ASSERT_OK_AND_ASSIGN(auto out_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[0], TensorBuffer::LockMode::kRead));
  LITERT_ASSERT_OK_AND_ASSIGN(auto state_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[1], TensorBuffer::LockMode::kRead));
  std::vector<float> expected_zeros(B * T * V_dim, 0.0f);
  EXPECT_THAT(absl::MakeSpan(out_lock.second, expected_zeros.size()),
              Pointwise(FloatNear(1e-6f), expected_zeros));
  EXPECT_THAT(absl::MakeSpan(state_lock.second, rec_data.size()),
              Pointwise(FloatNear(1e-6f), rec_data));
}

TEST(GatedDeltaUpdateLiteRtCustomOpTest, BlockModeFusedQkvzbaParity) {
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));
  GatedDeltaUpdateCustomOpKernel kernel;

  const int B = 1, T = 5, H_k = 2, H_v = 4, D_k = 8, D_v = 8;
  const int K_dim = H_k * D_k;
  const int V_dim = H_v * D_v;
  const int conv_ch = 2 * K_dim + V_dim;
  const int zba_ch = V_dim + 2 * H_v;
  const int fused_ch = conv_ch + zba_ch;
  const float eps = 1e-6f;
  const float norm_eps = 1e-5f;
  auto init_buf = BuildBlockInitData(H_k, H_v, D_k, D_v, eps, norm_eps);
  ASSERT_TRUE(kernel.Init(init_buf.data(), init_buf.size()));

  std::vector<float> conv_data(B * T * conv_ch);
  std::vector<float> zba_data(B * T * zba_ch);
  std::vector<float> fused_qkvzba_data(B * T * fused_ch, 77.0f);
  std::vector<float> a_log_data = {-1.2f, -0.7f, -0.3f, -2.0f};
  std::vector<float> dt_bias_data = {0.1f, -0.2f, 0.3f, 0.05f};
  std::vector<float> norm_w_data(D_v);
  std::vector<float> rec_data(B * H_v * D_k * D_v);
  std::vector<float> mask_data = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f};

  for (size_t i = 0; i < conv_data.size(); ++i)
    conv_data[i] = 0.4f * std::sin(i * 0.17f + 0.1f);
  for (size_t i = 0; i < zba_data.size(); ++i)
    zba_data[i] = 0.5f * std::cos(i * 0.23f - 0.2f);
  for (int bt = 0; bt < B * T; ++bt) {
    for (int c = 0; c < zba_ch; ++c) {
      fused_qkvzba_data[bt * fused_ch + conv_ch + c] =
          zba_data[bt * zba_ch + c];
    }
  }
  for (size_t i = 0; i < norm_w_data.size(); ++i)
    norm_w_data[i] = 0.8f + 0.1f * static_cast<float>(i);
  for (size_t i = 0; i < rec_data.size(); ++i)
    rec_data[i] = 0.1f * std::sin(i * 0.31f);

  std::vector<float> golden_out;
  std::vector<float> golden_state;
  ComputeGoldenGdnBlock(conv_data, zba_data, a_log_data, dt_bias_data,
                        norm_w_data, rec_data, &mask_data, golden_out,
                        golden_state, B, T, H_k, H_v, D_k, D_v, eps, norm_eps);

  auto make_f32_buf = [&](Dimensions dims, const std::vector<float>& vals) {
    auto ttype = MakeRankedTensorType<float>(dims);
    auto buf = TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                           ttype, sizeof(float) * vals.size());
    if (!vals.empty()) {
      (void)buf->Write<float>(absl::MakeConstSpan(vals));
    }
    return std::move(*buf);
  };

  std::vector<TensorBuffer> inputs;
  inputs.push_back(make_f32_buf({B, T, conv_ch}, conv_data));
  inputs.push_back(make_f32_buf({B, T, fused_ch}, fused_qkvzba_data));
  inputs.push_back(make_f32_buf({H_v}, a_log_data));
  inputs.push_back(make_f32_buf({H_v}, dt_bias_data));
  inputs.push_back(make_f32_buf({D_v}, norm_w_data));
  inputs.push_back(make_f32_buf({B, H_v, D_k, D_v}, rec_data));
  inputs.push_back(make_f32_buf({B, T}, mask_data));

  std::vector<float> zeros_out(B * T * V_dim, 0.0f);
  std::vector<float> zeros_state(B * H_v * D_k * D_v, 0.0f);
  std::vector<TensorBuffer> outputs;
  outputs.push_back(make_f32_buf({B, T, V_dim}, zeros_out));
  outputs.push_back(make_f32_buf({B, H_v, D_k, D_v}, zeros_state));

  std::vector<Layout> in_layouts = {Layout(Dimensions({B, T, conv_ch})),
                                    Layout(Dimensions({B, T, fused_ch})),
                                    Layout(Dimensions({H_v})),
                                    Layout(Dimensions({H_v})),
                                    Layout(Dimensions({D_v})),
                                    Layout(Dimensions({B, H_v, D_k, D_v})),
                                    Layout(Dimensions({B, T}))};
  std::vector<Layout> out_layouts(2);
  ASSERT_TRUE(kernel.GetOutputLayouts(in_layouts, out_layouts));
  EXPECT_EQ(out_layouts[0].Dimensions(), Dimensions({B, T, V_dim}));
  EXPECT_EQ(out_layouts[1].Dimensions(), Dimensions({B, H_v, D_k, D_v}));

  ASSERT_TRUE(kernel.Run(inputs, outputs));

  LITERT_ASSERT_OK_AND_ASSIGN(auto out_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[0], TensorBuffer::LockMode::kRead));
  LITERT_ASSERT_OK_AND_ASSIGN(auto state_lock,
                              TensorBufferScopedLock::Create<const float>(
                                  outputs[1], TensorBuffer::LockMode::kRead));
  EXPECT_THAT(absl::MakeSpan(out_lock.second, golden_out.size()),
              Pointwise(FloatNear(1e-5f), golden_out));
  EXPECT_THAT(absl::MakeSpan(state_lock.second, golden_state.size()),
              Pointwise(FloatNear(1e-5f), golden_state));
}

}  // namespace
}  // namespace gated_delta_net
}  // namespace litert
