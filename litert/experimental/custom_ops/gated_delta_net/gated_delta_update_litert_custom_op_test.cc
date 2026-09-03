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

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
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
    int B, int H, int N, int D_k, int D_v) {
  golden_out.resize(B * H * N * D_v);
  golden_final_state = initial_state;

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const int bh = b * H + h;
      float* S = golden_final_state.data() + bh * D_k * D_v;

      for (int t = 0; t < N; ++t) {
        const float* q_t = q.data() + (bh * N + t) * D_k;
        const float* k_t = k.data() + (bh * N + t) * D_k;
        const float* v_t = v.data() + (bh * N + t) * D_v;
        const float beta_t = beta[bh * N + t];
        const float g_decay = std::exp(g[bh * N + t]);
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
     RegisterGatedDeltaNetCustomOpsSucceeds) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, Options::Create());
  EXPECT_TRUE(RegisterGatedDeltaNetCustomOps(options));
}

}  // namespace
}  // namespace gated_delta_net
}  // namespace litert
