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

#include "litert/vendors/nvidia/trtllm/gated_delta.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"

namespace {

struct CudaDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};
using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation Upload(const std::vector<float>& values) {
  void* pointer = nullptr;
  EXPECT_EQ(cudaMalloc(&pointer, values.size() * sizeof(float)), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(pointer, values.data(), values.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  return CudaAllocation(pointer);
}

std::vector<float> Download(const void* pointer, size_t count) {
  std::vector<float> values(count);
  EXPECT_EQ(cudaMemcpy(values.data(), pointer, count * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return values;
}

// Deterministic values in [lo, hi).
std::vector<float> Fill(size_t count, uint32_t seed, float lo, float hi) {
  std::vector<float> values(count);
  uint32_t x = seed * 2654435761u + 1;
  for (auto& value : values) {
    x = x * 1664525u + 1013904223u;
    value = lo + (hi - lo) * static_cast<float>(x >> 8) / 16777216.0f;
  }
  return values;
}

// ComputeGatedDeltaUpdateRecurrent of
// litert/experimental/custom_ops/gated_delta_net, in double precision.
void ReferenceGatedDelta(
    const LiteRtNvidiaGatedDeltaParams& p, const std::vector<float>& q,
    const std::vector<float>& k, const std::vector<float>& v,
    const std::vector<float>& beta, const std::vector<float>& g,
    const std::vector<float>& state, std::optional<int32_t> valid_len,
    std::vector<float>& out, std::vector<float>& new_state) {
  const int n = p.tokens, dk = p.key_dim, dv = p.value_dim;
  const int ratio = p.heads / p.key_heads;
  const int active = valid_len.has_value() ? std::clamp(*valid_len, 0, n) : n;
  out.assign(static_cast<size_t>(p.heads) * n * dv, 0.0f);
  new_state.assign(state.size(), 0.0f);
  std::vector<double> s(static_cast<size_t>(dk) * dv);
  std::vector<double> memory(dv);
  for (int h = 0; h < p.heads; ++h) {
    const int hk = h / ratio;
    for (size_t i = 0; i < s.size(); ++i) {
      s[i] = state[static_cast<size_t>(h) * dk * dv + i];
    }
    for (int t = 0; t < active; ++t) {
      const float g_t = g[static_cast<size_t>(h) * n + t];
      const float b_t = beta[static_cast<size_t>(h) * n + t];
      const double decay = g_t <= 0.0f ? std::exp(g_t) : 1.0;
      const double rate = b_t > 0.0f ? b_t : 0.0;
      const float* k_t = &k[(static_cast<size_t>(hk) * n + t) * dk];
      const float* q_t = &q[(static_cast<size_t>(hk) * n + t) * dk];
      const float* v_t = &v[(static_cast<size_t>(h) * n + t) * dv];
      for (auto& value : s) value *= decay;
      for (int j = 0; j < dv; ++j) {
        double acc = 0.0;
        for (int i = 0; i < dk; ++i) acc += s[i * dv + j] * k_t[i];
        memory[j] = acc;
      }
      for (int j = 0; j < dv; ++j) {
        const double delta = (v_t[j] - memory[j]) * rate;
        for (int i = 0; i < dk; ++i) s[i * dv + j] += k_t[i] * delta;
      }
      for (int j = 0; j < dv; ++j) {
        double acc = 0.0;
        for (int i = 0; i < dk; ++i) acc += s[i * dv + j] * q_t[i];
        out[(static_cast<size_t>(h) * n + t) * dv + j] =
            static_cast<float>(acc);
      }
    }
    for (size_t i = 0; i < s.size(); ++i) {
      new_state[static_cast<size_t>(h) * dk * dv + i] =
          static_cast<float>(s[i]);
    }
  }
}

void ExpectNear(const std::vector<float>& got, const std::vector<float>& want,
                float tolerance, const char* what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  for (size_t i = 0; i < got.size(); ++i) {
    ASSERT_NEAR(got[i], want[i], tolerance * (1.0f + std::fabs(want[i])))
        << what << " index " << i;
  }
}

void RunGatedDelta(LiteRtNvidiaGatedDeltaParams p,
                   std::optional<int32_t> valid_len, bool in_place) {
  SCOPED_TRACE(::testing::Message()
               << "heads=" << p.heads << " key_heads=" << p.key_heads
               << " tokens=" << p.tokens << " dk=" << p.key_dim << " dv="
               << p.value_dim << " valid=" << (valid_len ? *valid_len : -1)
               << " in_place=" << in_place);
  ASSERT_TRUE(LiteRtNvidiaGatedDeltaSupports(&p));
  const size_t n = p.tokens;
  // L2-normalized-scale keys and queries, as the model feeds them.
  const float key_scale = 1.0f / std::sqrt(static_cast<float>(p.key_dim));
  auto q = Fill(p.key_heads * n * p.key_dim, 1, -key_scale, key_scale);
  auto k = Fill(p.key_heads * n * p.key_dim, 2, -key_scale, key_scale);
  auto v = Fill(p.heads * n * p.value_dim, 3, -1.0f, 1.0f);
  auto beta = Fill(p.heads * n, 4, 0.05f, 1.0f);
  auto g = Fill(p.heads * n, 5, -2.0f, 0.0f);
  auto state = Fill(static_cast<size_t>(p.heads) * p.key_dim * p.value_dim, 6,
                    -0.5f, 0.5f);
  // Right-padding sentinels on a few rows of the first head.
  if (n > 3) {
    g[1] = 0.5f;
    beta[2] = -1.0f;
  }
  std::vector<float> want_out, want_state;
  ReferenceGatedDelta(p, q, k, v, beta, g, state, valid_len, want_out,
                      want_state);

  auto d_q = Upload(q), d_k = Upload(k), d_v = Upload(v);
  auto d_beta = Upload(beta), d_g = Upload(g), d_state = Upload(state);
  auto d_out = Upload(std::vector<float>(want_out.size(), 7.0f));
  CudaAllocation d_new =
      in_place ? CudaAllocation() : Upload(std::vector<float>(state.size()));
  CudaAllocation d_valid;
  if (valid_len.has_value()) {
    void* pointer = nullptr;
    ASSERT_EQ(cudaMalloc(&pointer, sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(pointer, &*valid_len, sizeof(int32_t),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    d_valid.reset(pointer);
  }
  float* new_state =
      static_cast<float*>(in_place ? d_state.get() : d_new.get());
  ASSERT_EQ(LiteRtNvidiaLaunchGatedDeltaUpdate(
                &p, static_cast<const float*>(d_q.get()),
                static_cast<const float*>(d_k.get()),
                static_cast<const float*>(d_v.get()),
                static_cast<const float*>(d_beta.get()),
                static_cast<const float*>(d_g.get()),
                static_cast<const float*>(d_state.get()),
                static_cast<const int32_t*>(d_valid.get()),
                static_cast<float*>(d_out.get()), new_state, nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ExpectNear(Download(d_out.get(), want_out.size()), want_out, 2e-4f, "out");
  ExpectNear(Download(new_state, want_state.size()), want_state, 2e-4f,
             "state");
}

TEST(GatedDeltaTest, DecodeMatchesReference) {
  // Qwen 3.5 27B: 48 value heads over 16 key heads of 128 x 128.
  RunGatedDelta({48, 16, 1, 128, 128}, std::nullopt, /*in_place=*/false);
  RunGatedDelta({48, 16, 1, 128, 128}, std::nullopt, /*in_place=*/true);
}

TEST(GatedDeltaTest, PrefillMatchesReference) {
  RunGatedDelta({6, 2, 37, 128, 64}, std::nullopt, /*in_place=*/false);
  RunGatedDelta({6, 2, 128, 128, 128}, std::nullopt, /*in_place=*/true);
  RunGatedDelta({4, 4, 9, 64, 32}, std::nullopt, /*in_place=*/false);
  RunGatedDelta({2, 1, 5, 256, 32}, std::nullopt, /*in_place=*/false);
}

TEST(GatedDeltaTest, ValidLengthMasksPaddedRows) {
  RunGatedDelta({6, 2, 40, 128, 64}, 17, /*in_place=*/true);
  RunGatedDelta({6, 2, 40, 128, 64}, 0, /*in_place=*/false);
  RunGatedDelta({6, 2, 40, 128, 64}, 40, /*in_place=*/false);
  RunGatedDelta({6, 2, 40, 128, 64}, 1000, /*in_place=*/false);
}

TEST(GatedDeltaTest, RejectsUnsupportedShapes) {
  LiteRtNvidiaGatedDeltaParams p = {48, 16, 1, 128, 128};
  EXPECT_TRUE(LiteRtNvidiaGatedDeltaSupports(&p));
  p.heads = 40;  // not a multiple of the key heads
  EXPECT_FALSE(LiteRtNvidiaGatedDeltaSupports(&p));
  p = {48, 16, 1, 96, 128};
  EXPECT_FALSE(LiteRtNvidiaGatedDeltaSupports(&p));
  p = {48, 16, 1, 128, 48};
  EXPECT_FALSE(LiteRtNvidiaGatedDeltaSupports(&p));
  EXPECT_FALSE(LiteRtNvidiaGatedDeltaSupports(nullptr));
}

// The odml.short_conv_step decomposition.
void ReferenceShortConv(const LiteRtNvidiaShortConvParams& p,
                        const std::vector<float>& x,
                        const std::vector<float>& state,
                        const std::vector<float>& weight,
                        std::optional<int32_t> valid_len, std::vector<float>& y,
                        std::vector<float>& new_state) {
  const int c_count = p.channels, n = p.tokens, history = p.taps - 1;
  auto window = [&](int c, int position) -> double {
    return position < history
               ? state[static_cast<size_t>(c) * history + position]
               : x[static_cast<size_t>(position - history) * c_count + c];
  };
  y.assign(static_cast<size_t>(n) * c_count, 0.0f);
  for (int t = 0; t < n; ++t) {
    for (int c = 0; c < c_count; ++c) {
      double acc = 0.0;
      for (int j = 0; j < p.taps; ++j) {
        acc += weight[static_cast<size_t>(c) * p.taps + j] * window(c, t + j);
      }
      if (p.silu) acc = acc / (1.0 + std::exp(-acc));
      y[static_cast<size_t>(t) * c_count + c] = static_cast<float>(acc);
    }
  }
  const int valid = valid_len.has_value() ? std::clamp(*valid_len, 0, n) : n;
  new_state.assign(state.size(), 0.0f);
  for (int c = 0; c < c_count; ++c) {
    for (int j = 0; j < history; ++j) {
      new_state[static_cast<size_t>(c) * history + j] =
          static_cast<float>(window(c, valid + j));
    }
  }
}

void RunShortConv(LiteRtNvidiaShortConvParams p,
                  std::optional<int32_t> valid_len, bool in_place) {
  SCOPED_TRACE(::testing::Message()
               << "channels=" << p.channels << " tokens=" << p.tokens
               << " taps=" << p.taps << " silu=" << p.silu << " valid="
               << (valid_len ? *valid_len : -1) << " in_place=" << in_place);
  ASSERT_TRUE(LiteRtNvidiaShortConvSupports(&p));
  auto x = Fill(static_cast<size_t>(p.tokens) * p.channels, 11, -2.0f, 2.0f);
  auto state =
      Fill(static_cast<size_t>(p.channels) * (p.taps - 1), 12, -2.0f, 2.0f);
  auto weight = Fill(static_cast<size_t>(p.channels) * p.taps, 13, -1.0f, 1.0f);
  std::vector<float> want_y, want_state;
  ReferenceShortConv(p, x, state, weight, valid_len, want_y, want_state);
  auto d_x = Upload(x), d_state = Upload(state), d_w = Upload(weight);
  auto d_y = Upload(std::vector<float>(want_y.size(), 7.0f));
  CudaAllocation d_new =
      in_place ? CudaAllocation() : Upload(std::vector<float>(state.size()));
  CudaAllocation d_valid;
  if (valid_len.has_value()) {
    void* pointer = nullptr;
    ASSERT_EQ(cudaMalloc(&pointer, sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(pointer, &*valid_len, sizeof(int32_t),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    d_valid.reset(pointer);
  }
  float* new_state =
      static_cast<float*>(in_place ? d_state.get() : d_new.get());
  ASSERT_EQ(LiteRtNvidiaLaunchShortConv(
                &p, static_cast<const float*>(d_x.get()),
                static_cast<const float*>(d_state.get()),
                static_cast<const float*>(d_w.get()),
                static_cast<const int32_t*>(d_valid.get()),
                static_cast<float*>(d_y.get()), new_state, nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ExpectNear(Download(d_y.get(), want_y.size()), want_y, 1e-5f, "y");
  ExpectNear(Download(new_state, want_state.size()), want_state, 0.0f, "state");
}

TEST(ShortConvTest, MatchesReference) {
  // Qwen 3.5 27B: 10240 channels, 4 taps, SiLU; decode and a prefill chunk.
  RunShortConv({10240, 1, 4, 1}, std::nullopt, /*in_place=*/true);
  RunShortConv({10240, 128, 4, 1}, 128, /*in_place=*/true);
  RunShortConv({300, 33, 4, 0}, std::nullopt, /*in_place=*/false);
  RunShortConv({300, 33, 3, 1}, 10, /*in_place=*/true);
  // Fewer valid rows than the history keeps part of the old state.
  RunShortConv({300, 33, 4, 1}, 2, /*in_place=*/true);
  RunShortConv({300, 33, 4, 1}, 0, /*in_place=*/false);
}

}  // namespace
