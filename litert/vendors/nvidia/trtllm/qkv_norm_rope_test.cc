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

#include "litert/vendors/nvidia/trtllm/qkv_norm_rope.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"

namespace {

struct CudaDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};
using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation Upload(const void* data, size_t bytes) {
  void* pointer = nullptr;
  EXPECT_EQ(cudaMalloc(&pointer, bytes), cudaSuccess);
  if (data != nullptr) {
    EXPECT_EQ(cudaMemcpy(pointer, data, bytes, cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return CudaAllocation(pointer);
}

uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// The composite's decomposition: per-head RMS norm (scale for q and k, none
// for v), then the half-rotation RoPE over the first `rotary` dims.
void Reference(const LiteRtNvidiaQkvNormRopeParams& p,
               const std::vector<uint16_t>& qkv,
               const std::vector<int32_t>& positions,
               const std::vector<float>& q_scale,
               const std::vector<float>& k_scale, std::vector<float>& q,
               std::vector<float>& k, std::vector<float>& v) {
  const int total = p.heads + 2 * p.kv_heads;
  std::vector<double> head(p.depth);
  for (int t = 0; t < p.tokens; ++t) {
    for (int h = 0; h < total; ++h) {
      double sum = 0.0;
      for (int i = 0; i < p.depth; ++i) {
        head[i] = Bf16BitsToFloat(
            qkv[(static_cast<size_t>(t) * total + h) * p.depth + i]);
        sum += head[i] * head[i];
      }
      const bool is_q = h < p.heads;
      const bool is_k = !is_q && h < p.heads + p.kv_heads;
      if (is_q || is_k || p.has_v_norm) {
        const double inv = 1.0 / std::sqrt(sum / p.depth + p.epsilon);
        for (int i = 0; i < p.depth; ++i) {
          head[i] *= inv * (is_q ? q_scale[i] : (is_k ? k_scale[i] : 1.0));
        }
      }
      if (is_q || is_k) {
        const int half = p.rotary / 2;
        const double log_step =
            std::log(p.max_timescale / p.min_timescale) / half;
        for (int i = 0; i < half; ++i) {
          const double inv_timescale =
              std::exp(-i * log_step) / p.min_timescale;
          const double angle = positions[t] * inv_timescale;
          const double x1 = head[i];
          const double x2 = head[i + p.depth / 2];
          head[i] = x1 * std::cos(angle) - x2 * std::sin(angle);
          head[i + p.depth / 2] = x2 * std::cos(angle) + x1 * std::sin(angle);
        }
      }
      std::vector<float>& out = is_q ? q : (is_k ? k : v);
      const int local =
          is_q ? h : (is_k ? h - p.heads : h - p.heads - p.kv_heads);
      const int heads = is_q ? p.heads : p.kv_heads;
      static_cast<void>(heads);
      for (int i = 0; i < p.depth; ++i) {
        out[(static_cast<size_t>(local) * p.tokens + t) * p.depth + i] =
            static_cast<float>(head[i]);
      }
    }
  }
}

void RunCase(LiteRtNvidiaQkvNormRopeParams p) {
  SCOPED_TRACE(::testing::Message()
               << "tokens=" << p.tokens << " heads=" << p.heads
               << " kv=" << p.kv_heads << " depth=" << p.depth
               << " rotary=" << p.rotary << " v_norm=" << p.has_v_norm);
  ASSERT_TRUE(LiteRtNvidiaQkvNormRopeSupports(&p));
  const int total = p.heads + 2 * p.kv_heads;
  std::vector<uint16_t> qkv(static_cast<size_t>(p.tokens) * total * p.depth);
  for (size_t i = 0; i < qkv.size(); ++i) {
    qkv[i] =
        FloatToBf16Bits(static_cast<float>((i * 37 + 11) % 211) / 53.0f - 2.0f);
  }
  std::vector<int32_t> positions(p.tokens);
  for (int t = 0; t < p.tokens; ++t) positions[t] = 1000 + 7 * t;
  std::vector<float> q_scale(p.depth);
  std::vector<float> k_scale(p.depth);
  for (int i = 0; i < p.depth; ++i) {
    q_scale[i] = 0.5f + 0.01f * static_cast<float>(i % 17);
    k_scale[i] = 1.5f - 0.02f * static_cast<float>(i % 13);
  }
  auto d_qkv = Upload(qkv.data(), qkv.size() * 2);
  auto d_pos = Upload(positions.data(), positions.size() * 4);
  auto d_qs = Upload(q_scale.data(), q_scale.size() * 4);
  auto d_ks = Upload(k_scale.data(), k_scale.size() * 4);
  const size_t q_count = static_cast<size_t>(p.heads) * p.tokens * p.depth;
  const size_t kv_count = static_cast<size_t>(p.kv_heads) * p.tokens * p.depth;
  auto d_q = Upload(nullptr, q_count * 2);
  auto d_k = Upload(nullptr, kv_count * 2);
  auto d_v = Upload(nullptr, kv_count * 2);
  ASSERT_EQ(LiteRtNvidiaLaunchQkvNormRope(
                &p, d_qkv.get(), static_cast<const int32_t*>(d_pos.get()),
                static_cast<const float*>(d_qs.get()),
                static_cast<const float*>(d_ks.get()), d_q.get(), d_k.get(),
                d_v.get(), nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint16_t> q(q_count), k(kv_count), v(kv_count);
  ASSERT_EQ(
      cudaMemcpy(q.data(), d_q.get(), q_count * 2, cudaMemcpyDeviceToHost),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(k.data(), d_k.get(), kv_count * 2, cudaMemcpyDeviceToHost),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(v.data(), d_v.get(), kv_count * 2, cudaMemcpyDeviceToHost),
      cudaSuccess);
  std::vector<float> rq(q_count), rk(kv_count), rv(kv_count);
  Reference(p, qkv, positions, q_scale, k_scale, rq, rk, rv);
  const auto check = [](const std::vector<uint16_t>& got,
                        const std::vector<float>& want, const char* what) {
    for (size_t i = 0; i < got.size(); ++i) {
      const float value = Bf16BitsToFloat(got[i]);
      ASSERT_NEAR(value, want[i], 1e-2f * std::fabs(want[i]) + 2e-2f)
          << what << " index " << i;
    }
  };
  check(q, rq, "q");
  check(k, rk, "k");
  check(v, rv, "v");
}

TEST(QkvNormRopeTest, MatchesReference) {
  // Gemma 4 26B-A4B's local layers: 16 query heads, 8 KV heads of depth 256,
  // full rotation at base 10,000, value norm; decode and a prefill chunk.
  RunCase({1, 16, 8, 256, 256, 1, 1e-6f, 1.0f, 10000.0f});
  RunCase({37, 16, 8, 256, 256, 1, 1e-6f, 1.0f, 10000.0f});
  // Its global layers' geometry: depth 512, a quarter rotated at base 1e6.
  RunCase({5, 16, 2, 512, 128, 0, 1e-6f, 1.0f, 1000000.0f});
  // Small odd shapes.
  RunCase({3, 2, 1, 64, 32, 1, 1e-5f, 1.0f, 500.0f});
}

TEST(QkvNormRopeTest, RejectsInvalidParams) {
  LiteRtNvidiaQkvNormRopeParams p = {1, 16, 8, 256, 256, 1, 1e-6f, 1.0f, 1e4f};
  EXPECT_TRUE(LiteRtNvidiaQkvNormRopeSupports(&p));
  p.rotary = 258;
  EXPECT_FALSE(LiteRtNvidiaQkvNormRopeSupports(&p));
  p = {1, 16, 8, 2048, 256, 1, 1e-6f, 1.0f, 1e4f};
  EXPECT_FALSE(LiteRtNvidiaQkvNormRopeSupports(&p));
  p = {0, 16, 8, 256, 256, 1, 1e-6f, 1.0f, 1e4f};
  EXPECT_FALSE(LiteRtNvidiaQkvNormRopeSupports(&p));
  EXPECT_FALSE(LiteRtNvidiaQkvNormRopeSupports(nullptr));
}

}  // namespace
