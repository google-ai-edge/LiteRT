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

#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/int2_gemv.h"

namespace {

struct CudaDeleter {
  void operator()(void* pointer) const { cudaFree(pointer); }
};
using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation AllocateCuda(size_t bytes) {
  void* pointer = nullptr;
  const cudaError_t status = cudaMalloc(&pointer, bytes);
  if (status != cudaSuccess) {
    ADD_FAILURE() << "cudaMalloc(" << bytes
                  << ") failed: " << cudaGetErrorString(status);
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
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// Round-to-nearest-even FP32 -> FP16 for normal values.
uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0 || exponent >= 31) {
    ADD_FAILURE() << "FP16 test value out of the normal range: " << value;
    return static_cast<uint16_t>(sign);
  }
  uint32_t half = (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
    ++half;
  }
  return static_cast<uint16_t>(sign | half);
}

float HalfBitsToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  const uint32_t exponent = (value >> 10) & 0x1fu;
  const uint32_t mantissa = value & 0x3ffu;
  uint32_t bits = sign;
  if (exponent != 0) {
    bits |= ((exponent + 127 - 15) << 23) | (mantissa << 13);
  } else if (mantissa != 0) {
    float result = std::ldexp(static_cast<float>(mantissa), -24);
    return (value & 0x8000u) ? -result : result;
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

int8_t DecodeInt4(const std::vector<uint8_t>& packed, size_t index) {
  const uint8_t bits = (packed[index / 2] >> (4 * (index % 2))) & 0xf;
  return static_cast<int8_t>(static_cast<int>(bits ^ 8) - 8);
}

// Row-major [n, k] INT4 values, the even element in the low nibble.
std::vector<uint8_t> MakeInt4Weights(int n, int k, int seed) {
  std::vector<uint8_t> weights(static_cast<size_t>(n) * k / 2, 0);
  for (int row = 0; row < n; ++row) {
    for (int column = 0; column < k; ++column) {
      const int value = ((5 * row + 3 * column + seed + row / 3) % 16) - 8;
      const size_t index = static_cast<size_t>(row) * k + column;
      weights[index / 2] |= (static_cast<uint8_t>(value) & 0xf)
                            << (4 * (index % 2));
    }
  }
  return weights;
}

// FP16 scales [n, k / 32] in the range of Gemma 4 26B's blockwise scales.
std::vector<uint16_t> MakeScales(int n, int k, int seed) {
  const int blocks = k / 32;
  std::vector<uint16_t> scales(static_cast<size_t>(n) * blocks);
  for (int row = 0; row < n; ++row) {
    for (int b = 0; b < blocks; ++b) {
      scales[static_cast<size_t>(row) * blocks + b] = FloatToHalfBits(
          0.003f +
          0.0002f * static_cast<float>((row * 7 + b * 13 + seed) % 97));
    }
  }
  return scales;
}

float Gelu(float x, int gate) {
  if (gate == 1) {
    return 0.5f * x *
           (1.0f +
            std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  }
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

// y[r][n] of the blockwise GEMV, summing each block first as the kernel's
// reference contract.
float Reference(const std::vector<uint16_t>& activation,
                const std::vector<uint8_t>& weights,
                const std::vector<uint16_t>& scales, int r, int row, int k) {
  const int blocks = k / 32;
  float total = 0.0f;
  for (int b = 0; b < blocks; ++b) {
    float partial = 0.0f;
    for (int c = b * 32; c < (b + 1) * 32; ++c) {
      partial =
          std::fma(Bf16BitsToFloat(activation[static_cast<size_t>(r) * k + c]),
                   static_cast<float>(
                       DecodeInt4(weights, static_cast<size_t>(row) * k + c)),
                   partial);
    }
    total += partial *
             HalfBitsToFloat(scales[static_cast<size_t>(row) * blocks + b]);
  }
  return total;
}

TEST(Int4BlockGemvTest, MatchesCpuReference) {
  // Every row count, over the input sizes of the 26B model's projections
  // (the expert down projection of 22 blocks runs on one warp) and a short
  // odd case; 44 channels end in a partial tile of 4.
  constexpr int kN = 44;
  for (const int k : {64, 704, 2112, 2816}) {
    const std::vector<uint8_t> weights = MakeInt4Weights(kN, k, 1);
    const std::vector<uint16_t> scales = MakeScales(kN, k, 1);
    for (int rows = 1; rows <= kLiteRtNvidiaInt4BlockGemvMaxRows; ++rows) {
      SCOPED_TRACE(::testing::Message() << "K=" << k << " rows=" << rows);
      std::vector<uint16_t> activation(static_cast<size_t>(rows) * k);
      for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < k; ++i) {
          activation[static_cast<size_t>(r) * k + i] = FloatToBf16Bits(
              static_cast<float>((i * 17 + r * 29) % 101 - 50) / 37.0f);
        }
      }
      auto device_activation =
          AllocateCuda(activation.size() * sizeof(uint16_t));
      auto device_weights = AllocateCuda(weights.size());
      auto device_scales = AllocateCuda(scales.size() * sizeof(uint16_t));
      auto device_output =
          AllocateCuda(static_cast<size_t>(rows) * kN * sizeof(uint16_t));
      ASSERT_TRUE(device_activation && device_weights && device_scales &&
                  device_output);
      ASSERT_EQ(cudaMemcpy(device_activation.get(), activation.data(),
                           activation.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(device_weights.get(), weights.data(), weights.size(),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(
          cudaMemcpy(device_scales.get(), scales.data(),
                     scales.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
          cudaSuccess);
      ASSERT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
                    device_activation.get(),
                    static_cast<const uint8_t*>(device_weights.get()),
                    device_scales.get(), device_output.get(), rows, k, kN,
                    /*gate=*/0, /*stream=*/nullptr),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::vector<uint16_t> actual(static_cast<size_t>(rows) * kN);
      ASSERT_EQ(
          cudaMemcpy(actual.data(), device_output.get(),
                     actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
          cudaSuccess);
      for (int r = 0; r < rows; ++r) {
        for (int row = 0; row < kN; ++row) {
          const float expected =
              Reference(activation, weights, scales, r, row, k);
          const float got =
              Bf16BitsToFloat(actual[static_cast<size_t>(r) * kN + row]);
          EXPECT_NEAR(got, expected, 1e-2f * std::fabs(expected) + 1e-3f)
              << "r=" << r << " row=" << row;
        }
      }
    }
  }
}

// The RMS norm fused in front of the product, plain and gated, over the
// channel tiles of 8 (wide), 4 and 2 (narrow) outputs.
TEST(Int4BlockGemvTest, FusedNormMatchesCpuReference) {
  struct Case {
    int n;
    int k;
    int rows;
    int gate;
  };
  const Case cases[] = {
      {44, 2816, 1, 0},    // 2-channel tiles
      {4224, 2816, 3, 0},  // 4-channel tiles (the gate+up group's width)
      {8200, 2816, 2, 0},  // 8-channel tiles
      {2112, 2816, 1, 1},  // gated, 2-channel tiles
      {4100, 2816, 4, 2},  // gated, 4-channel tiles
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "n=" << c.n << " k=" << c.k << " rows="
                                      << c.rows << " gate=" << c.gate);
    const int channels = c.gate != 0 ? 2 * c.n : c.n;
    const std::vector<uint8_t> weights = MakeInt4Weights(channels, c.k, 7);
    const std::vector<uint16_t> scales = MakeScales(channels, c.k, 7);
    std::vector<uint16_t> activation(static_cast<size_t>(c.rows) * c.k);
    for (int r = 0; r < c.rows; ++r) {
      for (int i = 0; i < c.k; ++i) {
        activation[static_cast<size_t>(r) * c.k + i] = FloatToBf16Bits(
            static_cast<float>((i * 19 + r * 23) % 97 - 48) / 11.0f);
      }
    }
    std::vector<float> norm_scale(c.k);
    for (int i = 0; i < c.k; ++i) norm_scale[i] = 0.8f + 0.001f * (i % 50);
    const float eps = 1e-6f;
    const float multiplier = 0.6f;
    auto d_w = AllocateCuda(weights.size());
    auto d_s = AllocateCuda(scales.size() * 2);
    auto d_a = AllocateCuda(activation.size() * 2);
    auto d_ns = AllocateCuda(norm_scale.size() * 4);
    auto d_o = AllocateCuda(static_cast<size_t>(c.rows) * c.n * 2);
    cudaMemcpy(d_w.get(), weights.data(), weights.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.get(), scales.data(), scales.size() * 2,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_a.get(), activation.data(), activation.size() * 2,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ns.get(), norm_scale.data(), norm_scale.size() * 4,
               cudaMemcpyHostToDevice);
    ASSERT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRowsNormed(
                  d_a.get(), static_cast<const uint8_t*>(d_w.get()), d_s.get(),
                  static_cast<const float*>(d_ns.get()), eps, multiplier,
                  d_o.get(), c.rows, c.k, c.n, c.gate, nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint16_t> actual(static_cast<size_t>(c.rows) * c.n);
    ASSERT_EQ(cudaMemcpy(actual.data(), d_o.get(), actual.size() * 2,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int r = 0; r < c.rows; ++r) {
      // The normalized row as the reference sees it.
      double squares = 0.0;
      for (int i = 0; i < c.k; ++i) {
        const double x =
            Bf16BitsToFloat(activation[static_cast<size_t>(r) * c.k + i]);
        squares += x * x;
      }
      const double factor = 1.0 / std::sqrt(squares / c.k + eps) * multiplier;
      std::vector<uint16_t> normed(c.k);
      std::vector<float> normed_f(c.k);
      for (int i = 0; i < c.k; ++i) {
        normed_f[i] = static_cast<float>(
            Bf16BitsToFloat(activation[static_cast<size_t>(r) * c.k + i]) *
            factor * norm_scale[i]);
      }
      // Reference() takes BF16 rows: scale the raw row instead, in FP32
      // terms the kernel's order (x * scale, then the row factor).
      for (int row = 0; row < c.n; ++row) {
        const auto dot = [&](int channel) {
          const int blocks = c.k / 32;
          float total = 0.0f;
          for (int b = 0; b < blocks; ++b) {
            float partial = 0.0f;
            for (int col = b * 32; col < (b + 1) * 32; ++col) {
              partial = std::fma(
                  normed_f[col],
                  static_cast<float>(DecodeInt4(
                      weights, static_cast<size_t>(channel) * c.k + col)),
                  partial);
            }
            total += partial *
                     HalfBitsToFloat(
                         scales[static_cast<size_t>(channel) * blocks + b]);
          }
          return total;
        };
        float expected = dot(row);
        if (c.gate != 0) expected = Gelu(expected, c.gate) * dot(c.n + row);
        const float got =
            Bf16BitsToFloat(actual[static_cast<size_t>(r) * c.n + row]);
        EXPECT_NEAR(got, expected, 1.5e-2f * std::fabs(expected) + 2e-3f)
            << "r=" << r << " row=" << row;
      }
    }
  }
}

TEST(Int4BlockGemvTest, GatedFeedForwardMatchesCpuReference) {
  // Gate rows then up rows; gelu(gate) * up in both GELU forms.
  constexpr int kN = 40;
  constexpr int kK = 2816;
  const std::vector<uint8_t> weights = MakeInt4Weights(2 * kN, kK, 3);
  const std::vector<uint16_t> scales = MakeScales(2 * kN, kK, 3);
  for (const int gate : {1, 2}) {
    for (const int rows : {1, 3, 8}) {
      SCOPED_TRACE(::testing::Message() << "gate=" << gate << " rows=" << rows);
      std::vector<uint16_t> activation(static_cast<size_t>(rows) * kK);
      for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < kK; ++i) {
          activation[static_cast<size_t>(r) * kK + i] = FloatToBf16Bits(
              static_cast<float>((i * 23 + r * 31) % 89 - 44) / 41.0f);
        }
      }
      auto device_activation =
          AllocateCuda(activation.size() * sizeof(uint16_t));
      auto device_weights = AllocateCuda(weights.size());
      auto device_scales = AllocateCuda(scales.size() * sizeof(uint16_t));
      auto device_output =
          AllocateCuda(static_cast<size_t>(rows) * kN * sizeof(uint16_t));
      ASSERT_TRUE(device_activation && device_weights && device_scales &&
                  device_output);
      ASSERT_EQ(cudaMemcpy(device_activation.get(), activation.data(),
                           activation.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(device_weights.get(), weights.data(), weights.size(),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(
          cudaMemcpy(device_scales.get(), scales.data(),
                     scales.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
          cudaSuccess);
      ASSERT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
                    device_activation.get(),
                    static_cast<const uint8_t*>(device_weights.get()),
                    device_scales.get(), device_output.get(), rows, kK, kN,
                    gate, /*stream=*/nullptr),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::vector<uint16_t> actual(static_cast<size_t>(rows) * kN);
      ASSERT_EQ(
          cudaMemcpy(actual.data(), device_output.get(),
                     actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
          cudaSuccess);
      for (int r = 0; r < rows; ++r) {
        for (int row = 0; row < kN; ++row) {
          const float g = Reference(activation, weights, scales, r, row, kK);
          const float u =
              Reference(activation, weights, scales, r, kN + row, kK);
          const float expected = Gelu(g, gate) * u;
          const float got =
              Bf16BitsToFloat(actual[static_cast<size_t>(r) * kN + row]);
          EXPECT_NEAR(got, expected, 1e-2f * std::fabs(expected) + 1e-3f)
              << "r=" << r << " row=" << row;
        }
      }
    }
  }
}

// The experts' weights in the model's interleaved layout: row o of expert e
// is row o * experts + e.
struct MoeCase {
  int rows;
  int top_k;
  int experts;
  int model_dim;
  int hidden_dim;
  int gate;
};

void RunMoeCase(const MoeCase& c) {
  SCOPED_TRACE(::testing::Message()
               << "rows=" << c.rows << " top_k=" << c.top_k
               << " experts=" << c.experts << " model_dim=" << c.model_dim
               << " hidden_dim=" << c.hidden_dim << " gate=" << c.gate);
  const LiteRtNvidiaMoeShape shape = {c.rows,      c.top_k,      c.experts,
                                      c.model_dim, c.hidden_dim, c.gate};
  ASSERT_TRUE(LiteRtNvidiaMoeSupports(&shape));
  const int gate_rows = c.hidden_dim * c.experts;
  const int down_rows = c.model_dim * c.experts;
  const std::vector<uint8_t> gate_w =
      MakeInt4Weights(gate_rows, c.model_dim, 5);
  const std::vector<uint8_t> up_w = MakeInt4Weights(gate_rows, c.model_dim, 9);
  const std::vector<uint8_t> down_w =
      MakeInt4Weights(down_rows, c.hidden_dim, 13);
  const std::vector<uint16_t> gate_s = MakeScales(gate_rows, c.model_dim, 5);
  const std::vector<uint16_t> up_s = MakeScales(gate_rows, c.model_dim, 9);
  const std::vector<uint16_t> down_s = MakeScales(down_rows, c.hidden_dim, 13);
  std::vector<uint16_t> activation(static_cast<size_t>(c.rows) * c.model_dim);
  for (int r = 0; r < c.rows; ++r) {
    for (int i = 0; i < c.model_dim; ++i) {
      activation[static_cast<size_t>(r) * c.model_dim + i] = FloatToBf16Bits(
          static_cast<float>((i * 13 + r * 7) % 97 - 48) / 29.0f);
    }
  }
  // Distinct experts per row, some experts never routed, some popular.
  std::vector<int32_t> indices(static_cast<size_t>(c.rows) * c.top_k);
  std::vector<float> weights(indices.size());
  for (int r = 0; r < c.rows; ++r) {
    for (int j = 0; j < c.top_k; ++j) {
      indices[static_cast<size_t>(r) * c.top_k + j] =
          (r % 3 == 0 ? j : (r * 5 + j * 3 + r / 7)) % c.experts;
      weights[static_cast<size_t>(r) * c.top_k + j] =
          0.05f + 0.1f * static_cast<float>((r + 2 * j) % 7);
    }
  }
  std::vector<float> expert_scale(c.experts);
  for (int e = 0; e < c.experts; ++e) {
    expert_scale[e] = 0.98f + 0.0005f * static_cast<float>(e % 50);
  }
  const auto upload = [](const void* data, size_t bytes) {
    auto device = AllocateCuda(bytes);
    EXPECT_EQ(cudaMemcpy(device.get(), data, bytes, cudaMemcpyHostToDevice),
              cudaSuccess);
    return device;
  };
  auto d_gate = upload(gate_w.data(), gate_w.size());
  auto d_up = upload(up_w.data(), up_w.size());
  auto d_down = upload(down_w.data(), down_w.size());
  auto d_gate_s = upload(gate_s.data(), gate_s.size() * 2);
  auto d_up_s = upload(up_s.data(), up_s.size() * 2);
  auto d_down_s = upload(down_s.data(), down_s.size() * 2);
  auto d_act = upload(activation.data(), activation.size() * 2);
  auto d_idx = upload(indices.data(), indices.size() * 4);
  auto d_w = upload(weights.data(), weights.size() * 4);
  auto d_es = upload(expert_scale.data(), expert_scale.size() * 4);
  const size_t workspace_bytes = LiteRtNvidiaMoeWorkspaceBytes(&shape);
  ASSERT_GT(workspace_bytes, 0u);
  auto d_ws = AllocateCuda(workspace_bytes);
  auto d_out = AllocateCuda(static_cast<size_t>(c.rows) * c.model_dim * 2);
  const LiteRtNvidiaMoeArena arena = {static_cast<const uint8_t*>(d_gate.get()),
                                      static_cast<const uint8_t*>(d_up.get()),
                                      static_cast<const uint8_t*>(d_down.get()),
                                      d_gate_s.get(),
                                      d_up_s.get(),
                                      d_down_s.get()};
  ASSERT_EQ(
      LiteRtNvidiaLaunchMoeInt4Block(
          &shape, &arena, d_act.get(), static_cast<const int32_t*>(d_idx.get()),
          static_cast<const float*>(d_w.get()),
          static_cast<const float*>(d_es.get()), d_ws.get(), d_out.get(),
          /*stream=*/nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint16_t> actual(static_cast<size_t>(c.rows) * c.model_dim);
  ASSERT_EQ(cudaMemcpy(actual.data(), d_out.get(), actual.size() * 2,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  // Reference: expert e's row o of a projection is row o * experts + e.
  const auto dot = [&](const std::vector<uint8_t>& w,
                       const std::vector<uint16_t>& sc, int row, int k,
                       const std::vector<float>& x) {
    const int blocks = k / 32;
    float total = 0.0f;
    for (int b = 0; b < blocks; ++b) {
      float partial = 0.0f;
      for (int col = b * 32; col < (b + 1) * 32; ++col) {
        partial = std::fma(x[col],
                           static_cast<float>(DecodeInt4(
                               w, static_cast<size_t>(row) * k + col)),
                           partial);
      }
      total +=
          partial * HalfBitsToFloat(sc[static_cast<size_t>(row) * blocks + b]);
    }
    return total;
  };
  std::vector<float> x(c.model_dim);
  std::vector<float> hidden(c.hidden_dim);
  for (int r = 0; r < c.rows; ++r) {
    for (int i = 0; i < c.model_dim; ++i) {
      x[i] =
          Bf16BitsToFloat(activation[static_cast<size_t>(r) * c.model_dim + i]);
    }
    std::vector<float> expected(c.model_dim, 0.0f);
    for (int j = 0; j < c.top_k; ++j) {
      const int e = indices[static_cast<size_t>(r) * c.top_k + j];
      const float weight =
          weights[static_cast<size_t>(r) * c.top_k + j] * expert_scale[e];
      for (int o = 0; o < c.hidden_dim; ++o) {
        const int row = o * c.experts + e;
        hidden[o] = Gelu(dot(gate_w, gate_s, row, c.model_dim, x), c.gate) *
                    dot(up_w, up_s, row, c.model_dim, x);
      }
      for (int o = 0; o < c.model_dim; ++o) {
        expected[o] += weight * dot(down_w, down_s, o * c.experts + e,
                                    c.hidden_dim, hidden);
      }
    }
    double error = 0.0;
    double norm = 0.0;
    for (int o = 0; o < c.model_dim; ++o) {
      const float got =
          Bf16BitsToFloat(actual[static_cast<size_t>(r) * c.model_dim + o]);
      ASSERT_TRUE(std::isfinite(got)) << "r=" << r << " o=" << o;
      error += static_cast<double>(got - expected[o]) * (got - expected[o]);
      norm += static_cast<double>(expected[o]) * expected[o];
    }
    // Decode: BF16 output rounding (2^-9 relative) plus FP32 ordering.
    // Prefill: the tensor-core path also rounds the gated hidden rows to
    // BF16 and sums the down projection in FP16 over 128-dim slices, about
    // 0.7% RMS against the FP32 reference.
    const double tolerance =
        c.rows <= kLiteRtNvidiaInt4BlockGemvMaxRows ? 4e-3 : 1.5e-2;
    EXPECT_LE(std::sqrt(error), tolerance * std::sqrt(norm) + 1e-4)
        << "r=" << r << " relative "
        << std::sqrt(error / std::max(norm, 1e-30));
  }
}

TEST(Int4BlockGemvTest, MoeDecodeMatchesCpuReference) {
  // One row and a verify step's few rows: one GEMV per routed expert.
  RunMoeCase({1, 8, 16, 256, 64, 1});
  RunMoeCase({3, 4, 12, 2816, 704, 1});
  RunMoeCase({8, 2, 5, 128, 96, 2});
}

TEST(Int4BlockGemvTest, MoePrefillMatchesCpuReference) {
  // Many rows grouped by expert into GEMM tiles, including experts no row
  // routes to, experts with several tiles and partial last tiles.
  RunMoeCase({40, 3, 7, 256, 64, 1});
  RunMoeCase({256, 8, 16, 704, 128, 2});
  RunMoeCase({130, 8, 128, 2816, 704, 1});
  RunMoeCase({1024, 8, 128, 2816, 704, 1});
}

// The fused block (router norm, router GEMV, softmax, top-k, renormalized
// weights, expert norm) against the plain launch fed the CPU's routing.
TEST(Int4BlockGemvTest, MoeFusedRoutingMatchesPlainLaunch) {
  const LiteRtNvidiaMoeShape shape = {5, 8, 128, 2816, 704, 1};
  const int experts = shape.experts, dim = shape.model_dim,
            hidden = shape.hidden_dim;
  const std::vector<uint8_t> gate_w = MakeInt4Weights(hidden * experts, dim, 5);
  const std::vector<uint8_t> up_w = MakeInt4Weights(hidden * experts, dim, 9);
  const std::vector<uint8_t> down_w =
      MakeInt4Weights(dim * experts, hidden, 13);
  const std::vector<uint16_t> gate_s = MakeScales(hidden * experts, dim, 5);
  const std::vector<uint16_t> up_s = MakeScales(hidden * experts, dim, 9);
  const std::vector<uint16_t> down_s = MakeScales(dim * experts, hidden, 13);
  const std::vector<uint8_t> router_w = MakeInt4Weights(experts, dim, 21);
  const std::vector<uint16_t> router_s = MakeScales(experts, dim, 21);
  std::vector<uint16_t> x(static_cast<size_t>(shape.rows) * dim);
  for (int r = 0; r < shape.rows; ++r) {
    for (int i = 0; i < dim; ++i) {
      x[static_cast<size_t>(r) * dim + i] = FloatToBf16Bits(
          static_cast<float>((i * 31 + r * 17) % 103 - 51) / 7.0f);
    }
  }
  std::vector<float> router_norm(dim), expert_norm(dim), expert_scale(experts);
  for (int i = 0; i < dim; ++i) {
    router_norm[i] = 0.7f + 0.0005f * (i % 61);
    expert_norm[i] = 1.1f - 0.0004f * (i % 53);
  }
  for (int e = 0; e < experts; ++e) expert_scale[e] = 0.98f + 0.0003f * e;
  const float eps = 1e-6f, multiplier = 0.6f;
  // The CPU's routing: norm, logits, softmax, top-8, renormalize. The
  // device's logits are BF16 (the GEMV's output), so near-ties may pick
  // another expert: the probabilities are kept to judge those.
  std::vector<int32_t> indices(static_cast<size_t>(shape.rows) * shape.top_k);
  std::vector<float> weights(indices.size());
  std::vector<std::vector<double>> probabilities(shape.rows);
  std::vector<uint16_t> normed(x.size());
  for (int r = 0; r < shape.rows; ++r) {
    double squares = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double v = Bf16BitsToFloat(x[static_cast<size_t>(r) * dim + i]);
      squares += v * v;
    }
    const double inv = 1.0 / std::sqrt(squares / dim + eps);
    std::vector<float> xr(dim);
    for (int i = 0; i < dim; ++i) {
      const float v = Bf16BitsToFloat(x[static_cast<size_t>(r) * dim + i]);
      xr[i] = static_cast<float>(v * inv * multiplier * router_norm[i]);
      normed[static_cast<size_t>(r) * dim + i] =
          FloatToBf16Bits(static_cast<float>(v * inv * expert_norm[i]));
    }
    std::vector<double> logits(experts);
    for (int e = 0; e < experts; ++e) {
      double total = 0.0;
      for (int b = 0; b < dim / 32; ++b) {
        double partial = 0.0;
        for (int c = b * 32; c < (b + 1) * 32; ++c) {
          partial +=
              xr[c] * DecodeInt4(router_w, static_cast<size_t>(e) * dim + c);
        }
        total +=
            partial *
            HalfBitsToFloat(router_s[static_cast<size_t>(e) * (dim / 32) + b]);
      }
      logits[e] = total;
    }
    const double largest = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    std::vector<double> prob(experts);
    for (int e = 0; e < experts; ++e) {
      prob[e] = std::exp(logits[e] - largest);
      sum += prob[e];
    }
    for (int e = 0; e < experts; ++e) prob[e] /= sum;
    probabilities[r] = prob;
    double chosen = 0.0;
    for (int j = 0; j < shape.top_k; ++j) {
      int best = 0;
      for (int e = 1; e < experts; ++e) {
        if (prob[e] > prob[best]) best = e;
      }
      indices[static_cast<size_t>(r) * shape.top_k + j] = best;
      weights[static_cast<size_t>(r) * shape.top_k + j] =
          static_cast<float>(prob[best]);
      chosen += prob[best];
      prob[best] = -1.0;
    }
    for (int j = 0; j < shape.top_k; ++j) {
      weights[static_cast<size_t>(r) * shape.top_k + j] /=
          static_cast<float>(chosen);
    }
  }
  const auto upload = [](const void* data, size_t bytes) {
    auto device = AllocateCuda(bytes);
    cudaMemcpy(device.get(), data, bytes, cudaMemcpyHostToDevice);
    return device;
  };
  auto d_gate = upload(gate_w.data(), gate_w.size());
  auto d_up = upload(up_w.data(), up_w.size());
  auto d_down = upload(down_w.data(), down_w.size());
  auto d_gate_s = upload(gate_s.data(), gate_s.size() * 2);
  auto d_up_s = upload(up_s.data(), up_s.size() * 2);
  auto d_down_s = upload(down_s.data(), down_s.size() * 2);
  auto d_router = upload(router_w.data(), router_w.size());
  auto d_router_s = upload(router_s.data(), router_s.size() * 2);
  auto d_x = upload(x.data(), x.size() * 2);
  auto d_normed = upload(normed.data(), normed.size() * 2);
  auto d_idx = upload(indices.data(), indices.size() * 4);
  auto d_w = upload(weights.data(), weights.size() * 4);
  auto d_es = upload(expert_scale.data(), expert_scale.size() * 4);
  auto d_rn = upload(router_norm.data(), router_norm.size() * 4);
  auto d_en = upload(expert_norm.data(), expert_norm.size() * 4);
  const size_t workspace_bytes = LiteRtNvidiaMoeWorkspaceBytes(&shape);
  auto d_ws = AllocateCuda(workspace_bytes);
  const size_t out_bytes = static_cast<size_t>(shape.rows) * dim * 2;
  auto d_out_fused = AllocateCuda(out_bytes);
  auto d_out_plain = AllocateCuda(out_bytes);
  const LiteRtNvidiaMoeArena arena = {
      static_cast<const uint8_t*>(d_gate.get()),
      static_cast<const uint8_t*>(d_up.get()),
      static_cast<const uint8_t*>(d_down.get()),
      d_gate_s.get(),
      d_up_s.get(),
      d_down_s.get(),
      static_cast<const uint8_t*>(d_router.get()),
      d_router_s.get()};
  ASSERT_EQ(
      LiteRtNvidiaLaunchMoeInt4BlockFused(
          &shape, &arena, d_x.get(), static_cast<const float*>(d_es.get()),
          static_cast<const float*>(d_rn.get()), multiplier,
          static_cast<const float*>(d_en.get()), eps, /*renormalize=*/1,
          d_ws.get(), d_out_fused.get(), nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  // The routing the fused launch chose sits at the front of its workspace.
  std::vector<int32_t> got_indices(indices.size());
  std::vector<float> got_weights(weights.size());
  ASSERT_EQ(cudaMemcpy(got_indices.data(), d_ws.get(), got_indices.size() * 4,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(got_weights.data(),
                       static_cast<const char*>(d_ws.get()) +
                           ((got_indices.size() * 4 + 255) / 256) * 256,
                       got_weights.size() * 4, cudaMemcpyDeviceToHost),
            cudaSuccess);
  // The device's set of experts carries (nearly) the probability mass of
  // the CPU's, and its weights are those experts' probabilities
  // renormalized.
  for (int r = 0; r < shape.rows; ++r) {
    double expected_mass = 0.0, got_mass = 0.0;
    for (int j = 0; j < shape.top_k; ++j) {
      const size_t pair = static_cast<size_t>(r) * shape.top_k + j;
      ASSERT_GE(got_indices[pair], 0);
      ASSERT_LT(got_indices[pair], experts);
      expected_mass += probabilities[r][indices[pair]];
      got_mass += probabilities[r][got_indices[pair]];
    }
    EXPECT_GE(got_mass, expected_mass * (1.0 - 2e-3)) << "row " << r;
    for (int j = 0; j < shape.top_k; ++j) {
      const size_t pair = static_cast<size_t>(r) * shape.top_k + j;
      EXPECT_NEAR(got_weights[pair],
                  probabilities[r][got_indices[pair]] / got_mass, 3e-3f)
          << "row " << r << " slot " << j;
    }
  }
  // The plain launch over the device's own routing and host-normalized rows.
  ASSERT_EQ(cudaMemcpy(d_idx.get(), got_indices.data(), got_indices.size() * 4,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_w.get(), got_weights.data(), got_weights.size() * 4,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(
      LiteRtNvidiaLaunchMoeInt4Block(&shape, &arena, d_normed.get(),
                                     static_cast<const int32_t*>(d_idx.get()),
                                     static_cast<const float*>(d_w.get()),
                                     static_cast<const float*>(d_es.get()),
                                     d_ws.get(), d_out_plain.get(), nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint16_t> fused(out_bytes / 2), plain(out_bytes / 2);
  ASSERT_EQ(cudaMemcpy(fused.data(), d_out_fused.get(), out_bytes,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(plain.data(), d_out_plain.get(), out_bytes,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  double error = 0.0, norm = 0.0;
  for (size_t i = 0; i < fused.size(); ++i) {
    const double a = Bf16BitsToFloat(fused[i]), b = Bf16BitsToFloat(plain[i]);
    error += (a - b) * (a - b);
    norm += b * b;
  }
  // The plain launch saw the normalized rows rounded to BF16.
  EXPECT_LE(std::sqrt(error), 1e-2 * std::sqrt(norm) + 1e-6)
      << "relative " << std::sqrt(error / norm);
}

TEST(Int4BlockGemvTest, MoeRejectsInvalidShapes) {
  const auto supports = [](LiteRtNvidiaMoeShape shape) {
    return LiteRtNvidiaMoeSupports(&shape);
  };
  EXPECT_TRUE(supports({1024, 8, 128, 2816, 704, 1}));
  EXPECT_FALSE(supports({0, 8, 128, 2816, 704, 1}));
  EXPECT_FALSE(supports({1, 0, 128, 2816, 704, 1}));
  EXPECT_FALSE(supports({1, 8, 128, 2800, 704, 1}));
  EXPECT_FALSE(supports({1, 8, 128, 2816, 700, 1}));
  // Prefill needs the GEMM's input sizes (multiples of 64); decode does not.
  EXPECT_TRUE(supports({4, 8, 16, 256, 96, 1}));
  EXPECT_FALSE(supports({64, 8, 16, 256, 96, 1}));
  EXPECT_FALSE(supports({1, 8, 128, 2816, 704, 0}));
  EXPECT_FALSE(supports({1, 8, 2048, 2816, 704, 1}));
  EXPECT_FALSE(LiteRtNvidiaMoeSupports(nullptr));
  EXPECT_EQ(LiteRtNvidiaMoeWorkspaceBytes(nullptr), 0u);
}

// Times the decode launches at Gemma 4 26B-A4B's shapes (not a correctness
// test): INT4_BLOCK_GEMV_TIMING=1 prints microseconds and GB/s per launch.
TEST(Int4BlockGemvTest, TimesDecodeShapes) {
  if (std::getenv("INT4_BLOCK_GEMV_TIMING") == nullptr) {
    GTEST_SKIP() << "Set INT4_BLOCK_GEMV_TIMING=1.";
  }
  struct Shape {
    const char* name;
    int rows;
    int n;
    int k;
    int gate;
  };
  // The head (415 MB) streams from DRAM; the per-layer shapes fit the L2
  // when launched back to back, so their rates are L2 rates. Shapes of
  // 92 MB and more measure DRAM; the first entry warms the clocks.
  const Shape shapes[] = {
      {"warm-up [65536 x 2816]", 1, 65536, 2816, 0},
      {"qkv [8192 x 2816]", 1, 8192, 2816, 0},
      {"o [2816 x 4096]", 1, 2816, 4096, 0},
      {"gate+up group [4224 x 2816]", 1, 4224, 2816, 0},
      {"down [2816 x 2112]", 1, 2816, 2112, 0},
      {"router [128 x 2816]", 1, 128, 2816, 0},
      {"DRAM [65536 x 2816] 92 MB", 1, 65536, 2816, 0},
      {"head [262144 x 2816]", 1, 262144, 2816, 0},
      {"qkv, 4 rows", 4, 8192, 2816, 0},
      {"head, 4 rows", 4, 262144, 2816, 0},
  };
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int kIterations = 50;
  bool warmed = false;
  for (const auto& sh : shapes) {
    std::vector<uint8_t> weights(static_cast<size_t>(sh.n) * sh.k / 2, 0x5a);
    std::vector<uint16_t> scales(static_cast<size_t>(sh.n) * (sh.k / 32),
                                 FloatToHalfBits(0.01f));
    std::vector<uint16_t> activation(static_cast<size_t>(sh.rows) * sh.k,
                                     FloatToBf16Bits(0.5f));
    auto d_w = AllocateCuda(weights.size());
    auto d_s = AllocateCuda(scales.size() * 2);
    auto d_a = AllocateCuda(activation.size() * 2);
    auto d_o = AllocateCuda(static_cast<size_t>(sh.rows) * sh.n * 2);
    cudaMemcpy(d_w.get(), weights.data(), weights.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_s.get(), scales.data(), scales.size() * 2,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_a.get(), activation.data(), activation.size() * 2,
               cudaMemcpyHostToDevice);
    const auto launch = [&]() {
      return LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
          d_a.get(), static_cast<const uint8_t*>(d_w.get()), d_s.get(),
          d_o.get(), sh.rows, sh.k, sh.n, sh.gate, nullptr);
    };
    ASSERT_EQ(launch(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    if (!warmed) {
      // Half a second of launches so the clocks are up.
      for (int i = 0; i < 2000; ++i) ASSERT_EQ(launch(), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      warmed = true;
    }
    cudaEventRecord(start);
    for (int i = 0; i < kIterations; ++i) ASSERT_EQ(launch(), cudaSuccess);
    cudaEventRecord(stop);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    const double us = ms * 1000.0 / kIterations;
    const double bytes = weights.size() + scales.size() * 2.0;
    std::printf("%-30s %8.1f us  %6.0f GB/s\n", sh.name, us,
                bytes / (us * 1e-6) / 1e9);
    if (sh.n == 65536 && sh.rows == 1) {
      // The per-channel kernel on the same bytes (BF16 per-channel scales).
      std::vector<uint16_t> channel_scales(sh.n, FloatToBf16Bits(0.01f));
      auto d_cs = AllocateCuda(channel_scales.size() * 2);
      cudaMemcpy(d_cs.get(), channel_scales.data(), channel_scales.size() * 2,
                 cudaMemcpyHostToDevice);
      const auto launch_channel = [&]() {
        return LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRows(
            d_a.get(), static_cast<const uint8_t*>(d_w.get()), d_cs.get(),
            d_o.get(), 4, sh.rows, sh.k, sh.n, nullptr);
      };
      ASSERT_EQ(launch_channel(), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      cudaEventRecord(start);
      for (int i = 0; i < kIterations; ++i)
        ASSERT_EQ(launch_channel(), cudaSuccess);
      cudaEventRecord(stop);
      ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
      cudaEventElapsedTime(&ms, start, stop);
      const double channel_us = ms * 1000.0 / kIterations;
      std::printf("%-30s %8.1f us  %6.0f GB/s\n", "  per-channel kernel, same",
                  channel_us, weights.size() / (channel_us * 1e-6) / 1e9);
    }
  }
  // The expert launch of one decode token: 8 of 128 experts.
  {
    const LiteRtNvidiaMoeShape shape = {1, 8, 128, 2816, 704, 1};
    const int gate_rows = 704 * 128;
    const int down_rows = 2816 * 128;
    std::vector<uint8_t> gate_w(static_cast<size_t>(gate_rows) * 2816 / 2,
                                0x5a);
    std::vector<uint8_t> down_w(static_cast<size_t>(down_rows) * 704 / 2, 0x5a);
    std::vector<uint16_t> gate_s(static_cast<size_t>(gate_rows) * 88,
                                 FloatToHalfBits(0.01f));
    std::vector<uint16_t> down_s(static_cast<size_t>(down_rows) * 22,
                                 FloatToHalfBits(0.01f));
    std::vector<uint16_t> activation(2816, FloatToBf16Bits(0.5f));
    const std::vector<int32_t> indices = {3, 17, 40, 61, 77, 90, 101, 120};
    const std::vector<float> weights(8, 0.125f);
    std::vector<float> expert_scale(128, 1.0f);
    const auto upload = [](const void* data, size_t bytes) {
      auto device = AllocateCuda(bytes);
      cudaMemcpy(device.get(), data, bytes, cudaMemcpyHostToDevice);
      return device;
    };
    auto d_gate = upload(gate_w.data(), gate_w.size());
    auto d_up = upload(gate_w.data(), gate_w.size());
    auto d_down = upload(down_w.data(), down_w.size());
    auto d_gate_s = upload(gate_s.data(), gate_s.size() * 2);
    auto d_up_s = upload(gate_s.data(), gate_s.size() * 2);
    auto d_down_s = upload(down_s.data(), down_s.size() * 2);
    auto d_act = upload(activation.data(), activation.size() * 2);
    auto d_idx = upload(indices.data(), indices.size() * 4);
    auto d_w = upload(weights.data(), weights.size() * 4);
    auto d_es = upload(expert_scale.data(), expert_scale.size() * 4);
    auto d_ws = AllocateCuda(LiteRtNvidiaMoeWorkspaceBytes(&shape));
    auto d_out = AllocateCuda(2816 * 2);
    const LiteRtNvidiaMoeArena arena = {
        static_cast<const uint8_t*>(d_gate.get()),
        static_cast<const uint8_t*>(d_up.get()),
        static_cast<const uint8_t*>(d_down.get()),
        d_gate_s.get(),
        d_up_s.get(),
        d_down_s.get()};
    const auto launch = [&]() {
      return LiteRtNvidiaLaunchMoeInt4Block(
          &shape, &arena, d_act.get(), static_cast<const int32_t*>(d_idx.get()),
          static_cast<const float*>(d_w.get()),
          static_cast<const float*>(d_es.get()), d_ws.get(), d_out.get(),
          nullptr);
    };
    ASSERT_EQ(launch(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaEventRecord(start);
    for (int i = 0; i < kIterations; ++i) ASSERT_EQ(launch(), cudaSuccess);
    cudaEventRecord(stop);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    const double us = ms * 1000.0 / kIterations;
    const double bytes =
        8.0 * (2 * 704 * 1408 + 2816 * 352 + 2 * 704 * 88 * 2 + 2816 * 22 * 2);
    std::printf("%-30s %8.1f us  %6.0f GB/s (3 launches)\n",
                "moe decode, 8 experts", us, bytes / (us * 1e-6) / 1e9);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

TEST(Int4BlockGemvTest, RejectsInvalidArguments) {
  uint8_t byte = 0;
  EXPECT_EQ(
      LiteRtNvidiaLaunchBf16Int4BlockGemvRows(&byte, &byte, &byte, &byte,
                                              /*rows=*/0, 64, 8, 0, nullptr),
      cudaErrorInvalidValue);
  EXPECT_EQ(
      LiteRtNvidiaLaunchBf16Int4BlockGemvRows(&byte, &byte, &byte, &byte, 1,
                                              /*input_size=*/48, 8, 0, nullptr),
      cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
                &byte, &byte, &byte, &byte, 1, 64, 8, /*gate=*/3, nullptr),
            cudaErrorInvalidValue);
  EXPECT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
                &byte, &byte, &byte, &byte,
                kLiteRtNvidiaInt4BlockGemvMaxRows + 1, 64, 8, 0, nullptr),
            cudaErrorInvalidValue);
}

// The layer block against its parts: the fused expert launch for the experts'
// output, the GEMV entries (8 rows at a time) for the dense feed-forward, and
// the three norms and the residual update on the host. Both the decode path
// (3 rows) and the prefill path (128 rows: grouped GEMMs, the gathering
// epilogue) run.
TEST(Int4BlockGemvTest, MoeLayerBlockMatchesItsParts) {
  for (const int rows : {3, 128}) {
    const LiteRtNvidiaMoeShape shape = {rows, 8, 128, 2816, 704, 1};
    const int experts = shape.experts, dim = shape.model_dim,
              hidden = shape.hidden_dim;
    const int dense_hidden = 2112;
    const std::vector<uint8_t> gate_w =
        MakeInt4Weights(hidden * experts, dim, 5);
    const std::vector<uint8_t> up_w = MakeInt4Weights(hidden * experts, dim, 9);
    const std::vector<uint8_t> down_w =
        MakeInt4Weights(dim * experts, hidden, 13);
    const std::vector<uint16_t> gate_s = MakeScales(hidden * experts, dim, 5);
    const std::vector<uint16_t> up_s = MakeScales(hidden * experts, dim, 9);
    const std::vector<uint16_t> down_s = MakeScales(dim * experts, hidden, 13);
    const std::vector<uint8_t> router_w = MakeInt4Weights(experts, dim, 21);
    const std::vector<uint16_t> router_s = MakeScales(experts, dim, 21);
    const std::vector<uint8_t> dense_gu_w =
        MakeInt4Weights(2 * dense_hidden, dim, 33);
    const std::vector<uint16_t> dense_gu_s =
        MakeScales(2 * dense_hidden, dim, 33);
    const std::vector<uint8_t> dense_down_w =
        MakeInt4Weights(dim, dense_hidden, 37);
    const std::vector<uint16_t> dense_down_s =
        MakeScales(dim, dense_hidden, 37);
    std::vector<uint16_t> x(static_cast<size_t>(rows) * dim);
    for (int r = 0; r < rows; ++r) {
      for (int i = 0; i < dim; ++i) {
        x[static_cast<size_t>(r) * dim + i] = FloatToBf16Bits(
            static_cast<float>((i * 31 + r * 17) % 103 - 51) / 7.0f);
      }
    }
    std::vector<float> router_norm(dim), expert_norm(dim), dense_norm(dim),
        dense_post(dim), moe_post(dim), sum_post(dim), expert_scale(experts);
    for (int i = 0; i < dim; ++i) {
      router_norm[i] = 0.7f + 0.0005f * (i % 61);
      expert_norm[i] = 1.1f - 0.0004f * (i % 53);
      dense_norm[i] = 0.9f + 0.0003f * (i % 47);
      dense_post[i] = 1.2f - 0.0006f * (i % 43);
      moe_post[i] = 0.8f + 0.0004f * (i % 41);
      sum_post[i] = 1.05f - 0.0002f * (i % 37);
    }
    for (int e = 0; e < experts; ++e) expert_scale[e] = 0.98f + 0.0003f * e;
    const float eps = 1e-6f, router_multiplier = 0.6f,
                residual_multiplier = 1.25f;
    const auto upload = [](const void* data, size_t bytes) {
      auto device = AllocateCuda(bytes);
      cudaMemcpy(device.get(), data, bytes, cudaMemcpyHostToDevice);
      return device;
    };
    auto d_gate = upload(gate_w.data(), gate_w.size());
    auto d_up = upload(up_w.data(), up_w.size());
    auto d_down = upload(down_w.data(), down_w.size());
    auto d_gate_s = upload(gate_s.data(), gate_s.size() * 2);
    auto d_up_s = upload(up_s.data(), up_s.size() * 2);
    auto d_down_s = upload(down_s.data(), down_s.size() * 2);
    auto d_router = upload(router_w.data(), router_w.size());
    auto d_router_s = upload(router_s.data(), router_s.size() * 2);
    auto d_dense_gu = upload(dense_gu_w.data(), dense_gu_w.size());
    auto d_dense_gu_s = upload(dense_gu_s.data(), dense_gu_s.size() * 2);
    auto d_dense_down = upload(dense_down_w.data(), dense_down_w.size());
    auto d_dense_down_s = upload(dense_down_s.data(), dense_down_s.size() * 2);
    auto d_x = upload(x.data(), x.size() * 2);
    auto d_es = upload(expert_scale.data(), expert_scale.size() * 4);
    auto d_rn = upload(router_norm.data(), router_norm.size() * 4);
    auto d_en = upload(expert_norm.data(), expert_norm.size() * 4);
    auto d_dn = upload(dense_norm.data(), dense_norm.size() * 4);
    auto d_dp = upload(dense_post.data(), dense_post.size() * 4);
    auto d_mp = upload(moe_post.data(), moe_post.size() * 4);
    auto d_sp = upload(sum_post.data(), sum_post.size() * 4);
    const LiteRtNvidiaMoeArena arena = {
        static_cast<const uint8_t*>(d_gate.get()),
        static_cast<const uint8_t*>(d_up.get()),
        static_cast<const uint8_t*>(d_down.get()),
        d_gate_s.get(),
        d_up_s.get(),
        d_down_s.get(),
        static_cast<const uint8_t*>(d_router.get()),
        d_router_s.get(),
        static_cast<const uint8_t*>(d_dense_gu.get()),
        d_dense_gu_s.get(),
        static_cast<const uint8_t*>(d_dense_down.get()),
        d_dense_down_s.get()};
    const size_t out_bytes = static_cast<size_t>(rows) * dim * 2;
    // The experts on their own.
    auto d_ws_plain = AllocateCuda(LiteRtNvidiaMoeWorkspaceBytes(&shape));
    auto d_moe = AllocateCuda(out_bytes);
    ASSERT_EQ(
        LiteRtNvidiaLaunchMoeInt4BlockFused(
            &shape, &arena, d_x.get(), static_cast<const float*>(d_es.get()),
            static_cast<const float*>(d_rn.get()), router_multiplier,
            static_cast<const float*>(d_en.get()), eps, /*renormalize=*/1,
            d_ws_plain.get(), d_moe.get(), nullptr),
        cudaSuccess);
    // The dense feed-forward on its own, through the GEMV entries.
    auto d_dense_hidden =
        AllocateCuda(static_cast<size_t>(rows) * dense_hidden * 2);
    auto d_dense = AllocateCuda(out_bytes);
    for (int r0 = 0; r0 < rows; r0 += kLiteRtNvidiaInt4BlockGemvMaxRows) {
      const int n = std::min(kLiteRtNvidiaInt4BlockGemvMaxRows, rows - r0);
      ASSERT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRowsNormed(
                    static_cast<const char*>(d_x.get()) +
                        static_cast<size_t>(r0) * dim * 2,
                    arena.dense_gate_up, arena.dense_gate_up_scales,
                    static_cast<const float*>(d_dn.get()), eps, 1.0f,
                    static_cast<char*>(d_dense_hidden.get()) +
                        static_cast<size_t>(r0) * dense_hidden * 2,
                    n, dim, dense_hidden, /*gate=*/1, nullptr),
                cudaSuccess);
      ASSERT_EQ(LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
                    static_cast<const char*>(d_dense_hidden.get()) +
                        static_cast<size_t>(r0) * dense_hidden * 2,
                    arena.dense_down, arena.dense_down_scales,
                    static_cast<char*>(d_dense.get()) +
                        static_cast<size_t>(r0) * dim * 2,
                    n, dense_hidden, dim, /*gate=*/0, nullptr),
                cudaSuccess);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // The block, on streams of its own like the plugin's.
    cudaStream_t main_stream = nullptr, side_stream = nullptr;
    cudaEvent_t fork = nullptr, join = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking),
              cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&side_stream, cudaStreamNonBlocking),
              cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming),
              cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&join, cudaEventDisableTiming),
              cudaSuccess);
    const LiteRtNvidiaMoeBlockParams params = {
        dense_hidden,
        /*dense_gate=*/1,
        residual_multiplier,
        static_cast<const float*>(d_dn.get()),
        static_cast<const float*>(d_dp.get()),
        static_cast<const float*>(d_mp.get()),
        static_cast<const float*>(d_sp.get()),
        side_stream,
        fork,
        join};
    const size_t block_bytes =
        LiteRtNvidiaMoeBlockWorkspaceBytes(&shape, dense_hidden, 1);
    ASSERT_GT(block_bytes, LiteRtNvidiaMoeWorkspaceBytes(&shape));
    auto d_ws_block = AllocateCuda(block_bytes);
    auto d_y = AllocateCuda(out_bytes);
    ASSERT_EQ(
        LiteRtNvidiaLaunchMoeLayerBlock(
            &shape, &arena, d_x.get(), static_cast<const float*>(d_es.get()),
            static_cast<const float*>(d_rn.get()), router_multiplier,
            static_cast<const float*>(d_en.get()), eps, /*renormalize=*/1,
            &params, d_ws_block.get(), d_y.get(), main_stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(main_stream), cudaSuccess);
    std::vector<uint16_t> moe(x.size()), dense(x.size()), y(x.size());
    ASSERT_EQ(
        cudaMemcpy(moe.data(), d_moe.get(), out_bytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dense.data(), d_dense.get(), out_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(y.data(), d_y.get(), out_bytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    cudaEventDestroy(join);
    cudaEventDestroy(fork);
    cudaStreamDestroy(side_stream);
    cudaStreamDestroy(main_stream);
    // The norms and the residual update on the host. The prefill path's
    // dense feed-forward comes from the GEMM (FP16 partial sums) while the
    // reference's comes from the GEMV (FP32), so its tails are wider; the
    // RMS error over everything stays small either way.
    const double tail =
        rows > kLiteRtNvidiaInt4BlockGemvMaxRows ? 6e-2 : 2.5e-2;
    int mismatches = 0;
    double worst = 0.0;
    double squared = 0.0;
    for (int r = 0; r < rows; ++r) {
      const size_t base = static_cast<size_t>(r) * dim;
      double dense_sq = 0.0, moe_sq = 0.0;
      for (int o = 0; o < dim; ++o) {
        const double d = Bf16BitsToFloat(dense[base + o]);
        const double m = Bf16BitsToFloat(moe[base + o]);
        dense_sq += d * d;
        moe_sq += m * m;
      }
      const double dense_factor = 1.0 / std::sqrt(dense_sq / dim + eps);
      const double moe_factor = 1.0 / std::sqrt(moe_sq / dim + eps);
      std::vector<double> c(dim);
      double sum_sq = 0.0;
      for (int o = 0; o < dim; ++o) {
        c[o] = Bf16BitsToFloat(dense[base + o]) * dense_factor * dense_post[o] +
               Bf16BitsToFloat(moe[base + o]) * moe_factor * moe_post[o];
        sum_sq += c[o] * c[o];
      }
      const double sum_factor = 1.0 / std::sqrt(sum_sq / dim + eps);
      for (int o = 0; o < dim; ++o) {
        const double expected =
            (Bf16BitsToFloat(x[base + o]) + c[o] * sum_factor * sum_post[o]) *
            residual_multiplier;
        const double got = Bf16BitsToFloat(y[base + o]);
        const double error = std::fabs(got - expected);
        worst = std::max(worst, error);
        squared += error * error;
        if (error > tail + 1.5e-2 * std::fabs(expected)) ++mismatches;
      }
    }
    EXPECT_EQ(mismatches, 0) << "rows " << rows << " worst error " << worst;
    EXPECT_LT(std::sqrt(squared / x.size()), 1.5e-2) << "rows " << rows;
  }
}

// The compact expert scales (a byte per block as the fraction of the row's
// FP16 maximum) against the FP16 scales, decode and prefill paths, through
// the fused launch.
TEST(Int4BlockGemvTest, MoeCompactScalesMatchHalfScales) {
  for (const int rows : {3, 128}) {
    const LiteRtNvidiaMoeShape shape = {rows, 8, 128, 2816, 704, 1};
    const int experts = shape.experts, dim = shape.model_dim,
              hidden = shape.hidden_dim;
    const std::vector<uint8_t> gate_w =
        MakeInt4Weights(hidden * experts, dim, 5);
    const std::vector<uint8_t> up_w = MakeInt4Weights(hidden * experts, dim, 9);
    const std::vector<uint8_t> down_w =
        MakeInt4Weights(dim * experts, hidden, 13);
    const std::vector<uint16_t> gate_s = MakeScales(hidden * experts, dim, 5);
    const std::vector<uint16_t> up_s = MakeScales(hidden * experts, dim, 9);
    const std::vector<uint16_t> down_s = MakeScales(dim * experts, hidden, 13);
    const std::vector<uint8_t> router_w = MakeInt4Weights(experts, dim, 21);
    const std::vector<uint16_t> router_s = MakeScales(experts, dim, 21);
    std::vector<uint16_t> x(static_cast<size_t>(rows) * dim);
    for (int r = 0; r < rows; ++r) {
      for (int i = 0; i < dim; ++i) {
        x[static_cast<size_t>(r) * dim + i] = FloatToBf16Bits(
            static_cast<float>((i * 29 + r * 13) % 101 - 50) / 7.0f);
      }
    }
    std::vector<float> router_norm(dim), expert_norm(dim),
        expert_scale(experts);
    for (int i = 0; i < dim; ++i) {
      router_norm[i] = 0.7f + 0.0005f * (i % 61);
      expert_norm[i] = 1.1f - 0.0004f * (i % 53);
    }
    for (int e = 0; e < experts; ++e) expert_scale[e] = 0.98f + 0.0003f * e;
    // The compact form of each scale piece, as the plugin's loader makes it.
    auto compact = [](const std::vector<uint16_t>& halves, int row_blocks,
                      std::vector<uint8_t>* fractions,
                      std::vector<uint16_t>* maxima) {
      const size_t n_rows = halves.size() / row_blocks;
      fractions->resize(halves.size());
      maxima->resize(n_rows);
      std::vector<float> row(row_blocks);
      for (size_t r = 0; r < n_rows; ++r) {
        for (int b = 0; b < row_blocks; ++b)
          row[b] = HalfBitsToFloat(halves[r * row_blocks + b]);
        LiteRtNvidiaCompactScaleRow(row.data(), row_blocks,
                                    fractions->data() + r * row_blocks,
                                    maxima->data() + r);
      }
    };
    std::vector<uint8_t> gate_q, up_q, down_q;
    std::vector<uint16_t> gate_m, up_m, down_m;
    compact(gate_s, dim / 32, &gate_q, &gate_m);
    compact(up_s, dim / 32, &up_q, &up_m);
    compact(down_s, hidden / 32, &down_q, &down_m);
    const auto upload = [](const void* data, size_t bytes) {
      auto device = AllocateCuda(bytes);
      cudaMemcpy(device.get(), data, bytes, cudaMemcpyHostToDevice);
      return device;
    };
    auto d_gate = upload(gate_w.data(), gate_w.size());
    auto d_up = upload(up_w.data(), up_w.size());
    auto d_down = upload(down_w.data(), down_w.size());
    auto d_gate_s = upload(gate_s.data(), gate_s.size() * 2);
    auto d_up_s = upload(up_s.data(), up_s.size() * 2);
    auto d_down_s = upload(down_s.data(), down_s.size() * 2);
    auto d_gate_q = upload(gate_q.data(), gate_q.size());
    auto d_up_q = upload(up_q.data(), up_q.size());
    auto d_down_q = upload(down_q.data(), down_q.size());
    auto d_gate_m = upload(gate_m.data(), gate_m.size() * 2);
    auto d_up_m = upload(up_m.data(), up_m.size() * 2);
    auto d_down_m = upload(down_m.data(), down_m.size() * 2);
    auto d_router = upload(router_w.data(), router_w.size());
    auto d_router_s = upload(router_s.data(), router_s.size() * 2);
    auto d_x = upload(x.data(), x.size() * 2);
    auto d_es = upload(expert_scale.data(), expert_scale.size() * 4);
    auto d_rn = upload(router_norm.data(), router_norm.size() * 4);
    auto d_en = upload(expert_norm.data(), expert_norm.size() * 4);
    LiteRtNvidiaMoeArena half_arena = {
        static_cast<const uint8_t*>(d_gate.get()),
        static_cast<const uint8_t*>(d_up.get()),
        static_cast<const uint8_t*>(d_down.get()),
        d_gate_s.get(),
        d_up_s.get(),
        d_down_s.get(),
        static_cast<const uint8_t*>(d_router.get()),
        d_router_s.get()};
    LiteRtNvidiaMoeArena compact_arena = half_arena;
    compact_arena.gate_scales = d_gate_q.get();
    compact_arena.up_scales = d_up_q.get();
    compact_arena.down_scales = d_down_q.get();
    compact_arena.gate_scale_max = d_gate_m.get();
    compact_arena.up_scale_max = d_up_m.get();
    compact_arena.down_scale_max = d_down_m.get();
    compact_arena.compact_scales = 1;
    const size_t out_bytes = static_cast<size_t>(rows) * dim * 2;
    auto d_ws = AllocateCuda(LiteRtNvidiaMoeWorkspaceBytes(&shape));
    auto d_out_half = AllocateCuda(out_bytes);
    auto d_out_compact = AllocateCuda(out_bytes);
    for (const auto& [arena, out] :
         {std::make_pair(&half_arena, d_out_half.get()),
          std::make_pair(&compact_arena, d_out_compact.get())}) {
      ASSERT_EQ(
          LiteRtNvidiaLaunchMoeInt4BlockFused(
              &shape, arena, d_x.get(), static_cast<const float*>(d_es.get()),
              static_cast<const float*>(d_rn.get()), 0.6f,
              static_cast<const float*>(d_en.get()), 1e-6f, /*renormalize=*/1,
              d_ws.get(), out, nullptr),
          cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    std::vector<uint16_t> half_out(x.size()), compact_out(x.size());
    ASSERT_EQ(cudaMemcpy(half_out.data(), d_out_half.get(), out_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(compact_out.data(), d_out_compact.get(), out_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    double squared = 0.0, reference = 0.0, worst = 0.0;
    int mismatches = 0;
    for (size_t i = 0; i < half_out.size(); ++i) {
      const double a = Bf16BitsToFloat(half_out[i]),
                   b = Bf16BitsToFloat(compact_out[i]);
      const double error = std::fabs(a - b);
      squared += error * error;
      reference += a * a;
      worst = std::max(worst, error);
      if (error > 2e-2 + 2e-2 * std::fabs(a)) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0) << "rows " << rows << " worst " << worst;
    EXPECT_LT(std::sqrt(squared / std::max(reference, 1e-12)), 1e-2)
        << "rows " << rows;
  }
}

}  // namespace
