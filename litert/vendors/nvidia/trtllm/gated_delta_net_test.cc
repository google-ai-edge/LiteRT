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

// The fused gated DeltaNet layer against the model's ops in double precision:
// the short convolution, the q/k norms, the gates, the recurrent update and
// the gated norm, for decode, for a prefill chunk with padding past its rows,
// and with the states updated in place.

#include "litert/vendors/nvidia/trtllm/gated_delta_net.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

template <typename T>
CudaAllocation Upload(const std::vector<T>& values) {
  void* pointer = nullptr;
  EXPECT_EQ(
      cudaMalloc(&pointer, std::max<size_t>(values.size(), 1) * sizeof(T)),
      cudaSuccess);
  if (!values.empty()) {
    EXPECT_EQ(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
  return CudaAllocation(pointer);
}

template <typename T>
std::vector<T> Download(const void* pointer, size_t count) {
  std::vector<T> values(count);
  EXPECT_EQ(cudaMemcpy(values.data(), pointer, count * sizeof(T),
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

uint16_t ToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

double FromBf16(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

double Silu(double x) { return x / (1.0 + std::exp(-x)); }

struct Layer {
  LiteRtNvidiaGatedDeltaNetParams p;
  std::vector<uint16_t> proj;  // BF16 [tokens][proj_stride]
  std::vector<float> conv_state, state, constants;
};

// Qwen 3.5's head size with fewer heads; the constants as the model's: q/k
// norm weights 1/D and 1/sqrt(D), negative decays.
Layer MakeLayer(int tokens, int key_heads, int value_heads, uint32_t seed) {
  Layer layer;
  auto& p = layer.p;
  const int d = 128;
  const int channels = (2 * key_heads + value_heads) * d;
  const int values = value_heads * d;
  // Projections in the model's order within a GEMV group: qkv, b, a, z.
  p = {tokens,
       key_heads,
       value_heads,
       d,
       4,
       1,
       channels + 2 * value_heads + values + 8,
       /*qkv_offset=*/0,
       /*z_offset=*/channels + 2 * value_heads + 8,
       /*a_offset=*/channels + value_heads,
       /*b_offset=*/channels,
       /*q_eps=*/1e-6f / 128,
       /*k_eps=*/1e-6f / 128,
       /*out_eps=*/1e-6f};
  const auto raw =
      Fill(static_cast<size_t>(tokens) * p.proj_stride, seed, -2.0f, 2.0f);
  layer.proj.resize(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) layer.proj[i] = ToBf16(raw[i]);
  layer.conv_state =
      Fill(static_cast<size_t>(channels) * 3, seed + 1, -1.0f, 1.0f);
  layer.state =
      Fill(static_cast<size_t>(value_heads) * d * d, seed + 2, -0.2f, 0.2f);
  const auto decay = Fill(value_heads, seed + 3, -2.0f, -0.01f);
  const auto norm_weight = Fill(d, seed + 4, 0.6f, 1.1f);
  const auto conv_weight =
      Fill(static_cast<size_t>(channels) * 4, seed + 5, -0.5f, 0.5f);
  layer.constants = decay;
  for (int i = 0; i < d; ++i) layer.constants.push_back(1.0f / 128.0f);
  for (int i = 0; i < d; ++i) layer.constants.push_back(0.08838834764831845f);
  layer.constants.insert(layer.constants.end(), norm_weight.begin(),
                         norm_weight.end());
  layer.constants.insert(layer.constants.end(), conv_weight.begin(),
                         conv_weight.end());
  // The projections' biases: qkv, z, a (Qwen's dt bias, the one it has), b.
  const auto biases =
      Fill(static_cast<size_t>(channels) + values + 2 * value_heads, seed + 6,
           -3.0f, 3.0f);
  layer.constants.insert(layer.constants.end(), biases.begin(), biases.end());
  EXPECT_EQ(layer.constants.size(), LiteRtNvidiaGatedDeltaNetConstantCount(&p));
  return layer;
}

struct Result {
  std::vector<double> out;  // [tokens][value_heads * D]
  std::vector<double> conv_state, state;
};

Result Reference(const Layer& layer, std::optional<int32_t> valid_len) {
  const auto& p = layer.p;
  const int d = p.head_dim, hk = p.key_heads, hv = p.value_heads;
  const int channels = (2 * hk + hv) * d;
  const int n = p.tokens;
  const int valid = valid_len.has_value() ? std::clamp(*valid_len, 0, n) : n;
  const double* decay = nullptr;
  std::vector<double> c(layer.constants.begin(), layer.constants.end());
  decay = c.data();
  const double* q_scale = decay + hv;
  const double* k_scale = q_scale + d;
  const double* norm_weight = k_scale + d;
  const double* conv_weight = norm_weight + d;
  const double* qkv_bias = conv_weight + static_cast<size_t>(channels) * 4;
  const double* z_bias = qkv_bias + channels;
  const double* a_bias = z_bias + hv * d;
  const double* b_bias = a_bias + hv;
  const auto proj = [&](int t, int column) {
    return FromBf16(
        layer.proj[static_cast<size_t>(t) * p.proj_stride + column]);
  };
  const auto xx = [&](int ch, int position) {
    return position < 3
               ? static_cast<double>(layer.conv_state[ch * 3 + position])
               : proj(position - 3, p.qkv_offset + ch) + qkv_bias[ch];
  };
  Result result;
  result.out.assign(static_cast<size_t>(n) * hv * d, 0.0);
  result.conv_state.resize(layer.conv_state.size());
  for (int ch = 0; ch < channels; ++ch) {
    for (int j = 0; j < 3; ++j)
      result.conv_state[ch * 3 + j] = xx(ch, valid + j);
  }
  result.state.assign(layer.state.begin(), layer.state.end());
  std::vector<double> y(channels), memory(d), o(d);
  for (int t = 0; t < valid; ++t) {
    for (int ch = 0; ch < channels; ++ch) {
      double acc = 0.0;
      for (int j = 0; j < 4; ++j)
        acc += conv_weight[ch * 4 + j] * xx(ch, t + j);
      y[ch] = Silu(acc);
    }
    const auto normed = [&](int offset, const double* scale, double eps,
                            std::vector<double>& head) {
      double squares = 0.0;
      for (int i = 0; i < d; ++i) squares += y[offset + i] * y[offset + i];
      const double factor = 1.0 / std::sqrt(squares / d + eps);
      head.resize(d);
      for (int i = 0; i < d; ++i) head[i] = y[offset + i] * factor * scale[i];
    };
    for (int h = 0; h < hv; ++h) {
      const int key = h / (hv / hk);
      std::vector<double> q, k;
      normed(key * d, q_scale, p.q_eps, q);
      normed((hk + key) * d, k_scale, p.k_eps, k);
      const double* v = &y[(2 * hk + h) * d];
      const double beta =
          1.0 / (1.0 + std::exp(-(proj(t, p.b_offset + h) + b_bias[h])));
      const double g =
          decay[h] *
          std::log(1.0 + std::exp(std::min(proj(t, p.a_offset + h) + a_bias[h],
                                           50.0)));
      double* s = &result.state[static_cast<size_t>(h) * d * d];
      for (int i = 0; i < d * d; ++i) s[i] *= std::exp(g);
      for (int j = 0; j < d; ++j) {
        double acc = 0.0;
        for (int i = 0; i < d; ++i) acc += s[i * d + j] * k[i];
        memory[j] = acc;
      }
      for (int j = 0; j < d; ++j) {
        const double delta = (v[j] - memory[j]) * beta;
        for (int i = 0; i < d; ++i) s[i * d + j] += k[i] * delta;
      }
      double squares = 0.0;
      for (int j = 0; j < d; ++j) {
        double acc = 0.0;
        for (int i = 0; i < d; ++i) acc += s[i * d + j] * q[i];
        o[j] = acc;
        squares += acc * acc;
      }
      const double factor = 1.0 / std::sqrt(squares / d + p.out_eps);
      for (int j = 0; j < d; ++j) {
        result.out[(static_cast<size_t>(t) * hv + h) * d + j] =
            o[j] * factor * norm_weight[j] *
            Silu(proj(t, p.z_offset + h * d + j) + z_bias[h * d + j]);
      }
    }
  }
  return result;
}

// The RMS of got - want relative to the RMS of want.
double RelativeError(const std::vector<double>& got,
                     const std::vector<double>& want) {
  double error = 0.0, norm = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    error += (got[i] - want[i]) * (got[i] - want[i]);
    norm += want[i] * want[i];
  }
  return norm == 0.0 ? std::sqrt(error) : std::sqrt(error / norm);
}

void Check(const Layer& layer, std::optional<int32_t> valid_len,
           bool in_place) {
  const auto& p = layer.p;
  SCOPED_TRACE(::testing::Message()
               << "tokens=" << p.tokens << " valid="
               << (valid_len ? *valid_len : -1) << " in_place=" << in_place);
  ASSERT_TRUE(LiteRtNvidiaGatedDeltaNetSupports(&p));
  const Result want = Reference(layer, valid_len);
  auto proj = Upload(layer.proj);
  auto conv_state = Upload(layer.conv_state);
  auto state = Upload(layer.state);
  auto constants = Upload(layer.constants);
  auto valid = Upload(std::vector<int32_t>{valid_len.value_or(0)});
  const size_t outputs =
      static_cast<size_t>(p.tokens) * p.value_heads * p.head_dim;
  auto out =
      Upload(std::vector<uint16_t>(outputs, 0x7fc0));  // NaN until written
  auto new_conv_state = in_place ? CudaAllocation() : Upload(layer.conv_state);
  auto new_state = in_place ? CudaAllocation() : Upload(layer.state);
  float* conv_target =
      static_cast<float*>(in_place ? conv_state.get() : new_conv_state.get());
  float* state_target =
      static_cast<float*>(in_place ? state.get() : new_state.get());
  auto workspace =
      Upload(std::vector<uint8_t>(LiteRtNvidiaGatedDeltaNetWorkspaceBytes(&p)));
  ASSERT_EQ(LiteRtNvidiaLaunchGatedDeltaNet(
                &p, proj.get(), static_cast<const float*>(conv_state.get()),
                static_cast<const float*>(state.get()),
                static_cast<const float*>(constants.get()),
                valid_len ? static_cast<const int32_t*>(valid.get()) : nullptr,
                out.get(), conv_target, state_target, workspace.get(), nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto got_out = Download<uint16_t>(out.get(), outputs);
  const auto got_conv = Download<float>(conv_target, layer.conv_state.size());
  const auto got_state = Download<float>(state_target, layer.state.size());
  std::vector<double> out_values(outputs), conv_values(got_conv.size()),
      state_values(got_state.size());
  for (size_t i = 0; i < outputs; ++i) out_values[i] = FromBf16(got_out[i]);
  for (size_t i = 0; i < got_conv.size(); ++i) conv_values[i] = got_conv[i];
  for (size_t i = 0; i < got_state.size(); ++i) state_values[i] = got_state[i];
  // The BF16 output rounding is about 2e-3 of the RMS; the FP32 math adds
  // little on top. Padding rows come out exactly zero.
  EXPECT_LT(RelativeError(out_values, want.out), 4e-3);
  const int rows = valid_len ? std::clamp(*valid_len, 0, p.tokens) : p.tokens;
  for (size_t i = static_cast<size_t>(rows) * p.value_heads * p.head_dim;
       i < outputs; ++i) {
    ASSERT_EQ(out_values[i], 0.0) << "padding output " << i;
  }
  // The convolution state holds the biased inputs, each rounded once to FP32.
  EXPECT_LT(RelativeError(conv_values, want.conv_state), 1e-7);
  EXPECT_LT(RelativeError(state_values, want.state), 1e-5);
}

TEST(GatedDeltaNetTest, DecodeStep) {
  Check(MakeLayer(/*tokens=*/1, /*key_heads=*/2, /*value_heads=*/6, 7),
        std::nullopt,
        /*in_place=*/false);
}

TEST(GatedDeltaNetTest, DecodeStepInPlace) {
  Check(MakeLayer(1, 2, 6, 11), std::nullopt, /*in_place=*/true);
}

TEST(GatedDeltaNetTest, PrefillChunk) {
  Check(MakeLayer(/*tokens=*/40, 2, 6, 19), std::nullopt, false);
}

// LiteRT-LM pads a chunk past its rows: they neither advance the states nor
// produce outputs.
TEST(GatedDeltaNetTest, PrefillChunkWithPadding) {
  Check(MakeLayer(48, 2, 6, 23), /*valid_len=*/33, /*in_place=*/true);
  Check(MakeLayer(48, 2, 6, 29), /*valid_len=*/2,
        false);  // fewer rows than taps
  Check(MakeLayer(16, 1, 3, 31), /*valid_len=*/0, false);
}

TEST(GatedDeltaNetTest, Supports) {
  Layer layer = MakeLayer(4, 2, 6, 3);
  EXPECT_TRUE(LiteRtNvidiaGatedDeltaNetSupports(&layer.p));
  auto p = layer.p;
  p.head_dim = 64;
  EXPECT_FALSE(LiteRtNvidiaGatedDeltaNetSupports(&p));
  p = layer.p;
  p.value_heads = 5;  // not a multiple of the key heads
  EXPECT_FALSE(LiteRtNvidiaGatedDeltaNetSupports(&p));
  p = layer.p;
  p.z_offset = p.proj_stride - 10;  // past the row
  EXPECT_FALSE(LiteRtNvidiaGatedDeltaNetSupports(&p));
}

}  // namespace
