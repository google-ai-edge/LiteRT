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

// Qwen 4's per-layer embedding past its projections (the key and query
// norms, the sign-sqrt-sigmoid gate, the gated value, its norm and the
// dilated convolution with its state) against the model's ops in double
// precision: decode and a few rows, the state updated in place and apart.

#include "litert/vendors/nvidia/trtllm/ple_tail.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
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

std::vector<uint16_t> Bf16(const std::vector<float>& values) {
  std::vector<uint16_t> bits(values.size());
  for (size_t i = 0; i < values.size(); ++i) bits[i] = ToBf16(values[i]);
  return bits;
}

struct Reference {
  std::vector<double> out, state;
};

// The model's ops, from the BF16 inputs.
Reference Tail(const LiteRtNvidiaPleTailParams& p,
               const std::vector<uint16_t>& key,
               const std::vector<uint16_t>& value,
               const std::vector<uint16_t>& x, const std::vector<float>& state,
               const std::vector<float>& constants) {
  const int width = p.streams * p.hidden;
  const int history = p.taps - 1;
  const float* key_weight = constants.data();
  const float* query_weight = key_weight + width;
  const float* conv_weight = query_weight + width;
  const float* taps = conv_weight + width;
  Reference ref;
  ref.out.resize(static_cast<size_t>(p.rows) * width);
  ref.state.resize(static_cast<size_t>(width) * history);
  // The convolution's inputs over the rows, per channel.
  std::vector<double> inputs(static_cast<size_t>(p.rows) * width);
  std::vector<double> gated(static_cast<size_t>(p.rows) * width);
  for (int r = 0; r < p.rows; ++r) {
    double value_squares = 0.0;
    for (int h = 0; h < p.hidden; ++h) {
      const double v = FromBf16(value[static_cast<size_t>(r) * p.hidden + h]);
      value_squares += v * v;
    }
    for (int s = 0; s < p.streams; ++s) {
      const size_t at = static_cast<size_t>(r) * width + s * p.hidden;
      double key_squares = 0.0, x_squares = 0.0;
      for (int h = 0; h < p.hidden; ++h) {
        key_squares += FromBf16(key[at + h]) * FromBf16(key[at + h]);
        x_squares += FromBf16(x[at + h]) * FromBf16(x[at + h]);
      }
      const double key_factor =
          1.0 / std::sqrt(key_squares / p.hidden + p.eps_key);
      const double x_factor =
          1.0 / std::sqrt(x_squares / p.hidden + p.eps_query);
      double dot = 0.0;
      for (int h = 0; h < p.hidden; ++h) {
        dot += FromBf16(key[at + h]) * key_factor *
               key_weight[s * p.hidden + h] * FromBf16(x[at + h]) * x_factor *
               query_weight[s * p.hidden + h];
      }
      const double g = dot * p.dot_scale;
      const double sign = g > 0 ? 1.0 : (g < 0 ? -1.0 : 0.0);
      const double gate =
          1.0 / (1.0 + std::exp(-sign * std::sqrt(std::max(
                                            std::fabs(g),
                                            static_cast<double>(p.floor)))));
      const double factor =
          1.0 / std::sqrt(gate * gate * value_squares / p.hidden + p.eps_conv);
      for (int h = 0; h < p.hidden; ++h) {
        const double gv =
            gate * FromBf16(value[static_cast<size_t>(r) * p.hidden + h]);
        gated[at + h] = gv;
        inputs[at + h] = gv * factor * conv_weight[s * p.hidden + h];
      }
    }
  }
  for (int c = 0; c < width; ++c) {
    // state ++ inputs along the rows.
    std::vector<double> window(history + p.rows);
    for (int j = 0; j < history; ++j) {
      window[j] = state[static_cast<size_t>(c) * history + j];
    }
    for (int r = 0; r < p.rows; ++r) {
      window[history + r] = inputs[static_cast<size_t>(r) * width + c];
    }
    for (int r = 0; r < p.rows; ++r) {
      double acc = 0.0;
      for (int j = 0; j < p.taps; ++j) {
        acc += taps[static_cast<size_t>(c) * p.taps + j] * window[r + j];
      }
      if (p.silu != 0) acc = acc / (1.0 + std::exp(-acc));
      const size_t at = static_cast<size_t>(r) * width + c;
      ref.out[at] = FromBf16(x[at]) + gated[at] + acc;
    }
    for (int j = 0; j < history; ++j) {
      ref.state[static_cast<size_t>(c) * history + j] = window[p.rows + j];
    }
  }
  return ref;
}

// The RMS of got - want relative to the RMS of want.
template <typename T>
double RelativeError(const std::vector<T>& got,
                     const std::vector<double>& want) {
  double error = 0.0, norm = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    error += d * d;
    norm += want[i] * want[i];
  }
  return std::sqrt(error / norm);
}

void Check(int rows, int streams, int hidden, int taps, bool silu,
           bool time = false) {
  const LiteRtNvidiaPleTailParams p{
      rows,  streams,     hidden, taps,
      1e-6f, 1e-6f,       1e-6f,  1.0f / std::sqrt(static_cast<float>(hidden)),
      1e-6f, silu ? 1 : 0};
  SCOPED_TRACE(::testing::Message()
               << "rows=" << rows << " streams=" << streams
               << " hidden=" << hidden << " taps=" << taps << " silu=" << silu);
  ASSERT_TRUE(LiteRtNvidiaPleTailSupports(&p));
  const int width = streams * hidden;
  const int history = taps - 1;
  const auto key = Bf16(Fill(static_cast<size_t>(rows) * width, 3, -2, 2));
  const auto value = Bf16(Fill(static_cast<size_t>(rows) * hidden, 5, -1, 1));
  const auto x = Bf16(Fill(static_cast<size_t>(rows) * width, 7, -3, 3));
  const auto state = Fill(static_cast<size_t>(width) * history, 9, -1, 1);
  // The norm weights around one, the taps small.
  auto constants = Fill(3 * static_cast<size_t>(width), 11, 0.6f, 1.4f);
  const auto tap_values =
      Fill(static_cast<size_t>(width) * taps, 13, -0.5f, 0.5f);
  constants.insert(constants.end(), tap_values.begin(), tap_values.end());
  ASSERT_EQ(constants.size(), LiteRtNvidiaPleTailConstantCount(&p));
  auto d_key = Upload(key);
  auto d_value = Upload(value);
  auto d_x = Upload(x);
  auto d_constants = Upload(constants);
  const Reference want = Tail(p, key, value, x, state, constants);
  std::vector<uint16_t> first;
  for (const bool in_place : {false, true}) {
    auto d_state = Upload(state);
    auto d_new_state = Upload(std::vector<float>(state.size(), NAN));
    auto d_out = Upload(std::vector<uint16_t>(want.out.size(), 0x7fc0));
    void* new_state = in_place ? d_state.get() : d_new_state.get();
    ASSERT_EQ(LiteRtNvidiaLaunchPleTail(
                  &p, static_cast<const __nv_bfloat16*>(d_key.get()),
                  static_cast<const __nv_bfloat16*>(d_value.get()),
                  static_cast<const __nv_bfloat16*>(d_x.get()),
                  static_cast<const float*>(d_state.get()),
                  static_cast<const float*>(d_constants.get()),
                  static_cast<__nv_bfloat16*>(d_out.get()),
                  static_cast<float*>(new_state), nullptr),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto out_bits = Download<uint16_t>(d_out.get(), want.out.size());
    std::vector<double> out(out_bits.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = FromBf16(out_bits[i]);
    // One BF16 rounding of the output; the state is FP32.
    EXPECT_LT(RelativeError(out, want.out), 4e-3) << "in place " << in_place;
    EXPECT_LT(RelativeError(Download<float>(new_state, want.state.size()),
                            want.state),
              1e-5)
        << "in place " << in_place;
    if (in_place) {
      EXPECT_EQ(out_bits, first);
    } else {
      first = out_bits;
    }
  }
  if (!time) return;
  auto d_state = Upload(state);
  auto d_out = Upload(std::vector<uint16_t>(want.out.size()));
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  const auto launch = [&] {
    return LiteRtNvidiaLaunchPleTail(
        &p, static_cast<const __nv_bfloat16*>(d_key.get()),
        static_cast<const __nv_bfloat16*>(d_value.get()),
        static_cast<const __nv_bfloat16*>(d_x.get()),
        static_cast<const float*>(d_state.get()),
        static_cast<const float*>(d_constants.get()),
        static_cast<__nv_bfloat16*>(d_out.get()),
        static_cast<float*>(d_state.get()), stream);
  };
  // In a CUDA graph, as an engine runs it.
  constexpr int kCalls = 20;
  cudaGraph_t graph = nullptr;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < kCalls; ++i) launch();
  cudaStreamEndCapture(stream, &graph);
  cudaGraphExec_t exec = nullptr;
  cudaGraphInstantiate(&exec, graph, 0);
  for (int i = 0; i < 5; ++i) cudaGraphLaunch(exec, stream);
  cudaStreamSynchronize(stream);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int kIterations = 20;
  cudaEventRecord(start, stream);
  for (int i = 0; i < kIterations; ++i) cudaGraphLaunch(exec, stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf("ple tail rows %d [%d x %d] taps %d: graph %.1f us\n", rows,
              streams, hidden, taps, ms * 1000.0f / (kIterations * kCalls));
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
}

TEST(PleTailTest, MatchesReference) {
  // Qwen 4's widths (hidden 2560, hc 4, the 4 taps at dilation 3 as 10) and
  // the reduced model's; a verify step's rows; a plain 4-tap convolution.
  Check(1, 4, 2560, 10, true);
  Check(4, 4, 2560, 10, true);
  Check(1, 4, 256, 10, true);
  Check(8, 4, 256, 10, true);
  Check(3, 2, 512, 4, false);
  Check(1, 1, 8, 2, true);
  Check(2, 8, 8192, 16, true);
}

TEST(PleTailTest, TimesTheTail) {
  if (std::getenv("PLE_TAIL_TIMING") == nullptr) {
    GTEST_SKIP() << "Set PLE_TAIL_TIMING=1.";
  }
  Check(1, 4, 2560, 10, true, /*time=*/true);
}

}  // namespace
