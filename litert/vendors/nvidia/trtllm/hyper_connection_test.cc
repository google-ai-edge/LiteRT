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

// Qwen 4's fused hyper-connection mix against the model's ops in double
// precision (the grouped norm, the low-rank down projection, silu, the up
// projection, the sigmoid gates and the mean over the streams), with BF16 and
// INT4 block weights, for decode and for a verify step's rows.

#include "litert/vendors/nvidia/trtllm/hyper_connection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "cuda_fp16.h"
#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"

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

uint16_t ToHalf(float value) {
  return static_cast<uint16_t>(__half_as_ushort(__float2half_rn(value)));
}

// A [rows, columns] matrix as INT4 blocks of 32 (packed nibbles, low first;
// a half scale a block), and its dequantized values (`values` is replaced).
struct Int4 {
  std::vector<uint8_t> packed;
  std::vector<uint16_t> scales;
};

Int4 QuantizeInt4(std::vector<float>& values, int rows, int columns) {
  Int4 q;
  q.packed.assign(static_cast<size_t>(rows) * columns / 2, 0);
  q.scales.resize(static_cast<size_t>(rows) * columns / 32);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < columns / 32; ++b) {
      float* block = &values[static_cast<size_t>(r) * columns + b * 32];
      float largest = 0.0f;
      for (int i = 0; i < 32; ++i)
        largest = std::max(largest, std::fabs(block[i]));
      const uint16_t half = ToHalf(largest > 0.0f ? largest / 7.0f : 1.0f);
      const float scale = __half2float(__ushort_as_half(half));
      q.scales[static_cast<size_t>(r) * (columns / 32) + b] = half;
      for (int i = 0; i < 32; ++i) {
        const int n =
            std::clamp(static_cast<int>(std::lround(block[i] / scale)), -8, 7);
        block[i] = n * scale;
        const size_t at = (static_cast<size_t>(r) * columns + b * 32 + i) / 2;
        q.packed[at] |= static_cast<uint8_t>((n & 0xf) << ((i % 2) * 4));
      }
    }
  }
  return q;
}

// Qwen 4's hyper-connection mix at its widths in double precision, from the
// BF16 inputs: normed, then mixed.
struct Reference {
  std::vector<double> normed, mixed, extra;
};

Reference Mix(const LiteRtNvidiaHcMixParams& p, const std::vector<uint16_t>& x,
              const std::vector<float>& norm, const std::vector<float>& down,
              const std::vector<float>& up, const std::vector<float>& more) {
  const int width = p.streams * p.hidden;
  Reference ref;
  ref.normed.resize(static_cast<size_t>(p.rows) * width);
  ref.mixed.assign(static_cast<size_t>(p.rows) * p.hidden, 0.0);
  for (int r = 0; r < p.rows; ++r) {
    double* xn = &ref.normed[static_cast<size_t>(r) * width];
    for (int s = 0; s < p.streams; ++s) {
      double squares = 0.0;
      for (int h = 0; h < p.hidden; ++h) {
        const double v =
            FromBf16(x[static_cast<size_t>(r) * width + s * p.hidden + h]);
        squares += v * v;
      }
      const double factor = 1.0 / std::sqrt(squares / p.hidden + p.eps);
      for (int h = 0; h < p.hidden; ++h) {
        const int k = s * p.hidden + h;
        // The normed streams as the kernel holds them (BF16).
        xn[k] = FromBf16(ToBf16(
            static_cast<float>(FromBf16(x[static_cast<size_t>(r) * width + k]) *
                               factor * norm[k])));
      }
    }
    for (int e = 0; e < p.extra; ++e) {
      double dot = 0.0;
      for (int k = 0; k < width; ++k) {
        dot += more[static_cast<size_t>(e) * width + k] * xn[k];
      }
      ref.extra.push_back(dot);
    }
    std::vector<double> t(p.low_rank);
    for (int j = 0; j < p.low_rank; ++j) {
      double dot = 0.0;
      for (int k = 0; k < width; ++k) {
        dot += down[static_cast<size_t>(j) * width + k] * xn[k];
      }
      const double d = dot * p.down_scale;
      t[j] = d / (1.0 + std::exp(-d));
    }
    for (int h = 0; h < p.hidden; ++h) {
      double total = 0.0;
      for (int s = 0; s < p.streams; ++s) {
        const int row = s * p.hidden + h;
        double dot = 0.0;
        for (int j = 0; j < p.low_rank; ++j) {
          dot += up[static_cast<size_t>(row) * p.low_rank + j] * t[j];
        }
        total += xn[row] / (1.0 + std::exp(-dot));
      }
      ref.mixed[static_cast<size_t>(r) * p.hidden + h] = total / p.streams;
    }
  }
  return ref;
}

// The RMS of got - want relative to the RMS of want.
double RelativeError(const std::vector<uint16_t>& got,
                     const std::vector<double>& want) {
  double error = 0.0, norm = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = FromBf16(got[i]) - want[i];
    error += d * d;
    norm += want[i] * want[i];
  }
  return std::sqrt(error / norm);
}

// Microseconds per call: stream launches, and the same calls in a CUDA graph
// (as TensorRT-RTX runs an engine).
struct Timing {
  float stream_us;
  float graph_us;
};

Timing Time(const std::function<cudaError_t(cudaStream_t)>& launch) {
  cudaStream_t stream = nullptr;
  EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  for (int i = 0; i < 200; ++i) EXPECT_EQ(launch(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int kIterations = 200;
  cudaEventRecord(start, stream);
  for (int i = 0; i < kIterations; ++i) launch(stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  constexpr int kGraphCalls = 20;
  cudaGraph_t graph = nullptr;
  EXPECT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            cudaSuccess);
  for (int i = 0; i < kGraphCalls; ++i) launch(stream);
  EXPECT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
  cudaGraphExec_t exec = nullptr;
  EXPECT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
  for (int i = 0; i < 5; ++i) cudaGraphLaunch(exec, stream);
  cudaStreamSynchronize(stream);
  constexpr int kGraphIterations = 20;
  cudaEventRecord(start, stream);
  for (int i = 0; i < kGraphIterations; ++i) cudaGraphLaunch(exec, stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float graph_ms = 0.0f;
  cudaEventElapsedTime(&graph_ms, start, stop);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  return {ms * 1000.0f / kIterations,
          graph_ms * 1000.0f / (kGraphIterations * kGraphCalls)};
}

void Check(int rows, int hidden, int low_rank, bool int4, bool time,
           int extra = 0) {
  const LiteRtNvidiaHcMixParams p{rows,  4,     hidden,       low_rank,
                                  0.25f, 1e-6f, int4 ? 1 : 0, extra};
  SCOPED_TRACE(::testing::Message()
               << "rows=" << rows << " hidden=" << hidden
               << " low_rank=" << low_rank << " extra=" << extra);
  ASSERT_TRUE(LiteRtNvidiaHcMixSupports(&p));
  const int width = p.streams * hidden;
  const auto to_bf16 = [](const std::vector<float>& values) {
    std::vector<uint16_t> bits(values.size());
    for (size_t i = 0; i < values.size(); ++i) bits[i] = ToBf16(values[i]);
    return bits;
  };
  const auto x = to_bf16(Fill(static_cast<size_t>(rows) * width, 3, -2, 2));
  const auto norm = Fill(width, 5, 0.8f, 1.2f);
  const float down_gain = 2.0f / std::sqrt(static_cast<float>(width));
  const float up_gain = 2.0f / std::sqrt(static_cast<float>(low_rank));
  auto down =
      Fill(static_cast<size_t>(low_rank) * width, 7, -down_gain, down_gain);
  auto up = Fill(static_cast<size_t>(width) * low_rank, 11, -up_gain, up_gain);
  // The extra projection: a few rows over the normed streams.
  auto more =
      Fill(static_cast<size_t>(extra) * width, 13, -down_gain, down_gain);
  CudaAllocation d_down, d_up, d_down_scales, d_up_scales, d_more,
      d_more_scales;
  size_t weight_bytes = 0;
  if (int4) {
    const Int4 qd = QuantizeInt4(down, low_rank, width);
    const Int4 qu = QuantizeInt4(up, width, low_rank);
    const Int4 qm = QuantizeInt4(more, extra, width);
    d_down = Upload(qd.packed);
    d_up = Upload(qu.packed);
    d_more = Upload(qm.packed);
    d_down_scales = Upload(qd.scales);
    d_up_scales = Upload(qu.scales);
    d_more_scales = Upload(qm.scales);
    weight_bytes = qd.packed.size() + qu.packed.size() +
                   2 * (qd.scales.size() + qu.scales.size());
  } else {
    const auto down_bits = to_bf16(down);
    const auto up_bits = to_bf16(up);
    const auto more_bits = to_bf16(more);
    for (size_t i = 0; i < down.size(); ++i) down[i] = FromBf16(down_bits[i]);
    for (size_t i = 0; i < up.size(); ++i) up[i] = FromBf16(up_bits[i]);
    for (size_t i = 0; i < more.size(); ++i) more[i] = FromBf16(more_bits[i]);
    d_down = Upload(down_bits);
    d_up = Upload(up_bits);
    d_more = Upload(more_bits);
    weight_bytes = 2 * (down.size() + up.size());
  }
  auto d_x = Upload(x);
  auto d_norm = Upload(norm);
  auto d_mixed = Upload(std::vector<uint16_t>(
      static_cast<size_t>(rows) * hidden, 0x7fc0));  // NaN until written
  auto d_normed =
      Upload(std::vector<uint16_t>(static_cast<size_t>(rows) * width, 0x7fc0));
  auto d_extra =
      Upload(std::vector<uint16_t>(static_cast<size_t>(rows) * extra, 0x7fc0));
  // NaNs where the down projection's pieces go: a CTA past the barrier
  // before every piece is written shows.
  auto d_workspace =
      Upload(std::vector<uint8_t>(LiteRtNvidiaHcMixWorkspaceBytes(&p), 0xff));
  auto d_state = Upload(std::vector<uint8_t>(LiteRtNvidiaHcMixStateBytes()));
  void* state = d_state.get();
  const auto launch = [&](cudaStream_t stream) {
    return LiteRtNvidiaLaunchHcMixExtra(
        &p, static_cast<const __nv_bfloat16*>(d_x.get()),
        static_cast<const float*>(d_norm.get()), d_down.get(),
        static_cast<const __half*>(d_down_scales.get()), d_up.get(),
        static_cast<const __half*>(d_up_scales.get()), d_more.get(),
        static_cast<const __half*>(d_more_scales.get()),
        static_cast<__nv_bfloat16*>(d_mixed.get()),
        static_cast<__nv_bfloat16*>(d_normed.get()),
        static_cast<__nv_bfloat16*>(d_extra.get()), state, d_workspace.get(),
        stream);
  };
  ASSERT_EQ(launch(nullptr), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const Reference want = Mix(p, x, norm, down, up, more);
  const auto normed = Download<uint16_t>(d_normed.get(), want.normed.size());
  const auto mixed = Download<uint16_t>(d_mixed.get(), want.mixed.size());
  const auto extras = Download<uint16_t>(d_extra.get(), want.extra.size());
  // One BF16 rounding of each output, about 2e-3 of the RMS.
  EXPECT_LT(RelativeError(normed, want.normed), 4e-3);
  EXPECT_LT(RelativeError(mixed, want.mixed), 4e-3);
  if (extra > 0) EXPECT_LT(RelativeError(extras, want.extra), 4e-3);
  // The barrier's state is zero again, and a second launch gives the same
  // bits; so do the two launches without a barrier.
  EXPECT_EQ(Download<uint32_t>(d_state.get(), 2), std::vector<uint32_t>(2, 0));
  for (const bool barrier : {true, false}) {
    const std::vector<uint16_t> poison(want.normed.size(), 0x7fc0);
    ASSERT_EQ(cudaMemcpy(d_normed.get(), poison.data(), poison.size() * 2,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_mixed.get(), poison.data(), mixed.size() * 2,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    state = barrier ? d_state.get() : nullptr;
    ASSERT_EQ(cudaMemset(d_workspace.get(), 0xff,
                         LiteRtNvidiaHcMixWorkspaceBytes(&p)),
              cudaSuccess);
    ASSERT_EQ(launch(nullptr), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(Download<uint16_t>(d_normed.get(), normed.size()), normed)
        << "barrier " << barrier;
    EXPECT_EQ(Download<uint16_t>(d_mixed.get(), mixed.size()), mixed)
        << "barrier " << barrier;
    EXPECT_EQ(Download<uint16_t>(d_extra.get(), extras.size()), extras)
        << "barrier " << barrier;
  }
  state = d_state.get();
  if (!time) return;
  // The bytes it reads, and the time per launch (stream launches back to
  // back, and the same launches in a CUDA graph, as an engine runs them),
  // beside the two GEMV launches it replaces.
  const Timing mix = Time([&](cudaStream_t stream) { return launch(stream); });
  std::printf(
      "hc mix %s rows %d [%d x %d] + [%d x %d]: stream %.1f us, graph %.1f us "
      "(%.0f GB/s)\n",
      int4 ? "int4" : "bf16", rows, low_rank, width, width, low_rank,
      mix.stream_us, mix.graph_us,
      static_cast<double>(weight_bytes) / mix.graph_us * 1e-3);
  if (!int4) return;
  auto d_t = Upload(std::vector<uint16_t>(static_cast<size_t>(rows) * low_rank,
                                          ToBf16(0.25f)));
  auto d_u = Upload(std::vector<uint16_t>(static_cast<size_t>(rows) * width));
  const Timing gemvs = Time([&](cudaStream_t stream) {
    const cudaError_t status = LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
        d_x.get(), static_cast<const uint8_t*>(d_down.get()),
        d_down_scales.get(), d_t.get(), rows, width, low_rank, 0, stream);
    if (status != cudaSuccess) return status;
    return LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
        d_t.get(), static_cast<const uint8_t*>(d_up.get()), d_up_scales.get(),
        d_u.get(), rows, low_rank, width, 0, stream);
  });
  std::printf("   the two GEMV launches alone: stream %.1f us, graph %.1f us\n",
              gemvs.stream_us, gemvs.graph_us);
}

TEST(HyperConnectionTest, MixMatchesReference) {
  // Qwen 4's widths (hidden 2560, hc 4, low rank 320), decode and a verify
  // step's rows; the reduced model's (hidden 256, low rank 32).
  for (const bool int4 : {false, true}) {
    Check(1, 2560, 320, int4, false);
    Check(2, 2560, 320, int4, false);
    Check(3, 2560, 320, int4, false);
    Check(4, 2560, 320, int4, false);
    Check(1, 256, 32, int4, false);
    Check(5, 256, 32, int4, false);
    Check(8, 256, 32, int4, false);
    // Wider streams, and a low rank that leaves a warp's lanes idle.
    Check(2, 4096, 512, int4, false);
    Check(1, 8192, 64, int4, false);
    // With the combine's injection projection (hc rows), and more rows.
    Check(1, 2560, 320, int4, false, /*extra=*/4);
    Check(3, 2560, 320, int4, false, /*extra=*/4);
    Check(4, 256, 32, int4, false, /*extra=*/16);
  }
}

TEST(HyperConnectionTest, TimesTheMix) {
  if (std::getenv("HC_MIX_TIMING") == nullptr) {
    GTEST_SKIP() << "Set HC_MIX_TIMING=1.";
  }
  Check(1, 2560, 320, /*int4=*/false, true);
  Check(1, 2560, 320, /*int4=*/true, true);
  Check(1, 2560, 320, /*int4=*/true, true, /*extra=*/4);
}

}  // namespace
