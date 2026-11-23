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

// Timing of the QSA kernels (qsa_indexer.h, qsa_sparse_attention.h) at the
// shapes of a 27B Qwen 4 layer, against the bytes each launch must read, and
// of the dense decode attention over the same cache (not a correctness
// test). Prints microseconds per call, stream-launched and in a CUDA graph.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/qsa_indexer.h"
#include "litert/vendors/nvidia/trtllm/qsa_sparse_attention.h"

namespace {

struct CudaDeleter {
  void operator()(void* p) const { cudaFree(p); }
};
using Device = std::unique_ptr<void, CudaDeleter>;

Device Allocate(size_t bytes) {
  void* p = nullptr;
  if (cudaMalloc(&p, std::max<size_t>(bytes, 4)) != cudaSuccess) {
    return nullptr;
  }
  cudaMemset(p, 0, std::max<size_t>(bytes, 4));
  return Device(p);
}

template <typename T>
Device Upload(const std::vector<T>& values) {
  Device device = Allocate(values.size() * sizeof(T));
  cudaMemcpy(device.get(), values.data(), values.size() * sizeof(T),
             cudaMemcpyHostToDevice);
  return device;
}

// Random FP16 bits in [-1, 1).
std::vector<uint16_t> RandomHalves(size_t count, uint32_t seed) {
  std::vector<uint16_t> bits(count);
  uint32_t x = seed;
  for (auto& b : bits) {
    x = x * 1664525u + 1013904223u;
    // sign | exponent 13..14 | mantissa: magnitudes in [2^-2, 1).
    b = static_cast<uint16_t>(((x >> 16) & 0x8000u) |
                              (0x3400u + ((x >> 8) & 0x7FFu)));
  }
  return bits;
}

std::vector<float> RandomFloats(size_t count, uint32_t seed) {
  std::vector<float> values(count);
  uint32_t x = seed;
  for (auto& v : values) {
    x = x * 1664525u + 1013904223u;
    v = static_cast<float>(x >> 8) / 8388608.0f - 1.0f;
  }
  return values;
}

// Microseconds per call: stream launches, and the same calls in a CUDA
// graph (as TensorRT-RTX runs an engine).
struct Timing {
  float stream_us;
  float graph_us;
};

Timing Time(const std::function<cudaError_t(cudaStream_t)>& launch) {
  cudaStream_t stream = nullptr;
  EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  for (int i = 0; i < 20; ++i) EXPECT_EQ(launch(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int kIters = 200;
  cudaEventRecord(start, stream);
  for (int i = 0; i < kIters; ++i) launch(stream);
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
  for (int i = 0; i < 3; ++i) cudaGraphLaunch(exec, stream);
  cudaStreamSynchronize(stream);
  constexpr int kGraphIters = 10;
  cudaEventRecord(start, stream);
  for (int i = 0; i < kGraphIters; ++i) cudaGraphLaunch(exec, stream);
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
  return {ms * 1000.0f / kIters,
          graph_ms * 1000.0f / (kGraphIters * kGraphCalls)};
}

// The 27B's layer: 24 query heads on 4 KV heads of 256; an indexer of 8
// heads of 128 with 64 rotary dims and 512 blocks of 4 per row.
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kDepth = 256;
constexpr int kIndexHeads = 8;
constexpr int kIndexDim = 128;
constexpr int kRotary = 64;
constexpr int kTopk = 512;

// A causal mask [rows, len] whose rows end at len - rows + r.
std::vector<uint8_t> CausalMask(int rows, int len) {
  std::vector<uint8_t> mask(static_cast<size_t>(rows) * len, 0);
  const int first = len - rows - 3;
  for (int r = 0; r < rows; ++r) {
    std::fill(mask.begin() + static_cast<size_t>(r) * len,
              mask.begin() + static_cast<size_t>(r) * len + first + r + 1, 1);
  }
  return mask;
}

void TimeIndexer(int context, int rows) {
  LiteRtNvidiaQsaIndexerParams p = {};
  p.rows = rows;
  p.index_heads = kIndexHeads;
  p.index_dim = kIndexDim;
  p.rotary_dim = kRotary;
  p.max_blocks = context / 4;
  p.block_topk = kTopk;
  p.mask_len = context;
  p.q_type = kLiteRtNvidiaQsaBFloat16;
  p.state_type = kLiteRtNvidiaQsaFloat32;
  p.mask_type = kLiteRtNvidiaQsaMaskBool;
  p.rms_norm_eps = 1e-6f;
  p.scale = 0.0884f;
  ASSERT_TRUE(LiteRtNvidiaQsaIndexerSupports(&p));
  auto q = Upload(
      RandomHalves(static_cast<size_t>(rows) * kIndexHeads * kIndexDim, 1));
  auto k_new = Upload(RandomHalves(static_cast<size_t>(rows) * kIndexDim, 2));
  auto ring = Upload(RandomFloats(4 * kIndexDim, 3));
  auto pool =
      Upload(RandomFloats(static_cast<size_t>(p.max_blocks) * kIndexDim, 4));
  auto norm = Upload(std::vector<float>(kIndexDim, 1.0f));
  auto cos =
      Upload(RandomFloats(static_cast<size_t>(p.max_blocks) * kRotary, 5));
  auto sin =
      Upload(RandomFloats(static_cast<size_t>(p.max_blocks) * kRotary, 6));
  auto start = Upload(std::vector<int32_t>{context - rows - 3});
  auto mask = Upload(CausalMask(rows, context));
  auto topk = Allocate(static_cast<size_t>(rows) * kTopk * 4);
  auto work = Allocate(LiteRtNvidiaQsaIndexerWorkspaceBytes(&p));
  const Timing t = Time([&](cudaStream_t s) {
    return LiteRtNvidiaLaunchQsaIndexerUpdate(
        &p, q.get(), k_new.get(), ring.get(), pool.get(),
        static_cast<const float*>(norm.get()),
        static_cast<const float*>(cos.get()),
        static_cast<const float*>(sin.get()),
        static_cast<const int32_t*>(start.get()), mask.get(),
        static_cast<int32_t*>(topk.get()), work.get(), s);
  });
  // Must read: the pool once, the rows' mask rows (and write the scores).
  const double bytes = static_cast<double>(context) / 4 * kIndexDim * 4 +
                       static_cast<double>(rows) * context;
  std::printf(
      "QSA indexer      ctx=%6d rows=%4d  stream %8.2f us  graph %8.2f us  "
      "pool+mask %6.1f MB -> %6.0f GB/s\n",
      context, rows, t.stream_us, t.graph_us, bytes / 1e6,
      bytes / (t.graph_us * 1e3));
}

void TimeSparseAttention(int context, int rows) {
  LiteRtNvidiaQsaAttentionParams p = {};
  p.rows = rows;
  p.q_heads = kQueryHeads;
  p.kv_heads = kKvHeads;
  p.head_dim = kDepth;
  p.cache_len = context;
  p.block_topk = kTopk;
  p.q_type = kLiteRtNvidiaQsaBFloat16;
  p.cache_type = kLiteRtNvidiaQsaFloat16;
  p.value_columns = 0;
  p.mask_type = kLiteRtNvidiaQsaMaskBool;
  p.scale = 0.0625f;
  ASSERT_TRUE(LiteRtNvidiaQsaAttentionSupports(&p));
  const size_t cache = static_cast<size_t>(kKvHeads) * context * kDepth;
  auto q =
      Upload(RandomHalves(static_cast<size_t>(kQueryHeads) * rows * kDepth, 1));
  auto k = Upload(RandomHalves(cache, 2));
  auto v = Upload(RandomHalves(cache, 3));
  // Each row selects 512 distinct complete blocks spread over the context.
  std::vector<int32_t> topk(static_cast<size_t>(rows) * kTopk);
  const int complete = (context - rows - 3) / 4;
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < kTopk; ++i) {
      topk[static_cast<size_t>(r) * kTopk + i] =
          static_cast<int32_t>((static_cast<int64_t>(i) * complete) / kTopk);
    }
  }
  auto topk_d = Upload(topk);
  auto mask = Upload(CausalMask(rows, context));
  auto out = Allocate(static_cast<size_t>(rows) * kQueryHeads * kDepth * 2);
  auto work = Allocate(LiteRtNvidiaQsaAttentionWorkspaceBytes(&p));
  const Timing t = Time([&](cudaStream_t s) {
    return LiteRtNvidiaLaunchQsaAttention(
        &p, q.get(), k.get(), v.get(),
        static_cast<const int32_t*>(topk_d.get()), mask.get(), nullptr,
        out.get(), work.get(), s);
  });
  // Must read: the selected keys and values of every row once (a key read by
  // several rows counts once per row: the upper bound), and the mask rows.
  const double kv =
      static_cast<double>(rows) * (kTopk + 1) * 4 * kDepth * 2 * 2 * kKvHeads;
  const double bytes = kv + static_cast<double>(rows) * context;
  std::printf(
      "QSA attention    ctx=%6d rows=%4d  stream %8.2f us  graph %8.2f us  "
      "kv+mask %7.1f MB -> %6.0f GB/s\n",
      context, rows, t.stream_us, t.graph_us, bytes / 1e6,
      bytes / (t.graph_us * 1e3));
}

// The dense decode attention over the whole cache, as the 27B's attention
// would run without QSA: one launch over the 4 KV heads with the 6 query
// heads of each as rows.
void TimeDenseAttention(int context) {
  const int rows = kQueryHeads / kKvHeads;
  const size_t cache = static_cast<size_t>(kKvHeads) * context * kDepth;
  auto q = Upload(RandomHalves(static_cast<size_t>(kQueryHeads) * kDepth, 1));
  auto k = Upload(RandomHalves(cache, 2));
  auto v = Upload(RandomHalves(cache, 3));
  auto mask = Upload(CausalMask(1, context));
  auto out = Allocate(static_cast<size_t>(kQueryHeads) * kDepth * 2);
  auto work = Allocate(LiteRtNvidiaDecodeAttentionWorkspaceBytes(
      kKvHeads, rows, context, kDepth));
  const Timing t = Time([&](cudaStream_t s) {
    return LiteRtNvidiaLaunchDecodeAttention(
        q.get(), true, k.get(), v.get(), static_cast<const bool*>(mask.get()),
        1, kKvHeads, rows, context, kDepth, -1.0e9f, out.get(), work.get(), s);
  });
  const double bytes = static_cast<double>(cache) * 2 * 2;
  std::printf(
      "dense attention  ctx=%6d rows=%4d  stream %8.2f us  graph %8.2f us  "
      "kv %7.1f MB -> %6.0f GB/s\n",
      context, 1, t.stream_us, t.graph_us, bytes / 1e6,
      bytes / (t.graph_us * 1e3));
}

// QSA_BENCHMARK_CONTEXT and QSA_BENCHMARK_ROWS (decimal) limit the runs to
// one context or one row count (profiling).
int EnvValue(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? 0 : std::atoi(value);
}

TEST(QsaBenchmark, ModelShapes) {
  const int only_context = EnvValue("QSA_BENCHMARK_CONTEXT");
  const int only_rows = EnvValue("QSA_BENCHMARK_ROWS");
  for (int context : {8192, 32768, 131072}) {
    if (only_context != 0 && context != only_context) continue;
    for (int rows : {1, 4, 8, 1024}) {
      if (only_rows == 0 || rows == only_rows) TimeIndexer(context, rows);
    }
    for (int rows : {1, 4, 8, 1024}) {
      if (only_rows == 0 || rows == only_rows) {
        TimeSparseAttention(context, rows);
      }
    }
    if (only_rows == 0 || only_rows == 1) TimeDenseAttention(context);
  }
}

}  // namespace
