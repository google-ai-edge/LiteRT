// Timing of the decode attention kernels at Gemma 4 decode geometries (not a
// correctness test). Prints microseconds per launch pair.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/global_attention.h"
#include "litert/vendors/nvidia/trtllm/int2_gemv.h"

namespace {

struct CudaDeleter {
  void operator()(void* p) const { cudaFree(p); }
};
using CudaAllocation = std::unique_ptr<void, CudaDeleter>;
CudaAllocation AllocateCuda(size_t bytes) {
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
  cudaMemset(p, 0, bytes);
  return CudaAllocation(p);
}

struct Config {
  const char* name;
  bool int8;
  bool columns;
  int heads, rows, depth, seq;
  int window = 0;  // 0: every key visible; else the last `window` keys
  int read = 0;    // > 0: the window entry point reads `read` keys from the end
  // FP16 caches: the tensor-core streaming kernel (global_attention.h), one
  // launch per head, rows a multiple of 16.
  bool streaming = false;
};

void Time(const Config& c) {
  const size_t q_bytes = static_cast<size_t>(c.heads) * c.rows * c.depth * 2;
  const size_t cache_elems = static_cast<size_t>(c.heads) * c.seq * c.depth;
  const size_t cache_bytes = cache_elems * (c.int8 ? 1 : 2);
  auto q = AllocateCuda(q_bytes);
  auto k = AllocateCuda(cache_bytes);
  auto v = AllocateCuda(cache_bytes);
  auto mask = AllocateCuda(static_cast<size_t>(c.rows) * c.seq);
  {
    std::vector<uint8_t> host_mask(static_cast<size_t>(c.rows) * c.seq, 1);
    if (c.window > 0 && c.window < c.seq) {
      std::fill(host_mask.begin(), host_mask.end(), 0);
      for (int r = 0; r < c.rows; ++r) {
        for (int j = c.seq - c.window; j < c.seq; ++j)
          host_mask[r * c.seq + j] = 1;
      }
    }
    cudaMemcpy(mask.get(), host_mask.data(), host_mask.size(),
               cudaMemcpyHostToDevice);
  }
  auto out = AllocateCuda(q_bytes);
  const size_t ws_bytes =
      c.streaming ? LiteRtNvidiaGlobalAttentionWorkspaceBytes(c.rows, c.rows,
                                                              c.seq, c.depth)
                  : LiteRtNvidiaDecodeAttentionWorkspaceBytes(
                        c.heads, c.rows, c.read > 0 ? c.read : c.seq, c.depth);
  auto ws = AllocateCuda(ws_bytes);
  ASSERT_TRUE(q && k && v && mask && out && ws);
  const LiteRtNvidiaInt8CacheParams params = {0.0125f, -3, 0.02f, 5, c.columns};
  auto window_start = AllocateCuda(sizeof(int32_t));
  const int32_t start_value = c.read > 0 ? ((c.seq - c.read) / 64) * 64 : 0;
  cudaMemcpy(window_start.get(), &start_value, sizeof(int32_t),
             cudaMemcpyHostToDevice);
  auto launch_on = [&](cudaStream_t s) -> cudaError_t {
    if (c.streaming) {
      // As the plugin runs it: one launch per KV head, the launches in order
      // on the stream sharing the workspace.
      const size_t q_head = static_cast<size_t>(c.rows) * c.depth * 2;
      const size_t kv_head = static_cast<size_t>(c.seq) * c.depth * 2;
      cudaError_t status = cudaSuccess;
      for (int head = 0; head < c.heads && status == cudaSuccess; ++head) {
        status = LiteRtNvidiaLaunchGlobalAttention(
            static_cast<const char*>(q.get()) + head * q_head, true,
            static_cast<const char*>(k.get()) + head * kv_head,
            static_cast<const char*>(v.get()) + head * kv_head,
            static_cast<const bool*>(mask.get()), c.rows, c.rows, c.seq,
            c.depth, -1.0e9f, static_cast<char*>(out.get()) + head * q_head,
            ws.get(), s);
      }
      return status;
    }
    if (c.int8 && c.read > 0) {
      return LiteRtNvidiaLaunchDecodeAttentionInt8Window(
          q.get(), true, k.get(), v.get(), &params,
          static_cast<const bool*>(mask.get()), c.rows, c.heads, c.rows, c.seq,
          c.read, static_cast<const int32_t*>(window_start.get()), c.depth,
          -1.0e9f, out.get(), ws.get(), s);
    }
    if (c.int8) {
      return LiteRtNvidiaLaunchDecodeAttentionInt8(
          q.get(), true, k.get(), v.get(), &params,
          static_cast<const bool*>(mask.get()), c.rows, c.heads, c.rows, c.seq,
          c.depth, -1.0e9f, out.get(), ws.get(), s);
    }
    return LiteRtNvidiaLaunchDecodeAttention(
        q.get(), true, k.get(), v.get(), static_cast<const bool*>(mask.get()),
        c.rows, c.heads, c.rows, c.seq, c.depth, -1.0e9f, out.get(), ws.get(),
        s);
  };
  auto launch = [&]() { return launch_on(nullptr); };
  for (int i = 0; i < 500; ++i) ASSERT_EQ(launch(), cudaSuccess);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int kIters = 3000;
  cudaEventRecord(start);
  for (int i = 0; i < kIters; ++i) launch();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  // The same calls captured into a CUDA graph (how TensorRT-RTX runs the
  // engine): the per-call cost without stream submission overhead.
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  constexpr int kGraphCalls = 100;
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            cudaSuccess);
  for (int i = 0; i < kGraphCalls; ++i) {
    launch_on(stream);
  }
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
  for (int i = 0; i < 5; ++i)
    ASSERT_EQ(cudaGraphLaunch(exec, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  constexpr int kGraphIters = 30;
  cudaEventRecord(start, stream);
  for (int i = 0; i < kGraphIters; ++i) cudaGraphLaunch(exec, stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float graph_ms = 0.0f;
  cudaEventElapsedTime(&graph_ms, start, stop);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::printf(
      "ATTN %-28s heads=%d rows=%d depth=%d seq=%5d window=%d read=%d  stream "
      "%7.2f us/call  graph %7.2f us/call (workspace %zu KB)\n",
      c.name, c.heads, c.rows, c.depth, c.seq, c.window, c.read,
      ms * 1000.0f / kIters, graph_ms * 1000.0f / (kGraphIters * kGraphCalls),
      ws_bytes / 1024);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

TEST(DecodeAttentionBenchmark, LaunchFloor) {
  // A 256-byte stream memset stands in for the cheapest dependent stream op.
  auto out = AllocateCuda(256);
  for (int i = 0; i < 100; ++i) cudaMemsetAsync(out.get(), 0, 256, nullptr);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int kIters = 2000;
  cudaEventRecord(start);
  for (int i = 0; i < kIters; ++i) cudaMemsetAsync(out.get(), 0, 256, nullptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf("ATTN stream-op floor (256-byte memset): %.2f us/op\n",
              ms * 1000.0f / kIters);
}

TEST(DecodeAttentionBenchmark, KernelLaunchFloor) {
  // A tiny INT4 GEMV (one row, 256 x 256) is a real kernel launch with
  // negligible work: the per-launch floor of this host.
  auto act = AllocateCuda(256 * 2);
  auto w = AllocateCuda(256 * 256 / 2);
  auto scales = AllocateCuda(256 * 2);
  auto out = AllocateCuda(256 * 2);
  auto launch = [&]() {
    return LiteRtNvidiaLaunchBf16SubbytePerChannelGemvRows(
        act.get(), static_cast<const uint8_t*>(w.get()), scales.get(),
        out.get(), 4, 1, 256, 256, nullptr);
  };
  for (int i = 0; i < 200; ++i) ASSERT_EQ(launch(), cudaSuccess);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int kIters = 3000;
  cudaEventRecord(start);
  for (int i = 0; i < kIters; ++i) launch();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf("ATTN kernel launch floor (tiny GEMV): %.2f us/launch\n",
              ms * 1000.0f / kIters);
}

TEST(DecodeAttentionBenchmark, Geometries) {
  const Config configs[] = {
      {"int8-columns (E4B local)", true, true, 2, 4, 256, 512},
      {"int8-columns (E4B)", true, true, 2, 4, 256, 1024},
      {"int8-columns (E4B global 2k)", true, true, 2, 4, 256, 2048},
      {"int8-columns (E4B global 4k)", true, true, 2, 4, 256, 4096},
      {"int8-columns (E4B global 32k)", true, true, 2, 4, 256, 32768},
      {"int8-columns (E2B local)", true, true, 1, 8, 256, 512},
      {"int8-columns (E2B global 2k)", true, true, 1, 8, 256, 2048},
      {"int8-rows", true, false, 2, 4, 256, 2048},
      {"int8-rows (E4B 32k)", true, false, 2, 4, 256, 32768},
      {"int8-columns (E2B 8k)", true, true, 1, 8, 256, 8192},
      {"int8-columns (E2B 30k)", true, true, 1, 8, 256, 30720},
      {"int8-rows (E2B 30k)", true, false, 1, 8, 256, 30720},
      {"int8-columns (E4B 16k)", true, true, 2, 4, 256, 16384},
      {"E4B local 30k, 512 window", true, true, 2, 4, 256, 30720, 512},
      {"E4B verify local 2k (16 rows)", true, true, 2, 16, 256, 2048},
      {"E4B verify global 2k d512 (16 rows)", true, true, 2, 16, 512, 2048},
      {"E4B verify local 30k window (16 rows)", true, true, 2, 16, 256, 30720,
       512, 640},
      {"E4B verify global 30k d512 (16 rows)", true, true, 2, 16, 512, 30720},
      {"E4B local 30k, window entry", true, true, 2, 4, 256, 30720, 512, 576},
      {"E2B local 30k, window entry", true, true, 1, 8, 256, 30720, 512, 576},
      {"E4B local 2k, window entry", true, true, 2, 4, 256, 2048, 512, 576},
      {"E4B global 30k, depth 512", true, true, 2, 4, 512, 30720},
      {"E2B local 30k, 512 window", true, true, 1, 8, 256, 30720, 512},
      {"E2B global 30k, depth 512", true, true, 1, 8, 512, 30720},
      {"E2B global 2k, depth 512", true, true, 1, 8, 512, 2048},
      {"26B local 1k (8 kv, 2 rows)", false, true, 8, 2, 256, 1024},
      {"26B global 1k d512 (2 kv, 8 rows)", false, true, 2, 8, 512, 1024},
      {"26B global 8k d512", false, true, 2, 8, 512, 8192},
      {"26B global 32k d512", false, true, 2, 8, 512, 32768},
      {"26B verify local 1k (16 rows)", false, true, 8, 16, 256, 1024},
      // The tensor-core streaming kernel at the global layers' geometry with
      // the 8 rows padded to 16 (the plugin could pad), per head.
      {"26B global 1k streaming 16 rows", false, false, 2, 16, 512, 1024, 0, 0,
       true},
      {"26B global 8k streaming 16 rows", false, false, 2, 16, 512, 8192, 0, 0,
       true},
      {"26B global 32k streaming 16 rows", false, false, 2, 16, 512, 32768, 0,
       0, true},
      {"26B global 32k decode 16 rows", false, false, 2, 16, 512, 32768},
      {"fp16-rows same geometry", false, false, 2, 4, 256, 2048},
      {"fp16-rows (12B-like)", false, false, 8, 2, 256, 2048},
      {"fp16-rows (12B-like 32k)", false, false, 8, 2, 256, 32768},
  };
  for (const auto& c : configs) Time(c);
}

}  // namespace
