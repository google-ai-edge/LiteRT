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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_SLOT_TEST_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_SLOT_TEST_UTIL_H_

// Helpers of the slot kernels' GPU tests.

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "cuda_runtime_api.h"

namespace litert::nvidia::slot_test {

struct CudaDeleter {
  void operator()(void* p) const { cudaFree(p); }
};
using Device = std::unique_ptr<void, CudaDeleter>;

// Device memory of `bytes` bytes holding `data` (zeros with null data); null
// when the device has none.
inline Device Upload(const void* data, size_t bytes) {
  void* p = nullptr;
  if (cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
    return nullptr;
  }
  cudaError_t status = data != nullptr
                           ? cudaMemcpy(p, data, bytes, cudaMemcpyHostToDevice)
                           : cudaMemset(p, 0, bytes);
  // The memset runs on the default stream, which a non-blocking stream does
  // not wait for: it must not land on what a test's first launch writes.
  if (status == cudaSuccess && data == nullptr) {
    status = cudaDeviceSynchronize();
  }
  if (status != cudaSuccess) {
    cudaFree(p);
    return nullptr;
  }
  return Device(p);
}

template <typename T>
Device Upload(const std::vector<T>& values) {
  return Upload(values.data(), values.size() * sizeof(T));
}

inline std::vector<uint8_t> Download(const void* device, size_t bytes) {
  std::vector<uint8_t> host(bytes);
  if (cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    host.clear();
  }
  return host;
}

// Deterministic bytes: any INT8 values, or (floats) FP16 / BF16 values of
// moderate size and both signs when read two at a time (the high byte keeps
// the exponent near one).
inline std::vector<uint8_t> Bytes(size_t count, uint32_t seed, bool floats) {
  std::vector<uint8_t> bytes(count);
  uint32_t state = seed * 2654435761u + 12345u;
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    const uint8_t byte = static_cast<uint8_t>(state >> 24);
    bytes[i] = floats && i % 2 == 1 ? static_cast<uint8_t>(0x3c + (byte & 0x83))
                                    : byte;
  }
  return bytes;
}

// The median microseconds of two ways to do the same work, each captured
// into a CUDA graph on `stream` (as a decode step replays its kernels) and
// replayed `replays` times in turn, after a quarter of a second of replays
// that bring the device's clocks up (a few small kernels do not). False on a
// CUDA error.
inline bool TimeRound(cudaStream_t stream,
                      const std::function<cudaError_t()>* const ways[2],
                      int replays, float us[2]) {
  cudaGraphExec_t graphs[2] = {};
  cudaEvent_t begin = nullptr, end = nullptr;
  bool ok = cudaEventCreate(&begin) == cudaSuccess &&
            cudaEventCreate(&end) == cudaSuccess;
  for (int way = 0; ok && way < 2; ++way) {
    cudaGraph_t graph = nullptr;
    ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) ==
         cudaSuccess;
    const bool launched = ok && (*ways[way])() == cudaSuccess;
    ok = ok && cudaStreamEndCapture(stream, &graph) == cudaSuccess &&
         launched &&
         cudaGraphInstantiate(&graphs[way], graph, 0) == cudaSuccess;
    if (graph != nullptr) cudaGraphDestroy(graph);
  }
  std::vector<float> times[2];
  const auto start = std::chrono::steady_clock::now();
  int timed = 0;
  while (ok && timed < replays) {
    const bool warm = std::chrono::steady_clock::now() - start >=
                      std::chrono::milliseconds(250);
    for (int way = 0; ok && way < 2; ++way) {
      float ms = 0.0f;
      ok = cudaEventRecord(begin, stream) == cudaSuccess &&
           cudaGraphLaunch(graphs[way], stream) == cudaSuccess &&
           cudaEventRecord(end, stream) == cudaSuccess &&
           cudaEventSynchronize(end) == cudaSuccess &&
           cudaEventElapsedTime(&ms, begin, end) == cudaSuccess;
      if (ok && warm) times[way].push_back(ms * 1000.0f);
    }
    timed += warm ? 1 : 0;
  }
  for (cudaGraphExec_t graph : graphs) {
    if (graph != nullptr) cudaGraphExecDestroy(graph);
  }
  if (begin != nullptr) cudaEventDestroy(begin);
  if (end != nullptr) cudaEventDestroy(end);
  if (!ok) return false;
  for (int way = 0; way < 2; ++way) {
    std::sort(times[way].begin(), times[way].end());
    us[way] = times[way][times[way].size() / 2];
  }
  return true;
}

// The time of `first` and of `second`, two ways to do the same work: the
// smallest of three rounds' medians (TimeRound) for each. A launch of a few
// microseconds sometimes takes three times as long for a whole round on a
// shared or virtualized device (WSL), whichever way it belongs to; a round
// that was not disturbed gives the kernels' time. Both ways are launched
// once before the captures. False on a CUDA error.
inline bool TimeTwoWays(cudaStream_t stream,
                        const std::function<cudaError_t()>& first,
                        const std::function<cudaError_t()>& second, int replays,
                        float* first_us, float* second_us) {
  const std::function<cudaError_t()>* const ways[2] = {&first, &second};
  for (const auto* way : ways) {
    if ((*way)() != cudaSuccess) return false;
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
  float best[2] = {0.0f, 0.0f};
  for (int round = 0; round < 3; ++round) {
    float us[2];
    if (!TimeRound(stream, ways, replays, us)) return false;
    for (int way = 0; way < 2; ++way) {
      if (round == 0 || us[way] < best[way]) best[way] = us[way];
    }
  }
  *first_us = best[0];
  *second_us = best[1];
  return true;
}

}  // namespace litert::nvidia::slot_test

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TRTLLM_SLOT_TEST_UTIL_H_
