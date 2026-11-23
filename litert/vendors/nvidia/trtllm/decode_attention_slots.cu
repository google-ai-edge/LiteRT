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

#include "litert/vendors/nvidia/trtllm/decode_attention_slots.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/decode_attention_blocks.cuh"

namespace {

using namespace litert::nvidia::decode_attention;

constexpr int kMaxSlots = kLiteRtNvidiaDecodeAttentionMaxSlots;

// The slots' buffers as the kernels take them, by value: a captured CUDA
// graph then holds them as it holds a single-sequence launch's arguments.
struct SlotTable {
  LiteRtNvidiaDecodeAttentionSlot slot[kMaxSlots];
};

// Clang reports the unroll hints it does not honor in the inlined block at
// the kernel's declaration.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
// One block per (chunk of keys, head, slot): the block of the
// single-sequence launch (DecodeAttentionPartialBlock) on the slot's own
// buffers and its own part of the workspace, laid out as that launch's.
template <typename T, int kDepth, int kRowsCap, typename Cache, int kChunk>
__global__ void __launch_bounds__(ThreadsFor<kDepth>(), 2)
    DecodeAttentionPartialSlots(SlotTable table, Cache cache, int mask_rows,
                                int heads, int rows, int seq, int cache_len,
                                float fill, int splits,
                                float* __restrict__ workspace,
                                size_t slot_floats) {
  extern __shared__ float smem[];
  using Element = typename Cache::Element;
  const LiteRtNvidiaDecodeAttentionSlot& slot = table.slot[blockIdx.z];
  float* ws_acc = workspace + blockIdx.z * slot_floats;
  float* ws_m = ws_acc + static_cast<size_t>(heads) * splits * rows * kDepth;
  float* ws_l = ws_m + static_cast<size_t>(heads) * splits * rows;
  DecodeAttentionPartialBlock<T, kDepth, kRowsCap, Cache, kChunk>(
      static_cast<const T*>(slot.q), static_cast<const Element*>(slot.k),
      static_cast<const Element*>(slot.v), cache, slot.mask, mask_rows, rows,
      seq, cache_len, slot.window_start, fill, splits, ws_acc, ws_m, ws_l,
      blockIdx.x, blockIdx.y, gridDim.x, smem);
}
#pragma clang diagnostic pop

// One block per (row, head, slot) (DecodeAttentionCombineBlock).
template <typename T, int kDepth>
__global__ void __launch_bounds__(kCombineThreads)
    DecodeAttentionCombineSlots(SlotTable table, int heads, int rows,
                                int splits,
                                const float* __restrict__ workspace,
                                size_t slot_floats) {
  __shared__ float scratch[CombineSharedFloats<kDepth>()];
  const float* ws_acc = workspace + blockIdx.z * slot_floats;
  const float* ws_m =
      ws_acc + static_cast<size_t>(heads) * splits * rows * kDepth;
  const float* ws_l = ws_m + static_cast<size_t>(heads) * splits * rows;
  DecodeAttentionCombineBlock<T, kDepth>(
      ws_acc, ws_m, ws_l, rows, splits,
      static_cast<T*>(table.slot[blockIdx.z].out), blockIdx.x, blockIdx.y,
      scratch);
}

// LITERT_NVIDIA_ATTENTION_CARVEOUT as the single-sequence launch reads it
// (no strtol: the CUDA objects would bind the host glibc's newer symbol).
int Carveout() {
  static const int carveout = [] {
    const char* value = std::getenv("LITERT_NVIDIA_ATTENTION_CARVEOUT");
    if (value == nullptr || value[0] == '\0') return 100;
    int number = 0;
    for (const char* c = value; *c != '\0'; ++c) {
      if (*c < '0' || *c > '9' || number > 100) return 100;
      number = number * 10 + (*c - '0');
    }
    return number <= 100 ? number : 100;
  }();
  return carveout;
}

struct Shape {
  int slots, mask_rows, heads, rows, seq, cache_len;
  float fill;
  size_t slot_floats;
};

template <typename T, int kDepth, int kRowsCap, typename Cache, int kChunk>
cudaError_t LaunchChunk(const SlotTable& table, const Cache& cache,
                        const Shape& s, int splits, void* workspace,
                        cudaStream_t stream) {
  constexpr size_t kSharedBytes =
      SharedBytesFor<kDepth, kRowsCap, Cache::kValueColumns, kChunk>();
  static bool attribute_set = false;  // per instantiation
  if (!attribute_set) {
    cudaError_t status = cudaFuncSetAttribute(
        DecodeAttentionPartialSlots<T, kDepth, kRowsCap, Cache, kChunk>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes));
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(
        DecodeAttentionPartialSlots<T, kDepth, kRowsCap, Cache, kChunk>,
        cudaFuncAttributePreferredSharedMemoryCarveout, Carveout());
    if (status != cudaSuccess) return status;
    attribute_set = true;
  }
  float* ws = static_cast<float*>(workspace);
  DecodeAttentionPartialSlots<T, kDepth, kRowsCap, Cache, kChunk>
      <<<dim3(splits, s.heads, s.slots), ThreadsFor<kDepth>(), kSharedBytes,
         stream>>>(table, cache, s.mask_rows, s.heads, s.rows, s.seq,
                   s.cache_len, s.fill, splits, ws, s.slot_floats);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  DecodeAttentionCombineSlots<T, kDepth>
      <<<dim3(s.rows, s.heads, s.slots), kCombineThreads, 0, stream>>>(
          table, s.heads, s.rows, splits, ws, s.slot_floats);
  return cudaGetLastError();
}

// The keys per block and the blocks per head of the single-sequence launch
// (decode_attention.cu's kMaxSplits, DefaultChunkFor and
// LITERT_NVIDIA_ATTENTION_CHUNK, which says why): a slot's partial sums must
// be split as that launch splits them to add up to its bits, which
// decode_attention_slots_test checks on both sides of the threshold.
constexpr int kMaxSplits = 168;

int ChunkFor(int seq, bool few_rows_row_cache) {
  static const int pinned = [] {
    const char* value = std::getenv("LITERT_NVIDIA_ATTENTION_CHUNK");
    if (value == nullptr) return 0;
    const bool is32 = value[0] == '3' && value[1] == '2' && value[2] == '\0';
    const bool is64 = value[0] == '6' && value[1] == '4' && value[2] == '\0';
    return is32 ? 32 : is64 ? 64 : 0;
  }();
  if (pinned != 0) return pinned;
  return seq > 16 * kMaxSplits || few_rows_row_cache ? 64 : 32;
}

template <typename T, int kDepth, int kRowsCap, typename Cache>
cudaError_t Launch(const SlotTable& table, const Cache& cache, const Shape& s,
                   void* workspace, cudaStream_t stream) {
  const int chunk = ChunkFor(s.seq, !Cache::kValueColumns && kRowsCap <= 4);
  const int chunks = (s.seq + chunk - 1) / chunk;
  const int splits = chunks < kMaxSplits ? chunks : kMaxSplits;
  if (chunk == 32) {
    return LaunchChunk<T, kDepth, kRowsCap, Cache, 32>(table, cache, s, splits,
                                                       workspace, stream);
  }
  return LaunchChunk<T, kDepth, kRowsCap, Cache, 64>(table, cache, s, splits,
                                                     workspace, stream);
}

template <typename T, int kDepth, typename Cache>
cudaError_t LaunchRows(const SlotTable& table, const Cache& cache,
                       const Shape& s, void* workspace, cudaStream_t stream) {
  if (s.rows <= 4) {
    return Launch<T, kDepth, 4, Cache>(table, cache, s, workspace, stream);
  }
  if (s.rows <= 8) {
    return Launch<T, kDepth, 8, Cache>(table, cache, s, workspace, stream);
  }
  return Launch<T, kDepth, kMaxRows, Cache>(table, cache, s, workspace, stream);
}

template <typename T, typename Cache>
cudaError_t LaunchDepth(const SlotTable& table, const Cache& cache,
                        const Shape& s, int depth, void* workspace,
                        cudaStream_t stream) {
  switch (depth) {
    case 128:
      return LaunchRows<T, 128>(table, cache, s, workspace, stream);
    case 256:
      return LaunchRows<T, 256>(table, cache, s, workspace, stream);
    case 512:
      return LaunchRows<T, 512>(table, cache, s, workspace, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

template <typename T>
cudaError_t LaunchCache(const SlotTable& table,
                        const LiteRtNvidiaInt8CacheParams* int8,
                        const Shape& s, int depth, void* workspace,
                        cudaStream_t stream) {
  if (int8 == nullptr) {
    return LaunchDepth<T>(table, Fp16Cache(), s, depth, workspace, stream);
  }
  if (int8->value_columns) {
    const Int8Cache<true> cache{int8->k_scale, int8->v_scale,
                                int8->k_zero_point, int8->v_zero_point};
    return LaunchDepth<T>(table, cache, s, depth, workspace, stream);
  }
  const Int8Cache<false> cache{int8->k_scale, int8->v_scale,
                               int8->k_zero_point, int8->v_zero_point};
  return LaunchDepth<T>(table, cache, s, depth, workspace, stream);
}

}  // namespace

extern "C" size_t LiteRtNvidiaDecodeAttentionSlotsWorkspaceBytes(
    int32_t slots, int32_t heads, int32_t rows, int32_t seq, int32_t depth) {
  return static_cast<size_t>(slots) *
         LiteRtNvidiaDecodeAttentionWorkspaceBytes(heads, rows, seq, depth);
}

extern "C" cudaError_t LiteRtNvidiaLaunchDecodeAttentionSlots(
    const LiteRtNvidiaDecodeAttentionSlot* slots, int32_t num_slots,
    bool q_bf16, const LiteRtNvidiaInt8CacheParams* int8, int32_t mask_rows,
    int32_t heads, int32_t rows, int32_t cache_len, int32_t window,
    int32_t depth, float fill, void* workspace, cudaStream_t stream) {
  if (slots == nullptr || num_slots <= 0 || num_slots > kMaxSlots ||
      workspace == nullptr || heads <= 0 || rows <= 0 || rows > kMaxRows ||
      window <= 0 || cache_len < window ||
      (depth != 128 && depth != 256 && depth != 512) ||
      (int8 == nullptr && window != cache_len) ||
      (int8 != nullptr && (!LiteRtNvidiaDecodeAttentionInt8Supports(depth) ||
                           !std::isfinite(int8->k_scale) ||
                           !std::isfinite(int8->v_scale)))) {
    return cudaErrorInvalidValue;
  }
  SlotTable table{};
  for (int32_t i = 0; i < num_slots; ++i) {
    const LiteRtNvidiaDecodeAttentionSlot& slot = slots[i];
    if (slot.q == nullptr || slot.k == nullptr || slot.v == nullptr ||
        slot.out == nullptr ||
        (slot.mask != nullptr && mask_rows != 1 && mask_rows != rows) ||
        (slot.window_start == nullptr && window != cache_len) ||
        (slot.window_start != nullptr && int8 == nullptr)) {
      return cudaErrorInvalidValue;
    }
    table.slot[i] = slot;
  }
  Shape shape{};
  shape.slots = num_slots;
  shape.mask_rows = mask_rows;
  shape.heads = heads;
  shape.rows = rows;
  shape.seq = window;
  shape.cache_len = cache_len;
  shape.fill = fill;
  shape.slot_floats =
      LiteRtNvidiaDecodeAttentionWorkspaceBytes(heads, rows, window, depth) /
      sizeof(float);
  if (q_bf16) {
    return LaunchCache<__nv_bfloat16>(table, int8, shape, depth, workspace,
                                      stream);
  }
  return LaunchCache<__half>(table, int8, shape, depth, workspace, stream);
}
