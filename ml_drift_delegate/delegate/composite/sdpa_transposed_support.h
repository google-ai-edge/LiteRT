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

// Device and shape support checks of the fused SDPA (Flash-Attention) kernels
// of `odml.sdpa_transposed`. They live in their own library so that the
// parsers, which decide whether the BOOL causal mask can be pruned, and the
// kernel builder, which picks the kernel, share one implementation.

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_TRANSPOSED_SUPPORT_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_TRANSPOSED_SUPPORT_H_

#include <cstdint>

#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift

namespace litert::ml_drift {

// Launch and split-K parameters of the Flash-Decode kernels. They depend on the
// GPU, so they are chosen per device by `GetFlashDecodeTuning`.
struct FlashDecodeTuning {
  // Number of SIMD groups (waves of 32 lanes) cooperating on one query head in
  // the wave-SIMD kernel. At most 32, so wave 0 can reduce one value per wave.
  int num_simd_groups = 16;
  // Number of work items per work group in the work-group reduction kernel.
  int threads = 256;
  // Work items that combine the per-work-item values in the first stage of a
  // work-group reduction. Every work item then does the second stage itself.
  int reduce_threads = 32;
  // Maximum number of query heads sharing one K/V head to process per work
  // group in the work-group reduction kernel so per-head accumulators stay in
  // registers. Larger GQA groups are partitioned across multiple work groups.
  int max_heads_per_work_group = 8;
  // The keys of every query token and K/V head group are split across work
  // groups until there are at least `target_work_groups` work groups, but
  // every split keeps at least `min_split_keys` keys.
  int target_work_groups = 40;
  int min_split_keys = 256;
  // Local memory budget in bytes of the work-group reduction kernel. Apple GPUs
  // reject compute pipelines that use more than 32 KB of threadgroup memory,
  // and most OpenCL devices report at least 32 KB of local memory. WebGPU
  // devices report their limit (see `GetFlashDecodeTuning`).
  int local_memory_bytes = 32 * 1024;
};

// Returns the Flash-Decode tuning for `gpu_info`.
//
// The defaults were measured on Adreno 830 (work-group reduction kernel): with
// Qwen3 0.6B (8 K/V heads, 1280 cache entries), 5 splits of 256 keys read the
// KV cache at 58 GB/s, compared with 43 GB/s for one work group per K/V head.
// The wave-SIMD kernel uses 16 waves on Apple GPUs.
// TODO(b/553029558): Tune for Mali, Intel, Nvidia and AMD GPUs.
FlashDecodeTuning GetFlashDecodeTuning(const ::ml_drift::GpuInfo& gpu_info);

// Keys scored by each work item per chunk of the work-group Flash-Decode
// kernel: at most as many as keep the probabilities of a chunk within 8 KB of
// local memory.
int MaxFlashDecodeKeysPerThread(int heads, int threads);

// Upper bound of the local memory in bytes used by the work-group Flash-Decode
// kernel for `heads` query heads per work group and `slices` channel slices
// (head_dim / 4), assuming the largest chunk of probabilities: the float4 query
// slices, the probabilities of a chunk, the two reduction stages and the float4
// output accumulators (see `GenerateWorkGroupFlashDecodeCode`).
int FlashDecodeLocalMemoryBytes(const FlashDecodeTuning& tuning, int heads,
                                int slices);

// Whether the GPU supports fixed 32-lane wave-SIMD vector reductions
// (`ucl::WaveSum` on `half4` / `ucl::WaveMax`).
//
// Limited to Metal: OpenCL `sub_group_reduce_*` has no vector overloads and
// Adreno / Mali waves are not 32 lanes wide; WebGPU and Vulkan need the
// subgroup feature and a fixed subgroup size of 32, which `GpuInfo` does not
// guarantee.
// TODO(b/553029558): Support WebGPU / Vulkan when the device reports a fixed
// 32-lane subgroup size, and OpenCL with per-component reductions.
bool SupportsWave32Vec4SimdReduce(const ::ml_drift::GpuInfo& gpu_info);

// Whether the GPU supports the 8x8 wave-matrix MMA prefill kernel.
// TODO(b/553029558): Support more backends (e.g. Vulkan / WebGPU cooperative
// matrices and OpenCL subgroup matrix extensions).
bool SupportsWaveMatrixPrefill(const ::ml_drift::GpuInfo& gpu_info);

// Whether fused Flash-Attention kernels can be used on this GPU:
// - Metal on Apple GPUs: the wave-matrix MMA prefill kernel and the wave-SIMD
//   or work-group Flash-Decode kernel.
// - OpenCL and WebGPU: the work-group kernel, for decode and, since it handles
//   one query token per work group, also for prefill whose score tensor would
//   be too large (`UsesWorkGroupFlashPrefill`).
// Shape-specific conditions are checked separately (`UsesFusedSdpaKernel`).
bool SupportsFusedSdpaKernels(const ::ml_drift::GpuInfo& gpu_info);

// Largest `[1, heads, tokens, cache_size]` attention score tensor, in bytes,
// that multi-token prefill on backends without the wave-matrix MMA kernel
// (OpenCL, WebGPU) materializes in the decomposed BatchedMatMul + Softmax
// graph. The 2D register-tiled GEMM of that graph has a higher prefill
// throughput than the work-group Flash kernel, which handles one query token
// per work group, but its score tensor grows with the full cache size. Above
// the budget, prefill switches to the work-group Flash kernel, which never
// materializes the scores and only reads the active keys.
//
// 768 MiB keeps an 8K cache of 32 query heads with 1024-token prefill (544 MiB)
// on the GEMM and moves a 32K cache of 16 or more query heads (>= 1 GiB, which
// exceeds the maximum buffer allocation on Adreno) to the Flash kernel.
inline constexpr uint64_t kMaxDecomposedSdpaScoreBytes = 768ull << 20;

// Bytes of the score tensor of the decomposed graph for query `q_shape`
// [b, heads, tokens, head_dim] and K cache `k_shape`
// [b, kv_heads, cache_size, head_dim], assuming F16 storage.
uint64_t DecomposedSdpaScoreBytes(const ::ml_drift::BHWC& q_shape,
                                  const ::ml_drift::BHWC& k_shape);

// Whether multi-token prefill (`q_shape.w > 1`) on a backend without the
// wave-matrix MMA kernel runs on the work-group Flash kernel: when the score
// tensor of the decomposed graph exceeds `max_score_bytes` or, on OpenCL, the
// maximum buffer allocation of the device. Tests lower `max_score_bytes` to run
// small prefill shapes on the work-group kernel.
bool UsesWorkGroupFlashPrefill(
    const ::ml_drift::GpuInfo& gpu_info, const ::ml_drift::BHWC& q_shape,
    const ::ml_drift::BHWC& k_shape,
    uint64_t max_score_bytes = kMaxDecomposedSdpaScoreBytes);

// Whether single-token decode may use the Flash-Decode kernels. Always true on
// Metal and OpenCL. On WebGPU, the work-group kernel must process the whole
// GQA group of a K/V head in one work group (`FlashDecodeConfig::heads ==
// q_heads / kv_heads`) so every K/V read is shared by all its query heads;
// otherwise (e.g. head dim 256 / 512 under the default 16 KB workgroup storage
// limit) decode keeps the decomposed graph.
bool UsesFlashDecodeForShape(const ::ml_drift::GpuInfo& gpu_info,
                             const ::ml_drift::BHWC& q_shape,
                             const ::ml_drift::BHWC& k_shape);

// Execution and work-group configuration for the Flash-Decode kernel.
struct FlashDecodeConfig {
  // Whether to use the 32-lane wave-SIMD reduction kernel (`ucl::WaveSum` /
  // `ucl::WaveMax` across `tuning.num_simd_groups` waves of 32 lanes) or the
  // work-group reduction kernel with GQA head grouping and split-K.
  bool use_wave_simd = false;
  // Device-dependent launch parameters.
  FlashDecodeTuning tuning;
  // Query heads per work group (they share one K/V head).
  int heads = 1;
  // Number of work groups that split the keys of each query token and head
  // group. With more than one split, a combine kernel merges the partial
  // results.
  int splits = 1;
};

FlashDecodeConfig GetFlashDecodeConfig(const ::ml_drift::GpuInfo& gpu_info,
                                       const ::ml_drift::BHWC& q_shape,
                                       const ::ml_drift::BHWC& k_shape);

// Shape conditions of the Flash-Decode kernels (wave-SIMD or work-group), for
// query `q_shape` [b, heads, tokens, head_dim] and K cache `k_shape`
// [b, kv_heads, cache_size, head_dim]. The caller still has to check that K
// and V are BUFFER tensors in the packed `odml.cache_update` layout and the
// shape of a mask, if any.
bool IsSupportedFlashSdpaShape(const ::ml_drift::GpuInfo& gpu_info,
                               const ::ml_drift::BHWC& q_shape,
                               const ::ml_drift::BHWC& k_shape);

// Shape conditions of the wave-matrix MMA prefill kernel, for query `q_shape`
// and K cache `k_shape` as in `IsSupportedFlashSdpaShape`: the head dim is a
// positive multiple of 4 and at most 128 (32 channel slices), the query heads
// are a multiple of the K/V heads, and the cache holds whole blocks of 4 keys.
// The caller still has to check that K and V are BUFFER tensors in the packed
// `odml.cache_update` layout.
bool IsSupportedFlashPrefillShape(const ::ml_drift::BHWC& q_shape,
                                  const ::ml_drift::BHWC& k_shape);

// Whether `BuildSdpaTransposedGpuGraph` runs `odml.sdpa_transposed` as a fused
// kernel that bounds the keys of every query by `params` (active tokens and
// query start), so a BOOL causal mask (`is_causal`) can be pruned:
// - Prefill on Metal: the MMA prefill kernel when
//   `IsSupportedFlashPrefillShape` holds.
// - Prefill on OpenCL / WebGPU: the work-group kernel when
//   `UsesWorkGroupFlashPrefill` holds.
// - Single-token decode: the Flash-Decode kernels when
//   `UsesFlashDecodeForShape` holds.
// Requires K/V produced by `odml.cache_update` (`from_cache_update`), which
// the delegate stores as BUFFER tensors.
bool UsesFusedSdpaKernel(const ::ml_drift::GpuInfo& gpu_info,
                         const ::ml_drift::BHWC& q_shape,
                         const ::ml_drift::BHWC& k_shape, bool is_prefill,
                         bool from_cache_update);

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_COMPOSITE_SDPA_TRANSPOSED_SUPPORT_H_
