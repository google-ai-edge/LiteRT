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

#include "ml_drift_delegate/delegate/composite/sdpa_transposed_support.h"

#include <algorithm>
#include <cstdint>

#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift

namespace litert::ml_drift {

FlashDecodeTuning GetFlashDecodeTuning(const ::ml_drift::GpuInfo& gpu_info) {
  FlashDecodeTuning tuning;
  const int max_work_group_size = gpu_info.GetMaxWorkGroupTotalSize();
  if (max_work_group_size > 0) {
    // Keep both kernels within the work-group size limit of the device.
    while (tuning.threads > tuning.reduce_threads &&
           tuning.threads > max_work_group_size) {
      tuning.threads /= 2;
    }
    while (tuning.num_simd_groups > 1 &&
           tuning.num_simd_groups * 32 > max_work_group_size) {
      tuning.num_simd_groups /= 2;
    }
  }
  if (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()) {
    // The Adreno OpenCL compiler limits the work-group size of a kernel by its
    // register use, regardless of `reqd_work_group_size`. The work-group
    // kernel keeps the scores, running maximum, normalizer and output
    // accumulator of every query head of the work group in registers, and
    // with 4 heads per work group (e.g. GQA group 4 with head dim 256 or 512)
    // Adreno 830 limits the kernel to 192 work items, so the 256-work-item
    // dispatch fails with CL_INVALID_WORK_GROUP_SIZE. With 2 heads per work
    // group the limit is 384 work items and decode is as fast (b/578633071).
    tuning.max_heads_per_work_group = 2;
  }
  if (gpu_info.IsApiWebGpu() &&
      gpu_info.webgpu_info.max_compute_workgroup_storage_size > 0) {
    // WebGPU validates the workgroup storage of a pipeline against the limit
    // of the device: 16 KB by default, or the adapter limit when requested.
    tuning.local_memory_bytes =
        gpu_info.webgpu_info.max_compute_workgroup_storage_size;
  }
  // TODO(b/553029558): Use the local memory size of OpenCL
  // (`CL_DEVICE_LOCAL_MEM_SIZE`), Metal (`maxThreadgroupMemoryLength`) and
  // Vulkan (`maxComputeSharedMemorySize`) devices once `GpuInfo` reports it.
  return tuning;
}

int MaxFlashDecodeKeysPerThread(int heads, int threads) {
  return std::clamp(2048 / (heads * threads), 1, 4);
}

int FlashDecodeLocalMemoryBytes(const FlashDecodeTuning& tuning, int heads,
                                int slices) {
  constexpr int kFloatBytes = static_cast<int>(sizeof(float));
  const int threads = tuning.threads;
  const int chunk = threads * MaxFlashDecodeKeysPerThread(heads, threads);
  const int q_local = heads * slices * 4 * kFloatBytes;
  const int p_local = heads * chunk * kFloatBytes;
  const int red_local = heads * threads * kFloatBytes;
  const int red2_local = heads * tuning.reduce_threads * kFloatBytes;
  const int acc_local = threads * 4 * kFloatBytes;
  return q_local + p_local + red_local + red2_local + acc_local;
}

bool SupportsWave32Vec4SimdReduce(const ::ml_drift::GpuInfo& gpu_info) {
  return gpu_info.IsWaveSizeEqualTo32() && gpu_info.IsApiMetal();
}

bool SupportsWaveMatrixPrefill(const ::ml_drift::GpuInfo& gpu_info) {
  return gpu_info.IsApple() && gpu_info.IsApiMetal();
}

bool SupportsFusedSdpaKernels(const ::ml_drift::GpuInfo& gpu_info) {
  return (gpu_info.IsApple() && gpu_info.IsApiMetal()) ||
         gpu_info.IsApiOpenCl() || gpu_info.IsApiWebGpu();
}

FlashDecodeConfig GetFlashDecodeConfig(const ::ml_drift::GpuInfo& gpu_info,
                                       const ::ml_drift::BHWC& q_shape,
                                       const ::ml_drift::BHWC& k_shape) {
  FlashDecodeConfig config;
  config.tuning = GetFlashDecodeTuning(gpu_info);
  if (SupportsWave32Vec4SimdReduce(gpu_info) && q_shape.c == 128 &&
      config.tuning.num_simd_groups > 1) {
    config.use_wave_simd = true;
    return config;
  }
  if (k_shape.h <= 0 || q_shape.h % k_shape.h != 0) {
    return config;
  }
  // Process as many query heads of the GQA group per work group as divide the
  // group and keep the kernel within the local memory budget of the device.
  // With head dim 512 (128 slices), 8 heads need 37,888 B but 4 heads only
  // 25,088 B. If not even one head fits, `heads` stays 1 and
  // `IsSupportedFlashSdpaShape` rejects the config, so the op falls back to the
  // decomposed graph.
  const int group_size = q_shape.h / k_shape.h;
  const int slices = q_shape.c / 4;
  for (int h = std::min(group_size, config.tuning.max_heads_per_work_group);
       h >= 1; --h) {
    if (group_size % h == 0 &&
        FlashDecodeLocalMemoryBytes(config.tuning, h, slices) <=
            config.tuning.local_memory_bytes) {
      config.heads = h;
      break;
    }
  }
  const int work_groups = (q_shape.h / config.heads) * q_shape.w;
  const int wanted_splits =
      (config.tuning.target_work_groups + work_groups - 1) / work_groups;
  const int max_splits = std::max(1, k_shape.w / config.tuning.min_split_keys);
  config.splits = std::clamp(wanted_splits, 1, max_splits);
  return config;
}

namespace {

// Shape conditions shared by the fused kernels: whole channel slices (a
// positive head dim that is a multiple of 4), query heads that are a multiple
// of the K/V heads (query head `h` reads K/V head `h / (q_heads / kv_heads)`),
// and a packed cache layout of whole blocks of 4 keys.
bool IsSupportedFusedSdpaShape(const ::ml_drift::BHWC& q_shape,
                               const ::ml_drift::BHWC& k_shape) {
  return q_shape.c > 0 && q_shape.c % 4 == 0 && k_shape.h > 0 &&
         q_shape.h >= k_shape.h && q_shape.h % k_shape.h == 0 &&
         k_shape.w % 4 == 0;
}

}  // namespace

bool IsSupportedFlashSdpaShape(const ::ml_drift::GpuInfo& gpu_info,
                               const ::ml_drift::BHWC& q_shape,
                               const ::ml_drift::BHWC& k_shape) {
  if (!IsSupportedFusedSdpaShape(q_shape, k_shape)) {
    return false;
  }
  const FlashDecodeConfig config =
      GetFlashDecodeConfig(gpu_info, q_shape, k_shape);
  if (config.use_wave_simd) {
    return true;
  }
  const int slices = q_shape.c / 4;
  const int threads = config.tuning.threads;
  return slices <= threads && threads % slices == 0 &&
         FlashDecodeLocalMemoryBytes(config.tuning, config.heads, slices) <=
             config.tuning.local_memory_bytes;
}

bool IsSupportedFlashPrefillShape(const ::ml_drift::BHWC& q_shape,
                                  const ::ml_drift::BHWC& k_shape) {
  // The MMA prefill kernel covers at most 32 channel slices.
  return IsSupportedFusedSdpaShape(q_shape, k_shape) && q_shape.c <= 128;
}

uint64_t DecomposedSdpaScoreBytes(const ::ml_drift::BHWC& q_shape,
                                  const ::ml_drift::BHWC& k_shape) {
  constexpr uint64_t kHalfBytes = 2;
  return static_cast<uint64_t>(q_shape.b) * q_shape.h * q_shape.w * k_shape.w *
         kHalfBytes;
}

namespace {

// `bytes`, at most the maximum buffer allocation of an OpenCL device.
uint64_t ClampToMaxAllocation(const ::ml_drift::GpuInfo& gpu_info,
                              uint64_t bytes) {
  if (gpu_info.IsApiOpenCl() && gpu_info.opencl_info.max_allocation_size > 0) {
    return std::min(bytes, gpu_info.opencl_info.max_allocation_size);
  }
  return bytes;
}

}  // namespace

uint64_t DecomposedSdpaScoreChunkBudget(const ::ml_drift::GpuInfo& gpu_info,
                                        uint64_t max_score_bytes,
                                        uint64_t max_chunk_bytes) {
  const uint64_t budget =
      SupportsFusedSdpaKernels(gpu_info) && !SupportsWaveMatrixPrefill(gpu_info)
          ? std::max(max_chunk_bytes, max_score_bytes)
          : max_chunk_bytes;
  // Each chunk must also fit one buffer of an OpenCL device, even when its
  // maximum allocation is below `max_chunk_bytes`.
  return ClampToMaxAllocation(gpu_info, budget);
}

int DecomposedSdpaBytesPerScore(::ml_drift::DataType data_type,
                                bool from_cache_update) {
  const int bytes = ::ml_drift::SizeOf(data_type);
  return from_cache_update ? bytes : 2 * bytes;
}

int NumDecomposedSdpaScoreChunks(const ::ml_drift::BHWC& q_shape,
                                 const ::ml_drift::BHWC& k_shape,
                                 uint64_t max_chunk_bytes,
                                 int bytes_per_score) {
  if (k_shape.h <= 0 || q_shape.h < k_shape.h || q_shape.h % k_shape.h != 0) {
    return 1;
  }
  const int gqa_ratio = q_shape.h / k_shape.h;
  const uint64_t score_bytes = static_cast<uint64_t>(q_shape.b) * q_shape.h *
                               q_shape.w * k_shape.w * bytes_per_score;
  for (int chunks = 1; chunks < gqa_ratio; ++chunks) {
    // The score bytes are a multiple of `gqa_ratio`, so the division is exact.
    if (gqa_ratio % chunks == 0 && score_bytes / chunks <= max_chunk_bytes) {
      return chunks;
    }
  }
  return gqa_ratio;
}

bool UsesWorkGroupFlashPrefill(const ::ml_drift::GpuInfo& gpu_info,
                               const ::ml_drift::BHWC& q_shape,
                               const ::ml_drift::BHWC& k_shape,
                               uint64_t max_score_bytes) {
  if (SupportsWaveMatrixPrefill(gpu_info) || q_shape.w <= 1) {
    return false;
  }
  return DecomposedSdpaScoreBytes(q_shape, k_shape) >
         ClampToMaxAllocation(gpu_info, max_score_bytes);
}

bool UsesFlashDecodeForShape(const ::ml_drift::GpuInfo& gpu_info,
                             const ::ml_drift::BHWC& q_shape,
                             const ::ml_drift::BHWC& k_shape) {
  if (!gpu_info.IsApiWebGpu()) {
    return true;
  }
  if (k_shape.h <= 0 || q_shape.h % k_shape.h != 0) {
    return false;
  }
  const FlashDecodeConfig config =
      GetFlashDecodeConfig(gpu_info, q_shape, k_shape);
  return config.use_wave_simd || config.heads == q_shape.h / k_shape.h;
}

bool UsesFusedSdpaKernel(const ::ml_drift::GpuInfo& gpu_info,
                         const ::ml_drift::BHWC& q_shape,
                         const ::ml_drift::BHWC& k_shape, bool is_prefill,
                         bool from_cache_update) {
  if (!from_cache_update || !SupportsFusedSdpaKernels(gpu_info)) {
    return false;
  }
  if (is_prefill && SupportsWaveMatrixPrefill(gpu_info)) {
    return IsSupportedFlashPrefillShape(q_shape, k_shape);
  }
  if (is_prefill && q_shape.w > 1) {
    if (!UsesWorkGroupFlashPrefill(gpu_info, q_shape, k_shape)) {
      return false;
    }
  } else if (q_shape.w != 1) {
    // Multi-token decode (e.g. a speculative-decoding verifier) keeps its mask.
    return false;
  } else if (!UsesFlashDecodeForShape(gpu_info, q_shape, k_shape)) {
    return false;
  }
  return IsSupportedFlashSdpaShape(gpu_info, q_shape, k_shape);
}

}  // namespace litert::ml_drift
