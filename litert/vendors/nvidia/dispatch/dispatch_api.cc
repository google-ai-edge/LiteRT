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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_logging_helper_with_runtime_context.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/cache_layout.h"
#include "litert/vendors/nvidia/compiler/cache_update_plugin.h"
#include "litert/vendors/nvidia/compiler/decode_attention_plugin.h"
#include "litert/vendors/nvidia/compiler/int8_cache_write_plugin.h"
#include "litert/vendors/nvidia/compiler/moe_plugin.h"
#include "litert/vendors/nvidia/compiler/qkv_norm_rope_plugin.h"
#include "litert/vendors/nvidia/compiler/subbyte_gemv_plugin.h"
#include "litert/vendors/nvidia/dispatch/aot_artifact.h"
#include "litert/vendors/nvidia/dispatch/dispatch_profiler.h"
#include "litert/vendors/nvidia/dispatch/greedy_sampler_c_api.h"
#include "litert/vendors/nvidia/dispatch/greedy_sampler_kernel.h"
#include "litert/vendors/nvidia/dispatch/runtime_cache.h"
#include "litert/vendors/nvidia/dispatch/step_feed_kernel.h"
#include "litert/vendors/nvidia/dispatch/step_sample_kernel.h"
#include "litert/vendors/nvidia/dispatch/step_sampler_c_api.h"
#include "litert/vendors/nvidia/dispatch/tensor_buffer_layout.h"
#include "litert/vendors/nvidia/dispatch/tensor_buffer_view.h"
#include "litert/vendors/nvidia/dispatch/token_table_c_api.h"
#include "litert/vendors/nvidia/dispatch/topk_candidates_c_api.h"
#include "litert/vendors/nvidia/dispatch/topk_candidates_kernel.h"
#include "litert/vendors/nvidia/dispatch/weight_store.h"
#include "litert/vendors/nvidia/memory_profile.h"
#include "litert/vendors/nvidia/tensorrt_logger.h"
#include "NvInfer.h"
#include "NvInferVersion.h"
#include "litert/vendors/nvidia/trtllm/head_kernels.h"
#include "litert/vendors/nvidia/trtllm/int2_gemv.h"

namespace {

using ::litert::Error;
using ::litert::Expected;
using ::litert::nvidia::DispatchCpuTimer;
using ::litert::nvidia::DispatchProfileMetrics;
using ::litert::nvidia::DispatchProfilingEnabled;
using ::litert::nvidia::kNvidiaCudaTensorBufferType;
using ::litert::nvidia::MappedAotArtifact;

template <typename T>
using TrtPtr = std::unique_ptr<T>;

Expected<void> CudaOk(cudaError_t error, const char* what) {
  if (error == cudaSuccess) {
    return {};
  }
  return Error(kLiteRtStatusErrorRuntimeFailure,
               std::string(what) + ": " + cudaGetErrorString(error));
}

// A token table the runtime registered (token_table_c_api.h), on the device,
// for the step feeds whose bytecode has no table of its own.
struct RegisteredTokenTable {
  uint32_t vocab = 0;
  uint32_t dim = 0;
  uint32_t scale_block = 0;  // 0: FP32 scales per row, else FP16 per block
  float multiplier = 1.0f;
  void* table = nullptr;   // INT4 [vocab, dim]
  void* scales = nullptr;  // FP32 [vocab] or FP16 [vocab, dim / scale_block]
};

// The registered tables and the [vocab, dim] the step feeds await: a
// registration is taken only when a feed awaits it.
struct TokenTableRegistry {
  std::mutex mutex;
  std::vector<RegisteredTokenTable> tables;
  std::vector<std::pair<uint32_t, uint32_t>> awaited;
};

TokenTableRegistry& GetTokenTableRegistry() {
  static auto* registry = new TokenTableRegistry;
  return *registry;
}

// How the session samples, for the steps run ahead (step_sampler_c_api.h):
// the configuration, and the uniform draws of the host's next samples.
struct StepSamplerState {
  std::mutex mutex;
  bool told = false;  // the session's sampler has said how it samples
  int32_t k = 1;
  float p = 1.0f;
  float temperature = 1.0f;
  double uniforms[8] = {};
  int32_t count = 0;
};

StepSamplerState& GetStepSamplerState() {
  static auto* state = new StepSamplerState;
  return *state;
}

// The local attention's sliding window the runtime told (token_table_c_api.h),
// for a feed whose bytecode does not say (0: not told).
std::atomic<uint32_t>& StepFeedLocalWindow() {
  static std::atomic<uint32_t> window{0};
  return window;
}

// What a decode step's own token selection leaves for the host's sampler:
// the step's k candidates (ids, values, the row's flag), or its argmax (k of
// 1), in pinned memory the feed owns, by the logits buffer's device pointer,
// for the step the host samples next. The host's selection and argmax take
// them in place of reading the logits, which a step ahead may have
// overwritten by then.
struct StepHandoff {
  const int32_t* ids = nullptr;
  const float* values = nullptr;
  const int32_t* flags = nullptr;
  int32_t k = 0;
};

class StepHandoffRegistry {
 public:
  static StepHandoffRegistry& Get() {
    static auto* registry = new StepHandoffRegistry;
    return *registry;
  }
  void Set(const void* logits, const StepHandoff& handoff) {
    std::lock_guard<std::mutex> lock(mutex_);
    handoffs_[logits] = handoff;
  }
  void Clear(const void* logits) {
    std::lock_guard<std::mutex> lock(mutex_);
    handoffs_.erase(logits);
  }
  bool Find(const void* logits, StepHandoff* handoff) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = handoffs_.find(logits);
    if (it == handoffs_.end()) return false;
    *handoff = it->second;
    return true;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<const void*, StepHandoff> handoffs_;
};

// The registered table [vocab, dim]; none notes that a feed awaits one.
std::optional<RegisteredTokenTable> FindTokenTable(uint32_t vocab,
                                                   uint32_t dim) {
  auto& registry = GetTokenTableRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (const auto& table : registry.tables) {
    if (table.vocab == vocab && table.dim == dim) return table;
  }
  const std::pair<uint32_t, uint32_t> key(vocab, dim);
  if (std::find(registry.awaited.begin(), registry.awaited.end(), key) ==
      registry.awaited.end()) {
    registry.awaited.push_back(key);
  }
  return std::nullopt;
}

void DropFileBackedBytecodePages(const LiteRtMemBuffer& buffer) {
  if (buffer.fd < 0 || buffer.base_addr == nullptr || buffer.size == 0) {
    return;
  }
  const int64_t page_size = static_cast<int64_t>(sysconf(_SC_PAGESIZE));
  if (page_size <= 0) {
    return;
  }
  const uintptr_t page_mask = static_cast<uintptr_t>(page_size - 1);
  const uintptr_t bytecode_begin =
      reinterpret_cast<uintptr_t>(buffer.base_addr) + buffer.offset;
  const uintptr_t bytecode_end = bytecode_begin + buffer.size;
  if (bytecode_end < bytecode_begin) {
    return;
  }
  const uintptr_t aligned_begin = bytecode_begin & ~page_mask;
  const uintptr_t aligned_end = (bytecode_end + page_mask) & ~page_mask;
  if (aligned_end <= aligned_begin) {
    return;
  }
  const size_t aligned_size = aligned_end - aligned_begin;
  if (madvise(reinterpret_cast<void*>(aligned_begin), aligned_size,
              MADV_DONTNEED) == 0) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch released file-backed bytecode pages "
               "(bytes=%zu)",
               aligned_size);
  } else {
    LITERT_LOG(LITERT_WARNING,
               "NVIDIA dispatch could not release file-backed bytecode "
               "pages: %s",
               std::strerror(errno));
  }
}

// IEEE binary16 bits to float (the dumps' half-stored values).
float HalfBitsToFloat(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1fu;
  uint32_t mantissa = h & 0x3ffu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3ffu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

LiteRtStatus CudaStatus(cudaError_t error, const char* what) {
  if (error == cudaSuccess) {
    return kLiteRtStatusOk;
  }
  LITERT_LOG(LITERT_ERROR, "%s: %s", what, cudaGetErrorString(error));
  return kLiteRtStatusErrorRuntimeFailure;
}

Expected<nvinfer1::DataType> ToTensorRtDataType(
    litert::nvidia::TensorRtWeightDataType data_type) {
  switch (data_type) {
    case litert::nvidia::TensorRtWeightDataType::kFloat:
      return nvinfer1::DataType::kFLOAT;
    case litert::nvidia::TensorRtWeightDataType::kHalf:
      return nvinfer1::DataType::kHALF;
    case litert::nvidia::TensorRtWeightDataType::kInt8:
      return nvinfer1::DataType::kINT8;
    case litert::nvidia::TensorRtWeightDataType::kInt32:
      return nvinfer1::DataType::kINT32;
    case litert::nvidia::TensorRtWeightDataType::kBool:
      return nvinfer1::DataType::kBOOL;
    case litert::nvidia::TensorRtWeightDataType::kUint8:
      return nvinfer1::DataType::kUINT8;
    case litert::nvidia::TensorRtWeightDataType::kFp8:
      return nvinfer1::DataType::kFP8;
    case litert::nvidia::TensorRtWeightDataType::kBf16:
      return nvinfer1::DataType::kBF16;
    case litert::nvidia::TensorRtWeightDataType::kInt64:
      return nvinfer1::DataType::kINT64;
    case litert::nvidia::TensorRtWeightDataType::kInt4:
      return nvinfer1::DataType::kINT4;
    case litert::nvidia::TensorRtWeightDataType::kFp4:
      return nvinfer1::DataType::kFP4;
    case litert::nvidia::TensorRtWeightDataType::kE8m0:
      return nvinfer1::DataType::kE8M0;
  }
  return Error(kLiteRtStatusErrorInvalidArgument,
               "Unknown TensorRT refit weight type");
}

struct CudaTensorBufferInfo : HwMemoryInfo {
  // The device allocation the producer asked for: below the tensor's packed
  // size when the engine stores the tensor more compactly (a float32 KV cache
  // kept as FP16).
  size_t bytes = 0;
  // The tensor's packed size: the bytes the host sees through Lock.
  size_t packed_bytes = 0;
  void* host_cache = nullptr;
  bool host_cache_pinned = false;
  LiteRtTensorBufferLockMode lock_mode = kLiteRtTensorBufferLockModeRead;
  bool locked = false;
  bool owns_handle = true;
  // The host wrote host_cache after the device copy was last current.
  bool host_dirty = false;
  // host_cache holds the device copy as of the last step (the dispatch
  // copied it): a read lock needs no download.
  bool host_current = false;
};

struct NvidiaGreedySamplerContext {
  const LiteRtRuntimeContext* runtime_context = nullptr;
  int device = -1;
  cudaStream_t stream = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  int32_t* device_result = nullptr;
  int32_t* host_result = nullptr;
  size_t result_capacity = 0;
  uint64_t calls = 0;
  // Calls answered from the argmax the invocation ran itself (FusedGreedy).
  uint64_t fused_calls = 0;
};

CudaTensorBufferInfo* AsCudaTensorBufferInfo(HwMemoryInfoPtr info) {
  return static_cast<CudaTensorBufferInfo*>(info);
}

size_t NonZeroBytes(size_t bytes) { return std::max<size_t>(bytes, 1); }

// The bytes the host sees through Lock. LiteRT reads, writes and clears a
// locked buffer at the tensor's packed size, whatever the device holds.
size_t HostBytes(const CudaTensorBufferInfo& info) {
  return info.packed_bytes == 0 ? info.bytes : info.packed_bytes;
}

// The bytes copied between the host cache and the device, and cleared: the
// host's view, no more than the device allocation.
size_t CopyBytes(const CudaTensorBufferInfo& info) {
  const size_t host_bytes = HostBytes(info);
  return info.bytes == 0 ? host_bytes : std::min(host_bytes, info.bytes);
}

bool PinnedHostStagingEnabled() {
  const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_PINNED_HOST_STAGING");
  return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

// What the host writes to a CUDA tensor buffer reaches the device when an
// invocation reads the buffer, not when the host unlocks it: a runtime that
// locks a buffer several times before it invokes uploads it once. LiteRT-LM
// clears and then fills its attention masks before every prefill, 127 MiB
// each time at a capacity of 130048 tokens.
// LITERT_NVIDIA_DISPATCH_DEFERRED_UPLOAD=0 uploads at every unlock.
bool DeferredUploadEnabled() {
  const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_DEFERRED_UPLOAD");
  return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

// The CUDA tensor buffers by device pointer, for the invocations that read
// and write them through that pointer.
class CudaTensorBufferRegistry {
 public:
  static CudaTensorBufferRegistry& Get() {
    static auto* registry = new CudaTensorBufferRegistry;
    return *registry;
  }

  void Add(CudaTensorBufferInfo* info) {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_.emplace(info->memory_handle, info);
  }

  void Remove(CudaTensorBufferInfo* info) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto range = buffers_.equal_range(info->memory_handle);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == info) {
        buffers_.erase(it);
        return;
      }
    }
  }

  void MarkHostWrite(CudaTensorBufferInfo* info) {
    std::lock_guard<std::mutex> lock(mutex_);
    info->host_dirty = true;
    info->host_current = false;
  }

  // Whether the host copy is newer than the device copy.
  bool HostIsNewer(CudaTensorBufferInfo* info) {
    std::lock_guard<std::mutex> lock(mutex_);
    return info->host_dirty;
  }

  // Whether the host copy holds the device copy (a read needs no download).
  bool HostIsCurrent(CudaTensorBufferInfo* info) {
    std::lock_guard<std::mutex> lock(mutex_);
    return info->host_current;
  }

  // The host copy of the buffer at `handle` with at least `bytes` bytes, or
  // null: the dispatch's snapshot of the device copy goes there. `pinned`
  // tells whether a stream copy into it is asynchronous.
  void* HostCache(void* handle, size_t bytes, bool* pinned = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto range = buffers_.equal_range(handle);
    for (auto it = range.first; it != range.second; ++it) {
      CudaTensorBufferInfo* info = it->second;
      if (info->host_cache != nullptr && !info->host_dirty &&
          CopyBytes(*info) >= bytes) {
        if (pinned != nullptr) *pinned = info->host_cache_pinned;
        return info->host_cache;
      }
    }
    return nullptr;
  }

  // The host copy of the buffer at `handle` holds the device copy.
  void MarkHostCurrent(void* handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto range = buffers_.equal_range(handle);
    for (auto it = range.first; it != range.second; ++it) {
      if (!it->second->host_dirty) {
        it->second->host_current = true;
      }
    }
  }

  // Uploads what the host wrote to the buffers at `handle`, in `stream`
  // order. Returns the bytes uploaded in `bytes`.
  LiteRtStatus Upload(void* handle, cudaStream_t stream, size_t* bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto range = buffers_.equal_range(handle);
    for (auto it = range.first; it != range.second; ++it) {
      CudaTensorBufferInfo* info = it->second;
      if (!info->host_dirty || info->host_cache == nullptr) {
        continue;
      }
      const size_t copy_bytes = CopyBytes(*info);
      // Pageable memory is staged before cudaMemcpyAsync returns; pinned
      // memory must stay as it is until the stream gets to the copy, which
      // the invocation waits for.
      const cudaError_t status =
          cudaMemcpyAsync(info->memory_handle, info->host_cache, copy_bytes,
                          cudaMemcpyHostToDevice, stream);
      if (status != cudaSuccess) {
        LITERT_LOG(LITERT_ERROR, "cudaMemcpyAsync deferred H2D failed: %s",
                   cudaGetErrorString(status));
        return kLiteRtStatusErrorRuntimeFailure;
      }
      info->host_dirty = false;
      if (bytes != nullptr) {
        *bytes += copy_bytes;
      }
    }
    return kLiteRtStatusOk;
  }

  // Whether any buffer at `handle` has a host copy newer than the device's.
  bool AnyHostNewer(void* handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto range = buffers_.equal_range(handle);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second->host_dirty) {
        return true;
      }
    }
    return false;
  }

  // The device copy of the buffers at `handle` is about to be overwritten.
  void Discard(void* handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto range = buffers_.equal_range(handle);
    for (auto it = range.first; it != range.second; ++it) {
      it->second->host_dirty = false;
      it->second->host_current = false;
    }
  }

  // Copies the first `bytes` bytes of the host copy of the buffer at
  // `handle` (what the host last wrote, or the device copy as last downloaded)
  // to `out`. False without a host copy that long.
  bool ReadHostCopy(void* handle, void* out, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto range = buffers_.equal_range(handle);
    for (auto it = range.first; it != range.second; ++it) {
      CudaTensorBufferInfo* info = it->second;
      if (info->host_cache == nullptr || CopyBytes(*info) < bytes) {
        continue;
      }
      std::memcpy(out, info->host_cache, bytes);
      return true;
    }
    return false;
  }

  // Whether what the host wrote to the buffers at `handle`, if anything, is
  // the `bytes` bytes at `expected` (the device copy already holds them): a
  // dirty copy that matches is clean afterwards; one that differs stays
  // dirty. The first `bytes` bytes of the host copy are compared.
  bool VerifyHostWrite(void* handle, const void* expected, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto range = buffers_.equal_range(handle);
    for (auto it = range.first; it != range.second; ++it) {
      CudaTensorBufferInfo* info = it->second;
      if (!info->host_dirty) {
        continue;
      }
      if (info->host_cache == nullptr || CopyBytes(*info) < bytes ||
          std::memcmp(info->host_cache, expected, bytes) != 0) {
        return false;
      }
      info->host_dirty = false;
    }
    return true;
  }

 private:
  std::mutex mutex_;
  std::unordered_multimap<void*, CudaTensorBufferInfo*> buffers_;
};

// Greedy sampling fused into the invocation. Once a sampler has read a logits
// buffer (SampleNvidiaGreedy), the invocation that next writes that buffer
// runs the argmax on its own stream, before its synchronization, into the
// sampler's result buffers; the sampler's next call takes that result without
// a launch, a copy or a synchronization of its own (each costs a host round
// trip on WSL). A host write to the logits in between (a constrained decoder)
// leaves the host copy newer, and the sampler takes its usual path.
// LITERT_NVIDIA_DISPATCH_FUSED_SAMPLING=0 turns the fusion off.
bool FusedSamplingEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_FUSED_SAMPLING");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

struct FusedGreedyEntry {
  NvidiaGreedySamplerContext* context = nullptr;
  size_t rows = 0;
  size_t count = 0;
  bool fp16 = false;
  // The context's host_result holds the argmax of the buffer's current
  // contents.
  bool fresh = false;
};

// The logits buffers samplers read, by device pointer.
class FusedGreedyRegistry {
 public:
  static FusedGreedyRegistry& Get() {
    static auto* registry = new FusedGreedyRegistry;
    return *registry;
  }

  void Register(const void* logits, const FusedGreedyEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_[logits] = entry;
  }

  bool Find(const void* logits, FusedGreedyEntry* entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(logits);
    if (it == entries_.end()) {
      return false;
    }
    *entry = it->second;
    return true;
  }

  // Takes the fresh result for `logits` if it is `context`'s and matches the
  // shape; the entry is then stale until the next invocation.
  bool TakeFresh(const void* logits, const NvidiaGreedySamplerContext* context,
                 size_t rows, size_t count, bool fp16) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(logits);
    if (it == entries_.end() || !it->second.fresh ||
        it->second.context != context || it->second.rows != rows ||
        it->second.count != count || it->second.fp16 != fp16) {
      return false;
    }
    it->second.fresh = false;
    return true;
  }

  void SetFresh(const void* logits, bool fresh) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(logits);
    if (it != entries_.end()) {
      it->second.fresh = fresh;
    }
  }

  void ForgetContext(const NvidiaGreedySamplerContext* context) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
      it = it->second.context == context ? entries_.erase(it) : std::next(it);
    }
  }

  // A device allocation is going away: its views must not be sampled.
  void ForgetRange(const void* begin, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const uintptr_t first = reinterpret_cast<uintptr_t>(begin);
    for (auto it = entries_.begin(); it != entries_.end();) {
      const uintptr_t address = reinterpret_cast<uintptr_t>(it->first);
      it = address >= first && address - first < std::max<size_t>(bytes, 1)
               ? entries_.erase(it)
               : std::next(it);
    }
  }

 private:
  std::mutex mutex_;
  std::unordered_map<const void*, FusedGreedyEntry> entries_;
};

LiteRtStatus ValidateCudaDevicePointer(void* device_ptr) {
  if (device_ptr == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (reinterpret_cast<uintptr_t>(device_ptr) % 256 != 0) {
    LITERT_LOG(LITERT_ERROR,
               "Imported CUDA tensor buffer is not 256-byte aligned");
    return kLiteRtStatusErrorInvalidArgument;
  }

  cudaPointerAttributes attributes;
  const cudaError_t status = cudaPointerGetAttributes(&attributes, device_ptr);
  if (status != cudaSuccess) {
    const LiteRtStatus litert_status =
        CudaStatus(status, "cudaPointerGetAttributes CUDA tensor buffer");
    cudaGetLastError();
    return litert_status;
  }
  if (attributes.devicePointer == nullptr ||
      (attributes.type != cudaMemoryTypeDevice &&
       attributes.type != cudaMemoryTypeManaged)) {
    LITERT_LOG(LITERT_ERROR,
               "Imported CUDA tensor buffer must be device or managed memory");
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus CreateCudaTensorBuffer(LiteRtGpuDeviceId device_id,
                                    LiteRtGpuQueueId queue_id,
                                    const LiteRtRankedTensorType* tensor_type,
                                    LiteRtTensorBufferType buffer_type,
                                    size_t bytes, size_t packed_bytes,
                                    HwMemoryInfoPtr* hw_memory_info) {
  (void)device_id;
  (void)queue_id;
  (void)tensor_type;
  if (hw_memory_info == nullptr || buffer_type != kNvidiaCudaTensorBufferType) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto* info = new (std::nothrow) CudaTensorBufferInfo;
  if (info == nullptr) {
    return kLiteRtStatusErrorMemoryAllocationFailure;
  }
  info->bytes = bytes;
  info->packed_bytes = packed_bytes;
  // The buffer size comes from the producer's requirements; when it is
  // smaller than the tensor's packed size the producer stores the tensor
  // more compactly (a float32 KV cache kept as FP16), and the device
  // allocation, its copies and its clears are that size. The host cache
  // (Lock) stays at the tensor's packed size.
  void* device_ptr = nullptr;
  const LiteRtStatus status = CudaStatus(
      cudaMalloc(&device_ptr, NonZeroBytes(bytes != 0 ? bytes : packed_bytes)),
      "cudaMalloc CUDA tensor buffer");
  if (status != kLiteRtStatusOk) {
    delete info;
    return status;
  }
  info->memory_handle = device_ptr;
  info->raw_handle = device_ptr;
  info->owns_handle = true;
  CudaTensorBufferRegistry::Get().Add(info);
  *hw_memory_info = info;
  return kLiteRtStatusOk;
}

LiteRtStatus ImportCudaTensorBuffer(LiteRtGpuDeviceId device_id,
                                    LiteRtGpuQueueId queue_id,
                                    const LiteRtRankedTensorType* tensor_type,
                                    LiteRtTensorBufferType buffer_type,
                                    HwMemoryHandle hw_buffer_handle,
                                    size_t bytes, size_t packed_bytes,
                                    HwMemoryInfoPtr* hw_memory_info) {
  (void)device_id;
  (void)queue_id;
  (void)tensor_type;
  if (hw_memory_info == nullptr || buffer_type != kNvidiaCudaTensorBufferType) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtStatus status = ValidateCudaDevicePointer(hw_buffer_handle);
  if (status != kLiteRtStatusOk) {
    return status;
  }
  auto* info = new (std::nothrow) CudaTensorBufferInfo;
  if (info == nullptr) {
    return kLiteRtStatusErrorMemoryAllocationFailure;
  }
  info->memory_handle = hw_buffer_handle;
  info->raw_handle = hw_buffer_handle;
  info->bytes = bytes;
  info->packed_bytes = packed_bytes;
  info->owns_handle = false;
  CudaTensorBufferRegistry::Get().Add(info);
  *hw_memory_info = info;
  return kLiteRtStatusOk;
}

LiteRtStatus DestroyCudaTensorBuffer(HwMemoryInfoPtr hw_memory_info) {
  auto* info = AsCudaTensorBufferInfo(hw_memory_info);
  if (info == nullptr) {
    return kLiteRtStatusOk;
  }
  CudaTensorBufferRegistry::Get().Remove(info);
  if (info->memory_handle != nullptr) {
    // A sampler that still reads another view of the allocation registers
    // again on its next call.
    FusedGreedyRegistry::Get().ForgetRange(info->memory_handle, info->bytes);
  }
  LiteRtStatus status = kLiteRtStatusOk;
  if (info->host_cache != nullptr) {
    if (info->host_cache_pinned) {
      LiteRtStatus free_host_status =
          CudaStatus(cudaFreeHost(info->host_cache), "cudaFreeHost");
      if (free_host_status != kLiteRtStatusOk) {
        status = free_host_status;
      }
    } else {
      std::free(info->host_cache);
    }
  }
  if (info->owns_handle && info->memory_handle != nullptr) {
    LiteRtStatus free_device_status = CudaStatus(cudaFree(info->memory_handle),
                                                 "cudaFree CUDA tensor buffer");
    if (free_device_status != kLiteRtStatusOk) {
      status = free_device_status;
    }
  }
  delete info;
  return status;
}

LiteRtStatus LockCudaTensorBuffer(HwMemoryInfoPtr hw_memory_info,
                                  LiteRtTensorBufferLockMode mode,
                                  void** host_memory_ptr) {
  auto* info = AsCudaTensorBufferInfo(hw_memory_info);
  if (info == nullptr || host_memory_ptr == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (info->locked) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  const size_t copy_bytes = CopyBytes(*info);
  if (info->host_cache == nullptr) {
    const size_t alloc_bytes = NonZeroBytes(HostBytes(*info));
    if (PinnedHostStagingEnabled()) {
      const cudaError_t error = cudaMallocHost(&info->host_cache, alloc_bytes);
      if (error == cudaSuccess) {
        info->host_cache_pinned = true;
      } else {
        LITERT_LOG(LITERT_WARNING,
                   "cudaMallocHost CUDA tensor buffer staging failed for "
                   "%zu bytes (%s); falling back to pageable host staging",
                   alloc_bytes, cudaGetErrorString(error));
        cudaGetLastError();
      }
    }
    if (info->host_cache == nullptr) {
      info->host_cache = std::malloc(alloc_bytes);
      if (info->host_cache == nullptr) {
        return kLiteRtStatusErrorMemoryAllocationFailure;
      }
      info->host_cache_pinned = false;
    }
    // The bytes past the device's are the host's alone: read as zeros.
    if (alloc_bytes > copy_bytes) {
      std::memset(static_cast<char*>(info->host_cache) + copy_bytes, 0,
                  alloc_bytes - copy_bytes);
    }
  }
  // A host copy that is newer than the device copy, or that already holds
  // it, is what the host reads.
  if (copy_bytes > 0 &&
      (mode == kLiteRtTensorBufferLockModeRead ||
       mode == kLiteRtTensorBufferLockModeReadWrite) &&
      !CudaTensorBufferRegistry::Get().HostIsNewer(info) &&
      !CudaTensorBufferRegistry::Get().HostIsCurrent(info)) {
    LiteRtStatus status =
        CudaStatus(cudaMemcpy(info->host_cache, info->memory_handle, copy_bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy CUDA tensor buffer D2H lock");
    if (status != kLiteRtStatusOk) {
      return status;
    }
  }
  info->lock_mode = mode;
  info->locked = true;
  *host_memory_ptr = info->host_cache;
  return kLiteRtStatusOk;
}

LiteRtStatus UnlockCudaTensorBuffer(HwMemoryInfoPtr hw_memory_info) {
  auto* info = AsCudaTensorBufferInfo(hw_memory_info);
  if (info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!info->locked) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  LiteRtStatus status = kLiteRtStatusOk;
  const size_t copy_bytes = CopyBytes(*info);
  if (copy_bytes > 0 &&
      (info->lock_mode == kLiteRtTensorBufferLockModeWrite ||
       info->lock_mode == kLiteRtTensorBufferLockModeReadWrite)) {
    if (DeferredUploadEnabled()) {
      CudaTensorBufferRegistry::Get().MarkHostWrite(info);
    } else {
      // The device contents change without an invocation: no fused argmax
      // of them can be fresh (a deferred write says so through host_dirty).
      FusedGreedyRegistry::Get().ForgetRange(info->memory_handle,
                                             std::max(info->bytes, copy_bytes));
      status = CudaStatus(cudaMemcpy(info->memory_handle, info->host_cache,
                                     copy_bytes, cudaMemcpyHostToDevice),
                          "cudaMemcpy CUDA tensor buffer H2D unlock");
    }
  }
  info->locked = false;
  return status;
}

LiteRtStatus ClearCudaTensorBuffer(HwMemoryInfoPtr hw_memory_info) {
  auto* info = AsCudaTensorBufferInfo(hw_memory_info);
  if (info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const size_t copy_bytes = CopyBytes(*info);
  if (copy_bytes == 0) {
    return kLiteRtStatusOk;
  }
  CudaTensorBufferRegistry::Get().Discard(info->memory_handle);
  // The device contents change without an invocation: no fused argmax of them
  // can be fresh.
  FusedGreedyRegistry::Get().ForgetRange(info->memory_handle, info->bytes);
  return CudaStatus(cudaMemset(info->memory_handle, 0, copy_bytes),
                    "cudaMemset CUDA tensor buffer");
}

Expected<size_t> TensorBufferPackedSize(
    const LiteRtRuntimeContext* runtime_context, LiteRtTensorBuffer buffer) {
  size_t size = 0;
  LiteRtStatus status =
      runtime_context->get_tensor_buffer_packed_size(buffer, &size);
  if (status == kLiteRtStatusOk) {
    return size;
  }
  LITERT_RETURN_IF_ERROR(
      runtime_context->get_tensor_buffer_size(buffer, &size));
  return size;
}

void DestroyNvidiaGreedySamplerResources(NvidiaGreedySamplerContext* context) {
  if (context == nullptr) {
    return;
  }
  int previous_device = -1;
  const bool restore_device = context->device >= 0 &&
                              cudaGetDevice(&previous_device) == cudaSuccess &&
                              previous_device != context->device;
  if (restore_device) {
    cudaSetDevice(context->device);
  }
  if (context->stream != nullptr) {
    cudaStreamSynchronize(context->stream);
  }
  if (context->workspace != nullptr) {
    cudaFree(context->workspace);
  }
  if (context->device_result != nullptr) {
    cudaFree(context->device_result);
  }
  if (context->stream != nullptr) {
    cudaStreamDestroy(context->stream);
  }
  if (context->host_result != nullptr) {
    cudaFreeHost(context->host_result);
  }
  if (restore_device) {
    cudaSetDevice(previous_device);
  }
  FusedGreedyRegistry::Get().ForgetContext(context);
  if (context->calls > 0) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA CUDA greedy sampler completed %llu calls (%llu from "
               "the invocation's own argmax)",
               static_cast<unsigned long long>(context->calls),
               static_cast<unsigned long long>(context->fused_calls));
  }
}

LiteRtStatus EnsureNvidiaGreedySamplerResources(
    NvidiaGreedySamplerContext* context, int device, size_t count,
    size_t rows) {
  if (context->device >= 0 && context->device != device) {
    return kLiteRtStatusErrorUnsupported;
  }
  if (context->device < 0) {
    int compute_capability_major = 0;
    const cudaError_t capability_status = cudaDeviceGetAttribute(
        &compute_capability_major, cudaDevAttrComputeCapabilityMajor, device);
    if (capability_status != cudaSuccess) {
      return CudaStatus(capability_status,
                        "cudaDeviceGetAttribute CUDA greedy sampler");
    }
    // The archive carries compute_80 PTX. Older devices can still use the
    // TensorRT-RTX path, so keep them on the CPU sampler.
    if (compute_capability_major < 8) {
      return kLiteRtStatusErrorUnsupported;
    }

    cudaStream_t stream = nullptr;
    int32_t* device_result = nullptr;
    int32_t* host_result = nullptr;
    auto clean_up_partial_initialization = [&] {
      if (host_result != nullptr) {
        cudaFreeHost(host_result);
      }
      if (device_result != nullptr) {
        cudaFree(device_result);
      }
      if (stream != nullptr) {
        cudaStreamDestroy(stream);
      }
    };
    cudaError_t cuda_status =
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      clean_up_partial_initialization();
      return CudaStatus(cuda_status,
                        "cudaStreamCreateWithFlags CUDA greedy sampler");
    }
    cuda_status = cudaMalloc(reinterpret_cast<void**>(&device_result),
                             rows * sizeof(int32_t));
    if (cuda_status != cudaSuccess) {
      clean_up_partial_initialization();
      return CudaStatus(cuda_status, "cudaMalloc CUDA greedy sampler result");
    }
    cuda_status = cudaMallocHost(reinterpret_cast<void**>(&host_result),
                                 rows * sizeof(int32_t));
    if (cuda_status != cudaSuccess) {
      clean_up_partial_initialization();
      return CudaStatus(cuda_status,
                        "cudaMallocHost CUDA greedy sampler result");
    }
    context->device = device;
    context->stream = stream;
    context->device_result = device_result;
    context->host_result = host_result;
    context->result_capacity = rows;
    LITERT_LOG(LITERT_INFO,
               "NVIDIA CUDA greedy sampler initialized on device %d", device);
  }
  if (context->result_capacity < rows) {
    // Previous calls synchronize before returning. Keep old allocations valid
    // until both larger result buffers have been allocated successfully.
    int32_t* device_result = nullptr;
    int32_t* host_result = nullptr;
    LITERT_RETURN_IF_ERROR(
        CudaStatus(cudaMalloc(reinterpret_cast<void**>(&device_result),
                              rows * sizeof(int32_t)),
                   "cudaMalloc CUDA greedy sampler batched results"));
    const cudaError_t allocation_status = cudaMallocHost(
        reinterpret_cast<void**>(&host_result), rows * sizeof(int32_t));
    if (allocation_status != cudaSuccess) {
      cudaFree(device_result);
      return CudaStatus(allocation_status,
                        "cudaMallocHost CUDA greedy sampler batched results");
    }
    std::swap(context->device_result, device_result);
    std::swap(context->host_result, host_result);
    context->result_capacity = rows;
    const cudaError_t device_free_status = cudaFree(device_result);
    const cudaError_t host_free_status = cudaFreeHost(host_result);
    LITERT_RETURN_IF_ERROR(CudaStatus(
        device_free_status, "cudaFree CUDA greedy sampler old results"));
    LITERT_RETURN_IF_ERROR(CudaStatus(
        host_free_status, "cudaFreeHost CUDA greedy sampler old results"));
  }
  const size_t required = LiteRtNvidiaArgMaxRowsWorkspaceBytes(rows, count);
  if (required == 0) {
    return kLiteRtStatusErrorUnsupported;
  }
  if (context->workspace_bytes < required) {
    if (context->workspace != nullptr) {
      LITERT_RETURN_IF_ERROR(
          CudaStatus(cudaFree(context->workspace),
                     "cudaFree CUDA greedy sampler workspace"));
      context->workspace = nullptr;
      context->workspace_bytes = 0;
    }
    LITERT_RETURN_IF_ERROR(
        CudaStatus(cudaMalloc(&context->workspace, required),
                   "cudaMalloc CUDA greedy sampler workspace"));
    context->workspace_bytes = required;
  }
  return kLiteRtStatusOk;
}

// If a sampler reads the logits at `device_ptr`, enqueues their argmax and the
// copy of the ids to the sampler's host buffer on `stream`, and lists the
// pointer in `enqueued` (SetFresh once the stream has synchronized). The
// sampler's first, direct call set the context's device and registered the
// entry; a resource failure leaves the entry stale and the sampler on its own
// path, a launch failure is an error of the invocation.
// `host_destination` (null: the sampler's own host result) takes the ids and
// `keep_fresh` leaves the sampler's result as it is: a step run ahead of the
// host (StepFeed) must not stale the result the host is about to take.
Expected<void> EnqueueFusedGreedy(void* device_ptr, cudaStream_t stream,
                                  std::vector<const void*>* enqueued,
                                  int32_t* host_destination = nullptr,
                                  bool keep_fresh = false) {
  FusedGreedyEntry entry;
  if (device_ptr == nullptr ||
      !FusedGreedyRegistry::Get().Find(device_ptr, &entry) ||
      entry.context == nullptr || entry.context->device < 0) {
    return {};
  }
  NvidiaGreedySamplerContext* context = entry.context;
  if (!keep_fresh) {
    FusedGreedyRegistry::Get().SetFresh(device_ptr, false);
  }
  if (EnsureNvidiaGreedySamplerResources(context, context->device, entry.count,
                                         entry.rows) != kLiteRtStatusOk) {
    return {};
  }
  const cudaError_t launch_status =
      entry.fp16
          ? LiteRtNvidiaLaunchF16ArgMaxRows(
                static_cast<const uint16_t*>(device_ptr), entry.rows,
                entry.count, context->workspace, context->workspace_bytes,
                context->device_result, stream)
          : LiteRtNvidiaLaunchF32ArgMaxRows(
                static_cast<const float*>(device_ptr), entry.rows, entry.count,
                context->workspace, context->workspace_bytes,
                context->device_result, stream);
  LITERT_RETURN_IF_ERROR(CudaOk(launch_status, "Launch fused greedy argmax"));
  LITERT_RETURN_IF_ERROR(CudaOk(
      cudaMemcpyAsync(
          host_destination != nullptr ? host_destination : context->host_result,
          context->device_result, entry.rows * sizeof(int32_t),
          cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync fused greedy result"));
  enqueued->push_back(device_ptr);
  return {};
}

// IEEE half bits to float, on the host (the diagnostics).
float __half2float_host(uint16_t bits) {
  const uint32_t sign = (bits >> 15) & 1;
  const uint32_t exponent = (bits >> 10) & 0x1f;
  const uint32_t mantissa = bits & 0x3ff;
  float value;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 0x1f) {
    value = mantissa == 0 ? INFINITY : NAN;
  } else {
    value = std::ldexp(static_cast<float>(mantissa | 0x400),
                       static_cast<int>(exponent) - 25);
  }
  return sign ? -value : value;
}

// The step feed (bytecode.h, TensorRtStepFeed): a decode step run on the
// device before the host asks for it. LITERT_NVIDIA_DISPATCH_STEP_FEED=0
// keeps every step waiting for the host.
bool StepFeedEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_STEP_FEED");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

// The feed stops when the steps run ahead miss too often: at first when the
// misses outnumber the hits (an embedder the table is not tied to, an input
// the feed does not write), later when more than this share of the steps
// missed (a sampler that picks the argmax most of the time, not always): a
// miss wastes every step ahead, more than the hits between them save.
constexpr uint64_t kStepFeedMaxMisses = 4;
constexpr uint64_t kStepFeedMissPercent = 3;
constexpr uint64_t kStepFeedMissWindow = 64;

struct ResolvedSamplerLogits {
  HwMemoryHandle handle = nullptr;
  void* device_logits = nullptr;
  size_t allocation_size = 0;
  bool fp16 = false;
  int device = -1;
};

// A dense [1, rows, count] float32 (or, unless legacy_f32_only, float16)
// NVIDIA CUDA buffer, after its producer's event: its handle, its device
// pointer and its device. Unsupported buffers, element types and layouts are
// kLiteRtStatusErrorUnsupported, so a caller can keep its host sampler.
LiteRtStatus ResolveSamplerLogits(const LiteRtRuntimeContext* runtime_context,
                                  LiteRtTensorBuffer logits, size_t rows,
                                  size_t count, bool legacy_f32_only,
                                  ResolvedSamplerLogits* resolved) {
  if (runtime_context->get_tensor_buffer_type == nullptr ||
      runtime_context->get_tensor_buffer_tensor_type == nullptr ||
      runtime_context->get_tensor_buffer_packed_size == nullptr ||
      runtime_context->get_tensor_buffer_size == nullptr ||
      runtime_context->get_tensor_buffer_offset == nullptr ||
      runtime_context->has_tensor_buffer_event == nullptr ||
      runtime_context->get_tensor_buffer_event == nullptr ||
      runtime_context->wait_event == nullptr ||
      runtime_context->get_tensor_buffer_custom_tensor_buffer_handle ==
          nullptr) {
    return kLiteRtStatusErrorUnsupported;
  }
  LiteRtTensorBufferType buffer_type;
  LITERT_RETURN_IF_ERROR(
      runtime_context->get_tensor_buffer_type(logits, &buffer_type));
  if (buffer_type != kNvidiaCudaTensorBufferType) {
    return kLiteRtStatusErrorUnsupported;
  }
  LiteRtRankedTensorType tensor_type;
  LITERT_RETURN_IF_ERROR(
      runtime_context->get_tensor_buffer_tensor_type(logits, &tensor_type));
  const bool fp16 = tensor_type.element_type == kLiteRtElementTypeFloat16;
  if ((tensor_type.element_type != kLiteRtElementTypeFloat32 &&
       (legacy_f32_only || !fp16)) ||
      tensor_type.layout.rank != 3 || tensor_type.layout.dimensions[0] != 1 ||
      tensor_type.layout.dimensions[1] != static_cast<int32_t>(rows) ||
      tensor_type.layout.dimensions[2] != static_cast<int32_t>(count) ||
      (legacy_f32_only ? tensor_type.layout.has_strides
                       : !litert::nvidia::TensorBufferLayoutMatches(
                             tensor_type.layout, false))) {
    return kLiteRtStatusErrorUnsupported;
  }
  size_t packed_size = 0;
  size_t allocation_size = 0;
  size_t offset = 0;
  LITERT_RETURN_IF_ERROR(
      runtime_context->get_tensor_buffer_packed_size(logits, &packed_size));
  LITERT_RETURN_IF_ERROR(
      runtime_context->get_tensor_buffer_offset(logits, &offset));
  LITERT_RETURN_IF_ERROR(
      runtime_context->get_tensor_buffer_size(logits, &allocation_size));
  const size_t element_bytes = fp16 ? sizeof(uint16_t) : sizeof(float);
  if (packed_size != rows * count * element_bytes) {
    return kLiteRtStatusErrorUnsupported;
  }
  bool has_event = false;
  LITERT_RETURN_IF_ERROR(
      runtime_context->has_tensor_buffer_event(logits, &has_event));
  if (has_event) {
    LiteRtEvent event = nullptr;
    LITERT_RETURN_IF_ERROR(
        runtime_context->get_tensor_buffer_event(logits, &event));
    if (event != nullptr) {
      LITERT_RETURN_IF_ERROR(
          runtime_context->wait_event(event, /*timeout_in_ms=*/-1));
    }
  }
  HwMemoryHandle handle = nullptr;
  LITERT_RETURN_IF_ERROR(
      runtime_context->get_tensor_buffer_custom_tensor_buffer_handle(logits,
                                                                     &handle));
  if (handle == nullptr) {
    return kLiteRtStatusErrorUnsupported;
  }
  LITERT_ASSIGN_OR_RETURN(auto* device_logits,
                          litert::nvidia::ResolveCudaTensorBufferView(
                              handle, allocation_size, offset, packed_size));
  if (reinterpret_cast<uintptr_t>(device_logits) % element_bytes != 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  cudaPointerAttributes attributes{};
  const cudaError_t attributes_status =
      cudaPointerGetAttributes(&attributes, device_logits);
  if (attributes_status != cudaSuccess || attributes.devicePointer == nullptr ||
      (attributes.type != cudaMemoryTypeDevice &&
       attributes.type != cudaMemoryTypeManaged)) {
    cudaGetLastError();
    return kLiteRtStatusErrorUnsupported;
  }
  resolved->handle = handle;
  resolved->device_logits = device_logits;
  resolved->allocation_size = allocation_size;
  resolved->fp16 = fp16;
  resolved->device = attributes.device;
  return kLiteRtStatusOk;
}

// The k largest logits of every row (topk_candidates_c_api.h).
struct NvidiaTopKCandidatesContext {
  const LiteRtRuntimeContext* runtime_context = nullptr;
  int device = -1;
  cudaStream_t stream = nullptr;
  int32_t* device_ids = nullptr;
  float* device_values = nullptr;
  int32_t* device_flags = nullptr;
  int32_t* host_ids = nullptr;
  float* host_values = nullptr;
  int32_t* host_flags = nullptr;
  // Candidate slots (rows * k) and row slots the buffers hold.
  size_t slots = 0;
  size_t row_slots = 0;
  uint64_t calls = 0;
};

void FreeNvidiaTopKCandidatesBuffers(NvidiaTopKCandidatesContext* context) {
  if (context->device_ids != nullptr) cudaFree(context->device_ids);
  if (context->device_values != nullptr) cudaFree(context->device_values);
  if (context->device_flags != nullptr) cudaFree(context->device_flags);
  if (context->host_ids != nullptr) cudaFreeHost(context->host_ids);
  if (context->host_values != nullptr) cudaFreeHost(context->host_values);
  if (context->host_flags != nullptr) cudaFreeHost(context->host_flags);
  context->device_ids = nullptr;
  context->device_values = nullptr;
  context->device_flags = nullptr;
  context->host_ids = nullptr;
  context->host_values = nullptr;
  context->host_flags = nullptr;
  context->slots = 0;
  context->row_slots = 0;
}

void DestroyNvidiaTopKCandidatesResources(
    NvidiaTopKCandidatesContext* context) {
  if (context == nullptr) {
    return;
  }
  int previous_device = -1;
  const bool restore_device = context->device >= 0 &&
                              cudaGetDevice(&previous_device) == cudaSuccess &&
                              previous_device != context->device;
  if (restore_device) {
    cudaSetDevice(context->device);
  }
  if (context->stream != nullptr) {
    cudaStreamSynchronize(context->stream);
  }
  FreeNvidiaTopKCandidatesBuffers(context);
  if (context->stream != nullptr) {
    cudaStreamDestroy(context->stream);
    context->stream = nullptr;
  }
  if (restore_device) {
    cudaSetDevice(previous_device);
  }
  if (context->calls > 0) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA CUDA top-k candidate selection completed %llu calls",
               static_cast<unsigned long long>(context->calls));
  }
}

LiteRtStatus EnsureNvidiaTopKCandidatesResources(
    NvidiaTopKCandidatesContext* context, int device, size_t rows, size_t k) {
  if (context->device >= 0 && context->device != device) {
    return kLiteRtStatusErrorUnsupported;
  }
  if (context->device < 0) {
    int compute_capability_major = 0;
    const cudaError_t capability_status = cudaDeviceGetAttribute(
        &compute_capability_major, cudaDevAttrComputeCapabilityMajor, device);
    if (capability_status != cudaSuccess) {
      return CudaStatus(capability_status,
                        "cudaDeviceGetAttribute CUDA top-k candidates");
    }
    // The archive carries compute_80 PTX; older devices keep the host path.
    if (compute_capability_major < 8) {
      return kLiteRtStatusErrorUnsupported;
    }
    cudaStream_t stream = nullptr;
    LITERT_RETURN_IF_ERROR(
        CudaStatus(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags CUDA top-k candidates"));
    context->device = device;
    context->stream = stream;
    LITERT_LOG(LITERT_INFO,
               "NVIDIA CUDA top-k candidate selection initialized on device %d",
               device);
  }
  const size_t slots = rows * k;
  if (context->slots < slots || context->row_slots < rows) {
    // Previous calls synchronized before returning: the buffers are idle.
    FreeNvidiaTopKCandidatesBuffers(context);
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&context->device_ids),
                   slots * sizeof(int32_t)) != cudaSuccess ||
                cudaMalloc(reinterpret_cast<void**>(&context->device_values),
                           slots * sizeof(float)) != cudaSuccess ||
                cudaMalloc(reinterpret_cast<void**>(&context->device_flags),
                           rows * sizeof(int32_t)) != cudaSuccess ||
                cudaMallocHost(reinterpret_cast<void**>(&context->host_ids),
                               slots * sizeof(int32_t)) != cudaSuccess ||
                cudaMallocHost(reinterpret_cast<void**>(&context->host_values),
                               slots * sizeof(float)) != cudaSuccess ||
                cudaMallocHost(reinterpret_cast<void**>(&context->host_flags),
                               rows * sizeof(int32_t)) != cudaSuccess
            ? cudaGetLastError()
            : cudaSuccess;
    if (status != cudaSuccess) {
      FreeNvidiaTopKCandidatesBuffers(context);
      return CudaStatus(status, "allocate CUDA top-k candidate buffers");
    }
    context->slots = slots;
    context->row_slots = rows;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus SelectNvidiaTopKCandidates(NvidiaTopKCandidatesContext* context,
                                        LiteRtTensorBuffer logits, size_t rows,
                                        size_t count, int32_t k, int32_t* ids,
                                        float* values, int32_t* row_flags) {
  if (context == nullptr || logits == nullptr || ids == nullptr ||
      values == nullptr || row_flags == nullptr || rows == 0 || count == 0 ||
      rows > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      rows > std::numeric_limits<size_t>::max() / count / sizeof(float) ||
      k < 1 || k > LiteRtNvidiaTopKCandidatesMaxK() ||
      static_cast<size_t>(k) > count || context->runtime_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  ResolvedSamplerLogits resolved;
  LITERT_RETURN_IF_ERROR(
      ResolveSamplerLogits(context->runtime_context, logits, rows, count,
                           /*legacy_f32_only=*/false, &resolved));
  if (context->device >= 0 && context->device != resolved.device) {
    return kLiteRtStatusErrorUnsupported;
  }
  StepHandoff handoff;
  if (rows == 1 &&
      !CudaTensorBufferRegistry::Get().AnyHostNewer(resolved.handle) &&
      StepHandoffRegistry::Get().Find(resolved.device_logits, &handoff) &&
      handoff.k == k) {
    // The decode step that feeds itself selected these for the host already
    // (its own sampling of the step's logits), and the step is synchronized.
    std::copy_n(handoff.ids, k, ids);
    std::copy_n(handoff.values, k, values);
    row_flags[0] = handoff.flags[0];
    ++context->calls;
    return kLiteRtStatusOk;
  }
  int previous_device = -1;
  LITERT_RETURN_IF_ERROR(CudaStatus(cudaGetDevice(&previous_device),
                                    "cudaGetDevice top-k candidates"));
  const bool restore_device = previous_device != resolved.device;
  if (restore_device) {
    LITERT_RETURN_IF_ERROR(CudaStatus(cudaSetDevice(resolved.device),
                                      "cudaSetDevice top-k candidates"));
  }
  LiteRtStatus status = EnsureNvidiaTopKCandidatesResources(
      context, resolved.device, rows, static_cast<size_t>(k));
  if (status == kLiteRtStatusOk) {
    // What the host wrote to the logits since the device last had them (a
    // constraint's mask, applied through a host lock) reaches the device in
    // stream order before the kernel reads them, as it would before an
    // invocation read them. The upload takes the "host copy is newer" mark
    // with it, which the greedy sampler's fused path reads to decline the
    // argmax an invocation ran before the write: that argmax is forgotten
    // here, so a greedy call on the same logits afterwards reads them too.
    const bool host_newer =
        CudaTensorBufferRegistry::Get().AnyHostNewer(resolved.handle);
    status = CudaTensorBufferRegistry::Get().Upload(resolved.handle,
                                                    context->stream, nullptr);
    if (status == kLiteRtStatusOk && host_newer) {
      FusedGreedyRegistry::Get().ForgetRange(resolved.handle,
                                             resolved.allocation_size);
    }
  }
  if (status == kLiteRtStatusOk) {
    const cudaError_t launch_status =
        resolved.fp16
            ? LiteRtNvidiaLaunchF16TopKCandidateRows(
                  static_cast<const uint16_t*>(resolved.device_logits), rows,
                  count, k, context->device_ids, context->device_values,
                  context->device_flags, context->stream)
            : LiteRtNvidiaLaunchF32TopKCandidateRows(
                  static_cast<const float*>(resolved.device_logits), rows,
                  count, k, context->device_ids, context->device_values,
                  context->device_flags, context->stream);
    status = CudaStatus(launch_status, "Launch CUDA top-k candidates");
  }
  const size_t slots = rows * static_cast<size_t>(k);
  if (status == kLiteRtStatusOk) {
    status =
        CudaStatus(cudaMemcpyAsync(context->host_ids, context->device_ids,
                                   slots * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, context->stream),
                   "cudaMemcpyAsync CUDA top-k candidate ids");
  }
  if (status == kLiteRtStatusOk) {
    status =
        CudaStatus(cudaMemcpyAsync(context->host_values, context->device_values,
                                   slots * sizeof(float),
                                   cudaMemcpyDeviceToHost, context->stream),
                   "cudaMemcpyAsync CUDA top-k candidate values");
  }
  if (status == kLiteRtStatusOk) {
    status =
        CudaStatus(cudaMemcpyAsync(context->host_flags, context->device_flags,
                                   rows * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, context->stream),
                   "cudaMemcpyAsync CUDA top-k candidate flags");
  }
  if (context->stream != nullptr) {
    // Also complete any earlier work if a later launch or copy fails, so the
    // caller may release the logits or fall back to the host after an error.
    const LiteRtStatus sync_status =
        CudaStatus(cudaStreamSynchronize(context->stream),
                   "cudaStreamSynchronize CUDA top-k candidates");
    if (status == kLiteRtStatusOk) {
      status = sync_status;
    }
  }
  if (status == kLiteRtStatusOk) {
    for (size_t row = 0; row < rows; ++row) {
      if ((context->host_flags[row] &
           kLiteRtNvidiaTopKCandidatesInconsistent) != 0) {
        LITERT_LOG(LITERT_ERROR,
                   "NVIDIA CUDA top-k candidate selection failed for row %zu",
                   row);
        status = kLiteRtStatusErrorRuntimeFailure;
        break;
      }
    }
  }
  if (status == kLiteRtStatusOk) {
    std::copy_n(context->host_ids, slots, ids);
    std::copy_n(context->host_values, slots, values);
    for (size_t row = 0; row < rows; ++row) {
      row_flags[row] =
          (context->host_flags[row] & kLiteRtNvidiaTopKCandidatesNaN)
              ? LITERT_DISPATCH_NVIDIA_TOPK_CANDIDATES_NAN
              : 0;
    }
    ++context->calls;
  }
  if (restore_device) {
    const LiteRtStatus restore_status =
        CudaStatus(cudaSetDevice(previous_device), "restore CUDA device");
    if (restore_status != kLiteRtStatusOk) {
      status = restore_status;
    }
  }
  return status;
}

LiteRtStatus SampleNvidiaGreedy(NvidiaGreedySamplerContext* context,
                                LiteRtTensorBuffer logits, size_t rows,
                                size_t count, int32_t* token_ids,
                                bool legacy_f32_only) {
  if (context == nullptr || logits == nullptr || token_ids == nullptr ||
      rows == 0 || count == 0 ||
      rows > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      rows > std::numeric_limits<size_t>::max() / count / sizeof(float) ||
      context->runtime_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  ResolvedSamplerLogits resolved;
  LITERT_RETURN_IF_ERROR(ResolveSamplerLogits(context->runtime_context, logits,
                                              rows, count, legacy_f32_only,
                                              &resolved));
  HwMemoryHandle handle = resolved.handle;
  void* device_logits = resolved.device_logits;
  const bool fp16 = resolved.fp16;
  if (context->device >= 0 && context->device != resolved.device) {
    return kLiteRtStatusErrorUnsupported;
  }
  if (FusedSamplingEnabled() &&
      !CudaTensorBufferRegistry::Get().AnyHostNewer(handle) &&
      FusedGreedyRegistry::Get().TakeFresh(device_logits, context, rows, count,
                                           fp16)) {
    // The invocation that wrote these logits ran the argmax and the copy on
    // its stream and synchronized; the result is in host memory already.
    std::copy_n(context->host_result, rows, token_ids);
    ++context->calls;
    ++context->fused_calls;
    return kLiteRtStatusOk;
  }
  StepHandoff handoff;
  if (rows == 1 && !CudaTensorBufferRegistry::Get().AnyHostNewer(handle) &&
      StepHandoffRegistry::Get().Find(device_logits, &handoff) &&
      handoff.k == 1) {
    // The decode step that feeds itself argmaxed the step's logits already.
    token_ids[0] = handoff.ids[0];
    ++context->calls;
    ++context->fused_calls;
    return kLiteRtStatusOk;
  }

  int previous_device = -1;
  LITERT_RETURN_IF_ERROR(
      CudaStatus(cudaGetDevice(&previous_device), "cudaGetDevice sampler"));
  const bool restore_device = previous_device != resolved.device;
  if (restore_device) {
    LITERT_RETURN_IF_ERROR(
        CudaStatus(cudaSetDevice(resolved.device), "cudaSetDevice sampler"));
  }
  LiteRtStatus status =
      EnsureNvidiaGreedySamplerResources(context, resolved.device, count, rows);
  if (status == kLiteRtStatusOk) {
    // What the host wrote to the logits since the device last had them (a
    // constraint's mask, applied through a host lock) reaches the device in
    // stream order before the kernel reads them, as it would before an
    // invocation read them.
    status = CudaTensorBufferRegistry::Get().Upload(handle, context->stream,
                                                    nullptr);
  }
  if (status == kLiteRtStatusOk) {
    // Every row in one launch pair; only the small ID array is copied back.
    const cudaError_t launch_status =
        fp16 ? LiteRtNvidiaLaunchF16ArgMaxRows(
                   static_cast<const uint16_t*>(device_logits), rows, count,
                   context->workspace, context->workspace_bytes,
                   context->device_result, context->stream)
             : LiteRtNvidiaLaunchF32ArgMaxRows(
                   static_cast<const float*>(device_logits), rows, count,
                   context->workspace, context->workspace_bytes,
                   context->device_result, context->stream);
    status = CudaStatus(launch_status, "Launch CUDA greedy sampler argmax");
  }
  if (status == kLiteRtStatusOk) {
    status =
        CudaStatus(cudaMemcpyAsync(context->host_result, context->device_result,
                                   rows * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, context->stream),
                   "cudaMemcpyAsync CUDA greedy sampler result");
  }
  if (context->stream != nullptr) {
    // Also complete any earlier rows if a later launch/copy fails, so callers
    // may safely release the logits or fall back to CPU after an error.
    const LiteRtStatus sync_status =
        CudaStatus(cudaStreamSynchronize(context->stream),
                   "cudaStreamSynchronize CUDA greedy sampler");
    if (status == kLiteRtStatusOk) {
      status = sync_status;
    }
  }
  if (status == kLiteRtStatusOk) {
    std::copy_n(context->host_result, rows, token_ids);
    ++context->calls;
    if (FusedSamplingEnabled()) {
      // From now on the invocation that writes these logits runs the argmax.
      FusedGreedyRegistry::Get().Register(
          device_logits, FusedGreedyEntry{context, rows, count, fp16, false});
    }
  }
  if (restore_device) {
    const LiteRtStatus restore_status =
        CudaStatus(cudaSetDevice(previous_device), "restore CUDA device");
    if (restore_status != kLiteRtStatusOk) {
      status = restore_status;
    }
  }
  return status;
}

// The LiteRT runtime binds partition boundary tensors to the first supported
// buffer type, but its planner does not yet force host buffers for boundary
// tensors that CPU kernels also consume; a CUDA-first preference then leaves
// those CPU consumers with unmapped data (b/observed as XNNPack "null data
// pointer" / cast-kernel crashes on Gemma4). Preferring host IO trades extra
// boundary copies for correctness until the planner accounts for mixed
// consumers.
bool PreferHostBoundaryIo() {
  const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_PREFER_HOST_IO");
  if (value != nullptr && value[0] != '\0') {
    return std::strcmp(value, "0") != 0;
  }
  return false;
}

std::string RuntimeCacheDir() {
  const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("LITERT_NVIDIA_TENSORRT_RUNTIME_CACHE_DIR");
  }
  return value == nullptr ? std::string() : std::string(value);
}

bool IsCacheFileChar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
}

std::string SanitizeCacheComponent(std::string component) {
  if (component.empty()) {
    return "anonymous";
  }
  for (char& c : component) {
    if (!IsCacheFileChar(c)) {
      c = '_';
    }
  }
  return component;
}

uint64_t Fnv1a64(const uint8_t* data, size_t size) {
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string Hex64(uint64_t value) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx",
                static_cast<unsigned long long>(value));
  return buffer;
}

std::string JoinPath(const std::string& dir, const std::string& file) {
  if (dir.empty() || dir.back() == '/') {
    return dir + file;
  }
  return dir + "/" + file;
}

bool ReadBinaryFile(const std::string& path, std::vector<uint8_t>* bytes) {
  bytes->clear();
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  if (std::fseek(file, 0, SEEK_END) != 0) {
    std::fclose(file);
    return false;
  }
  const long size = std::ftell(file);
  if (size < 0 || std::fseek(file, 0, SEEK_SET) != 0) {
    std::fclose(file);
    return false;
  }
  bytes->resize(static_cast<size_t>(size));
  if (!bytes->empty() &&
      std::fread(bytes->data(), 1, bytes->size(), file) != bytes->size()) {
    std::fclose(file);
    bytes->clear();
    return false;
  }
  std::fclose(file);
  return true;
}

bool WriteBinaryFile(const std::string& path, const void* data, size_t size) {
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return false;
  }
  const bool ok = size == 0 ||
                  std::fwrite(data, 1, size, file) == static_cast<size_t>(size);
  const int close_status = std::fclose(file);
  return ok && close_status == 0;
}

// `half_storage`: the engine keeps this float32 tensor as FP16 (a KV cache,
// Fp16CacheStorageEnabled in the compiler), so the buffer it needs is half
// the tensor's size. LiteRT's buffer keeps the tensor's type; the dispatch
// binds and clears the bytes the engine uses.
Expected<LiteRtTensorBufferRequirements> TensorBufferRequirements(
    const LiteRtRuntimeContext* runtime_context,
    const LiteRtRankedTensorType& tensor_type, bool prefer_cuda,
    bool transposed_value_cache = false, bool half_storage = false) {
  litert::RankedTensorType ranked_type(tensor_type);
  uint32_t strides[4];
  if (transposed_value_cache && !litert::nvidia::GetTransposedValueCacheStrides(
                                    tensor_type.layout, strides)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid sequence-major NVIDIA value-cache shape");
  }
  if (ranked_type.Layout().HasStrides() &&
      !litert::nvidia::TensorBufferLayoutMatches(tensor_type.layout,
                                                 transposed_value_cache)) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Tensor strides do not match the NVIDIA engine layout");
  }
  LITERT_ASSIGN_OR_RETURN(size_t size, ranked_type.Bytes());
  if (PreferHostBoundaryIo()) {
    prefer_cuda = false;
  }
  // Only a CUDA buffer is allocated at the engine's size: LiteRT reads,
  // writes and clears a host memory buffer at the tensor's packed size.
  if (half_storage && prefer_cuda) {
    size /= 2;
  }
  const LiteRtTensorBufferType cuda_buffer_types[] = {
      kNvidiaCudaTensorBufferType, kLiteRtTensorBufferTypeHostMemory};
  const LiteRtTensorBufferType host_buffer_types[] = {
      kLiteRtTensorBufferTypeHostMemory};
  const LiteRtTensorBufferType* buffer_types =
      prefer_cuda ? cuda_buffer_types : host_buffer_types;
  const int num_buffer_types = prefer_cuda ? 2 : 1;
  if (DispatchProfilingEnabled()) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch requirements prefer_cuda=%d first_type=%d "
               "num_types=%d element_type=%d rank=%d size=%zu",
               prefer_cuda, static_cast<int>(buffer_types[0]), num_buffer_types,
               tensor_type.element_type, tensor_type.layout.rank, size);
  }
  LiteRtTensorBufferRequirements requirements = nullptr;
  LITERT_RETURN_IF_ERROR(runtime_context->create_tensor_buffer_requirements(
      num_buffer_types, buffer_types, size, transposed_value_cache ? 4 : 0,
      transposed_value_cache ? strides : nullptr, &requirements));
  return requirements;
}

std::string DescribeTensorType(const LiteRtRankedTensorType& tensor_type) {
  std::string description =
      "element_type=" + std::to_string(tensor_type.element_type) + " dims=[";
  for (int i = 0; i < tensor_type.layout.rank; ++i) {
    if (i > 0) {
      description += ",";
    }
    description += std::to_string(tensor_type.layout.dimensions[i]);
  }
  description += "]";
  return description;
}

struct LockedHostBuffer {
  LiteRtTensorBuffer buffer = nullptr;
  void* host = nullptr;
  bool locked = false;
};

class ScopedHostBufferLocks {
 public:
  explicit ScopedHostBufferLocks(const LiteRtRuntimeContext* runtime_context)
      : runtime_context_(runtime_context) {}

  ~ScopedHostBufferLocks() {
    for (auto& host : locked_) {
      if (host.locked) {
        const LiteRtStatus status =
            runtime_context_->unlock_tensor_buffer(host.buffer);
        if (status != kLiteRtStatusOk) {
          LITERT_LOG(LITERT_WARNING,
                     "Failed to unlock LiteRT tensor buffer after TensorRT "
                     "dispatch error: %d",
                     status);
        }
      }
    }
  }

  void Add(LockedHostBuffer host) { locked_.push_back(host); }

  Expected<void> UnlockAll() {
    for (auto& host : locked_) {
      if (host.locked) {
        LITERT_RETURN_IF_ERROR(
            runtime_context_->unlock_tensor_buffer(host.buffer));
        host.locked = false;
      }
    }
    return {};
  }

 private:
  const LiteRtRuntimeContext* runtime_context_;
  std::vector<LockedHostBuffer> locked_;
};

// Opt-in AOT residency is scoped to one dispatch device context. Keep this
// small coordination object alive independently of that context so invocation
// destruction never has to dereference an externally owned device context.
struct AotEngineResidency {
  std::mutex mutex;
  LiteRtDispatchInvocationContext active = nullptr;
};

bool LazyAotEnginesEnabled() {
  const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_LAZY_AOT_ENGINES");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

}  // namespace

class LiteRtDispatchDeviceContextT {
 public:
  explicit LiteRtDispatchDeviceContextT(
      const LiteRtRuntimeContext* runtime_context)
      : runtime_context_(runtime_context),
        aot_residency_(LazyAotEnginesEnabled()
                           ? std::make_shared<AotEngineResidency>()
                           : nullptr) {}

  ~LiteRtDispatchDeviceContextT() {
    for (auto& [handle, record] : records_) {
      if (record.owns_device_ptr && record.device_ptr != nullptr) {
        cudaFree(record.device_ptr);
      }
    }
    if (arena_ptr_ != nullptr) {
      cudaFree(arena_ptr_);
    }
  }

  Expected<LiteRtTensorBufferHandle> RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer) {
    if (tensor_buffer == nullptr) {
      return Error(kLiteRtStatusErrorInvalidArgument, "Null tensor buffer");
    }
    LiteRtTensorBufferType buffer_type = kLiteRtTensorBufferTypeUnknown;
    LITERT_RETURN_IF_ERROR(
        runtime_context_->get_tensor_buffer_type(tensor_buffer, &buffer_type));
    if (DispatchProfilingEnabled()) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA dispatch registering tensor buffer type=%d",
                 static_cast<int>(buffer_type));
    }
    if (buffer_type != kLiteRtTensorBufferTypeHostMemory &&
        buffer_type != kNvidiaCudaTensorBufferType) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "NVIDIA dispatch currently supports host memory and "
                   "user-custom CUDA buffers only");
    }

    LITERT_ASSIGN_OR_RETURN(
        size_t packed_size,
        TensorBufferPackedSize(runtime_context_, tensor_buffer));
    // A float32 cache the engine stores as FP16 has a buffer of half the
    // tensor's packed size (TensorBufferRequirements half_storage): the
    // engine's bytes are the buffer's.
    size_t allocation_size = 0;
    LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_size(
        tensor_buffer, &allocation_size));
    Record record;
    record.tensor_buffer = tensor_buffer;
    record.size = std::min(packed_size, allocation_size);
    LiteRtRankedTensorType tensor_type;
    LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_tensor_type(
        tensor_buffer, &tensor_type));
    // Classify once at registration, not on every token's attachment path.
    record.dense_layout =
        litert::nvidia::TensorBufferLayoutMatches(tensor_type.layout, false);
    record.transposed_value_cache_layout =
        litert::nvidia::TensorBufferLayoutMatches(tensor_type.layout, true);
    if (!record.dense_layout && !record.transposed_value_cache_layout) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported NVIDIA tensor-buffer strides");
    }
    record.host_tensor_buffer =
        buffer_type == kLiteRtTensorBufferTypeHostMemory;
    record.owns_device_ptr = false;

    if (buffer_type == kNvidiaCudaTensorBufferType) {
      HwMemoryHandle handle = nullptr;
      LiteRtStatus status =
          runtime_context_->get_tensor_buffer_custom_tensor_buffer_handle(
              tensor_buffer, &handle);
      if (status == kLiteRtStatusOk && handle != nullptr) {
        size_t buffer_size = 0;
        size_t buffer_offset = 0;
        LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_size(
            tensor_buffer, &buffer_size));
        LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_offset(
            tensor_buffer, &buffer_offset));
        LITERT_ASSIGN_OR_RETURN(
            record.device_ptr,
            litert::nvidia::ResolveCudaTensorBufferView(
                handle, buffer_size, buffer_offset, record.size));
        record.owns_device_ptr = false;
        record.direct_cuda_buffer = true;
        record.cuda_handle = handle;
      }
    }

    // Host-backed boundary tensors only need a device staging pointer while
    // this TensorRT partition is executing. Allocating those staging buffers
    // for the full tensor-buffer lifetime makes fragmented prefill graphs hold
    // many large temporary tensors at once.
    if (record.device_ptr == nullptr && !record.host_tensor_buffer) {
      void* device_ptr = nullptr;
      LITERT_RETURN_IF_ERROR(
          CudaOk(cudaMalloc(&device_ptr, std::max<size_t>(packed_size, 1)),
                 "cudaMalloc"));
      record.device_ptr = device_ptr;
      record.owns_device_ptr = true;
    }

    const LiteRtTensorBufferHandle handle = next_handle_++;
    records_[handle] = record;
    return handle;
  }

  Expected<void> UnregisterTensorBuffer(LiteRtTensorBufferHandle handle) {
    auto it = records_.find(handle);
    if (it == records_.end()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unknown tensor buffer handle");
    }
    if (it->second.owns_device_ptr && it->second.device_ptr != nullptr) {
      LITERT_RETURN_IF_ERROR(
          CudaOk(cudaFree(it->second.device_ptr), "cudaFree"));
    }
    records_.erase(it);
    return {};
  }

  struct Record {
    LiteRtTensorBuffer tensor_buffer = nullptr;
    void* device_ptr = nullptr;
    size_t size = 0;
    bool owns_device_ptr = true;
    bool direct_cuda_buffer = false;
    // The device pointer of the CUDA tensor buffer `device_ptr` is a view of.
    void* cuda_handle = nullptr;
    bool host_tensor_buffer = false;
    bool transient_device_ptr = false;
    bool dense_layout = true;
    bool transposed_value_cache_layout = false;
  };

  Expected<void> EnsureDevicePtr(Record* record) {
    if (record == nullptr) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Null tensor buffer record");
    }
    if (record->device_ptr != nullptr) {
      return {};
    }
    void* device_ptr = nullptr;
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaMalloc(&device_ptr, std::max<size_t>(record->size, 1)),
               "cudaMalloc CUDA tensor buffer"));
    record->device_ptr = device_ptr;
    record->owns_device_ptr = true;
    record->transient_device_ptr = record->host_tensor_buffer;
    return {};
  }

  // Host-backed boundary tensors up to this size keep their device staging
  // buffer across invocations. Recreating it costs a cudaMalloc plus a
  // device-synchronizing cudaFree per invocation, which for the few small
  // int32 position/flag inputs of a decode step is pure overhead. Larger
  // staging buffers (fragmented prefill boundaries) are still released.
  static constexpr size_t kRetainedTransientDeviceBytes = 1 << 20;

  void ReleaseTransientDevicePtr(Record* record) {
    if (record == nullptr || !record->transient_device_ptr ||
        record->device_ptr == nullptr ||
        record->size <= kRetainedTransientDeviceBytes) {
      return;
    }
    const cudaError_t status = cudaFree(record->device_ptr);
    if (status != cudaSuccess) {
      LITERT_LOG(LITERT_WARNING,
                 "cudaFree transient CUDA tensor buffer failed: %s",
                 cudaGetErrorString(status));
    }
    record->device_ptr = nullptr;
    record->owns_device_ptr = false;
    record->transient_device_ptr = false;
  }

  // Execution contexts of this device context run sequentially, so they share
  // one activation-memory arena sized for the largest engine instead of each
  // holding its own. Contexts re-bind when the arena is reallocated (grown).
  Expected<std::pair<void*, uint64_t>> EnsureSharedActivationArena(
      size_t bytes) {
    if (bytes > arena_size_) {
      if (arena_ptr_ != nullptr) {
        // cudaFree synchronizes the device, so no in-flight work can still
        // reference the old arena afterwards.
        LITERT_RETURN_IF_ERROR(
            CudaOk(cudaFree(arena_ptr_), "cudaFree shared arena"));
        arena_ptr_ = nullptr;
        arena_size_ = 0;
      }
      LITERT_RETURN_IF_ERROR(
          CudaOk(cudaMalloc(&arena_ptr_, bytes), "cudaMalloc shared arena"));
      arena_size_ = bytes;
      ++arena_version_;
    }
    return std::make_pair(arena_ptr_, arena_version_);
  }

  size_t shared_arena_size() const { return arena_size_; }

  Expected<Record*> GetRecord(LiteRtTensorBufferHandle handle) {
    auto it = records_.find(handle);
    if (it == records_.end()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unknown tensor buffer handle");
    }
    return &it->second;
  }

  const LiteRtRuntimeContext* runtime_context() const {
    return runtime_context_;
  }

  const std::shared_ptr<AotEngineResidency>& aot_residency() const {
    return aot_residency_;
  }

  // The step a decode context ran ahead of the host (StepFeed): the streams
  // of the other contexts of this device wait for it before they run, so a
  // prefill on another stream never races its cache rows. LiteRT-LM runs its
  // steps from one thread; this is not synchronized.
  void SetStepAhead(const void* owner, cudaEvent_t done) {
    step_ahead_owner_ = owner;
    step_ahead_done_ = done;
  }
  void ClearStepAhead(const void* owner) {
    if (step_ahead_owner_ == owner) {
      step_ahead_owner_ = nullptr;
      step_ahead_done_ = nullptr;
    }
  }
  Expected<void> WaitStepAhead(const void* caller, cudaStream_t stream) {
    if (step_ahead_owner_ == nullptr || step_ahead_owner_ == caller ||
        step_ahead_done_ == nullptr) {
      return {};
    }
    return CudaOk(cudaStreamWaitEvent(stream, step_ahead_done_, 0),
                  "cudaStreamWaitEvent step ahead");
  }

 private:
  const LiteRtRuntimeContext* runtime_context_;
  std::shared_ptr<AotEngineResidency> aot_residency_;
  const void* step_ahead_owner_ = nullptr;
  cudaEvent_t step_ahead_done_ = nullptr;
  LiteRtTensorBufferHandle next_handle_ = 1;
  std::unordered_map<LiteRtTensorBufferHandle, Record> records_;
  void* arena_ptr_ = nullptr;
  size_t arena_size_ = 0;
  uint64_t arena_version_ = 0;
};

class LiteRtDispatchInvocationContextT {
 public:
  static Expected<std::unique_ptr<LiteRtDispatchInvocationContextT>> Create(
      const LiteRtRuntimeContext* runtime_context,
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs) {
    if (runtime_context == nullptr || device_context == nullptr ||
        exec_bytecode_buffer == nullptr ||
        exec_bytecode_buffer->base_addr == nullptr) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Null dispatch invocation input");
    }
    const litert::nvidia::MemoryProfiler memory_profiler("dispatch");
    const uint8_t* bytecode_data = nullptr;
    size_t bytecode_size = 0;
    std::unique_ptr<MappedAotArtifact> aot_artifact;
    if (exec_type == kLiteRtDispatchExecutableTypeJitHandle) {
      const auto* jit_executable =
          static_cast<const litert::nvidia::TensorRtJitExecutable*>(
              exec_bytecode_buffer->base_addr);
      if (jit_executable->magic !=
              litert::nvidia::kTensorRtJitExecutableMagic ||
          jit_executable->version !=
              litert::nvidia::kTensorRtJitExecutableVersion ||
          jit_executable->reserved != 0 ||
          jit_executable->bytecode_data == nullptr ||
          jit_executable->bytecode_size == 0 ||
          jit_executable->bytecode_size > std::numeric_limits<size_t>::max()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Invalid NVIDIA TensorRT JIT executable handle");
      }
      bytecode_data =
          static_cast<const uint8_t*>(jit_executable->bytecode_data);
      bytecode_size = static_cast<size_t>(jit_executable->bytecode_size);
      memory_profiler.Log("jit_handle_resolved", function_name);
    } else if (exec_type == kLiteRtDispatchExecutableTypeMlModel) {
      const auto* base =
          static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr);
      if (exec_bytecode_buffer->offset >
          std::numeric_limits<size_t>::max() - exec_bytecode_buffer->size) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT bytecode range overflows");
      }
      bytecode_data = base + exec_bytecode_buffer->offset;
      bytecode_size = exec_bytecode_buffer->size;
    } else {
      return Error(kLiteRtStatusErrorUnsupported,
                   "NVIDIA dispatch expects TensorRT bytecode or JIT handle");
    }
    LITERT_ASSIGN_OR_RETURN(auto aot_locator,
                            litert::nvidia::TryParseTensorRtAotLocator(
                                bytecode_data, bytecode_size));
    if (device_context->aot_residency() && !aot_locator.has_value()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "LITERT_NVIDIA_DISPATCH_LAZY_AOT_ENGINES requires "
                   "persistent AOT artifacts for every engine");
    }
    if (aot_locator.has_value()) {
      memory_profiler.Log("aot_artifact_map_begin", function_name);
      LITERT_ASSIGN_OR_RETURN(aot_artifact,
                              MappedAotArtifact::Open(*aot_locator));
      bytecode_data = aot_artifact->data();
      bytecode_size = aot_artifact->size();
      LITERT_LOG(
          LITERT_INFO,
          "NVIDIA dispatch mapped TensorRT AOT artifact for %s "
          "(bytes=%zu validation=%s path=%s)",
          function_name == nullptr ? "<null>" : function_name, bytecode_size,
          litert::nvidia::AotArtifactValidationName(aot_artifact->validation()),
          aot_locator->path.c_str());
      const char* validation_phase = "aot_artifact_validated";
      if (aot_artifact->validation() ==
          litert::nvidia::AotArtifactValidation::kTrustedFileIdentity) {
        validation_phase = "aot_artifact_validation_trusted";
      } else if (aot_artifact->validation() ==
                 litert::nvidia::AotArtifactValidation::kProcessCache) {
        validation_phase = "aot_artifact_validation_reused";
      }
      memory_profiler.Log(validation_phase, function_name);
    }
    LITERT_ASSIGN_OR_RETURN(auto bytecode,
                            litert::nvidia::ParseTensorRtBytecode(
                                bytecode_data, bytecode_size, function_name));
    if (function_name != nullptr && !bytecode.function_name.empty() &&
        bytecode.function_name != function_name) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "TensorRT bytecode function name does not match call info");
    }
    if (num_inputs < 0 || num_outputs < 0 ||
        static_cast<size_t>(num_inputs) != bytecode.input_names.size() ||
        static_cast<size_t>(num_outputs) != bytecode.output_names.size()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "LiteRT dispatch IO count does not match TensorRT engine");
    }

    auto context = std::unique_ptr<LiteRtDispatchInvocationContextT>(
        new LiteRtDispatchInvocationContextT(runtime_context, device_context,
                                             std::move(bytecode)));
    if (context->aot_residency_) {
      context->aot_locator_ = std::move(aot_locator);
      std::lock_guard<std::mutex> lock(context->aot_residency_->mutex);
      if (auto* active = context->aot_residency_->active) {
        LITERT_RETURN_IF_ERROR(active->EvictAotEngine());
        context->aot_residency_->active = nullptr;
      }
      // Locations and aliases are properties of the built engine, even for
      // stripped plans before refit. Probe them without creating a context or
      // uploading refit/external-head weights, then release the probe engine.
      LITERT_RETURN_IF_ERROR(context->ProbeAotIoMetadata());
    } else {
      LITERT_RETURN_IF_ERROR(context->Initialize());
    }
    if (aot_artifact) {
      // Eager contexts have consumed the serialized data. Lazy contexts keep
      // an owning locator and reopen its validated artifact before execution.
      context->DetachSerializedBacking();
      aot_artifact.reset();
      memory_profiler.Log("aot_artifact_unmapped", function_name);
    }
    // TensorRT-RTX owns its deserialized engine and the external head has
    // copied packed weights and scales to CUDA memory. Serialized-model
    // bytecode can therefore release clean file-backed pages. JIT-handle
    // bytecode remains owned by the compiler result for handle validity.
    if (exec_type == kLiteRtDispatchExecutableTypeMlModel) {
      DropFileBackedBytecodePages(*exec_bytecode_buffer);
    }
    return context;
  }

  ~LiteRtDispatchInvocationContextT() {
    if (aot_residency_) {
      std::lock_guard<std::mutex> lock(aot_residency_->mutex);
      if (aot_residency_->active == this) {
        aot_residency_->active = nullptr;
      }
      SynchronizeForCleanup();
      ReleaseExecutionResources(/*save_runtime_cache=*/resident_has_run_);
      if (layer_profiler_) {
        layer_profiler_->Dump(bytecode_.function_name);
      }
      return;
    }
    if (bytecode_.trtllm_head.has_value() && stream_ != nullptr) {
      const cudaError_t status = cudaStreamSynchronize(stream_);
      if (status != cudaSuccess) {
        LITERT_LOG(LITERT_ERROR,
                   "cudaStreamSynchronize external W2 head teardown: %s",
                   cudaGetErrorString(status));
      }
    }
    if (layer_profiler_) {
      layer_profiler_->Dump(bytecode_.function_name);
    }
    SaveRuntimeCache();
    LogOwnGraphUse();
    DestroyStepFeed();
    DestroyOwnGraphs();
    DestroyExternalHeadResources();
    for (cudaStream_t aux_stream : aux_streams_) {
      cudaStreamDestroy(aux_stream);
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  void LogOwnGraphUse() const {
    if (own_graph_captures_ == 0 && own_graph_direct_ == 0) {
      return;
    }
    int64_t replays = 0;
    for (const auto& graph : own_graphs_) {
      replays += graph.replays;
    }
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch %s step graphs: %lld captures, %lld replays, "
               "%lld direct steps%s",
               bytecode_.function_name.c_str(),
               static_cast<long long>(own_graph_captures_),
               static_cast<long long>(replays),
               static_cast<long long>(own_graph_direct_),
               own_graph_enabled_ ? "" : " (capture gave up)");
  }

  Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int index, const LiteRtRankedTensorType& tensor_type) const {
    if (index < 0 ||
        static_cast<size_t>(index) >= bytecode_.input_names.size()) {
      return Error(kLiteRtStatusErrorIndexOOB, "Input index out of bounds");
    }
    LITERT_ASSIGN_OR_RETURN(
        const bool prefer_cuda,
        PrefersCudaTensorBuffer(bytecode_.input_names[index]));
    LITERT_ASSIGN_OR_RETURN(
        const bool half_storage,
        StoredAsHalf(bytecode_.input_names[index], tensor_type));
    return TensorBufferRequirements(
        runtime_context_, tensor_type, prefer_cuda,
        litert::nvidia::IsTransposedValueCacheTensor(
            bytecode_.input_names[index]),
        half_storage);
  }

  Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int index, const LiteRtRankedTensorType& tensor_type) const {
    if (index < 0 ||
        static_cast<size_t>(index) >= bytecode_.output_names.size()) {
      return Error(kLiteRtStatusErrorIndexOOB, "Output index out of bounds");
    }
    if (bytecode_.output_names[index].empty()) {
      return TensorBufferRequirements(runtime_context_, tensor_type,
                                      /*prefer_cuda=*/true);
    }
    LITERT_ASSIGN_OR_RETURN(
        const bool prefer_cuda,
        PrefersCudaTensorBuffer(bytecode_.output_names[index]));
    LITERT_ASSIGN_OR_RETURN(
        const bool half_storage,
        StoredAsHalf(bytecode_.output_names[index], tensor_type));
    return TensorBufferRequirements(
        runtime_context_, tensor_type, prefer_cuda,
        litert::nvidia::IsTransposedValueCacheTensor(
            bytecode_.output_names[index]),
        half_storage);
  }

  Expected<void> AttachInput(int index, LiteRtTensorBufferHandle handle) {
    if (index < 0 || static_cast<size_t>(index) >= input_handles_.size()) {
      return Error(kLiteRtStatusErrorIndexOOB, "Input index out of bounds");
    }
    if (handle == 0) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Cannot attach null input tensor buffer handle");
    }
    LITERT_ASSIGN_OR_RETURN(auto* record, device_context_->GetRecord(handle));
    const bool transposed = litert::nvidia::IsTransposedValueCacheTensor(
        bytecode_.input_names[index]);
    if (!(transposed ? record->transposed_value_cache_layout
                     : record->dense_layout)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Input tensor-buffer layout does not match NVIDIA engine: " +
                       bytecode_.input_names[index]);
    }
    input_handles_[index] = handle;
    return {};
  }

  Expected<void> AttachOutput(int index, LiteRtTensorBufferHandle handle) {
    if (index < 0 || static_cast<size_t>(index) >= output_handles_.size()) {
      return Error(kLiteRtStatusErrorIndexOOB, "Output index out of bounds");
    }
    if (handle == 0) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Cannot attach null output tensor buffer handle");
    }
    LITERT_ASSIGN_OR_RETURN(auto* record, device_context_->GetRecord(handle));
    const bool transposed = litert::nvidia::IsTransposedValueCacheTensor(
        bytecode_.output_names[index]);
    if (!(transposed ? record->transposed_value_cache_layout
                     : record->dense_layout)) {
      return Error(
          kLiteRtStatusErrorInvalidArgument,
          "Output tensor-buffer layout does not match NVIDIA engine: " +
              bytecode_.output_names[index]);
    }
    output_handles_[index] = handle;
    return {};
  }

  Expected<void> DetachInput(int index, LiteRtTensorBufferHandle handle) {
    if (index < 0 || static_cast<size_t>(index) >= input_handles_.size()) {
      return Error(kLiteRtStatusErrorIndexOOB, "Input index out of bounds");
    }
    if (input_handles_[index] == handle) {
      input_handles_[index] = 0;
      bound_input_ptrs_[index] = nullptr;
    }
    return {};
  }

  Expected<void> DetachOutput(int index, LiteRtTensorBufferHandle handle) {
    if (index < 0 || static_cast<size_t>(index) >= output_handles_.size()) {
      return Error(kLiteRtStatusErrorIndexOOB, "Output index out of bounds");
    }
    if (output_handles_[index] == handle) {
      output_handles_[index] = 0;
      bound_output_ptrs_[index] = nullptr;
    }
    return {};
  }

  Expected<void> Invoke() {
    std::unique_lock<std::mutex> residency_lock;
    if (aot_residency_) {
      // Hold through completion, not just engine selection: another thread
      // must not evict this engine or reuse its activation arena mid-enqueue.
      residency_lock = std::unique_lock<std::mutex>(aot_residency_->mutex);
      LITERT_RETURN_IF_ERROR(EnsureAotEngineResident());
    }
    const bool memory_profile =
        memory_profiler_.enabled() && invocation_count_ == 0;
    ++invocation_count_;
    if (memory_profile) {
      memory_profiler_.Log("invoke_begin", bytecode_.function_name.c_str());
    }
    const bool profile = DispatchProfilingEnabled();
    if (profile) {
      LITERT_RETURN_IF_ERROR(invocation_profiler_.Begin(stream_));
    }
    if (feed_.pending) {
      // The runtime may have registered the token table since.
      LITERT_RETURN_IF_ERROR(InitStepFeed());
    }
    if (feed_.in_flight != 0) {
      // Steps ran ahead of this call. The host's inputs for it must be the
      // ones the oldest used (its mirrors are complete once its feed's event
      // is); then it is this step. Otherwise every step ahead ran for
      // nothing: the real step below overwrites their logits and cache rows.
      LITERT_RETURN_IF_ERROR(
          CudaOk(cudaEventSynchronize(feed_.slots[feed_.head].feed_done),
                 "cudaEventSynchronize step feed"));
      LITERT_ASSIGN_OR_RETURN(const bool matched, VerifyStepFeedInputs());
      if (matched) {
        const double host_gap_ms =
            feed_.returned_valid
                ? std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - feed_.returned)
                      .count()
                : 0.0;
        const StepFeedSlot& taken = feed_.slots[feed_.head];
        const cudaEvent_t taken_feed = taken.feed_done;
        const cudaEvent_t taken_step = taken.step_done;
        DispatchProfileMetrics ahead_metrics;
        LITERT_RETURN_IF_ERROR(TakeStepAhead(ahead_metrics, profile));
        if (profile) {
          float ahead_gpu_ms = 0.0f;
          cudaEventElapsedTime(&ahead_gpu_ms, taken_feed, taken_step);
          LITERT_LOG(LITERT_INFO,
                     "NVIDIA dispatch profile function=%s step_ahead=1 "
                     "host_between_ms=%.3f cpu_sync_ms=%.3f ahead_gpu_ms=%.3f",
                     bytecode_.function_name.c_str(), host_gap_ms,
                     ahead_metrics.cpu_sync_ms, ahead_gpu_ms);
        }
        feed_.returned = std::chrono::steady_clock::now();
        feed_.returned_valid = true;
        return {};
      }
      LITERT_RETURN_IF_ERROR(DropStepsAhead());
      ++feed_.misses;
      const uint64_t steps = feed_.hits + feed_.misses;
      if ((feed_.misses >= kStepFeedMaxMisses && feed_.hits < feed_.misses) ||
          (steps >= kStepFeedMissWindow &&
           feed_.misses * 100 > steps * kStepFeedMissPercent)) {
        feed_.stopped = true;
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA dispatch %s: the step feed stops after %llu of "
                   "%llu steps whose inputs the host did not confirm",
                   bytecode_.function_name.c_str(),
                   static_cast<unsigned long long>(feed_.misses),
                   static_cast<unsigned long long>(steps));
      }
    }
    LITERT_RETURN_IF_ERROR(device_context_->WaitStepAhead(this, stream_));

    ScopedHostBufferLocks locked(runtime_context_);
    struct ScopedTransientDevicePtrs {
      explicit ScopedTransientDevicePtrs(
          LiteRtDispatchDeviceContext device_context)
          : device_context(device_context) {}
      ~ScopedTransientDevicePtrs() { ReleaseAll(); }
      void ReleaseAll() {
        for (auto* record : records) {
          device_context->ReleaseTransientDevicePtr(record);
        }
        records.clear();
      }
      void Add(LiteRtDispatchDeviceContextT::Record* record) {
        if (record == nullptr || !record->transient_device_ptr ||
            std::find(records.begin(), records.end(), record) !=
                records.end()) {
          return;
        }
        records.push_back(record);
      }

      LiteRtDispatchDeviceContext device_context;
      std::vector<LiteRtDispatchDeviceContextT::Record*> records;
    } transient_device_ptrs(device_context_);
    DispatchCpuTimer input_setup_timer(profile);
    DispatchProfileMetrics profile_metrics;

    for (int i = 0; i < static_cast<int>(input_handles_.size()); ++i) {
      LITERT_ASSIGN_OR_RETURN(auto* record,
                              device_context_->GetRecord(input_handles_[i]));
      LITERT_RETURN_IF_ERROR(device_context_->EnsureDevicePtr(record));
      transient_device_ptrs.Add(record);
      if (!record->direct_cuda_buffer) {
        ++profile_metrics.host_inputs;
        profile_metrics.h2d_bytes += record->size;
        LITERT_ASSIGN_OR_RETURN(auto host,
                                GetReadableHostPointer(record->tensor_buffer));
        locked.Add(host);
        LITERT_RETURN_IF_ERROR(
            CudaOk(cudaMemcpyAsync(record->device_ptr, host.host, record->size,
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync H2D"));
      } else {
        ++profile_metrics.direct_inputs;
        size_t uploaded = 0;
        LITERT_RETURN_IF_ERROR(CudaTensorBufferRegistry::Get().Upload(
            record->cuda_handle, stream_, &uploaded));
        profile_metrics.h2d_bytes += uploaded;
      }
      LITERT_ASSIGN_OR_RETURN(
          const bool did_bind,
          BindTensorAddressIfNeeded(bytecode_.input_names[i],
                                    record->device_ptr, &bound_input_ptrs_[i]));
      did_bind ? ++profile_metrics.set_address_calls
               : ++profile_metrics.set_address_skips;
    }
    profile_metrics.cpu_input_setup_ms = input_setup_timer.ElapsedMs();
    if (profile) {
      LITERT_RETURN_IF_ERROR(invocation_profiler_.RecordInputsReady(stream_));
    }
    if (memory_profile) {
      memory_profiler_.Log("invoke_inputs_ready",
                           bytecode_.function_name.c_str());
    }

    DispatchCpuTimer output_setup_timer(profile);
    for (int i = 0; i < static_cast<int>(output_handles_.size()); ++i) {
      LITERT_ASSIGN_OR_RETURN(auto* record,
                              device_context_->GetRecord(output_handles_[i]));
      LITERT_RETURN_IF_ERROR(device_context_->EnsureDevicePtr(record));
      transient_device_ptrs.Add(record);
      if (record->direct_cuda_buffer) {
        // The invocation overwrites the buffer, host writes included.
        CudaTensorBufferRegistry::Get().Discard(record->cuda_handle);
      }
      if (bytecode_.output_names[i].empty()) {
        // The external W2 vocabulary head writes this LiteRT output after the
        // TensorRT-RTX prefix, so it has no TensorRT engine binding.
        bound_output_ptrs_[i] = record->device_ptr;
        ++profile_metrics.set_address_skips;
        continue;
      }
      if (aliased_input_of_output_[i] >= 0) {
        // In-place cache update: the engine writes this output into the
        // aliased input's memory. LiteRT-LM passes the same buffer for both;
        // otherwise seed the output with the input and run in place there.
        const int input_index = aliased_input_of_output_[i];
        LITERT_ASSIGN_OR_RETURN(
            auto* input_record,
            device_context_->GetRecord(input_handles_[input_index]));
        if (input_record->device_ptr != record->device_ptr) {
          LITERT_RETURN_IF_ERROR(CudaOk(
              cudaMemcpyAsync(record->device_ptr, input_record->device_ptr,
                              std::min(record->size, input_record->size),
                              cudaMemcpyDeviceToDevice, stream_),
              "cudaMemcpyAsync aliased cache input"));
          LITERT_ASSIGN_OR_RETURN(
              const bool rebound,
              BindTensorAddressIfNeeded(bytecode_.input_names[input_index],
                                        record->device_ptr,
                                        &bound_input_ptrs_[input_index]));
          rebound ? ++profile_metrics.set_address_calls
                  : ++profile_metrics.set_address_skips;
        }
      }
      LITERT_ASSIGN_OR_RETURN(const bool did_bind,
                              BindTensorAddressIfNeeded(
                                  bytecode_.output_names[i], record->device_ptr,
                                  &bound_output_ptrs_[i]));
      did_bind ? ++profile_metrics.set_address_calls
               : ++profile_metrics.set_address_skips;
    }
    profile_metrics.cpu_output_setup_ms = output_setup_timer.ElapsedMs();

    LITERT_RETURN_IF_ERROR(BindSharedActivationArena());
    if (memory_profile) {
      memory_profiler_.Log("invoke_outputs_and_arena_ready",
                           bytecode_.function_name.c_str());
    }
    if (DumpOutputsSpec() != nullptr) {
      ++dump_outputs_step_;
    }
    if (DispatchDumpIoEnabled()) {
      ++dump_step_;
      for (int i = 0; i < static_cast<int>(input_handles_.size()); ++i) {
        LITERT_ASSIGN_OR_RETURN(auto* record,
                                device_context_->GetRecord(input_handles_[i]));
        if (DispatchDumpTopCount() == 0) {
          DumpDeviceBufferPrefix("in", bytecode_.input_names[i],
                                 record->device_ptr, record->size);
        }
        DumpDeviceBufferToFile("input", i, record->device_ptr, record->size);
      }
    }
    DispatchCpuTimer enqueue_call_timer(profile);
    if (!aux_streams_.empty()) {
      execution_context_->setAuxStreams(
          aux_streams_.data(), static_cast<int32_t>(aux_streams_.size()));
    }
    LITERT_RETURN_IF_ERROR(EnqueueStep());
    profile_metrics.cpu_enqueue_call_ms = enqueue_call_timer.ElapsedMs();
    if (profile) {
      LITERT_RETURN_IF_ERROR(invocation_profiler_.RecordEnqueued(stream_));
    }
    if (memory_profile) {
      memory_profiler_.Log("invoke_enqueued", bytecode_.function_name.c_str());
    }

    DispatchCpuTimer output_copy_setup_timer(profile);
    for (int i = 0; i < static_cast<int>(output_handles_.size()); ++i) {
      LITERT_ASSIGN_OR_RETURN(auto* record,
                              device_context_->GetRecord(output_handles_[i]));
      if (!record->direct_cuda_buffer) {
        ++profile_metrics.host_outputs;
        profile_metrics.d2h_bytes += record->size;
        LITERT_ASSIGN_OR_RETURN(auto host,
                                GetWritableHostPointer(record->tensor_buffer));
        locked.Add(host);
        LITERT_RETURN_IF_ERROR(
            CudaOk(cudaMemcpyAsync(host.host, record->device_ptr, record->size,
                                   cudaMemcpyDeviceToHost, stream_),
                   "cudaMemcpyAsync D2H"));
      } else {
        ++profile_metrics.direct_outputs;
      }
    }
    profile_metrics.cpu_output_copy_setup_ms =
        output_copy_setup_timer.ElapsedMs();
    if (profile) {
      LITERT_RETURN_IF_ERROR(invocation_profiler_.RecordOutputsReady(stream_));
    }

    // The greedy argmax of the logits a sampler reads, on this stream before
    // the synchronization (FusedGreedyRegistry).
    std::vector<const void*> fused_logits;
    if (FusedSamplingEnabled()) {
      for (int i = 0; i < static_cast<int>(output_handles_.size()); ++i) {
        LITERT_ASSIGN_OR_RETURN(auto* record,
                                device_context_->GetRecord(output_handles_[i]));
        LITERT_RETURN_IF_ERROR(
            EnqueueFusedGreedy(record->device_ptr, stream_, &fused_logits));
      }
    }

    LITERT_RETURN_IF_ERROR(EnqueueFeedToken(0, nullptr));
    LITERT_RETURN_IF_ERROR(PrefetchReadOutputs(nullptr));
    DispatchCpuTimer sync_timer(profile);
    const char* sync_context =
        bytecode_.trtllm_head.has_value()
            ? "cudaStreamSynchronize TensorRT-RTX prefix and external W2 head"
            : "cudaStreamSynchronize";
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaStreamSynchronize(stream_), sync_context));
    for (const void* logits : fused_logits) {
      FusedGreedyRegistry::Get().SetFresh(logits, true);
    }
    MarkPrefetched();
    if (feed_.argmax_ready) {
      PublishHandoff(feed_.host_handoff);
    } else if (feed_.enabled || feed_.stopped) {
      ClearHandoff();
    }
    if (aot_residency_) {
      resident_has_run_ = true;
    }
    profile_metrics.cpu_sync_ms = sync_timer.ElapsedMs();
    if (memory_profile) {
      memory_profiler_.Log("invoke_synchronized",
                           bytecode_.function_name.c_str());
    }
    if (DispatchDumpIoEnabled()) {
      for (int i = 0; i < static_cast<int>(output_handles_.size()); ++i) {
        LITERT_ASSIGN_OR_RETURN(auto* record,
                                device_context_->GetRecord(output_handles_[i]));
        if (DispatchDumpTopCount() == 0) {
          DumpDeviceBufferPrefix("out", bytecode_.output_names[i],
                                 record->device_ptr, record->size);
        }
        DumpDeviceBufferTop(bytecode_.output_names[i], record->tensor_buffer,
                            record->device_ptr, record->size);
        DumpDeviceBufferToFile("output", i, record->device_ptr, record->size);
      }
    }
    if (DumpOutputsSpec() != nullptr) {
      // The KV caches are state: LiteRT-LM attaches one buffer as the cache
      // input and the cache output of a step (in place, whether or not the
      // engine declares the alias). An output that shares its device memory
      // with an input is not what the step computed for its caller. A
      // prefill's whole output is its caches, one chunk per step, so a
      // context that emits nothing but state has its state dumped at every
      // step (a change in a later chunk is then named at that chunk); a
      // context with other outputs (a decode step's logits) has its state
      // dumped at its first step only, where the caches hold the prefill's
      // work.
      std::vector<const void*> input_ptrs;
      input_ptrs.reserve(input_handles_.size());
      for (const auto handle : input_handles_) {
        if (handle == 0) continue;
        auto input_record = device_context_->GetRecord(handle);
        if (input_record.HasValue() && (*input_record)->device_ptr != nullptr) {
          input_ptrs.push_back((*input_record)->device_ptr);
        }
      }
      std::vector<bool> state_of_output(output_handles_.size(), false);
      bool any_non_state = false;
      for (int i = 0; i < static_cast<int>(output_handles_.size()); ++i) {
        LITERT_ASSIGN_OR_RETURN(auto* record,
                                device_context_->GetRecord(output_handles_[i]));
        state_of_output[i] =
            (i < static_cast<int>(aliased_input_of_output_.size()) &&
             aliased_input_of_output_[i] >= 0) ||
            std::find(input_ptrs.begin(), input_ptrs.end(),
                      record->device_ptr) != input_ptrs.end();
        any_non_state |= !state_of_output[i];
      }
      const std::string input_dims = DumpInputDims();
      for (int i = 0; i < static_cast<int>(output_handles_.size()); ++i) {
        if (state_of_output[i] && any_non_state && dump_outputs_step_ != 1) {
          continue;
        }
        LITERT_ASSIGN_OR_RETURN(auto* record,
                                device_context_->GetRecord(output_handles_[i]));
        DumpOutputToDir(i, bytecode_.output_names[i], record->tensor_buffer,
                        record->device_ptr, record->size, state_of_output[i],
                        input_dims);
      }
    }
    LITERT_RETURN_IF_ERROR(SpeculateNextStep());
    feed_.returned = std::chrono::steady_clock::now();
    feed_.returned_valid = true;
    DispatchCpuTimer unlock_timer(profile);
    LITERT_RETURN_IF_ERROR(locked.UnlockAll());
    profile_metrics.cpu_unlock_ms = unlock_timer.ElapsedMs();
    if (profile) {
      LITERT_RETURN_IF_ERROR(invocation_profiler_.Finish(
          bytecode_.function_name.c_str(), profile_metrics));
    }
    if (memory_profile) {
      transient_device_ptrs.ReleaseAll();
      memory_profiler_.Log("invoke_transients_released",
                           bytecode_.function_name.c_str());
    }
    return {};
  }

  Expected<void> SetOptions(LiteRtOptions options) { return {}; }

  Expected<void> SetSchedulingInfo(
      const LiteRtSchedulingInfo* scheduling_info) {
    if (scheduling_info != nullptr) {
      scheduling_info_ = *scheduling_info;
      has_scheduling_info_ = true;
    } else {
      has_scheduling_info_ = false;
    }
    return {};
  }

 private:
  LiteRtDispatchInvocationContextT(const LiteRtRuntimeContext* runtime_context,
                                   LiteRtDispatchDeviceContext device_context,
                                   litert::nvidia::TensorRtBytecode bytecode)
      : runtime_context_(runtime_context),
        device_context_(device_context),
        aot_residency_(device_context->aot_residency()),
        bytecode_(std::move(bytecode)),
        memory_profiler_("dispatch"),
        input_handles_(bytecode_.input_names.size(), 0),
        output_handles_(bytecode_.output_names.size(), 0),
        bound_input_ptrs_(bytecode_.input_names.size(), nullptr),
        bound_output_ptrs_(bytecode_.output_names.size(), nullptr) {}

  void DetachSerializedBacking() {
    bytecode_.engine_data = nullptr;
    bytecode_.engine_size = 0;
    bytecode_.refit_weights.clear();
    bytecode_.weight_store.reset();
    if (bytecode_.trtllm_head.has_value()) {
      bytecode_.trtllm_head->packed_weights = nullptr;
      bytecode_.trtllm_head->packed_weights_size = 0;
      bytecode_.trtllm_head->bf16_scales = nullptr;
      bytecode_.trtllm_head->bf16_scales_size = 0;
    }
  }

  Expected<void> ProbeAotIoMetadata() {
    DispatchCpuTimer probe_timer(/*enabled=*/true);
    memory_profiler_.Log("engine_metadata_probe_begin",
                         bytecode_.function_name.c_str());
    LITERT_RETURN_IF_ERROR(DeserializeEngine());
    LITERT_RETURN_IF_ERROR(RecordAliasedOutputs());
    for (const auto& name : bytecode_.input_names) {
      tensor_locations_.emplace(name, engine_->getTensorLocation(name.c_str()));
      tensor_types_.emplace(name, engine_->getTensorDataType(name.c_str()));
    }
    for (const auto& name : bytecode_.output_names) {
      if (!name.empty()) {
        tensor_locations_.emplace(name,
                                  engine_->getTensorLocation(name.c_str()));
        tensor_types_.emplace(name, engine_->getTensorDataType(name.c_str()));
      }
    }
    engine_.reset();
    runtime_.reset();
    memory_profiler_.Log("engine_metadata_probe_end",
                         bytecode_.function_name.c_str());
    LITERT_LOG(LITERT_INFO,
               "NVIDIA AOT engine metadata probed: %s elapsed_ms=%.3f",
               bytecode_.function_name.c_str(), probe_timer.ElapsedMs());
    return {};
  }

  // Error/destructor cleanup cannot return a synchronization failure, but must
  // still dispose of partially created state in dependency order.
  void SynchronizeForCleanup() {
    if (stream_ != nullptr) {
      const cudaError_t status = cudaStreamSynchronize(stream_);
      if (status != cudaSuccess) {
        LITERT_LOG(
            LITERT_ERROR,
            "NVIDIA AOT engine cleanup synchronization failed for %s: %s",
            bytecode_.function_name.c_str(), cudaGetErrorString(status));
      }
    }
  }

  // The caller has synchronized the stream. Buffer handles and the shared
  // activation arena belong to the device context, not to a resident engine;
  // keep them while clearing every address cached by the old execution context.
  void ReleaseExecutionResources(bool save_runtime_cache) {
    if (save_runtime_cache) {
      SaveRuntimeCache();
    }
    LogOwnGraphUse();
    DestroyStepFeed();
    DestroyOwnGraphs();
    execution_context_.reset();
    runtime_config_.reset();
    runtime_cache_.reset();
    DestroyExternalHeadResources();
    for (cudaStream_t aux_stream : aux_streams_) {
      cudaStreamDestroy(aux_stream);
    }
    aux_streams_.clear();
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    weight_mapping_.reset();
    engine_.reset();
    runtime_.reset();
    weight_memory_.reset();
    std::fill(bound_input_ptrs_.begin(), bound_input_ptrs_.end(), nullptr);
    std::fill(bound_output_ptrs_.begin(), bound_output_ptrs_.end(), nullptr);
    bound_arena_ptr_ = nullptr;
    bound_arena_version_ = 0;
    device_memory_bytes_ = -1;
    resident_has_run_ = false;
  }

  Expected<void> EvictAotEngine() {
    DispatchCpuTimer eviction_timer(/*enabled=*/true);
    memory_profiler_.Log("engine_eviction_begin",
                         bytecode_.function_name.c_str());
    if (stream_ != nullptr) {
      LITERT_RETURN_IF_ERROR(CudaOk(cudaStreamSynchronize(stream_),
                                    "cudaStreamSynchronize AOT eviction"));
    }
    ReleaseExecutionResources(/*save_runtime_cache=*/resident_has_run_);
    memory_profiler_.Log("engine_evicted", bytecode_.function_name.c_str());
    LITERT_LOG(LITERT_INFO, "NVIDIA AOT engine evicted: %s elapsed_ms=%.3f",
               bytecode_.function_name.c_str(), eviction_timer.ElapsedMs());
    return {};
  }

  Expected<void> EnsureAotEngineResident() {
    if (aot_residency_->active == this) {
      return {};
    }
    if (auto* active = aot_residency_->active) {
      LITERT_RETURN_IF_ERROR(active->EvictAotEngine());
      aot_residency_->active = nullptr;
    }
    DispatchCpuTimer reload_timer(/*enabled=*/true);
    memory_profiler_.Log("engine_reload_begin",
                         bytecode_.function_name.c_str());
    LITERT_ASSIGN_OR_RETURN(auto artifact,
                            MappedAotArtifact::Open(*aot_locator_));
    LITERT_ASSIGN_OR_RETURN(auto bytecode,
                            litert::nvidia::ParseTensorRtBytecode(
                                artifact->data(), artifact->size(),
                                bytecode_.function_name.c_str()));
    if (bytecode.version != bytecode_.version ||
        bytecode.function_name != bytecode_.function_name ||
        bytecode.input_names != bytecode_.input_names ||
        bytecode.output_names != bytecode_.output_names) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Reloaded NVIDIA AOT engine has different IO metadata");
    }
    bytecode_ = std::move(bytecode);
    auto initialized = Initialize();
    if (!initialized) {
      SynchronizeForCleanup();
      ReleaseExecutionResources(/*save_runtime_cache=*/false);
      DetachSerializedBacking();
      return initialized.Error();
    }
    DetachSerializedBacking();
    artifact.reset();
    aot_residency_->active = this;
    memory_profiler_.Log("engine_reload_end", bytecode_.function_name.c_str());
    LITERT_LOG(LITERT_INFO, "NVIDIA AOT engine reloaded: %s elapsed_ms=%.3f",
               bytecode_.function_name.c_str(), reload_timer.ElapsedMs());
    return {};
  }

  Expected<void> DeserializeEngine() {
    litert::nvidia::EnsureSubbyteGemvPluginRegistered();
    litert::nvidia::EnsureDecodeAttentionPluginRegistered();
    litert::nvidia::EnsureCacheUpdatePluginRegistered();
    litert::nvidia::EnsureInt8CacheWritePluginRegistered();
    litert::nvidia::EnsureMoePluginRegistered();
    litert::nvidia::EnsureQkvNormRopePluginRegistered();
    // Execution contexts of this process load their expert arenas.
    litert::nvidia::SetMoeArenaLoadingEnabled(true);
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch creating TensorRT runtime for function %s "
               "(inputs=%zu outputs=%zu engine_bytes=%zu)",
               bytecode_.function_name.c_str(), bytecode_.input_names.size(),
               bytecode_.output_names.size(), bytecode_.engine_size);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to create TensorRT runtime");
    }
    memory_profiler_.Log("runtime_created", bytecode_.function_name.c_str());
    engine_.reset(runtime_->deserializeCudaEngine(bytecode_.engine_data,
                                                  bytecode_.engine_size));
    if (!engine_) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to deserialize TensorRT engine");
    }
    memory_profiler_.Log("engine_deserialized",
                         bytecode_.function_name.c_str());
    return {};
  }

  Expected<void> Initialize() {
    memory_profiler_.Log("context_initialize_begin",
                         bytecode_.function_name.c_str());
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags"));
    memory_profiler_.Log("stream_created", bytecode_.function_name.c_str());
    LITERT_RETURN_IF_ERROR(DeserializeEngine());
    LITERT_RETURN_IF_ERROR(RecordAliasedOutputs());
    LITERT_RETURN_IF_ERROR(MapEngineWeights());
    LITERT_RETURN_IF_ERROR(RefitEngine());
    LITERT_RETURN_IF_ERROR(InitStepFeed());
    DumpEngineInfo();
    const bool own_graph = OwnCudaGraph(&own_graph_mode_);
    if (UseCudaGraph() || own_graph) {
      // TensorRT-RTX owns whole-graph CUDA-graph capture/replay through the
      // runtime config; this collapses per-kernel launch overhead, which
      // dominates decode-step time on WSL. With OwnCudaGraph the context
      // captures the step itself instead (EnqueueStep).
      runtime_config_.reset(engine_->createRuntimeConfig());
      if (runtime_config_) {
        if (UseSharedActivationArena()) {
          runtime_config_->setExecutionContextAllocationStrategy(
              nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
        }
        runtime_config_->setCudaGraphStrategy(
            own_graph ? nvinfer1::CudaGraphStrategy::kDISABLED
                      : nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
        LITERT_RETURN_IF_ERROR(ConfigureRuntimeCache());
        execution_context_.reset(
            engine_->createExecutionContext(runtime_config_.get()));
        if (execution_context_) {
          if (UseSharedActivationArena()) {
            device_memory_bytes_ = engine_->getDeviceMemorySizeV2();
          }
          own_graph_enabled_ = own_graph;
          LITERT_LOG(LITERT_INFO,
                     "NVIDIA dispatch context %s uses %s "
                     "(device_memory_bytes=%lld)",
                     bytecode_.function_name.c_str(),
                     own_graph ? "its own step graph" : "CUDA graph capture",
                     static_cast<long long>(device_memory_bytes_));
        }
      }
    }
    if (!execution_context_ && UseSharedActivationArena()) {
      execution_context_.reset(engine_->createExecutionContext(
          nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
      if (execution_context_) {
        device_memory_bytes_ = engine_->getDeviceMemorySizeV2();
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA dispatch context %s shares activation arena "
                   "(device_memory_bytes=%lld)",
                   bytecode_.function_name.c_str(),
                   static_cast<long long>(device_memory_bytes_));
      }
    }
    if (!execution_context_) {
      execution_context_.reset(engine_->createExecutionContext());
    }
    if (!execution_context_) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to create TensorRT execution context");
    }
    memory_profiler_.Log("execution_context_created",
                         bytecode_.function_name.c_str());
    LITERT_RETURN_IF_ERROR(CreateAuxStreams());
    if (litert::nvidia::DispatchLayerProfilingEnabled()) {
      if (!layer_profiler_) {
        layer_profiler_ =
            std::make_unique<litert::nvidia::TensorRtLayerProfiler>();
      }
      execution_context_->setProfiler(layer_profiler_.get());
      LITERT_LOG(LITERT_INFO, "NVIDIA layer profiling enabled for %s",
                 bytecode_.function_name.c_str());
    }
    LITERT_RETURN_IF_ERROR(InitializeExternalHead());
    memory_profiler_.Log("context_initialize_end",
                         bytecode_.function_name.c_str());
    return {};
  }

  // Outputs of in-place KV cache updates alias a network input; the engine
  // requires both to be bound to the same device address.
  Expected<void> RecordAliasedOutputs() {
    aliased_input_of_output_.assign(bytecode_.output_names.size(), -1);
    size_t aliased = 0;
    for (size_t i = 0; i < bytecode_.output_names.size(); ++i) {
      if (bytecode_.output_names[i].empty()) {
        continue;
      }
      const char* input_name =
          engine_->getAliasedInputTensor(bytecode_.output_names[i].c_str());
      if (input_name == nullptr) {
        continue;
      }
      const auto it = std::find(bytecode_.input_names.begin(),
                                bytecode_.input_names.end(), input_name);
      if (it == bytecode_.input_names.end()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT output aliases an unknown input tensor");
      }
      aliased_input_of_output_[i] =
          static_cast<int>(it - bytecode_.input_names.begin());
      ++aliased;
    }
    if (aliased > 0) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA dispatch %s binds %zu in-place cache outputs to "
                 "their input buffers",
                 bytecode_.function_name.c_str(), aliased);
    }
    return {};
  }

  // The step feed (bytecode.h, TensorRtStepFeed): the decode steps run on
  // the device ahead of the host, kStepFeedDepth at most.
  static constexpr int kStepFeedDepth = 2;
  struct StepFeedSlot {
    // Host-mapped mirrors of the inputs the feed wrote, in one allocation:
    // the token, the embeddings, the positions, the cache parameter, the
    // local mask; the global mask's mirror is the host's copy plus the keys
    // of the steps ahead.
    void* mirror = nullptr;
    int32_t* token_mirror = nullptr;
    float* embeddings_mirror = nullptr;
    int32_t* positions_mirror = nullptr;
    int32_t* param_mirror = nullptr;
    uint8_t* local_mask_mirror = nullptr;
    uint8_t* global_mask_mirror = nullptr;
    cudaEvent_t feed_done = nullptr;
    cudaEvent_t step_done = nullptr;
    int32_t position = 0;  // the step's position
    int handoff = 0;       // the set its token selection went to
    // The outputs the host reads, staged in pinned memory per output port.
    std::vector<void*> staging;
    std::vector<size_t> staging_bytes;
    struct Prefetched {
      size_t port;
      void* handle;
      void* host;
      size_t bytes;
    };
    std::vector<Prefetched> prefetched;
  };
  struct StepFeed {
    bool enabled = false;  // the bytecode has a feed and the setup succeeded
    bool stopped = false;  // too many misses
    // The bytecode's feed has no table: the runtime's registration is awaited
    // (token_table_c_api.h) and the setup retried at each invocation.
    bool pending = false;
    bool owns_table = false;      // the table and the scales are this context's
    void* table = nullptr;        // device INT4 [vocab, dim]
    float* row_scales = nullptr;  // device [vocab], or null
    const uint16_t* block_scales = nullptr;  // device FP16 per block, or null
    uint32_t scale_block = 0;
    float multiplier = 1.0f;
    unsigned int* blocks_done = nullptr;  // the kernel's block counter
    int32_t param_count = 0;
    size_t global_bytes = 0;
    StepFeedSlot slots[kStepFeedDepth];
    int head = 0;                 // the oldest step ahead's slot
    int in_flight = 0;            // the steps ahead
    int32_t queued_position = 0;  // the newest step ahead's position
    // The feed's own token from the logits: the argmax, or the session's
    // sampling on the device (EnqueueFeedToken); the host samples on its
    // own and the feed's token is verified through the inputs it fed.
    NvidiaGreedySamplerContext sampler;
    int32_t* candidate_ids = nullptr;    // device [max k]
    float* candidate_values = nullptr;   // device [max k]
    int32_t* candidate_flags = nullptr;  // device [1]
    // The token selections handed to the host (StepHandoff), pinned sets
    // taken in turn: the one the host reads, the steps ahead's, the one
    // being enqueued.
    struct Handoff {
      int32_t* ids = nullptr;
      float* values = nullptr;
      int32_t* flags = nullptr;
      int32_t k = 0;
    };
    Handoff handoff[kStepFeedDepth + 2];
    int next_handoff = 0;
    int host_handoff = 0;       // the set of the step the host invoked
    uint64_t sampled = 0;       // tokens sampled on the device, not argmaxed
    bool argmax_ready = false;  // the token of the last step is enqueued
    uint64_t speculated = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    // The local mask's sliding window when the bytecode does not say: the
    // keys the host's mask holds once it holds fewer than the position
    // allows (0: not seen yet, the next mask keeps every key).
    uint32_t learned_window = 0;
    // When the last invocation returned (the host's time between the steps).
    std::chrono::steady_clock::time_point returned;
    bool returned_valid = false;
  };
  StepFeed feed_;

  // The step feed's token table, slots and sampler (bytecode.h,
  // TensorRtStepFeed). Nothing to set up when the bytecode has no feed, the
  // feed is off (StepFeedEnabled), the engines are swapped in and out
  // (aot_residency_), or the step's inputs and outputs are dumped. A setup
  // that fails leaves the feed off and the step as it is.
  Expected<void> InitStepFeed() {
    if (!bytecode_.step_feed.has_value() || !StepFeedEnabled() ||
        aot_residency_ || DispatchDumpIoEnabled() ||
        DumpOutputsSpec() != nullptr) {
      return {};
    }
    const auto& feed = *bytecode_.step_feed;
    const size_t table_bytes = static_cast<size_t>(feed.vocab) * feed.dim / 2;
    const auto fail = [&](const char* what, cudaError_t status) {
      LITERT_LOG(LITERT_WARNING, "NVIDIA dispatch %s: the step feed's %s: %s",
                 bytecode_.function_name.c_str(), what,
                 cudaGetErrorString(status));
      DestroyStepFeed();
      return Expected<void>();
    };
    std::string table_source;
    cudaError_t status = cudaSuccess;
    if (feed.table_offset == litert::nvidia::kTensorRtStepFeedNoTable) {
      // The embedder's table, registered by the runtime.
      auto registered = FindTokenTable(feed.vocab, feed.dim);
      if (!registered.has_value()) {
        if (!feed_.pending) {
          LITERT_LOG(LITERT_INFO,
                     "NVIDIA dispatch %s: the step feed awaits the runtime's "
                     "token table [%u, %u]",
                     bytecode_.function_name.c_str(), feed.vocab, feed.dim);
        }
        feed_.pending = true;
        return {};
      }
      feed_.pending = false;
      feed_.owns_table = false;
      feed_.table = registered->table;
      if (registered->scale_block == 0) {
        feed_.row_scales = static_cast<float*>(registered->scales);
      } else {
        feed_.block_scales = static_cast<const uint16_t*>(registered->scales);
        feed_.scale_block = registered->scale_block;
      }
      feed_.multiplier = registered->multiplier;
      table_source = "the runtime";
    } else {
      // The table in the model file, by the weight store's path (the store
      // is there at the setup: it goes once the engine's weights are mapped).
      if (!bytecode_.weight_store.has_value()) return {};
      const std::string& path = bytecode_.weight_store->source_path;
      const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
      if (fd < 0) {
        LITERT_LOG(LITERT_WARNING,
                   "NVIDIA dispatch %s: the step feed cannot open %s: %s",
                   bytecode_.function_name.c_str(), path.c_str(),
                   std::strerror(errno));
        return {};
      }
      std::vector<uint8_t> table(table_bytes);
      bool read_ok = true;
      for (size_t done = 0; done < table_bytes;) {
        const ssize_t result =
            pread(fd, table.data() + done, table_bytes - done,
                  static_cast<off_t>(feed.table_offset + done));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) {
          read_ok = false;
          break;
        }
        done += static_cast<size_t>(result);
      }
      close(fd);
      if (!read_ok) {
        LITERT_LOG(LITERT_WARNING,
                   "NVIDIA dispatch %s: the step feed cannot read the token "
                   "table (%zu bytes at %llu of %s)",
                   bytecode_.function_name.c_str(), table_bytes,
                   static_cast<unsigned long long>(feed.table_offset),
                   path.c_str());
        return {};
      }
      feed_.owns_table = true;
      feed_.multiplier = feed.multiplier;
      status = cudaMalloc(&feed_.table, table_bytes);
      if (status != cudaSuccess) return fail("table", status);
      status = cudaMemcpy(feed_.table, table.data(), table_bytes,
                          cudaMemcpyHostToDevice);
      if (status != cudaSuccess) return fail("table copy", status);
      status = cudaMalloc(reinterpret_cast<void**>(&feed_.row_scales),
                          feed.row_scales_size);
      if (status != cudaSuccess) return fail("row scales", status);
      status = cudaMemcpy(feed_.row_scales, feed.row_scales,
                          feed.row_scales_size, cudaMemcpyHostToDevice);
      if (status != cudaSuccess) return fail("row scales copy", status);
      table_source = path;
    }
    status = cudaMalloc(reinterpret_cast<void**>(&feed_.blocks_done),
                        sizeof(unsigned int));
    if (status != cudaSuccess) return fail("block counter", status);
    status = cudaMemset(feed_.blocks_done, 0, sizeof(unsigned int));
    if (status != cudaSuccess) return fail("block counter", status);
    const size_t max_k = static_cast<size_t>(LiteRtNvidiaTopKCandidatesMaxK());
    status = cudaMalloc(reinterpret_cast<void**>(&feed_.candidate_ids),
                        max_k * sizeof(int32_t));
    if (status != cudaSuccess) return fail("candidates", status);
    status = cudaMalloc(reinterpret_cast<void**>(&feed_.candidate_values),
                        max_k * sizeof(float));
    if (status != cudaSuccess) return fail("candidates", status);
    status = cudaMalloc(reinterpret_cast<void**>(&feed_.candidate_flags),
                        sizeof(int32_t));
    if (status != cudaSuccess) return fail("candidates", status);
    for (StepFeed::Handoff& handoff : feed_.handoff) {
      status = cudaHostAlloc(reinterpret_cast<void**>(&handoff.ids),
                             max_k * sizeof(int32_t), cudaHostAllocDefault);
      if (status != cudaSuccess) return fail("handoff", status);
      status = cudaHostAlloc(reinterpret_cast<void**>(&handoff.values),
                             max_k * sizeof(float), cudaHostAllocDefault);
      if (status != cudaSuccess) return fail("handoff", status);
      status = cudaHostAlloc(reinterpret_cast<void**>(&handoff.flags),
                             sizeof(int32_t), cudaHostAllocDefault);
      if (status != cudaSuccess) return fail("handoff", status);
    }
    // A slot's mirrors: the token, the embeddings, the positions, the
    // parameter (up to 16 values), the local mask.
    const size_t mirror_bytes =
        16 + static_cast<size_t>(feed.dim) * sizeof(float) + 16 +
        16 * sizeof(int32_t) + static_cast<size_t>(feed.ring) + 64;
    for (StepFeedSlot& slot : feed_.slots) {
      status = cudaHostAlloc(&slot.mirror, mirror_bytes,
                             cudaHostAllocMapped | cudaHostAllocPortable);
      if (status != cudaSuccess) return fail("mirrors", status);
      std::memset(slot.mirror, 0, mirror_bytes);
      uint8_t* cursor = static_cast<uint8_t*>(slot.mirror);
      slot.token_mirror = reinterpret_cast<int32_t*>(cursor);
      cursor += 16;
      slot.embeddings_mirror = reinterpret_cast<float*>(cursor);
      cursor += static_cast<size_t>(feed.dim) * sizeof(float);
      slot.positions_mirror = reinterpret_cast<int32_t*>(cursor);
      cursor += 16;
      slot.param_mirror = reinterpret_cast<int32_t*>(cursor);
      cursor += 16 * sizeof(int32_t);
      slot.local_mask_mirror = cursor;
      // With timing: the profile reports a step ahead's device time.
      status = cudaEventCreateWithFlags(&slot.feed_done, cudaEventDefault);
      if (status != cudaSuccess) return fail("event", status);
      status = cudaEventCreateWithFlags(&slot.step_done, cudaEventDefault);
      if (status != cudaSuccess) return fail("event", status);
      slot.staging.assign(output_handles_.size(), nullptr);
      slot.staging_bytes.assign(output_handles_.size(), 0);
    }
    int device = -1;
    status = cudaGetDevice(&device);
    if (status != cudaSuccess) return fail("device", status);
    feed_.sampler.runtime_context = runtime_context_;
    feed_.sampler.device = device;
    feed_.sampler.stream = stream_;
    if (EnsureNvidiaGreedySamplerResources(&feed_.sampler, device, feed.vocab,
                                           /*rows=*/1) != kLiteRtStatusOk) {
      return fail("argmax workspace", cudaErrorMemoryAllocation);
    }
    feed_.enabled = true;
    const std::string scales_text =
        feed_.scale_block == 0 ? "row"
                               : "block " + std::to_string(feed_.scale_block);
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch %s: step feed ready (table [%u, %u] %zu bytes "
               "from %s, %s scales, multiplier %g, ring %u, window %u, %d "
               "steps ahead)",
               bytecode_.function_name.c_str(), feed.vocab, feed.dim,
               table_bytes, table_source.c_str(), scales_text.c_str(),
               feed_.multiplier, feed.ring, feed.window, kStepFeedDepth);
    return {};
  }

  // The newest step ahead's event, for the waits.
  cudaEvent_t LastStepAheadEvent() const {
    if (feed_.in_flight == 0) return nullptr;
    return feed_.slots[(feed_.head + feed_.in_flight - 1) % kStepFeedDepth]
        .step_done;
  }

  void DestroyStepFeed() {
    if (cudaEvent_t last = LastStepAheadEvent(); last != nullptr) {
      cudaEventSynchronize(last);
    }
    feed_.in_flight = 0;
    device_context_->ClearStepAhead(this);
    if (feed_.enabled || feed_.speculated != 0) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA dispatch %s: step feed ran %llu steps ahead: %llu "
                 "taken, %llu missed, %llu sampled on the device%s",
                 bytecode_.function_name.c_str(),
                 static_cast<unsigned long long>(feed_.speculated),
                 static_cast<unsigned long long>(feed_.hits),
                 static_cast<unsigned long long>(feed_.misses),
                 static_cast<unsigned long long>(feed_.sampled),
                 feed_.stopped ? " (stopped)" : "");
    }
    for (StepFeedSlot& slot : feed_.slots) {
      if (slot.feed_done != nullptr) cudaEventDestroy(slot.feed_done);
      if (slot.step_done != nullptr) cudaEventDestroy(slot.step_done);
      if (slot.mirror != nullptr) cudaFreeHost(slot.mirror);
      if (slot.global_mask_mirror != nullptr)
        std::free(slot.global_mask_mirror);
      for (void* staging : slot.staging) {
        if (staging != nullptr) cudaFreeHost(staging);
      }
      slot = StepFeedSlot();
    }
    feed_.sampler.stream =
        nullptr;  // the context's stream is not the sampler's to destroy
    DestroyNvidiaGreedySamplerResources(&feed_.sampler);
    ClearHandoff();
    if (feed_.blocks_done != nullptr) cudaFree(feed_.blocks_done);
    if (feed_.candidate_ids != nullptr) cudaFree(feed_.candidate_ids);
    if (feed_.candidate_values != nullptr) cudaFree(feed_.candidate_values);
    if (feed_.candidate_flags != nullptr) cudaFree(feed_.candidate_flags);
    for (StepFeed::Handoff& handoff : feed_.handoff) {
      if (handoff.ids != nullptr) cudaFreeHost(handoff.ids);
      if (handoff.values != nullptr) cudaFreeHost(handoff.values);
      if (handoff.flags != nullptr) cudaFreeHost(handoff.flags);
    }
    if (feed_.owns_table) {
      if (feed_.row_scales != nullptr) cudaFree(feed_.row_scales);
      if (feed_.table != nullptr) cudaFree(feed_.table);
    }
    const bool stopped = feed_.stopped;
    const bool pending = feed_.pending;  // the table may still be registered
    const uint64_t counts[4] = {feed_.speculated, feed_.hits, feed_.misses,
                                feed_.sampled};
    feed_ = StepFeed();
    feed_.stopped = stopped;
    feed_.pending = pending;
    feed_.speculated = counts[0];
    feed_.hits = counts[1];
    feed_.misses = counts[2];
    feed_.sampled = counts[3];
  }

  // The device buffer and the bytes of the feed's input at `port`
  // (kTensorRtStepFeedNoPort: none).
  Expected<LiteRtDispatchDeviceContextT::Record*> FeedRecord(uint32_t port) {
    if (port == litert::nvidia::kTensorRtStepFeedNoPort ||
        port >= input_handles_.size() || input_handles_[port] == 0) {
      return nullptr;
    }
    LITERT_ASSIGN_OR_RETURN(auto* record,
                            device_context_->GetRecord(input_handles_[port]));
    if (!record->direct_cuda_buffer || record->transient_device_ptr ||
        record->device_ptr == nullptr) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "A step feed input is not a device buffer");
    }
    return record;
  }

  // Why the host's inputs are not the step ahead's: logged for the first
  // misses.
  void LogStepFeedMiss(const char* what, uint32_t port, size_t at,
                       double host_value, double mirror_value) {
    if (feed_.misses >= 8) return;
    const StepFeedSlot& slot = feed_.slots[feed_.head];
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch %s: the step ahead's %s (port %u) is not the "
               "host's at byte %zu: host %g, step ahead %g (position %d, "
               "token %d, handoff k %d)",
               bytecode_.function_name.c_str(), what, port, at, host_value,
               mirror_value, slot.positions_mirror[0], slot.token_mirror[0],
               feed_.handoff[slot.handoff].k);
  }

  // On an embeddings miss: the token the step ahead used against the argmax
  // of the logits the host read (its host copy), to tell a sampler that is
  // not greedy from a stale read.
  void DiagnoseEmbeddingsMiss(const StepFeedSlot& slot) {
    if (feed_.misses >= 8) return;
    const auto& feed = *bytecode_.step_feed;
    {
      // What the step ahead sampled from: ties among the candidates reorder
      // the CPU's sort by probability.
      const StepFeed::Handoff& handoff = feed_.handoff[slot.handoff];
      int ties = 0;
      for (int i = 1; i < handoff.k; ++i) {
        if (handoff.values[i] == handoff.values[i - 1]) ++ties;
      }
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA dispatch %s: the step ahead at position %d used "
                 "token %d from %d candidates (flag %d, %d ties; the first "
                 "%d %d %d, values %g %g %g)",
                 bytecode_.function_name.c_str(), slot.positions_mirror[0] - 1,
                 slot.token_mirror[0], handoff.k,
                 handoff.k > 0 ? handoff.flags[0] : -1, ties,
                 handoff.k > 0 ? handoff.ids[0] : -1,
                 handoff.k > 1 ? handoff.ids[1] : -1,
                 handoff.k > 2 ? handoff.ids[2] : -1,
                 handoff.k > 0 ? handoff.values[0] : 0.0f,
                 handoff.k > 1 ? handoff.values[1] : 0.0f,
                 handoff.k > 2 ? handoff.values[2] : 0.0f);
    }
    auto record =
        device_context_->GetRecord(output_handles_[feed.logits_output_port]);
    if (!record.HasValue()) return;
    const size_t count = feed.vocab;
    std::vector<uint8_t> host((*record)->size);
    if (!CudaTensorBufferRegistry::Get().ReadHostCopy(
            (*record)->cuda_handle, host.data(), host.size())) {
      return;
    }
    int32_t best = -1;  // first maximum, the logits read as their type
    if (host.size() == count * sizeof(float)) {
      const float* values = reinterpret_cast<const float*>(host.data());
      float best_value = -INFINITY;
      for (size_t i = 0; i < count; ++i) {
        if (values[i] > best_value) {
          best_value = values[i];
          best = static_cast<int32_t>(i);
        }
      }
    } else if (host.size() == count * sizeof(uint16_t)) {
      const uint16_t* bits = reinterpret_cast<const uint16_t*>(host.data());
      float best_value = -INFINITY;
      for (size_t i = 0; i < count; ++i) {
        const float half = __half2float_host(bits[i]);
        if (half > best_value) {
          best_value = half;
          best = static_cast<int32_t>(i);
        }
      }
    }
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch %s: the step ahead used token %d; the host "
               "copy of the logits (%zu bytes) has first-maximum argmax %d "
               "at position %d",
               bytecode_.function_name.c_str(), slot.token_mirror[0],
               host.size(), best, slot.positions_mirror[0] - 1);
  }

  // Whether the host's inputs for this step are the ones the oldest step
  // ahead used: the bindings, no host write to the logits, every host write
  // to a fed input matching its mirror (and clean afterwards), no other
  // input written.
  Expected<bool> VerifyStepFeedInputs() {
    const auto& feed = *bytecode_.step_feed;
    const StepFeedSlot& slot = feed_.slots[feed_.head];
    for (size_t i = 0; i < input_handles_.size(); ++i) {
      if (input_handles_[i] == 0) continue;
      auto record = device_context_->GetRecord(input_handles_[i]);
      if (!record.HasValue() || i >= bound_input_ptrs_.size() ||
          (*record)->device_ptr != bound_input_ptrs_[i]) {
        LogStepFeedMiss("input binding", static_cast<uint32_t>(i), 0, 0, 0);
        return false;
      }
    }
    for (size_t i = 0; i < output_handles_.size(); ++i) {
      if (output_handles_[i] == 0) continue;
      auto record = device_context_->GetRecord(output_handles_[i]);
      if (!record.HasValue() || i >= bound_output_ptrs_.size() ||
          (*record)->device_ptr != bound_output_ptrs_[i]) {
        LogStepFeedMiss("output binding", static_cast<uint32_t>(i), 0, 0, 0);
        return false;
      }
    }
    auto& registry = CudaTensorBufferRegistry::Get();
    {
      // The logits: a constraint's mask or a penalty.
      auto record =
          device_context_->GetRecord(output_handles_[feed.logits_output_port]);
      if (record.HasValue() && registry.AnyHostNewer((*record)->cuda_handle)) {
        LogStepFeedMiss("logits (written by the host)", feed.logits_output_port,
                        0, 0, 0);
        return false;
      }
    }
    const auto verify = [&](const char* what, uint32_t port, const void* mirror,
                            size_t bytes, size_t element) -> Expected<bool> {
      LITERT_ASSIGN_OR_RETURN(auto* record, FeedRecord(port));
      if (record == nullptr) return true;
      const size_t compared = std::min(bytes, record->size);
      if (registry.VerifyHostWrite(record->cuda_handle, mirror, compared)) {
        return true;
      }
      std::vector<uint8_t> host(compared);
      size_t at = compared;
      double host_value = 0.0;
      double mirror_value = 0.0;
      if (registry.ReadHostCopy(record->cuda_handle, host.data(), compared)) {
        const auto* expected = static_cast<const uint8_t*>(mirror);
        for (size_t i = 0; i < compared; ++i) {
          if (host[i] == expected[i]) continue;
          at = i - i % element;
          if (element == 4 && what[0] == 'e') {
            float h = 0.0f, m = 0.0f;
            std::memcpy(&h, host.data() + at, 4);
            std::memcpy(&m, expected + at, 4);
            host_value = h;
            mirror_value = m;
          } else if (element == 4) {
            int32_t h = 0, m = 0;
            std::memcpy(&h, host.data() + at, 4);
            std::memcpy(&m, expected + at, 4);
            host_value = h;
            mirror_value = m;
          } else {
            host_value = host[i];
            mirror_value = expected[i];
          }
          break;
        }
      }
      LogStepFeedMiss(what, port, at, host_value, mirror_value);
      if (what[0] == 'e') DiagnoseEmbeddingsMiss(slot);
      return false;
    };
    LITERT_ASSIGN_OR_RETURN(
        bool ok,
        verify("embeddings", feed.embeddings_port, slot.embeddings_mirror,
               static_cast<size_t>(feed.dim) * sizeof(float), 4));
    if (!ok) return false;
    LITERT_ASSIGN_OR_RETURN(
        ok, verify("positions", feed.positions_port, slot.positions_mirror,
                   sizeof(int32_t), 4));
    if (!ok) return false;
    LITERT_ASSIGN_OR_RETURN(
        ok,
        verify("param", feed.param_port, slot.param_mirror,
               static_cast<size_t>(feed_.param_count) * sizeof(int32_t), 4));
    if (!ok) return false;
    LITERT_ASSIGN_OR_RETURN(ok, verify("local mask", feed.local_mask_port,
                                       slot.local_mask_mirror, feed.ring, 1));
    if (!ok) return false;
    LITERT_ASSIGN_OR_RETURN(
        ok, verify("global mask", feed.global_mask_port,
                   slot.global_mask_mirror, feed_.global_bytes, 1));
    if (!ok) return false;
    for (size_t i = 0; i < input_handles_.size(); ++i) {
      if (input_handles_[i] == 0) continue;
      auto record = device_context_->GetRecord(input_handles_[i]);
      if (record.HasValue() && (*record)->direct_cuda_buffer &&
          registry.AnyHostNewer((*record)->cuda_handle)) {
        LogStepFeedMiss("other input (written by the host)",
                        static_cast<uint32_t>(i), 0, 0, 0);
        return false;
      }
    }
    return true;
  }

  // After a step's work on the stream: the outputs the host has read before
  // (their host copies exist and are pinned) are copied there now, so the
  // host's next read needs no download of its own and a step run ahead may
  // overwrite the device copy. LiteRT-LM reads the logits every step: the
  // copy (half a megabyte at a 262k vocabulary) and the lock's own
  // synchronization were most of its time between the steps. A step ahead
  // copies into its slot's pinned staging instead (the host may still be
  // reading the previous step's host copy); the staging goes to the host
  // copy when the host takes the step.
  // LITERT_NVIDIA_DISPATCH_PREFETCH_OUTPUTS=0 leaves the reads to the lock.
  static bool PrefetchOutputsEnabled() {
    static const bool enabled = [] {
      const char* value =
          std::getenv("LITERT_NVIDIA_DISPATCH_PREFETCH_OUTPUTS");
      return value == nullptr || value[0] == '\0' ||
             std::strcmp(value, "0") != 0;
    }();
    return enabled;
  }

  Expected<void> PrefetchReadOutputs(StepFeedSlot* slot) {
    if (slot == nullptr) prefetched_.clear();
    if (!PrefetchOutputsEnabled()) return {};
    constexpr size_t kMaxPrefetchBytes = 8u << 20;
    for (size_t i = 0; i < output_handles_.size(); ++i) {
      const auto handle = output_handles_[i];
      if (handle == 0) continue;
      LITERT_ASSIGN_OR_RETURN(auto* record, device_context_->GetRecord(handle));
      if (!record->direct_cuda_buffer || record->device_ptr == nullptr ||
          record->size == 0 || record->size > kMaxPrefetchBytes) {
        continue;
      }
      bool pinned = false;
      void* host = CudaTensorBufferRegistry::Get().HostCache(
          record->cuda_handle, record->size, &pinned);
      if (host == nullptr || !pinned) continue;
      void* destination = host;
      if (slot != nullptr) {
        if (slot->staging.size() != output_handles_.size()) {
          slot->staging.assign(output_handles_.size(), nullptr);
          slot->staging_bytes.assign(output_handles_.size(), 0);
        }
        if (slot->staging[i] == nullptr ||
            slot->staging_bytes[i] < record->size) {
          if (slot->staging[i] != nullptr) cudaFreeHost(slot->staging[i]);
          slot->staging[i] = nullptr;
          slot->staging_bytes[i] = 0;
          if (cudaMallocHost(&slot->staging[i], record->size) != cudaSuccess) {
            slot->staging[i] = nullptr;
            continue;
          }
          slot->staging_bytes[i] = record->size;
        }
        destination = slot->staging[i];
      }
      LITERT_RETURN_IF_ERROR(
          CudaOk(cudaMemcpyAsync(destination, record->device_ptr, record->size,
                                 cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpyAsync output prefetch"));
      if (slot == nullptr) {
        prefetched_.push_back(record->cuda_handle);
      } else {
        slot->prefetched.push_back(
            {i, record->cuda_handle, host, record->size});
      }
    }
    return {};
  }

  // Once the stream reached the prefetches of the step the host ran: the
  // host copies hold the outputs.
  void MarkPrefetched() {
    for (void* handle : prefetched_) {
      CudaTensorBufferRegistry::Get().MarkHostCurrent(handle);
    }
    prefetched_.clear();
  }

  // The host takes a step ahead: its staged outputs go to their host copies.
  void TakePrefetched(StepFeedSlot& slot) {
    for (const auto& prefetched : slot.prefetched) {
      std::memcpy(prefetched.host, slot.staging[prefetched.port],
                  prefetched.bytes);
      CudaTensorBufferRegistry::Get().MarkHostCurrent(prefetched.handle);
    }
    slot.prefetched.clear();
  }

  // After a step's work on the stream: the feed's greedy argmax of the
  // logits. Nothing when the feed is off or a sampler takes the logits'
  // argmax from the invocation (FusedGreedyRegistry).
  // The feed's token selection of the step the host samples next, for the
  // host (StepHandoff): the set's pinned copies, complete once the step is
  // synchronized or taken.
  void PublishHandoff(int set) {
    const auto& feed = *bytecode_.step_feed;
    auto record =
        device_context_->GetRecord(output_handles_[feed.logits_output_port]);
    if (!record.HasValue()) return;
    const StepFeed::Handoff& handoff = feed_.handoff[set];
    StepHandoffRegistry::Get().Set(
        (*record)->device_ptr,
        StepHandoff{handoff.ids, handoff.values, handoff.flags, handoff.k});
  }
  void ClearHandoff() {
    if (!bytecode_.step_feed.has_value() || output_handles_.empty()) return;
    const auto& feed = *bytecode_.step_feed;
    if (feed.logits_output_port >= output_handles_.size()) return;
    auto record =
        device_context_->GetRecord(output_handles_[feed.logits_output_port]);
    if (record.HasValue())
      StepHandoffRegistry::Get().Clear((*record)->device_ptr);
  }

  // The token of the logits the step just enqueued leaves, on the device:
  // the host's next sample but `uniform_offset` (0: the step the host
  // invoked, 1: the first step ahead, ...). The session's sampling when it
  // said how it samples and gave that draw (step_sampler_c_api.h): its
  // candidates and arithmetic on the device; otherwise the argmax. What was
  // selected goes to the next handoff set, noted on `slot` (a step ahead)
  // or as the host's own step's.
  Expected<void> EnqueueFeedToken(int uniform_offset, StepFeedSlot* slot) {
    feed_.argmax_ready = false;
    if (!feed_.enabled || feed_.stopped) return {};
    const auto& feed = *bytecode_.step_feed;
    LITERT_ASSIGN_OR_RETURN(
        auto* logits,
        device_context_->GetRecord(output_handles_[feed.logits_output_port]));
    if (!logits->direct_cuda_buffer || logits->device_ptr == nullptr) {
      return {};
    }
    const int set = feed_.next_handoff;
    feed_.next_handoff = (set + 1) % (kStepFeedDepth + 2);
    if (slot != nullptr) {
      slot->handoff = set;
    } else {
      feed_.host_handoff = set;
    }
    StepFeed::Handoff& handoff = feed_.handoff[set];
    handoff.k = 0;
    const size_t count = feed.vocab;
    const bool f32 = logits->size == count * sizeof(float);
    const bool f16 = logits->size == count * sizeof(uint16_t);
    int32_t k = 1;
    float p = 1.0f;
    float temperature = 1.0f;
    double uniform = 0.0;
    {
      auto& state = GetStepSamplerState();
      std::lock_guard<std::mutex> lock(state.mutex);
      if (state.k > 1 && uniform_offset >= 0 && uniform_offset < state.count) {
        k = state.k;
        p = state.p;
        temperature = state.temperature;
        uniform = state.uniforms[uniform_offset];
      }
    }
    if (k > 1 && k <= LiteRtNvidiaTopKCandidatesMaxK() && (f32 || f16) &&
        static_cast<size_t>(k) <= count) {
      const cudaError_t selected =
          f32 ? LiteRtNvidiaLaunchF32TopKCandidateRows(
                    static_cast<const float*>(logits->device_ptr), 1, count, k,
                    feed_.candidate_ids, feed_.candidate_values,
                    feed_.candidate_flags, stream_)
              : LiteRtNvidiaLaunchF16TopKCandidateRows(
                    static_cast<const uint16_t*>(logits->device_ptr), 1, count,
                    k, feed_.candidate_ids, feed_.candidate_values,
                    feed_.candidate_flags, stream_);
      LITERT_RETURN_IF_ERROR(CudaOk(selected, "Launch step feed candidates"));
      LITERT_RETURN_IF_ERROR(
          CudaOk(LiteRtNvidiaLaunchStepSample(
                     feed_.candidate_ids, feed_.candidate_values,
                     feed_.candidate_flags, k, p, temperature, uniform,
                     feed_.sampler.device_result, stream_),
                 "Launch step feed sample"));
      LITERT_RETURN_IF_ERROR(
          CudaOk(cudaMemcpyAsync(handoff.ids, feed_.candidate_ids,
                                 static_cast<size_t>(k) * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpyAsync step feed candidates"));
      LITERT_RETURN_IF_ERROR(
          CudaOk(cudaMemcpyAsync(handoff.values, feed_.candidate_values,
                                 static_cast<size_t>(k) * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpyAsync step feed candidates"));
      LITERT_RETURN_IF_ERROR(CudaOk(
          cudaMemcpyAsync(handoff.flags, feed_.candidate_flags, sizeof(int32_t),
                          cudaMemcpyDeviceToHost, stream_),
          "cudaMemcpyAsync step feed candidates"));
      handoff.k = k;
      ++feed_.sampled;
      feed_.argmax_ready = true;
      return {};
    }
    cudaError_t launch_status = cudaErrorInvalidValue;
    if (f32) {
      launch_status = LiteRtNvidiaLaunchF32ArgMaxRows(
          static_cast<const float*>(logits->device_ptr), 1, count,
          feed_.sampler.workspace, feed_.sampler.workspace_bytes,
          feed_.sampler.device_result, stream_);
    } else if (f16) {
      launch_status = LiteRtNvidiaLaunchF16ArgMaxRows(
          static_cast<const uint16_t*>(logits->device_ptr), 1, count,
          feed_.sampler.workspace, feed_.sampler.workspace_bytes,
          feed_.sampler.device_result, stream_);
    }
    LITERT_RETURN_IF_ERROR(CudaOk(launch_status, "Launch step feed argmax"));
    LITERT_RETURN_IF_ERROR(CudaOk(
        cudaMemcpyAsync(handoff.ids, feed_.sampler.device_result,
                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync step feed argmax"));
    handoff.flags[0] = 0;
    handoff.k = 1;
    feed_.argmax_ready = true;
    return {};
  }

  // Runs the steps ahead the slots allow, each over the greedy token of the
  // step before it: the feed kernel into the slot's mirrors, the step, its
  // argmax, its prefetches, its event. Nothing runs ahead without the last
  // step's argmax or after the feed stopped.
  Expected<void> SpeculateNextStep() {
    if (!feed_.enabled || feed_.stopped || !feed_.argmax_ready) return {};
    {
      // The session's sampler says how it samples at its first sample, after
      // the first step: until then, or past the second step, the argmax.
      auto& state = GetStepSamplerState();
      std::lock_guard<std::mutex> lock(state.mutex);
      if (!state.told && invocation_count_ <= 2) return {};
    }
    const auto& feed = *bytecode_.step_feed;
    LITERT_ASSIGN_OR_RETURN(auto* embeddings, FeedRecord(feed.embeddings_port));
    LITERT_ASSIGN_OR_RETURN(auto* positions, FeedRecord(feed.positions_port));
    LITERT_ASSIGN_OR_RETURN(auto* param, FeedRecord(feed.param_port));
    LITERT_ASSIGN_OR_RETURN(auto* local_mask, FeedRecord(feed.local_mask_port));
    LITERT_ASSIGN_OR_RETURN(auto* global_mask,
                            FeedRecord(feed.global_mask_port));
    if (embeddings == nullptr || positions == nullptr ||
        embeddings->size < static_cast<size_t>(feed.dim) * sizeof(float) ||
        positions->size < sizeof(int32_t) ||
        (param != nullptr && param->size < 3 * sizeof(int32_t)) ||
        (local_mask != nullptr && local_mask->size < feed.ring)) {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA dispatch %s: the step feed's inputs are not the "
                 "sizes the bytecode describes; the feed stops",
                 bytecode_.function_name.c_str());
      feed_.stopped = true;
      return {};
    }
    if (param != nullptr) {
      feed_.param_count = static_cast<int32_t>(
          std::min<size_t>(param->size / sizeof(int32_t), 16));
    }
    auto& registry = CudaTensorBufferRegistry::Get();
    // The position the host last ran (its copy), and the host's global mask
    // as it stands: the steps ahead follow from them.
    int32_t host_position = -1;
    if (!registry.ReadHostCopy(positions->cuda_handle, &host_position,
                               sizeof(host_position))) {
      return {};
    }
    std::vector<uint8_t> host_global_mask;
    if (global_mask != nullptr) {
      feed_.global_bytes = global_mask->size;
      host_global_mask.resize(feed_.global_bytes);
      if (!registry.ReadHostCopy(global_mask->cuda_handle,
                                 host_global_mask.data(), feed_.global_bytes)) {
        feed_.stopped = true;
        return {};
      }
    }
    uint32_t window = feed.window;
    if (local_mask != nullptr && window == 0) {
      if (feed_.learned_window == 0) {
        // The runtime's word on the window, before the host's masks show it.
        const uint32_t told = StepFeedLocalWindow().load();
        if (told != 0 && told <= feed.ring) feed_.learned_window = told;
      }
      if (feed_.learned_window == 0) {
        // The host's mask for this position: fewer keys than the position
        // allows means the window is full and that count.
        std::vector<uint8_t> host_mask(feed.ring);
        if (registry.ReadHostCopy(local_mask->cuda_handle, host_mask.data(),
                                  feed.ring)) {
          uint32_t count = 0;
          for (const uint8_t key : host_mask) count += key != 0 ? 1 : 0;
          if (host_position >= 0 && count > 0 &&
              count < static_cast<uint32_t>(host_position) + 1) {
            feed_.learned_window = count;
            LITERT_LOG(LITERT_INFO,
                       "NVIDIA dispatch %s: the step feed's local mask holds "
                       "%u keys of %u at position %d: the sliding window",
                       bytecode_.function_name.c_str(), count, feed.ring,
                       host_position);
          }
        }
      }
      window = feed_.learned_window != 0 ? feed_.learned_window : feed.ring;
    }
    while (feed_.in_flight < kStepFeedDepth) {
      StepFeedSlot& slot =
          feed_.slots[(feed_.head + feed_.in_flight) % kStepFeedDepth];
      const int32_t base =
          feed_.in_flight == 0 ? host_position : feed_.queued_position;
      slot.position = base + 1;
      if (global_mask != nullptr) {
        if (slot.global_mask_mirror == nullptr) {
          slot.global_mask_mirror =
              static_cast<uint8_t*>(std::malloc(feed_.global_bytes));
          if (slot.global_mask_mirror == nullptr) {
            feed_.stopped = true;
            return {};
          }
        }
        std::memcpy(slot.global_mask_mirror, host_global_mask.data(),
                    feed_.global_bytes);
        for (int32_t key = host_position + 1; key <= slot.position; ++key) {
          if (key >= 0 && static_cast<size_t>(key) < feed_.global_bytes) {
            slot.global_mask_mirror[key] = 1;
          }
        }
      }
      LiteRtNvidiaStepFeedParams params{};
      params.token = feed_.sampler.device_result;
      params.table = static_cast<const uint8_t*>(feed_.table);
      params.row_scales = feed_.row_scales;
      params.block_scales = feed_.block_scales;
      params.scale_block = static_cast<int32_t>(feed_.scale_block);
      params.multiplier = feed_.multiplier;
      params.vocab = static_cast<int32_t>(feed.vocab);
      params.dim = static_cast<int32_t>(feed.dim);
      params.embeddings = static_cast<float*>(embeddings->device_ptr);
      params.positions = static_cast<int32_t*>(positions->device_ptr);
      params.param =
          param != nullptr ? static_cast<int32_t*>(param->device_ptr) : nullptr;
      params.local_mask = local_mask != nullptr
                              ? static_cast<uint8_t*>(local_mask->device_ptr)
                              : nullptr;
      params.ring = static_cast<int32_t>(feed.ring);
      params.window = static_cast<int32_t>(window);
      params.global_mask = global_mask != nullptr
                               ? static_cast<uint8_t*>(global_mask->device_ptr)
                               : nullptr;
      params.global_keys =
          global_mask != nullptr ? static_cast<int32_t>(global_mask->size) : 0;
      params.blocks_done = feed_.blocks_done;
      params.token_mirror = slot.token_mirror;
      params.embeddings_mirror = slot.embeddings_mirror;
      params.positions_mirror = slot.positions_mirror;
      params.param_mirror = param != nullptr ? slot.param_mirror : nullptr;
      params.param_count = feed_.param_count;
      params.local_mask_mirror =
          local_mask != nullptr ? slot.local_mask_mirror : nullptr;
      LITERT_RETURN_IF_ERROR(CudaOk(
          LiteRtNvidiaLaunchStepFeed(&params, stream_), "LaunchStepFeed"));
      LITERT_RETURN_IF_ERROR(CudaOk(cudaEventRecord(slot.feed_done, stream_),
                                    "cudaEventRecord step feed"));
      if (!aux_streams_.empty()) {
        execution_context_->setAuxStreams(
            aux_streams_.data(), static_cast<int32_t>(aux_streams_.size()));
      }
      LITERT_RETURN_IF_ERROR(EnqueueStep());
      LITERT_RETURN_IF_ERROR(EnqueueFeedToken(feed_.in_flight + 1, &slot));
      slot.prefetched.clear();
      LITERT_RETURN_IF_ERROR(PrefetchReadOutputs(&slot));
      LITERT_RETURN_IF_ERROR(CudaOk(cudaEventRecord(slot.step_done, stream_),
                                    "cudaEventRecord step ahead"));
      ++feed_.in_flight;
      feed_.queued_position = slot.position;
      ++feed_.speculated;
      device_context_->SetStepAhead(this, slot.step_done);
    }
    // The host goes on to its own work now: the queued launches must reach
    // the device without waiting for the next synchronization (on WSL the
    // driver batches submissions; a query flushes them).
    static_cast<void>(cudaStreamQuery(stream_));
    return {};
  }

  // Waits for a step ahead: polling returns as soon as it is done, where
  // sleeping on the event wakes later (WSL); past 50 ms the event is slept
  // on.
  Expected<void> WaitStepAheadDone(cudaEvent_t event) {
    const auto poll_start = std::chrono::steady_clock::now();
    cudaError_t query = cudaErrorNotReady;
    while ((query = cudaEventQuery(event)) == cudaErrorNotReady &&
           std::chrono::steady_clock::now() - poll_start <
               std::chrono::milliseconds(50)) {
    }
    if (query == cudaErrorNotReady) {
      return CudaOk(cudaEventSynchronize(event),
                    "cudaEventSynchronize step ahead");
    }
    return CudaOk(query, "cudaEventQuery step ahead");
  }

  // The host asks for the oldest step ahead: its outputs are in place, its
  // staged outputs go to the host, and the slots refill.
  Expected<void> TakeStepAhead(DispatchProfileMetrics& profile_metrics,
                               bool profile) {
    StepFeedSlot& slot = feed_.slots[feed_.head];
    DispatchCpuTimer sync_timer(profile);
    LITERT_RETURN_IF_ERROR(WaitStepAheadDone(slot.step_done));
    profile_metrics.cpu_sync_ms = sync_timer.ElapsedMs();
    TakePrefetched(slot);
    PublishHandoff(slot.handoff);
    feed_.head = (feed_.head + 1) % kStepFeedDepth;
    --feed_.in_flight;
    if (feed_.in_flight == 0) device_context_->ClearStepAhead(this);
    ++feed_.hits;
    return SpeculateNextStep();
  }

  // The host's inputs are not the oldest step ahead's: every step ahead ran
  // for nothing; the real step overwrites their logits and cache rows.
  Expected<void> DropStepsAhead() {
    if (cudaEvent_t last = LastStepAheadEvent(); last != nullptr) {
      LITERT_RETURN_IF_ERROR(WaitStepAheadDone(last));
    }
    for (StepFeedSlot& slot : feed_.slots) slot.prefetched.clear();
    feed_.in_flight = 0;
    feed_.head = 0;
    device_context_->ClearStepAhead(this);
    return {};
  }

  // A plan built without its packed plugin weights (bytecode.h,
  // TensorRtWeightStore) gets them mapped in from device memory that is filled
  // from the model file and shared by the engines of the process.
  Expected<void> MapEngineWeights() {
    if (!bytecode_.weight_store.has_value()) {
      return {};
    }
    DispatchCpuTimer map_timer(/*enabled=*/true);
    memory_profiler_.Log("engine_weights_map_begin",
                         bytecode_.function_name.c_str());
    LITERT_ASSIGN_OR_RETURN(weight_memory_,
                            litert::nvidia::EngineWeightMemory::Create(
                                *bytecode_.weight_store, stream_));
    memory_profiler_.Log("engine_weights_on_device",
                         bytecode_.function_name.c_str());
    LITERT_ASSIGN_OR_RETURN(weight_mapping_,
                            litert::nvidia::EngineWeightMapping::Create(
                                *engine_, *weight_memory_, stream_));
    LITERT_LOG(
        LITERT_INFO,
        "NVIDIA dispatch mapped the weights of %s "
        "(weight_data_bytes=%llu segment_bytes=%llu "
        "shared_segment_bytes=%llu private_bytes=%llu elapsed_ms=%.3f)",
        bytecode_.function_name.c_str(),
        static_cast<unsigned long long>(weight_memory_->weight_data_size()),
        static_cast<unsigned long long>(weight_memory_->segment_bytes()),
        static_cast<unsigned long long>(weight_memory_->shared_segment_bytes()),
        static_cast<unsigned long long>(weight_memory_->private_bytes()),
        map_timer.ElapsedMs());
    memory_profiler_.Log("engine_weights_map_end",
                         bytecode_.function_name.c_str());
    return {};
  }

  Expected<void> RefitEngine() {
    if (bytecode_.refit_weights.empty()) {
      return {};
    }
    memory_profiler_.Log("engine_refit_begin", bytecode_.function_name.c_str());
    TrtPtr<nvinfer1::IRefitter> refitter(
        nvinfer1::createInferRefitter(*engine_, logger_));
    if (!refitter) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to create TensorRT refitter");
    }

    const int32_t refittable_count = refitter->getAllWeights(0, nullptr);
    if (refittable_count < 0) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to enumerate TensorRT refit weights");
    }
    std::vector<const char*> refittable_names(refittable_count);
    if (refitter->getAllWeights(refittable_count, refittable_names.data()) !=
        refittable_count) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "TensorRT refit weight enumeration changed");
    }
    std::unordered_set<std::string> refittable;
    refittable.reserve(refittable_names.size());
    for (const char* name : refittable_names) {
      if (name != nullptr) {
        refittable.emplace(name);
      }
    }

    size_t bound_weights = 0;
    size_t bound_bytes = 0;
    for (const auto& weight : bytecode_.refit_weights) {
      LITERT_ASSIGN_OR_RETURN(auto data_type,
                              ToTensorRtDataType(weight.data_type));
      if (weight.count >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT refit weight count is too large");
      }
      // TensorRT can eliminate named constants while optimizing the graph.
      // They are harmless in the shared store and need not be supplied.
      if (refittable.find(weight.name) == refittable.end()) {
        continue;
      }
      const nvinfer1::Weights prototype =
          refitter->getWeightsPrototype(weight.name.c_str());
      if (prototype.type != data_type ||
          prototype.count != static_cast<int64_t>(weight.count)) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "TensorRT refit weight metadata does not match plan");
      }
      if (!refitter->setNamedWeights(
              weight.name.c_str(),
              nvinfer1::Weights{data_type, weight.data,
                                static_cast<int64_t>(weight.count)})) {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "TensorRT rejected a named refit weight");
      }
      ++bound_weights;
      bound_bytes += weight.size;
    }
    memory_profiler_.Log("engine_refit_weights_bound",
                         bytecode_.function_name.c_str());
    const int32_t missing_count = refitter->getMissingWeights(0, nullptr);
    if (missing_count < 0) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to enumerate missing TensorRT refit weights");
    }
    if (missing_count != 0) {
      std::vector<const char*> missing_names(missing_count);
      if (refitter->getMissingWeights(missing_count, missing_names.data()) !=
          missing_count) {
        return Error(kLiteRtStatusErrorRuntimeFailure,
                     "TensorRT missing refit weight enumeration changed");
      }
      for (int32_t i = 0; i < std::min<int32_t>(missing_count, 8); ++i) {
        LITERT_LOG(LITERT_ERROR,
                   "NVIDIA TensorRT-RTX missing refit weight for %s: %s",
                   bytecode_.function_name.c_str(), missing_names[i]);
      }
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "TensorRT stripped plan has missing refit weights");
    }
    if (!refitter->refitCudaEngine()) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to refit TensorRT stripped plan");
    }
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch refitted TensorRT plan for %s "
               "(weights=%zu bytes=%zu total_engine_weight_bytes=%lld "
               "stripped_engine_weight_bytes=%lld)",
               bytecode_.function_name.c_str(), bound_weights, bound_bytes,
               static_cast<long long>(engine_->getEngineStat(
                   nvinfer1::EngineStat::kTOTAL_WEIGHTS_SIZE)),
               static_cast<long long>(engine_->getEngineStat(
                   nvinfer1::EngineStat::kSTRIPPED_WEIGHTS_SIZE)));
    memory_profiler_.Log("engine_refit_end", bytecode_.function_name.c_str());
    return {};
  }

  Expected<void> InitializeExternalHead() {
    if (!bytecode_.trtllm_head.has_value()) {
      for (const auto& output_name : bytecode_.output_names) {
        if (output_name.empty()) {
          return Error(kLiteRtStatusErrorInvalidArgument,
                       "TensorRT bytecode has an empty output binding without "
                       "an external vocabulary head");
        }
      }
      return {};
    }

    const auto& head = *bytecode_.trtllm_head;
    if (head.weight_format !=
        litert::nvidia::TensorRtLlmHeadWeightFormat::kInt2TfliteRowMajor) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "TensorRT-RTX dispatch supports only the raw W2 external "
                   "vocabulary-head format");
    }

    int device = 0;
    cudaDeviceProp properties{};
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaGetDevice(&device), "cudaGetDevice external W2 head"));
    LITERT_RETURN_IF_ERROR(CudaOk(cudaGetDeviceProperties(&properties, device),
                                  "cudaGetDeviceProperties external W2 head"));
    const int compute_capability = properties.major * 10 + properties.minor;
    if (!litert::nvidia::IsInt2GemvComputeCapabilitySupported(
            compute_capability)) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "The external W2 vocabulary head requires SM80 or newer");
    }

    const size_t num_outputs = bytecode_.output_names.size();
    if (head.hidden_output_port >= num_outputs ||
        head.logits_output_port >= num_outputs ||
        head.hidden_output_port == head.logits_output_port ||
        bytecode_.output_names[head.hidden_output_port].empty() ||
        !bytecode_.output_names[head.logits_output_port].empty()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "External W2 head output bindings are invalid");
    }
    for (size_t i = 0; i < num_outputs; ++i) {
      if (i != head.logits_output_port && bytecode_.output_names[i].empty()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "External W2 head has an unexpected empty output "
                     "binding");
      }
    }

    if (head.k == 0 || head.n == 0 || head.k % 16 != 0 ||
        head.k > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        head.n > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        !std::isfinite(head.soft_cap) || head.soft_cap <= 0.0f) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "External W2 head dimensions or soft cap are invalid");
    }
    const uint64_t weight_elements =
        static_cast<uint64_t>(head.k) * static_cast<uint64_t>(head.n);
    const uint64_t expected_weights_size = (weight_elements + 3) / 4;
    const uint64_t expected_scales_size =
        static_cast<uint64_t>(head.n) * sizeof(uint16_t);
    if (head.packed_weights == nullptr ||
        head.packed_weights_size != expected_weights_size ||
        head.bf16_scales == nullptr ||
        head.bf16_scales_size != expected_scales_size) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "External W2 head payload size is invalid");
    }

    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaMalloc(&head_weights_, head.packed_weights_size),
               "cudaMalloc external W2 head weights"));
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaMemcpy(head_weights_, head.packed_weights,
                          head.packed_weights_size, cudaMemcpyHostToDevice),
               "cudaMemcpy external W2 head weights H2D"));
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaMalloc(&head_scales_, head.bf16_scales_size),
               "cudaMalloc external W2 head BF16 scales"));
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaMemcpy(head_scales_, head.bf16_scales, head.bf16_scales_size,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy external W2 head BF16 scales H2D"));
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaMalloc(&head_input_bf16_,
                          static_cast<size_t>(head.k) * sizeof(uint16_t)),
               "cudaMalloc external W2 head BF16 input scratch"));
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaMalloc(&head_output_bf16_,
                          static_cast<size_t>(head.n) * sizeof(uint16_t)),
               "cudaMalloc external W2 head BF16 output scratch"));

    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch initialized external W2 vocabulary head for "
               "%s (hidden_output=%u logits_output=%u k=%u n=%u "
               "weights_bytes=%zu scales_bytes=%zu)",
               bytecode_.function_name.c_str(), head.hidden_output_port,
               head.logits_output_port, head.k, head.n,
               head.packed_weights_size, head.bf16_scales_size);
    return {};
  }

  Expected<void> LaunchExternalHead() {
    if (!bytecode_.trtllm_head.has_value()) {
      return {};
    }
    const auto& head = *bytecode_.trtllm_head;
    LITERT_ASSIGN_OR_RETURN(
        auto* hidden_record,
        device_context_->GetRecord(output_handles_[head.hidden_output_port]));
    LITERT_ASSIGN_OR_RETURN(
        auto* logits_record,
        device_context_->GetRecord(output_handles_[head.logits_output_port]));
    const size_t required_hidden_bytes =
        static_cast<size_t>(head.k) * sizeof(float);
    const size_t required_logits_bytes =
        static_cast<size_t>(head.n) * sizeof(float);
    if (hidden_record->device_ptr == nullptr ||
        hidden_record->size < required_hidden_bytes ||
        logits_record->device_ptr == nullptr ||
        logits_record->size < required_logits_bytes) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "External W2 head LiteRT output buffer is too small");
    }

    LITERT_RETURN_IF_ERROR(
        CudaOk(LiteRtNvidiaLaunchF32ToBf16(
                   static_cast<const float*>(hidden_record->device_ptr),
                   head_input_bf16_, head.k, stream_),
               "LaunchF32ToBf16 external W2 head"));
    LITERT_RETURN_IF_ERROR(
        CudaOk(LiteRtNvidiaLaunchBf16Int2PerChannelGemv(
                   head_input_bf16_, static_cast<const uint8_t*>(head_weights_),
                   head_scales_, head_output_bf16_, static_cast<int>(head.k),
                   static_cast<int>(head.n), stream_),
               "LaunchBf16Int2PerChannelGemv external W2 head"));
    LITERT_RETURN_IF_ERROR(CudaOk(
        LiteRtNvidiaLaunchBf16SoftCapToF32(
            head_output_bf16_, static_cast<float*>(logits_record->device_ptr),
            head.n, head.soft_cap, stream_),
        "LaunchBf16SoftCapToF32 external W2 head"));
    return {};
  }

  void DestroyExternalHeadResources() {
    auto free_device_memory = [](void*& pointer, const char* what) {
      if (pointer == nullptr) {
        return;
      }
      const cudaError_t status = cudaFree(pointer);
      if (status != cudaSuccess) {
        LITERT_LOG(LITERT_ERROR, "%s: %s", what, cudaGetErrorString(status));
      }
      pointer = nullptr;
    };
    free_device_memory(head_output_bf16_,
                       "cudaFree external W2 head BF16 output scratch");
    free_device_memory(head_input_bf16_,
                       "cudaFree external W2 head BF16 input scratch");
    free_device_memory(head_scales_, "cudaFree external W2 head BF16 scales");
    free_device_memory(head_weights_, "cudaFree external W2 head weights");
  }

  // TensorRT-RTX runs parallel graph branches on auxiliary streams. Unless
  // the application provides them, the runtime creates and destroys its own
  // on every enqueue (about fifteen stream create/destroy calls per Gemma 4
  // decode step in the CUDA API trace), so hand it persistent streams once.
  Expected<void> CreateAuxStreams() {
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_AUX_STREAMS");
    if (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") == 0) {
      return {};
    }
    const int32_t count = engine_->getNbAuxStreams();
    for (int32_t i = 0; i < count; ++i) {
      cudaStream_t aux_stream = nullptr;
      LITERT_RETURN_IF_ERROR(
          CudaOk(cudaStreamCreateWithFlags(&aux_stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags auxiliary"));
      aux_streams_.push_back(aux_stream);
    }
    if (count > 0) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA dispatch %s provides %d auxiliary CUDA streams",
                 bytecode_.function_name.c_str(), count);
    }
    return {};
  }

  static bool UseSharedActivationArena() {
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_SHARED_ARENA");
    if (value != nullptr && value[0] != '\0') {
      return std::strcmp(value, "0") != 0;
    }
    return true;
  }

  static bool UseCudaGraph() {
    // Layer profiling requires per-kernel synchronization; it supersedes
    // graph capture.
    if (litert::nvidia::DispatchLayerProfilingEnabled()) {
      return false;
    }
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_CUDA_GRAPH");
    if (value != nullptr && value[0] != '\0') {
      return std::strcmp(value, "0") != 0 && std::strncmp(value, "own", 3) != 0;
    }
    return true;
  }

  // LITERT_NVIDIA_DISPATCH_CUDA_GRAPH=own|own-relaxed: TensorRT-RTX's graph
  // strategy stays off and this context captures its own step graph around
  // enqueueV3 and the external head (EnqueueStep). TensorRT-RTX's whole-graph
  // capture leaves plugin layers out, so each of their launches pays WSL's
  // launch latency every step; a capture around the whole enqueue takes them
  // all. `own` captures in thread-local mode (an unsafe CUDA call during the
  // capture fails it, loudly); `own-relaxed` lets such calls through.
  static bool OwnCudaGraph(cudaStreamCaptureMode* mode) {
    if (litert::nvidia::DispatchLayerProfilingEnabled()) {
      return false;
    }
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_CUDA_GRAPH");
    if (value == nullptr) {
      return false;
    }
    if (std::strcmp(value, "own") == 0) {
      *mode = cudaStreamCaptureModeThreadLocal;
      return true;
    }
    if (std::strcmp(value, "own-relaxed") == 0) {
      *mode = cudaStreamCaptureModeRelaxed;
      return true;
    }
    return false;
  }

  // The step graphs of this context's own (OwnCudaGraph): the bindings they
  // were captured for, or seen once without a graph yet (exec null).
  struct OwnGraph {
    std::vector<void*> inputs;
    std::vector<void*> outputs;
    void* arena = nullptr;
    cudaGraphExec_t exec = nullptr;
    int64_t replays = 0;
  };

  bool SameBindings(const OwnGraph& graph) const {
    return graph.inputs == bound_input_ptrs_ &&
           graph.outputs == bound_output_ptrs_ &&
           graph.arena == bound_arena_ptr_;
  }

  void DestroyOwnGraphs() {
    for (auto& graph : own_graphs_) {
      if (graph.exec != nullptr) {
        cudaGraphExecDestroy(graph.exec);
      }
    }
    own_graphs_.clear();
  }

  // The engine, then the external head, on stream_.
  Expected<void> EnqueueDirect() {
    if (!execution_context_->enqueueV3(stream_)) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "TensorRT enqueueV3 failed");
    }
    return LaunchExternalHead();
  }

  // The step's work: enqueued directly, or (OwnCudaGraph) replayed from the
  // graph captured for the current bindings. Bindings seen for the first
  // time run directly, so that lazy work (expert arenas, kernel
  // specialization) stays out of the graph; their second step captures.
  Expected<void> EnqueueStep() {
    if (!own_graph_enabled_) {
      return EnqueueDirect();
    }
    for (auto& graph : own_graphs_) {
      if (!SameBindings(graph)) {
        continue;
      }
      if (graph.exec == nullptr) {
        return CaptureOwnGraph(&graph);
      }
      LITERT_RETURN_IF_ERROR(CudaOk(cudaGraphLaunch(graph.exec, stream_),
                                    "cudaGraphLaunch step graph"));
      ++graph.replays;
      return {};
    }
    constexpr size_t kMaxOwnGraphs = 4;
    if (own_graphs_.size() >= kMaxOwnGraphs) {
      if (own_graphs_.front().exec != nullptr) {
        cudaGraphExecDestroy(own_graphs_.front().exec);
      }
      own_graphs_.erase(own_graphs_.begin());
    }
    own_graphs_.push_back(OwnGraph{bound_input_ptrs_, bound_output_ptrs_,
                                   bound_arena_ptr_, nullptr, 0});
    ++own_graph_direct_;
    return EnqueueDirect();
  }

  Expected<void> CaptureOwnGraph(OwnGraph* graph) {
    const char* name = bytecode_.function_name.c_str();
    cudaError_t status = cudaStreamBeginCapture(stream_, own_graph_mode_);
    if (status != cudaSuccess) {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA dispatch %s: cudaStreamBeginCapture: %s; the step "
                 "runs without its own graph",
                 name, cudaGetErrorString(status));
      static_cast<void>(cudaGetLastError());
      own_graph_enabled_ = false;
      return EnqueueDirect();
    }
    const auto enqueued = EnqueueDirect();
    cudaGraph_t captured = nullptr;
    status = cudaStreamEndCapture(stream_, &captured);
    if (!enqueued || status != cudaSuccess || captured == nullptr) {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA dispatch %s: capturing the step failed (%s%s); the "
                 "step runs without its own graph",
                 name, enqueued ? "" : "enqueue failed, ",
                 cudaGetErrorString(status));
      if (captured != nullptr) {
        cudaGraphDestroy(captured);
      }
      static_cast<void>(cudaGetLastError());
      own_graph_enabled_ = false;
      // Captured work never ran: enqueue the step for real.
      return EnqueueDirect();
    }
    ++own_graph_captures_;
    LogOwnGraph(captured);
    AutopsyOwnGraph(captured);
    TimelineOwnGraph(captured);
    cudaGraphExec_t exec = nullptr;
    status = cudaGraphInstantiate(&exec, captured, 0);
    cudaGraphDestroy(captured);
    if (status != cudaSuccess || exec == nullptr) {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA dispatch %s: cudaGraphInstantiate: %s; the step runs "
                 "without its own graph",
                 name, cudaGetErrorString(status));
      static_cast<void>(cudaGetLastError());
      own_graph_enabled_ = false;
      return EnqueueDirect();
    }
    graph->exec = exec;
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaGraphLaunch(exec, stream_), "cudaGraphLaunch step graph"));
    ++graph->replays;
    return {};
  }

  // LITERT_NVIDIA_DISPATCH_CUDA_GRAPH_AUTOPSY=path: every kernel node of the
  // first captured step graph of each function timed on its own (a clone of
  // the graph with the other nodes removed, written to path.<function>), warm
  // and cold (two passes of a kernel that rewrites
  // 128 MB ahead of it flush L2; their own time is subtracted). A graph launch
  // costs 10-16 us here, more than most kernels, so each measurement captures
  // several launches of the node's graph into one graph and divides. One line
  // per node: index (the DOT's ID), grid, block, cold us, warm us. The CUPTI
  // tools do not see kernels on this host; this is the step's per-kernel
  // profile.
  void AutopsyOwnGraph(cudaGraph_t graph) {
    const char* path = std::getenv("LITERT_NVIDIA_DISPATCH_CUDA_GRAPH_AUTOPSY");
    if (path == nullptr || path[0] == '\0' || autopsy_done_) {
      return;
    }
    autopsy_done_ = true;
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess ||
        count == 0) {
      return;
    }
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) {
      return;
    }
    constexpr size_t kFlushBytes = size_t{128} << 20;
    void* flush = nullptr;
    if (cudaMalloc(&flush, kFlushBytes) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      return;
    }
    // One file per function: <path>.<function>, since every context that
    // captures a step writes its own autopsy.
    const std::string function_path =
        std::string(path) + "." +
        SanitizeCacheComponent(bytecode_.function_name);
    FILE* out = std::fopen(function_path.c_str(), "w");
    if (out == nullptr) {
      cudaFree(flush);
      return;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int kIters = 10;
    auto time_graph = [&](cudaGraph_t g) -> double {
      cudaGraphExec_t exec = nullptr;
      if (cudaGraphInstantiate(&exec, g, 0) != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return -1.0;
      }
      for (int i = 0; i < 2; ++i) {
        cudaGraphLaunch(exec, stream_);
      }
      cudaEventRecord(start, stream_);
      for (int i = 0; i < kIters; ++i) {
        cudaGraphLaunch(exec, stream_);
      }
      cudaEventRecord(stop, stream_);
      cudaEventSynchronize(stop);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start, stop);
      cudaGraphExecDestroy(exec);
      return ms * 1000.0 / kIters;
    };
    constexpr int kRepeatWarm = 10;
    constexpr int kRepeatCold = 4;
    // Two passes of the flush, captured once as a graph.
    cudaGraph_t flush_graph = nullptr;
    if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) ==
        cudaSuccess) {
      LiteRtNvidiaLaunchL2Flush(flush, kFlushBytes, stream_);
      LiteRtNvidiaLaunchL2Flush(flush, kFlushBytes, stream_);
      cudaStreamEndCapture(stream_, &flush_graph);
    }
    if (flush_graph == nullptr) {
      static_cast<void>(cudaGetLastError());
      std::fclose(out);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(flush);
      return;
    }
    // `repeat` rounds of [flush,] node graph, chained as child graph nodes
    // of one graph; its time per round.
    auto time_sequence = [&](cudaGraph_t node_graph, bool with_flush,
                             int repeat) -> double {
      cudaGraph_t g = nullptr;
      if (cudaGraphCreate(&g, 0) != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return -1.0;
      }
      cudaGraphNode_t previous = nullptr;
      bool ok = true;
      auto append = [&](cudaGraph_t child) {
        cudaGraphNode_t node = nullptr;
        ok = ok && cudaGraphAddChildGraphNode(
                       &node, g, previous == nullptr ? nullptr : &previous,
                       previous == nullptr ? 0 : 1, child) == cudaSuccess;
        previous = node;
      };
      for (int i = 0; i < repeat && ok; ++i) {
        if (with_flush) {
          append(flush_graph);
        }
        if (node_graph != nullptr) {
          append(node_graph);
        }
      }
      double us = -1.0;
      if (ok) {
        us = time_graph(g);
      } else {
        static_cast<void>(cudaGetLastError());
      }
      cudaGraphDestroy(g);
      return us < 0.0 ? us : us / repeat;
    };
    const double flush_us = time_sequence(nullptr, true, kRepeatCold);
    std::fprintf(out, "# flush_us %.1f\n", flush_us);
    std::fprintf(out, "# node grid_x grid_y grid_z block_x cold_us warm_us\n");
    size_t timed = 0;
    for (size_t i = 0; i < count; ++i) {
      cudaGraphNodeType type;
      if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess ||
          type != cudaGraphNodeTypeKernel) {
        continue;
      }
      cudaGraph_t clone = nullptr;
      if (cudaGraphClone(&clone, graph) != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        break;
      }
      cudaGraphNode_t keep = nullptr;
      if (cudaGraphNodeFindInClone(&keep, nodes[i], clone) != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        cudaGraphDestroy(clone);
        break;
      }
      size_t clone_count = 0;
      cudaGraphGetNodes(clone, nullptr, &clone_count);
      std::vector<cudaGraphNode_t> clone_nodes(clone_count);
      cudaGraphGetNodes(clone, clone_nodes.data(), &clone_count);
      for (cudaGraphNode_t n : clone_nodes) {
        if (n != keep) {
          cudaGraphDestroyNode(n);
        }
      }
      const double warm = time_sequence(clone, false, kRepeatWarm);
      const double cold = time_sequence(clone, true, kRepeatCold) - flush_us;
      cudaKernelNodeParams params{};
      dim3 grid(0, 0, 0);
      dim3 block(0, 0, 0);
      if (cudaGraphKernelNodeGetParams(keep, &params) == cudaSuccess) {
        grid = params.gridDim;
        block = params.blockDim;
      } else {
        static_cast<void>(cudaGetLastError());
      }
      std::fprintf(out, "%zu %u %u %u %u %.2f %.2f\n", i, grid.x, grid.y,
                   grid.z, block.x, cold, warm);
      cudaGraphDestroy(clone);
      ++timed;
    }
    std::fclose(out);
    cudaGraphDestroy(flush_graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(flush);
    static_cast<void>(cudaGetLastError());
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch %s: autopsy of %zu kernel nodes written to %s "
               "(flush %.1f us)",
               bytecode_.function_name.c_str(), timed, path, flush_us);
  }

  // LITERT_NVIDIA_DISPATCH_CUDA_GRAPH_TIMELINE=path: the first captured step
  // graph of each function replayed with an event-record node after every
  // node, so that each node's time in the real step, with its launch gap and
  // its real data (the autopsy above replays a node alone, which a routed
  // expert kernel or a chunked attention may not survive), is the time
  // between its event and the one before. Written to path.<function>: one
  // line per node, index (the DOT's ID), type (K kernel, M memcpy, S memset,
  // O other), end us from the step's start and its duration; the third
  // launch of the instrumented graph is the one read.
  void TimelineOwnGraph(cudaGraph_t graph) {
    const char* path =
        std::getenv("LITERT_NVIDIA_DISPATCH_CUDA_GRAPH_TIMELINE");
    if (path == nullptr || path[0] == '\0' || timeline_done_) {
      return;
    }
    timeline_done_ = true;
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess ||
        count == 0) {
      return;
    }
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) {
      return;
    }
    // Only graphs of kernel and memset nodes: a replay of a step graph that
    // copies (an aliased cache input seeded from its output, a deferred
    // upload from host memory) is not the step's own work, and the 26B's
    // prefill graph did not survive three replays with events.
    std::vector<char> types(count, 'O');
    for (size_t i = 0; i < count; ++i) {
      cudaGraphNodeType type;
      if (cudaGraphNodeGetType(nodes[i], &type) == cudaSuccess) {
        types[i] = type == cudaGraphNodeTypeKernel   ? 'K'
                   : type == cudaGraphNodeTypeMemcpy ? 'M'
                   : type == cudaGraphNodeTypeMemset ? 'S'
                                                     : 'O';
      }
      if (types[i] != 'K' && types[i] != 'S') {
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA dispatch %s: no timeline (node %zu is not a kernel "
                   "or a memset)",
                   bytecode_.function_name.c_str(), i);
        return;
      }
    }
    cudaGraph_t clone = nullptr;
    if (cudaGraphClone(&clone, graph) != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      return;
    }
    std::vector<cudaEvent_t> events(count + 1, nullptr);
    bool ok = true;
    for (size_t i = 0; ok && i <= count; ++i) {
      ok =
          cudaEventCreateWithFlags(&events[i], cudaEventDefault) == cudaSuccess;
    }
    // Every event node sits on the chain: it depends on its node and the
    // node's dependents depend on it, so an event records when its node is
    // done and before the next node starts. The start event precedes the
    // roots.
    std::vector<cudaGraphNode_t> in_clone(count, nullptr);
    for (size_t i = 0; ok && i < count; ++i) {
      ok = cudaGraphNodeFindInClone(&in_clone[i], nodes[i], clone) ==
           cudaSuccess;
    }
    cudaGraphNode_t event_node = nullptr;
    if (ok) {
      ok = cudaGraphAddEventRecordNode(&event_node, clone, nullptr, 0,
                                       events[count]) == cudaSuccess;
    }
    for (size_t i = 0; ok && i < count; ++i) {
      size_t dependencies = 0;
      cudaGraphNodeGetDependencies(in_clone[i], nullptr, &dependencies);
      if (dependencies == 0) {
        ok = cudaGraphAddDependencies(clone, &event_node, &in_clone[i], 1) ==
             cudaSuccess;
      }
    }
    for (size_t i = 0; ok && i < count; ++i) {
      size_t dependents = 0;
      cudaGraphNodeGetDependentNodes(in_clone[i], nullptr, &dependents);
      std::vector<cudaGraphNode_t> next(dependents);
      if (dependents > 0) {
        ok = cudaGraphNodeGetDependentNodes(in_clone[i], next.data(),
                                            &dependents) == cudaSuccess;
      }
      cudaGraphNode_t after = nullptr;
      ok = ok && cudaGraphAddEventRecordNode(&after, clone, &in_clone[i], 1,
                                             events[i]) == cudaSuccess;
      for (size_t d = 0; ok && d < dependents; ++d) {
        ok =
            cudaGraphAddDependencies(clone, &after, &next[d], 1) == cudaSuccess;
      }
    }
    cudaGraphExec_t exec = nullptr;
    if (ok) {
      ok = cudaGraphInstantiate(&exec, clone, 0) == cudaSuccess;
    }
    for (int launch = 0; ok && launch < 3; ++launch) {
      ok = cudaGraphLaunch(exec, stream_) == cudaSuccess &&
           cudaStreamSynchronize(stream_) == cudaSuccess;
    }
    FILE* out =
        ok ? std::fopen((std::string(path) + "." +
                         SanitizeCacheComponent(bytecode_.function_name))
                            .c_str(),
                        "w")
           : nullptr;
    if (out != nullptr) {
      std::vector<std::pair<float, size_t>> ends;
      ends.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, events[count], events[i]) ==
            cudaSuccess) {
          ends.emplace_back(ms * 1000.0f, i);
        } else {
          static_cast<void>(cudaGetLastError());
        }
      }
      std::sort(ends.begin(), ends.end());
      const float total = ends.empty() ? 0.0f : ends.back().first;
      std::fprintf(out, "# total_us %.1f nodes %zu\n", total, count);
      std::fprintf(out, "# node type end_us dur_us\n");
      float previous = 0.0f;
      for (const auto& [end, i] : ends) {
        std::fprintf(out, "%zu %c %.1f %.1f\n", i, types[i], end,
                     end - previous);
        previous = end;
      }
      std::fclose(out);
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA dispatch %s: timeline of %zu nodes written to %s "
                 "(%.1f us)",
                 bytecode_.function_name.c_str(), count, path, total);
    } else {
      LITERT_LOG(LITERT_WARNING, "NVIDIA dispatch %s: no timeline (%s)",
                 bytecode_.function_name.c_str(),
                 cudaGetErrorString(cudaGetLastError()));
    }
    if (exec != nullptr) cudaGraphExecDestroy(exec);
    cudaGraphDestroy(clone);
    for (cudaEvent_t e : events) {
      if (e != nullptr) cudaEventDestroy(e);
    }
    static_cast<void>(cudaGetLastError());
  }

  // What the captured step holds; LITERT_NVIDIA_DISPATCH_CUDA_GRAPH_DOT=dir
  // also writes it as a DOT file (kernel names, grids and arguments).
  void LogOwnGraph(cudaGraph_t graph) {
    const char* name = bytecode_.function_name.c_str();
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) {
      return;
    }
    std::vector<cudaGraphNode_t> nodes(count);
    if (count > 0 &&
        cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) {
      return;
    }
    int kernels = 0, copies = 0, memsets = 0, events = 0, others = 0;
    for (cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type;
      if (cudaGraphNodeGetType(node, &type) != cudaSuccess) {
        continue;
      }
      switch (type) {
        case cudaGraphNodeTypeKernel:
          ++kernels;
          break;
        case cudaGraphNodeTypeMemcpy:
          ++copies;
          break;
        case cudaGraphNodeTypeMemset:
          ++memsets;
          break;
        case cudaGraphNodeTypeWaitEvent:
        case cudaGraphNodeTypeEventRecord:
          ++events;
          break;
        default:
          ++others;
          break;
      }
    }
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch %s captured its step: %zu graph nodes (%d "
               "kernels, %d copies, %d memsets, %d events, %d other)",
               name, count, kernels, copies, memsets, events, others);
    const char* dot_dir = std::getenv("LITERT_NVIDIA_DISPATCH_CUDA_GRAPH_DOT");
    if (dot_dir != nullptr && dot_dir[0] != '\0') {
      const std::string path = JoinPath(
          dot_dir, SanitizeCacheComponent(bytecode_.function_name) + "_" +
                       std::to_string(own_graph_captures_) + ".dot");
      const cudaError_t status = cudaGraphDebugDotPrint(
          graph, path.c_str(), cudaGraphDebugDotFlagsVerbose);
      LITERT_LOG(LITERT_INFO, "NVIDIA dispatch %s step graph to %s: %s", name,
                 path.c_str(), cudaGetErrorString(status));
    }
  }

  std::string RuntimeCachePath(const std::string& dir) const {
    const std::string function_name =
        SanitizeCacheComponent(bytecode_.function_name);
    const uint64_t engine_hash =
        Fnv1a64(bytecode_.engine_data, bytecode_.engine_size);
    return JoinPath(dir, function_name + "_" + Hex64(engine_hash) +
                             ".trt_rtx_runtime_cache");
  }

  Expected<void> ConfigureRuntimeCache() {
    const std::string cache_dir = RuntimeCacheDir();
    if (cache_dir.empty() || runtime_config_ == nullptr) {
      return {};
    }
    // A lazy engine always reloads the same validated AOT artifact. Reuse its
    // cache key rather than rescan a multi-GiB plan at every phase transition.
    if (!aot_residency_ || runtime_cache_path_.empty()) {
      runtime_cache_path_ = RuntimeCachePath(cache_dir);
    }
    runtime_cache_.reset(runtime_config_->createRuntimeCache());
    if (!runtime_cache_) {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA TensorRT-RTX failed to create runtime cache for %s",
                 bytecode_.function_name.c_str());
      runtime_cache_path_.clear();
      return {};
    }

    std::vector<uint8_t> serialized_cache;
    if (ReadBinaryFile(runtime_cache_path_, &serialized_cache) &&
        !serialized_cache.empty()) {
      if (runtime_cache_->deserialize(serialized_cache.data(),
                                      serialized_cache.size())) {
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT-RTX loaded runtime cache for %s "
                   "(bytes=%zu path=%s)",
                   bytecode_.function_name.c_str(), serialized_cache.size(),
                   runtime_cache_path_.c_str());
      } else {
        LITERT_LOG(LITERT_WARNING,
                   "NVIDIA TensorRT-RTX ignored invalid runtime cache for %s "
                   "(path=%s)",
                   bytecode_.function_name.c_str(),
                   runtime_cache_path_.c_str());
      }
    }

    if (!runtime_config_->setRuntimeCache(*runtime_cache_)) {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA TensorRT-RTX failed to enable runtime cache for %s",
                 bytecode_.function_name.c_str());
      runtime_cache_.reset();
      runtime_cache_path_.clear();
      return {};
    }

    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX runtime cache enabled for %s (path=%s)",
               bytecode_.function_name.c_str(), runtime_cache_path_.c_str());
    return {};
  }

  void SaveRuntimeCache() {
    if (!runtime_cache_ || !runtime_config_ || runtime_cache_path_.empty()) {
      return;
    }
    TrtPtr<nvinfer1::IHostMemory> serialized_cache(runtime_cache_->serialize());
    if (!serialized_cache || serialized_cache->data() == nullptr ||
        serialized_cache->size() == 0) {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA TensorRT-RTX runtime cache for %s serialized empty",
                 bytecode_.function_name.c_str());
      return;
    }
    auto saved = litert::nvidia::PersistValidatedRuntimeCache(
        *runtime_config_, serialized_cache->data(), serialized_cache->size(),
        runtime_cache_path_);
    if (saved) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX saved runtime cache for %s "
                 "(bytes=%zu path=%s)",
                 bytecode_.function_name.c_str(), serialized_cache->size(),
                 runtime_cache_path_.c_str());
    } else {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA TensorRT-RTX failed to save runtime cache for %s "
                 "(path=%s): %s",
                 bytecode_.function_name.c_str(), runtime_cache_path_.c_str(),
                 saved.Error().Message().c_str());
    }
  }

  // Debug helper: writes the deserialized engine's detailed layer/tensor
  // information as JSON to
  // $LITERT_NVIDIA_DISPATCH_ENGINE_INFO_DIR/<function>.json so post-fusion
  // kernels seen by profilers can be mapped back to shapes.
  void DumpEngineInfo() const {
    const char* dir = std::getenv("LITERT_NVIDIA_DISPATCH_ENGINE_INFO_DIR");
    if (dir == nullptr || dir[0] == '\0' || engine_ == nullptr) {
      return;
    }
    TrtPtr<nvinfer1::IEngineInspector> inspector(
        engine_->createEngineInspector());
    const char* info = inspector ? inspector->getEngineInformation(
                                       nvinfer1::LayerInformationFormat::kJSON)
                                 : nullptr;
    if (info == nullptr) {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA dispatch engine info unavailable for %s",
                 bytecode_.function_name.c_str());
      return;
    }
    const std::string path = JoinPath(
        dir, SanitizeCacheComponent(bytecode_.function_name) + ".json");
    if (WriteBinaryFile(path, info, std::strlen(info))) {
      LITERT_LOG(LITERT_INFO, "NVIDIA dispatch wrote engine info for %s to %s",
                 bytecode_.function_name.c_str(), path.c_str());
    }
  }

  static bool DispatchDumpIoEnabled() {
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_DUMP_IO");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }

  // LITERT_NVIDIA_DISPATCH_DUMP_IO=top<K>[:<file>] also records the K largest
  // values of every row of every large float output (the logits) with their
  // indices: a numerics probe of one step against the same step on another
  // build or backend. Without <file> the rows go to the log; with it, one JSON
  // line per output and step is appended to <file>:
  //   {"context":C,"function":"decode","step":N,"output":"logits","rows":R,
  //    "columns":V,"top":[[[index,value],...],...]}
  // where C numbers the invocation contexts of the process (two models, the
  // base and its drafter, share function names).
  static int DispatchDumpTopCount() {
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_DUMP_IO");
    if (value == nullptr || std::strncmp(value, "top", 3) != 0) return 0;
    return std::max(0, std::atoi(value + 3));
  }

  static const char* DispatchDumpTopFile() {
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_DUMP_IO");
    if (value == nullptr || std::strncmp(value, "top", 3) != 0) return nullptr;
    const char* colon = std::strchr(value + 3, ':');
    return colon == nullptr || colon[1] == '\0' ? nullptr : colon + 1;
  }

  // LITERT_NVIDIA_DISPATCH_DUMP_IO=dir:<path> also writes every input and
  // output of every step to <path>/<function>/step<N>/<input|output>_<i>.raw,
  // the step's data for a run of the same signature elsewhere (run_model).
  static const char* DispatchDumpDir() {
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_DUMP_IO");
    if (value == nullptr || std::strncmp(value, "dir:", 4) != 0) return nullptr;
    return value + 4;
  }

  // LITERT_NVIDIA_DISPATCH_DUMP_OUTPUTS=<dir>[:<max_steps>[:<max_mb>]] (any
  // LITERT_NVIDIA_DISPATCH_DUMP_IO mode may run beside it) writes every
  // output of the first <max_steps> steps of every context (default 8) that
  // is at most <max_mb> MB (default 16), as
  // <dir>/c<context>/<function>/step<N>/out_<i>.raw, and one
  // JSON line per file to <dir>/index.jsonl, in the order the outputs were
  // produced: {"seq":S,"context":C,"function":F,"step":N,"output":i,
  // "name":...,"element_type":T,"dims":[...],"stored_half":H,"state":K,
  // "input_dims":[...],"bytes":B,"skipped":"","file":"c<C>/<F>/step<N>/out_<i>.raw"}
  // (an output above <max_mb> keeps its line with "skipped":"size" and no
  // file, so a reader knows it existed);
  // "state" marks an output whose device memory is an input's (the in-place
  // KV caches, declared as aliases by the engine or not), dumped at every
  // step of a context that emits nothing else (a prefill) and at the first
  // step of any other; "input_dims" are the step's token input's (the first
  // integer input of rank at most 2: a prefill chunk's count, a decode
  // step's one, a verify step's few), which tell the signatures apart. A
  // numerics guard compares two runs' files in that order and names the first
  // tensor that differs.
  static const char* DumpOutputsSpec() {
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_DUMP_OUTPUTS");
    return value == nullptr || value[0] == '\0' ? nullptr : value;
  }

  static std::string DumpOutputsDir() {
    const char* spec = DumpOutputsSpec();
    if (spec == nullptr) return "";
    const char* colon = std::strchr(spec, ':');
    return colon == nullptr ? std::string(spec) : std::string(spec, colon);
  }

  static int DumpOutputsMaxSteps() {
    const char* spec = DumpOutputsSpec();
    const char* colon = spec == nullptr ? nullptr : std::strchr(spec, ':');
    if (colon == nullptr) return 8;
    const int steps = std::atoi(colon + 1);
    return steps > 0 ? steps : 8;
  }

  static size_t DumpOutputsMaxBytes() {
    const char* spec = DumpOutputsSpec();
    const char* colon = spec == nullptr ? nullptr : std::strchr(spec, ':');
    const char* second =
        colon == nullptr ? nullptr : std::strchr(colon + 1, ':');
    if (second == nullptr) return size_t{16} << 20;
    const long mb = std::atol(second + 1);
    return (mb > 0 ? static_cast<size_t>(mb) : size_t{16}) << 20;
  }

  static uint64_t NextDumpOutputsSeq() {
    static std::atomic<uint64_t> next{0};
    return next.fetch_add(1);
  }

  // The dims of the step's token input (the first integer input of rank at
  // most 2: the tokens or the positions, "[1024]" for a prefill chunk, "[1]"
  // for a decode step, "[4]" for a verify step, as Gemma's signatures
  // declare them), as a JSON array, or "[]" when there is none.
  std::string DumpInputDims() {
    if (runtime_context_ == nullptr ||
        runtime_context_->get_tensor_buffer_tensor_type == nullptr) {
      return "[]";
    }
    for (const auto handle : input_handles_) {
      if (handle == 0) continue;
      auto record = device_context_->GetRecord(handle);
      if (!record.HasValue() || (*record)->tensor_buffer == nullptr) continue;
      LiteRtRankedTensorType type{};
      if (runtime_context_->get_tensor_buffer_tensor_type(
              (*record)->tensor_buffer, &type) != kLiteRtStatusOk) {
        continue;
      }
      const bool integer = type.element_type == kLiteRtElementTypeInt32 ||
                           type.element_type == kLiteRtElementTypeInt64;
      if (!integer || type.layout.rank > 2) continue;
      std::string dims = "[";
      for (uint32_t d = 0; d < type.layout.rank; ++d) {
        dims += (d == 0 ? "" : ",") + std::to_string(type.layout.dimensions[d]);
      }
      return dims + "]";
    }
    return "[]";
  }

  void DumpOutputToDir(int index, const std::string& name,
                       LiteRtTensorBuffer tensor_buffer, void* device_ptr,
                       size_t size, bool state, const std::string& input_dims) {
    const std::string dir = DumpOutputsDir();
    if (dir.empty() || device_ptr == nullptr || size == 0 ||
        dump_outputs_step_ > DumpOutputsMaxSteps()) {
      return;
    }
    // Above the cap: an index line without a file, so a reader knows the
    // output existed and was not dumped (the caches of a long context).
    const bool skipped = size > DumpOutputsMaxBytes();
    LiteRtRankedTensorType type{};
    bool have_type = false;
    if (tensor_buffer != nullptr && runtime_context_ != nullptr &&
        runtime_context_->get_tensor_buffer_tensor_type != nullptr) {
      have_type = runtime_context_->get_tensor_buffer_tensor_type(
                      tensor_buffer, &type) == kLiteRtStatusOk;
    }
    const std::string function =
        SanitizeCacheComponent(bytecode_.function_name);
    const std::string relative = "c" + std::to_string(dump_context_id_) + "/" +
                                 function + "/step" +
                                 std::to_string(dump_outputs_step_) + "/out_" +
                                 std::to_string(index) + ".raw";
    const std::string step_dir =
        JoinPath(JoinPath(JoinPath(dir, "c" + std::to_string(dump_context_id_)),
                          function),
                 "step" + std::to_string(dump_outputs_step_));
    (void)std::system(("mkdir -p '" + step_dir + "'").c_str());
    if (!skipped) {
      std::vector<unsigned char> bytes(size);
      cudaStreamSynchronize(stream_);
      if (cudaMemcpy(bytes.data(), device_ptr, size, cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        (void)cudaGetLastError();
        return;
      }
      if (!WriteBinaryFile(JoinPath(dir, relative), bytes.data(),
                           bytes.size())) {
        LITERT_LOG(LITERT_WARNING, "NVIDIA dispatch could not write %s",
                   relative.c_str());
        return;
      }
    }
    std::string dims;
    if (have_type) {
      for (uint32_t d = 0; d < type.layout.rank; ++d) {
        dims += (d == 0 ? "" : ",") + std::to_string(type.layout.dimensions[d]);
      }
    }
    bool stored_half = false;
    if (have_type) {
      const auto half = StoredAsHalf(name, type);
      stored_half = half.HasValue() && *half;
    }
    const std::string line =
        "{\"seq\":" + std::to_string(NextDumpOutputsSeq()) +
        ",\"context\":" + std::to_string(dump_context_id_) +
        ",\"function\":\"" + function +
        "\",\"step\":" + std::to_string(dump_outputs_step_) +
        ",\"output\":" + std::to_string(index) + ",\"name\":\"" +
        SanitizeCacheComponent(name) + "\",\"element_type\":" +
        std::to_string(have_type ? static_cast<int>(type.element_type) : -1) +
        ",\"dims\":[" + dims +
        "],\"stored_half\":" + (stored_half ? "true" : "false") +
        ",\"state\":" + (state ? "true" : "false") +
        ",\"input_dims\":" + input_dims + ",\"bytes\":" + std::to_string(size) +
        ",\"skipped\":" + (skipped ? "\"size\"" : "\"\"") + ",\"file\":\"" +
        (skipped ? std::string() : relative) + "\"}\n";
    static std::mutex index_mutex;
    std::lock_guard<std::mutex> lock(index_mutex);
    FILE* out = std::fopen(JoinPath(dir, "index.jsonl").c_str(), "a");
    if (out != nullptr) {
      std::fputs(line.c_str(), out);
      std::fclose(out);
    }
  }

  // The step counter of this context for the outputs dump (every Invoke).
  int dump_outputs_step_ = 0;

  void DumpDeviceBufferToFile(const char* kind, int index, void* device_ptr,
                              size_t size) {
    const char* dir = DispatchDumpDir();
    if (dir == nullptr || device_ptr == nullptr || size == 0) return;
    const std::string step_dir =
        JoinPath(JoinPath(dir, SanitizeCacheComponent(bytecode_.function_name)),
                 "step" + std::to_string(dump_step_));
    (void)std::system(("mkdir -p '" + step_dir + "'").c_str());
    std::vector<unsigned char> bytes(size);
    cudaStreamSynchronize(stream_);
    if (cudaMemcpy(bytes.data(), device_ptr, size, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }
    const std::string path = JoinPath(
        step_dir, std::string(kind) + "_" + std::to_string(index) + ".raw");
    if (!WriteBinaryFile(path, bytes.data(), bytes.size())) {
      LITERT_LOG(LITERT_WARNING, "NVIDIA dispatch could not write %s",
                 path.c_str());
    }
  }

  // The step counter of this context (each Invoke) and the context's number
  // in the process, both for the dumps above.
  int dump_step_ = 0;
  const int dump_context_id_ = NextDumpContextId();
  static int NextDumpContextId() {
    static std::atomic<int> next{0};
    return next.fetch_add(1);
  }

  void DumpDeviceBufferTop(const std::string& name,
                           LiteRtTensorBuffer tensor_buffer, void* device_ptr,
                           size_t size) {
    const int top = DispatchDumpTopCount();
    if (top <= 0 || device_ptr == nullptr || tensor_buffer == nullptr ||
        runtime_context_ == nullptr ||
        runtime_context_->get_tensor_buffer_tensor_type == nullptr) {
      return;
    }
    LiteRtRankedTensorType type;
    if (runtime_context_->get_tensor_buffer_tensor_type(tensor_buffer, &type) !=
            kLiteRtStatusOk ||
        (type.element_type != kLiteRtElementTypeFloat32 &&
         type.element_type != kLiteRtElementTypeFloat16) ||
        type.layout.rank == 0) {
      return;
    }
    const size_t columns =
        static_cast<size_t>(type.layout.dimensions[type.layout.rank - 1]);
    // Float16 logits (the 12B's) hold two bytes per value, as does a float32
    // tensor the engine keeps as FP16.
    const auto half = StoredAsHalf(name, type);
    const size_t element_bytes =
        type.element_type == kLiteRtElementTypeFloat16 ||
                (half.HasValue() && *half)
            ? 2
            : 4;
    const size_t count = size / element_bytes;
    if (dump_step_ == 1) {
      // An inventory of the outputs, once, for whoever wonders why a tensor
      // is not among the dumps.
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA dispatch dump inventory context=%d %s output=%s "
                 "type=%d rank=%u columns=%zu bytes=%zu element_bytes=%zu",
                 dump_context_id_, bytecode_.function_name.c_str(),
                 name.c_str(), static_cast<int>(type.element_type),
                 type.layout.rank, columns, size, element_bytes);
    }
    // Logits only: a vocabulary's width, not an activation's.
    if (columns < 65536 || count < columns || count % columns != 0) return;
    const size_t rows = count / columns;
    const size_t k = std::min(static_cast<size_t>(top), columns);
    std::vector<float> values(count);
    cudaStreamSynchronize(stream_);
    if (element_bytes == 2) {
      std::vector<uint16_t> bits(count);
      if (cudaMemcpy(bits.data(), device_ptr, count * sizeof(uint16_t),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError();
        return;
      }
      for (size_t i = 0; i < count; ++i) values[i] = HalfBitsToFloat(bits[i]);
    } else if (cudaMemcpy(values.data(), device_ptr, count * sizeof(float),
                          cudaMemcpyDeviceToHost) != cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }
    std::vector<int> order(columns);
    std::string json;
    for (size_t row = 0; row < rows; ++row) {
      const float* row_values = values.data() + row * columns;
      for (size_t i = 0; i < columns; ++i) order[i] = static_cast<int>(i);
      // Ties broken by index, as the CPU sampler's top-k orders them.
      std::partial_sort(order.begin(), order.begin() + k, order.end(),
                        [&](int a, int b) {
                          return row_values[a] > row_values[b] ||
                                 (row_values[a] == row_values[b] && a < b);
                        });
      std::string text;
      for (size_t i = 0; i < k; ++i) {
        char item[48];
        std::snprintf(item, sizeof(item), i == 0 ? "[%d,%.7g]" : ",[%d,%.7g]",
                      order[i], row_values[order[i]]);
        text += item;
      }
      if (DispatchDumpTopFile() == nullptr) {
        LITERT_LOG(
            LITERT_INFO,
            "NVIDIA dispatch top%zu context=%d %s step=%d %s row=%zu/%zu "
            "n=%zu: %s",
            k, dump_context_id_, bytecode_.function_name.c_str(), dump_step_,
            name.c_str(), row, rows, columns, text.c_str());
      } else {
        json += row == 0 ? "[" : ",[";
        json += text;
        json += "]";
      }
    }
    const char* file = DispatchDumpTopFile();
    if (file == nullptr) return;
    static std::mutex file_mutex;
    std::lock_guard<std::mutex> lock(file_mutex);
    FILE* out = std::fopen(file, "a");
    if (out == nullptr) {
      LITERT_LOG(LITERT_WARNING, "NVIDIA dispatch could not append to %s",
                 file);
      return;
    }
    std::fprintf(out,
                 "{\"context\":%d,\"function\":\"%s\",\"step\":%d,"
                 "\"output\":\"%s\",\"rows\":%zu,\"columns\":%zu,"
                 "\"top\":[%s]}\n",
                 dump_context_id_, bytecode_.function_name.c_str(), dump_step_,
                 name.c_str(), rows, columns, json.c_str());
    std::fclose(out);
  }

  // Debug helper: logs the first bytes of a device buffer as floats and a
  // cheap checksum, synchronizing the stream first.
  void DumpDeviceBufferPrefix(const char* tag, const std::string& name,
                              void* device_ptr, size_t size) {
    float values[4] = {0, 0, 0, 0};
    const size_t copy_bytes = std::min(sizeof(values), size);
    cudaStreamSynchronize(stream_);
    if (cudaMemcpy(values, device_ptr, copy_bytes, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }
    unsigned char sample[64] = {};
    const size_t sample_bytes = std::min(sizeof(sample), size - size / 2);
    cudaMemcpy(sample, static_cast<const char*>(device_ptr) + size / 2,
               sample_bytes, cudaMemcpyDeviceToHost);
    // A failed debug copy must not leave its error for the next launch's
    // cudaGetLastError (the plugins report it as their own failure).
    (void)cudaGetLastError();
    unsigned int checksum = 0;
    for (size_t i = 0; i < sample_bytes; ++i) {
      checksum = checksum * 31 + sample[i];
    }
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch io %s %s=%s f32[%g %g %g %g] midsum=%u", tag,
               bytecode_.function_name.c_str(), name.c_str(), values[0],
               values[1], values[2], values[3], checksum);
  }

  Expected<void> BindSharedActivationArena() {
    if (device_memory_bytes_ <= 0) {
      // Nothing to bind: the engine has no activation memory (or the context
      // owns its own allocation).
      return {};
    }
    LITERT_ASSIGN_OR_RETURN(auto arena,
                            device_context_->EnsureSharedActivationArena(
                                static_cast<size_t>(device_memory_bytes_)));
    if (arena.first != bound_arena_ptr_ ||
        arena.second != bound_arena_version_) {
      execution_context_->setDeviceMemoryV2(
          arena.first,
          static_cast<int64_t>(device_context_->shared_arena_size()));
      bound_arena_ptr_ = arena.first;
      bound_arena_version_ = arena.second;
    }
    return {};
  }

  // Whether the engine keeps a float32 boundary tensor as FP16 (a KV cache
  // the compiler declared half: Fp16CacheStorageEnabled), from the probed
  // metadata or the engine.
  Expected<bool> StoredAsHalf(const std::string& tensor_name,
                              const LiteRtRankedTensorType& tensor_type) const {
    if (tensor_type.element_type != kLiteRtElementTypeFloat32) {
      return false;
    }
    if (aot_residency_) {
      const auto type = tensor_types_.find(tensor_name);
      if (type == tensor_types_.end()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Missing NVIDIA AOT tensor type metadata");
      }
      return type->second == nvinfer1::DataType::kHALF;
    }
    if (engine_ == nullptr) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "TensorRT engine is not initialized");
    }
    return engine_->getTensorDataType(tensor_name.c_str()) ==
           nvinfer1::DataType::kHALF;
  }

  Expected<bool> PrefersCudaTensorBuffer(const std::string& tensor_name) const {
    if (aot_residency_) {
      const auto location = tensor_locations_.find(tensor_name);
      if (location == tensor_locations_.end()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Missing NVIDIA AOT tensor location metadata");
      }
      return location->second == nvinfer1::TensorLocation::kDEVICE;
    }
    if (engine_ == nullptr) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "TensorRT engine is not initialized");
    }
    const nvinfer1::TensorLocation location =
        engine_->getTensorLocation(tensor_name.c_str());
    if (DispatchProfilingEnabled()) {
      LITERT_LOG(
          LITERT_INFO, "NVIDIA dispatch tensor location name=%s location=%s",
          tensor_name.c_str(),
          location == nvinfer1::TensorLocation::kHOST ? "HOST" : "DEVICE");
    }
    return location == nvinfer1::TensorLocation::kDEVICE;
  }

  Expected<bool> BindTensorAddressIfNeeded(const std::string& tensor_name,
                                           void* device_ptr, void** bound_ptr) {
    if (bound_ptr == nullptr) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Null TensorRT binding cache");
    }
    if (*bound_ptr == device_ptr) {
      return false;
    }
    if (!execution_context_->setTensorAddress(tensor_name.c_str(),
                                              device_ptr)) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to bind TensorRT tensor");
    }
    *bound_ptr = device_ptr;
    return true;
  }

  Expected<LockedHostBuffer> GetReadableHostPointer(
      LiteRtTensorBuffer buffer) const {
    return GetHostPointer(buffer, kLiteRtTensorBufferLockModeRead);
  }

  Expected<LockedHostBuffer> GetWritableHostPointer(
      LiteRtTensorBuffer buffer) const {
    return GetHostPointer(buffer, kLiteRtTensorBufferLockModeWrite);
  }

  Expected<LockedHostBuffer> GetHostPointer(
      LiteRtTensorBuffer buffer, LiteRtTensorBufferLockMode mode) const {
    void* host = nullptr;
    LiteRtStatus status =
        runtime_context_->get_tensor_buffer_host_memory(buffer, &host);
    if (status == kLiteRtStatusOk && host != nullptr) {
      return LockedHostBuffer{buffer, host, false};
    }
    LITERT_RETURN_IF_ERROR(
        runtime_context_->lock_tensor_buffer(buffer, &host, mode));
    return LockedHostBuffer{buffer, host, true};
  }

  const LiteRtRuntimeContext* runtime_context_;
  LiteRtDispatchDeviceContext device_context_;
  std::shared_ptr<AotEngineResidency> aot_residency_;
  std::optional<litert::nvidia::TensorRtAotLocator> aot_locator_;
  std::unordered_map<std::string, nvinfer1::TensorLocation> tensor_locations_;
  std::unordered_map<std::string, nvinfer1::DataType> tensor_types_;
  bool resident_has_run_ = false;
  litert::nvidia::TensorRtBytecode bytecode_;
  litert::nvidia::MemoryProfiler memory_profiler_;
  litert::nvidia::TensorRtLogger logger_;
  // The device memory of the engine's weight store outlives the engine, and
  // its mapping into the engine goes before the engine.
  std::unique_ptr<litert::nvidia::EngineWeightMemory> weight_memory_;
  TrtPtr<nvinfer1::IRuntime> runtime_;
  TrtPtr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<litert::nvidia::EngineWeightMapping> weight_mapping_;
  TrtPtr<nvinfer1::IRuntimeCache> runtime_cache_;
  TrtPtr<nvinfer1::IRuntimeConfig> runtime_config_;
  TrtPtr<nvinfer1::IExecutionContext> execution_context_;
  std::unique_ptr<litert::nvidia::TensorRtLayerProfiler> layer_profiler_;
  litert::nvidia::DispatchInvocationProfiler invocation_profiler_;
  cudaStream_t stream_ = nullptr;
  std::vector<cudaStream_t> aux_streams_;
  bool own_graph_enabled_ = false;
  cudaStreamCaptureMode own_graph_mode_ = cudaStreamCaptureModeThreadLocal;
  std::vector<OwnGraph> own_graphs_;
  int64_t own_graph_captures_ = 0;
  int64_t own_graph_direct_ = 0;
  bool autopsy_done_ = false;
  bool timeline_done_ = false;
  std::vector<LiteRtTensorBufferHandle> input_handles_;
  std::vector<LiteRtTensorBufferHandle> output_handles_;
  std::vector<void*> bound_input_ptrs_;
  std::vector<void*> bound_output_ptrs_;
  std::vector<int> aliased_input_of_output_;
  void* head_weights_ = nullptr;
  void* head_scales_ = nullptr;
  void* head_input_bf16_ = nullptr;
  void* head_output_bf16_ = nullptr;
  bool has_scheduling_info_ = false;
  LiteRtSchedulingInfo scheduling_info_{};
  // Shared activation arena bookkeeping; negative size means the context owns
  // its activation memory (kSTATIC allocation strategy).
  int64_t device_memory_bytes_ = -1;
  void* bound_arena_ptr_ = nullptr;
  uint64_t bound_arena_version_ = 0;
  uint64_t invocation_count_ = 0;
  std::string runtime_cache_path_;

  // The outputs copied to their host caches by the step on the stream.
  std::vector<void*> prefetched_;
};

namespace {

LiteRtEnvironment static_environment = nullptr;
LiteRtOptions static_options = nullptr;
const LiteRtRuntimeContext* static_runtime_context = nullptr;
char build_id[128];

LiteRtStatus Initialize(const LiteRtRuntimeContext* runtime_context,
                        LiteRtEnvironment environment, LiteRtOptions options) {
  static_environment = environment;
  static_options = options;
  static_runtime_context = runtime_context;
  LiteRtEnvironmentOptions environment_options = nullptr;
  if (runtime_context != nullptr &&
      runtime_context->get_environment_options(
          environment, &environment_options) == kLiteRtStatusOk) {
    LiteRtPropagateMinLoggerSeverityWithRuntimeContext(runtime_context,
                                                       environment_options);
  }
  snprintf(build_id, sizeof(build_id),
           "NVIDIA TensorRT-RTX via NvInfer headers %d.%d.%d",
           NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
  return kLiteRtStatusOk;
}

LiteRtStatus GetVendorId(const char** vendor_id) {
  if (vendor_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *vendor_id = "NVIDIA";
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** out_build_id) {
  if (out_build_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *out_build_id = build_id;
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  if (capabilities == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextCreate(const LiteRtRuntimeContext* runtime_context,
                                 LiteRtOptions options,
                                 LiteRtDispatchDeviceContext* device_context) {
  if (runtime_context == nullptr || device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *device_context = new LiteRtDispatchDeviceContextT(runtime_context);
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextDestroy(LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus GetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (invocation_context == nullptr || tensor_type == nullptr ||
      tensor_buffer_requirements == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto requirements =
      invocation_context->GetInputRequirements(input_index, *tensor_type);
  if (!requirements) {
    LITERT_LOG(LITERT_ERROR,
               "NVIDIA dispatch input requirements failed: index=%d %s: %s",
               input_index, DescribeTensorType(*tensor_type).c_str(),
               requirements.Error().Message().c_str());
    return requirements.Error().Status();
  }
  *tensor_buffer_requirements = *requirements;
  return kLiteRtStatusOk;
}

LiteRtStatus GetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (invocation_context == nullptr || tensor_type == nullptr ||
      tensor_buffer_requirements == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto requirements =
      invocation_context->GetOutputRequirements(output_index, *tensor_type);
  if (!requirements) {
    LITERT_LOG(LITERT_ERROR,
               "NVIDIA dispatch output requirements failed: index=%d %s: %s",
               output_index, DescribeTensorType(*tensor_type).c_str(),
               requirements.Error().Message().c_str());
    return requirements.Error().Status();
  }
  *tensor_buffer_requirements = *requirements;
  return kLiteRtStatusOk;
}

LiteRtStatus RegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context, LiteRtTensorBuffer buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  if (device_context == nullptr || tensor_buffer_handle == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto handle = device_context->RegisterTensorBuffer(buffer);
  if (!handle) {
    LITERT_LOG(LITERT_ERROR, "NVIDIA dispatch buffer registration failed: %s",
               handle.Error().Message().c_str());
    return handle.Error().Status();
  }
  *tensor_buffer_handle = *handle;
  return kLiteRtStatusOk;
}

LiteRtStatus UnregisterTensorBuffer(LiteRtDispatchDeviceContext device_context,
                                    LiteRtTensorBufferHandle handle) {
  if (device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(device_context->UnregisterTensorBuffer(handle));
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextCreate(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto context = LiteRtDispatchInvocationContextT::Create(
      runtime_context, device_context, exec_type, exec_bytecode_buffer,
      function_name, num_inputs, num_outputs);
  if (!context) {
    LITERT_LOG(LITERT_ERROR,
               "NVIDIA dispatch invocation context creation failed for %s "
               "(inputs=%d outputs=%d bytecode_bytes=%zu): %s",
               function_name == nullptr ? "<null>" : function_name, num_inputs,
               num_outputs,
               exec_bytecode_buffer == nullptr ? 0 : exec_bytecode_buffer->size,
               context.Error().Message().c_str());
    return context.Error().Status();
  }
  *invocation_context = context->release();
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextSetSchedulingInfo(
    LiteRtDispatchInvocationContext invocation_context,
    const LiteRtSchedulingInfo* scheduling_info) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      invocation_context->SetSchedulingInfo(scheduling_info));
  return kLiteRtStatusOk;
}

LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto attached =
      invocation_context->AttachInput(graph_input_index, tensor_buffer_handle);
  if (!attached) {
    // The delegate reports only the status; the reason is logged here.
    LITERT_LOG(LITERT_ERROR, "NVIDIA dispatch failed to attach input %d: %s",
               graph_input_index, attached.Error().Message().c_str());
    return attached.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto attached = invocation_context->AttachOutput(graph_output_index,
                                                   tensor_buffer_handle);
  if (!attached) {
    LITERT_LOG(LITERT_ERROR, "NVIDIA dispatch failed to attach output %d: %s",
               graph_output_index, attached.Error().Message().c_str());
    return attached.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(
      invocation_context->DetachInput(graph_input_index, tensor_buffer_handle));
  return kLiteRtStatusOk;
}

LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(invocation_context->DetachOutput(
      graph_output_index, tensor_buffer_handle));
  return kLiteRtStatusOk;
}

LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto status = invocation_context->Invoke();
  if (!status) {
    LITERT_LOG(LITERT_ERROR, "NVIDIA dispatch invoke failed: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                       LiteRtEnvironmentOptions env,
                                       LiteRtOptions options) {
  static constexpr LiteRtApiVersion kApiVersion{LITERT_API_VERSION_MAJOR,
                                                LITERT_API_VERSION_MINOR,
                                                LITERT_API_VERSION_PATCH};
  if (LiteRtCompareApiVersion(api_version, kApiVersion) > 0) {
    return kLiteRtStatusErrorUnsupportedCompilerVersion;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextSetOptions(
    LiteRtDispatchInvocationContext invocation_context, LiteRtOptions options) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(invocation_context->SetOptions(options));
  return kLiteRtStatusOk;
}

static const LiteRtDispatchInterface_V1 NvidiaInterface = {
    /*.abi_header=*/{
        .struct_size = sizeof(LiteRtDispatchInterface_V1),
        .major_version = 1,
        .minor_version = 0,
        .reserved = 0,
    },
    /*.initialize=*/Initialize,
    /*.get_vendor_id=*/GetVendorId,
    /*.get_build_id=*/GetBuildId,
    /*.get_capabilities=*/GetCapabilities,
    /*.device_context_create=*/DeviceContextCreate,
    /*.device_context_destroy=*/DeviceContextDestroy,
    /*.get_input_requirements=*/GetInputRequirements,
    /*.get_output_requirements=*/GetOutputRequirements,
    /*.register_tensor_buffer=*/RegisterTensorBuffer,
    /*.unregister_tensor_buffer=*/UnregisterTensorBuffer,
    /*.invocation_context_create=*/InvocationContextCreate,
    /*.invocation_context_destroy=*/InvocationContextDestroy,
    /*.invocation_context_set_scheduling_info=*/
    InvocationContextSetSchedulingInfo,
    /*.attach_input=*/AttachInput,
    /*.attach_output=*/AttachOutput,
    /*.detach_input=*/DetachInput,
    /*.detach_output=*/DetachOutput,
    /*.invoke=*/Invoke,
    /*.start_metrics_collection=*/nullptr,
    /*.stop_metrics_collection=*/nullptr,
    /*.get_num_metrics=*/nullptr,
    /*.get_metric=*/nullptr,
    /*.destroy_metrics=*/nullptr,
    /*.check_runtime_compatibility=*/CheckRuntimeCompatibility,
    /*.invocation_context_set_options=*/InvocationContextSetOptions,
};

static const LiteRtCustomTensorBufferHandlersDef_V1 NvidiaTensorBufferHandlers =
    {
        .abi_header =
            {
                .struct_size = sizeof(LiteRtCustomTensorBufferHandlersDef_V1),
                .major_version = 1,
                .minor_version = 0,
                .reserved = 0,
            },
        .create_func = CreateCudaTensorBuffer,
        .destroy_func = DestroyCudaTensorBuffer,
        .lock_func = LockCudaTensorBuffer,
        .unlock_func = UnlockCudaTensorBuffer,
        .clear_func = ClearCudaTensorBuffer,
        .import_func = ImportCudaTensorBuffer,
        .device_tag = kLiteRtEnvOptionTagNull,
        .queue_tag = kLiteRtEnvOptionTagNull,
        .num_supported_buffer_types = 1,
        .supported_buffer_types = {kNvidiaCudaTensorBufferType},
    };

}  // namespace

extern "C" LiteRtStatus LiteRtDispatchNvidiaGreedySamplerCreate(
    LiteRtDispatchNvidiaGreedySampler* sampler) {
  if (sampler == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sampler = nullptr;
  auto* context = new (std::nothrow) NvidiaGreedySamplerContext;
  if (context == nullptr) {
    return kLiteRtStatusErrorMemoryAllocationFailure;
  }
  if (static_runtime_context == nullptr) {
    delete context;
    return kLiteRtStatusErrorUnsupported;
  }
  context->runtime_context = static_runtime_context;
  *sampler = context;
  return kLiteRtStatusOk;
}

extern "C" void LiteRtDispatchNvidiaGreedySamplerDestroy(
    LiteRtDispatchNvidiaGreedySampler sampler) {
  auto* context = static_cast<NvidiaGreedySamplerContext*>(sampler);
  DestroyNvidiaGreedySamplerResources(context);
  delete context;
}

extern "C" LiteRtStatus LiteRtDispatchNvidiaGreedySamplerSampleF32(
    LiteRtDispatchNvidiaGreedySampler sampler, LiteRtTensorBuffer logits,
    size_t count, int32_t* token_id) {
  return SampleNvidiaGreedy(static_cast<NvidiaGreedySamplerContext*>(sampler),
                            logits, 1, count, token_id,
                            /*legacy_f32_only=*/true);
}

extern "C" LiteRtStatus LiteRtDispatchNvidiaGreedySamplerSampleBatched(
    LiteRtDispatchNvidiaGreedySampler sampler, LiteRtTensorBuffer logits,
    size_t rows, size_t vocab_size, int32_t* token_ids) {
  return SampleNvidiaGreedy(static_cast<NvidiaGreedySamplerContext*>(sampler),
                            logits, rows, vocab_size, token_ids,
                            /*legacy_f32_only=*/false);
}

extern "C" int32_t LiteRtDispatchNvidiaTopKCandidatesMaxK(void) {
  return LiteRtNvidiaTopKCandidatesMaxK();
}

extern "C" LiteRtStatus LiteRtDispatchNvidiaTopKCandidatesCreate(
    LiteRtDispatchNvidiaTopKCandidates* selector) {
  if (selector == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *selector = nullptr;
  auto* context = new (std::nothrow) NvidiaTopKCandidatesContext;
  if (context == nullptr) {
    return kLiteRtStatusErrorMemoryAllocationFailure;
  }
  if (static_runtime_context == nullptr) {
    delete context;
    return kLiteRtStatusErrorUnsupported;
  }
  context->runtime_context = static_runtime_context;
  *selector = context;
  return kLiteRtStatusOk;
}

extern "C" void LiteRtDispatchNvidiaTopKCandidatesDestroy(
    LiteRtDispatchNvidiaTopKCandidates selector) {
  auto* context = static_cast<NvidiaTopKCandidatesContext*>(selector);
  DestroyNvidiaTopKCandidatesResources(context);
  delete context;
}

extern "C" LiteRtStatus LiteRtDispatchNvidiaTopKCandidatesSelect(
    LiteRtDispatchNvidiaTopKCandidates selector, LiteRtTensorBuffer logits,
    size_t rows, size_t vocab_size, int32_t k, int32_t* ids, float* values,
    int32_t* row_flags) {
  return SelectNvidiaTopKCandidates(
      static_cast<NvidiaTopKCandidatesContext*>(selector), logits, rows,
      vocab_size, k, ids, values, row_flags);
}

extern "C" LiteRtStatus LiteRtDispatchNvidiaStepSamplerSet(int32_t k, float p,
                                                           float temperature) {
  if (k <= 0 || !(p >= 0.0f && p <= 1.0f) || !(temperature >= 0.0f)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& state = GetStepSamplerState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.told = true;
  state.k = k;
  state.p = p;
  state.temperature = temperature;
  state.count = 0;
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtDispatchNvidiaStepSamplerPushUniforms(
    const double* uniforms, int32_t count) {
  if (count < 0 || count > 8 || (count > 0 && uniforms == nullptr)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  for (int32_t i = 0; i < count; ++i) {
    if (!(uniforms[i] >= 0.0 && uniforms[i] < 1.0)) {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }
  auto& state = GetStepSamplerState();
  std::lock_guard<std::mutex> lock(state.mutex);
  for (int32_t i = 0; i < count; ++i) state.uniforms[i] = uniforms[i];
  state.count = count;
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtDispatchNvidiaStepFeedSetLocalWindow(
    uint32_t window) {
  StepFeedLocalWindow().store(window);
  return kLiteRtStatusOk;
}

extern "C" LiteRtStatus LiteRtDispatchNvidiaRegisterTokenTable(
    const LiteRtDispatchNvidiaTokenTable* table) {
  if (table == nullptr || (table->table == nullptr && table->table_fd < 0) ||
      table->scales == nullptr || table->vocab == 0 || table->dim == 0 ||
      table->dim % 2 != 0 ||
      table->table_bytes !=
          static_cast<size_t>(table->vocab) * table->dim / 2 ||
      !std::isfinite(table->multiplier) || table->multiplier == 0.0f) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  size_t scales_bytes = 0;
  if (table->scale_block == 0) {
    scales_bytes = static_cast<size_t>(table->vocab) * sizeof(float);
  } else if (table->dim % table->scale_block == 0) {
    scales_bytes = static_cast<size_t>(table->vocab) *
                   (table->dim / table->scale_block) * sizeof(uint16_t);
  }
  if (scales_bytes == 0 || table->scales_bytes != scales_bytes) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (static_runtime_context == nullptr) {
    return kLiteRtStatusErrorUnsupported;
  }
  auto& registry = GetTokenTableRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const std::pair<uint32_t, uint32_t> key(table->vocab, table->dim);
  if (std::find(registry.awaited.begin(), registry.awaited.end(), key) ==
      registry.awaited.end()) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA dispatch: no decode step awaits a token table [%u, %u]",
               table->vocab, table->dim);
    return kLiteRtStatusErrorNotFound;
  }
  for (const auto& registered : registry.tables) {
    if (registered.vocab == table->vocab && registered.dim == table->dim) {
      return kLiteRtStatusOk;  // the first registration stands
    }
  }
  // The table's bytes: given, or read from the file.
  const void* table_bytes = table->table;
  std::vector<uint8_t> read_table;
  if (table_bytes == nullptr) {
    read_table.resize(table->table_bytes);
    for (size_t done = 0; done < read_table.size();) {
      const ssize_t result = pread(
          table->table_fd, read_table.data() + done, read_table.size() - done,
          static_cast<off_t>(table->table_offset + done));
      if (result < 0 && errno == EINTR) continue;
      if (result <= 0) {
        LITERT_LOG(LITERT_WARNING,
                   "NVIDIA dispatch: the token table [%u, %u] cannot be read "
                   "(%zu bytes at %llu): %s",
                   table->vocab, table->dim, read_table.size(),
                   static_cast<unsigned long long>(table->table_offset),
                   result < 0 ? std::strerror(errno) : "end of file");
        return kLiteRtStatusErrorRuntimeFailure;
      }
      done += static_cast<size_t>(result);
    }
    table_bytes = read_table.data();
  }
  RegisteredTokenTable registered;
  registered.vocab = table->vocab;
  registered.dim = table->dim;
  registered.scale_block = table->scale_block;
  registered.multiplier = table->multiplier;
  cudaError_t status = cudaMalloc(&registered.table, table->table_bytes);
  if (status == cudaSuccess) {
    status = cudaMemcpy(registered.table, table_bytes, table->table_bytes,
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    status = cudaMalloc(&registered.scales, scales_bytes);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(registered.scales, table->scales, scales_bytes,
                        cudaMemcpyHostToDevice);
  }
  if (status != cudaSuccess) {
    LITERT_LOG(
        LITERT_WARNING,
        "NVIDIA dispatch: the token table [%u, %u] is not registered: %s",
        table->vocab, table->dim, cudaGetErrorString(status));
    if (registered.scales != nullptr) cudaFree(registered.scales);
    if (registered.table != nullptr) cudaFree(registered.table);
    return status == cudaErrorMemoryAllocation
               ? kLiteRtStatusErrorMemoryAllocationFailure
               : kLiteRtStatusErrorRuntimeFailure;
  }
  registry.tables.push_back(registered);
  LITERT_LOG(LITERT_INFO,
             "NVIDIA dispatch: token table [%u, %u] registered (%zu bytes, "
             "%s scales, multiplier %g) for the step feed",
             table->vocab, table->dim, table->table_bytes + scales_bytes,
             table->scale_block == 0 ? "row" : "block", table->multiplier);
  return kLiteRtStatusOk;
}

extern "C" LITERT_CAPI_EXPORT LiteRtStatus LiteRtDispatchQueryInterface(
    LiteRtDispatchInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface) {
  if (out_interface == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (litert_runtime_version.major >= 1) {
    if (interface_id == kLiteRtInterfaceBasic) {
      *out_interface = &NvidiaInterface;
      return kLiteRtStatusOk;
    } else if (interface_id == kLiteRtInterfaceCustomTensorBufferHandlers) {
      *out_interface = &NvidiaTensorBufferHandlers;
      return kLiteRtStatusOk;
    }
  }
  return kLiteRtStatusErrorUnsupported;
}
