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

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cuda_runtime_api.h"
#include "driver_types.h"
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
#include "litert/vendors/nvidia/compiler/subbyte_gemv_plugin.h"
#include "litert/vendors/nvidia/dispatch/aot_artifact.h"
#include "litert/vendors/nvidia/dispatch/dispatch_profiler.h"
#include "litert/vendors/nvidia/dispatch/greedy_sampler_c_api.h"
#include "litert/vendors/nvidia/dispatch/greedy_sampler_kernel.h"
#include "litert/vendors/nvidia/dispatch/tensor_buffer_view.h"
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
using ::litert::nvidia::MappedAotArtifact;

template <typename T>
using TrtPtr = std::unique_ptr<T>;

constexpr LiteRtTensorBufferType kNvidiaCudaTensorBufferType =
    static_cast<LiteRtTensorBufferType>(
        kLiteRtTensorBufferTypeUserCustomBuffer + 1);

Expected<void> CudaOk(cudaError_t error, const char* what) {
  if (error == cudaSuccess) {
    return {};
  }
  return Error(kLiteRtStatusErrorRuntimeFailure,
               std::string(what) + ": " + cudaGetErrorString(error));
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
  size_t bytes = 0;
  size_t packed_bytes = 0;
  void* host_cache = nullptr;
  bool host_cache_pinned = false;
  LiteRtTensorBufferLockMode lock_mode = kLiteRtTensorBufferLockModeRead;
  bool locked = false;
  bool owns_handle = true;
};

struct NvidiaGreedySamplerContext {
  const LiteRtRuntimeContext* runtime_context = nullptr;
  int device = -1;
  cudaStream_t stream = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  int32_t* device_result = nullptr;
  int32_t* host_result = nullptr;
  uint64_t calls = 0;
};

CudaTensorBufferInfo* AsCudaTensorBufferInfo(HwMemoryInfoPtr info) {
  return static_cast<CudaTensorBufferInfo*>(info);
}

size_t NonZeroBytes(size_t bytes) { return std::max<size_t>(bytes, 1); }

size_t CopyBytes(const CudaTensorBufferInfo& info) {
  return info.packed_bytes == 0 ? info.bytes : info.packed_bytes;
}

bool PinnedHostStagingEnabled() {
  const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_PINNED_HOST_STAGING");
  return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

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
  void* device_ptr = nullptr;
  const LiteRtStatus status = CudaStatus(
      cudaMalloc(&device_ptr, NonZeroBytes(std::max(bytes, packed_bytes))),
      "cudaMalloc CUDA tensor buffer");
  if (status != kLiteRtStatusOk) {
    delete info;
    return status;
  }
  info->memory_handle = device_ptr;
  info->raw_handle = device_ptr;
  info->bytes = bytes;
  info->packed_bytes = packed_bytes;
  info->owns_handle = true;
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
  *hw_memory_info = info;
  return kLiteRtStatusOk;
}

LiteRtStatus DestroyCudaTensorBuffer(HwMemoryInfoPtr hw_memory_info) {
  auto* info = AsCudaTensorBufferInfo(hw_memory_info);
  if (info == nullptr) {
    return kLiteRtStatusOk;
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
  if (info->host_cache == nullptr) {
    const size_t alloc_bytes = NonZeroBytes(CopyBytes(*info));
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
  }
  const size_t copy_bytes = CopyBytes(*info);
  if (copy_bytes > 0 && (mode == kLiteRtTensorBufferLockModeRead ||
                         mode == kLiteRtTensorBufferLockModeReadWrite)) {
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
    status = CudaStatus(cudaMemcpy(info->memory_handle, info->host_cache,
                                   copy_bytes, cudaMemcpyHostToDevice),
                        "cudaMemcpy CUDA tensor buffer H2D unlock");
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
  if (context->calls > 0) {
    LITERT_LOG(LITERT_INFO, "NVIDIA CUDA greedy sampler completed %llu calls",
               static_cast<unsigned long long>(context->calls));
  }
}

LiteRtStatus EnsureNvidiaGreedySamplerResources(
    NvidiaGreedySamplerContext* context, int device, size_t count) {
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
    cuda_status =
        cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(int32_t));
    if (cuda_status != cudaSuccess) {
      clean_up_partial_initialization();
      return CudaStatus(cuda_status, "cudaMalloc CUDA greedy sampler result");
    }
    cuda_status =
        cudaMallocHost(reinterpret_cast<void**>(&host_result), sizeof(int32_t));
    if (cuda_status != cudaSuccess) {
      clean_up_partial_initialization();
      return CudaStatus(cuda_status,
                        "cudaMallocHost CUDA greedy sampler result");
    }
    context->device = device;
    context->stream = stream;
    context->device_result = device_result;
    context->host_result = host_result;
    LITERT_LOG(LITERT_INFO,
               "NVIDIA CUDA greedy sampler initialized on device %d", device);
  }
  const size_t required = LiteRtNvidiaF32ArgMaxWorkspaceBytes(count);
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

LiteRtStatus SampleNvidiaGreedyF32(NvidiaGreedySamplerContext* context,
                                   LiteRtTensorBuffer logits, size_t count,
                                   int32_t* token_id) {
  if (context == nullptr || logits == nullptr || token_id == nullptr ||
      count == 0 || context->runtime_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const LiteRtRuntimeContext* runtime_context = context->runtime_context;
  if (runtime_context->get_tensor_buffer_type == nullptr ||
      runtime_context->get_tensor_buffer_tensor_type == nullptr ||
      runtime_context->get_tensor_buffer_packed_size == nullptr ||
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
  if (tensor_type.element_type != kLiteRtElementTypeFloat32 ||
      tensor_type.layout.has_strides || tensor_type.layout.rank != 3 ||
      tensor_type.layout.dimensions[0] != 1 ||
      tensor_type.layout.dimensions[1] != 1 ||
      tensor_type.layout.dimensions[2] != static_cast<int32_t>(count)) {
    return kLiteRtStatusErrorUnsupported;
  }
  size_t packed_size = 0;
  size_t offset = 0;
  LITERT_RETURN_IF_ERROR(
      runtime_context->get_tensor_buffer_packed_size(logits, &packed_size));
  LITERT_RETURN_IF_ERROR(
      runtime_context->get_tensor_buffer_offset(logits, &offset));
  if (packed_size != count * sizeof(float)) {
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
  auto* device_logits = reinterpret_cast<const float*>(
      static_cast<const uint8_t*>(handle) + offset);
  cudaPointerAttributes attributes{};
  const cudaError_t attributes_status =
      cudaPointerGetAttributes(&attributes, device_logits);
  if (attributes_status != cudaSuccess || attributes.devicePointer == nullptr ||
      (attributes.type != cudaMemoryTypeDevice &&
       attributes.type != cudaMemoryTypeManaged)) {
    cudaGetLastError();
    return kLiteRtStatusErrorUnsupported;
  }

  int previous_device = -1;
  LITERT_RETURN_IF_ERROR(
      CudaStatus(cudaGetDevice(&previous_device), "cudaGetDevice sampler"));
  const bool restore_device = previous_device != attributes.device;
  if (restore_device) {
    LITERT_RETURN_IF_ERROR(
        CudaStatus(cudaSetDevice(attributes.device), "cudaSetDevice sampler"));
  }
  LiteRtStatus status =
      EnsureNvidiaGreedySamplerResources(context, attributes.device, count);
  if (status == kLiteRtStatusOk) {
    status = CudaStatus(
        LiteRtNvidiaLaunchF32ArgMax(device_logits, count, context->workspace,
                                    context->workspace_bytes,
                                    context->device_result, context->stream),
        "LiteRtNvidiaLaunchF32ArgMax");
  }
  if (status == kLiteRtStatusOk) {
    status =
        CudaStatus(cudaMemcpyAsync(context->host_result, context->device_result,
                                   sizeof(int32_t), cudaMemcpyDeviceToHost,
                                   context->stream),
                   "cudaMemcpyAsync CUDA greedy sampler result");
  }
  if (status == kLiteRtStatusOk) {
    status = CudaStatus(cudaStreamSynchronize(context->stream),
                        "cudaStreamSynchronize CUDA greedy sampler");
  }
  if (status == kLiteRtStatusOk) {
    *token_id = *context->host_result;
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

Expected<LiteRtTensorBufferRequirements> TensorBufferRequirements(
    const LiteRtRuntimeContext* runtime_context,
    const LiteRtRankedTensorType& tensor_type, bool prefer_cuda) {
  litert::RankedTensorType ranked_type(tensor_type);
  if (ranked_type.Layout().HasStrides()) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "NVIDIA dispatch does not support strided tensors");
  }
  LITERT_ASSIGN_OR_RETURN(size_t size, ranked_type.Bytes());
  if (PreferHostBoundaryIo()) {
    prefer_cuda = false;
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
      num_buffer_types, buffer_types, size,
      /*num_strides=*/0, /*strides=*/nullptr, &requirements));
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

}  // namespace

class LiteRtDispatchDeviceContextT {
 public:
  explicit LiteRtDispatchDeviceContextT(
      const LiteRtRuntimeContext* runtime_context)
      : runtime_context_(runtime_context) {}

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
    Record record;
    record.tensor_buffer = tensor_buffer;
    record.size = packed_size;
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
                handle, buffer_size, buffer_offset, packed_size));
        record.owns_device_ptr = false;
        record.direct_cuda_buffer = true;
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
    bool host_tensor_buffer = false;
    bool transient_device_ptr = false;
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

  void ReleaseTransientDevicePtr(Record* record) {
    if (record == nullptr || !record->transient_device_ptr ||
        record->device_ptr == nullptr) {
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

 private:
  const LiteRtRuntimeContext* runtime_context_;
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
    LITERT_RETURN_IF_ERROR(context->Initialize());
    if (aot_artifact) {
      // deserializeCudaEngine(), refit, runtime-cache setup, and the external
      // head's H2D copies are complete. Keep only owned TensorRT/CUDA state;
      // no invocation-time operation reads the serialized artifact views.
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
    DestroyExternalHeadResources();
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
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
    return TensorBufferRequirements(runtime_context_, tensor_type, prefer_cuda);
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
    return TensorBufferRequirements(runtime_context_, tensor_type, prefer_cuda);
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
    (void)record;
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
    (void)record;
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
      if (bytecode_.output_names[i].empty()) {
        // The external W2 vocabulary head writes this LiteRT output after the
        // TensorRT-RTX prefix, so it has no TensorRT engine binding.
        bound_output_ptrs_[i] = record->device_ptr;
        ++profile_metrics.set_address_skips;
        continue;
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
    if (DispatchDumpIoEnabled()) {
      for (int i = 0; i < static_cast<int>(input_handles_.size()); ++i) {
        LITERT_ASSIGN_OR_RETURN(auto* record,
                                device_context_->GetRecord(input_handles_[i]));
        DumpDeviceBufferPrefix("in", bytecode_.input_names[i],
                               record->device_ptr, record->size);
      }
    }
    DispatchCpuTimer enqueue_call_timer(profile);
    if (!execution_context_->enqueueV3(stream_)) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "TensorRT enqueueV3 failed");
    }
    LITERT_RETURN_IF_ERROR(LaunchExternalHead());
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

    DispatchCpuTimer sync_timer(profile);
    const char* sync_context =
        bytecode_.trtllm_head.has_value()
            ? "cudaStreamSynchronize TensorRT-RTX prefix and external W2 head"
            : "cudaStreamSynchronize";
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaStreamSynchronize(stream_), sync_context));
    profile_metrics.cpu_sync_ms = sync_timer.ElapsedMs();
    if (memory_profile) {
      memory_profiler_.Log("invoke_synchronized",
                           bytecode_.function_name.c_str());
    }
    if (DispatchDumpIoEnabled()) {
      for (int i = 0; i < static_cast<int>(output_handles_.size()); ++i) {
        LITERT_ASSIGN_OR_RETURN(auto* record,
                                device_context_->GetRecord(output_handles_[i]));
        DumpDeviceBufferPrefix("out", bytecode_.output_names[i],
                               record->device_ptr, record->size);
      }
    }
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
    if (bytecode_.trtllm_head.has_value()) {
      bytecode_.trtllm_head->packed_weights = nullptr;
      bytecode_.trtllm_head->packed_weights_size = 0;
      bytecode_.trtllm_head->bf16_scales = nullptr;
      bytecode_.trtllm_head->bf16_scales_size = 0;
    }
  }

  Expected<void> Initialize() {
    memory_profiler_.Log("context_initialize_begin",
                         bytecode_.function_name.c_str());
    litert::nvidia::EnsureSubbyteGemvPluginRegistered();
    LITERT_RETURN_IF_ERROR(
        CudaOk(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags"));
    memory_profiler_.Log("stream_created", bytecode_.function_name.c_str());
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
    LITERT_RETURN_IF_ERROR(RefitEngine());
    if (UseCudaGraph()) {
      // TensorRT-RTX owns whole-graph CUDA-graph capture/replay through the
      // runtime config; this collapses per-kernel launch overhead, which
      // dominates decode-step time on WSL.
      runtime_config_.reset(engine_->createRuntimeConfig());
      if (runtime_config_) {
        if (UseSharedActivationArena()) {
          runtime_config_->setExecutionContextAllocationStrategy(
              nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
        }
        runtime_config_->setCudaGraphStrategy(
            nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
        LITERT_RETURN_IF_ERROR(ConfigureRuntimeCache());
        execution_context_.reset(
            engine_->createExecutionContext(runtime_config_.get()));
        if (execution_context_) {
          if (UseSharedActivationArena()) {
            device_memory_bytes_ = engine_->getDeviceMemorySizeV2();
          }
          LITERT_LOG(LITERT_INFO,
                     "NVIDIA dispatch context %s uses CUDA graph capture "
                     "(device_memory_bytes=%lld)",
                     bytecode_.function_name.c_str(),
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
    if (litert::nvidia::DispatchLayerProfilingEnabled()) {
      layer_profiler_ =
          std::make_unique<litert::nvidia::TensorRtLayerProfiler>();
      execution_context_->setProfiler(layer_profiler_.get());
      LITERT_LOG(LITERT_INFO, "NVIDIA layer profiling enabled for %s",
                 bytecode_.function_name.c_str());
    }
    LITERT_RETURN_IF_ERROR(InitializeExternalHead());
    memory_profiler_.Log("context_initialize_end",
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
      return std::strcmp(value, "0") != 0;
    }
    return true;
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
    runtime_cache_path_ = RuntimeCachePath(cache_dir);
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
    if (!runtime_cache_ || runtime_cache_path_.empty()) {
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
    if (WriteBinaryFile(runtime_cache_path_, serialized_cache->data(),
                        serialized_cache->size())) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX saved runtime cache for %s "
                 "(bytes=%zu path=%s)",
                 bytecode_.function_name.c_str(), serialized_cache->size(),
                 runtime_cache_path_.c_str());
    } else {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA TensorRT-RTX failed to save runtime cache for %s "
                 "(path=%s)",
                 bytecode_.function_name.c_str(), runtime_cache_path_.c_str());
    }
  }

  static bool DispatchDumpIoEnabled() {
    const char* value = std::getenv("LITERT_NVIDIA_DISPATCH_DUMP_IO");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
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
      return;
    }
    unsigned char sample[64] = {};
    const size_t sample_bytes = std::min(sizeof(sample), size);
    cudaMemcpy(sample, static_cast<const char*>(device_ptr) + size / 2,
               sample_bytes, cudaMemcpyDeviceToHost);
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

  Expected<bool> PrefersCudaTensorBuffer(const std::string& tensor_name) const {
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
  litert::nvidia::TensorRtBytecode bytecode_;
  litert::nvidia::MemoryProfiler memory_profiler_;
  litert::nvidia::TensorRtLogger logger_;
  TrtPtr<nvinfer1::IRuntime> runtime_;
  TrtPtr<nvinfer1::ICudaEngine> engine_;
  TrtPtr<nvinfer1::IRuntimeCache> runtime_cache_;
  TrtPtr<nvinfer1::IRuntimeConfig> runtime_config_;
  TrtPtr<nvinfer1::IExecutionContext> execution_context_;
  std::unique_ptr<litert::nvidia::TensorRtLayerProfiler> layer_profiler_;
  litert::nvidia::DispatchInvocationProfiler invocation_profiler_;
  cudaStream_t stream_ = nullptr;
  std::vector<LiteRtTensorBufferHandle> input_handles_;
  std::vector<LiteRtTensorBufferHandle> output_handles_;
  std::vector<void*> bound_input_ptrs_;
  std::vector<void*> bound_output_ptrs_;
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
  LITERT_RETURN_IF_ERROR(
      invocation_context->AttachInput(graph_input_index, tensor_buffer_handle));
  return kLiteRtStatusOk;
}

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LITERT_RETURN_IF_ERROR(invocation_context->AttachOutput(
      graph_output_index, tensor_buffer_handle));
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

LiteRtDispatchInterface NvidiaInterface = {
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

LiteRtCustomTensorBufferHandlersDef NvidiaTensorBufferHandlers = {
    /*.abi_header=*/
    {
        /*.struct_size=*/sizeof(LiteRtCustomTensorBufferHandlersDef),
        /*.major_version=*/1,
        /*.minor_version=*/0,
        /*.reserved=*/0,
    },
    /*.create_func=*/CreateCudaTensorBuffer,
    /*.destroy_func=*/DestroyCudaTensorBuffer,
    /*.lock_func=*/LockCudaTensorBuffer,
    /*.unlock_func=*/UnlockCudaTensorBuffer,
    /*.clear_func=*/ClearCudaTensorBuffer,
    /*.import_func=*/ImportCudaTensorBuffer,
    /*.device_tag=*/kLiteRtEnvOptionTagNull,
    /*.queue_tag=*/kLiteRtEnvOptionTagNull,
    /*.num_supported_buffer_types=*/1,
    /*.supported_buffer_types=*/{kNvidiaCudaTensorBufferType},
};

LiteRtDispatchApi NvidiaApi = {
    /*.abi_header=*/
    {
        /*.struct_size=*/sizeof(LiteRtDispatchApi),
        /*.major_version=*/1,
        /*.minor_version=*/0,
        /*.reserved=*/0,
    },
    /*.version=*/
    {/*.major=*/LITERT_API_VERSION_MAJOR,
     /*.minor=*/LITERT_API_VERSION_MINOR,
     /*.patch=*/LITERT_API_VERSION_PATCH},
    /*.interface=*/&NvidiaInterface,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
    /*.tensor_buffer_handlers_def=*/&NvidiaTensorBufferHandlers,
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
  return SampleNvidiaGreedyF32(
      static_cast<NvidiaGreedySamplerContext*>(sampler), logits, count,
      token_id);
}

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  if (api == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *api = NvidiaApi;
  return kLiteRtStatusOk;
}
