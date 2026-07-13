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

#include "third_party/odml/litert/ml_drift/delegate/buffer_handler_opencl.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/cl/cl_command_queue.h"  // from @ml_drift
#include "ml_drift/cl/cl_context.h"  // from @ml_drift
#include "ml_drift/cl/cl_memory.h"  // from @ml_drift
#if LITERT_HAS_OPENGL_SUPPORT
#include "ml_drift/cl/gl_interop.h"  // from @ml_drift
#endif
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/access_type.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_tensor_buffer_utils.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer_lockstate.h"
#include "third_party/odml/litert/ml_drift/delegate/buffer_handler_utils.h"
#include <CL/cl.h>
#include "tflite/delegates/gpu/cl/opencl_wrapper.h" // NOLINT: Required for OpenCL backend.

using ::litert::internal::LockState;

namespace {

// A `HwMemoryInfo` implementation for OpenCL Custom Buffer integration.
struct OpenClMemoryInfo : public HwMemoryInfo {
  // MLD Tensor owns an OpenCL memory.
  ::ml_drift::cl::Tensor cl_tensor;
  bool owns_tensor;
  cl_mem imported_cl_mem =
      nullptr;  // For tracking ownership of imported buffers.
  LiteRtRankedTensorType tensor_type;
  LiteRtTensorBufferType buffer_type;
  size_t packed_bytes;
  void* host_memory;
  LockState lock_state;
  cl_context context;
  cl_command_queue command_queue;
};

}  // namespace

LiteRtStatus LiteRtDestroyOpenClMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = static_cast<OpenClMemoryInfo*>(hw_memory_info);
  if (memory_info->host_memory != nullptr) {
    if (memory_info->buffer_type == kLiteRtTensorBufferTypeOpenClBufferPacked) {
      ::ml_drift::cl::clEnqueueUnmapMemObject(
          memory_info->command_queue, memory_info->cl_tensor.GetMemoryPtr(),
          memory_info->host_memory, 0, nullptr, nullptr);
      memory_info->host_memory = nullptr;
    } else {
      litert_aligned_free(memory_info->host_memory);
    }
  }
  if (memory_info->imported_cl_mem) {
    ml_drift::cl::clReleaseMemObject(memory_info->imported_cl_mem);
  }
  delete memory_info;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtLockOpenClMemory(HwMemoryInfoPtr hw_memory_info,
                                    LiteRtTensorBufferLockMode mode,
                                    void** host_memory_ptr) {
  if (hw_memory_info == nullptr || host_memory_ptr == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = reinterpret_cast<OpenClMemoryInfo*>(hw_memory_info);

  LITERT_RETURN_IF_ERROR(
      memory_info->lock_state == LockState::kUnlocked,
      litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
          << "The OpenCL memory is already locked.");

  LockState new_lock_state = litert::internal::ToLockState(mode);
  auto& gpu_tensor = memory_info->cl_tensor;

  if (memory_info->buffer_type == kLiteRtTensorBufferTypeOpenClBufferPacked) {
    if (memory_info->host_memory == nullptr) {
      // Fallback in case it wasn't mapped at creation
      cl_int error_code = CL_SUCCESS;
      void* mapped_ptr = ::ml_drift::cl::clEnqueueMapBuffer(
          memory_info->command_queue, gpu_tensor.GetMemoryPtr(), CL_TRUE,
          CL_MAP_READ | CL_MAP_WRITE, 0, memory_info->packed_bytes, 0, nullptr,
          nullptr, &error_code);
      if (error_code != CL_SUCCESS) {
        ABSL_LOG(ERROR) << "clEnqueueMapBuffer failed with: " << error_code;
        return kLiteRtStatusErrorRuntimeFailure;
      }
      memory_info->host_memory = mapped_ptr;
    }
    *host_memory_ptr = memory_info->host_memory;
    memory_info->lock_state = new_lock_state;
    return kLiteRtStatusOk;
  }

  if (memory_info->host_memory == nullptr) {
    // Ensure the data is aligned.
    if (auto rc = posix_memalign(&memory_info->host_memory,
                                 LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                 memory_info->packed_bytes);
        rc) {
      ABSL_LOG(ERROR) << "Failed to allocate aligned memory";
      return kLiteRtStatusErrorRuntimeFailure;
    }
  }
  *host_memory_ptr = memory_info->host_memory;

  // Don't download data if write only.
  if (new_lock_state == LockState::kWriteLocked) {
    memory_info->lock_state = new_lock_state;
    return kLiteRtStatusOk;
  }

  ::ml_drift::cl::CLCommandQueue queue(memory_info->command_queue,
                                       /*has_ownership=*/false);

  auto descriptor_with_data = gpu_tensor.GetDescriptor();
  LITERT_RETURN_IF_ERROR(
      gpu_tensor.ToDescriptor(&descriptor_with_data, &queue));

  ::litert::ml_drift::ConvertDataFromDescriptor(
      descriptor_with_data, memory_info->host_memory,
      memory_info->tensor_type.element_type);

  memory_info->lock_state = new_lock_state;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtUnlockOpenClMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto* memory_info = static_cast<OpenClMemoryInfo*>(hw_memory_info);

  LITERT_RETURN_IF_ERROR(
      memory_info->lock_state != LockState::kUnlocked,
      litert::ErrorStatusBuilder(kLiteRtStatusErrorRuntimeFailure)
          << "The OpenCL memory is already unlocked.");
  absl::Cleanup unlock = [&memory_info] {
    memory_info->lock_state = LockState::kUnlocked;
  };

  auto& gpu_tensor = memory_info->cl_tensor;

  if (memory_info->buffer_type == kLiteRtTensorBufferTypeOpenClBufferPacked) {
    ::ml_drift::cl::clEnqueueUnmapMemObject(
        memory_info->command_queue, gpu_tensor.GetMemoryPtr(),
        memory_info->host_memory, 0, nullptr, nullptr);
    memory_info->host_memory = nullptr;
    return kLiteRtStatusOk;
  }

  // Don't upload data if read only.
  if (memory_info->lock_state == LockState::kReadLocked) {
    return kLiteRtStatusOk;
  }

  ::ml_drift::cl::CLCommandQueue queue(memory_info->command_queue,
                                       /*has_ownership=*/false);

  ::ml_drift::cl::CLContext context(memory_info->context,
                                    /*has_ownership=*/false);

  auto descriptor_with_data = gpu_tensor.GetDescriptor();
  ::litert::ml_drift::ConvertDataToDescriptor(
      memory_info->host_memory, descriptor_with_data,
      memory_info->tensor_type.element_type);
  auto status = gpu_tensor.UploadDescriptorData(descriptor_with_data, &queue);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to upload data to tensor: " << status;
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtClearOpenClMemory(HwMemoryInfoPtr hw_memory_info) {
  if (hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // Not implemented natively by OpenCL 1.2 commands. We will enqueue write /
  // fill instead. Note: clEnqueueFillBuffer is available in OpenCL 1.2
  // clEnqueueFillImage is available in OpenCL 1.2
  // For ML Drift tensor which does not expose that directly, we'll write a
  // zero'd block.
  auto* memory_info = reinterpret_cast<OpenClMemoryInfo*>(hw_memory_info);
  auto& gpu_tensor = memory_info->cl_tensor;
  ::ml_drift::cl::CLCommandQueue queue(memory_info->command_queue,
                                       /*has_ownership=*/false);

  if (memory_info->buffer_type == kLiteRtTensorBufferTypeOpenClBufferPacked) {
    std::vector<uint8_t> zeros(memory_info->packed_bytes, 0);
    LITERT_RETURN_IF_ERROR(queue.EnqueueWriteBuffer(
        gpu_tensor.GetMemoryPtr(), memory_info->packed_bytes, zeros.data(),
        /*async=*/false));
  } else {
    // Write via ML drift tensor API
    // Need a descriptor for uploading.
    auto desc_with_data = gpu_tensor.GetDescriptor();
    std::vector<uint8_t> zeros(desc_with_data.GetMemorySizeInBytes(), 0);
    desc_with_data.SetData(std::move(zeros));
    LITERT_RETURN_IF_ERROR(
        gpu_tensor.UploadDescriptorData(desc_with_data, &queue));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCreateOpenClMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* opencl_memory_info) {
  int fp16_scale = IsGpuFloat16Memory(buffer_type) ? 2 : 1;
  auto tensor_desc =
      ::litert::ml_drift::CreateTensorDescriptor(*tensor_type, buffer_type);
  if (!tensor_desc.ok()) {
    ABSL_LOG(ERROR) << "Failed to create tensor descriptor: "
                    << tensor_desc.status();
    return kLiteRtStatusErrorUnsupported;
  } else if (tensor_desc->GetMemorySizeInBytes() * fp16_scale < packed_bytes) {
    ABSL_LOG(ERROR) << "Too big memory requested: max_size="
                    << tensor_desc->GetMemorySizeInBytes()
                    << " vs requested=" << packed_bytes;
    return kLiteRtStatusErrorInvalidArgument;
  }

  if (device_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  cl_context cl_ctx_handle = static_cast<cl_context>(device_id);
  ::ml_drift::cl::CLContext context(cl_ctx_handle, /*has_ownership=*/false);

  auto memory_info = std::make_unique<OpenClMemoryInfo>(OpenClMemoryInfo{
      .cl_tensor = {},
      .owns_tensor = true,
      .tensor_type = *tensor_type,
      .buffer_type = buffer_type,
      .packed_bytes = packed_bytes,
      .host_memory = nullptr,
      .lock_state = LockState::kUnlocked,
      .context = cl_ctx_handle,
      .command_queue = static_cast<cl_command_queue>(queue_id)});

  LITERT_RETURN_IF_ERROR(::ml_drift::cl::CreateTensor(context, *tensor_desc,
                                                      &memory_info->cl_tensor));

  // Returns `Tensor*` as the `memory_handle`.
  memory_info->memory_handle = &memory_info->cl_tensor;
  memory_info->raw_handle = memory_info->cl_tensor.GetMemoryPtrForWriting();

  *opencl_memory_info = memory_info.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtImportOpenClMemory(LiteRtGpuDeviceId device_id,
                                      LiteRtGpuQueueId queue_id,
                                      const LiteRtRankedTensorType* tensor_type,
                                      LiteRtTensorBufferType buffer_type,
                                      HwMemoryHandle hw_buffer_handle,
                                      size_t bytes, size_t packed_bytes,
                                      HwMemoryInfoPtr* hw_memory_info) {
  if (hw_buffer_handle == nullptr || hw_memory_info == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto tensor_desc =
      ::litert::ml_drift::CreateTensorDescriptor(*tensor_type, buffer_type);
  if (!tensor_desc.ok()) {
    ABSL_LOG(ERROR) << "Failed to create tensor descriptor for import: "
                    << tensor_desc.status();
    return kLiteRtStatusErrorUnsupported;
  }

  if (device_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  cl_context cl_ctx_handle = static_cast<cl_context>(device_id);
  ::ml_drift::cl::CLContext context(cl_ctx_handle, /*has_ownership=*/false);

  auto memory_info = std::make_unique<OpenClMemoryInfo>(OpenClMemoryInfo{
      .cl_tensor = {},
      .owns_tensor = false,
      .tensor_type = *tensor_type,
      .buffer_type = buffer_type,
      .packed_bytes = packed_bytes,
      .host_memory = nullptr,
      .lock_state = LockState::kUnlocked,
      .context = cl_ctx_handle,
      .command_queue = static_cast<cl_command_queue>(queue_id)});

  // TODO(487801415): Add support for AHWB and OpenGL memory.
  cl_mem cl_memory = nullptr;

  // Try to infer the source buffer type.
  cl_mem candidate_cl_mem = static_cast<cl_mem>(hw_buffer_handle);
  cl_mem_object_type cl_mem_type;
  if (ml_drift::cl::clGetMemObjectInfo(candidate_cl_mem, CL_MEM_TYPE,
                                       sizeof(cl_mem_type), &cl_mem_type,
                                       nullptr) == CL_SUCCESS) {
    cl_memory = candidate_cl_mem;
  }
#if LITERT_HAS_AHWB_SUPPORT
  else if (tflite::gpu::cl::clImportMemoryARM != nullptr &&
           hw_buffer_handle != nullptr) {
    const cl_import_properties_arm properties[] = {
        CL_IMPORT_TYPE_ARM,
        CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM,
        0,
    };

    cl_int error = CL_SUCCESS;
    cl_mem candidate_cl_mem = tflite::gpu::cl::clImportMemoryARM(
        cl_ctx_handle, CL_MEM_READ_WRITE, properties, hw_buffer_handle, bytes,
        &error);
    if (error == CL_SUCCESS) {
      cl_memory = candidate_cl_mem;
    }
  }
#endif
  else {
#if LITERT_HAS_OPENGL_SUPPORT
    // Assume it is an OpenGL buffer ID.
    cl_GLuint gl_buffer_id =
        static_cast<cl_GLuint>(reinterpret_cast<uintptr_t>(hw_buffer_handle));
    ::ml_drift::cl::CLMemory cl_mem_obj;
    auto status = ::ml_drift::cl::CreateClMemoryFromGlBuffer(
        gl_buffer_id, ::ml_drift::AccessType::READ_WRITE, &context,
        &cl_mem_obj);
    if (status.ok()) {
      cl_memory = cl_mem_obj.Release();
      memory_info->imported_cl_mem = cl_memory;
    }
#else
    ABSL_LOG(ERROR)
        << "OpenGL support is disabled, cannot import OpenGL buffer.";
    return kLiteRtStatusErrorUnsupported;
#endif
  }
  // Re-use `buffer_handler_webgpu.cc`'s logic for parsing
  // `kLiteRtTensorBufferTypeWebGpuBuffer` /
  // `kLiteRtTensorBufferTypeWebGpuTexture`
  if (buffer_type == kLiteRtTensorBufferTypeOpenClBuffer ||
      buffer_type == kLiteRtTensorBufferTypeOpenClBufferFp16 ||
      buffer_type == kLiteRtTensorBufferTypeOpenClBufferPacked) {
    LITERT_RETURN_IF_ERROR(::ml_drift::cl::CreateTensorShared(
        context, cl_memory, *tensor_desc, &memory_info->cl_tensor));
  } else if (buffer_type == kLiteRtTensorBufferTypeOpenClTexture ||
             buffer_type == kLiteRtTensorBufferTypeOpenClTextureFp16 ||
             buffer_type == kLiteRtTensorBufferTypeOpenClImageBuffer ||
             buffer_type == kLiteRtTensorBufferTypeOpenClImageBufferFp16) {
    // For Image buffer, use `CreateTensorSharedImage2DBuffer`
    // However, it is not general enough, so we re-use `CreateTensorShared`.
    LITERT_RETURN_IF_ERROR(::ml_drift::cl::CreateTensorShared(
        context, cl_memory, *tensor_desc, &memory_info->cl_tensor));
  } else {
    ABSL_LOG(ERROR) << "Unsupported buffer type for import: "
                    << litert::BufferTypeToString(buffer_type);
    return kLiteRtStatusErrorUnsupported;
  }

  // Returns `Tensor*` as the `memory_handle`.
  memory_info->memory_handle = &memory_info->cl_tensor;
  memory_info->raw_handle = memory_info->cl_tensor.GetMemoryPtrForWriting();
  *hw_memory_info = memory_info.release();
  return kLiteRtStatusOk;
}
