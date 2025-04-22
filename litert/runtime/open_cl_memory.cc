// Copyright 2024 Google LLC.
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

#include "litert/runtime/open_cl_memory.h"

#include <stdlib.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/ahwb_buffer.h"
#include "litert/runtime/gpu_environment.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>
#include "tflite/delegates/gpu/cl/buffer.h"
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_context.h"
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"

// GPU memory access functions defined in GPU Accelerator.
extern "C" bool (*LiteRtGpuMemoryUpload)(
    void* cl_mem, size_t bytes, const void* ptr,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type) = nullptr;
extern "C" bool (*LiteRtGpuMemoryDownload)(
    void* cl_mem, size_t bytes, void* ptr,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type) = nullptr;


namespace litert {
namespace internal {

template Expected<float*> OpenClMemory::Lock<float>(LiteRtLockMode mode);
template Expected<char*> OpenClMemory::Lock<char>(LiteRtLockMode mode);
template Expected<void> OpenClMemory::Unlock<float>();
template Expected<void> OpenClMemory::Unlock<char>();

template <typename T>
Expected<T*> OpenClMemory::Lock(LiteRtLockMode mode) {
  absl::MutexLock lock(&mutex_);
  lock_mode_ = mode;
  // The buffer has not been locked, so we need to read from the OpenCL
  // buffer.
  if (data_ == nullptr) {
    cpu_buffer_size_ =
        (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferFp16 ||
         buffer_type_ == kLiteRtTensorBufferTypeOpenClImageBufferFp16)
            ? size_ * 2
            : size_;
    LITERT_LOG(LITERT_INFO, "cpu_buffer_size_: %zu data: %p", cpu_buffer_size_,
               data_);
    // Ensure the data is aligned.
    if (auto rc = posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                 cpu_buffer_size_);
        rc) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to allocate aligned memory");
    }
    // For write-only mode, we don't need to download the data from the
    // OpenCL buffer.
    if (lock_mode_ == kLiteRtLockWriteMode) {
      return Expected<T*>(static_cast<T*>(data_));
    }

    if (LiteRtGpuMemoryDownload) {
      // Use the GPU Accelerator API to download the data from the OpenCL buffer
      // to the aligned memory.
      bool res = LiteRtGpuMemoryDownload(GetMemoryPtr(), cpu_buffer_size_,
                                         data_, &tensor_type_, buffer_type_);
      if (!res) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to download data from GPU memory");
      }
    } else if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBuffer) {
      // When the GPU Accelerator API is not available, we can use the OpenCL
      // API to download the data from the OpenCL buffer to the aligned memory.
      tflite::gpu::cl::CLCommandQueue* queue =
          GpuEnvironmentSingleton::GetInstance().getCommandQueue();
      std::vector<T> result;
      auto status = buffer_.ReadData(queue, &result);
      if (!status.ok()) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to read OpenCL buffer");
      }
      // Copy the data from the OpenCL buffer to the aligned memory.
      // TODO(piyu): Consider adding support in MLD OpenCL buffer to directly
      // write to the aligned memory.
      std::copy(result.begin(), result.end(), static_cast<T*>(data_));
    } else {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "LiteRtGpuMemoryDownload is not supported");
    }
  }
  return Expected<T*>(static_cast<T*>(data_));
}

template <typename T>
Expected<void> OpenClMemory::Unlock() {
  absl::MutexLock lock(&mutex_);
  // For read-only mode, we don't need to write the data to the OpenCL buffer.
  if (lock_mode_ == kLiteRtLockReadMode) {
    return Expected<void>();
  }

  // The buffer has not been locked, so we don't need to write back.
  if (data_ == nullptr) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "Cannot unlock a buffer that wasn't locked in the first place");
  }

  if (LiteRtGpuMemoryUpload) {
    // Use the GPU Accelerator API to upload the data from the OpenCL buffer
    // to the aligned memory.
    bool res = LiteRtGpuMemoryUpload(GetMemoryPtr(), cpu_buffer_size_, data_,
                                     &tensor_type_, buffer_type_);
    if (res) {
      return Expected<void>();
    }
  } else if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBuffer) {
    // When the GPU Accelerator API is not available, we can use the OpenCL //
    // API to upload the data from the aligned memory to the OpenCL buffer.
    tflite::gpu::cl::CLCommandQueue* queue =
        GpuEnvironmentSingleton::GetInstance().getCommandQueue();
    size_t write_size = (size_ + sizeof(T) - 1) / sizeof(T);
    auto status = buffer_.WriteData(
        queue, absl::MakeSpan(static_cast<T*>(data_), write_size));

    if (status.ok()) {
      return Expected<void>();
    }
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      "The data failed to write to the OpenCL buffer when unlocked");
}

bool OpenClMemory::IsSupported() {
  static bool is_supported = ::tflite::gpu::cl::LoadOpenCL().ok();
  return is_supported;
}

Expected<OpenClMemory> OpenClMemory::Alloc(
    const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes_size) {
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));

  tflite::gpu::cl::Buffer buffer;

  tflite::gpu::cl::CLContext* cl_context =
      GpuEnvironmentSingleton::GetInstance().getContext();
  auto result =
      tflite::gpu::cl::CreateReadWriteBuffer(bytes_size, cl_context, &buffer);
  if (!result.ok()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create OpenCL buffer");
  }

  return Expected<OpenClMemory>(tensor_type, buffer_type, std::move(buffer));
}

bool IsAhwbToClInteropSupported() {
  return ::tflite::gpu::cl::clImportMemoryARM != nullptr;
}

Expected<OpenClMemory> OpenClMemory::AllocFromAhwbBuffer(
    const LiteRtRankedTensorType& tensor_type, AhwbBuffer& ahwb_buffer) {
  cl_int error = CL_SUCCESS;
  const cl_import_properties_arm properties[] = {
      CL_IMPORT_TYPE_ARM,
      CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM,
      0,
  };

  cl_context context =
      GpuEnvironmentSingleton::GetInstance().getContext()->context();
  LITERT_RETURN_IF_ERROR(IsAhwbToClInteropSupported(),
                         Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                    "clImportMemoryARM is not supported"));
  LITERT_ASSIGN_OR_RETURN(size_t size_bytes,
                          AhwbBuffer::GetSize(ahwb_buffer.ahwb));
  cl_mem buffer =
      tflite::gpu::cl::clImportMemoryARM(context, CL_MEM_READ_WRITE, properties,
                                         ahwb_buffer.ahwb, size_bytes, &error);
  LITERT_RETURN_IF_ERROR(
      error == CL_SUCCESS,
      Unexpected(kLiteRtStatusErrorRuntimeFailure,
                 absl::StrCat("Failed to create OpenCL buffer from "
                              "AHardwareBuffer, cl_int error code:",
                              error)));

  tflite::gpu::cl::Buffer cl_buffer(buffer, size_bytes);

  return OpenClMemory(tensor_type, kLiteRtTensorBufferTypeOpenClBuffer,
                      std::move(cl_buffer), ahwb_buffer.ahwb);
}

}  // namespace internal
}  // namespace litert
