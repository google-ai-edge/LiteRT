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

#if defined(__EMSCRIPTEN__) || defined(WIN32) || defined(_WIN32) || \
    defined(__WIN32__) || defined(__NT__) || defined(_WIN64)
#define LITERT_NATIVE_FP16_NOT_SUPPORTED
#endif

namespace litert {
namespace internal {

template Expected<float*> OpenClMemory::Lock<float>();
template Expected<char*> OpenClMemory::Lock<char>();
template Expected<void> OpenClMemory::Unlock<float>();
template Expected<void> OpenClMemory::Unlock<char>();

template <typename T>
Expected<T*> OpenClMemory::Lock() {
  absl::MutexLock lock(&mutex_);
  // The buffer has not been locked, so we need to read from the OpenCL
  // buffer.
  if (data_ == nullptr) {
    tflite::gpu::cl::CLCommandQueue* queue =
        GpuEnvironmentSingleton::GetInstance().getCommandQueue();
    if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferFp16) {
#ifdef LITERT_NATIVE_FP16_NOT_SUPPORTED
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "fp16 is not supported");
#else   // LITERT_NATIVE_FP16_NOT_SUPPORTED
      std::vector<_Float16> result;
      auto status = buffer_.ReadData(queue, &result);
      if (!status.ok()) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to read OpenCL buffer");
      }
      // Ensure the data is aligned.
      if (auto rc = posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                   size_ * 2);
          rc) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to allocate aligned memory");
      }
      // TODO b/407551791 - Vectorize the conversion.
      float* data_fp32 = static_cast<float*>(data_);
      for (int i = 0; i < result.size(); ++i) {
        data_fp32[i] = result[i];  // fp16 to fp32
      }
#endif  // LITERT_NATIVE_FP16_NOT_SUPPORTED
    } else {
      std::vector<T> result;
      auto status = buffer_.ReadData(queue, &result);
      if (!status.ok()) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to read OpenCL buffer");
      }
      // Ensure the data is aligned.
      if (auto rc = posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                   size_);
          rc) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to allocate aligned memory");
      }
      // Copy the data from the OpenCL buffer to the aligned memory.
      // TODO(piyu): Consider adding support in MLD OpenCL buffer to directly
      // write to the aligned memory.
      std::copy(result.begin(), result.end(), static_cast<T*>(data_));
    }
  }
  return Expected<T*>(static_cast<T*>(data_));
}

template <typename T>
Expected<void> OpenClMemory::Unlock() {
  absl::MutexLock lock(&mutex_);
  tflite::gpu::cl::CLCommandQueue* queue =
      GpuEnvironmentSingleton::GetInstance().getCommandQueue();
  // The buffer has not been locked, so we don't need to write back.
  if (data_ == nullptr) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "Cannot unlock a buffer that wasn't locked in the first place");
  }
  if (buffer_type_ == kLiteRtTensorBufferTypeOpenClBufferFp16) {
#ifdef LITERT_NATIVE_FP16_NOT_SUPPORTED
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "fp16 is not supported");
#else   // LITERT_NATIVE_FP16_NOT_SUPPORTED
    float* data_fp32 = static_cast<float*>(data_);
    size_t num_elements = size_ / sizeof(_Float16);
    // TODO b/407551791 - Vectorize the conversion.
    auto data_fp16 = std::make_unique<_Float16[]>(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      data_fp16[i] = static_cast<_Float16>(data_fp32[i]);
    }
    auto status = buffer_.WriteData(
        queue, absl::MakeSpan(reinterpret_cast<_Float16*>(data_fp16.get()),
                              num_elements));
    if (status.ok()) {
      return Expected<void>();
    }
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      std::string(status.message()));
#endif  // LITERT_NATIVE_FP16_NOT_SUPPORTED
  } else {
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

Expected<OpenClMemory> OpenClMemory::Alloc(LiteRtTensorBufferType buffer_type,
                                           size_t bytes_size) {
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

  return Expected<OpenClMemory>(buffer_type, std::move(buffer));
}

bool IsAhwbToClInteropSupported() {
  return ::tflite::gpu::cl::clImportMemoryARM != nullptr;
}

Expected<OpenClMemory> OpenClMemory::AllocFromAhwbBuffer(
    AhwbBuffer& ahwb_buffer) {
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

  return OpenClMemory(kLiteRtTensorBufferTypeOpenClBuffer, std::move(cl_buffer),
                      ahwb_buffer.ahwb);
}

}  // namespace internal
}  // namespace litert
