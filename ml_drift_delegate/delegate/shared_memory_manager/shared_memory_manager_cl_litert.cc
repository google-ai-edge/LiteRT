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

#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_cl_litert.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/graph_adapter.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "ml_drift_delegate/tflite/shared_const_tensor_map.h"
#include "weight_loader/external_weight_loader_litert.h"
#include "tflite/c/common.h"

#ifdef __ANDROID__
#include <android/hardware_buffer.h>

#include "ml_drift/cl/cl_image_format.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include <CL/cl.h>
#endif

namespace ml_drift {
namespace internal {

#ifdef __ANDROID__
// Tries to allocate a cl::Tensor backed by an AHardwareBuffer via the
// cl_arm_import_memory extension. On Mali GPUs, standard clCreateBuffer /
// clCreateImage maps GPU memory into the process address space, inflating RSS.
// This is problematic on Android because the low memory killer (lmk) uses RSS
// to decide which processes to terminate under memory pressure — inflated RSS
// causes the app to be killed even when the system has sufficient physical
// memory. Using clImportMemoryARM with AHardwareBuffer avoids this because the
// memory is allocated through Android's graphics subsystem and is not counted
// toward process RSS.
//
// Currently only supports TEXTURE_2D tensors, which are the primary storage
// type on Mali GPUs and account for the vast majority of weight memory.
//
// Returns true if the tensor was successfully created via AHWB; |tensor| is
// populated. Returns false on any failure; the caller should fall through to
// the standard CreateTensor path.
bool TryCreateTensorViaAhwb(
    const cl::Environment& env,
    ml_drift::TensorDescriptor& tensor_desc,
    std::unique_ptr<GpuSpatialTensor>& tensor) {
  if (cl::clImportMemoryARM == nullptr) return false;
  if (tensor_desc.GetStorageType() != TensorStorageType::TEXTURE_2D) {
    return false;
  }

  // Compute the buffer size: width * height * channels * sizeof(element).
  std::vector<uint64_t> storage_dims = tensor_desc.GetStorageDims();
  const int element_size = tensor_desc.GetElementSize();
  const DataType data_type = tensor_desc.GetDataType();
  const size_t data_size =
      storage_dims[0] * storage_dims[1] * element_size * SizeOf(data_type);

  // All AHardwareBuffer APIs require Android 26+. The __builtin_available
  // check must be an if-condition (not an early return) so the compiler
  // can verify availability for every call site within the block.
  if (__builtin_available(android 26, *)) {
    // Allocate an AHardwareBuffer.
    AHardwareBuffer_Desc ahwb_desc = {};
    ahwb_desc.width = data_size;
    ahwb_desc.height = 1;
    ahwb_desc.layers = 1;
    ahwb_desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
    ahwb_desc.usage = AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;

    AHardwareBuffer* ahwb = nullptr;
    if (AHardwareBuffer_allocate(&ahwb_desc, &ahwb) != 0) return false;

    // Import the AHWB into OpenCL as a buffer via cl_arm_import_memory.
    const cl_import_properties_arm properties[] = {
        CL_IMPORT_TYPE_ARM,
        CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM,
        0,
    };
    cl_int error_code;
    cl_mem buffer_memory = cl::clImportMemoryARM(
        env.context().context(), CL_MEM_READ_WRITE, properties, ahwb,
        data_size, &error_code);
    if (!buffer_memory || error_code != CL_SUCCESS) {
      if (buffer_memory) cl::clReleaseMemObject(buffer_memory);
      AHardwareBuffer_release(ahwb);
      return false;
    }

    // Register a destructor callback to release the AHardwareBuffer when
    // the cl_mem is freed.
    cl::clSetMemObjectDestructorCallback(
        buffer_memory,
        [](cl_mem /*memobj*/, void* user_data) {
          if (__builtin_available(android 26, *)) {
            AHardwareBuffer_release(
                static_cast<AHardwareBuffer*>(user_data));
          }
        },
        ahwb);

    // Upload initial data (weights) to the buffer if present.
    if (!tensor_desc.GetData().empty()) {
      cl_int write_err = cl::clEnqueueWriteBuffer(
          const_cast<cl::Environment&>(env).queue()->queue(), buffer_memory,
          /*blocking_write=*/CL_TRUE, /*offset=*/0, data_size,
          tensor_desc.GetData().data(),
          /*num_events_in_wait_list=*/0,
          /*event_wait_list=*/nullptr, /*event=*/nullptr);
      if (write_err != CL_SUCCESS) {
        cl::clReleaseMemObject(buffer_memory);
        // AHWB released by the destructor callback.
        return false;
      }
    }

    // Create an Image2D view backed by the imported buffer. The Tensor
    // constructor will set buffer_based_ = true, so GPU reads use the
    // image view while the underlying buffer holds the actual memory.
    cl_image_desc image_desc = {};
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = storage_dims[0];
    image_desc.image_height = storage_dims[1];
    image_desc.image_row_pitch =
        storage_dims[0] * element_size * SizeOf(data_type);
    image_desc.buffer = buffer_memory;

    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = cl::DataTypeToChannelType(data_type);

    cl_int img_error;
    cl_mem image_memory = cl::CreateImage2DLegacy(
        env.context().context(), CL_MEM_READ_WRITE, &format, &image_desc,
        nullptr, &img_error);
    if (!image_memory || img_error != CL_SUCCESS) {
      if (image_memory) cl::clReleaseMemObject(image_memory);
      cl::clReleaseMemObject(buffer_memory);
      // AHWB released by the destructor callback.
      return false;
    }

    TensorDescriptor desc_copy;
    tensor_desc.CopyWithoutData(&desc_copy);
    tensor = std::make_unique<cl::Tensor>(
        buffer_memory, /*memory_owner=*/true, image_memory, desc_copy);
    return true;
  }
  return false;
}
#endif  // __ANDROID__

}  // namespace internal

std::unique_ptr<::ml_drift::SharedMemoryManager>
MakeSharedMemoryManagerClLitert(
    const ::ml_drift::cl::Environment& env,
    const ::LiteRtRuntimeContext* runtime_context,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    std::unique_ptr<::ml_drift::GraphAdapter> graph_adapter,
    TfLiteContext* context,
    ::ml_drift::ValueIdToSharedTensorMap& value_to_tensor_map,
    ::ml_drift::ValueIdToSharedTensorMap& quant_param_tensors,
    bool has_prepacked_external_tensors,
    ::ml_drift::SerializationWeightCache* serialization_cache,
    bool madvise_original_tensors, weight_loader::WeightLoader* weight_loader,
    const TensorIndexToExternalBufferIdMap* external_buffer_id_map) {
  ::ml_drift::SharedMemoryManager::CreateTensorFromDeviceBufferFunc
      device_buffer_import =
          [&env, weight_loader, external_buffer_id_map, runtime_context](
              const ::litert::ml_drift::SharedTfliteTensor&
                  shared_tflite_tensor,
              const ::ml_drift::TensorDescriptor& tensor_desc,
              std::unique_ptr<::ml_drift::GpuSpatialTensor>& tensor)
      -> absl::Status {
    if (!weight_loader) {
      return absl::NotFoundError("Weight loader not available");
    }
    if (external_buffer_id_map == nullptr) {
      return absl::NotFoundError("External buffer map not available");
    }
    auto it = external_buffer_id_map->find(
        static_cast<size_t>(shared_tflite_tensor.tflite_tensor_id));
    if (it == external_buffer_id_map->end() || it->second == 0) {
      return absl::NotFoundError("Tensor lacks external buffer id");
    }
    const auto* access = weight_loader->GetExternalWeightByBuffer(
        static_cast<uint32_t>(it->second));
    if (access == nullptr || access->GetDeviceBuffer() == nullptr) {
      return absl::NotFoundError("No external device buffer");
    }
    cl_mem cl_memory;
    if (runtime_context->get_tensor_buffer_opencl_memory(
            access->GetDeviceBuffer(), &cl_memory) != kLiteRtStatusOk) {
      return absl::InternalError(
          "Failed to get OpenCL memory from device buffer");
    }

    ::ml_drift::TensorDescriptor desc_copy;
    tensor_desc.CopyWithoutData(&desc_copy);

    auto cl_tensor = std::make_unique<::ml_drift::cl::Tensor>();
    ABSL_RETURN_IF_ERROR(::ml_drift::cl::CreateTensorShared(
        env.context(), cl_memory, desc_copy, cl_tensor.get()));

    tensor = std::move(cl_tensor);
    return absl::OkStatus();
  };

  ::ml_drift::SharedMemoryManager::CreateTensorFunc create_tensor_func =
      [&env](ml_drift::TensorDescriptor& tensor_desc,
             size_t page_adjusted_offset,
             ::litert::ml_drift::ReleaseDataCallback release_data_callback,
             std::unique_ptr<GpuSpatialTensor>& tensor) {
        if (tensor) {
          return absl::InternalError("Tensor is already initialized.");
        }
        if (release_data_callback) {
          return absl::InvalidArgumentError(
              "Release data callback is not currently supported on OpenCL.");
        }
#ifdef __ANDROID__
        if (internal::TryCreateTensorViaAhwb(env, tensor_desc, tensor)) {
          return absl::OkStatus();
        }
#endif  // __ANDROID__
        tensor = std::make_unique<cl::Tensor>();
        return CreateTensor(env.context(), tensor_desc,
                            dynamic_cast<cl::Tensor*>(tensor.get()));
      };

  ::ml_drift::SharedMemoryManager::MaybeBindTensorDataFunc maybe_bind_data =
      [weight_loader, external_buffer_id_map, runtime_context](
          const ::litert::ml_drift::SharedTfliteTensor& shared_tflite_tensor,
          TfLiteTensor& tensor) -> absl::Status {
    if (!weight_loader) {
      return absl::OkStatus();
    }
    if (external_buffer_id_map == nullptr) {
      return absl::OkStatus();
    }
    auto it = external_buffer_id_map->find(
        static_cast<size_t>(shared_tflite_tensor.tflite_tensor_id));
    if (it == external_buffer_id_map->end() || it->second == 0) {
      // This is invoked for shared tensor broadly, not only external weight
      // tensors. For non-external tensors, "not in map / id is zero" is
      // expected.
      return absl::OkStatus();
    }
    const uint32_t external_buffer_id = static_cast<uint32_t>(it->second);
    weight_loader::WeightAccessRequest request;
    request.cpu = true;
    absl::Status prepare_status = weight_loader->PrepareAccessForBuffer(
        external_buffer_id, request, /*env=*/nullptr);
    if (!prepare_status.ok()) {
      return prepare_status;
    }
    const auto* access = weight_loader->GetExternalWeightByBuffer(
        external_buffer_id);
    if (access == nullptr || access->GetHostBuffer() == nullptr) {
      return absl::NotFoundError(
          "Prepared external OpenCL host weight not found.");
    }
    void* host_memory = nullptr;
    if (runtime_context->get_tensor_buffer_host_memory(
            access->GetHostBuffer(), &host_memory) != kLiteRtStatusOk ||
        host_memory == nullptr) {
      return absl::InternalError("Failed to get host memory.");
    }
    size_t buffer_size = 0;
    if (runtime_context->get_tensor_buffer_size(
            access->GetHostBuffer(), &buffer_size) != kLiteRtStatusOk) {
      return absl::InternalError("Failed to get buffer size.");
    }
    if (static_cast<size_t>(tensor.bytes) > buffer_size) {
      return absl::InternalError("Tensor size is larger than buffer size.");
    }
    size_t buffer_offset = 0;
    if (runtime_context->get_tensor_buffer_offset(
            access->GetHostBuffer(), &buffer_offset) != kLiteRtStatusOk) {
      return absl::InternalError("Failed to get buffer offset.");
    }
    const char* raw_ptr =
        reinterpret_cast<const char*>(host_memory) + buffer_offset;
    tensor.data.raw_const = raw_ptr;
    tensor.data.raw = const_cast<char*>(raw_ptr);
    // Mark as custom allocation so TFLite won't try to free this memory
    // during cleanup. The memory is owned by the weight_loader.
    tensor.allocation_type = kTfLiteCustom;
    return absl::OkStatus();
  };

  ::ml_drift::SharedMemoryManager::PackingLookupFunc packing_lookup =
      [weight_loader](uint32_t global_id) -> absl::StatusOr<std::string> {
    if (weight_loader == nullptr) {
      return absl::NotFoundError("Weight loader not available");
    }
    if (global_id == 0) {
      return absl::InvalidArgumentError("Global id is zero.");
    }
    const auto* info = weight_loader->FindWeightInfoByBuffer(global_id);
    if (info == nullptr || info->packing.empty()) {
      return absl::NotFoundError("Packing info not found.");
    }
    return std::string(info->packing);
  };

  return std::make_unique<::ml_drift::SharedMemoryManager>(
      env.GetDevicePtr()->GetInfo(), create_info, std::move(graph_adapter),
      create_tensor_func, context, value_to_tensor_map, quant_param_tensors,
      has_prepacked_external_tensors, serialization_cache,
      madvise_original_tensors, /*experimental_int4_unpacking=*/true,
      /*experimental_int2_unpacking=*/false, std::move(device_buffer_import),
      std::move(maybe_bind_data), std::move(packing_lookup));
}

}  // namespace ml_drift
