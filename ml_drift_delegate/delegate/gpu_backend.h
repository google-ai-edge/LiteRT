// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
// clang-format off
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
// clang-format on
#include "litert/c/litert_tensor_buffer_types.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "tflite/c/common.h"

// Forward declaration to depend on LiteRT only when needed.
class LiteRtEnvironmentT;
using LiteRtEnvironment = LiteRtEnvironmentT*;

// A unique pointer implementing a deleter for LiteRtTensorBufferT.
class LiteRtTensorBufferT;
struct GpuTensorBufferDeleter;
using GpuTensorBufferPtr =
    std::unique_ptr<LiteRtTensorBufferT, GpuTensorBufferDeleter>;

struct GpuTensorBufferDeleter {
  void (*destroy_tensor_buffer)(LiteRtTensorBufferT*) = nullptr;
  void operator()(LiteRtTensorBufferT* buffer) const {
    if (buffer && destroy_tensor_buffer) {
      destroy_tensor_buffer(buffer);
    }
  }
};

namespace litert::ml_drift {

// Forward declarations to avoid circular dependency.
struct MlDriftDelegateData;
// Forward declarations defined after GpuBackend below.
class GpuInferenceContext;
class GpuIOBuffer;
class GpuTensorWrapper;
class Tensor2BufferConverter;
class Buffer2TensorConverter;

// Backend-specific memory handles. The actual type is determined by the
// backend, for example, cl_mem for OpenCL backend.
using GpuMemoryHandle = void*;

// Backend-specific event handles. The actual type is determined by the backend,
// for example, cl_event for OpenCL backend.
using GpuEventHandle = void*;

// Base classes to provide the environment and objects specific to a given GPU
// backend used by ML Drift delegates. It's to abstract the differences among
// different ML Drift GPU backends.
// Delegate implementation for a given GPU backend would be instantiated and
// owned by ML Drift delegate of the given GPU backend, for example,
// delegate_opencl.cc for OpenCL backend with LiteRT.
// [REQUIRED]
class GpuBackend {
 public:
  // GPU buffer requirements consist of the buffer types in order and the
  // corresponding strides in bytes.
  struct GpuBufferRequirements {
    std::vector<LiteRtTensorBufferType> buffer_types;
    std::vector<uint32_t> strides;
  };

  virtual ~GpuBackend() = default;

  // Returns the name of the backend.
  // [REQUIRED]
  virtual absl::string_view GetBackendName() = 0;



  // Returns the prefix of the serialized data.
  // [REQUIRED]
  virtual absl::string_view GetSerializedDataPrefix() = 0;

  // Returns the GPU device info.
  // [REQUIRED]
  virtual absl::StatusOr<::ml_drift::GpuInfo> GetInfo() = 0;

  // Returns the fastest storage type for the GPU device.
  // [REQUIRED]
  virtual absl::StatusOr<::ml_drift::TensorStorageType>
  GetFastestStorageType() = 0;

  // Gets GPU memory allocated for LiteRT TensorBuffer.
  // [OPTIONAL] Only for LiteRT delegates.
  virtual absl::StatusOr<GpuMemoryHandle> GetGpuMemoryAllocated(
      const GpuTensorBufferPtr& tensor_buffer) = 0;

  // Gets GPU events associated with the given tensor buffer.
  // [OPTIONAL] Only for asynchronous mode in LiteRT delegates.
  virtual absl::StatusOr<GpuEventHandle> GetGpuEventAssociated(
      const GpuTensorBufferPtr& tensor_buffer) = 0;

  // Associates a GPU event with the given tensor buffer. The event can be
  // retrieved later by calling GetGpuEventAssociated().
  // [OPTIONAL] Only for asynchronous mode in LiteRT delegates.
  virtual absl::Status AssociateGpuEvent(
      GpuEventHandle event, LiteRtEnvironment env,
      GpuTensorBufferPtr& tensor_buffer) = 0;

  // Waits for all pending GPU operations to complete.
  // [REQUIRED]
  virtual absl::Status WaitForCompletion() = 0;

  // Returns the GPU buffer requirements for the given storage & data type.
  // [OPTIONAL] Only for LiteRT delegates.
  virtual absl::StatusOr<GpuBufferRequirements> GetGpuBufferRequirements(
      ::ml_drift::TensorStorageType used_storage_type,
      ::ml_drift::DataType data_type) = 0;

  // Returns the GPU buffer requirements for the non-external tensors.
  // [OPTIONAL] Only for no-external tensor mode in LiteRT delegates.
  virtual absl::StatusOr<GpuBufferRequirements>
  GetGpuBufferRequirementsForNonExternalTensors() = 0;

  // Creates a `GpuInferenceContext`. If serialized_model is not null, the
  // serialized model will be stored there. If may_share_memory_manager is true,
  // memory manager might be shared among inference contexts for some backends.
  // [REQUIRED]
  virtual absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
  CreateInferenceContext(const ::ml_drift::CreateGpuModelInfo& create_info,
                         ::ml_drift::GpuModel& gpu_model,
                         std::vector<uint8_t>* serialized_model,
                         bool may_share_memory_manager) = 0;

  // Creates a `GpuInferenceContext` without serialized model output parameter.
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> CreateInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GpuModel& gpu_model) {
    return CreateInferenceContext(create_info, gpu_model,
                                  /*serialized_model=*/nullptr,
                                  /*may_share_memory_manager=*/false);
  }

  // Creates a `GpuInferenceContext` and restores content from the given
  // serialized data.
  // [REQUIRED]
  virtual absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
  RestoreInferenceContext(const ::ml_drift::CreateGpuModelInfo& create_info,
                          absl::Span<const uint8_t> serialized_model) = 0;

  // Creates a `SharedMemoryManager`.
  // [REQUIRED]
  virtual absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
  CreateSharedMemoryManager(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GraphFloat32& graph, TfLiteContext* context,
      MlDriftDelegateData& delegate_data,
      ::ml_drift::SerializationWeightCache* serialization_cache) = 0;

  // Creates a `WeightsManager` to manage the weights for GPU backend.
  // [REQUIRED]
  virtual absl::StatusOr<std::shared_ptr<::ml_drift::WeightsManager>>
  CreateWeightsManager() {
    return std::make_shared<::ml_drift::WeightsManager>();
  }

  // Returns the batches for weights preparation.
  // [OPTIONAL]
  virtual absl::StatusOr<std::vector<
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
  GetBatchesForWeightsPreparation(
      ::ml_drift::WeightsManager* weights_manager) = 0;

  // Prepares the weights in one batch gotten from
  // GetBatchesForWeightsPreparation.
  // [OPTIONAL]
  virtual absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatch(
      ::ml_drift::WeightsManager* weights_manager,
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&
          op_infos) = 0;

  // Prepares the weights in batches for GPU backend. This is only being used
  // when GPU weights preparation is enabled.
  // [OPTIONAL]
  virtual absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatches(::ml_drift::WeightsManager* weights_manager) = 0;

  // Creates a `GpuTensorWrapper` with the given descriptor and GPU memory.
  // Note that the GPU memory is not owned by the tensor. The caller should
  // make sure the GPU memory is valid during the lifetime of the tensor, and
  // deallocate it when the gpu memory is no longer needed.
  // Note that the returned tensor is not a `::ml_drift::GpuSpatialTensor`
  // instance, but its wrapper for flexibility.
  // [OPTIONAL] Only for LiteRT delegates.
  virtual absl::StatusOr<std::unique_ptr<GpuTensorWrapper>> CreateTensorWrapper(
      const ::ml_drift::TensorDescriptor& desc, GpuMemoryHandle gpu_memory) = 0;

  virtual absl::Status ReadSpatialTensorToDescriptor(
      ::ml_drift::GpuSpatialTensor& tensor,
      ::ml_drift::TensorDescriptor& desc) = 0;

  // Updates the given spatial tensor with the given descriptor. If the tensor
  // already has GPU memory, it will be released.
  //
  // If this function should take responsibility for freeing the desc's data,
  // the user is required to pass a valid release_data_callback to release the
  // data. This is most likely to happen when the TensorDescriptor is an
  // UnownedDataTensorDescriptor so the data won't automatically be released
  // when the TensorDescriptor is destroyed.
  // The page_adjusted_offset is the offset of the buffer data within the mmap
  // region.
  // [OPTIONAL] Only for LiteRT delegates.
  virtual absl::Status UpdateSpatialTensor(
      ::ml_drift::GpuSpatialTensor* tensor,
      const ::ml_drift::TensorDescriptor& desc, size_t page_adjusted_offset,
      ReleaseDataCallback release_data_callback) = 0;

  // Releases the memory of the given spatial tensor without destroying the
  // tensor.
  // [OPTIONAL] Only for LiteRT delegates.
  virtual absl::Status ReleaseSpatialTensorMemory(
      ::ml_drift::GpuSpatialTensor* tensor) = 0;

  // Creates a `GpuIOBuffer` with the given GPU memory for IO, i.e. data
  // transfer between CPU and GPU.
  // Note that the GPU memory is not owned by the buffer. The caller should
  // make sure the GPU memory is valid during the lifetime of the buffer, and
  // deallocate it when the gpu memory is no longer needed.
  // [OPTOINAL] Only for no-external tensor mode in LiteRT delegates.
  virtual absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBuffer(
      GpuMemoryHandle gpu_memory) = 0;

  // Creates a `GpuIOBuffer` of the given size for IO, i.e. data transfer
  // between CPU and GPU. The buffer will be used either input or output, but
  // not both.
  // [OPTIONAL] Only for legacy TfLite delegates.
  virtual absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBufferWithSize(
      ::ml_drift::DataType data_type, size_t size, bool input) = 0;

  // Creates a `Tensor2BufferConverter`.
  // [OPTIONAL] Only for no-external tensor mode in LiteRT delegates or for
  // legacy TfLite delegates.
  virtual absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
  CreateTensor2BufferConverter(
      const ::ml_drift::TensorDescriptor& src_desc,
      const ::ml_drift::BufferDescriptor& dst_desc) = 0;

  // Creates a `Buffer2TensorConverter`.
  // [OPTIONAL] Only for no-external tensor mode in LiteRT delegates or for
  // legacy TfLite delegates.
  virtual absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
  CreateBuffer2TensorConverter(
      const ::ml_drift::BufferDescriptor& src_desc,
      const ::ml_drift::TensorDescriptor& dst_desc) = 0;
};

// Inference context of a GPU backend initialized with a GPU model. It's a
// wrapper of ML Drift's inference context, e.g. cl::InferenceContext for
// OpenCL backend.
// [REQUIRED]
class GpuInferenceContext {
 public:
  virtual ~GpuInferenceContext() = default;

  // Gets the spatial tensor bound to the given id.
  // [REQUIRED]
  virtual absl::StatusOr<::ml_drift::GpuSpatialTensor*> GetSpatialTensor(
      ::ml_drift::ValueId id) = 0;

  // Binds the spatial tensor to the given id.
  // [OPTIONAL] Only when external tensors exist.
  virtual absl::Status BindSpatialTensor(
      ::ml_drift::ValueId id, ::ml_drift::GpuSpatialTensor* tensor) = 0;

  virtual absl::Status UploadWeightsOnWeb(
      weight_loader::WeightLoader* weight_loader,
      const ::ml_drift::GpuModel& gpu_model,
      const absl::flat_hash_map<::ml_drift::ValueId, ::ml_drift::ValueId>&
          io_mapping,
      const absl::flat_hash_map<::ml_drift::ValueId, uint32_t>&
          weight_id_to_external_buffer_id,
      std::vector<::ml_drift::WeightsManager::UploadWeightsInfo>&
          upload_infos) = 0;

  // Writes weight data to a spatial tensor bound to the given id synchronously.
  // [OPTIONAL] Only used by Gpu-weights-preparation enabled when
  // `convert_weights_on_gpu` is true.
  virtual absl::Status WriteDataToWeightTensor(
      ::ml_drift::ValueId id, absl::Span<const uint8_t> data) = 0;

  // Reads a spatial tensor bound to the given id to a tensor descriptor
  // synchronously.
  // [OPTIONAL] Only used by Gpu-weights-preparation enabled when
  // `convert_weights_on_gpu` is true and cache is enabled.
  virtual absl::Status ReadWeightTensorToDescriptor(
      ::ml_drift::ValueId id, ::ml_drift::TensorDescriptor& desc) = 0;

  // Executes inference by dispatching the inference context to the GPU queue.
  // [REQUIRED]
  virtual absl::Status Dispatch() = 0;

  // Gets a GPU event used to synchronize with the inference context before
  // dispatching. If it returns with a GPU event, the client must wait for it
  // completed by calling WaitForEventsCompleted() before Dispatch().
  // If it returns absl::StatusCode::kNotFoundError, synchronization is not
  // needed.
  // [OPTIONAL] Only for asynchronous mode in LiteRT delegates.
  virtual absl::StatusOr<GpuEventHandle> GetPreDispatchEvent() = 0;

  // Gets a GPU event used to synchronize with the inference context after
  // dispatching. If it returns with a GPU event, the client should either wait
  // for it completed by calling WaitForEventsCompleted() or associate it with a
  // tensor buffer to wait for it later by calling
  // GpuBackend::AssociateGpuEvent().
  // If it returns absl::StatusCode::kNotFoundError, synchronization is not
  // needed.
  // [OPTIONAL] Mostly only for asynchronous mode in LiteRT delegates.
  virtual absl::StatusOr<GpuEventHandle> GetPostDispatchEvent(
      bool is_async_execution_mode) = 0;

  // Waits for all the given GPU events to complete.
  // If `force_sync` is true, waits for events synchronously and the current
  // thread will be blocked until all the events are fulfilled.
  // If `force_sync` is false, waits for events synchronously or asynchronously
  // depending on the backend. For example, OpenCL backend may enqueue a
  // barrier to block following commands from executing until all the events are
  // fulfilled, but doesn't block the current thread.
  // [OPTIONAL] Only for asynchronous mode in LiteRT delegates.
  virtual absl::Status WaitForEventsCompleted(absl::Span<GpuEventHandle> events,
                                              bool force_sync) = 0;

  // Called before input (if `input` is true) or output (if `input` is false)
  // tensor layout conversions between BHWC and PHWC4. It may have additional
  // works specific to GPU backend, e.g. initializing command buffers.
  // [OPTIONAL] Only for no-external tensor mode in LiteRT delegates or for
  // legacy TfLite delegates.
  virtual absl::Status PreConvert(bool input) = 0;

  // Called after input (if `input` is true) or output (if `input` is false)
  // tensor layout conversions between BHWC and PHWC4. It may have additional
  // works specific to GPU backend, e.g. finalizing command buffers.
  // [OPTIONAL] Only for no-external tensor mode in LiteRT delegates or for
  // legacy TfLite delegates.
  virtual absl::Status PostConvert(bool input) = 0;

  // Profiles the inference context.
  // [OPTIONAL] Only when backend supports profiling.
  virtual absl::Status Profile(::ml_drift::ProfilingInfo& profiling_info) = 0;

  // Returns the size of memory allocated for intermediate tensors.
  // [OPTIONAL] Only when backend supports profiling.
  virtual absl::StatusOr<size_t>
  GetSizeOfMemoryAllocatedForIntermediateTensors() const = 0;

  // Returns the size of memory allocated for constant tensors.
  // [OPTIONAL] Only when backend supports profiling.
  virtual absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForConstantTensors()
      const = 0;

  // Returns the size of memory allocated for external tensors.
  // [OPTIONAL] Only when backend supports profiling.
  virtual absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForExternalTensors()
      const = 0;

  // Reports memory benchmark to stdout if enabled.
  // [OPTIONAL] Only when backend supports memory benchmark.
  virtual absl::Status ReportMemoryBenchmarkIfEnabled(
      const ::ml_drift::CreateGpuModelInfo& create_info) = 0;

  // The larger computation nodes are supposed to get proportionally smaller
  // command batch sizes (eg. otherwise, WebGPU backend will consume additional
  // memory and even errors).
  // [OPTIONAL] Only when backend supports setting the number of nodes per
  // command encoder.
  virtual absl::Status SetCommandBufferHint(
      int num_nodes_per_command_encoder) = 0;
};

// A wrapper class of `::ml_drift::GpuSpatialTensor` for flexibility.
// [OPTIONAL] Only for LiteRT delegates.
class GpuTensorWrapper {
 public:
  virtual ~GpuTensorWrapper() = default;

  // Returns the underlying `::ml_drift::GpuSpatialTensor`. Always succeeds.
  virtual ::ml_drift::GpuSpatialTensor& Get() = 0;
};

// A wrapper class to provide the GPU buffer object for a given GPU backend used
// mainly to transfer data between CPU and GPU. For example,
// ml_drift::cl::Buffer for OpenCL backend.
// [OPTIONAL] Only for no-external tensor mode in LiteRT delegates or for legacy
// TfLite delegates.
class GpuIOBuffer {
 public:
  virtual ~GpuIOBuffer() = default;

  // Reads the content of this buffer and fills the given data span.
  // Note that the implementation may read the data asynchronously.
  // GpuBackend::WaitForCompletion() should be called before accessing the data.
  virtual absl::Status Read(absl::Span<uint8_t> data) = 0;

  // Writes the content of the given data span to this buffer.
  // Note that the implementation may write the data asynchronously.
  // GpuBackend::WaitForCompletion() should be called before deallocating the
  // buffer of data.
  virtual absl::Status Write(absl::Span<const uint8_t> data) = 0;
};

// Converter (i.e. data transferer with potential memory layout change) from a
// GPU spatial tensor to an IO buffer which is used to transfer data from GPU to
// CPU or GPU.
// [OPTIONAL] Only for no-external tensor mode in LiteRT delegates or for legacy
// TfLite delegates.
class Tensor2BufferConverter {
 public:
  virtual ~Tensor2BufferConverter() = default;

  // Converts a tensor to a buffer.
  virtual absl::Status Convert(::ml_drift::GpuSpatialTensor& src_tensor,
                               GpuIOBuffer& dst_buffer) = 0;
};

// Converter (i.e. data transferer with potential memory layout change) from an
// IO buffer to a GPU spatial tensor which is used to transfer data from CPU or
// GPU to GPU.
// [OPTIONAL] Only for no-external tensor mode in LiteRT delegates or for legacy
// TfLite delegates.
class Buffer2TensorConverter {
 public:
  virtual ~Buffer2TensorConverter() = default;

  // Converts a buffer to a tensor.
  virtual absl::Status Convert(GpuIOBuffer& src_buffer,
                               ::ml_drift::GpuSpatialTensor& dst_tensor) = 0;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_H_
