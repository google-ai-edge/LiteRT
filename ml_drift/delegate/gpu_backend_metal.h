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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_METAL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_METAL_H_

#import <Metal/Metal.h>

#ifdef __cplusplus

#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/metal/buffer.h"  // from @ml_drift
#include "ml_drift/metal/converter.h"  // from @ml_drift
#include "ml_drift/metal/inference_context.h"  // from @ml_drift
#include "ml_drift/metal/memory_manager.h"  // from @ml_drift
#include "ml_drift/metal/metal_device.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "third_party/odml/infra/ml_drift_delegate/shared_memory_manager.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_data.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_options.h"
#include "third_party/odml/litert/ml_drift/delegate/gpu_backend.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

// GpuBackend for Metal.
class GpuBackendMetal : public GpuBackend {
 public:
  explicit GpuBackendMetal(GpuDelegateWaitType wait_type);
  GpuBackendMetal(::ml_drift::metal::MetalDevice* device,
                  GpuDelegateWaitType wait_type);
  GpuBackendMetal(::ml_drift::metal::MetalDevice* device,
                  GpuDelegateWaitType wait_type,
                  id<MTLCommandQueue> command_queue,
                  bool enable_residency_set = false);
  ~GpuBackendMetal() override;

  // Implementation of GpuBackend.
  absl::string_view GetBackendName() override;
  absl::string_view GetSerializedDataPrefix() override;
  absl::StatusOr<::ml_drift::GpuInfo> GetInfo() override;
  absl::StatusOr<::ml_drift::TensorStorageType> GetFastestStorageType()
      override;
  absl::StatusOr<GpuMemoryHandle> GetGpuMemoryAllocated(
      const GpuTensorBufferPtr& tensor_buffer) override;
  absl::StatusOr<GpuEventHandle> GetGpuEventAssociated(
      const GpuTensorBufferPtr& tensor_buffer) override;
  absl::Status AssociateGpuEvent(GpuEventHandle event, LiteRtEnvironment env,
                                 GpuTensorBufferPtr& tensor_buffer) override;
  absl::Status WaitForCompletion() override;
  absl::StatusOr<GpuBufferRequirements> GetGpuBufferRequirements(
      ::ml_drift::TensorStorageType used_storage_type,
      ::ml_drift::DataType data_type) override;
  absl::StatusOr<GpuBufferRequirements>
  GetGpuBufferRequirementsForNonExternalTensors() override;
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> CreateInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
      bool may_share_memory_manager) override;
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> RestoreInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      absl::Span<const uint8_t> serialized_model) override;
  absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
  CreateSharedMemoryManager(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GraphFloat32& graph, TfLiteContext* context,
      MlDriftDelegateData& delegate_data,
      ::ml_drift::SerializationWeightCache* serialization_cache) override;
  absl::StatusOr<std::shared_ptr<::ml_drift::WeightsManager>>
  CreateWeightsManager() override;
  absl::StatusOr<std::vector<
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
  GetBatchesForWeightsPreparation(
      ::ml_drift::WeightsManager* weights_manager) override;
  absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatch(
      ::ml_drift::WeightsManager* weights_manager,
      std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&
          op_infos) override;
  absl::StatusOr<absl::flat_hash_map<
      ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
  PrepareWeightsInBatches(::ml_drift::WeightsManager* weights_manager) override;
  absl::StatusOr<std::unique_ptr<GpuTensorWrapper>> CreateTensorWrapper(
      const ::ml_drift::TensorDescriptor& desc,
      GpuMemoryHandle gpu_memory) override;
  absl::Status ReadSpatialTensorToDescriptor(
      ::ml_drift::GpuSpatialTensor& tensor,
      ::ml_drift::TensorDescriptor& desc) override;
  absl::Status UpdateSpatialTensor(
      ::ml_drift::GpuSpatialTensor* tensor,
      const ::ml_drift::TensorDescriptor& desc, size_t page_adjusted_offset,
      ::ml_drift_delegate::ReleaseDataCallback release_data_callback) override;
  absl::Status ReleaseSpatialTensorMemory(
      ::ml_drift::GpuSpatialTensor* tensor) override;
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBuffer(
      GpuMemoryHandle gpu_memory) override;
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBufferWithSize(
      ::ml_drift::DataType data_type, size_t size, bool input) override;
  absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
  CreateTensor2BufferConverter(
      const ::ml_drift::TensorDescriptor& src_desc,
      const ::ml_drift::BufferDescriptor& dst_desc) override;
  absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
  CreateBuffer2TensorConverter(
      const ::ml_drift::BufferDescriptor& src_desc,
      const ::ml_drift::TensorDescriptor& dst_desc) override;

  // Metal Delegate wait type options.
  GpuDelegateWaitType wait_type() const { return wait_type_; }

  // Underlying Metal device.
  ::ml_drift::metal::MetalDevice* metal_device() const { return device_; }
  ::ml_drift::metal::MemoryManager& memory_manager() { return memory_manager_; }
  // Underlying Metal command queue.
  id<MTLCommandQueue> command_queue() const { return command_queue_; }
  id<MTLResidencySet> residency_set() const { return residency_set_; }
  // Set up a new command queue to the underying Metal command queue.
  void set_command_queue(id<MTLCommandQueue> command_queue) {
    command_queue_ = command_queue;
  }

  // Underlying Metal command buffer.
  id<MTLCommandBuffer> command_buffer() const { return command_buffer_; }
  void set_command_buffer(id<MTLCommandBuffer> command_buffer) {
    command_buffer_ = command_buffer;
  }

  // Underlying Metal compute command encoder.
  id<MTLComputeCommandEncoder> compute_command_encoder() const {
    return compute_command_encoder_;
  }
  void set_compute_command_encoder(id<MTLComputeCommandEncoder> encoder) {
    compute_command_encoder_ = encoder;
  }

 protected:
  void PopulateResidencySet(const ::ml_drift::CreateGpuModelInfo& create_info,
                            ::ml_drift::metal::InferenceContext& metal_ctx);

 private:
  void InitResidencySet();
  void StartHeartbeat();
  void StopHeartbeat();

  std::unique_ptr<::ml_drift::metal::MetalDevice> device_owned_;
  ::ml_drift::metal::MetalDevice* const device_;
  const GpuDelegateWaitType wait_type_;
  id<MTLCommandQueue> command_queue_;
  // Note that command buffer is created on demand whenever it's needed.
  // It's committed and reset only by WaitForCompletion() in this class.
  // Subclass may commit and reset it in different ways, e.g. on asynchronous
  // execution mode.
  id<MTLCommandBuffer> command_buffer_ = nullptr;
  id<MTLComputeCommandEncoder> compute_command_encoder_ = nullptr;
  ::ml_drift::metal::MemoryManager memory_manager_;
  id<MTLResidencySet> residency_set_ = nil;
  std::thread heartbeat_thread_;
  absl::Notification stop_heartbeat_;
};

class GpuInferenceContextMetal : public GpuInferenceContext {
 public:
  explicit GpuInferenceContextMetal(
      GpuBackendMetal* backend,
      ::ml_drift::metal::MemoryManager* memory_manager = nullptr);
  ~GpuInferenceContextMetal() override = default;

  // Implementation of GpuInferenceContext.
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> GetSpatialTensor(
      ::ml_drift::ValueId id) override;
  absl::Status BindSpatialTensor(::ml_drift::ValueId id,
                                 ::ml_drift::GpuSpatialTensor* tensor) override;
  absl::Status WriteDataToWeightTensor(::ml_drift::ValueId id,
                                       absl::Span<const uint8_t> data) override;
  absl::Status ReadWeightTensorToDescriptor(
      ::ml_drift::ValueId id, ::ml_drift::TensorDescriptor& desc) override;
  absl::Status Dispatch() override;
  absl::StatusOr<GpuEventHandle> GetPreDispatchEvent() override;
  absl::StatusOr<GpuEventHandle> GetPostDispatchEvent(
      bool is_async_execution_mode) override;
  absl::Status WaitForEventsCompleted(absl::Span<GpuEventHandle> events,
                                      bool force_sync) override;
  absl::Status PreConvert(bool input) override;
  absl::Status PostConvert(bool input) override;
  absl::Status Profile(::ml_drift::ProfilingInfo& profiling_info) override;
  absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForIntermediateTensors()
      const override;
  absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForConstantTensors()
      const override;
  absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForExternalTensors()
      const override;
  absl::Status ReportMemoryBenchmarkIfEnabled(
      const ::ml_drift::CreateGpuModelInfo& create_info) override;
  absl::Status SetCommandBufferHint(int num_nodes_per_command_encoder) override;

  // Underlying Metal inference context.
  ::ml_drift::metal::InferenceContext& metal_ctx() { return *ctx_; }
  const ::ml_drift::metal::InferenceContext& metal_ctx() const { return *ctx_; }

  GpuBackendMetal* metal_backend() const { return backend_; }

 private:
  GpuBackendMetal* const backend_;
  std::unique_ptr<::ml_drift::metal::InferenceContext> ctx_;
};

class GpuTensorWrapperMetal : public GpuTensorWrapper {
 public:
  GpuTensorWrapperMetal() = default;
  ~GpuTensorWrapperMetal() override = default;

  // Implementation of GpuTensorWrapper.
  ::ml_drift::GpuSpatialTensor& Get() override { return tensor_; }

  // Underlying Metal spatial tensor.
  ::ml_drift::metal::MetalSpatialTensor& metal_tensor() { return tensor_; }
  const ::ml_drift::metal::MetalSpatialTensor& metal_tensor() const {
    return tensor_;
  }

 private:
  ::ml_drift::metal::MetalSpatialTensor tensor_;
};

class GpuIOBufferMetal : public GpuIOBuffer {
 public:
  explicit GpuIOBufferMetal(GpuBackendMetal* backend_,
                            ::ml_drift::metal::Buffer buffer);
  ~GpuIOBufferMetal() override = default;

  // Implementation of GpuIOBuffer.
  absl::Status Read(absl::Span<uint8_t> data) override;
  absl::Status Write(absl::Span<const uint8_t> data) override;

  // Underlying immutable Metal buffer.
  const ::ml_drift::metal::Buffer& metal_buffer() const { return buffer_; }
  // Underlying mutable Metal buffer.
  ::ml_drift::metal::Buffer& metal_buffer() { return buffer_; }

 private:
  GpuBackendMetal* const backend_;
  ::ml_drift::metal::Buffer buffer_;
};

class Tensor2BufferConverterMetal : public Tensor2BufferConverter {
 public:
  explicit Tensor2BufferConverterMetal(
      GpuBackendMetal* backend,
      std::unique_ptr<::ml_drift::metal::TensorToBHWCBufferConverter>
          converter);
  ~Tensor2BufferConverterMetal() override = default;

  // Implementation of Tensor2BufferConverter.
  absl::Status Convert(::ml_drift::GpuSpatialTensor& src_tensor,
                       GpuIOBuffer& dst_buffer) override;

 private:
  GpuBackendMetal* const backend_;
  const std::unique_ptr<::ml_drift::metal::TensorToBHWCBufferConverter>
      converter_;
};

class Buffer2TensorConverterMetal : public Buffer2TensorConverter {
 public:
  explicit Buffer2TensorConverterMetal(
      GpuBackendMetal* backend,
      std::unique_ptr<::ml_drift::metal::BHWCBufferToTensorConverter>
          converter);
  ~Buffer2TensorConverterMetal() override = default;

  // Implementation of Buffer2TensorConverter.
  absl::Status Convert(GpuIOBuffer& src_buffer,
                       ::ml_drift::GpuSpatialTensor& dst_tensor) override;

 private:
  GpuBackendMetal* const backend_;
  const std::unique_ptr<::ml_drift::metal::BHWCBufferToTensorConverter>
      converter_;
};

}  // namespace litert::ml_drift

#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_METAL_H_
