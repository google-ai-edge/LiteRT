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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_WEBGPU_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_WEBGPU_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>  // NOLINT (Open source code)
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
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
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/webgpu/buffer.h"  // from @ml_drift
#include "ml_drift/webgpu/converter.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/inference_context.h"  // from @ml_drift
#include "ml_drift/webgpu/memory_manager.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
// clang-format off
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
// clang-format on
#include "third_party/odml/infra/ml_drift_delegate/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

class GpuIOBufferWebGpu;

// GpuBackend for WebGPU.
class GpuBackendWebGpu : public GpuBackend {
 public:
  GpuBackendWebGpu();
  explicit GpuBackendWebGpu(::ml_drift::webgpu::ExecutionEnvironment* env);
  ~GpuBackendWebGpu() override = default;

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
      ReleaseDataCallback release_data_callback) override;
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

  // Underlying WebGPU execution environment.
  ::ml_drift::webgpu::ExecutionEnvironment& wgpu_env() { return *env_; }
  const ::ml_drift::webgpu::ExecutionEnvironment& wgpu_env() const {
    return *env_;
  }

  ::ml_drift::webgpu::MemoryManager& memory_manager() {
    return memory_manager_;
  }

  // Encoders for IO tensors.
  wgpu::CommandEncoder* command_encoder() const {
    return command_encoder_.get();
  }
  void set_command_encoder(
      std::unique_ptr<wgpu::CommandEncoder> command_encoder) {
    command_encoder_ = std::move(command_encoder);
  }

  wgpu::ComputePassEncoder* compute_pass_encoder() const {
    return compute_pass_encoder_.get();
  }
  void set_compute_pass_encoder(
      std::unique_ptr<wgpu::ComputePassEncoder> compute_pass_encoder) {
    compute_pass_encoder_ = std::move(compute_pass_encoder);
  }

  // Output buffers for PreRead when output conversion is done.
  absl::flat_hash_set<GpuIOBufferWebGpu*>& output_buffers() {
    return output_buffers_;
  }

  int num_steps_of_command_buffer_preparations() const {
    return num_steps_of_command_buffer_preparations_;
  }
  void set_num_steps_of_command_buffer_preparations(
      int num_steps_of_command_buffer_preparations) {
    num_steps_of_command_buffer_preparations_ =
        num_steps_of_command_buffer_preparations;
  }

 private:
  std::unique_ptr<::ml_drift::webgpu::ExecutionEnvironment> env_owned_;
  ::ml_drift::webgpu::ExecutionEnvironment* const env_;
  ::ml_drift::webgpu::MemoryManager memory_manager_;
  int num_steps_of_command_buffer_preparations_ = 0;

  // Encoder for batch of IO tensor conversions.
  std::unique_ptr<wgpu::CommandEncoder> command_encoder_;
  // Encoder for actual computation of IO tensor conversions.
  std::unique_ptr<wgpu::ComputePassEncoder> compute_pass_encoder_;

  // Output buffers for PreRead when output conversion is done. They are added
  // by GpuIOBufferWebGpu constructor when mappable buffer is not nullptr.
  // Key is the GpuIOBufferWebGpu instance pointer. The actual instance is owned
  // by the caller of CreateIOBufferWithSize().
  absl::flat_hash_set<GpuIOBufferWebGpu*> output_buffers_;
};

class GpuInferenceContextWebGpu : public GpuInferenceContext {
 public:
  explicit GpuInferenceContextWebGpu(
      GpuBackendWebGpu* backend,
      ::ml_drift::webgpu::MemoryManager* memory_manager,
      const ::ml_drift::CreateGpuModelInfo& create_info,
      int num_steps_of_command_buffer_preparations);
  ~GpuInferenceContextWebGpu() override;

  // Implementation of GpuInferenceContext.
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> GetSpatialTensor(
      ::ml_drift::ValueId id) override;
  absl::Status BindSpatialTensor(::ml_drift::ValueId id,
                                 ::ml_drift::GpuSpatialTensor* tensor) override;
  absl::Status WriteDataToWeightTensor(::ml_drift::ValueId id,
                                       absl::Span<const uint8_t> data) override;
  absl::Status ReadWeightTensorToDescriptor(
      ::ml_drift::ValueId id, ::ml_drift::TensorDescriptor& desc) override;
  absl::Status UploadWeightsOnWeb(
      weight_loader::WeightLoader* weight_loader,
      const ::ml_drift::GpuModel& gpu_model,
      const absl::flat_hash_map<::ml_drift::ValueId, ::ml_drift::ValueId>&
          io_mapping,
      const absl::flat_hash_map<::ml_drift::ValueId, uint32_t>&
          weight_id_to_external_buffer_id,
      std::vector<::ml_drift::WeightsManager::UploadWeightsInfo>& upload_infos)
      override;
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

  // Underlying WebGPU inference context.
  ::ml_drift::webgpu::InferenceContext& wgpu_ctx() { return *ctx_; }
  const ::ml_drift::webgpu::InferenceContext& wgpu_ctx() const { return *ctx_; }

  GpuBackendWebGpu* backend() const { return backend_; };

 private:
  // Prepares command buffers for Dispatch() call.
  absl::Status PrepareCommandBuffers(
      std::vector<wgpu::CommandBuffer>& command_buffers,
      bool submit_command_buffers);
  // Updates command buffers from next_command_buffers_ which would already been
  // prepared by a thread scheduled in the previous Dispatch() call.
  // If next_command_buffers_ is not available, i.e. at the first Dispatch()
  // call, it just calls PrepareCommandBuffers().
  absl::Status PrepareCommandBuffersFromCached(
      std::vector<wgpu::CommandBuffer>& command_buffers);

  GpuBackendWebGpu* const backend_;
  std::unique_ptr<::ml_drift::webgpu::InferenceContext> ctx_;
  std::vector<std::shared_ptr<absl::Notification>> async_upload_notifications_;

  // Command buffers prepared in advance for the next Dispatch() call.
  std::vector<std::vector<wgpu::CommandBuffer>> next_command_buffers_;
  int num_nodes_per_command_encoder_;
  // Index of next_command_buffers_ to use and prepare when
  // next_command_buffers_.size() > 1.
  int next_command_buffers_index_ = 0;
  // Whether next_command_buffers_ is ready to use. It's set to true when
  // all next_command_buffers_ are prepared.
  bool next_command_buffers_ready_ = false;
  std::unique_ptr<std::thread> next_command_buffers_thread_;

  // Static member var indicating the last context which built the command
  // buffers. It's used to invalidate the next_command_buffers_ if it is not
  // the same as the current context. For example, during multiple conversions
  // with a LLM, prefill inference starts after decode and it would have command
  // buffers prepared by the previous prefill inference which must be
  // invalidated since the gpu resource binding would be changed.
  // No synchronization is needed as the inference context is supposed to be
  // running on a single thread.
  static GpuInferenceContextWebGpu* last_ctx_;
};

class GpuTensorWrapperWebGpu : public GpuTensorWrapper {
 public:
  explicit GpuTensorWrapperWebGpu(::ml_drift::webgpu::SpatialTensor* tensor)
      : tensor_(tensor) {}
  ~GpuTensorWrapperWebGpu() override = default;

  // Implementation of GpuTensorWrapper.
  ::ml_drift::GpuSpatialTensor& Get() override { return *tensor_; }

 private:
  ::ml_drift::webgpu::SpatialTensor* const tensor_;
};

class GpuIOBufferWebGpu : public GpuIOBuffer {
 public:
  explicit GpuIOBufferWebGpu(
      GpuBackendWebGpu* backend, ::ml_drift::webgpu::Buffer&& buffer,
      std::unique_ptr<::ml_drift::webgpu::Buffer> mappable_buffer = nullptr);
  ~GpuIOBufferWebGpu() override;

  // Implementation of GpuIOBuffer.
  absl::Status Read(absl::Span<uint8_t> data) override;
  absl::Status Write(absl::Span<const uint8_t> data) override;

  // Prepares the buffer for read.
  absl::Status PreRead(wgpu::CommandEncoder& command_encoder);

  // Underlying immutable WebGPU buffer.
  const ::ml_drift::webgpu::Buffer& wgpu_buffer() const { return buffer_; }
  // Underlying mutable WebGPU buffer.
  ::ml_drift::webgpu::Buffer& wgpu_buffer() { return buffer_; }
  // Underlying mappable WebGPU buffer.
  ::ml_drift::webgpu::Buffer* wgpu_mappable_buffer() const {
    return mappable_buffer_.get();
  }

 private:
  GpuBackendWebGpu* const backend_;
  ::ml_drift::webgpu::Buffer buffer_;
  std::unique_ptr<::ml_drift::webgpu::Buffer> mappable_buffer_;
};

class Tensor2BufferConverterWebGpu : public Tensor2BufferConverter {
 public:
  explicit Tensor2BufferConverterWebGpu(
      GpuBackendWebGpu* backend,
      std::unique_ptr<::ml_drift::webgpu::TensorToBHWCBufferConverter>
          converter);
  ~Tensor2BufferConverterWebGpu() override = default;

  // Implementation of Tensor2BufferConverter.
  absl::Status Convert(::ml_drift::GpuSpatialTensor& src_tensor,
                       GpuIOBuffer& dst_buffer) override;

 private:
  GpuBackendWebGpu* const backend_;
  const std::unique_ptr<::ml_drift::webgpu::TensorToBHWCBufferConverter>
      converter_;
};

class Buffer2TensorConverterWebGpu : public Buffer2TensorConverter {
 public:
  explicit Buffer2TensorConverterWebGpu(
      GpuBackendWebGpu* backend,
      std::unique_ptr<::ml_drift::webgpu::BHWCBufferToTensorConverter>
          converter);
  ~Buffer2TensorConverterWebGpu() override = default;

  // Implementation of Buffer2TensorConverter.
  absl::Status Convert(GpuIOBuffer& src_buffer,
                       ::ml_drift::GpuSpatialTensor& dst_tensor) override;

 private:
  GpuBackendWebGpu* const backend_;
  const std::unique_ptr<::ml_drift::webgpu::BHWCBufferToTensorConverter>
      converter_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_WEBGPU_H_
