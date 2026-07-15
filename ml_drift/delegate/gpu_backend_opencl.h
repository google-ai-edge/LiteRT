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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_OPENCL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_OPENCL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/buffer.h"  // from @ml_drift
#include "ml_drift/cl/converter.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "third_party/odml/infra/ml_drift_delegate/shared_memory_manager.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_data.h"
#include "third_party/odml/litert/ml_drift/delegate/gpu_backend.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

// GpuBackend for OpenCL.
class GpuBackendOpenCl : public GpuBackend {
 public:
  explicit GpuBackendOpenCl(::ml_drift::cl::Environment* env);
  explicit GpuBackendOpenCl(std::unique_ptr<::ml_drift::cl::Environment> env);
  ~GpuBackendOpenCl() override = default;

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

  // Underlying OpenCL environment.
  ::ml_drift::cl::Environment* cl_env() const { return env_; }

  ::ml_drift::cl::MemoryManager& memory_manager() { return memory_manager_; }

 private:
  std::unique_ptr<::ml_drift::cl::Environment> env_owned_;
  ::ml_drift::cl::Environment* const env_;
  ::ml_drift::cl::MemoryManager memory_manager_;
};

class GpuInferenceContextOpenCl : public GpuInferenceContext {
 public:
  explicit GpuInferenceContextOpenCl(
      GpuBackendOpenCl* backend,
      ::ml_drift::cl::MemoryManager* memory_manager = nullptr);
  ~GpuInferenceContextOpenCl() override = default;

  // Implementation of GpuInferenceContext.
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> GetSpatialTensor(
      ::ml_drift::ValueId id) override;
  absl::Status BindSpatialTensor(::ml_drift::ValueId id,
                                 ::ml_drift::GpuSpatialTensor* tensor) override;
  absl::Status UploadWeightsOnWeb(
      weight_loader::WeightLoader* weight_loader,
      const ::ml_drift::GpuModel& gpu_model,
      const absl::flat_hash_map<::ml_drift::ValueId, ::ml_drift::ValueId>&
          io_mapping,
      const absl::flat_hash_map<::ml_drift::ValueId, uint32_t>&
          weight_id_to_external_buffer_id,
      std::vector<::ml_drift::WeightsManager::UploadWeightsInfo>& upload_infos)
      override {
    return absl::UnimplementedError("Not implemented for this backend.");
  }
  absl::Status WriteDataToWeightTensor(
      ::ml_drift::ValueId id, absl::Span<const uint8_t> data) override;
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

  // Underlying OpenCL backend.
  GpuBackendOpenCl* backend() const { return backend_; }
  // Underlying OpenCL environment.
  ::ml_drift::cl::Environment* cl_env() const { return backend_->cl_env(); }
  // Underlying OpenCL inference context.
  ::ml_drift::cl::InferenceContext& cl_ctx() { return *ctx_; }

 private:
  GpuBackendOpenCl* const backend_;
  std::unique_ptr<::ml_drift::cl::InferenceContext> ctx_;
};

class GpuTensorWrapperOpenCl : public GpuTensorWrapper {
 public:
  GpuTensorWrapperOpenCl() = default;
  ~GpuTensorWrapperOpenCl() override = default;

  // Implementation of GpuTensorWrapper.
  ::ml_drift::GpuSpatialTensor& Get() override { return tensor_; }

  // Underlying OpenCL tensor.
  ::ml_drift::cl::Tensor& cl_tensor() { return tensor_; }
  const ::ml_drift::cl::Tensor& cl_tensor() const { return tensor_; }

 private:
  ::ml_drift::cl::Tensor tensor_;
};

class GpuIOBufferOpenCl : public GpuIOBuffer {
 public:
  explicit GpuIOBufferOpenCl(::ml_drift::cl::Environment* env,
                             ::ml_drift::cl::Buffer&& buffer);
  ~GpuIOBufferOpenCl() override = default;

  // Implementation of GpuIOBuffer.
  absl::Status Read(absl::Span<uint8_t> data) override;
  absl::Status Write(absl::Span<const uint8_t> data) override;

  // Underlying immutable OpenCL buffer.
  const ::ml_drift::cl::Buffer& cl_buffer() const { return buffer_; }
  // Underlying mutable OpenCL buffer.
  ::ml_drift::cl::Buffer& cl_buffer() { return buffer_; }

 private:
  ::ml_drift::cl::Environment* const env_;
  ::ml_drift::cl::Buffer buffer_;
};

class Tensor2BufferConverterOpenCl : public Tensor2BufferConverter {
 public:
  explicit Tensor2BufferConverterOpenCl(
      ::ml_drift::cl::Environment* env,
      std::unique_ptr<::ml_drift::cl::TensorToBHWCBufferConverter> converter);
  ~Tensor2BufferConverterOpenCl() override = default;

  // Implementation of Tensor2BufferConverter.
  absl::Status Convert(::ml_drift::GpuSpatialTensor& src_tensor,
                       GpuIOBuffer& dst_buffer) override;

 private:
  ::ml_drift::cl::Environment* const env_;
  const std::unique_ptr<::ml_drift::cl::TensorToBHWCBufferConverter> converter_;
};

class Buffer2TensorConverterOpenCl : public Buffer2TensorConverter {
 public:
  explicit Buffer2TensorConverterOpenCl(
      ::ml_drift::cl::Environment* env,
      std::unique_ptr<::ml_drift::cl::BHWCBufferToTensorConverter> converter);
  ~Buffer2TensorConverterOpenCl() override = default;

  // Implementation of Buffer2TensorConverter.
  absl::Status Convert(GpuIOBuffer& src_buffer,
                       ::ml_drift::GpuSpatialTensor& dst_tensor) override;

 private:
  ::ml_drift::cl::Environment* const env_;
  const std::unique_ptr<::ml_drift::cl::BHWCBufferToTensorConverter> converter_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_OPENCL_H_
