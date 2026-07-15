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

#include "ml_drift_delegate/delegate/gpu_backend_opencl.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/buffer.h"  // from @ml_drift
#include "ml_drift/cl/cl_operation.h"  // from @ml_drift
#include "ml_drift/cl/converter.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#include "ml_drift/cl/memory_manager.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/delegate_data_util.h"
#include "third_party/odml/infra/ml_drift_delegate/ml_drift_cl_benchmark_util.h"  // IWYU pragma: keep
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "third_party/odml/infra/ml_drift_delegate/shared_memory_manager.h"
#include "third_party/odml/infra/ml_drift_delegate/shared_memory_manager_cl.h"
#include "third_party/odml/infra/ml_drift_delegate/util.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include <CL/cl.h>
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

constexpr absl::string_view kBackendName = "OpenCL";

// Serialization data prefix. FYI, Jet uses "gpuv2_data_".
constexpr absl::string_view kSerializedDataPrefix = "gpuv3_data_";

}  // namespace

GpuBackendOpenCl::GpuBackendOpenCl(::ml_drift::cl::Environment* env)
    : env_(env) {}

GpuBackendOpenCl::GpuBackendOpenCl(
    std::unique_ptr<::ml_drift::cl::Environment> env)
    : env_owned_(std::move(env)), env_(env_owned_.get()) {}

absl::string_view GpuBackendOpenCl::GetBackendName() { return kBackendName; }

absl::string_view GpuBackendOpenCl::GetSerializedDataPrefix() {
  return kSerializedDataPrefix;
}

absl::StatusOr<::ml_drift::GpuInfo> GpuBackendOpenCl::GetInfo() {
  return env_->GetDevicePtr()->GetInfo();
}

absl::StatusOr<::ml_drift::TensorStorageType>
GpuBackendOpenCl::GetFastestStorageType() {
  return ::ml_drift::cl::GetFastestStorageType(env_->GetDevicePtr()->GetInfo());
}

absl::StatusOr<GpuMemoryHandle> GpuBackendOpenCl::GetGpuMemoryAllocated(
    const GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("GetGpuMemoryAllocated is not implemented.");
}

absl::StatusOr<GpuEventHandle> GpuBackendOpenCl::GetGpuEventAssociated(
    const GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("GetGpuEventAssociated is not implemented.");
}

absl::Status GpuBackendOpenCl::AssociateGpuEvent(
    GpuEventHandle event, LiteRtEnvironment env,
    GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("AssociateGpuEvent is not implemented.");
}

absl::Status GpuBackendOpenCl::WaitForCompletion() {
  return env_->queue()->WaitForCompletion();
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendOpenCl::GetGpuBufferRequirements(
    ::ml_drift::TensorStorageType used_storage_type,
    ::ml_drift::DataType data_type) {
  return absl::UnimplementedError(
      "GetGpuBufferRequirements is not implemented.");
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendOpenCl::GetGpuBufferRequirementsForNonExternalTensors() {
  return absl::UnimplementedError(
      "GetGpuBufferRequirementsForNonExternalTensors is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendOpenCl::CreateInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
    bool may_share_memory_manager) {
  auto ctx = std::make_unique<GpuInferenceContextOpenCl>(
      this, may_share_memory_manager ? &memory_manager_ : nullptr);
  RETURN_IF_ERROR(ctx->cl_ctx().InitFromGpuModel(create_info, &gpu_model, env_,
                                                 serialized_model));
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendOpenCl::RestoreInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const absl::Span<const uint8_t> serialized_model) {
  auto ctx =
      std::make_unique<GpuInferenceContextOpenCl>(this, &memory_manager_);
  RETURN_IF_ERROR(
      ctx->cl_ctx().RestoreDeserialized(serialized_model, env_, &create_info));
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
GpuBackendOpenCl::CreateSharedMemoryManager(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GraphFloat32& graph, TfLiteContext* context,
    MlDriftDelegateData& delegate_data,
    ::ml_drift::SerializationWeightCache* serialization_cache) {
  return ::ml_drift::MakeSharedMemoryManagerCl(
      *env_, create_info, graph, context,
      GetBufferIdToSpatialTensorMap(delegate_data),
      GetQuantParamIdToSpatialTensorMap(delegate_data),
      delegate_data.options->has_prepacked_external_tflite_tensors,
      serialization_cache,
      delegate_data.options->madvise_original_shared_tensors);
}

absl::StatusOr<std::shared_ptr<::ml_drift::WeightsManager>>
GpuBackendOpenCl::CreateWeightsManager() {
  return std::make_shared<::ml_drift::WeightsManager>();
}

absl::StatusOr<std::vector<
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
GpuBackendOpenCl::GetBatchesForWeightsPreparation(
    ::ml_drift::WeightsManager* weights_manager) {
  return absl::UnimplementedError(
      "GetBatchesForWeightsPreparation is not implemented.");
}

absl::StatusOr<absl::flat_hash_map<
    ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendOpenCl::PrepareWeightsInBatch(
    ::ml_drift::WeightsManager* weights_manager,
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&
        op_infos) {
  return absl::UnimplementedError("PrepareWeightsInBatch is not implemented.");
}

absl::StatusOr<absl::flat_hash_map<
    ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendOpenCl::PrepareWeightsInBatches(
    ::ml_drift::WeightsManager* weights_manager) {
  return absl::UnimplementedError(
      "PrepareWeightsInBatches is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuTensorWrapper>>
GpuBackendOpenCl::CreateTensorWrapper(const ::ml_drift::TensorDescriptor& desc,
                                      GpuMemoryHandle gpu_memory) {
  auto cl_tensor = std::make_unique<GpuTensorWrapperOpenCl>();
  RETURN_IF_ERROR(::ml_drift::cl::CreateTensorShared(
      env_->context(), static_cast<cl_mem>(gpu_memory), desc,
      &cl_tensor->cl_tensor()));
  return std::move(cl_tensor);
}

absl::Status GpuBackendOpenCl::ReadSpatialTensorToDescriptor(
    ::ml_drift::GpuSpatialTensor& tensor, ::ml_drift::TensorDescriptor& desc) {
  return absl::UnimplementedError(
      "ReadSpatialTensorToDescriptor is not implemented");
}

absl::Status GpuBackendOpenCl::UpdateSpatialTensor(
    ::ml_drift::GpuSpatialTensor* tensor,
    const ::ml_drift::TensorDescriptor& desc, size_t page_adjusted_offset,
    ::ml_drift_delegate::ReleaseDataCallback release_data_callback) {
  return absl::UnimplementedError("UpdateSpatialTensor is not implemented.");
}

absl::Status GpuBackendOpenCl::ReleaseSpatialTensorMemory(
    ::ml_drift::GpuSpatialTensor* tensor) {
  return absl::UnimplementedError(
      "ReleaseSpatialTensorMemory is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>> GpuBackendOpenCl::CreateIOBuffer(
    GpuMemoryHandle gpu_memory) {
  return std::make_unique<GpuIOBufferOpenCl>(
      env_,
      ::ml_drift::cl::CreateBufferShared(static_cast<cl_mem>(gpu_memory)));
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>>
GpuBackendOpenCl::CreateIOBufferWithSize(::ml_drift::DataType data_type,
                                         size_t size, bool input) {
  ::ml_drift::cl::Buffer cl_buffer;
  RETURN_IF_ERROR(::ml_drift::cl::CreateReadWriteBuffer(size, &env_->context(),
                                                        &cl_buffer));
  return std::make_unique<GpuIOBufferOpenCl>(env_, std::move(cl_buffer));
}

absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
GpuBackendOpenCl::CreateTensor2BufferConverter(
    const ::ml_drift::TensorDescriptor& src_desc,
    const ::ml_drift::BufferDescriptor& dst_desc) {
  auto converter =
      std::make_unique<::ml_drift::cl::TensorToBHWCBufferConverter>();
  RETURN_IF_ERROR(
      converter->InitExplicit(env_->GetDevicePtr(), &env_->context(),
                              env_->program_cache(), src_desc, dst_desc));
  return std::make_unique<Tensor2BufferConverterOpenCl>(env_,
                                                        std::move(converter));
}

absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
GpuBackendOpenCl::CreateBuffer2TensorConverter(
    const ::ml_drift::BufferDescriptor& src_desc,
    const ::ml_drift::TensorDescriptor& dst_desc) {
  auto converter =
      std::make_unique<::ml_drift::cl::BHWCBufferToTensorConverter>();
  RETURN_IF_ERROR(
      converter->InitExplicit(env_->GetDevicePtr(), &env_->context(),
                              env_->program_cache(), src_desc, dst_desc));
  return std::make_unique<Buffer2TensorConverterOpenCl>(env_,
                                                        std::move(converter));
}

GpuInferenceContextOpenCl::GpuInferenceContextOpenCl(
    GpuBackendOpenCl* backend, ::ml_drift::cl::MemoryManager* memory_manager)
    : backend_(backend),
      ctx_(memory_manager != nullptr
               ? std::make_unique<::ml_drift::cl::InferenceContext>(
                     memory_manager)
               : std::make_unique<::ml_drift::cl::InferenceContext>()) {}

absl::StatusOr<::ml_drift::GpuSpatialTensor*>
GpuInferenceContextOpenCl::GetSpatialTensor(::ml_drift::ValueId id) {
  return ctx_->GetTensor(id);
}

absl::Status GpuInferenceContextOpenCl::BindSpatialTensor(
    ::ml_drift::ValueId id, ::ml_drift::GpuSpatialTensor* tensor) {
  return ctx_->SetTensor(id, static_cast<::ml_drift::cl::Tensor*>(tensor));
}

absl::Status GpuInferenceContextOpenCl::WriteDataToWeightTensor(
    ::ml_drift::ValueId id, absl::Span<const uint8_t> data) {
  auto* cl_tensor = ctx_->GetTensor(id);
  // Mali GPU returns incorrect results when utilizing staging buffer to upload
  // data to GPU.
  ASSIGN_OR_RETURN(auto gpu_info, backend_->GetInfo());
  if (gpu_info.IsMali()) {
    return cl_tensor->WriteData(data.data(), cl_env()->queue(),
                                /*async=*/false);
  } else {
    return cl_tensor->WriteDataViaStaging(data.data(), cl_env()->queue(),
                                          &cl_env()->context());
  }
}

absl::Status GpuInferenceContextOpenCl::ReadWeightTensorToDescriptor(
    ::ml_drift::ValueId id, ::ml_drift::TensorDescriptor& desc) {
  auto* cl_tensor = ctx_->GetTensor(id);
  return cl_tensor->ToDescriptor(&desc, cl_env()->queue());
}

absl::Status GpuInferenceContextOpenCl::Dispatch() {
  return ctx_->AddToQueue(cl_env()->queue());
}

absl::StatusOr<GpuEventHandle>
GpuInferenceContextOpenCl::GetPreDispatchEvent() {
  return absl::UnimplementedError("GetPreDispatchEvent is not implemented.");
}

absl::StatusOr<GpuEventHandle> GpuInferenceContextOpenCl::GetPostDispatchEvent(
    bool is_async_execution_mode) {
  return absl::UnimplementedError("GetPostDispatchEvent is not implemented.");
}

absl::Status GpuInferenceContextOpenCl::WaitForEventsCompleted(
    absl::Span<GpuEventHandle> events, bool force_sync) {
  return absl::UnimplementedError("WaitForEventsCompleted is not implemented.");
}

absl::Status GpuInferenceContextOpenCl::PreConvert(bool input) {
  return absl::OkStatus();
}

absl::Status GpuInferenceContextOpenCl::PostConvert(bool input) {
  return absl::OkStatus();
}

absl::Status GpuInferenceContextOpenCl::Profile(
    ::ml_drift::ProfilingInfo& profiling_info) {
  return ctx_->Profile(cl_env()->profiling_queue(), &profiling_info);
}

absl::StatusOr<size_t>
GpuInferenceContextOpenCl::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  return ctx_->GetSizeOfMemoryAllocatedForIntermediateTensors();
}

absl::StatusOr<size_t>
GpuInferenceContextOpenCl::GetSizeOfMemoryAllocatedForConstantTensors() const {
  return ctx_->GetConstantTensorsSize();
}

absl::StatusOr<size_t>
GpuInferenceContextOpenCl::GetSizeOfMemoryAllocatedForExternalTensors() const {
  return ctx_->GetExternalTensorsSize();
}

absl::Status GpuInferenceContextOpenCl::ReportMemoryBenchmarkIfEnabled(
    const ::ml_drift::CreateGpuModelInfo& create_info) {
#ifdef ML_DRIFT_MEM_STATS
  std::cout << ml_drift_delegate::GetMemoryBenchmarkReport(*ctx_, create_info)
            << std::endl;
#endif
  return absl::OkStatus();
}

absl::Status GpuInferenceContextOpenCl::SetCommandBufferHint(
    int num_nodes_per_command_encoder) {
  return absl::UnimplementedError("SetCommandBufferHint is not implemented.");
}

GpuIOBufferOpenCl::GpuIOBufferOpenCl(::ml_drift::cl::Environment* env,
                                     ::ml_drift::cl::Buffer&& buffer)
    : env_(env), buffer_(std::move(buffer)) {}

absl::Status GpuIOBufferOpenCl::Read(absl::Span<uint8_t> data) {
  return env_->queue()->EnqueueReadBuffer(buffer_.GetMemoryPtr(), data.size(),
                                          data.data(), /*async=*/true);
}

absl::Status GpuIOBufferOpenCl::Write(absl::Span<const uint8_t> data) {
  return env_->queue()->EnqueueWriteBuffer(buffer_.GetMemoryPtr(), data.size(),
                                           data.data(), /*async=*/true);
}

Tensor2BufferConverterOpenCl::Tensor2BufferConverterOpenCl(
    ::ml_drift::cl::Environment* env,
    std::unique_ptr<::ml_drift::cl::TensorToBHWCBufferConverter> converter)
    : env_(env), converter_(std::move(converter)) {}

absl::Status Tensor2BufferConverterOpenCl::Convert(
    ::ml_drift::GpuSpatialTensor& src_tensor, GpuIOBuffer& dst_buffer) {
  return converter_->ConvertExplicit(
      env_->queue(), static_cast<::ml_drift::cl::Tensor*>(&src_tensor),
      &(static_cast<GpuIOBufferOpenCl&>(dst_buffer).cl_buffer()));
}

Buffer2TensorConverterOpenCl::Buffer2TensorConverterOpenCl(
    ::ml_drift::cl::Environment* env,
    std::unique_ptr<::ml_drift::cl::BHWCBufferToTensorConverter> converter)
    : env_(env), converter_(std::move(converter)) {}

absl::Status Buffer2TensorConverterOpenCl::Convert(
    GpuIOBuffer& src_buffer, ::ml_drift::GpuSpatialTensor& dst_tensor) {
  return converter_->ConvertExplicit(
      env_->queue(), &(static_cast<GpuIOBufferOpenCl&>(src_buffer).cl_buffer()),
      static_cast<::ml_drift::cl::Tensor*>(&dst_tensor));
}

}  // namespace litert::ml_drift
