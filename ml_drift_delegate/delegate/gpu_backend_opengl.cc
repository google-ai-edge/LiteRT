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

#include "ml_drift_delegate/delegate/gpu_backend_opengl.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/die_if_null.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
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
#include "ml_drift/pelong/converter.h"  // from @ml_drift
#include "ml_drift/pelong/egl_environment.h"  // from @ml_drift
#include "ml_drift/pelong/gl_buffer.h"  // from @ml_drift
#include "ml_drift/pelong/gl_inference_context.h"  // from @ml_drift
#include "ml_drift/pelong/gl_spatial_tensor.h"  // from @ml_drift
// clang-format off
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
// clang-format on
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

constexpr absl::string_view kBackendName = "OpenGL";

// Serialization data prefix. FYI, Jet uses "gpuv2_data_".
constexpr absl::string_view kSerializedDataPrefix = "gpuv3_data_";

}  // namespace

GpuBackendOpenGl::GpuBackendOpenGl(::ml_drift::gl::EglEnvironment* env,
                                   const LiteRtRuntimeContext* runtime_context)
    : env_(env), runtime_context_(ABSL_DIE_IF_NULL(runtime_context)) {}

GpuBackendOpenGl::GpuBackendOpenGl(
    std::unique_ptr<::ml_drift::gl::EglEnvironment> env,
    const LiteRtRuntimeContext* runtime_context)
    : env_owned_(std::move(env)),
      env_(env_owned_.get()),
      runtime_context_(runtime_context) {}

absl::string_view GpuBackendOpenGl::GetBackendName() { return kBackendName; }

absl::string_view GpuBackendOpenGl::GetSerializedDataPrefix() {
  return kSerializedDataPrefix;
}

absl::StatusOr<::ml_drift::GpuInfo> GpuBackendOpenGl::GetInfo() {
  return env_->gpu_info();
}

absl::StatusOr<::ml_drift::TensorStorageType>
GpuBackendOpenGl::GetFastestStorageType() {
  return ::ml_drift::pelong::GetFastestStorageType(env_->gpu_info());
}

absl::StatusOr<GpuMemoryHandle> GpuBackendOpenGl::GetGpuMemoryAllocated(
    const GpuTensorBufferPtr& tensor_buffer) {
#if LITERT_HAS_OPENGL_SUPPORT
  LiteRtGLenum target;
  LiteRtGLuint id;
  size_t size;
  size_t offset;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_gl_buffer(
      tensor_buffer.get(), &target, &id, &size, &offset));
  return reinterpret_cast<GpuMemoryHandle>(id);
#else
  return absl::UnimplementedError(
      "GetGpuMemoryAllocated requires OpenGL support.");
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

absl::StatusOr<GpuEventHandle> GpuBackendOpenGl::GetGpuEventAssociated(
    const GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("GetGpuEventAssociated is not implemented.");
}

absl::Status GpuBackendOpenGl::AssociateGpuEvent(
    GpuEventHandle event, LiteRtEnvironment env,
    GpuTensorBufferPtr& tensor_buffer) {
  return absl::UnimplementedError("AssociateGpuEvent is not implemented.");
}

absl::Status GpuBackendOpenGl::WaitForCompletion() {
  glFlush();
  glFinish();
  return absl::OkStatus();
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendOpenGl::GetGpuBufferRequirements(
    ::ml_drift::TensorStorageType used_storage_type,
    ::ml_drift::DataType data_type) {
  return absl::UnimplementedError(
      "OpenGL Accelerator doesn't support ExternalTensors.");
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendOpenGl::GetGpuBufferRequirementsForNonExternalTensors() {
  return GpuBufferRequirements{
      .buffer_types = {kLiteRtTensorBufferTypeGlBuffer},
      // No strides for packed buffer.
      .strides = {0},
  };
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendOpenGl::CreateInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
    bool may_share_memory_manager) {
  auto ctx = std::make_unique<GpuInferenceContextOpenGl>(env_);
  ABSL_RETURN_IF_ERROR(ctx->gl_ctx().InitFromGpuModel(create_info, &gpu_model,
                                                      serialized_model));
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendOpenGl::RestoreInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const absl::Span<const uint8_t> serialized_model) {
  auto ctx = std::make_unique<GpuInferenceContextOpenGl>(env_);
  ABSL_RETURN_IF_ERROR(ctx->gl_ctx().RestoreDeserialized(
      serialized_model,
      const_cast<::ml_drift::CreateGpuModelInfo*>(&create_info)));
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
GpuBackendOpenGl::CreateSharedMemoryManager(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GraphFloat32& graph, TfLiteContext* context,
    MlDriftDelegateData& delegate_data,
    ::ml_drift::SerializationWeightCache* serialization_cache) {
  return absl::UnimplementedError(
      "CreateSharedMemoryManager is not implemented.");
}

absl::StatusOr<std::shared_ptr<::ml_drift::WeightsManager>>
GpuBackendOpenGl::CreateWeightsManager() {
  return std::make_shared<::ml_drift::WeightsManager>();
}

absl::StatusOr<std::vector<
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
GpuBackendOpenGl::GetBatchesForWeightsPreparation(
    ::ml_drift::WeightsManager* weights_manager,
    size_t total_shared_tensor_size) {
  return absl::UnimplementedError(
      "GetBatchesForWeightsPreparation is not implemented.");
}

absl::StatusOr<
    absl::flat_hash_map<::ml_drift::ValueId,
                        std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendOpenGl::PrepareWeightsInBatch(
    ::ml_drift::WeightsManager* weights_manager,
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&
        op_infos) {
  return absl::UnimplementedError("PrepareWeightsInBatch is not implemented.");
}

absl::StatusOr<absl::flat_hash_map<
    ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendOpenGl::PrepareWeightsInBatches(
    ::ml_drift::WeightsManager* weights_manager,
    size_t total_shared_tensor_size) {
  return absl::UnimplementedError(
      "PrepareWeightsInBatches is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuTensorWrapper>>
GpuBackendOpenGl::CreateTensorWrapper(const ::ml_drift::TensorDescriptor& desc,
                                      GpuMemoryHandle gpu_memory) {
  auto gl_tensor = std::make_unique<GpuTensorWrapperOpenGl>();
  ABSL_RETURN_IF_ERROR(::ml_drift::pelong::CreateTensorShared(
      static_cast<GLuint>(reinterpret_cast<uintptr_t>(gpu_memory)), desc,
      &gl_tensor->gl_tensor()));
  return std::move(gl_tensor);
}

absl::Status GpuBackendOpenGl::ReadSpatialTensorToDescriptor(
    ::ml_drift::GpuSpatialTensor& tensor, ::ml_drift::TensorDescriptor& desc) {
  return absl::UnimplementedError(
      "ReadSpatialTensorToDescriptor is not implemented");
}

absl::Status GpuBackendOpenGl::UpdateSpatialTensor(
    ::ml_drift::GpuSpatialTensor* tensor,
    const ::ml_drift::TensorDescriptor& desc, size_t page_adjusted_offset,
    ReleaseDataCallback release_data_callback) {
  return absl::UnimplementedError("UpdateSpatialTensor is not implemented.");
}

absl::Status GpuBackendOpenGl::ReleaseSpatialTensorMemory(
    ::ml_drift::GpuSpatialTensor* tensor) {
  return absl::UnimplementedError(
      "ReleaseSpatialTensorMemory is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>> GpuBackendOpenGl::CreateIOBuffer(
    GpuMemoryHandle gpu_memory) {
  ::ml_drift::pelong::GlBuffer gl_buffer;
  ::ml_drift::pelong::CreateSharedSSBOBuffer(
      static_cast<GLuint>(reinterpret_cast<uintptr_t>(gpu_memory)), &gl_buffer);
  return std::make_unique<GpuIOBufferOpenGl>(env_, std::move(gl_buffer));
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>>
GpuBackendOpenGl::CreateIOBufferWithSize(::ml_drift::DataType data_type,
                                         size_t size, bool input) {
  ::ml_drift::pelong::GlBuffer gl_buffer;
  ABSL_RETURN_IF_ERROR(
      ::ml_drift::pelong::CreateReadWriteBuffer(size, &gl_buffer));
  return std::make_unique<GpuIOBufferOpenGl>(env_, std::move(gl_buffer));
}

absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
GpuBackendOpenGl::CreateTensor2BufferConverter(
    const ::ml_drift::TensorDescriptor& src_desc,
    const ::ml_drift::BufferDescriptor& dst_desc) {
  auto converter =
      std::make_unique<::ml_drift::pelong::TensorToBHWCBufferConverter>();
  ABSL_RETURN_IF_ERROR(converter->Init(env_->gpu_info(), src_desc, dst_desc));
  return std::make_unique<Tensor2BufferConverterOpenGl>(env_,
                                                        std::move(converter));
}

absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
GpuBackendOpenGl::CreateBuffer2TensorConverter(
    const ::ml_drift::BufferDescriptor& src_desc,
    const ::ml_drift::TensorDescriptor& dst_desc) {
  auto converter =
      std::make_unique<::ml_drift::pelong::BHWCBufferToTensorConverter>();
  ABSL_RETURN_IF_ERROR(converter->Init(env_->gpu_info(), src_desc, dst_desc));
  return std::make_unique<Buffer2TensorConverterOpenGl>(env_,
                                                        std::move(converter));
}

GpuInferenceContextOpenGl::GpuInferenceContextOpenGl(
    ::ml_drift::gl::EglEnvironment* env)
    : env_(env) {}

absl::StatusOr<::ml_drift::GpuSpatialTensor*>
GpuInferenceContextOpenGl::GetSpatialTensor(::ml_drift::ValueId id) {
  return ctx_.GetTensor(id);
}

absl::Status GpuInferenceContextOpenGl::BindSpatialTensor(
    ::ml_drift::ValueId id, ::ml_drift::GpuSpatialTensor* tensor) {
  return absl::UnimplementedError("BindSpatialTensor is not implemented.");
}

absl::Status GpuInferenceContextOpenGl::WriteDataToWeightTensor(
    ::ml_drift::ValueId id, absl::Span<const uint8_t> data) {
  return absl::UnimplementedError(
      "WriteDataToWeightTensor is not implemented.");
}

absl::Status GpuInferenceContextOpenGl::ReadWeightTensorToDescriptor(
    ::ml_drift::ValueId id, ::ml_drift::TensorDescriptor& desc) {
  return absl::UnimplementedError(
      "ReadWeightTensorToDescriptor is not implemented.");
}

absl::Status GpuInferenceContextOpenGl::Dispatch() { return ctx_.AddToQueue(); }

absl::StatusOr<GpuEventHandle>
GpuInferenceContextOpenGl::GetPreDispatchEvent() {
  return absl::UnimplementedError("GetPreDispatchEvent is not implemented.");
}

absl::StatusOr<GpuEventHandle> GpuInferenceContextOpenGl::GetPostDispatchEvent(
    bool is_async_execution_mode) {
  return absl::UnimplementedError("GetPostDispatchEvent is not implemented.");
}

absl::Status GpuInferenceContextOpenGl::WaitForEventsCompleted(
    absl::Span<GpuEventHandle> events, bool force_sync) {
  return absl::UnimplementedError("WaitForEventsCompleted is not implemented.");
}

absl::Status GpuInferenceContextOpenGl::PreConvert(bool input) {
  return absl::OkStatus();
}

absl::Status GpuInferenceContextOpenGl::PostConvert(bool input) {
  return absl::OkStatus();
}

absl::Status GpuInferenceContextOpenGl::Profile(
    ::ml_drift::ProfilingInfo& profiling_info) {
  return ctx_.Profile(&profiling_info);
}

absl::StatusOr<size_t>
GpuInferenceContextOpenGl::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  return ctx_.GetSizeOfMemoryAllocatedForIntermediateTensors();
}

absl::StatusOr<size_t>
GpuInferenceContextOpenGl::GetSizeOfMemoryAllocatedForConstantTensors() const {
  return absl::UnimplementedError(
      "GetSizeOfMemoryAllocatedForConstantTensors is not implemented.");
}

absl::StatusOr<size_t>
GpuInferenceContextOpenGl::GetSizeOfMemoryAllocatedForExternalTensors() const {
  return absl::UnimplementedError(
      "GetSizeOfMemoryAllocatedForExternalTensors is not implemented.");
}

absl::Status GpuInferenceContextOpenGl::ReportMemoryBenchmarkIfEnabled(
    const ::ml_drift::CreateGpuModelInfo& create_info) {
#ifdef ML_DRIFT_MEM_STATS
  std::cout << ml_drift_delegate::GetMemoryBenchmarkReport(ctx_, create_info)
            << std::endl;
#endif
  return absl::OkStatus();
}

absl::Status GpuInferenceContextOpenGl::SetCommandBufferHint(
    int num_nodes_per_command_encoder) {
  return absl::UnimplementedError("SetCommandBufferHint is not implemented.");
}

GpuIOBufferOpenGl::GpuIOBufferOpenGl(::ml_drift::gl::EglEnvironment* env,
                                     ::ml_drift::pelong::GlBuffer&& buffer)
    : env_(env), buffer_(std::move(buffer)) {}

absl::Status GpuIOBufferOpenGl::Read(absl::Span<uint8_t> data) {
  std::vector<uint8_t> data_vector(data.begin(), data.end());
  return buffer_.ReadData(&data_vector);
}

absl::Status GpuIOBufferOpenGl::Write(absl::Span<const uint8_t> data) {
  // TODO(terryheo): Implement this.
  return absl::UnimplementedError("Write is not implemented.");
}

Tensor2BufferConverterOpenGl::Tensor2BufferConverterOpenGl(
    ::ml_drift::gl::EglEnvironment* env,
    std::unique_ptr<::ml_drift::pelong::TensorToBHWCBufferConverter> converter)
    : env_(env), converter_(std::move(converter)) {}

absl::Status Tensor2BufferConverterOpenGl::Convert(
    ::ml_drift::GpuSpatialTensor& src_tensor, GpuIOBuffer& dst_buffer) {
  return converter_->Convert(
      static_cast<::ml_drift::pelong::GlSpatialTensor*>(&src_tensor),
      &(static_cast<GpuIOBufferOpenGl&>(dst_buffer).gl_buffer()));
}

Buffer2TensorConverterOpenGl::Buffer2TensorConverterOpenGl(
    ::ml_drift::gl::EglEnvironment* env,
    std::unique_ptr<::ml_drift::pelong::BHWCBufferToTensorConverter> converter)
    : env_(env), converter_(std::move(converter)) {}

absl::Status Buffer2TensorConverterOpenGl::Convert(
    GpuIOBuffer& src_buffer, ::ml_drift::GpuSpatialTensor& dst_tensor) {
  return converter_->Convert(
      &(static_cast<GpuIOBufferOpenGl&>(src_buffer).gl_buffer()),
      static_cast<::ml_drift::pelong::GlSpatialTensor*>(&dst_tensor));
}

}  // namespace litert::ml_drift
