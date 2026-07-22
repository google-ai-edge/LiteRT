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

#include "ml_drift_delegate/delegate/gpu_backend_vulkan.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>  // NOLINT (Open source code)
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/die_if_null.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
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
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/syrtis/buffer.h"  // from @ml_drift
#include "ml_drift/syrtis/command_buffer.h"  // from @ml_drift
#include "ml_drift/syrtis/converter.h"  // from @ml_drift
#include "ml_drift/syrtis/environment.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_inference_context.h"  // from @ml_drift
#include "ml_drift/syrtis/vulkan_spatial_tensor.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "ml_drift_delegate/delegate/delegate_data.h"
#include "ml_drift_delegate/delegate/delegate_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_vulkan.h"
#include "ml_drift_delegate/delegate/shared_vulkan_env.h"
#include "ml_drift_delegate/delegate/unowned_tensor_desc.h"
#include "tflite/c/common.h"

using ::ml_drift::syrtis::VulkanSpatialTensor;

namespace litert::ml_drift {
namespace {
constexpr absl::string_view kBackendName = "Vulkan";
constexpr absl::string_view kSerializedDataPrefix = "gpuv3_vulkan_data_";

constexpr int kDefaultMaxNumNodesPerCommandBuffer = 256;
}  // namespace

GpuBackendVulkan::GpuBackendVulkan(SharedVulkanEnv* env,
                                   const LiteRtRuntimeContext* runtime_context)
    : env_(env), runtime_context_(ABSL_DIE_IF_NULL(runtime_context)) {}

absl::string_view GpuBackendVulkan::GetBackendName() { return kBackendName; }

absl::string_view GpuBackendVulkan::GetSerializedDataPrefix() {
  return kSerializedDataPrefix;
}

absl::StatusOr<::ml_drift::GpuInfo> GpuBackendVulkan::GetInfo() {
  return env_->vulkan_env().GetInfo();
}

absl::StatusOr<::ml_drift::TensorStorageType>
GpuBackendVulkan::GetFastestStorageType() {
  return ::ml_drift::syrtis::GetFastestStorageType(
      env_->vulkan_env().GetInfo());
}

absl::StatusOr<GpuMemoryHandle> GpuBackendVulkan::GetGpuMemoryAllocated(
    const GpuTensorBufferPtr& tensor_buffer) {
  HwMemoryHandle hw_memory_handle;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_custom_tensor_buffer_handle(
          tensor_buffer.get(), &hw_memory_handle));
  return static_cast<GpuMemoryHandle>(hw_memory_handle);
}

absl::StatusOr<GpuEventHandle> GpuBackendVulkan::GetGpuEventAssociated(
    const GpuTensorBufferPtr& tensor_buffer) {
  bool has_event;
  LITERT_RETURN_IF_ERROR(runtime_context_->has_tensor_buffer_event(
      tensor_buffer.get(), &has_event));
  if (!has_event) {
    return absl::NotFoundError("Tensor buffer does not have an event.");
  }

  LiteRtEvent event;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_event(tensor_buffer.get(), &event));
  LiteRtEventType event_type;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_event_event_type(event, &event_type));
  if (event_type != LiteRtEventTypeCustom) {
    return absl::InternalError("Tensor buffer has a non-custom event.");
  }
  LiteRtCustomEvent custom_event;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_custom_event(event, &custom_event));
  return custom_event;
}

absl::Status GpuBackendVulkan::AssociateGpuEvent(
    GpuEventHandle event, LiteRtEnvironment env,
    GpuTensorBufferPtr& tensor_buffer) {
  LiteRtEvent liter_event;
  LITERT_RETURN_IF_ERROR(runtime_context_->create_managed_event(
      env, LiteRtEventTypeCustom, &liter_event));
  LITERT_RETURN_IF_ERROR(runtime_context_->set_custom_event(
      liter_event, reinterpret_cast<LiteRtCustomEvent>(event)));
  LITERT_RETURN_IF_ERROR(runtime_context_->set_tensor_buffer_event(
      tensor_buffer.get(), liter_event));
  return absl::OkStatus();
}

absl::Status GpuBackendVulkan::WaitForCompletion() {
  return env_->SubmitCommandBuffer(/*wait_for_completion=*/true);
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendVulkan::GetGpuBufferRequirements(
    ::ml_drift::TensorStorageType used_storage_type,
    ::ml_drift::DataType data_type) {
  GpuBufferRequirements requirements;
  if (used_storage_type == ::ml_drift::TensorStorageType::TEXTURE_2D) {
    if (data_type == ::ml_drift::DataType::FLOAT16) {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeVulkanTextureFp16);
    } else {
      requirements.buffer_types.push_back(kLiteRtTensorBufferTypeVulkanTexture);
    }
  } else if (used_storage_type == ::ml_drift::TensorStorageType::IMAGE_BUFFER) {
    if (data_type == ::ml_drift::DataType::FLOAT16) {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeVulkanImageBufferFp16);
    } else {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeVulkanImageBuffer);
    }
  } else {
    if (data_type == ::ml_drift::DataType::FLOAT16) {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeVulkanBufferFp16);
    } else {
      requirements.buffer_types.push_back(kLiteRtTensorBufferTypeVulkanBuffer);
    }
  }
  // MLD uses PHWC4, 16 bytes strides.
  requirements.strides.push_back(16);
  return std::move(requirements);
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendVulkan::GetGpuBufferRequirementsForNonExternalTensors() {
  return GpuBufferRequirements{
      .buffer_types = {kLiteRtTensorBufferTypeVulkanBufferPacked},
      // No strides for packed buffer.
      .strides = {0},
  };
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendVulkan::CreateInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
    bool may_share_memory_manager) {
  auto ctx = std::make_unique<GpuInferenceContextVulkan>(
      this, num_steps_of_command_buffer_preparations_);

  ::ml_drift::syrtis::CreateInferenceInfo inference_info;
  inference_info.options.tuning_type =
      create_info.hints.Check(::ml_drift::ModelHints::kFastTuning)
          ? ::ml_drift::TuningType::kFast
          : ::ml_drift::TuningType::kExhaustive;
  // NOLINTBEGIN(misc-include-cleaner) to avoid shaderc.h inclusion.
  inference_info.options.optimization_level =
      optimize_shader_compilation_ ? shaderc_optimization_level_performance
                                   : shaderc_optimization_level_zero;
  // NOLINTEND(misc-include-cleaner)
  inference_info.external_tensors.immutable_tensors =
      create_info.external_immutable_tensors;
  inference_info.external_tensors.mutable_tensors =
      create_info.external_mutable_tensors;
  RETURN_IF_ERROR(ctx->vk_ctx().InitFromGpuModel(
      inference_info, &gpu_model, &env_->vulkan_env(), serialized_model));
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendVulkan::RestoreInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const absl::Span<const uint8_t> serialized_model) {
  auto ctx = std::make_unique<GpuInferenceContextVulkan>(
      this, num_steps_of_command_buffer_preparations_);
  RETURN_IF_ERROR(ctx->vk_ctx().RestoreDeserialized(
      serialized_model, &env_->vulkan_env(), &create_info));
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
GpuBackendVulkan::CreateSharedMemoryManager(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GraphFloat32& graph, TfLiteContext* context,
    MlDriftDelegateData& delegate_data,
    ::ml_drift::SerializationWeightCache* serialization_cache) {
  return ::ml_drift::MakeSharedMemoryManagerVulkan(
      env_, create_info, graph, context,
      GetBufferIdToSpatialTensorMap(delegate_data),
      GetQuantParamIdToSpatialTensorMap(delegate_data),
      delegate_data.options->has_prepacked_external_tflite_tensors,
      serialization_cache,
      delegate_data.options->madvise_original_shared_tensors);
}

absl::StatusOr<std::shared_ptr<::ml_drift::WeightsManager>>
GpuBackendVulkan::CreateWeightsManager() {
  return std::make_shared<::ml_drift::WeightsManager>();
}

absl::StatusOr<std::vector<
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>>>
GpuBackendVulkan::GetBatchesForWeightsPreparation(
    ::ml_drift::WeightsManager* weights_manager,
    size_t total_shared_tensor_size) {
  return absl::UnimplementedError(
      "GetBatchesForWeightsPreparation is not implemented.");
}

absl::StatusOr<absl::flat_hash_map<
    ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendVulkan::PrepareWeightsInBatch(
    ::ml_drift::WeightsManager* weights_manager,
    std::vector<::ml_drift::WeightsManager::WeightsPrepOperationInfo>&
        op_infos) {
  return absl::UnimplementedError("PrepareWeightsInBatch is not implemented.");
}

absl::StatusOr<absl::flat_hash_map<
    ::ml_drift::ValueId, std::unique_ptr<::ml_drift::GpuSpatialTensor>>>
GpuBackendVulkan::PrepareWeightsInBatches(
    ::ml_drift::WeightsManager* weights_manager,
    size_t total_shared_tensor_size) {
  return absl::UnimplementedError(
      "PrepareWeightsInBatches is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuTensorWrapper>>
GpuBackendVulkan::CreateTensorWrapper(const ::ml_drift::TensorDescriptor& desc,
                                      GpuMemoryHandle gpu_memory) {
  return std::make_unique<GpuTensorWrapperVulkan>(
      reinterpret_cast<::ml_drift::syrtis::VulkanSpatialTensor*>(gpu_memory));
}

absl::Status GpuBackendVulkan::ReadSpatialTensorToDescriptor(
    ::ml_drift::GpuSpatialTensor& tensor, ::ml_drift::TensorDescriptor& desc) {
  return absl::UnimplementedError(
      "ReadSpatialTensorToDescriptor is not implemented");
}

absl::Status GpuBackendVulkan::UpdateSpatialTensor(
    ::ml_drift::GpuSpatialTensor* tensor,
    const ::ml_drift::TensorDescriptor& desc, size_t page_adjusted_offset,
    ReleaseDataCallback release_data_callback) {
  return absl::UnimplementedError("UpdateSpatialTensor is not implemented.");
}

absl::Status GpuBackendVulkan::ReleaseSpatialTensorMemory(
    ::ml_drift::GpuSpatialTensor* tensor) {
  return absl::UnimplementedError(
      "ReleaseSpatialTensorMemory is not implemented.");
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>> GpuBackendVulkan::CreateIOBuffer(
    GpuMemoryHandle gpu_memory) {
  auto* vk_tensor =
      reinterpret_cast<::ml_drift::syrtis::VulkanSpatialTensor*>(gpu_memory);
  ::ml_drift::syrtis::BufferInfo buffer_info{
      .buffer = vk_tensor->GetBufferHandle(),
      .memory = vk_tensor->GetMemoryHandle(),
      .memory_size = vk_tensor->GetMemorySizeInBytes()};
  ::ml_drift::syrtis::Buffer buffer(env_->vulkan_env().GetDevice(), buffer_info,
                                    /*owns=*/false);
  return std::make_unique<GpuIOBufferVulkan>(std::move(buffer));
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>>
GpuBackendVulkan::CreateIOBufferWithSize(::ml_drift::DataType data_type,
                                         size_t size, bool input) {
  return absl::UnimplementedError("CreateIOBufferWithSize is not implemented.");
}

absl::StatusOr<std::unique_ptr<Tensor2BufferConverter>>
GpuBackendVulkan::CreateTensor2BufferConverter(
    const ::ml_drift::TensorDescriptor& src_desc,
    const ::ml_drift::BufferDescriptor& dst_desc) {
  auto converter =
      std::make_unique<::ml_drift::syrtis::TensorToBHWCBufferConverter>();
  RETURN_IF_ERROR(converter->Init(&env_->vulkan_env(), src_desc, dst_desc));
  return std::make_unique<Tensor2BufferConverterVulkan>(this,
                                                        std::move(converter));
}

absl::StatusOr<std::unique_ptr<Buffer2TensorConverter>>
GpuBackendVulkan::CreateBuffer2TensorConverter(
    const ::ml_drift::BufferDescriptor& src_desc,
    const ::ml_drift::TensorDescriptor& dst_desc) {
  auto converter =
      std::make_unique<::ml_drift::syrtis::BHWCBufferToTensorConverter>();
  RETURN_IF_ERROR(converter->Init(&env_->vulkan_env(), src_desc, dst_desc));
  return std::make_unique<Buffer2TensorConverterVulkan>(this,
                                                        std::move(converter));
}

VulkanCustomEvent::VulkanCustomEvent(GpuBackendVulkan* backend)
    : backend_(backend), ref_count_(1) {
  Retain = RetainStatic;
  Release = ReleaseStatic;
  Wait = WaitStatic;
  IsSignaled = IsSignaledStatic;
  GetNative = nullptr;
}

void VulkanCustomEvent::RetainStatic(LiteRtCustomEvent event) {
  auto* self = static_cast<VulkanCustomEvent*>(event);
  ++self->ref_count_;
}

void VulkanCustomEvent::ReleaseStatic(LiteRtCustomEvent event) {
  auto* self = static_cast<VulkanCustomEvent*>(event);
  if (--self->ref_count_ <= 0) {
    delete self;
  }
}

void VulkanCustomEvent::WaitStatic(LiteRtCustomEvent event,
                                   int64_t timeout_in_ms) {
  auto* self = static_cast<VulkanCustomEvent*>(event);
  // TODO: b/403337563 - Implement it with a VkEvent when necessary.
  self->backend_->WaitForCompletion().IgnoreError();
}

int VulkanCustomEvent::IsSignaledStatic(LiteRtCustomEvent event) {
  // TODO: b/403337563 - Implement it with VkEvent.
  return true;
}

GpuInferenceContextVulkan::GpuInferenceContextVulkan(
    GpuBackendVulkan* backend, int num_steps_of_command_buffer_preparations)
    : backend_(backend),
      next_command_buffers_(num_steps_of_command_buffer_preparations),
      num_nodes_per_command_buffer_(kDefaultMaxNumNodesPerCommandBuffer) {}

GpuInferenceContextVulkan::~GpuInferenceContextVulkan() {
  if (next_command_buffers_thread_) {
    next_command_buffers_thread_->join();
  }
  if (post_dispatch_event_ != nullptr) {
    post_dispatch_event_->Release(post_dispatch_event_);
  }
}

absl::StatusOr<::ml_drift::GpuSpatialTensor*>
GpuInferenceContextVulkan::GetSpatialTensor(::ml_drift::ValueId id) {
  return ctx_.GetTensor(id);
}

absl::Status GpuInferenceContextVulkan::BindSpatialTensor(
    ::ml_drift::ValueId id, ::ml_drift::GpuSpatialTensor* tensor) {
  return ctx_.SetTensor(id, static_cast<VulkanSpatialTensor*>(tensor));
}

absl::Status GpuInferenceContextVulkan::WriteDataToWeightTensor(
    ::ml_drift::ValueId id, absl::Span<const uint8_t> data) {
  return absl::UnimplementedError(
      "WriteDataToWeightTensor is not implemented.");
}

absl::Status GpuInferenceContextVulkan::ReadWeightTensorToDescriptor(
    ::ml_drift::ValueId id, ::ml_drift::TensorDescriptor& desc) {
  return absl::UnimplementedError(
      "ReadWeightTensorToDescriptor is not implemented.");
}

GpuInferenceContextVulkan* GpuInferenceContextVulkan::last_ctx_ = nullptr;

absl::Status GpuInferenceContextVulkan::Dispatch() {
  auto& shared_env = backend_->vk_env();

  // Submit any command buffers used so far, e.g. to upload input tensors.
  RETURN_IF_ERROR(shared_env.SubmitCommandBuffer());

  // No command buffer preparation in advance. Add inference nodes to command
  // buffers newly created and added in shared_env.pending_command_buffers().
  if (next_command_buffers_.empty()) {
    ASSIGN_OR_RETURN(auto command_buffers, ctx_.AddToQueueAsync(
        &shared_env.vulkan_env(), num_nodes_per_command_buffer_));
    return shared_env.AddToPendingCommandBuffers(std::move(command_buffers));
  }

  if (next_command_buffers_thread_) {
    next_command_buffers_thread_->join();
  }

  if (last_ctx_ != this) {
    last_ctx_ = this;
    // The next_command_buffers_ must be invalidated since the last Dispatch()
    // call was done by a different context. next_command_buffers_ is valid only
    // when Dispatch() is called by the same context continuously.
    next_command_buffers_index_ = 0;
    next_command_buffers_ready_ = false;
    for (auto& b : next_command_buffers_) {
      b = ::ml_drift::syrtis::CommandBuffer();
    }
  }

  // Create a new command buffer to prepare in the shared_env with a new
  // command pool as it can't be shared with existing command buffers.
  ASSIGN_OR_RETURN(auto* command_buffer,
                   shared_env.GetCommandBuffer(/*new_command_pool=*/true));
  // Swap it with one prepared by the previous next_command_buffers_thread_.
  auto& next_buffer = next_command_buffers_[next_command_buffers_index_];
  std::swap(next_buffer, *command_buffer);

  if (next_command_buffers_ready_) {
    RETURN_IF_ERROR(shared_env.SubmitCommandBuffer());
  } else {
    // Prepare and use command buffers immediately since cached ones are not
    // ready to use yet.
    ASSIGN_OR_RETURN(auto command_buffers, ctx_.AddToQueueAsync(
        &shared_env.vulkan_env(), num_nodes_per_command_buffer_));
    RETURN_IF_ERROR(
        shared_env.AddToPendingCommandBuffers(std::move(command_buffers)));
  }

  // Schedule another next_command_buffers_thread_ to prepare command buffers
  // in parallel with the current Dispatch() call.
  next_command_buffers_thread_ = std::make_unique<std::thread>([this]() {
    auto& next_buffer = next_command_buffers_[next_command_buffers_index_];
    if (auto s = ctx_.AddToCommandBuffer(next_buffer.VkCB());
        !s.ok()) {
      ABSL_LOG(ERROR) << "Failed to prepare next command buffers: " << s;
    } else {
      next_command_buffers_index_ =
          (next_command_buffers_index_ + 1) % next_command_buffers_.size();
      if (!next_command_buffers_ready_ && next_command_buffers_index_ == 0) {
        next_command_buffers_ready_ = true;
      }
    }
  });

  return absl::OkStatus();
}

absl::StatusOr<GpuEventHandle>
GpuInferenceContextVulkan::GetPreDispatchEvent() {
  return absl::NotFoundError("No pre-dispatch event.");
}

absl::StatusOr<GpuEventHandle> GpuInferenceContextVulkan::GetPostDispatchEvent(
    bool is_async_execution_mode) {
  if (!is_async_execution_mode) {
    return absl::NotFoundError("No post-dispatch event.");
  }

  if (post_dispatch_event_ != nullptr) {
    post_dispatch_event_->Release(post_dispatch_event_);
  }

  return (post_dispatch_event_ = new VulkanCustomEvent(backend_));
}

absl::Status GpuInferenceContextVulkan::WaitForEventsCompleted(
    absl::Span<GpuEventHandle> events, bool force_sync) {
  if (force_sync && !events.empty()) {
    return backend_->WaitForCompletion();
  }
  // TODO: b/403337563 - Implement it with VkEvent.
  return absl::OkStatus();
}

absl::Status GpuInferenceContextVulkan::PreConvert(bool input) {
  return absl::OkStatus();
}

absl::Status GpuInferenceContextVulkan::PostConvert(bool input) {
  return absl::OkStatus();
}

absl::Status GpuInferenceContextVulkan::Profile(
    ::ml_drift::ProfilingInfo& profiling_info) {
  return ctx_.Profile(&backend_->vk_env().vulkan_env(), &profiling_info);
}

absl::StatusOr<size_t>
GpuInferenceContextVulkan::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  return ctx_.GetIntermediateTensorsSize();
}

absl::StatusOr<size_t>
GpuInferenceContextVulkan::GetSizeOfMemoryAllocatedForConstantTensors() const {
  return ctx_.GetConstantTensorsSize();
}

absl::StatusOr<size_t>
GpuInferenceContextVulkan::GetSizeOfMemoryAllocatedForExternalTensors() const {
  return 0;
}

absl::Status GpuInferenceContextVulkan::ReportMemoryBenchmarkIfEnabled(
    const ::ml_drift::CreateGpuModelInfo& create_info) {
  return absl::OkStatus();
}

absl::Status GpuInferenceContextVulkan::SetCommandBufferHint(
    int num_nodes_per_command_buffer) {
  num_nodes_per_command_buffer_ = num_nodes_per_command_buffer;
  return absl::OkStatus();
}

GpuIOBufferVulkan::GpuIOBufferVulkan(::ml_drift::syrtis::Buffer&& buffer)
    : buffer_(std::move(buffer)) {}

absl::Status GpuIOBufferVulkan::Read(absl::Span<uint8_t> data) {
  return absl::UnimplementedError("Read is not implemented.");
}

absl::Status GpuIOBufferVulkan::Write(absl::Span<const uint8_t> data) {
  return absl::UnimplementedError("Write is not implemented.");
}

Tensor2BufferConverterVulkan::Tensor2BufferConverterVulkan(
    GpuBackendVulkan* backend,
    std::unique_ptr<::ml_drift::syrtis::TensorToBHWCBufferConverter> converter)
    : backend_(backend), converter_(std::move(converter)) {}

absl::Status Tensor2BufferConverterVulkan::Convert(
    ::ml_drift::GpuSpatialTensor& src_tensor, GpuIOBuffer& dst_buffer) {
  ASSIGN_OR_RETURN(auto* command_buffer, backend_->vk_env().GetCommandBuffer());
  return converter_->Convert(
      backend_->vk_env().vulkan_env().GetDevice(), command_buffer->VkCB(),
      static_cast<::ml_drift::syrtis::VulkanSpatialTensor*>(&src_tensor),
      &(static_cast<GpuIOBufferVulkan&>(dst_buffer).buffer()));
}

Buffer2TensorConverterVulkan::Buffer2TensorConverterVulkan(
    GpuBackendVulkan* backend,
    std::unique_ptr<::ml_drift::syrtis::BHWCBufferToTensorConverter> converter)
    : backend_(backend), converter_(std::move(converter)) {}

absl::Status Buffer2TensorConverterVulkan::Convert(
    GpuIOBuffer& src_buffer, ::ml_drift::GpuSpatialTensor& dst_tensor) {
  ASSIGN_OR_RETURN(auto* command_buffer, backend_->vk_env().GetCommandBuffer());
  return converter_->Convert(
      backend_->vk_env().vulkan_env().GetDevice(), command_buffer->VkCB(),
      &(static_cast<GpuIOBufferVulkan&>(src_buffer).buffer()),
      static_cast<::ml_drift::syrtis::VulkanSpatialTensor*>(&dst_tensor));
}

}  // namespace litert::ml_drift
