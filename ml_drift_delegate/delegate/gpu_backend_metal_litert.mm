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

#import "third_party/odml/litert/ml_drift/delegate/gpu_backend_metal_litert.h"
#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/die_if_null.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "ml_drift_delegate/delegate/custom_event_metal.h"
#include "ml_drift_delegate/delegate/delegate_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend_metal.h"
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_metal.h"
#include "tflite/core/subgraph.h"

namespace litert::ml_drift {

GpuBackendMetalLitert::GpuBackendMetalLitert(::ml_drift::metal::MetalDevice* device,
                                             GpuDelegateWaitType wait_type,
                                             id<MTLCommandQueue> command_queue,
                                             const LiteRtRuntimeContext* runtime_context,
                                             bool enable_residency_set)
    : GpuBackendMetal(device, wait_type, command_queue, enable_residency_set),
      runtime_context_(ABSL_DIE_IF_NULL(runtime_context)) {}

GpuBackendMetalLitert::~GpuBackendMetalLitert() = default;

absl::StatusOr<::ml_drift::TensorStorageType> GpuBackendMetalLitert::GetFastestStorageType() {
  return ::ml_drift::metal::GetFastestStorageType(metal_device()->GetInfo());
}

absl::StatusOr<GpuMemoryHandle> GpuBackendMetalLitert::GetGpuMemoryAllocated(
    const GpuTensorBufferPtr& tensor_buffer) {
  HwMemoryHandle hw_memory_handle;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_custom_tensor_buffer_handle(
      tensor_buffer.get(), &hw_memory_handle));
  return reinterpret_cast<GpuMemoryHandle>(hw_memory_handle);
}

absl::StatusOr<GpuEventHandle> GpuBackendMetalLitert::GetGpuEventAssociated(
    const GpuTensorBufferPtr& tensor_buffer) {
  bool has_event;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->has_tensor_buffer_event(tensor_buffer.get(), &has_event));
  if (!has_event) {
    return absl::NotFoundError("Tensor buffer does not have an event.");
  }

  LiteRtEvent event;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_event(tensor_buffer.get(), &event));
  LiteRtEventType event_type;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_event_event_type(event, &event_type));
  if (event_type != LiteRtEventTypeCustom) {
    return absl::InternalError("Tensor buffer has a non-custom event.");
  }
  LiteRtCustomEvent custom_event;
  LITERT_RETURN_IF_ERROR(runtime_context_->get_custom_event(event, &custom_event));
  return custom_event;
}

absl::Status GpuBackendMetalLitert::AssociateGpuEvent(GpuEventHandle event, LiteRtEnvironment env,
                                                      GpuTensorBufferPtr& tensor_buffer) {
  if (runtime_context_ == nullptr) {
    return absl::InternalError("Runtime context is not set.");
  }
  LiteRtEvent liter_event;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->create_managed_event(env, LiteRtEventTypeCustom, &liter_event));
  LITERT_RETURN_IF_ERROR(
      runtime_context_->set_custom_event(liter_event, reinterpret_cast<LiteRtCustomEvent>(event)));
  LITERT_RETURN_IF_ERROR(
      runtime_context_->set_tensor_buffer_event(tensor_buffer.get(), liter_event));
  return absl::OkStatus();
}

absl::StatusOr<GpuBackend::GpuBufferRequirements> GpuBackendMetalLitert::GetGpuBufferRequirements(
    ::ml_drift::TensorStorageType used_storage_type, ::ml_drift::DataType data_type) {
  GpuBufferRequirements requirements;
  if (used_storage_type == ::ml_drift::TensorStorageType::TEXTURE_2D) {
    if (data_type == ::ml_drift::DataType::FLOAT16) {
      requirements.buffer_types.push_back(kLiteRtTensorBufferTypeMetalTextureFp16);
    } else {
      requirements.buffer_types.push_back(kLiteRtTensorBufferTypeMetalTexture);
    }
  } else {
    if (data_type == ::ml_drift::DataType::FLOAT16) {
      requirements.buffer_types.push_back(kLiteRtTensorBufferTypeMetalBufferFp16);
    } else {
      requirements.buffer_types.push_back(kLiteRtTensorBufferTypeMetalBuffer);
    }
  }
  // MLD uses PHWC4, 16 bytes strides.
  requirements.strides.push_back(16);
  return std::move(requirements);
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendMetalLitert::GetGpuBufferRequirementsForNonExternalTensors() {
  return GpuBufferRequirements{
      .buffer_types = {kLiteRtTensorBufferTypeMetalBufferPacked},
      // No strides for packed buffer.
      .strides = {0},
  };
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>> GpuBackendMetalLitert::CreateInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info, ::ml_drift::GpuModel& gpu_model,
    std::vector<uint8_t>* serialized_model, bool may_share_memory_manager) {
  auto ctx = std::make_unique<GpuInferenceContextMetalLitert>(
      this, may_share_memory_manager ? &memory_manager() : nullptr);
  RETURN_IF_ERROR(ctx->metal_ctx().InitFromGpuModel(create_info, &gpu_model,
                                                    metal_device()->device(), serialized_model));
  if (@available(macOS 15.0, iOS 18.0, *)) {
    PopulateResidencySet(create_info, ctx->metal_ctx());
  }
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>> GpuBackendMetalLitert::RestoreInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info, absl::Span<const uint8_t> serialized_model) {
  auto ctx = std::make_unique<GpuInferenceContextMetalLitert>(this, &memory_manager());
  RETURN_IF_ERROR(ctx->metal_ctx().RestoreDeserialized(serialized_model, metal_device()->device(),
                                                       &create_info));
  if (@available(macOS 15.0, iOS 18.0, *)) {
    PopulateResidencySet(create_info, ctx->metal_ctx());
  }
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
GpuBackendMetalLitert::CreateSharedMemoryManager(
    const ::ml_drift::CreateGpuModelInfo& create_info, ::ml_drift::GraphFloat32& graph,
    TfLiteContext* context, MlDriftDelegateData& delegate_data,
    ::ml_drift::SerializationWeightCache* serialization_cache) {
  const std::unordered_map<size_t, size_t>* external_buffer_id_map =
      &(reinterpret_cast<const tflite::Subgraph*>(context->impl_)
            ->GetExternalTensorBufferIdentifiers());
  return ::ml_drift::MakeSharedMemoryManagerMetal(
      metal_device(), create_info, graph, context, GetBufferIdToSpatialTensorMap(delegate_data),
      GetQuantParamIdToSpatialTensorMap(delegate_data),
      delegate_data.options->has_prepacked_external_tflite_tensors, serialization_cache,
      delegate_data.options->madvise_original_shared_tensors, runtime_context_,
      delegate_data.weight_loader, external_buffer_id_map);
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>> GpuBackendMetalLitert::CreateIOBuffer(
    GpuMemoryHandle gpu_memory) {
  auto* spatial_tensor = reinterpret_cast<::ml_drift::metal::MetalSpatialTensor*>(gpu_memory);
  ::ml_drift::metal::Buffer wgpu_buffer(spatial_tensor->GetBufferHandle(),
                                        spatial_tensor->GetMemorySizeInBytes());
  return std::make_unique<GpuIOBufferMetal>(this, std::move(wgpu_buffer));
}

GpuInferenceContextMetalLitert::GpuInferenceContextMetalLitert(
    GpuBackendMetal* backend, ::ml_drift::metal::MemoryManager* memory_manager)
    : GpuInferenceContextMetal(backend, memory_manager) {}

GpuInferenceContextMetalLitert::~GpuInferenceContextMetalLitert() {
  if (post_dispatch_event_ != nullptr) {
    post_dispatch_event_->Release(post_dispatch_event_);
  }
}

absl::StatusOr<GpuEventHandle> GpuInferenceContextMetalLitert::GetPreDispatchEvent() {
  return absl::NotFoundError("No pre-dispatch event.");
}

absl::StatusOr<GpuEventHandle> GpuInferenceContextMetalLitert::GetPostDispatchEvent(
    bool is_async_execution_mode) {
  if (!is_async_execution_mode) {
    // Commit the command buffer here for DoNotWait mode as it won't call WaitForCompletion().
    if (metal_backend()->wait_type() == kGpuDelegateWaitTypeDoNotWait) {
      [metal_backend()->command_buffer() commit];
      metal_backend()->set_command_buffer(nullptr);
    }
    return absl::NotFoundError("No post-dispatch event.");
  }

  if (post_dispatch_event_ == nullptr) {
    return absl::NotFoundError("No post-dispatch event.");
  }

  if (metal_backend()->command_buffer() == nullptr) {
    return absl::FailedPreconditionError("Command buffer is not set.");
  }

  // Commit the command buffer here which has signaling post_dispatch_event_ encoded by
  // PostConvert() below.
  [metal_backend()->command_buffer() commit];
  metal_backend()->set_command_buffer(nullptr);

  return post_dispatch_event_;
}

absl::Status GpuInferenceContextMetalLitert::WaitForEventsCompleted(
    absl::Span<GpuEventHandle> events, bool force_sync) {
  if (force_sync) {
    for (auto event : events) {
      auto* metal_event = reinterpret_cast<CustomEventMetal*>(event);
      metal_event->Wait(metal_event, /*timeout_in_ms=*/0);
    }
    return absl::OkStatus();
  }

  if (metal_backend()->command_buffer() == nullptr) {
    metal_backend()->set_command_buffer([metal_backend()->command_queue() commandBuffer]);
  }

  absl::flat_hash_set<CustomEventMetal*> events_to_wait;
  for (auto event : events) {
    auto* metal_event = reinterpret_cast<CustomEventMetal*>(event);
    if (events_to_wait.insert(metal_event).second) {
      metal_event->EncodeWait(metal_backend()->command_buffer());
    }
  }

  return absl::OkStatus();
}

// Same as GpuInferenceContextMetal::PostConvert except that it doesn't wait for output conversion
// to finish if post_dispatch_event_ is not null as it is asynchronous mode.
absl::Status GpuInferenceContextMetalLitert::PostConvert(bool input) {
  [metal_backend()->compute_command_encoder() endEncoding];
  metal_backend()->set_compute_command_encoder(nullptr);

  // If output conversion is done, wait for conversion to finish, or replace post_dispatch_event_
  // with a new one.
  if (!input) {
    if (post_dispatch_event_ == nullptr) {
      RETURN_IF_ERROR(metal_backend()->WaitForCompletion());
    } else {
      if (metal_backend()->command_buffer() == nullptr) {
        metal_backend()->set_command_buffer([metal_backend()->command_queue() commandBuffer]);
      }

      if (post_dispatch_event_ != nullptr) {
        post_dispatch_event_->Release(post_dispatch_event_);
      }

      post_dispatch_event_ = new CustomEventMetal(metal_backend()->metal_device()->device());
      post_dispatch_event_->EncodeSignal(metal_backend()->command_buffer());
    }
  }

  return absl::OkStatus();
}

}  // namespace litert::ml_drift
