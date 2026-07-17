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

#include "ml_drift_delegate/delegate/gpu_backend_webgpu_litert.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/die_if_null.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/webgpu/buffer.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/memory_manager.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_api_util.h"  // from @ml_drift
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/gpu_backend_webgpu.h"

namespace {

static bool sHasWebGpuError = false;

void WebGpuError(const char* message) {
  sHasWebGpuError = true;

  LITERT_LOG(LITERT_ERROR, "WebGPU error: %s", message);
}

#define RETURN_IF_WEBGPU_ERROR()                        \
  if (sHasWebGpuError) {                                \
    sHasWebGpuError = false;                            \
    return absl::InternalError("WebGPU runtime error"); \
  }

}  // namespace

namespace litert::ml_drift {

GpuBackendWebGpuLitert::GpuBackendWebGpuLitert(
    ::ml_drift::webgpu::ExecutionEnvironment* env, bool strict_error_handling,
    const LiteRtRuntimeContext* runtime_context)
    : GpuBackendWebGpu(env),
      runtime_context_(ABSL_DIE_IF_NULL(runtime_context)) {
  if (strict_error_handling) {
    ::ml_drift::webgpu::ExecutionEnvironment::SetErrorFn(WebGpuError);
  }
}

absl::StatusOr<GpuMemoryHandle> GpuBackendWebGpuLitert::GetGpuMemoryAllocated(
    const GpuTensorBufferPtr& tensor_buffer) {
  HwMemoryHandle hw_memory_handle;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_custom_tensor_buffer_handle(
          tensor_buffer.get(), &hw_memory_handle));
  return reinterpret_cast<GpuMemoryHandle>(hw_memory_handle);
}

absl::StatusOr<GpuEventHandle> GpuBackendWebGpuLitert::GetGpuEventAssociated(
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

absl::Status GpuBackendWebGpuLitert::AssociateGpuEvent(
    GpuEventHandle event, LiteRtEnvironment env,
    GpuTensorBufferPtr& tensor_buffer) {
  if (runtime_context_ == nullptr) {
    return absl::InternalError("Runtime context is not set.");
  }
  LiteRtEvent liter_event;
  LITERT_RETURN_IF_ERROR(runtime_context_->create_managed_event(
      env, LiteRtEventTypeCustom, &liter_event));
  LITERT_RETURN_IF_ERROR(runtime_context_->set_custom_event(
      liter_event, reinterpret_cast<LiteRtCustomEvent>(event)));
  LITERT_RETURN_IF_ERROR(runtime_context_->set_tensor_buffer_event(
      tensor_buffer.get(), liter_event));
  return absl::OkStatus();
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendWebGpuLitert::GetGpuBufferRequirements(
    ::ml_drift::TensorStorageType used_storage_type,
    ::ml_drift::DataType data_type) {
  GpuBufferRequirements requirements;
  if (used_storage_type == ::ml_drift::TensorStorageType::TEXTURE_2D) {
    if (data_type == ::ml_drift::DataType::FLOAT16) {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeWebGpuTextureFp16);
    } else {
      requirements.buffer_types.push_back(kLiteRtTensorBufferTypeWebGpuTexture);
    }
  } else if (used_storage_type == ::ml_drift::TensorStorageType::IMAGE_BUFFER) {
    if (data_type == ::ml_drift::DataType::FLOAT16) {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeWebGpuImageBufferFp16);
    } else {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeWebGpuImageBuffer);
    }
  } else {
    if (data_type == ::ml_drift::DataType::FLOAT16) {
      requirements.buffer_types.push_back(
          kLiteRtTensorBufferTypeWebGpuBufferFp16);
    } else {
      requirements.buffer_types.push_back(kLiteRtTensorBufferTypeWebGpuBuffer);
    }
  }
  // MLD uses PHWC4, 16 bytes strides.
  requirements.strides.push_back(16);
  return std::move(requirements);
}

absl::StatusOr<GpuBackend::GpuBufferRequirements>
GpuBackendWebGpuLitert::GetGpuBufferRequirementsForNonExternalTensors() {
  return GpuBufferRequirements{
      .buffer_types = {kLiteRtTensorBufferTypeWebGpuBufferPacked},
      // No strides for packed buffer.
      .strides = {0},
  };
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendWebGpuLitert::CreateInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
    bool may_share_memory_manager) {
  auto ctx = std::make_unique<GpuInferenceContextWebGpuLitert>(
      this, may_share_memory_manager ? &memory_manager() : nullptr, create_info,
      num_steps_of_command_buffer_preparations());
  RETURN_IF_ERROR(ctx->wgpu_ctx().InitFromGpuModel(
      wgpu_env(), create_info, &gpu_model, serialized_model));
  RETURN_IF_WEBGPU_ERROR();
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<GpuInferenceContext>>
GpuBackendWebGpuLitert::RestoreInferenceContext(
    const ::ml_drift::CreateGpuModelInfo& create_info,
    const absl::Span<const uint8_t> serialized_model) {
  auto ctx = std::make_unique<GpuInferenceContextWebGpuLitert>(
      this, &memory_manager(), create_info,
      num_steps_of_command_buffer_preparations());
  RETURN_IF_ERROR(ctx->wgpu_ctx().RestoreDeserialized(
      serialized_model, wgpu_env(), &create_info));
  RETURN_IF_WEBGPU_ERROR();
  return std::move(ctx);
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>>
GpuBackendWebGpuLitert::CreateIOBuffer(GpuMemoryHandle gpu_memory) {
  ::ml_drift::webgpu::SpatialTensor* wgpu_tensor =
      reinterpret_cast<::ml_drift::webgpu::SpatialTensor*>(gpu_memory);
  ::ml_drift::webgpu::Buffer wgpu_buffer = ::ml_drift::webgpu::Buffer(
      wgpu_tensor->GetBufferHandle(), wgpu_tensor->GetBufferHandle().GetSize());
  auto buffer =
      std::make_unique<GpuIOBufferWebGpu>(this, std::move(wgpu_buffer));
  return std::move(buffer);
}

absl::Status GpuBackendWebGpuLitert::WaitForCompletion() {
  constexpr int kMaxWaitRetries = 6;
  constexpr absl::Duration kWaitDuration = absl::Seconds(10);
  for (int wait_cnt = 1; wait_cnt <= kMaxWaitRetries; ++wait_cnt) {
    auto res = ::ml_drift::webgpu::WaitUntilCompleted(
        wgpu_env().queue(), wgpu_env().device(), kWaitDuration);
    if (res.ok()) {
      return absl::OkStatus();
    }
    if (res.code() == absl::StatusCode::kDeadlineExceeded) {
      // Warning if the wait fails. It's possible that the GPU is not as fast
      // to process the commands in time.
      LITERT_LOG(LITERT_WARNING, "WaitForCompletion timeout. Retry: %d",
                 wait_cnt);
    } else {
      return res;
    }
  }
  return absl::OkStatus();
}

WebGpuCustomEvent::WebGpuCustomEvent() : ref_count_(1) {
  Retain = RetainStatic;
  Release = ReleaseStatic;
  Wait = WaitStatic;
  IsSignaled = IsSignaledStatic;
  GetNative = nullptr;
}

void WebGpuCustomEvent::RetainStatic(LiteRtCustomEvent event) {
  auto* self = static_cast<WebGpuCustomEvent*>(event);
  ++self->ref_count_;
}

void WebGpuCustomEvent::ReleaseStatic(LiteRtCustomEvent event) {
  auto* self = static_cast<WebGpuCustomEvent*>(event);
  if (--self->ref_count_ <= 0) {
    delete self;
  }
}

void WebGpuCustomEvent::WaitStatic(LiteRtCustomEvent event,
                                   int64_t timeout_in_ms) {
  // As WebGPU guarantees the commands are executed in order, no need to wait
  // for queue completed here. Otherwise, event->Wait() would sync up to the
  // all the commands completed in queue which would be too much if event was
  // actually enqueued much earlier. It's crucial to implement fully-async async
  // pipeline.
}

int WebGpuCustomEvent::IsSignaledStatic(LiteRtCustomEvent event) {
  // As WebGPU guarantees the commands are executed in order, the event can be
  // marked as signaled immediately.
  return true;
}

GpuInferenceContextWebGpuLitert::GpuInferenceContextWebGpuLitert(
    GpuBackendWebGpuLitert* backend,
    ::ml_drift::webgpu::MemoryManager* memory_manager,
    const ::ml_drift::CreateGpuModelInfo& create_info,
    int num_steps_of_command_buffer_preparations)
    : GpuInferenceContextWebGpu(backend, memory_manager, create_info,
                                num_steps_of_command_buffer_preparations) {}

GpuInferenceContextWebGpuLitert::~GpuInferenceContextWebGpuLitert() {
  if (post_dispatch_event_ != nullptr) {
    post_dispatch_event_->Release(post_dispatch_event_);
  }
}

absl::StatusOr<GpuEventHandle>
GpuInferenceContextWebGpuLitert::GetPreDispatchEvent() {
  return absl::NotFoundError("No pre-dispatch event.");
}

absl::StatusOr<GpuEventHandle>
GpuInferenceContextWebGpuLitert::GetPostDispatchEvent(
    bool is_async_execution_mode) {
  if (!is_async_execution_mode) {
    return absl::NotFoundError("No post-dispatch event.");
  }

  if (post_dispatch_event_ != nullptr) {
    post_dispatch_event_->Release(post_dispatch_event_);
  }

  return (post_dispatch_event_ = new WebGpuCustomEvent());
}

absl::Status GpuInferenceContextWebGpuLitert::WaitForEventsCompleted(
    absl::Span<GpuEventHandle> events, bool force_sync) {
  if (force_sync && !events.empty()) {
    return backend()->WaitForCompletion();
  }
  // If not force_sync or no events to wait for, we don't need to wait as webgpu
  // guarantees that the commands are executed in order.
  return absl::OkStatus();
}

}  // namespace litert::ml_drift
