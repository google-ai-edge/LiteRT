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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_WEBGPU_LITERT_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_WEBGPU_LITERT_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/memory_manager.h"  // from @ml_drift
#include "litert/c/litert_event_type.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/gpu_backend_webgpu.h"

namespace litert::ml_drift {

// GpuBackend for WebGPU with litert tensor buffer.
class GpuBackendWebGpuLitert : public GpuBackendWebGpu {
 public:
  // Creates a GpuBackendWebGpuLitert instance.
  //
  // If `strict_error_handling` is set, internal WebGPU errors will be returned
  // as absl::InternalError.
  explicit GpuBackendWebGpuLitert(::ml_drift::webgpu::ExecutionEnvironment* env,
                                  bool strict_error_handling,
                                  const LiteRtRuntimeContext* runtime_context);
  ~GpuBackendWebGpuLitert() override = default;

  // Implementation of GpuBackendWebGpu.
  absl::StatusOr<GpuMemoryHandle> GetGpuMemoryAllocated(
      const GpuTensorBufferPtr& tensor_buffer) override;
  absl::StatusOr<GpuEventHandle> GetGpuEventAssociated(
      const GpuTensorBufferPtr& tensor_buffer) override;
  absl::Status AssociateGpuEvent(GpuEventHandle event, LiteRtEnvironment env,
                                 GpuTensorBufferPtr& tensor_buffer) override;
  absl::StatusOr<GpuBufferRequirements> GetGpuBufferRequirements(
      ::ml_drift::TensorStorageType used_storage_type,
      ::ml_drift::DataType data_type) override;
  absl::StatusOr<GpuBackend::GpuBufferRequirements>
  GetGpuBufferRequirementsForNonExternalTensors() override;
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> CreateInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GpuModel& gpu_model, std::vector<uint8_t>* serialized_model,
      bool may_share_memory_manager) override;
  absl::StatusOr<std::unique_ptr<GpuInferenceContext>> RestoreInferenceContext(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      absl::Span<const uint8_t> serialized_model) override;
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBuffer(
      GpuMemoryHandle gpu_memory) override;
  absl::StatusOr<std::unique_ptr<::ml_drift::SharedMemoryManager>>
  CreateSharedMemoryManager(
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GraphFloat32& graph, TfLiteContext* context,
      MlDriftDelegateData& delegate_data,
      ::ml_drift::SerializationWeightCache* serialization_cache) override;
  absl::Status WaitForCompletion() override;

 private:
  const LiteRtRuntimeContext* const runtime_context_;
};

// Custom event for WebGPU to emulate the asynchronous execution.
class WebGpuCustomEvent : public LiteRtCustomEventT {
 public:
  WebGpuCustomEvent();
  ~WebGpuCustomEvent() = default;

 private:
  static void RetainStatic(LiteRtCustomEvent event);
  static void ReleaseStatic(LiteRtCustomEvent event);
  static void WaitStatic(LiteRtCustomEvent event, int64_t timeout_in_ms);
  static int IsSignaledStatic(LiteRtCustomEvent event);

  int ref_count_;
};

// GpuBackend for WebGPU with litert tensor buffer.
class GpuInferenceContextWebGpuLitert : public GpuInferenceContextWebGpu {
 public:
  explicit GpuInferenceContextWebGpuLitert(
      GpuBackendWebGpuLitert* backend,
      ::ml_drift::webgpu::MemoryManager* memory_manager,
      const ::ml_drift::CreateGpuModelInfo& create_info,
      int num_steps_of_command_buffer_preparations);
  ~GpuInferenceContextWebGpuLitert() override;

  absl::StatusOr<GpuEventHandle> GetPreDispatchEvent() override;
  absl::StatusOr<GpuEventHandle> GetPostDispatchEvent(
      bool is_async_execution_mode) override;
  absl::Status WaitForEventsCompleted(absl::Span<GpuEventHandle> events,
                                      bool force_sync) override;

 private:
  // Cache the last post dispatch event for correct ref counting.
  WebGpuCustomEvent* post_dispatch_event_ = nullptr;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_WEBGPU_LITERT_H_
