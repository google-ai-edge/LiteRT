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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_METAL_LITERT_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_METAL_LITERT_H_

#ifdef __cplusplus
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/metal/metal_device.h"  // from @ml_drift
#include "litert/c/litert_event_type.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_options.h"
#include "third_party/odml/litert/ml_drift/delegate/gpu_backend.h"
#include "third_party/odml/litert/ml_drift/delegate/gpu_backend_metal.h"

namespace litert::ml_drift {

// GpuBackend for Metal with litert tensor buffer.
class GpuBackendMetalLitert : public GpuBackendMetal {
 public:
  explicit GpuBackendMetalLitert(::ml_drift::metal::MetalDevice* device,
                                 GpuDelegateWaitType wait_type,
                                 id<MTLCommandQueue> command_queue,
                                 const LiteRtRuntimeContext* runtime_context,
                                 bool enable_residency_set = false);
  ~GpuBackendMetalLitert() override;

  // Implementation of GpuBackend.
  absl::StatusOr<::ml_drift::TensorStorageType> GetFastestStorageType()
      override;
  absl::StatusOr<GpuMemoryHandle> GetGpuMemoryAllocated(
      const GpuTensorBufferPtr& tensor_buffer) override;
  absl::StatusOr<GpuEventHandle> GetGpuEventAssociated(
      const GpuTensorBufferPtr& tensor_buffer) override;
  absl::Status AssociateGpuEvent(GpuEventHandle event, LiteRtEnvironment env,
                                 GpuTensorBufferPtr& tensor_buffer) override;
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
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateIOBuffer(
      GpuMemoryHandle gpu_memory) override;

 private:
  const LiteRtRuntimeContext* const runtime_context_;
};

// Custom event for Metal wrapping a MTLSharedEvent for asynchronous execution.
class MetalCustomEvent : public LiteRtCustomEventT {
 public:
  explicit MetalCustomEvent(GpuBackendMetal* backend);
  ~MetalCustomEvent() = default;

  // Encodes the signal command in the command buffer.
  void EncodeSignal(id<MTLCommandBuffer> command_buffer);
  // Encodes the wait command in the command buffer.
  void EncodeWait(id<MTLCommandBuffer> command_buffer);

 private:
  // Callbacks of litert_custom_event_t.
  static void RetainStatic(LiteRtCustomEvent event);
  static void ReleaseStatic(LiteRtCustomEvent event);
  static void WaitStatic(LiteRtCustomEvent event, int64_t timeout_in_ms);
  static int IsSignaledStatic(LiteRtCustomEvent event);
  static void* GetNativeStatic(LiteRtCustomEvent event);

  int ref_count_;
  id<MTLSharedEvent> shared_event_;
  uint64_t value_to_wait_ = 0;
};

class GpuInferenceContextMetalLitert : public GpuInferenceContextMetal {
 public:
  explicit GpuInferenceContextMetalLitert(
      GpuBackendMetal* backend,
      ::ml_drift::metal::MemoryManager* memory_manager = nullptr);
  ~GpuInferenceContextMetalLitert() override;

  absl::StatusOr<GpuEventHandle> GetPreDispatchEvent() override;
  absl::StatusOr<GpuEventHandle> GetPostDispatchEvent(
      bool is_async_execution_mode) override;
  absl::Status WaitForEventsCompleted(absl::Span<GpuEventHandle> events,
                                      bool force_sync) override;
  absl::Status PostConvert(bool input) override;

 private:
  // Cache the last post dispatch event for correct ref counting.
  MetalCustomEvent* post_dispatch_event_ = nullptr;
};

}  // namespace litert::ml_drift
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_METAL_LITERT_H_
