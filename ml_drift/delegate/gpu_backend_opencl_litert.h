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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_OPENCL_LITERT_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_OPENCL_LITERT_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/cl/cl_event.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_gl_types.h"
#include "third_party/odml/litert/ml_drift/delegate/cache/simple_cache.h"
#include "third_party/odml/litert/ml_drift/delegate/gpu_backend.h"
#include "third_party/odml/litert/ml_drift/delegate/gpu_backend_opencl.h"

namespace litert::ml_drift {

#if LITERT_HAS_OPENGL_SUPPORT
namespace internal {
class GlInteropFabricLiteRt;
}  // namespace internal
#endif

// GpuBackend for OpenCL with litert tensor buffer.
class GpuBackendOpenClLitert : public GpuBackendOpenCl {
 public:
  GpuBackendOpenClLitert(::ml_drift::cl::Environment* env,
                         LiteRtEglDisplay egl_display,
                         SimpleCache&& compiled_cache,
                         const LiteRtRuntimeContext* runtime_context);
  ~GpuBackendOpenClLitert() override;

  // Implementation of GpuBackendOpenCl.
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

  // Flushes the compiled shader program cache to the file if needed
  // heuristically.
  void FlushCompiledCacheIfNeeded();

  int kernel_batch_size() const { return kernel_batch_size_; }
  void set_kernel_batch_size(int kernel_batch_size) {
    kernel_batch_size_ = kernel_batch_size;
  }

 private:
  absl::Status UploadCompiledCache();

  const LiteRtEglDisplay egl_display_;
#if LITERT_HAS_OPENGL_SUPPORT
  std::unique_ptr<internal::GlInteropFabricLiteRt> gl_interop_fabric_;
#endif

  // The cache for compiled shader programs.
  SimpleCache compiled_cache_;

  const LiteRtRuntimeContext* const runtime_context_;
  // Number of compiled programs in the cache not to flush the cache if no
  // programs are added.
  int num_compiled_programs_;
  // A counter to flush the compiled shader program cache heuristically.
  int invoke_count_to_flush_compiled_cache_;
  // If > 0, specifies the kernel (op) batch size, for a flush.
  int kernel_batch_size_ = 0;
};

class GpuInferenceContextOpenClLitert : public GpuInferenceContextOpenCl {
 public:
  explicit GpuInferenceContextOpenClLitert(
      GpuBackendOpenClLitert* backend,
      ::ml_drift::cl::MemoryManager* memory_manager = nullptr,
      void* gl_interop_fabric = nullptr);
  ~GpuInferenceContextOpenClLitert() override = default;

  // Implementation of GpuInferenceContextOpenCl.
  absl::Status Dispatch() override;
  absl::StatusOr<GpuEventHandle> GetPreDispatchEvent() override;
  absl::StatusOr<GpuEventHandle> GetPostDispatchEvent(
      bool is_async_execution_mode) override;
  absl::Status WaitForEventsCompleted(absl::Span<GpuEventHandle> events,
                                      bool force_sync) override;
  absl::Status PreConvert(bool input) override;
  absl::Status PostConvert(bool input) override;

 private:
#if LITERT_HAS_OPENGL_SUPPORT
  internal::GlInteropFabricLiteRt* const gl_interop_fabric_;
#endif

  // Post dispatch event is a marker event that is used to signal the completion
  // of all output events (see GetPostDispatchEvent()) and is not released until
  // inference context is destroyed.
  ::ml_drift::cl::CLEvent post_dispatch_event_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_GPU_BACKEND_OPENCL_LITERT_H_
