/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_ML_DRIFT_METAL_MODEL_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_ML_DRIFT_METAL_MODEL_RUNNER_H_

// ML Drift Metal (Apple GPU) backend for `CompiledMlDriftModelRunner`.
//
// This header uses Metal types, so it must only be included from
// Objective-C++ translation units compiled with ARC (`objc_library`). Plain
// C++ code can use `CompiledMlDriftMetalModelRunner`
// (compiled_ml_drift_metal_model_runner.h) instead.
//
// Execution model: `MlDriftMetalEnvironment` owns a single in-order
// `MTLCommandQueue`. GPU work (inference, blit copies, uploads) is encoded
// into an open `MTLCommandBuffer` that is committed by `Flush()`; command
// buffers execute in commit order and Metal's automatic hazard tracking
// orders accesses to the same buffers. `WaitForCompletion()` commits pending
// work and blocks until the GPU is idle, after which the CPU may read the
// (unified-memory, `MTLResourceStorageModeShared`) tensor buffers.

#import <Metal/Metal.h>

#include <cstddef>
#include <memory>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/metal/environment.h"  // from @ml_drift
#include "ml_drift/metal/inference_context.h"  // from @ml_drift
#include "ml_drift/metal/metal_spatial_tensor.h"  // from @ml_drift
#include "tensor/buffer.h"

namespace litert::tensor {

// Metal device, GPU info and one in-order command queue shared by every
// inference context compiled against it.
class MlDriftMetalEnvironment {
 public:
  MlDriftMetalEnvironment();
  ~MlDriftMetalEnvironment();

  MlDriftMetalEnvironment(const MlDriftMetalEnvironment&) = delete;
  MlDriftMetalEnvironment& operator=(const MlDriftMetalEnvironment&) = delete;

  // Creates the system default Metal device and the command queue. Fails with
  // kFailedPrecondition when the machine has no Metal device. Idempotent.
  absl::Status Init();
  bool initialized() const { return metal_env_ != nullptr; }

  // All accessors below require `initialized()`.
  id<MTLDevice> device() const;
  id<MTLCommandQueue> queue() const { return queue_; }
  const ml_drift::GpuInfo& GetInfo() const;
  ml_drift::metal::Environment* metal_env() const { return metal_env_.get(); }

  // `CompiledMlDriftModelRunner::Profile()` passes `profiling_queue()` to
  // `InferenceContext::Profile()`; Metal profiles against the device (it
  // creates its own command queue).
  id<MTLDevice> profiling_queue() const { return device(); }

  // Returns the open command buffer, creating it if needed. Work encoded
  // into it runs after everything committed before.
  id<MTLCommandBuffer> command_buffer();

  // Commits the open command buffer, if any, without waiting.
  void Flush();

  // True if a committed command buffer is still executing on the GPU (i.e. has
  // not yet been waited on via `WaitForCompletion()`). When false, no GPU work
  // is currently in flight, so the CPU may write directly into unified-memory
  // (`MTLResourceStorageModeShared`) input buffers before encoding commands.
  bool has_in_flight_work() const {
    return last_committed_command_buffer_ != nil;
  }

  // Commits the open command buffer and blocks until all committed work has
  // completed. Returns an error if any committed command buffer failed since
  // the last call.
  absl::Status WaitForCompletion();

  // Number of ML Drift tasks encoded per `MTLComputeCommandEncoder` by
  // `MlDriftMetalBackend::AddToQueue` (env `LRT_METAL_TASKS_PER_ENCODER`).
  int tasks_per_encoder() const { return tasks_per_encoder_; }

 private:
  struct ErrorState;

  std::unique_ptr<ml_drift::metal::Environment> metal_env_;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLCommandBuffer> open_command_buffer_ = nil;
  id<MTLCommandBuffer> last_committed_command_buffer_ = nil;
  std::shared_ptr<ErrorState> error_state_;
  int tasks_per_encoder_ = 32;
};

// `ml_drift::metal::InferenceContext` with a status-returning `Profile()` so
// that the API-agnostic `CompiledMlDriftModelRunner::Profile()` compiles for
// Metal.
class MlDriftMetalInferenceContext : public ml_drift::metal::InferenceContext {
 public:
  using ml_drift::metal::InferenceContext::InferenceContext;

  absl::Status Profile(id<MTLDevice> device, ml_drift::ProfilingInfo* result);
};

// Returns the byte offset of `tensor`'s data inside `GetBufferHandle()`.
// Buffer-backed tensors allocated by the ML Drift Metal memory manager are
// sub-allocations of a shared `MTLBuffer`.
size_t MetalTensorBufferOffset(const ml_drift::metal::MetalSpatialTensor& tensor);

// True if the tensor data lives in a (CPU-visible, shared storage) MTLBuffer
// in ML Drift's linear BUFFER layout.
bool IsMetalLinearBufferTensor(const ml_drift::metal::MetalSpatialTensor& tensor);

struct MlDriftMetalBackend {
  using Environment = MlDriftMetalEnvironment;
  using InferenceContext = MlDriftMetalInferenceContext;
  using Tensor = ml_drift::metal::MetalSpatialTensor;

  static absl::Status InitEnvironment(Environment* env) { return env->Init(); }

  static ml_drift::GpuInfo GetGpuInfoForBuilder(const Environment& env,
                                                bool env_initialized) {
    if (env_initialized && env.initialized()) {
      return env.GetInfo();
    }
    return ml_drift::GpuInfo();
  }

  static ml_drift::TensorStorageType GetStorageTypeForBuilder(
      const Environment& env, bool env_initialized);

  static absl::Status InitInferenceContext(
      InferenceContext* ctx, const ml_drift::CreateGpuModelInfo& create_info,
      ml_drift::GpuModel* model, Environment* env,
      const ml_drift::ExternalTensorsInfo& external_tensors = {});

  static absl::Status SetInputTensorFloat(
      InferenceContext* ctx, Environment* env, ml_drift::ValueId id,
      const ml_drift::TensorFloat32& tensor);

  static absl::Status SetInputTensorInt32(
      InferenceContext* ctx, Environment* env, ml_drift::ValueId id,
      const ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32>&
          tensor);

  static absl::Status GetOutputTensorFloat(InferenceContext* ctx,
                                           Environment* env,
                                           ml_drift::ValueId id,
                                           ml_drift::TensorFloat32* tensor);

  static absl::Status GetOutputTensorInt32(InferenceContext* ctx,
                                           Environment* env,
                                           ml_drift::ValueId id,
                                           ml_drift::TensorInt32* tensor);

  static Tensor* GetTensor(InferenceContext* ctx, ml_drift::ValueId id) {
    return ctx->GetTensor(id);
  }

  static absl::Status CreateConstTensor(Environment* env,
                                        const ml_drift::TensorDescriptor& desc,
                                        Tensor* tensor);

  static absl::Status UploadTensorDescriptorData(
      Tensor* tensor, Environment* env, const ml_drift::TensorDescriptor& desc);

  static absl::Status TensorToDescriptor(Tensor* tensor, Environment* env,
                                         ml_drift::TensorDescriptor* desc);

  // Encodes the whole model into the environment's command buffer and
  // commits it (non-blocking).
  static absl::Status AddToQueue(InferenceContext* ctx, Environment* env);

  static absl::Status WaitForCompletion(Environment* env) {
    return env->WaitForCompletion();
  }

  // Zero-copy interop with LiteRT / MlDriftBuffer buffers is OpenCL-only for
  // now. `CompiledMlDriftModelRunner::SetInput` falls back to a host upload
  // when `CopyBufferToTensor` fails.
  static absl::Status CopyBufferToTensor(InferenceContext* ctx,
                                         Environment* env, ml_drift::ValueId id,
                                         Buffer* generic_buffer);
  static absl::StatusOr<std::unique_ptr<Buffer>> CreateInputBuffer(
      InferenceContext* ctx, Environment* env, ml_drift::ValueId id);
  static absl::Status CopyTensorToBuffer(InferenceContext* ctx,
                                         Environment* env, ml_drift::ValueId id,
                                         Buffer* generic_buffer);
  static absl::StatusOr<std::shared_ptr<Buffer>> CreateOutputBuffer(
      InferenceContext* ctx, Environment* env, ml_drift::ValueId id);
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_ML_DRIFT_METAL_MODEL_RUNNER_H_
