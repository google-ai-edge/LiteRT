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

#include "third_party/odml/litert/tensor/runners/ml_drift/ml_drift_metal_model_runner.h"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
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
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/utils/macros.h"

namespace litert::tensor {

// Errors reported by command buffer completion handlers, which run on a Metal
// thread and may outlive the environment.
struct MlDriftMetalEnvironment::ErrorState {
  absl::Mutex mutex;
  std::string first_error ABSL_GUARDED_BY(mutex);
};

MlDriftMetalEnvironment::MlDriftMetalEnvironment()
    : error_state_(std::make_shared<ErrorState>()) {
  if (const char* env = std::getenv("LRT_METAL_TASKS_PER_ENCODER")) {
    int value = 0;
    if (absl::SimpleAtoi(env, &value) && value > 0) {
      tasks_per_encoder_ = value;
    }
  }
}

MlDriftMetalEnvironment::~MlDriftMetalEnvironment() {
  if (initialized()) {
    // Do not destroy resources that are still referenced by in-flight GPU
    // work.
    WaitForCompletion().IgnoreError();
  }
}

absl::Status MlDriftMetalEnvironment::Init() {
  if (initialized()) return absl::OkStatus();
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return absl::FailedPreconditionError(
          "The ml_drift backend needs a Metal GPU and "
          "MTLCreateSystemDefaultDevice() returned nil.");
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      return absl::InternalError("Failed to create an MTLCommandQueue.");
    }
    metal_env_ = std::make_unique<ml_drift::metal::Environment>(device);
    queue_ = queue;
    ABSL_LOG(INFO) << "Initialized ML Drift Metal device: '"
                   << std::string([[device name] UTF8String])
                   << "' (gpu_type="
                   << static_cast<int>(metal_env_->GetInfo().apple_info.gpu_type)
                   << ", compute_units="
                   << metal_env_->GetInfo().GetComputeUnitsCount()
                   << ", tasks_per_encoder=" << tasks_per_encoder_ << ")";
  }
  return absl::OkStatus();
}

id<MTLDevice> MlDriftMetalEnvironment::device() const {
  return metal_env_ != nullptr ? metal_env_->device() : nil;
}

const ml_drift::GpuInfo& MlDriftMetalEnvironment::GetInfo() const {
  return metal_env_->GetInfo();
}

id<MTLCommandBuffer> MlDriftMetalEnvironment::command_buffer() {
  if (open_command_buffer_ == nil) {
    @autoreleasepool {
      open_command_buffer_ = [queue_ commandBuffer];
    }
  }
  return open_command_buffer_;
}

void MlDriftMetalEnvironment::Flush() {
  if (open_command_buffer_ == nil) return;
  @autoreleasepool {
    std::shared_ptr<ErrorState> error_state = error_state_;
    [open_command_buffer_ addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      if (cb.status != MTLCommandBufferStatusError) return;
      NSString* description = cb.error != nil
                                  ? [cb.error localizedDescription]
                                  : @"unknown error";
      absl::MutexLock lock(error_state->mutex);
      if (error_state->first_error.empty()) {
        error_state->first_error = std::string([description UTF8String]);
      }
    }];
    [open_command_buffer_ commit];
    last_committed_command_buffer_ = open_command_buffer_;
    open_command_buffer_ = nil;
  }
}

absl::Status MlDriftMetalEnvironment::WaitForCompletion() {
  Flush();
  if (last_committed_command_buffer_ != nil) {
    @autoreleasepool {
      // Command buffers of one queue complete in commit order, so waiting for
      // the last one waits for all of them.
      [last_committed_command_buffer_ waitUntilCompleted];
      last_committed_command_buffer_ = nil;
    }
  }
  std::string error;
  {
    absl::MutexLock lock(error_state_->mutex);
    error.swap(error_state_->first_error);
  }
  if (!error.empty()) {
    return absl::InternalError(
        absl::StrCat("Metal command buffer failed: ", error));
  }
  return absl::OkStatus();
}

absl::Status MlDriftMetalInferenceContext::Profile(
    id<MTLDevice> device, ml_drift::ProfilingInfo* result) {
  @autoreleasepool {
    ml_drift::metal::InferenceContext::Profile(device, result);
  }
  return absl::OkStatus();
}

bool IsMetalLinearBufferTensor(
    const ml_drift::metal::MetalSpatialTensor& tensor) {
  const ml_drift::TensorStorageType storage_type = tensor.GetStorageType();
  if (storage_type != ml_drift::TensorStorageType::kBuffer &&
      storage_type != ml_drift::TensorStorageType::kImageBuffer) {
    return false;
  }
  id<MTLBuffer> buffer = tensor.GetBufferHandle();
  return buffer != nil && [buffer contents] != nullptr;
}

size_t MetalTensorBufferOffset(
    const ml_drift::metal::MetalSpatialTensor& tensor) {
  id<MTLBuffer> buffer = tensor.GetBufferHandle();
  if (buffer == nil || [buffer contents] == nullptr) return 0;
  return static_cast<size_t>(static_cast<const uint8_t*>(
                                 tensor.GetBufferPointer()) -
                             static_cast<const uint8_t*>([buffer contents]));
}

ml_drift::TensorStorageType MlDriftMetalBackend::GetStorageTypeForBuilder(
    const Environment& env, bool env_initialized) {
  if (!env_initialized || !env.initialized()) {
    return ml_drift::TensorStorageType::kBuffer;
  }
  return ml_drift::metal::GetFastestStorageType(env.GetInfo());
}

absl::Status MlDriftMetalBackend::InitInferenceContext(
    InferenceContext* ctx, const ml_drift::CreateGpuModelInfo& create_info,
    ml_drift::GpuModel* model, Environment* env,
    const ml_drift::ExternalTensorsInfo& external_tensors) {
  if (!env->initialized()) {
    return absl::FailedPreconditionError(
        "MlDriftMetalBackend: environment is not initialized");
  }
  ml_drift::CreateGpuModelInfo info = create_info;
  for (const auto& [id, tensor] : external_tensors.immutable_tensors) {
    info.external_immutable_tensors[id] = tensor;
  }
  for (const auto& [id, desc] : external_tensors.mutable_tensors) {
    info.external_mutable_tensors[id] = desc;
  }
  @autoreleasepool {
    return ctx->InitFromGpuModel(info, model, env->metal_env());
  }
}

absl::Status MlDriftMetalBackend::SetInputTensorFloat(
    InferenceContext* ctx, Environment* env, ml_drift::ValueId id,
    const ml_drift::TensorFloat32& tensor) {
  Tensor* gpu_tensor = ctx->GetTensor(id);
  if (gpu_tensor == nullptr) {
    return absl::NotFoundError(absl::StrCat("No Metal tensor for id ", id));
  }
  ml_drift::TensorDescriptor desc = gpu_tensor->GetDescriptor();
  desc.UploadData(tensor);
  return UploadTensorDescriptorData(gpu_tensor, env, desc);
}

absl::Status MlDriftMetalBackend::SetInputTensorInt32(
    InferenceContext* ctx, Environment* env, ml_drift::ValueId id,
    const ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32>&
        tensor) {
  Tensor* gpu_tensor = ctx->GetTensor(id);
  if (gpu_tensor == nullptr) {
    return absl::NotFoundError(absl::StrCat("No Metal tensor for id ", id));
  }
  ml_drift::TensorDescriptor desc = gpu_tensor->GetDescriptor();
  desc.UploadData(tensor);
  return UploadTensorDescriptorData(gpu_tensor, env, desc);
}

absl::Status MlDriftMetalBackend::GetOutputTensorFloat(
    InferenceContext* ctx, Environment* env, ml_drift::ValueId id,
    ml_drift::TensorFloat32* tensor) {
  // The Metal read path does not synchronize with `env`'s queue.
  LRT_TENSOR_RETURN_IF_ERROR(env->WaitForCompletion());
  if (ctx->GetTensor(id) == nullptr) {
    return absl::NotFoundError(absl::StrCat("No Metal tensor for id ", id));
  }
  @autoreleasepool {
    return ctx->GetOutputTensor(id, tensor);
  }
}

absl::Status MlDriftMetalBackend::GetOutputTensorInt32(
    InferenceContext* ctx, Environment* env, ml_drift::ValueId id,
    ml_drift::TensorInt32* tensor) {
  LRT_TENSOR_RETURN_IF_ERROR(env->WaitForCompletion());
  if (ctx->GetTensor(id) == nullptr) {
    return absl::NotFoundError(absl::StrCat("No Metal tensor for id ", id));
  }
  @autoreleasepool {
    return ctx->GetOutputTensor(id, tensor);
  }
}

absl::Status MlDriftMetalBackend::CreateConstTensor(
    Environment* env, const ml_drift::TensorDescriptor& desc, Tensor* tensor) {
  @autoreleasepool {
    return tensor->CreateFromDescriptor(desc, env->device());
  }
}

absl::Status MlDriftMetalBackend::UploadTensorDescriptorData(
    Tensor* tensor, Environment* env, const ml_drift::TensorDescriptor& desc) {
  if (desc.GetData().size() < tensor->GetMemorySizeInBytes()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "UploadTensorDescriptorData: descriptor holds ", desc.GetData().size(),
        " bytes, tensor needs ", tensor->GetMemorySizeInBytes()));
  }
  // `WriteData` commits its own (blit) command buffer to `queue()`; commit
  // the open one first so that GPU work keeps host program order.
  env->Flush();
  @autoreleasepool {
    return tensor->UploadDescriptorData(desc, env->queue(),
                                        /*wait_for_completion=*/false);
  }
}

absl::Status MlDriftMetalBackend::TensorToDescriptor(
    Tensor* tensor, Environment* env, ml_drift::TensorDescriptor* desc) {
  LRT_TENSOR_RETURN_IF_ERROR(env->WaitForCompletion());
  @autoreleasepool {
    return tensor->ToDescriptor(desc, env->device());
  }
}

absl::Status MlDriftMetalBackend::AddToQueue(InferenceContext* ctx,
                                             Environment* env) {
  if (!env->initialized()) {
    return absl::FailedPreconditionError(
        "MlDriftMetalBackend: environment is not initialized");
  }
  @autoreleasepool {
    ctx->EncodeWithCommandBuffer(env->command_buffer(),
                                 env->tasks_per_encoder());
  }
  return absl::OkStatus();
}

absl::Status MlDriftMetalBackend::CopyBufferToTensor(InferenceContext* ctx,
                                                     Environment* env,
                                                     ml_drift::ValueId id,
                                                     Buffer* generic_buffer) {
  return absl::UnimplementedError(
      "MlDriftMetalBackend::CopyBufferToTensor is not implemented");
}

absl::StatusOr<std::unique_ptr<Buffer>> MlDriftMetalBackend::CreateInputBuffer(
    InferenceContext* ctx, Environment* env, ml_drift::ValueId id) {
  return absl::UnimplementedError(
      "MlDriftMetalBackend::CreateInputBuffer is not implemented");
}

absl::Status MlDriftMetalBackend::CopyTensorToBuffer(InferenceContext* ctx,
                                                     Environment* env,
                                                     ml_drift::ValueId id,
                                                     Buffer* generic_buffer) {
  return absl::UnimplementedError(
      "MlDriftMetalBackend::CopyTensorToBuffer is not implemented");
}

absl::StatusOr<std::shared_ptr<Buffer>>
MlDriftMetalBackend::CreateOutputBuffer(InferenceContext* ctx, Environment* env,
                                        ml_drift::ValueId id) {
  return absl::UnimplementedError(
      "MlDriftMetalBackend::CreateOutputBuffer is not implemented");
}

}  // namespace litert::tensor
