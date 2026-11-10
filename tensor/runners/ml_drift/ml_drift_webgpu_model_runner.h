/* Copyright 2025 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_WEBGPU_MODEL_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_WEBGPU_MODEL_RUNNER_H_

#include <string>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/environment.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/inference_context.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_api_util.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "tensor/buffer.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/runners/ml_drift/ml_drift_model_runner_base.h"

namespace litert::tensor {

struct MlDriftWebGpuBackend {
  using Environment = ml_drift::webgpu::ExecutionEnvironment;
  using InferenceContext = ml_drift::webgpu::InferenceContext;
  using Tensor = ml_drift::webgpu::SpatialTensor;

  static absl::Status InitEnvironment(Environment* env) {
    ABSL_LOG(INFO) << "Initializing WebGPU environment";
    env->RequestExtension("shader_f16");
    return env->Initialize();
  }

  static ml_drift::GpuInfo GetGpuInfoForBuilder(const Environment& env,
                                                bool env_initialized) {
    return env_initialized ? env.GetInfo() : ml_drift::GpuInfo{};
  }

  static ml_drift::TensorStorageType GetStorageTypeForBuilder(
      const Environment& env, bool env_initialized) {
    if (!env_initialized) {
      return ml_drift::TensorStorageType::kBuffer;
    }
    return ml_drift::webgpu::GetFastestStorageType(env.GetInfo());
  }

  static absl::Status InitInferenceContext(
      InferenceContext* ctx, const ml_drift::CreateGpuModelInfo& create_info,
      ml_drift::GpuModel* model, Environment* env) {
    return ctx->InitFromGpuModel(*env, create_info, model);
  }

  static absl::Status SetInputTensorFloat(
      InferenceContext* ctx, Environment* env, ml_drift::ValueId id,
      const ml_drift::TensorFloat32& tensor) {
    return ctx->SetInputTensor(*env, id, tensor);
  }

  static absl::Status SetInputTensorInt32(
      InferenceContext* ctx, Environment* env, ml_drift::ValueId id,
      const ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32>&
          tensor) {
    return ctx->SetInputTensor(*env, id, tensor);
  }

  static absl::Status GetOutputTensorFloat(InferenceContext* ctx,
                                           Environment* env,
                                           ml_drift::ValueId id,
                                           ml_drift::TensorFloat32* tensor) {
    return ctx->GetOutputTensor(*env, id, tensor);
  }

  static absl::Status GetOutputTensorInt32(InferenceContext* ctx,
                                           Environment* env,
                                           ml_drift::ValueId id,
                                           ml_drift::TensorInt32* tensor) {
    return ctx->GetOutputTensor(*env, id, tensor);
  }

  static Tensor* GetTensor(InferenceContext* ctx, ml_drift::ValueId id) {
    return ctx->GetTensor(id);
  }

  static absl::Status UploadTensorDescriptorData(
      Tensor* tensor, Environment* env,
      const ml_drift::TensorDescriptor& desc) {
    return tensor->UploadDescriptorData(env->queue(), desc);
  }

  static absl::Status TensorToDescriptor(Tensor* tensor, Environment* env,
                                         ml_drift::TensorDescriptor* desc) {
    return tensor->ToDescriptor(env->device(), desc);
  }

  static absl::Status AddToQueue(InferenceContext* ctx, Environment* env) {
    return ctx->AddToQueue(*env);
  }

  static absl::Status WaitForCompletion(Environment* env) {
    return ml_drift::webgpu::WaitUntilCompleted(env->queue(), env->device(),
                                                absl::Seconds(10));
  }

  static absl::Status CopyBufferToTensor(InferenceContext* ctx,
                                         Environment* env, ml_drift::ValueId id,
                                         Buffer* generic_buffer) {
    auto* tensor = ctx->GetTensor(id);
    if (!tensor) {
      return absl::InternalError(
          absl::StrCat("Internal tensor not found for id: ", id));
    }
    if (tensor->GetStorageType() != ml_drift::TensorStorageType::kBuffer) {
      return absl::UnimplementedError(
          "WebGPU zero-copy buffer transfer currently requires BUFFER storage "
          "type");
    }
    wgpu::Buffer dst_buffer = tensor->GetBufferHandle();
    if (!dst_buffer) {
      return absl::InternalError("Destination WebGPU tensor buffer is null");
    }

    if (auto litert_buffer = generic_buffer->As<LitertBuffer>();
        litert_buffer.ok()) {
      auto hw_handle = litert_buffer->tensor_buffer().GetWebGpuBuffer();
      if (!hw_handle) {
        return absl::InvalidArgumentError(
            "LitertBuffer does not contain a WebGPU buffer handle");
      }
      wgpu::Buffer src_buffer(static_cast<WGPUBuffer>(*hw_handle));
      if (!src_buffer) {
        return absl::InvalidArgumentError(
            "Source WebGPU buffer handle is null");
      }
      // Zero-copy path: if internal tensor already aliases the provided buffer,
      // no copy or queue submission is needed.
      if (src_buffer.Get() == dst_buffer.Get() && tensor->offset() == 0) {
        return absl::OkStatus();
      }
      // Non-aliased buffer copy fallback.
      // TODO: When multiple non-aliased buffers require copying across a stage
      // boundary, batch copies into a single CommandBuffer rather than
      // submitting to the queue per buffer.
      wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();
      encoder.CopyBufferToBuffer(src_buffer, 0, dst_buffer, tensor->offset(),
                                 tensor->GetMemorySizeInBytes());
      wgpu::CommandBuffer cb = encoder.Finish();
      env->queue().Submit(1, &cb);
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(
        "Unsupported buffer type for MlDriftWebGpuBackend::CopyBufferToTensor");
  }

  static absl::Status CopyTensorToBuffer(InferenceContext* ctx,
                                         Environment* env, ml_drift::ValueId id,
                                         Buffer* generic_buffer) {
    auto* tensor = ctx->GetTensor(id);
    if (!tensor) {
      return absl::InternalError("Internal tensor not found");
    }
    if (tensor->GetStorageType() != ml_drift::TensorStorageType::kBuffer) {
      return absl::UnimplementedError(
          "WebGPU zero-copy buffer transfer currently requires BUFFER storage "
          "type");
    }
    wgpu::Buffer src_buffer = tensor->GetBufferHandle();
    if (!src_buffer) {
      return absl::InternalError("Source WebGPU tensor buffer is null");
    }

    if (auto litert_buffer = generic_buffer->As<LitertBuffer>();
        litert_buffer.ok()) {
      auto hw_handle = litert_buffer->tensor_buffer().GetWebGpuBuffer();
      if (!hw_handle) {
        return absl::InvalidArgumentError(
            "LitertBuffer does not contain a WebGPU buffer handle");
      }
      wgpu::Buffer dst_buffer(static_cast<WGPUBuffer>(*hw_handle));
      if (!dst_buffer) {
        return absl::InvalidArgumentError(
            "Destination WebGPU buffer handle is null");
      }
      // Zero-copy path: if internal tensor already aliases the destination
      // buffer, no copy or queue submission is needed.
      if (src_buffer.Get() == dst_buffer.Get() && tensor->offset() == 0) {
        return absl::OkStatus();
      }
      // Non-aliased buffer copy fallback.
      // TODO: When multiple non-aliased buffers require copying across a stage
      // boundary, batch copies into a single CommandBuffer rather than
      // submitting to the queue per buffer.
      wgpu::CommandEncoder encoder = env->device().CreateCommandEncoder();
      encoder.CopyBufferToBuffer(src_buffer, tensor->offset(), dst_buffer, 0,
                                 tensor->GetMemorySizeInBytes());
      wgpu::CommandBuffer cb = encoder.Finish();
      env->queue().Submit(1, &cb);
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(
        "Unsupported buffer type for MlDriftWebGpuBackend::CopyTensorToBuffer");
  }
};

template <typename ModelFunctor, typename Inputs, typename Outputs>
using MlDriftWebGpuModelRunner =
    MlDriftModelRunnerBase<MlDriftWebGpuBackend, ModelFunctor, Inputs, Outputs>;

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_WEBGPU_MODEL_RUNNER_H_
