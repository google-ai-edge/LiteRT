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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_CL_MODEL_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_CL_MODEL_RUNNER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/inference_context.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/cl/util.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "tensor/backends/ml_drift/ml_drift_buffer.h"
#include "tensor/buffer.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/runners/ml_drift/ml_drift_model_runner_base.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

struct MlDriftClBackend {
  using Environment = ml_drift::cl::Environment;
  using InferenceContext = ml_drift::cl::InferenceContext;
  using Tensor = ml_drift::cl::Tensor;

  static absl::Status InitEnvironment(Environment* env) {
    ABSL_LOG(INFO) << "Loading OpenCL and creating environment";
    LRT_TENSOR_RETURN_IF_ERROR(ml_drift::cl::LoadOpenCL());
    return ml_drift::cl::CreateEnvironment(env);
  }

  static ml_drift::GpuInfo GetGpuInfoForBuilder(Environment& env,
                                                bool env_initialized) {
    if (env_initialized) {
      return env.device().GetInfo();
    }
    return ml_drift::GpuInfo();
  }

  static ml_drift::TensorStorageType GetStorageTypeForBuilder(
      const Environment& env, bool /*env_initialized*/) {
    return ml_drift::cl::GetFastestStorageType(env.device().GetInfo());
  }

  static absl::Status InitInferenceContext(
      InferenceContext* ctx, const ml_drift::CreateGpuModelInfo& create_info,
      ml_drift::GpuModel* model, Environment* env,
      const ml_drift::ExternalTensorsInfo& external_tensors = {}) {
    ml_drift::cl::CreateInferenceInfo cii;
    cii.options = ml_drift::cl::CreateFromModelHints(create_info.hints);
    cii.external_tensors = external_tensors;
    return ctx->InitFromGpuModel(cii, model, env);
  }

  static absl::Status SetInputTensorFloat(
      InferenceContext* ctx, Environment* env, ml_drift::ValueId id,
      const ml_drift::TensorFloat32& tensor) {
    return ctx->SetInputTensor(id, tensor, env->queue());
  }

  static absl::Status SetInputTensorInt32(
      InferenceContext* ctx, Environment* env, ml_drift::ValueId id,
      const ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kInt32>&
          tensor) {
    return ctx->SetInputTensor(id, tensor, env->queue());
  }

  static absl::Status GetOutputTensorFloat(InferenceContext* ctx,
                                           Environment* env,
                                           ml_drift::ValueId id,
                                           ml_drift::TensorFloat32* tensor) {
    return ctx->GetOutputTensor(id, env->queue(), tensor);
  }

  static absl::Status GetOutputTensorInt32(InferenceContext* ctx,
                                           Environment* env,
                                           ml_drift::ValueId id,
                                           ml_drift::TensorInt32* tensor) {
    return ctx->GetOutputTensor(id, env->queue(), tensor);
  }

  static Tensor* GetTensor(InferenceContext* ctx, ml_drift::ValueId id) {
    return ctx->GetTensor(id);
  }

  static absl::Status UploadTensorDescriptorData(
      Tensor* tensor, Environment* env,
      const ml_drift::TensorDescriptor& desc) {
    return tensor->UploadDescriptorData(desc, env->queue());
  }

  static absl::Status TensorToDescriptor(Tensor* tensor, Environment* env,
                                         ml_drift::TensorDescriptor* desc) {
    return tensor->ToDescriptor(desc, env->queue());
  }

  static absl::Status AddToQueue(InferenceContext* ctx, Environment* env) {
    return ctx->AddToQueue(env->queue());
  }

  static absl::Status WaitForCompletion(Environment* env) {
    return env->queue()->WaitForCompletion();
  }

  static absl::Status CopyBufferToTensor(InferenceContext* ctx,
                                         Environment* env, ml_drift::ValueId id,
                                         Buffer* generic_buffer) {
    cl_mem src_mem = nullptr;
    if (auto mld_buffer = generic_buffer->As<MlDriftBuffer>();
        mld_buffer.ok()) {
      src_mem = mld_buffer->GetMemoryForWriting();
    } else if (auto litert_buffer = generic_buffer->As<LitertBuffer>();
               litert_buffer.ok()) {
      auto cl_mem_res = litert_buffer->tensor_buffer().GetOpenClMemory();
      if (!cl_mem_res) {
        return absl::InvalidArgumentError(
            "LitertBuffer does not contain an OpenCL memory handle");
      }
      src_mem = reinterpret_cast<cl_mem>(*cl_mem_res);
    } else {
      return absl::InvalidArgumentError(
          "Unsupported buffer type for MlDriftClBackend::CopyBufferToTensor");
    }
    auto* tensor = ctx->GetTensor(id);
    if (!tensor) {
      return absl::InternalError(
          absl::StrCat("Internal tensor not found for id: ", id));
    }
    auto dst_mem = tensor->GetMemoryPtrForWriting();
    if (!src_mem || !dst_mem) {
      return absl::InternalError("Buffer memory is null");
    }
    if (src_mem == dst_mem) return absl::OkStatus();

    cl_int error_code;
    auto storage_type = tensor->GetStorageType();
    if (storage_type == ml_drift::TensorStorageType::kBuffer ||
        storage_type == ml_drift::TensorStorageType::kImageBuffer) {
      error_code = ml_drift::cl::clEnqueueCopyBuffer(
          env->queue()->queue(), src_mem, dst_mem, 0, 0,
          tensor->GetMemorySizeInBytes(), 0, nullptr, nullptr);
    } else {
      auto region3d = tensor->GetDescriptor().GetFullTensorRegion();
      size_t region[3] = {static_cast<size_t>(region3d.x),
                          static_cast<size_t>(region3d.y),
                          static_cast<size_t>(region3d.z)};
      size_t origin[3] = {0, 0, 0};
      error_code = ml_drift::cl::clEnqueueCopyBufferToImage(
          env->queue()->queue(), src_mem, dst_mem, 0, origin, region, 0,
          nullptr, nullptr);
    }
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(
          absl::StrCat("Failed to copy buffer to tensor - ",
                       ml_drift::cl::CLErrorCodeToString(error_code)));
    }
    return absl::OkStatus();
  }

  static absl::StatusOr<std::unique_ptr<Buffer>> CreateInputBuffer(
      InferenceContext* ctx, Environment* env, ml_drift::ValueId id) {
    auto* tensor = ctx->GetTensor(id);
    if (!tensor) {
      return absl::InternalError("Internal tensor not found");
    }
    size_t size_bytes = tensor->GetMemorySizeInBytes();
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto buf, MlDriftBuffer::Create(id, *ctx, *env->queue(), size_bytes));
    return std::unique_ptr<Buffer>(std::move(buf));
  }

  static absl::Status CopyTensorToBuffer(InferenceContext* ctx,
                                         Environment* env, ml_drift::ValueId id,
                                         Buffer* generic_buffer) {
    auto* tensor = ctx->GetTensor(id);
    if (!tensor) {
      return absl::InternalError("Internal tensor not found");
    }
    auto src_mem = tensor->GetMemoryPtrForWriting();
    cl_mem dst_mem = nullptr;
    const Tensor* dst_tensor = nullptr;
    MlDriftBuffer* mld_buffer_ptr = nullptr;
    if (auto mld_buffer = generic_buffer->As<MlDriftBuffer>();
        mld_buffer.ok()) {
      mld_buffer_ptr = &(*mld_buffer);
      dst_mem = mld_buffer_ptr->GetMemoryForWriting();
      dst_tensor = mld_buffer_ptr->context()
                       ? mld_buffer_ptr->context()->GetTensor(
                             mld_buffer_ptr->value_id())
                       : nullptr;
    } else if (auto litert_buffer = generic_buffer->As<LitertBuffer>();
               litert_buffer.ok()) {
      auto cl_mem_res = litert_buffer->tensor_buffer().GetOpenClMemory();
      if (!cl_mem_res) {
        return absl::InvalidArgumentError(
            "LitertBuffer does not contain an OpenCL memory handle");
      }
      dst_mem = reinterpret_cast<cl_mem>(*cl_mem_res);
    } else {
      return absl::InvalidArgumentError(
          "Unsupported buffer type for MlDriftClBackend::CopyTensorToBuffer");
    }
    if (src_mem == dst_mem) {
      return absl::OkStatus();
    }

    auto IsImage = [](ml_drift::TensorStorageType type) {
      return type == ml_drift::TensorStorageType::kTexture2D ||
             type == ml_drift::TensorStorageType::kTexture3D ||
             type == ml_drift::TensorStorageType::kTextureArray ||
             type == ml_drift::TensorStorageType::kSingleTexture2D;
    };
    bool src_is_image = IsImage(tensor->GetStorageType());
    bool dst_is_image = false;
    if (dst_tensor) {
      dst_is_image = IsImage(dst_tensor->GetStorageType());
    } else if (dst_mem) {
      cl_mem_object_type mem_type = 0;
      if (ml_drift::cl::clGetMemObjectInfo(
              dst_mem, CL_MEM_TYPE, sizeof(mem_type), &mem_type, nullptr) ==
          CL_SUCCESS) {
        dst_is_image = (mem_type == CL_MEM_OBJECT_IMAGE2D ||
                        mem_type == CL_MEM_OBJECT_IMAGE3D ||
                        mem_type == CL_MEM_OBJECT_IMAGE2D_ARRAY ||
                        mem_type == CL_MEM_OBJECT_IMAGE1D ||
                        mem_type == CL_MEM_OBJECT_IMAGE1D_ARRAY ||
                        mem_type == CL_MEM_OBJECT_IMAGE1D_BUFFER);
      }
    }

    size_t origin[3] = {0, 0, 0};
    auto region3d = tensor->GetDescriptor().GetFullTensorRegion();
    size_t region[3] = {static_cast<size_t>(region3d.x),
                        static_cast<size_t>(region3d.y),
                        static_cast<size_t>(region3d.z)};

    cl_int error_code;
    if (!src_is_image && !dst_is_image) {
      error_code = ml_drift::cl::clEnqueueCopyBuffer(
          env->queue()->queue(), src_mem, dst_mem, 0, 0,
          tensor->GetMemorySizeInBytes(), 0, nullptr, nullptr);
    } else if (src_is_image && !dst_is_image) {
      error_code = ml_drift::cl::clEnqueueCopyImageToBuffer(
          env->queue()->queue(), src_mem, dst_mem, origin, region, 0, 0,
          nullptr, nullptr);
    } else if (!src_is_image && dst_is_image) {
      error_code = ml_drift::cl::clEnqueueCopyBufferToImage(
          env->queue()->queue(), src_mem, dst_mem, 0, origin, region, 0,
          nullptr, nullptr);
    } else {
      error_code = ml_drift::cl::clEnqueueCopyImage(
          env->queue()->queue(), src_mem, dst_mem, origin, origin, region, 0,
          nullptr, nullptr);
    }

    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(
          absl::StrCat("Failed to copy tensor to buffer - ",
                       ml_drift::cl::CLErrorCodeToString(error_code)));
    }
    if (mld_buffer_ptr) {
      mld_buffer_ptr->Invalidate();
    }
    return absl::OkStatus();
  }

  static absl::StatusOr<std::shared_ptr<Buffer>> CreateOutputBuffer(
      InferenceContext* ctx, Environment* env, ml_drift::ValueId id) {
    auto* tensor = ctx->GetTensor(id);
    if (!tensor) {
      return absl::InternalError("Internal tensor not found");
    }
    size_t size_bytes = tensor->GetMemorySizeInBytes();
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto buf, MlDriftBuffer::Create(id, *ctx, *env->queue(), size_bytes));
    return std::shared_ptr<Buffer>(std::move(buf));
  }
};

template <typename ModelFunctor, typename Inputs, typename Outputs>
class MlDriftModelRunner
    : public MlDriftModelRunnerBase<MlDriftClBackend, ModelFunctor, Inputs,
                                    Outputs> {
 public:
  using Base =
      MlDriftModelRunnerBase<MlDriftClBackend, ModelFunctor, Inputs, Outputs>;
  using Base::Base;
  using Base::GetOutput;
  using Base::SetInput;

  absl::StatusOr<std::unique_ptr<MlDriftBuffer>> AllocateBuffer(
      size_t size_bytes) {
    return MlDriftBuffer::Create(this->env().context(), *this->env().queue(),
                                 size_bytes);
  }

  absl::Status SetInput(const std::string& name, MlDriftBuffer& buffer);
  absl::Status GetOutput(const std::string& name, MlDriftBuffer& buffer);
  absl::StatusOr<std::unique_ptr<MlDriftBuffer>> GetInputBuffer(
      const std::string& name);

  absl::StatusOr<ml_drift::ProfilingInfo> Profile() {
    LRT_TENSOR_RETURN_IF_ERROR(this->EnsureEnvInitialized());
    ml_drift::ProfilingInfo profiling_info;
    LRT_TENSOR_RETURN_IF_ERROR(this->inference_context().Profile(
        this->env().profiling_queue(), &profiling_info));
    return profiling_info;
  }
};

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status MlDriftModelRunner<ModelFunctor, Inputs, Outputs>::SetInput(
    const std::string& name, MlDriftBuffer& buffer) {
  LRT_TENSOR_RETURN_IF_ERROR(this->EnsureEnvInitialized());
  const auto& tensors = this->cached_input_tensors_;
  if (!tensors.contains(name)) {
    return absl::NotFoundError(absl::StrCat("Input tensor not found: ", name));
  }
  const auto raw_tensor_ptr = tensors.at(name)->GetRaw();
  auto it = this->conversion_ctx().graph_tensor_to_handle.find(raw_tensor_ptr);
  if (it == this->conversion_ctx().graph_tensor_to_handle.end()) {
    return absl::NotFoundError(
        absl::StrCat("Input tensor handle not found: ", name));
  }
  const auto& th = it->second;
  auto* tensor = this->inference_context().GetTensor(th.id);
  if (!tensor) {
    return absl::InternalError(
        absl::StrCat("Internal tensor not found for id: ", th.id));
  }
  auto src_mem = buffer.GetMemoryForWriting();
  auto dst_mem = tensor->GetMemoryPtrForWriting();
  if (!src_mem) {
    return absl::InternalError("Source buffer memory is null");
  }
  if (!dst_mem) {
    return absl::InternalError(
        absl::StrCat("Destination tensor memory is null for id: ", th.id));
  }
  if (src_mem == dst_mem) {
    return absl::OkStatus();
  }

  cl_int error_code;
  auto storage_type = tensor->GetStorageType();
  if (storage_type == ml_drift::TensorStorageType::kBuffer ||
      storage_type == ml_drift::TensorStorageType::kImageBuffer) {
    error_code = ml_drift::cl::clEnqueueCopyBuffer(
        this->env().queue()->queue(), src_mem, dst_mem, 0, 0,
        tensor->GetMemorySizeInBytes(), 0, nullptr, nullptr);
  } else {
    auto region3d = tensor->GetDescriptor().GetFullTensorRegion();
    size_t region[3] = {static_cast<size_t>(region3d.x),
                        static_cast<size_t>(region3d.y),
                        static_cast<size_t>(region3d.z)};
    size_t origin[3] = {0, 0, 0};
    error_code = ml_drift::cl::clEnqueueCopyBufferToImage(
        this->env().queue()->queue(), src_mem, dst_mem, 0, origin, region, 0,
        nullptr, nullptr);
  }
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to copy buffer to tensor - ",
                     ml_drift::cl::CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status MlDriftModelRunner<ModelFunctor, Inputs, Outputs>::GetOutput(
    const std::string& name, MlDriftBuffer& buffer) {
  LRT_TENSOR_RETURN_IF_ERROR(this->EnsureEnvInitialized());
  if (!this->name_to_tensor_map().contains(name)) {
    return absl::NotFoundError(absl::StrCat("Output tensor not found: ", name));
  }
  const auto& tensor_obj = this->name_to_tensor_map().at(name);
  if (!this->conversion_ctx().graph_tensor_to_handle.contains(tensor_obj)) {
    return absl::NotFoundError(absl::StrCat(
        "Output tensor ", name, " was not found in the ML Drift model map."));
  }
  const auto& tensor_handle =
      this->conversion_ctx().graph_tensor_to_handle.at(tensor_obj);
  auto* tensor = this->inference_context().GetTensor(tensor_handle.id);
  if (!tensor) {
    return absl::InternalError("Internal tensor not found");
  }
  auto src_mem = tensor->GetMemoryPtrForWriting();
  auto dst_mem = buffer.GetMemoryForWriting();
  auto* dst_tensor = buffer.context()
                         ? buffer.context()->GetTensor(buffer.value_id())
                         : nullptr;
  if (src_mem == dst_mem) {
    return absl::OkStatus();
  }

  auto IsImage = [](ml_drift::TensorStorageType type) {
    return type == ml_drift::TensorStorageType::kTexture2D ||
           type == ml_drift::TensorStorageType::kTexture3D ||
           type == ml_drift::TensorStorageType::kTextureArray ||
           type == ml_drift::TensorStorageType::kSingleTexture2D;
  };
  auto dst_storage_type = dst_tensor ? dst_tensor->GetStorageType()
                                     : ml_drift::TensorStorageType::kBuffer;
  bool src_is_image = IsImage(tensor->GetStorageType());
  bool dst_is_image = IsImage(dst_storage_type);

  size_t origin[3] = {0, 0, 0};
  auto region3d = tensor->GetDescriptor().GetFullTensorRegion();
  size_t region[3] = {static_cast<size_t>(region3d.x),
                      static_cast<size_t>(region3d.y),
                      static_cast<size_t>(region3d.z)};

  cl_int error_code;
  if (!src_is_image && !dst_is_image) {
    error_code = ml_drift::cl::clEnqueueCopyBuffer(
        this->env().queue()->queue(), src_mem, dst_mem, 0, 0,
        tensor->GetMemorySizeInBytes(), 0, nullptr, nullptr);
  } else if (src_is_image && !dst_is_image) {
    error_code = ml_drift::cl::clEnqueueCopyImageToBuffer(
        this->env().queue()->queue(), src_mem, dst_mem, origin, region, 0, 0,
        nullptr, nullptr);
  } else if (!src_is_image && dst_is_image) {
    error_code = ml_drift::cl::clEnqueueCopyBufferToImage(
        this->env().queue()->queue(), src_mem, dst_mem, 0, origin, region, 0,
        nullptr, nullptr);
  } else {
    error_code = ml_drift::cl::clEnqueueCopyImage(
        this->env().queue()->queue(), src_mem, dst_mem, origin, origin, region,
        0, nullptr, nullptr);
  }

  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to copy tensor to buffer - ",
                     ml_drift::cl::CLErrorCodeToString(error_code)));
  }
  buffer.Invalidate();
  return absl::OkStatus();
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::StatusOr<std::unique_ptr<MlDriftBuffer>>
MlDriftModelRunner<ModelFunctor, Inputs, Outputs>::GetInputBuffer(
    const std::string& name) {
  LRT_TENSOR_RETURN_IF_ERROR(this->EnsureEnvInitialized());
  const auto& tensors = this->cached_input_tensors_;
  if (!tensors.contains(name)) {
    return absl::NotFoundError(absl::StrCat("Input tensor not found: ", name));
  }
  const auto raw_tensor_ptr = tensors.at(name)->GetRaw();
  auto it = this->conversion_ctx().graph_tensor_to_handle.find(raw_tensor_ptr);
  if (it == this->conversion_ctx().graph_tensor_to_handle.end()) {
    return absl::NotFoundError(
        absl::StrCat("Input tensor handle not found: ", name));
  }
  const auto& th = it->second;
  auto* tensor = this->inference_context().GetTensor(th.id);
  if (!tensor) {
    return absl::InternalError(
        absl::StrCat("Internal tensor not found for id: ", th.id));
  }
  return MlDriftBuffer::Create(th.id, this->inference_context(),
                               *this->env().queue(),
                               tensor->GetMemorySizeInBytes());
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_CL_MODEL_RUNNER_H_
