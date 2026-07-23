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

#include "ml_drift_delegate/delegate/delegate_kernel_litert.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_object_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "ml_drift_delegate/delegate/delegate_kernel.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/tflite/model_builder_helper.h"
#include "tflite/core/c/common.h"

namespace litert::ml_drift {
namespace {

// Creates TensorDescriptor from BHWC shape and data type, if the batch size is
// 1, then the tensor is created as HWC tensor, otherwise it is created as
// BHWC tensor.
absl::StatusOr<::ml_drift::TensorDescriptor> CreateTensorDescriptor(
    const ::ml_drift::GpuInfo& gpu_info,
    ::ml_drift::TensorRef<::ml_drift::StrongShape<::ml_drift::Layout::BHWC>>&
        tensor,
    ::ml_drift::CalculationsPrecision calculation_precision,
    ::ml_drift::TensorStorageType storage_type) {
  ::ml_drift::BHWC shape = tensor.shape;
  ::ml_drift::DataType data_type = tensor.type;
  if (data_type == ::ml_drift::DataType::FLOAT32) {
    data_type = DeduceDataTypeFromPrecision(calculation_precision);
  }

  auto tensor_desc =
      (shape.b == 1)
          ? ::ml_drift::CreateHwcTensorDescriptor(
                data_type, storage_type,
                ::ml_drift::HWC(shape.h, shape.w, shape.c))
          : ::ml_drift::CreateBhwcTensorDescriptor(
                data_type, storage_type,
                ::ml_drift::BHWC(shape.b, shape.h, shape.w, shape.c));
  ABSL_RETURN_IF_ERROR(tensor_desc.UpdateToSupportedStorageType(
      gpu_info, tensor_desc.GetBHWCShape()));
  return tensor_desc;
}

Expected<LiteRtEnvironment> GetEnvironment(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtExternalLiteRtBufferContext buffer_context) {
  LiteRtEnvironment env;
  LITERT_RETURN_IF_ERROR(
      runtime_context->external_litert_buffer_context_get_environment(
          buffer_context, &env));
  return env;
}

Expected<GpuTensorBufferPtr> GetTensorBuffer(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtExternalLiteRtBufferContext buffer_context,
    const TfLiteTensor* tflite_tensor) {
  LiteRtTensorBuffer tensor_buffer;
  LITERT_RETURN_IF_ERROR(
      runtime_context->get_external_litert_buffer_context_tensor_buffer(
          buffer_context, tflite_tensor, &tensor_buffer));
  return GpuTensorBufferPtr(
      tensor_buffer,
      GpuTensorBufferDeleter{runtime_context->destroy_tensor_buffer});
}

Expected<GpuTensorBufferPtr> AllocateTensorBuffer(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtExternalLiteRtBufferContext buffer_context,
    const TfLiteTensor* tflite_tensor) {
  // Since it is always called right after GetTensorBuffer() fails, no need to
  // check if the tensor buffer already exists.
  LiteRtTensorBuffer tensor_buffer_rawptr;
  LITERT_RETURN_IF_ERROR(
      runtime_context->external_litert_buffer_context_create_tensor_buffer(
          buffer_context, tflite_tensor, &tensor_buffer_rawptr));
  auto tensor_buffer = GpuTensorBufferPtr(
      tensor_buffer_rawptr,
      GpuTensorBufferDeleter{runtime_context->destroy_tensor_buffer});
  // Note that the buffer context takes ownership of the tensor buffer passed.
  LITERT_RETURN_IF_ERROR(
      runtime_context->external_litert_buffer_context_register_tensor_buffer(
          buffer_context, tflite_tensor, tensor_buffer.get()));
  // Release the raw pointer since it's owned by the buffer context now.
  tensor_buffer.release();
  return GetTensorBuffer(runtime_context, buffer_context, tflite_tensor);
}

Expected<void> CopyFromTensor(const LiteRtRuntimeContext* runtime_context,
                              GpuTensorBufferPtr& tensor_buffer,
                              const TfLiteTensor* tflite_tensor) {
  if (tflite_tensor && tflite_tensor->data.raw != nullptr) {
    size_t buffer_size;
    LITERT_RETURN_IF_ERROR(runtime_context->get_tensor_buffer_size(
        tensor_buffer.get(), &buffer_size));
    void* memory;
    LITERT_RETURN_IF_ERROR(runtime_context->lock_tensor_buffer(
        tensor_buffer.get(), &memory, kLiteRtTensorBufferLockModeWrite));
    size_t bytes_to_copy = std::min(tflite_tensor->bytes, buffer_size);
    if (tflite_tensor->bytes != buffer_size) {
      ABSL_LOG_FIRST_N(WARNING, 10)
          << "TFLite tensor size (" << tflite_tensor->bytes
          << ") is different from LiteRT buffer size (" << buffer_size << ").";
    }
    std::memcpy(memory, tflite_tensor->data.raw, bytes_to_copy);
    LITERT_RETURN_IF_ERROR(
        runtime_context->unlock_tensor_buffer(tensor_buffer.get()));
  }
  return {};
}

}  // namespace

absl::StatusOr<DelegateKernelLiteRt*> DelegateKernelLiteRt::Create(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params) {
  auto delegate_data =
      reinterpret_cast<MlDriftDelegateData*>(delegate_params->delegate->data_);
  if (delegate_data == nullptr || delegate_data->options == nullptr ||
      delegate_data->options->runtime_context == nullptr) {
    return absl::InternalError(
        "Missing runtime context for DelegateKernelLiteRt.");
  }
  auto delegate_kernel = std::make_unique<DelegateKernelLiteRt>(
      delegate_data->options->runtime_context);

  ABSL_RETURN_IF_ERROR(delegate_kernel->Initialize(context, delegate_params));
  if (delegate_kernel->NoExternalTensorsMode()) {
    ABSL_RETURN_IF_ERROR(delegate_kernel->InitTensorConverters(context));
  }

  delegate_kernel->buffer_context_ =
      reinterpret_cast<LiteRtExternalLiteRtBufferContext>(
          context->GetExternalContext(context, kTfLiteLiteRtBufferContext));

  return delegate_kernel.release();
}

absl::Status DelegateKernelLiteRt::BindGpuMemoryToInferenceContext(
    ::ml_drift::ValueId tensor_id,
    const ::ml_drift::TensorDescriptor& tensor_desc, GpuMemoryHandle gpu_memory,
    absl::flat_hash_map<GpuMemoryHandle, std::unique_ptr<GpuTensorWrapper>>&
        tensors) {
  auto it = tensors.find(gpu_memory);
  if (it != tensors.end()) {
    auto* gpu_tensor = it->second.get();
    ABSL_RETURN_IF_ERROR(
        ctx_->BindSpatialTensor(tensor_id, &gpu_tensor->Get()));
  } else {
    ABSL_ASSIGN_OR_RETURN(auto gpu_tensor, backend_->CreateTensorWrapper(
                                               tensor_desc, gpu_memory));
    ABSL_RETURN_IF_ERROR(
        ctx_->BindSpatialTensor(tensor_id, &gpu_tensor->Get()));
    tensors[gpu_memory] = std::move(gpu_tensor);
  }
  return absl::OkStatus();
}

// Create internal SpatialTensor from I/O TensorBuffers and bind them to the
// inference context. The created SpatialTensors are stored in
// input_tensors_ and output_tensors_.
absl::Status DelegateKernelLiteRt::BindTensorBuffers(TfLiteContext* context) {
  for (int i = 0; i < input_indices_.size(); ++i) {
    if (IsExternalSharedConstantTensor(input_ids_[i])) {
      continue;
    }
    TfLiteTensor* tflite_tensor = &context->tensors[input_indices_[i]];
    auto tensor_buffer =
        GetTensorBuffer(runtime_context_, buffer_context_, tflite_tensor);
    if (!tensor_buffer) {
      LITERT_LOG(LITERT_VERBOSE,
                 "GPU TensorBuffer not found for %s Tensor",
                 tflite_tensor->name);
      tensor_buffer = AllocateTensorBuffer(runtime_context_, buffer_context_,
                                           tflite_tensor);
      LITERT_RETURN_IF_ERROR(tensor_buffer.HasValue());
      LITERT_RETURN_IF_ERROR(
          CopyFromTensor(runtime_context_, *tensor_buffer, tflite_tensor));
    }
    LITERT_ASSIGN_OR_RETURN(
        auto gpu_memory, backend_->GetGpuMemoryAllocated(*tensor_buffer),
        _ << absl::StrCat("Input#", i, " tensor does not have a GPU Memory."));
    ABSL_RETURN_IF_ERROR(BindGpuMemoryToInferenceContext(
        input_ids_[i], input_tensor_descriptors_[i], gpu_memory,
        input_tensors_));
  }

  for (int i = 0; i < output_indices_.size(); ++i) {
    TfLiteTensor* tflite_tensor = &context->tensors[output_indices_[i]];
    auto tensor_buffer =
        GetTensorBuffer(runtime_context_, buffer_context_, tflite_tensor);
    if (!tensor_buffer) {
      tensor_buffer = AllocateTensorBuffer(runtime_context_, buffer_context_,
                                           tflite_tensor);
      LITERT_RETURN_IF_ERROR(tensor_buffer.HasValue());
    }
    LITERT_ASSIGN_OR_RETURN(
        auto gpu_memory, backend_->GetGpuMemoryAllocated(*tensor_buffer),
        _ << absl::StrCat("Output#", i, " tensor does not have a GPU Memory."));
    ABSL_RETURN_IF_ERROR(BindGpuMemoryToInferenceContext(
        output_ids_[i], output_tensor_descriptors_[i], gpu_memory,
        output_tensors_));
  }
  return absl::OkStatus();
}

absl::Status DelegateKernelLiteRt::UploadOrBindTensorBuffer(
    TfLiteContext* context) {
  struct InputPrepInfo {
    int index;
    GpuMemoryHandle gpu_memory;
  };
  std::vector<InputPrepInfo> prep_infos;
  prep_infos.reserve(input_indices_.size());

  // Allocate, upload, and register inputs.
  for (int i = 0; i < input_indices_.size(); ++i) {
    if (IsExternalSharedConstantTensor(input_ids_[i])) {
      continue;
    }
    TfLiteTensor* tflite_tensor = &context->tensors[input_indices_[i]];
    auto tensor_buffer =
        GetTensorBuffer(runtime_context_, buffer_context_, tflite_tensor);
    if (!tensor_buffer) {
      LITERT_LOG(LITERT_VERBOSE, "GPU TensorBuffer not found for %s Tensor",
                 tflite_tensor->name);
      tensor_buffer = AllocateTensorBuffer(runtime_context_, buffer_context_,
                                           tflite_tensor);
      LITERT_RETURN_IF_ERROR(tensor_buffer.HasValue());
      input_needs_upload_.insert(i);
    }
    if (input_needs_upload_.contains(i)) {
      LITERT_RETURN_IF_ERROR(
          CopyFromTensor(runtime_context_, *tensor_buffer, tflite_tensor));
    }
    LITERT_ASSIGN_OR_RETURN(
        auto gpu_memory, backend_->GetGpuMemoryAllocated(*tensor_buffer),
        _ << absl::StrCat("Input#", i, " tensor does not have a GPU Memory."));
    LiteRtTensorBufferType buffer_type;
    LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_type(
        tensor_buffer->get(), &buffer_type));
    if (is_opencl_backend_ && buffer_type == kLiteRtTensorBufferTypeGlBuffer) {
      tensors_to_flush_[tflite_tensor] = gpu_memory;
    }

    // Push back inputs for use during conversion.
    prep_infos.push_back({i, gpu_memory});
  }

  // Allocate and register outputs.
  for (int i = 0; i < output_indices_.size(); ++i) {
    TfLiteTensor* tflite_tensor = &context->tensors[output_indices_[i]];
    auto tensor_buffer =
        GetTensorBuffer(runtime_context_, buffer_context_, tflite_tensor);
    if (!tensor_buffer) {
      tensor_buffer = AllocateTensorBuffer(runtime_context_, buffer_context_,
                                           tflite_tensor);
      LITERT_RETURN_IF_ERROR(tensor_buffer.HasValue());
    }
    // Note: GetGpuMemoryAllocated has side-effect of registering memory with GL
    // interop fabric and must always be called.
    LITERT_ASSIGN_OR_RETURN(
        auto gpu_memory, backend_->GetGpuMemoryAllocated(*tensor_buffer),
        _ << absl::StrCat("Output#", i, " tensor does not have a GPU Memory."));
    LiteRtTensorBufferType buffer_type;
    LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_type(
        tensor_buffer->get(), &buffer_type));
    if (is_opencl_backend_ && buffer_type == kLiteRtTensorBufferTypeGlBuffer) {
      tensors_to_flush_[tflite_tensor] = gpu_memory;
    }
    if (external_tensor_ids_.contains(output_ids_[i])) {
      // Use direct Tensor binding for external tensors even with no external
      // tensors mode.
      ABSL_RETURN_IF_ERROR(BindGpuMemoryToInferenceContext(
          output_ids_[i], output_tensor_descriptors_[i], gpu_memory,
          output_tensors_));
    }
  }

  // Handle input events for Metal backend before conversion.
  // TODO: b/537754749 - Refactor delegate kernel to order operations correctly
  // for all backends.
  if (is_metal_backend_) {
    ABSL_RETURN_IF_ERROR(HandleInputEvents(context));
  }

  // PreConvert after all inputs and outputs are registered.
  ABSL_RETURN_IF_ERROR(ctx_->PreConvert(/*input=*/true));

  // Enqueue conversion kernels for inputs.
  for (const auto& [i, gpu_memory] : prep_infos) {
    if (external_tensor_ids_.contains(input_ids_[i])) {
      // Use direct Tensor binding for external tensors even with no external
      // tensors mode.
      ABSL_RETURN_IF_ERROR(BindGpuMemoryToInferenceContext(
          input_ids_[i], input_tensor_descriptors_[i], gpu_memory,
          input_tensors_));
    } else {
      // Normal GPU to GPU copy for no external tensors mode.
      ABSL_ASSIGN_OR_RETURN(auto src_buffer,
                            backend_->CreateIOBuffer(gpu_memory));
      ABSL_ASSIGN_OR_RETURN(auto dst_tensor,
                            ctx_->GetSpatialTensor(input_ids_[i]));
      ABSL_RETURN_IF_ERROR(
          input_converters_[i]->Convert(*src_buffer, *dst_tensor));
    }
  }

  ABSL_RETURN_IF_ERROR(ctx_->PostConvert(/*input=*/true));
  return absl::OkStatus();
}

absl::Status DelegateKernelLiteRt::DownloadGpuMemoryToTensorBufferGpuMemory(
    TfLiteContext* context) {
  struct OutputPrepInfo {
    int index;
    GpuMemoryHandle gpu_memory;
  };
  std::vector<OutputPrepInfo> prep_infos;
  prep_infos.reserve(output_ids_.size());

  // Get outputs.
  for (int i = 0; i < output_ids_.size(); ++i) {
    if (external_tensor_ids_.contains(output_ids_[i])) {
      // External tensors don't need to download since they're updated directly
      // in the inference context.
      continue;
    }
    TfLiteTensor* tflite_tensor = &context->tensors[output_indices_[i]];
    LITERT_ASSIGN_OR_RETURN(
        auto tensor_buffer,
        GetTensorBuffer(runtime_context_, buffer_context_, tflite_tensor),
        _ << absl::StrCat("Output#", i, " doesn't have a tensor buffer."));
    LITERT_ASSIGN_OR_RETURN(
        auto gpu_memory, backend_->GetGpuMemoryAllocated(tensor_buffer),
        _ << absl::StrCat("Output#", i, " tensor does not have a GPU Memory."));

    prep_infos.push_back({i, gpu_memory});
  }

  ABSL_RETURN_IF_ERROR(ctx_->PreConvert(/*input=*/false));

  // Enqueue conversion kernels for outputs.
  for (const auto& [i, gpu_memory] : prep_infos) {
    ABSL_ASSIGN_OR_RETURN(auto dst_buffer,
                          backend_->CreateIOBuffer(gpu_memory));
    ABSL_ASSIGN_OR_RETURN(auto src_tensor,
                          ctx_->GetSpatialTensor(output_ids_[i]));
    ABSL_RETURN_IF_ERROR(
        output_converters_[i]->Convert(*src_tensor, *dst_buffer));
  }
  ABSL_RETURN_IF_ERROR(ctx_->PostConvert(/*input=*/false));

  return absl::OkStatus();
}

// Registers LiteRT buffer requirements for the given tensor.
// The required buffer size is calculated from the tensor descriptor.
// Note: This method is only used in LiteRt mode.
absl::Status DelegateKernelLiteRt::RegisterLiteRtBufferRequirements(
    LiteRtExternalLiteRtBufferContextT* buffer_context,
    TfLiteTensor* tflite_tensor,
    const ::ml_drift::TensorDescriptor& tensor_desc,
    ::ml_drift::TensorStorageType used_storage_type) {
  const ::ml_drift::DataType data_type = tensor_desc.GetDataType();
  std::vector<uint64_t> storage_dims = tensor_desc.GetStorageDims();
  size_t required_data_size =
      storage_dims[0] * tensor_desc.GetElementSize() * SizeOf(data_type);

  LiteRtTensorBufferRequirements gpu_buffer_requirements;
  ABSL_ASSIGN_OR_RETURN(auto requirements,
                        backend_->GetGpuBufferRequirements(
                            used_storage_type, tensor_desc.GetDataType()));
  LITERT_RETURN_IF_ERROR(runtime_context_->create_tensor_buffer_requirements(
      requirements.buffer_types.size(), requirements.buffer_types.data(),
      required_data_size, requirements.strides.size(),
      requirements.strides.data(), &gpu_buffer_requirements));
  LITERT_RETURN_IF_ERROR(
      runtime_context_
          ->external_litert_buffer_context_register_buffer_requirements(
              buffer_context, tflite_tensor, gpu_buffer_requirements),
      _ << "Failed to register buffer requirement");
  return absl::OkStatus();
}

absl::Status DelegateKernelLiteRt::
    RegisterLiteRtBufferRequirementsNonImmutableExternalTensorsMode(
        LiteRtExternalLiteRtBufferContextT* buffer_context,
        TfLiteTensor* tflite_tensor) {
  size_t required_data_size = tflite_tensor->bytes;
  LiteRtTensorBufferRequirements gpu_buffer_requirements;
  ABSL_ASSIGN_OR_RETURN(
      auto requirements,
      backend_->GetGpuBufferRequirementsForNonExternalTensors());
  LITERT_RETURN_IF_ERROR(runtime_context_->create_tensor_buffer_requirements(
      requirements.buffer_types.size(), requirements.buffer_types.data(),
      required_data_size, requirements.strides.size(),
      requirements.strides.data(), &gpu_buffer_requirements));
  LITERT_RETURN_IF_ERROR(
      runtime_context_
          ->external_litert_buffer_context_register_buffer_requirements(
              buffer_context, tflite_tensor, gpu_buffer_requirements),
      _ << "Failed to register buffer requirement");
  return absl::OkStatus();
}

absl::Status DelegateKernelLiteRt::UpdateCreateInfoWithExternalTensors(
    TfLiteContext* context, const std::vector<::ml_drift::Value*>& inputs,
    const std::vector<::ml_drift::Value*>& outputs,
    ::ml_drift::CreateGpuModelInfo& create_info) {
  auto* buffer_context = reinterpret_cast<LiteRtExternalLiteRtBufferContext>(
      context->GetExternalContext(context, kTfLiteLiteRtBufferContext));

  // Initialize tensor descriptors
  input_tensor_descriptors_.resize(inputs.size());
  output_tensor_descriptors_.resize(outputs.size());

  // Create processing context to pass common data to helper methods
  ABSL_ASSIGN_OR_RETURN(auto gpu_info, backend_->GetInfo());
  TensorProcessingContext proc_context{.buffer_context = buffer_context,
                                       .gpu_info = gpu_info,
                                       .create_info = create_info};

  // Process tensors based on the mode
  bool no_external_mode = NoExternalTensorsMode();

  // Process input tensors
  for (int i = 0; i < inputs.size(); ++i) {
    auto& input = inputs[i];
    if (IsExternalSharedConstantTensor(input->id)) {
      continue;
    }
    ABSL_RETURN_IF_ERROR(ProcessTensor(context, input, i, proc_context,
                                       no_external_mode,
                                       input_tensor_descriptors_));
  }

  // Process output tensors
  for (int i = 0; i < outputs.size(); ++i) {
    auto& output = outputs[i];
    ABSL_RETURN_IF_ERROR(ProcessTensor(context, output, i, proc_context,
                                       no_external_mode,
                                       output_tensor_descriptors_));
  }

  return absl::OkStatus();
}

absl::Status DelegateKernelLiteRt::ProcessTensor(
    TfLiteContext* context, ::ml_drift::Value* value, int index,
    const TensorProcessingContext& proc_context, bool no_external_tensor_mode,
    std::vector<::ml_drift::TensorDescriptor>& tensor_descriptors) {
  uint32_t tensor_index = value->tensor.ref;
  TfLiteTensor* tflite_tensor = &context->tensors[tensor_index];

  // Special handling for NoExternalTensorsMode with non-external tensors
  if (no_external_tensor_mode && !IsExternalTensorName(tflite_tensor->name)) {
    ABSL_RETURN_IF_ERROR(
        RegisterLiteRtBufferRequirementsNonImmutableExternalTensorsMode(
            proc_context.buffer_context, tflite_tensor));
  } else {
    // Common path for external tensors (both modes) and standard mode
    if (no_external_tensor_mode) {
      external_tensor_ids_.insert(value->id);
    }
    ABSL_ASSIGN_OR_RETURN(
        ::ml_drift::TensorDescriptor tensor_desc,
        CreateTensorDescriptor(proc_context.gpu_info, value->tensor,
                               delegate_data_->calculation_precision,
                               GetStorageType(tflite_tensor->name)));
    auto tensor_storage_type = tensor_desc.GetStorageType();

    // Store tensor descriptor
    tensor_descriptors[index] = tensor_desc;

    proc_context.create_info.external_mutable_tensors.try_emplace(value->id,
                                                                  tensor_desc);
    ABSL_RETURN_IF_ERROR(RegisterLiteRtBufferRequirements(
        proc_context.buffer_context, tflite_tensor, tensor_desc,
        tensor_storage_type));
  }
  return absl::OkStatus();
}

absl::Status DelegateKernelLiteRt::HandleInputEvents(TfLiteContext* context) {
  std::vector<GpuEventHandle> input_events;
  for (int i = 0; i < input_indices_.size(); ++i) {
    // Skip external shared constant tensors - they are handled internally by
    // the GPU and don't need TensorBuffer registration.
    if (IsExternalSharedConstantTensor(input_ids_[i])) {
      continue;
    }
    TfLiteTensor* tflite_tensor = &context->tensors[input_indices_[i]];
    LITERT_ASSIGN_OR_RETURN(
        auto tensor_buffer,
        GetTensorBuffer(runtime_context_, buffer_context_, tflite_tensor),
        _ << "TensorBuffer is not registered for input#" << i << " tensor T#"
          << input_indices_[i] << ".");
    auto event = backend_->GetGpuEventAssociated(tensor_buffer);
    if (event.ok()) {
      input_events.push_back(event.value());
    } else if (event.status().code() != absl::StatusCode::kNotFound) {
      return event.status();
    }
  }

  auto pre_dispatch_event = ctx_->GetPreDispatchEvent();
  if (pre_dispatch_event.ok()) {
    input_events.push_back(pre_dispatch_event.value());
  } else if (pre_dispatch_event.status().code() !=
             absl::StatusCode::kNotFound) {
    return pre_dispatch_event.status();
  }

  if (input_events.empty()) {
    return absl::OkStatus();
  }

  LITERT_LOG(LITERT_DEBUG, "Waiting for %zu input GPU events",
             input_events.size());
  return ctx_->WaitForEventsCompleted(absl::MakeSpan(input_events),
                                      /*force_sync=*/false);
}

absl::Status DelegateKernelLiteRt::HandleOutputEvents(
    TfLiteContext* context, bool is_async_execution_mode) {
  auto post_dispatch_event =
      ctx_->GetPostDispatchEvent(is_async_execution_mode);
  if (!post_dispatch_event.ok()) {
    if (post_dispatch_event.status().code() != absl::StatusCode::kNotFound) {
      return post_dispatch_event.status();
    }

    if (is_async_execution_mode) {
      LITERT_LOG(LITERT_DEBUG, "Backend doesn't support async execution mode");
      return absl::OkStatus();
    }

    if (delegate_data_->options->wait_type == kGpuDelegateWaitTypeDoNotWait) {
      return absl::OkStatus();
    }

    // Don't distinguish wait_type except DoNotWait. Some backend, e.g. Metal,
    // may distinguish wait_type in WaitForCompletion().
    return backend_->WaitForCompletion();
  }

  if (!is_async_execution_mode) {
    // If not running in async execution mode, we wait for output_event and
    // we do not attach output event to tensor buffer.
    std::vector<GpuEventHandle> output_events{*post_dispatch_event};
    return ctx_->WaitForEventsCompleted(absl::MakeSpan(output_events),
                                        /*force_sync=*/true);
  }

  // Bind the output event to the all output tensor buffers.
  // Note: The output event is still owned by the DelegateKernelLiteRt without
  // transferring ownership to the tensor buffers.
  LITERT_ASSIGN_OR_RETURN(auto env,
                          GetEnvironment(runtime_context_, buffer_context_));
  for (int i = 0; i < output_indices_.size(); ++i) {
    TfLiteTensor* tflite_tensor = &context->tensors[output_indices_[i]];
    LITERT_ASSIGN_OR_RETURN(
        auto tensor_buffer,
        GetTensorBuffer(runtime_context_, buffer_context_, tflite_tensor),
        _ << "TensorBuffer is not registered for output#" << i << " tensor T#"
          << output_indices_[i] << ".");
    ABSL_RETURN_IF_ERROR(
        backend_->AssociateGpuEvent(*post_dispatch_event, env, tensor_buffer));
  }
  return absl::OkStatus();
}

absl::Status DelegateKernelLiteRt::DownloadGpuMemoryToCpuMemory(
    TfLiteContext* context) {
  for (int i = 0; i < output_ids_.size(); ++i) {
    if (external_tensor_ids_.contains(output_ids_[i])) {
      continue;
    }
    TfLiteTensor* tflite_tensor = &context->tensors[output_indices_[i]];
    if (tflite_tensor->data.raw != nullptr) {
      LITERT_ASSIGN_OR_RETURN(
          auto tensor_buffer,
          GetTensorBuffer(runtime_context_, buffer_context_, tflite_tensor),
          _ << "TensorBuffer is not registered for output#" << i << " tensor T#"
            << output_indices_[i] << ".");
      size_t buffer_size;
      LITERT_RETURN_IF_ERROR(runtime_context_->get_tensor_buffer_size(
          tensor_buffer.get(), &buffer_size));
      void* host_memory;
      LITERT_RETURN_IF_ERROR(runtime_context_->lock_tensor_buffer(
          tensor_buffer.get(), &host_memory, kLiteRtTensorBufferLockModeRead));
      size_t bytes_to_copy = std::min(tflite_tensor->bytes, buffer_size);
      if (tflite_tensor->bytes != buffer_size) {
        ABSL_LOG_FIRST_N(WARNING, 10)
            << "TFLite tensor size (" << tflite_tensor->bytes
            << ") is different from LiteRT buffer size (" << buffer_size
            << ").";
      }
      std::memcpy(tflite_tensor->data.raw, host_memory, bytes_to_copy);
      LITERT_RETURN_IF_ERROR(
          runtime_context_->unlock_tensor_buffer(tensor_buffer.get()));
    }
  }
  return absl::OkStatus();
}

absl::Status DelegateKernelLiteRt::FlushBufferCacheIfNeeded(
    TfLiteContext* context) {
  for (const auto& [tflite_tensor, gpu_memory] : tensors_to_flush_) {
    // Flush from buffer context.
    LITERT_RETURN_IF_ERROR(
        runtime_context_
            ->external_litert_buffer_context_unregister_tensor_buffer(
                buffer_context_, tflite_tensor));
    // Flush from delegate kernel cache.
    input_tensors_.erase(gpu_memory);
    output_tensors_.erase(gpu_memory);
  }
  tensors_to_flush_.clear();

  return absl::OkStatus();
}

// Initializes tensor converters used for synchronizing input and output
// tensors.
absl::Status DelegateKernelLiteRt::InitTensorConverters(
    TfLiteContext* context) {
  input_converters_.resize(input_indices_.size());

  for (int i = 0; i < input_indices_.size(); ++i) {
    if (IsExternalSharedConstantTensor(input_ids_[i]) ||
        external_tensor_ids_.contains(input_ids_[i])) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(auto gpu_tensor,
                          ctx_->GetSpatialTensor(input_ids_[i]));
    TfLiteTensor* tflite_tensor = &context->tensors[input_indices_[i]];
    ::ml_drift::BufferDescriptor src_desc;
    src_desc.element_type = ToDataType(tflite_tensor->type);
    if (src_desc.element_type == ::ml_drift::DataType::UNKNOWN) {
      return absl::InvalidArgumentError("model input type is not supported.");
    }
    src_desc.element_size = 1;
    src_desc.memory_type = ::ml_drift::MemoryType::GLOBAL;
    ABSL_ASSIGN_OR_RETURN(input_converters_[i],
                          backend_->CreateBuffer2TensorConverter(
                              src_desc, gpu_tensor->GetDescriptor()));
  }

  output_converters_.resize(output_indices_.size());
  for (int i = 0; i < output_indices_.size(); ++i) {
    if (external_tensor_ids_.contains(output_ids_[i])) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(auto gpu_tensor,
                          ctx_->GetSpatialTensor(output_ids_[i]));
    TfLiteTensor* tflite_tensor = &context->tensors[output_indices_[i]];
    ::ml_drift::BufferDescriptor dst_desc;
    dst_desc.element_type = ToDataType(tflite_tensor->type);
    if (dst_desc.element_type == ::ml_drift::DataType::UNKNOWN) {
      return absl::InvalidArgumentError("model output type is not supported.");
    }
    dst_desc.element_size = 1;
    dst_desc.memory_type = ::ml_drift::MemoryType::GLOBAL;
    ABSL_ASSIGN_OR_RETURN(output_converters_[i],
                          backend_->CreateTensor2BufferConverter(
                              gpu_tensor->GetDescriptor(), dst_desc));
  }

  return absl::OkStatus();
}

absl::Status DelegateKernelLiteRt::Dispatch(TfLiteContext* context) {
  return DelegateKernel::Dispatch(context);
}

}  // namespace litert::ml_drift
