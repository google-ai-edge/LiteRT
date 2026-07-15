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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_KERNEL_LITERT_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_KERNEL_LITERT_H_

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/litert_common.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_kernel.h"
#include "third_party/odml/litert/ml_drift/delegate/gpu_backend.h"
#include "tflite/core/c/common.h"

// Forward declaration of LiteRtRuntimeContext.
struct LiteRtRuntimeContext;

namespace litert::ml_drift {

class DelegateKernelLiteRt : public DelegateKernel {
 public:
  explicit DelegateKernelLiteRt(const LiteRtRuntimeContext* runtime_context)
      : DelegateKernel(), runtime_context_(runtime_context) {}

  ~DelegateKernelLiteRt() override = default;

  absl::Status Dispatch(TfLiteContext* context) override;

  // Creates a new DelegateKernel which will be stored in TfLiteNode::user_data.
  static absl::StatusOr<DelegateKernelLiteRt*> Create(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params);

  // Create internal SpatialTensor from I/O TensorBuffers and bind them to the
  // inference context. The created SpatialTensors are stored in
  // input_tensors_ and output_tensors_.
  absl::Status BindTensorBuffers(TfLiteContext* context);

  // Enqueues Barrier command that will be used to wait for all the input events
  // to be completed.
  // Note: This method should be called before Dispatch() is called.
  // Note: If there is non OpenCL event, it will wait for 5 seconds for the
  // event to be signaled.
  absl::Status HandleInputEvents(TfLiteContext* context);

  // Enqueues Marker command that will be used to notify for all the output
  // events to be completed. If is_async_execution_mode is true, attaches the
  // output event to the output tensors. Note: This method should be called
  // after Dispatch() is called.
  absl::Status HandleOutputEvents(TfLiteContext* context,
                                  bool is_async_execution_mode);

  // Downloads GPU memory to CPU memory. This method will be called after
  // Dispatch() is called, and it meant to only download the output tensors that
  // are intermediate tensors and not allocated by the users.
  absl::Status DownloadGpuMemoryToCpuMemory(TfLiteContext* context);

  // Flushes the buffer cache if needed.
  absl::Status FlushBufferCacheIfNeeded(TfLiteContext* context);

  // Initializes tensor converters used for synchronizing input and output
  // tensors.
  absl::Status InitTensorConverters(TfLiteContext* context);

  // Returns true if the delegate is running in no external tensors mode.
  bool NoExternalTensorsMode() const {
    return !delegate_data_->options->litert_external_tensors_mode;
  }

  // Returns true if the tensor name matches any of the external tensor
  // patterns.
  bool IsExternalTensorName(const char* tensor_name) const {
    if (tensor_name == nullptr) {
      return false;
    }
    for (const auto& pattern :
         delegate_data_->options->litert_external_tensor_patterns) {
      if (strncmp(tensor_name, pattern.c_str(), pattern.size()) == 0) {
        return true;
      }
    }
    return false;
  }

  // In NoExternalTensorsMode, we need to upload GPU memory in
  // input TensorBuffer to GPU memory before Dispatch() is called.
  // Additionaly, it allocates output TensorBuffers if they're not provided
  // similar to BindTensorBuffers().
  // If external tensors are present, binds them directly to the inference
  // context.
  absl::Status UploadOrBindTensorBuffer(TfLiteContext* context);

  // In NoExternalTensorsMode, we need to download GPU memory to
  // output TensorBuffer after Dispatch() is called.
  absl::Status DownloadGpuMemoryToTensorBufferGpuMemory(TfLiteContext* context);

  // Returns the LiteRT runtime context.
  const LiteRtRuntimeContext* runtime_context() const {
    return runtime_context_;
  }

 protected:
  // A virtual function to update the create info with external tensors.
  // In addition to this, it conducts the following tasks:
  // - It creates EnvironmentSingleton to share OpenCL resources with LiteRt
  // - It registers LiteRt BufferRequirements for the given tensor via
  // RegisterLiteRtBufferRequirements().
  // - It create TensorDescriptors which are later used to create SpatialTensor
  // in BindTensorBuffers().
  absl::Status UpdateCreateInfoWithExternalTensors(
      TfLiteContext* context, const std::vector<::ml_drift::Value*>& inputs,
      const std::vector<::ml_drift::Value*>& outputs,
      ::ml_drift::CreateGpuModelInfo& create_info) override;

  // Returns the storage type for the given tensor name.
  // If the tensor name matches any of the buffer storage type patterns,
  // BUFFER storage type is returned. Otherwise, the default storage type is
  // returned.
  ::ml_drift::TensorStorageType GetStorageType(const char* tensor_name) const {
    for (const auto& pattern :
         delegate_data_->options->litert_buffer_storage_tensor_patterns) {
      if (strncmp(tensor_name, pattern.c_str(), pattern.size()) == 0) {
        return ::ml_drift::TensorStorageType::BUFFER;
      }
    }
    return DelegateKernel::GetStorageType();
  }

 private:
  const LiteRtRuntimeContext* runtime_context_;
  LiteRtExternalLiteRtBufferContext buffer_context_ = nullptr;

  // Context structure that encapsulates the common parameters needed for tensor
  // processing operations in the ML Drift delegate. This structure helps reduce
  // parameter passing complexity and improves code maintainability by grouping
  // related tensor processing configuration together.
  struct TensorProcessingContext {
    // External buffer context for managing LiteRT tensor buffer requirements
    LiteRtExternalLiteRtBufferContext buffer_context;

    // GPU device information used for creating appropriate tensor descriptors
    ::ml_drift::GpuInfo gpu_info;

    // Reference to the GPU model creation info that will be populated with
    // tensor descriptors and external tensor configurations
    ::ml_drift::CreateGpuModelInfo& create_info;
  };

  // Registers LiteRT buffer requirements for the given tensor.
  // The required buffer size is calculated from the tensor descriptor.
  absl::Status RegisterLiteRtBufferRequirements(
      LiteRtExternalLiteRtBufferContext buffer_context,
      TfLiteTensor* tflite_tensor,
      const ::ml_drift::TensorDescriptor& tensor_desc,
      ::ml_drift::TensorStorageType used_storage_type);

  // The same with above, but it registers the requirements for non immutable
  // external tensors mode.
  absl::Status RegisterLiteRtBufferRequirementsNonImmutableExternalTensorsMode(
      LiteRtExternalLiteRtBufferContext buffer_context,
      TfLiteTensor* tflite_tensor);

  // Unified tensor processing method that handles both standard and
  // NoExternalTensorsMode. The behavior is controlled by the parameters:
  // - no_external_tensor_mode: if true, indicates NoExternalTensorsMode is
  // active
  //   in NoExternalTensorsMode
  // - tensor_descriptors: reference to either input_tensor_descriptors_ or
  //   output_tensor_descriptors_ where the descriptor will be stored
  absl::Status ProcessTensor(
      TfLiteContext* context, ::ml_drift::Value* value, int index,
      const TensorProcessingContext& proc_context, bool no_external_tensor_mode,
      std::vector<::ml_drift::TensorDescriptor>& tensor_descriptors);

  // Binds the GPU memory to the inference context.
  absl::Status BindGpuMemoryToInferenceContext(
      ::ml_drift::ValueId tensor_id,
      const ::ml_drift::TensorDescriptor& tensor_desc,
      GpuMemoryHandle gpu_memory,
      absl::flat_hash_map<GpuMemoryHandle, std::unique_ptr<GpuTensorWrapper>>&
          tensors);

  // Tensor descriptors for I/O tensors.
  // The size and order of these vectors should be the same as input_indices_
  // and output_indices_ of the base class.
  std::vector<::ml_drift::TensorDescriptor> input_tensor_descriptors_;
  std::vector<::ml_drift::TensorDescriptor> output_tensor_descriptors_;

  // SpatialTensor for I/O tensors.
  // The size and order of these vectors should be the same as input_indices_
  // and output_indices_ of the base class.
  absl::flat_hash_map<GpuMemoryHandle, std::unique_ptr<GpuTensorWrapper>>
      input_tensors_;
  absl::flat_hash_map<GpuMemoryHandle, std::unique_ptr<GpuTensorWrapper>>
      output_tensors_;

  // Converter to sync input and output tensors. These are used only in
  // NoExternalTensorsMode.
  std::vector<std::unique_ptr<Buffer2TensorConverter>> input_converters_;
  std::vector<std::unique_ptr<Tensor2BufferConverter>> output_converters_;

  // Set of tensors that need to be flushed from the cache and buffer context
  // during `FlushBufferCacheIfNeeded`.
  absl::flat_hash_map<TfLiteTensor*, GpuMemoryHandle> tensors_to_flush_;

  // Set of input indices whether this input needs to be uploaded to GPU memory
  // before inference. This is only used in NoExternalTensorsMode.
  absl::flat_hash_set<int> input_needs_upload_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_KERNEL_LITERT_H_
