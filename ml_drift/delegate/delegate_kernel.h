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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_KERNEL_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_KERNEL_H_

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/status.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/serialization_weight_cache.h"
#include "third_party/odml/litert/ml_drift/delegate/composite/litert_op_selector.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_data.h"
#include "third_party/odml/litert/ml_drift/delegate/delegate_options.h"
#include "third_party/odml/litert/ml_drift/delegate/gpu_backend.h"
#include "third_party/odml/litert/ml_drift/tflite/shared_const_tensor_map.h"
#include "tflite/core/c/common.h"
#include "tflite/delegates/serialization.h"

namespace litert::ml_drift {

class DelegateKernel {
 public:
  virtual ~DelegateKernel();

  absl::Status Dispatch(TfLiteContext* context);

  const std::vector<int64_t>& GetInputIndices() const { return input_indices_; }

  const std::vector<int64_t>& GetOutputIndices() const {
    return output_indices_;
  }

  // Returns true if there are any quantized tensors in the model.
  inline bool HasQuantizedTensors() const {
    return !quant_conversion_map_.empty();
  }

  // Calls the delegate to get the list of tensors that need to be temporarily
  // storage. It's usually needed for quantized tensors. And it's eventually
  // stored in TfLiteNode::temporaries.
  absl::Status GetRequiredTemporaries(TfLiteContext* context, TfLiteNode* node,
                                      TfLiteIntArray** temporaries_array_ptr);

  // Dequantizes the input tensors using `quant_conversion_map_`.
  absl::Status DequantizeInputs(TfLiteContext* context);

  // Quantizes the output tensors using `quant_conversion_map_`.
  absl::Status QuantizeOutputs(TfLiteContext* context);

  // Initializes the delegate kernel. This method is called by the Create()
  // method of the derived classes.
  absl::Status Initialize(TfLiteContext* context,
                          const TfLiteDelegateParams* delegate_params);

  // Returns true if the delegate is running in benchmark mode.
  bool IsBenchmarkMode() const {
    return delegate_data_->options->litert_benchmark_mode;
  }

  // Returns true if the delegate is hinting waiting for completion.
  bool IsWaitingForCompletionHinted() const {
    return delegate_data_->options->hint_waiting_for_completion;
  }

  GpuBackend* backend() const { return backend_; }

 protected:
  // A virtual function to update the create_info with external tensors.
  // inputs and outputs are the list of Value* that are used in the GraphFloat32
  // model.
  // Used external tensors are added to external_tensor_ids_.
  virtual absl::Status UpdateCreateInfoWithExternalTensors(
      TfLiteContext* context, const std::vector<::ml_drift::Value*>& inputs,
      const std::vector<::ml_drift::Value*>& outputs,
      ::ml_drift::CreateGpuModelInfo& create_info) = 0;

  // Returns true if the given id is an external shared constant tensor.
  bool IsExternalSharedConstantTensor(::ml_drift::ValueId id) {
    return (external_shared_constant_tensor_ids_.find(id) !=
            external_shared_constant_tensor_ids_.end());
  }

  // Returns storage type for GraphFloat32 creation.
  ::ml_drift::TensorStorageType GetStorageType() const {
    if (delegate_data_ != nullptr &&
        delegate_data_->options->use_buffer_storage_type) {
      return ::ml_drift::TensorStorageType::BUFFER;
    }
    auto type = backend_->GetFastestStorageType();
    return type.ok() ? *type : ::ml_drift::TensorStorageType::BUFFER;
  }

 private:
  // Initializes external shared constant tensors.
  // Note: This is only called when `enable_constant_tensors_sharing` is true.
  absl::Status InitializeExternalSharedConstantTensors(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      const SharedConstTensorsMap& shared_tensors,
      ::ml_drift::GraphFloat32& graph,
      ::ml_drift::CreateGpuModelInfo& create_info);

  // Returns false if serialization prerequisites are not initialized. Otherwise
  // initializes the serialization params and returns true.
  bool ReadFromSerialzedData();

  // Builds the GPU model, while preparing weights on GPU. This will be used
  // when convert_weights_on_gpu is enabled.
  absl::Status GraphToGpuModelWithGpuConverters(
      const ::ml_drift::GraphFloat32& graph,
      ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GpuModel* gpu_model, LiteRtOpSelector& op_selector);

  // Restores inference context from the serialized data.
  absl::Status InitInferenceContextFromSerializedData(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GraphFloat32* graph, LiteRtOpSelector& op_selector);

  // Builds inference context from the graph.
  absl::Status InitInferenceContextFromGraph(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GraphFloat32* graph, LiteRtOpSelector& op_selector,
      std::vector<uint8_t>* serialized_model);

  // Initializes the inference context `ctx_`. If the model has been serialized,
  // restores the inference context from the serialized data. Otherwise,
  // initializes the inference context from the graph.
  absl::Status InitInferenceContext(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      const ::ml_drift::CreateGpuModelInfo& create_info,
      ::ml_drift::GraphFloat32* graph);

  // Creates a new SerializationWeightCache and returns it. If the serialization
  // is disabled, it will return a nullptr. If there is an error, it will return
  // the error status.
  absl::StatusOr<::ml_drift::SerializationWeightCache*>
  TryInitializingExternalTensorsSerialization(
      TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
      bool prepare_weights_in_batches);

  // Cleans up the external tensors serialization cache. If the cache was never
  // initialized, this is a no-op.
  absl::Status CleanupExternalTensorsSerialization(
      ::ml_drift::SerializationWeightCache* shared_memory_serialization_cache);

 protected:
  // MlDriftDelegateData referenced by delegate->data_.
  MlDriftDelegateData* delegate_data_;

  std::vector<int64_t> input_indices_;
  std::vector<int64_t> output_indices_;
  std::vector<::ml_drift::ValueId> output_ids_;
  std::vector<::ml_drift::ValueId> input_ids_;

  GpuBackend* backend_ = nullptr;
  bool is_opencl_backend_ = false;
  std::unique_ptr<GpuInferenceContext> ctx_;

  // Set of external tensor ids. It's updated by
  // InitializeExternalSharedConstantTensors().
  absl::flat_hash_set<::ml_drift::ValueId> external_tensor_ids_;

 private:
  // Ordered set of external shared constant tensor ids.
  std::set<::ml_drift::ValueId> external_shared_constant_tensor_ids_;
  // Conversion context holds converted weights Tensors.
  std::unique_ptr<GpuInferenceContext> conversion_context_;
  // # of weights being converted on GPU. Dispatch must be called after all
  // weights are converted.
  std::unique_ptr<absl::BlockingCounter> weights_converting_;
  // Object needed for the model serialization. It is initialized in
  // InitInferenceContext().
  std::unique_ptr<tflite::delegates::Serialization> serialization_;
  // Map the tensor index of each originally quantized (8-bit) tensor to its
  // float version added in model_builder.
  absl::flat_hash_map<int, int> quant_conversion_map_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_DELEGATE_KERNEL_H_
