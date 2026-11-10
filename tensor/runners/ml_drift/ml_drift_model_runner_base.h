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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_MODEL_RUNNER_BASE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_MODEL_RUNNER_BASE_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "tensor/arithmetic.h"
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/backends/ml_drift/ml_drift_conversion.h"
#include "tensor/backends/ml_drift/ml_drift_helpers.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_probe.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/runners/ml_drift/ml_drift_dummy_weight_creator.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

// Backend concept required by MlDriftModelRunnerBase.
// Required:
//  - Environment, InferenceContext, Tensor types
//  - InitEnvironment(Environment* env)
//  - GetGpuInfoForBuilder(const Environment& env, bool env_initialized)
//  - GetStorageTypeForBuilder(const Environment& env, bool env_initialized)
//  - InitInferenceContext(InferenceContext*, const CreateGpuModelInfo&,
//  GpuModel*, Environment*)
//  - SetInputTensorFloat/SetInputTensorInt32
//  - GetOutputTensorFloat/GetOutputTensorInt32
//  - GetTensor(InferenceContext*, ValueId)
//  - UploadTensorDescriptorData(Tensor*, Environment*, const TensorDescriptor&)
//  - TensorToDescriptor(Tensor*, Environment*, TensorDescriptor*)
//  - AddToQueue(InferenceContext*, Environment*)
//  - WaitForCompletion(Environment*)

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
class MlDriftModelRunnerBase {
 public:
  using Environment = typename Backend::Environment;
  using InferenceContext = typename Backend::InferenceContext;
  using BackendTensor = typename Backend::Tensor;
  using TensorMLD = Tensor<MlDriftMixinTag>;

  explicit MlDriftModelRunnerBase(
      ModelFunctor model_func, Inputs& inputs, Outputs& outputs,
      const absl::flat_hash_map<std::string, absl::Span<const float>>&
          float_weights,
      const absl::flat_hash_map<std::string, absl::Span<const int8_t>>&
          int8_weights,
      bool build_gpu_model = false,
      ml_drift::CalculationsPrecision precision =
          ml_drift::CalculationsPrecision::kF16,
      typename Backend::Environment* shared_env = nullptr,
      MlDriftBuildContext* shared_build_ctx = nullptr);

  absl::Status BuildModel(
      const std::vector<Tensor<MlDriftMixinTag>>& all_output_tensors = {});

  ml_drift::GpuModel& GetGpuModel() { return gpu_model_; }

  // Sets the input tensor data by name.
  absl::Status SetInput(absl::string_view name, absl::Span<const float> data);
  absl::Status SetInput(absl::string_view name, absl::Span<const fp16_t> data);
  absl::Status SetInput(absl::string_view name, absl::Span<const int32_t> data);
  absl::Status SetInput(absl::string_view name, absl::Span<const uint8_t> data);
  absl::Status SetInput(absl::string_view name, absl::Span<const int8_t> data);
  absl::Status SetInput(absl::string_view name, absl::Span<const bool> data);
  // Packed version of the above SetInput for bool. Note that std::vector<bool>
  // is specialized to allocate one bit per element.
  absl::Status SetInput(absl::string_view name, const std::vector<bool>& data);
  // Set the input tensor with type neutral span by name.
  absl::Status SetInput(absl::string_view name,
                        absl::Span<const std::byte> data);

  // Executes the model inference.
  absl::Status Init(const ml_drift::ExternalTensorsInfo& external_tensors = {});
  absl::Status Run();
  absl::Status Enqueue();
  absl::Status Wait();

  absl::StatusOr<ml_drift::ValueId> GetWeightValueId(
      const std::string& name) const;
  absl::flat_hash_map<ml_drift::ValueId, BackendTensor*> GetWeightTensors();

  // Retrieves the output tensor data by name.
  absl::StatusOr<std::vector<float>> GetFloatOutput(absl::string_view name);
  absl::StatusOr<std::vector<int32_t>> GetInt32Output(absl::string_view name);

  const Tensor<MlDriftMixinTag>& GetIntermediateTensor(
      absl::string_view name) const;

  // Retrieves the output tensor data by name and converts it to float.
  absl::StatusOr<absl::Span<const float>> GetOutput(absl::string_view name);
  // Retrieves the output tensor data by name into buffer which must have
  // sufficient capacity.
  absl::Status ReadOutput(absl::string_view name, absl::Span<std::byte> data);

  // Returns the compilation time.
  absl::Duration compilation_time() const { return compilation_time_; }

  // Returns the inference time.
  absl::Duration inference_time() const { return inference_time_; }

  absl::Status AddTensorsAsOutputs(
      const absl::flat_hash_map<GraphProbe::StableTensorId, std::string,
                                GraphProbe::StableTensorIdHash>& probe_tensors);

  absl::Status AddTensorsAsOutputsByName(
      const absl::flat_hash_map<std::string, std::string>& probe_tensors);

 protected:
  template <typename TensorT>
  absl::Status SetInputInternal(absl::string_view name,
                                absl::Span<const TensorT> data);
  template <typename StorageT, typename TensorT>
  absl::StatusOr<std::vector<StorageT>> GetOutputInternal(
      absl::string_view name);

  Environment& env() { return *env_ptr_; }
  const Environment& env() const { return *env_ptr_; }
  InferenceContext& inference_context() { return inference_context_; }
  const InferenceContext& inference_context() const {
    return inference_context_;
  }
  MlDriftBuildContext& conversion_ctx() { return *conversion_ctx_ptr_; }
  const MlDriftBuildContext& conversion_ctx() const {
    return *conversion_ctx_ptr_;
  }
  Inputs& inputs() { return inputs_; }
  const Inputs& inputs() const { return inputs_; }
  const absl::flat_hash_map<std::string, graph::Tensor>& name_to_tensor_map()
      const {
    return name_to_tensor_map_;
  }

  absl::Status EnsureEnvInitialized();

  absl::Status CreateWeights(
      const absl::flat_hash_map<std::string, absl::Span<const float>>&
          float_weights,
      const absl::flat_hash_map<std::string, absl::Span<const int8_t>>&
          int8_weights);

  ModelFunctor model_func_;
  Inputs& inputs_;
  Outputs& outputs_;

  absl::flat_hash_map<std::string, TensorMLD*> cached_input_tensors_;
  absl::flat_hash_map<std::string, TensorMLD*> cached_output_tensors_;

  absl::flat_hash_map<GraphProbe::StableTensorId, std::string,
                      GraphProbe::StableTensorIdHash>
      probed_tensors_;

  absl::flat_hash_map<std::string, graph::Tensor> name_to_tensor_map_;

  Environment env_;
  Environment* env_ptr_ = &env_;
  std::unique_ptr<ml_drift::GpuModelBuilder> model_builder_;
  ml_drift::GpuModel gpu_model_;
  InferenceContext inference_context_;
  MlDriftBuildContext conversion_ctx_;
  MlDriftBuildContext* conversion_ctx_ptr_ = &conversion_ctx_;

  std::vector<ml_drift::ValueId> graph_input_ids_;
  std::vector<ml_drift::ValueId> graph_output_ids_;

  absl::Duration compilation_time_;
  absl::Duration inference_time_;

  ml_drift::CalculationsPrecision precision_;
  bool env_initialized_ = false;

  absl::flat_hash_map<std::string, std::vector<float>> float_inputs_cache_;
  absl::flat_hash_map<std::string, std::vector<int32_t>> int32_inputs_cache_;
  absl::flat_hash_map<std::string, std::vector<bool>> bool_inputs_cache_;
  absl::flat_hash_map<std::string, std::vector<int8_t>> int8_inputs_cache_;

  // Cache for output tensors returned as span.
  absl::flat_hash_map<std::string, std::vector<std::byte>> outputs_cache_;

  ml_drift::CreateGpuModelInfo DefaultCreateGpuModelInfo(
      ml_drift::CalculationsPrecision precision) {
    ml_drift::CreateGpuModelInfo ci = {
        .precision = precision,
        .storage_type =
            Backend::GetStorageTypeForBuilder(*env_ptr_, env_initialized_),
    };
    ci.hints.Add(ml_drift::ModelHints::kFastTuning);
    ci.hints.Add(ml_drift::ModelHints::kPreferTextureWeights);
    ci.hints.Add(ml_drift::ModelHints::kEnableHostMappedPointer);
    ci.hints.Add(ml_drift::ModelHints::kDisallow8bitConvs);
    ci.hints.Add(ml_drift::ModelHints::kAllowSpecialKernels);
    const char* allow_cb_env = std::getenv("LRT_ALLOW_COMMAND_BUFFER");
    ci.hints.allow_cl_khr_command_buffer =
        (allow_cb_env == nullptr) || (std::string(allow_cb_env) == "true");
    return ci;
  }
};

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::
    MlDriftModelRunnerBase(
        ModelFunctor model_func, Inputs& inputs, Outputs& outputs,
        const absl::flat_hash_map<std::string, absl::Span<const float>>&
            float_weights,
        const absl::flat_hash_map<std::string, absl::Span<const int8_t>>&
            int8_weights,
        bool build_gpu_model, ml_drift::CalculationsPrecision precision,
        typename Backend::Environment* shared_env,
        MlDriftBuildContext* shared_build_ctx)
    : model_func_(std::move(model_func)),
      inputs_(inputs),
      outputs_(outputs),
      precision_(precision) {
  cached_input_tensors_.insert(inputs_.tensors().begin(),
                               inputs_.tensors().end());
  if (shared_env != nullptr) {
    env_ptr_ = shared_env;
    env_initialized_ = true;
  }
  ABSL_CHECK_OK(EnsureEnvInitialized());

  ml_drift::CreateGpuModelInfo ci = DefaultCreateGpuModelInfo(precision);

  if (shared_build_ctx != nullptr) {
    conversion_ctx_ptr_ = shared_build_ctx;
    if (conversion_ctx_ptr_->gpu_model_builder == nullptr) {
      const ml_drift::GpuInfo gpu_info =
          Backend::GetGpuInfoForBuilder(*env_ptr_, env_initialized_);
      ABSL_LOG(INFO) << "Creating shared model builder";
      model_builder_ = std::make_unique<ml_drift::GpuModelBuilder>(
          gpu_info, ci.hints, ci.precision, ci.storage_type);
      conversion_ctx_ptr_->gpu_model_builder = model_builder_.get();
      conversion_ctx_ptr_->default_float_type =
          ml_drift::DeduceDataTypeFromPrecision(ci.precision);
    }
  } else {
    conversion_ctx_ptr_ = &conversion_ctx_;
    const ml_drift::GpuInfo gpu_info =
        Backend::GetGpuInfoForBuilder(*env_ptr_, env_initialized_);
    ABSL_LOG(INFO) << "Creating local model builder";
    model_builder_ = std::make_unique<ml_drift::GpuModelBuilder>(
        gpu_info, ci.hints, ci.precision, ci.storage_type);
    conversion_ctx_ptr_->gpu_model_builder = model_builder_.get();
    conversion_ctx_ptr_->default_float_type =
        ml_drift::DeduceDataTypeFromPrecision(ci.precision);
  }

  outputs_ = model_func_(inputs_);
  cached_output_tensors_.insert(outputs_.tensors().begin(),
                                outputs_.tensors().end());

  ABSL_LOG(INFO) << "Creating FC weights: float: " << float_weights.size()
                 << " int8:" << int8_weights.size();
  ABSL_CHECK_OK(CreateWeights(float_weights, int8_weights));

  const absl::Time compilation_start = absl::Now();
  if (build_gpu_model) {
    std::vector<TensorMLD> output_tensors;
    for (auto const& [name, tensor_ptr] : cached_output_tensors_) {
      tensor_ptr->SetName(name);
      output_tensors.push_back(*tensor_ptr);
    }
    ABSL_CHECK_OK(BuildModel(output_tensors));
    ABSL_CHECK_OK(Backend::InitInferenceContext(&inference_context_, ci,
                                                &gpu_model_, env_ptr_));
  }
  compilation_time_ = absl::Now() - compilation_start;
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs,
                                    Outputs>::EnsureEnvInitialized() {
  if (env_initialized_) {
    return absl::OkStatus();
  }
  LRT_TENSOR_RETURN_IF_ERROR(Backend::InitEnvironment(env_ptr_));
  env_initialized_ = true;
  return absl::OkStatus();
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::
    AddTensorsAsOutputs(
        const absl::flat_hash_map<GraphProbe::StableTensorId, std::string,
                                  GraphProbe::StableTensorIdHash>&
            probe_tensors) {
  probed_tensors_ = probe_tensors;

  std::vector<TensorHandle> original_output_handles;
  for (auto const& [name, tensor_ptr] : cached_output_tensors_) {
    original_output_handles.push_back(*tensor_ptr);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto execution_plan,
                              GetExecutionPlan(original_output_handles));
  absl::flat_hash_map<const graph::Operation*, int> op_to_id;
  for (int i = 0; i < execution_plan.size(); ++i) {
    op_to_id[execution_plan[i]] = i;
  }

  absl::flat_hash_set<graph::Tensor> non_leaf_tensors;
  for (const auto& op : execution_plan) {
    for (const auto& input : op->inputs) {
      non_leaf_tensors.insert(input);
    }
  }

  std::vector<TensorMLD> all_output_tensors;
  if (probed_tensors_.empty()) {
    all_output_tensors.reserve(original_output_handles.size());
    for (const auto& t : original_output_handles) {
      all_output_tensors.push_back(TensorMLD(t));
    }
  } else {
    for (const auto& op : execution_plan) {
      auto outputs_group = op->outputs_group.lock();
      if (!outputs_group) {
        return absl::InternalError("Outputs group is null.");
      }
      for (int i = 0; i < outputs_group->tensor_infos.size(); ++i) {
        graph::Tensor tensor = graph::GetTensor(i, outputs_group);
        auto it = op_to_id.find(op);
        if (it == op_to_id.end()) {
          continue;
        }
        const int op_id = it->second;
        const GraphProbe::StableTensorId stable_id = {op_id, i};
        auto probe_it = probed_tensors_.find(stable_id);
        if (probe_it != probed_tensors_.end()) {
          if (non_leaf_tensors.contains(tensor)) {
            TensorMLD original_tensor(tensor);
            TensorMLD probed_tensor = Probe(original_tensor);
            probed_tensor.SetName(probe_it->second);
            all_output_tensors.push_back(probed_tensor);
          } else {
            LRT_TENSOR_RETURN_IF_ERROR(
                graph::SetName(tensor, probe_it->second));
            all_output_tensors.push_back(TensorMLD(tensor));
          }
        }
      }
    }
  }

  const absl::Time compilation_start = absl::Now();
  LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());
  LRT_TENSOR_RETURN_IF_ERROR(BuildModel(all_output_tensors));
  ml_drift::CreateGpuModelInfo ci =
      DefaultCreateGpuModelInfo(ml_drift::CalculationsPrecision::kF32);
  LRT_TENSOR_RETURN_IF_ERROR(Backend::InitInferenceContext(
      &inference_context_, ci, &gpu_model_, env_ptr_));
  compilation_time_ = absl::Now() - compilation_start;
  return absl::OkStatus();
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::
    AddTensorsAsOutputsByName(
        const absl::flat_hash_map<std::string, std::string>& probe_tensors) {
  std::vector<TensorMLD> all_output_tensors;
  // Always keep original outputs and ensure they are named.
  for (auto const& [name, tensor_ptr] : cached_output_tensors_) {
    tensor_ptr->SetName(name);
    all_output_tensors.push_back(*tensor_ptr);
  }

  std::vector<TensorHandle> original_output_handles;
  original_output_handles.reserve(all_output_tensors.size());
  for (const auto& t : all_output_tensors) {
    original_output_handles.push_back(t);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto execution_plan,
                              GetExecutionPlan(original_output_handles));

  absl::flat_hash_set<graph::Tensor> non_leaf_tensors;
  for (const auto& op : execution_plan) {
    for (const auto& input : op->inputs) {
      non_leaf_tensors.insert(input);
    }
  }

  if (!probe_tensors.empty()) {
    for (const auto& op : execution_plan) {
      auto outputs_group = op->outputs_group.lock();
      if (!outputs_group) {
        return absl::InternalError("Outputs group is null.");
      }
      for (int i = 0; i < outputs_group->tensor_infos.size(); ++i) {
        graph::Tensor tensor = graph::GetTensor(i, outputs_group);
        LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
        auto it = probe_tensors.find(info.name);
        if (it != probe_tensors.end()) {
          if (non_leaf_tensors.contains(tensor)) {
            TensorMLD original_tensor(tensor);
            TensorMLD probed_tensor = Probe(original_tensor);
            probed_tensor.SetName(it->second);
            all_output_tensors.push_back(probed_tensor);
          } else {
            TensorMLD original_tensor(tensor);
            TensorMLD probed_tensor = Probe(original_tensor);
            probed_tensor.SetName(it->second);
            all_output_tensors.push_back(probed_tensor);
          }
        }
      }
    }
  }

  const absl::Time compilation_start = absl::Now();
  LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());
  LRT_TENSOR_RETURN_IF_ERROR(BuildModel(all_output_tensors));
  ml_drift::CreateGpuModelInfo ci =
      DefaultCreateGpuModelInfo(ml_drift::CalculationsPrecision::kF32);
  LRT_TENSOR_RETURN_IF_ERROR(Backend::InitInferenceContext(
      &inference_context_, ci, &gpu_model_, env_ptr_));
  compilation_time_ = absl::Now() - compilation_start;
  return absl::OkStatus();
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::BuildModel(
    const std::vector<TensorMLD>& all_output_tensors) {
  std::vector<TensorHandle> all_output_handles;
  all_output_handles.reserve(all_output_tensors.size());
  for (const auto& t : all_output_tensors) {
    all_output_handles.push_back(t);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto tensor_to_handle,
      AppendGpuModelBuilder(*conversion_ctx_ptr_, all_output_handles));
  conversion_ctx_ptr_->graph_tensor_to_handle = tensor_to_handle;

  name_to_tensor_map_.clear();
  for (const auto& tensor : all_output_tensors) {
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                graph::GetInfo(tensor.GetRaw()));
    if (!info.name.empty()) {
      name_to_tensor_map_[info.name] = tensor.GetRaw();
    }
  }

  graph_input_ids_.clear();
  for (auto const& [name, tensor_ptr] : cached_input_tensors_) {
    if (conversion_ctx_ptr_->graph_tensor_to_handle.contains(
            tensor_ptr->GetRaw())) {
      graph_input_ids_.push_back(
          conversion_ctx_ptr_->graph_tensor_to_handle.at(tensor_ptr->GetRaw())
              .id);
    }
  }
  graph_output_ids_.clear();
  for (const auto& tensor : all_output_tensors) {
    if (conversion_ctx_ptr_->graph_tensor_to_handle.contains(tensor.GetRaw())) {
      graph_output_ids_.push_back(
          conversion_ctx_ptr_->graph_tensor_to_handle.at(tensor.GetRaw()).id);
    }
  }
  return conversion_ctx_ptr_->gpu_model_builder->GetGpuModel(
      graph_input_ids_, graph_output_ids_, &gpu_model_);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
template <typename TensorT>
absl::Status MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::
    SetInputInternal(absl::string_view name, absl::Span<const TensorT> data) {
  if (absl::StartsWith(name, "w_fc_weight")) {
    return absl::OkStatus();
  }
  const auto& tensors = inputs_.tensors();
  auto inputs_it = tensors.find(std::string(name));
  if (inputs_it == tensors.end()) {
    ABSL_LOG(WARNING) << "Input tensor not found: " << name
                      << ", data_size=" << data.size();
    return absl::OkStatus();
  }
  const auto& tensor = inputs_it->second->GetRaw();
  LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
  if (data.size() != info.GetSize() ||
      sizeof(TensorT) != BufferSize(info.type, 1)) {
    return absl::InternalError(
        absl::StrCat("Input tensor size does not match the data type: ", name,
                     ", tensor=[type=", info.type, ", size=", info.GetSize(),
                     "], data=[type=", typeid(TensorT).name(),
                     ", size=", data.size(), "]"));
  }

  auto handle_it = conversion_ctx_ptr_->graph_tensor_to_handle.find(tensor);
  if (handle_it == conversion_ctx_ptr_->graph_tensor_to_handle.end()) {
    return absl::OkStatus();
  }
  const auto& th = handle_it->second;
  BackendTensor* gpu_tensor = Backend::GetTensor(&inference_context_, th.id);
  if (gpu_tensor == nullptr) {
    return absl::InternalError(
        absl::StrCat("Tensor not found for input: ", name));
  }

  ml_drift::TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(data.data());
  return Backend::UploadTensorDescriptorData(gpu_tensor, env_ptr_,
                                             descriptor_with_data);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::SetInput(
    absl::string_view name, absl::Span<const bool> data) {
  return SetInputInternal<bool>(name, data);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::SetInput(
    absl::string_view name, const std::vector<bool>& data) {
  std::vector<uint8_t> unpacked_bool(data.begin(), data.end());
  return SetInputInternal<uint8_t>(name, unpacked_bool);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::SetInput(
    absl::string_view name, absl::Span<const uint8_t> data) {
  return SetInputInternal<uint8_t>(name, data);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::SetInput(
    absl::string_view name, absl::Span<const int8_t> data) {
  return SetInputInternal<int8_t>(name, data);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::SetInput(
    absl::string_view name, absl::Span<const float> data) {
  return SetInputInternal<float>(name, data);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::SetInput(
    absl::string_view name, absl::Span<const fp16_t> data) {
  return SetInputInternal<fp16_t>(name, data);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::SetInput(
    absl::string_view name, absl::Span<const int32_t> data) {
  return SetInputInternal<int32_t>(name, data);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::SetInput(
    absl::string_view name, absl::Span<const std::byte> data) {
  const auto& tensors = inputs_.tensors();
  auto inputs_it = tensors.find(std::string(name));
  if (inputs_it == tensors.end()) {
    ABSL_LOG(WARNING) << "Input tensor not found: " << name
                      << " data size: " << data.size();
    return absl::OkStatus();
  }
  switch (inputs_it->second->GetType()) {
    case Type::kFP32:
      return SetInput(
          name, absl::MakeConstSpan(reinterpret_cast<const float*>(data.data()),
                                    data.size() / sizeof(float)));
    case Type::kFP16:
      return SetInput(name, absl::MakeConstSpan(
                                reinterpret_cast<const fp16_t*>(data.data()),
                                data.size() / sizeof(fp16_t)));
    case Type::kI32:
      return SetInput(name, absl::MakeConstSpan(
                                reinterpret_cast<const int32_t*>(data.data()),
                                data.size() / sizeof(int32_t)));
    case Type::kI8:
      return SetInput(name, absl::MakeConstSpan(
                                reinterpret_cast<const int8_t*>(data.data()),
                                data.size() / sizeof(int8_t)));
    case Type::kU8:
      return SetInput(name, absl::MakeConstSpan(
                                reinterpret_cast<const uint8_t*>(data.data()),
                                data.size() / sizeof(uint8_t)));
    case Type::kBOOL:
      return SetInput(
          name, absl::MakeConstSpan(reinterpret_cast<const bool*>(data.data()),
                                    data.size() / sizeof(bool)));
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported input type for tensor ", name, ": ",
                       "type=", tensors.at(name)->GetType()));
  }
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::Init(
    const ml_drift::ExternalTensorsInfo& external_tensors) {
  LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());
  ml_drift::CreateGpuModelInfo ci = DefaultCreateGpuModelInfo(precision_);
  return Backend::InitInferenceContext(&inference_context_, ci, &gpu_model_,
                                       env_ptr_, external_tensors);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
const Tensor<MlDriftMixinTag>& MlDriftModelRunnerBase<
    Backend, ModelFunctor, Inputs,
    Outputs>::GetIntermediateTensor(absl::string_view name) const {
  return outputs_.intermediate_tensors.at(name);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::Run() {
  LRT_TENSOR_RETURN_IF_ERROR(Enqueue());
  return Wait();
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::Enqueue() {
  LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());
  return Backend::AddToQueue(&inference_context_, env_ptr_);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::Wait() {
  return Backend::WaitForCompletion(env_ptr_);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
template <typename StorageT, typename TensorT>
absl::StatusOr<std::vector<StorageT>>
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs,
                       Outputs>::GetOutputInternal(absl::string_view name) {
  if (!name_to_tensor_map_.contains(name)) {
    return absl::NotFoundError(absl::StrCat("Output tensor not found: ", name));
  }
  LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());
  const auto& tensor = name_to_tensor_map_.at(name);
  auto it = conversion_ctx_ptr_->graph_tensor_to_handle.find(tensor);
  if (it == conversion_ctx_ptr_->graph_tensor_to_handle.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Output tensor '", name, "' was not found in the ML Drift model map."));
  }

  const auto& th = it->second;
  BackendTensor* gpu_tensor = Backend::GetTensor(&inference_context_, th.id);
  ml_drift::TensorDescriptor descriptor;
  LRT_TENSOR_RETURN_IF_ERROR(
      Backend::TensorToDescriptor(gpu_tensor, env_ptr_, &descriptor));
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
  size_t data_size = BufferSize(info.type, info.GetSize()) / sizeof(StorageT);
  std::vector<StorageT> data(data_size);
  descriptor.DownloadData(reinterpret_cast<TensorT*>(data.data()));
  return data;
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::StatusOr<std::vector<float>>
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::GetFloatOutput(
    absl::string_view name) {
  return GetOutputInternal<float, float>(name);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::StatusOr<std::vector<int32_t>>
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::GetInt32Output(
    absl::string_view name) {
  return GetOutputInternal<int32_t, int32_t>(name);
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::StatusOr<absl::Span<const float>>
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::GetOutput(
    absl::string_view name) {
  if (!name_to_tensor_map_.contains(name)) {
    return absl::NotFoundError(absl::StrCat("Output tensor not found: ", name));
  }
  LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());
  const auto& tensor = name_to_tensor_map_.at(name);
  if (!conversion_ctx_ptr_->graph_tensor_to_handle.contains(tensor)) {
    return absl::NotFoundError(absl::StrCat(
        "Output tensor ", name, " was not found in the ML Drift model map."));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
  switch (info.type) {
    case Type::kFP32: {
      LRT_TENSOR_ASSIGN_OR_RETURN(outputs_cache_[name],
                                  (GetOutputInternal<std::byte, float>(name)));
      const auto& output_data = outputs_cache_.at(name);
      return absl::MakeConstSpan(
          reinterpret_cast<const float*>(output_data.data()),
          output_data.size() / sizeof(float));
    }
    case Type::kFP16: {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto fp16_data,
                                  (GetOutputInternal<fp16_t, fp16_t>(name)));
      outputs_cache_[name].resize(fp16_data.size() * sizeof(float));
      absl::Span<float> float_data = absl::MakeConstSpan(
          reinterpret_cast<float*>(outputs_cache_.at(name).data()),
          fp16_data.size());
      for (size_t i = 0; i < fp16_data.size(); ++i) {
        float_data[i] = static_cast<float>(fp16_data[i]);
      }
      return float_data;
    }
    case Type::kI32: {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto int32_data,
                                  (GetOutputInternal<int32_t, int32_t>(name)));
      outputs_cache_[name].resize(int32_data.size() * sizeof(float));
      absl::Span<float> float_data = absl::MakeConstSpan(
          reinterpret_cast<float*>(outputs_cache_.at(name).data()),
          int32_data.size());
      for (size_t i = 0; i < int32_data.size(); ++i) {
        float_data[i] = static_cast<float>(int32_data[i]);
      }
      return float_data;
    }
    case Type::kI8: {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto int8_data,
                                  (GetOutputInternal<int8_t, int8_t>(name)));
      outputs_cache_[name].resize(int8_data.size() * sizeof(float));
      absl::Span<float> float_data = absl::MakeConstSpan(
          reinterpret_cast<float*>(outputs_cache_.at(name).data()),
          int8_data.size());
      for (size_t i = 0; i < int8_data.size(); ++i) {
        float_data[i] = static_cast<float>(int8_data[i]);
      }
      return float_data;
    }
    case Type::kBOOL: {
      LRT_TENSOR_ASSIGN_OR_RETURN(auto bool_data,
                                  (GetOutputInternal<bool, bool>(name)));
      outputs_cache_[name].resize(bool_data.size() * sizeof(float));
      absl::Span<float> float_data = absl::MakeConstSpan(
          reinterpret_cast<float*>(outputs_cache_.at(name).data()),
          bool_data.size());
      for (size_t i = 0; i < bool_data.size(); ++i) {
        float_data[i] = static_cast<float>(bool_data[i]);
      }
      return float_data;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported output type for tensor ", name, ": ",
                       "type=", info.type));
  }
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::ReadOutput(
    absl::string_view name, absl::Span<std::byte> data) {
  if (!name_to_tensor_map_.contains(name)) {
    return absl::NotFoundError(absl::StrCat("Output tensor not found: ", name));
  }
  LRT_TENSOR_RETURN_IF_ERROR(EnsureEnvInitialized());
  const auto& tensor = name_to_tensor_map_.at(name);
  auto it = conversion_ctx_ptr_->graph_tensor_to_handle.find(tensor);
  if (it == conversion_ctx_ptr_->graph_tensor_to_handle.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Output tensor '", name, "' was not found in the ML Drift model map."));
  }

  const auto& th = it->second;
  BackendTensor* gpu_tensor = Backend::GetTensor(&inference_context_, th.id);
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
  size_t required_size = BufferSize(info.type, info.GetSize());
  if (data.size() < required_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Output buffer is too small for tensor ", name,
        ". Required size: ", required_size, ", provided size: ", data.size()));
  }

  ml_drift::TensorDescriptor descriptor;
  LRT_TENSOR_RETURN_IF_ERROR(
      Backend::TensorToDescriptor(gpu_tensor, env_ptr_, &descriptor));
  switch (info.type) {
    case Type::kFP32:
      descriptor.DownloadData(reinterpret_cast<float*>(data.data()));
      return absl::OkStatus();
    case Type::kFP16:
      // Use ml_drift::half as DownlaodData() doesn't work with fp16_t.
      descriptor.DownloadData(reinterpret_cast<ml_drift::half*>(data.data()));
      return absl::OkStatus();
    case Type::kI32:
      descriptor.DownloadData(reinterpret_cast<int32_t*>(data.data()));
      return absl::OkStatus();
    case Type::kI8:
      descriptor.DownloadData(reinterpret_cast<int8_t*>(data.data()));
      return absl::OkStatus();
    case Type::kBOOL:
      descriptor.DownloadData(reinterpret_cast<bool*>(data.data()));
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported output type for tensor ", name, ": ",
                       "type=", info.type));
  }
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::Status
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs, Outputs>::CreateWeights(
    const absl::flat_hash_map<std::string, absl::Span<const float>>&
        float_weights,
    const absl::flat_hash_map<std::string, absl::Span<const int8_t>>&
        int8_weights) {
  MlDriftDummyWeightCreator weight_creator(
      conversion_ctx_ptr_->gpu_model_builder);
  const auto& input_tensors = inputs_.tensors();

  for (const auto& [weight_name, data] : float_weights) {
    auto input_it = input_tensors.find(weight_name);
    if (input_it == input_tensors.end()) {
      return absl::NotFoundError(
          absl::StrCat("Weight tensor not found in inputs: ", weight_name));
    }
    TensorMLD* weight_tensor = input_it->second;
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                graph::GetInfo(weight_tensor->GetRaw()));

    auto weight_handle_it =
        conversion_ctx_ptr_->weight_name_to_handle.find(weight_name);
    if (weight_handle_it != conversion_ctx_ptr_->weight_name_to_handle.end()) {
      conversion_ctx_ptr_->graph_tensor_to_handle[weight_tensor->GetRaw()] =
          weight_handle_it->second;
      continue;
    }

    if (info.shape.size() != 2) {
      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto handle_or,
          weight_creator.CreateConstantTensor(
              info.shape, data, conversion_ctx_ptr_->default_float_type));
      conversion_ctx_ptr_->graph_tensor_to_handle[weight_tensor->GetRaw()] =
          handle_or;
      conversion_ctx_ptr_->weight_name_to_handle[weight_name] = handle_or;
      continue;
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto handle_or,
        weight_creator.CreateFcWeights<float>(
            *weight_tensor, data, conversion_ctx_ptr_->default_float_type));
    conversion_ctx_ptr_->graph_tensor_to_handle[weight_tensor->GetRaw()] =
        handle_or;
    conversion_ctx_ptr_->weight_name_to_handle[weight_name] = handle_or;
  }

  for (const auto& [weight_name, data] : int8_weights) {
    auto input_it = input_tensors.find(weight_name);
    if (input_it == input_tensors.end()) {
      return absl::NotFoundError(
          absl::StrCat("Weight tensor not found in inputs: ", weight_name));
    }
    TensorMLD* weight_tensor = input_it->second;
    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                graph::GetInfo(weight_tensor->GetRaw()));

    auto weight_handle_it =
        conversion_ctx_ptr_->weight_name_to_handle.find(weight_name);
    if (weight_handle_it != conversion_ctx_ptr_->weight_name_to_handle.end()) {
      conversion_ctx_ptr_->graph_tensor_to_handle[weight_tensor->GetRaw()] =
          weight_handle_it->second;
      continue;
    }

    if (info.shape.size() == 1) {
      ABSL_LOG(WARNING) << "Skipping 1D int8/int4/int2 weight: " << weight_name
                        << ", treating as regular input.";
      continue;
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto handle_or,
        weight_creator.CreateFcWeights<int8_t>(*weight_tensor, data));
    conversion_ctx_ptr_->graph_tensor_to_handle[weight_tensor->GetRaw()] =
        handle_or;
    conversion_ctx_ptr_->weight_name_to_handle[weight_name] = handle_or;
  }
  return absl::OkStatus();
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::StatusOr<ml_drift::ValueId> MlDriftModelRunnerBase<
    Backend, ModelFunctor, Inputs, Outputs>::GetWeightValueId(const std::string&
                                                                  name) const {
  const auto& input_tensors = cached_input_tensors_;
  if (!input_tensors.contains(name)) {
    return absl::NotFoundError(absl::StrCat("Weight name not found: ", name));
  }
  const auto& raw = input_tensors.at(name)->GetRaw();
  auto it = conversion_ctx_ptr_->graph_tensor_to_handle.find(raw);
  if (it == conversion_ctx_ptr_->graph_tensor_to_handle.end()) {
    return absl::NotFoundError(
        absl::StrCat("Mapping not found for weight: ", name));
  }
  return it->second.id;
}

template <typename Backend, typename ModelFunctor, typename Inputs,
          typename Outputs>
absl::flat_hash_map<ml_drift::ValueId,
                    typename MlDriftModelRunnerBase<
                        Backend, ModelFunctor, Inputs, Outputs>::BackendTensor*>
MlDriftModelRunnerBase<Backend, ModelFunctor, Inputs,
                       Outputs>::GetWeightTensors() {
  absl::flat_hash_map<ml_drift::ValueId, BackendTensor*> map;
  for (const auto& [name, handle] :
       conversion_ctx_ptr_->weight_name_to_handle) {
    auto* tensor = Backend::GetTensor(&inference_context_, handle.id);
    if (tensor != nullptr) {
      map[handle.id] = tensor;
    }
  }
  return map;
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_ML_DRIFT_MODEL_RUNNER_BASE_H_
