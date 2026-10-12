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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_TRACE_ML_DRIFT_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_TRACE_ML_DRIFT_RUNNER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "tensor/backends/ml_drift/arithmetic_ml_drift.h"
#include "tensor/backends/ml_drift/ml_drift_buffer.h"
#include "tensor/backends/ml_drift/ml_drift_conversion.h"
#include "tensor/buffer.h"
#include "tensor/internal/graph.h"
#include "tensor/runners/ml_drift/ml_drift_cl_model_runner.h"
#include "tensor/runners/ml_drift/ml_drift_model_runner_base.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

struct DummyInputs {
  absl::flat_hash_map<std::string, Tensor<MlDriftMixinTag>*> tensors() const {
    return {};
  }
};

struct DummyOutputs {
  absl::flat_hash_map<std::string, Tensor<MlDriftMixinTag>*> tensors() const {
    return map;
  }
  absl::flat_hash_map<std::string, Tensor<MlDriftMixinTag>*> map;
  absl::flat_hash_map<std::string, Tensor<MlDriftMixinTag>>
      intermediate_tensors;
};

struct DummyFunctor {
  std::vector<TensorHandle> flat_outputs;
  explicit DummyFunctor(std::vector<TensorHandle> flat_outputs)
      : flat_outputs(flat_outputs) {}
  DummyOutputs operator()(DummyInputs&) const { return {}; }
};

// A specialized runner that inherits the execution capabilities of
// MlDriftModelRunnerBase but overrides the graph building and input mapping
// to directly consume a topologically sorted execution plan from `trace()`.
class TraceMlDriftRunner
    : public MlDriftModelRunnerBase<MlDriftClBackend, DummyFunctor, DummyInputs,
                                    DummyOutputs> {
 private:
  std::vector<TensorHandle> flat_inputs_;
  std::vector<TensorHandle> flat_outputs_;

 public:
  using Base = MlDriftModelRunnerBase<MlDriftClBackend, DummyFunctor,
                                      DummyInputs, DummyOutputs>;

  static DummyInputs& GetDummyInputs() {
    static absl::NoDestructor<DummyInputs> dummy_inputs;
    return *dummy_inputs;
  }

  static DummyOutputs& GetDummyOutputs() {
    static absl::NoDestructor<DummyOutputs> dummy_outputs;
    return *dummy_outputs;
  }

  explicit TraceMlDriftRunner(std::vector<TensorHandle> flat_inputs,
                              std::vector<TensorHandle> flat_outputs,
                              ml_drift::CalculationsPrecision precision =
                                  ml_drift::CalculationsPrecision::kF32)
      : Base(DummyFunctor(flat_outputs), GetDummyInputs(), GetDummyOutputs(),
             /*float_weights=*/{}, /*int8_weights=*/{},
             /*build_gpu_model=*/false, precision),
        flat_inputs_(std::move(flat_inputs)),
        flat_outputs_(std::move(flat_outputs)) {}

  absl::Status BuildModel() {
    const absl::Time compilation_start = absl::Now();

    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto tensor_to_handle,
        AppendGpuModelBuilder(this->conversion_ctx(), flat_outputs_));
    this->conversion_ctx().graph_tensor_to_handle = std::move(tensor_to_handle);

    // Populate name to tensor map
    this->name_to_tensor_map_.clear();
    for (const auto& handle : flat_outputs_) {
      LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                  graph::GetInfo(handle.GetRaw()));
      if (!info.name.empty()) {
        this->name_to_tensor_map_[info.name] = handle.GetRaw();
      }
    }

    // Explicitly add inputs to the name map so GetFloatOutput / SetInput work
    for (const auto& handle : flat_inputs_) {
      LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info,
                                  graph::GetInfo(handle.GetRaw()));
      if (!info.name.empty()) {
        this->name_to_tensor_map_[info.name] = handle.GetRaw();
      }
    }

    this->graph_input_ids_.clear();
    for (const auto& input_handle : flat_inputs_) {
      if (this->conversion_ctx().graph_tensor_to_handle.contains(
              input_handle.GetRaw())) {
        this->graph_input_ids_.push_back(
            this->conversion_ctx()
                .graph_tensor_to_handle.at(input_handle.GetRaw())
                .id);
      }
    }

    this->graph_output_ids_.clear();
    for (const auto& output_handle : flat_outputs_) {
      if (this->conversion_ctx().graph_tensor_to_handle.contains(
              output_handle.GetRaw())) {
        this->graph_output_ids_.push_back(
            this->conversion_ctx()
                .graph_tensor_to_handle.at(output_handle.GetRaw())
                .id);
      }
    }

    LRT_TENSOR_RETURN_IF_ERROR(
        this->conversion_ctx().gpu_model_builder->GetGpuModel(
            this->graph_input_ids_, this->graph_output_ids_,
            &this->gpu_model_));

    // Initialize the inference context with the precision and storage type.
    // Note: precision is defaulted to F32 for now.
    ml_drift::CreateGpuModelInfo ci = {
        .precision = ml_drift::CalculationsPrecision::kF32,
        .storage_type = MlDriftClBackend::GetStorageTypeForBuilder(
            this->env(), /*env_initialized=*/true),
    };
    LRT_TENSOR_RETURN_IF_ERROR(MlDriftClBackend::InitInferenceContext(
        &this->inference_context(), ci, &this->gpu_model_, &this->env()));

    this->compilation_time_ = absl::Now() - compilation_start;
    return absl::OkStatus();
  }

  // Override SetInput to use flat_inputs_ instead of the dummy inputs_ struct.
  absl::Status SetInput(const std::string& name,
                        const std::vector<float>& data) {
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      ABSL_LOG(WARNING) << "Input tensor not found: " << name
                        << " data size: " << data.size();
      return absl::InvalidArgumentError("Input tensor not found: " + name);
    }

    auto it =
        this->conversion_ctx().graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == this->conversion_ctx().graph_tensor_to_handle.end()) {
      return absl::InvalidArgumentError("Input tensor not found: " + name);
    }

    LRT_TENSOR_RETURN_IF_ERROR(this->EnsureEnvInitialized());
    const auto& th = it->second;
    ml_drift::Tensor<ml_drift::BHWC, ml_drift::DataType::kFloat32> t;
    t.id = th.id;
    t.data.resize(th.tensor_desc.GetBHWCShape().DimensionsProduct());
    t.shape = th.tensor_desc.GetBHWCShape();
    t.data.assign(data.begin(), data.end());
    return MlDriftClBackend::SetInputTensorFloat(&this->inference_context(),
                                                 &this->env(), t.id, t);
  }

  absl::Status SetInput(const std::string& name,
                        Tensor<MlDriftMixinTag>& tensor_input) {
    auto buffer_or = tensor_input.GetBuffer();
    if (!buffer_or.ok())
      return absl::InvalidArgumentError("Tensor has no buffer");

    // Extracting float data (fallback for CpuBuffers).
    auto locked = buffer_or.value().Lock();
    auto span = std::move(locked).template As<const float>();
    std::vector<float> data(span.begin(), span.end());
    return SetInput(name, data);
  }

  absl::StatusOr<std::shared_ptr<Buffer>> GetOutputBuffer(
      const std::string& name) {
    auto raw_tensor_ptr = FindTensorByName(name);
    if (!raw_tensor_ptr) {
      return absl::NotFoundError(
          absl::StrCat("Output tensor not found: ", name));
    }
    auto it =
        this->conversion_ctx().graph_tensor_to_handle.find(*raw_tensor_ptr);
    if (it == this->conversion_ctx().graph_tensor_to_handle.end()) {
      return absl::NotFoundError(
          absl::StrCat("Output tensor handle not found: ", name));
    }
    const auto& th = it->second;
    auto* tensor = this->inference_context().GetTensor(th.id);
    if (!tensor) {
      return absl::InternalError("Internal tensor not found");
    }
    size_t size_bytes = tensor->GetMemorySizeInBytes();
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto buf, MlDriftBuffer::Create(th.id, this->inference_context(),
                                        *this->env().queue(), size_bytes));
    return std::shared_ptr<Buffer>(std::move(buf));
  }

 private:
  std::optional<graph::Tensor> FindTensorByName(const std::string& name) {
    auto it = this->name_to_tensor_map_.find(name);
    if (it != this->name_to_tensor_map_.end()) {
      return it->second;
    }
    return std::nullopt;
  }
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_ML_DRIFT_TRACE_ML_DRIFT_RUNNER_H_
