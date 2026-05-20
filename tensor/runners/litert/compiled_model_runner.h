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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_COMPILED_MODEL_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_COMPILED_MODEL_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_probe.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/tensor.h"

namespace litert {
namespace tensor {

template <typename ModelFunctor, typename Inputs, typename Outputs>
class CompiledModelRunner {
 public:
  explicit CompiledModelRunner(Environment& env, Options& options,
                               ModelFunctor model_func,
                               bool build_model_now = true);

  absl::Status BuildModel(
      const std::vector<Tensor<TfLiteMixinTag>>& output_tensors = {});

  absl::Status SetInput(const std::string& name,
                        const std::vector<float>& data);
  absl::Status SetInput(const std::string& name,
                        const std::vector<int32_t>& data);
  absl::Status SetInput(const std::string& name,
                        const std::vector<int8_t>& data);
  absl::Status SetInput(const std::string& name, const std::vector<bool>& data);
  absl::Status SetInput(const std::string& name,
                        const std::vector<uint8_t>& data);
  absl::Status SetInput(const std::string& name, const TensorHandle& tensor);
  absl::Status SetInput(const std::string& name,
                        absl::Span<const uint8_t> data);

  absl::Status Run();

  absl::StatusOr<std::vector<float>> GetFloatOutput(const std::string& name);
  absl::StatusOr<std::vector<int32_t>> GetInt32Output(const std::string& name);
  absl::StatusOr<std::vector<bool>> GetBoolOutput(const std::string& name);

  absl::StatusOr<TensorHandle> GetOutput(const std::string& name);
  absl::Status SetOutput(const std::string& name, const TensorHandle& tensor);

  absl::StatusOr<std::vector<std::string>> GetInputNames() const;

  static Type ConvertType(ElementType et) {
    switch (et) {
      case ElementType::Float32:
        return Type::kFP32;
      case ElementType::Int32:
        return Type::kI32;
      case ElementType::Bool:
        return Type::kBOOL;
      default:
        return Type::kUnknown;
    }
  }

  absl::StatusOr<TensorHandle> GetInput(const std::string& name) {
    LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
    for (size_t i = 0; i < signature.InputNames().size(); ++i) {
      if (signature.InputNames()[i] == name) {
        TensorInit init;
        init.name = name;

        LITERT_ASSIGN_OR_RETURN(auto tensor_type, signature.InputTensorType(i));
        init.type = ConvertType(tensor_type.ElementType());

        auto shape_vector = tensor_type.Layout().Dimensions();
        init.shape =
            std::vector<int32_t>(shape_vector.begin(), shape_vector.end());

        auto dup_or = input_buffers_[i].Duplicate();
        if (!dup_or.HasValue())
          return absl::InternalError("Failed to duplicate buffer");

        init.buffer = std::make_shared<LitertBuffer>(std::move(*dup_or));
        return TensorHandle(init);
      }
    }
    return absl::NotFoundError(absl::StrCat("Input tensor not found: ", name));
  }

  absl::Status AddTensorsAsOutputs(
      const absl::flat_hash_map<GraphProbe::StableTensorId, std::string,
                                GraphProbe::StableTensorIdHash>& probe_tensors);

 private:
  static Environment CreateEnvironmentOrDie() {
    LITERT_ASSIGN_OR_ABORT(auto env, Environment::Create({}));
    return env;
  }

  using TensorTf = Tensor<TfLiteMixinTag>;

  ModelFunctor model_func_;
  Inputs inputs_;
  Outputs outputs_;

  absl::flat_hash_map<GraphProbe::StableTensorId, std::string,
                      GraphProbe::StableTensorIdHash>
      probed_tensors_;
  std::vector<char> model_buffer_;
  Environment& env_;
  Options& options_;
  CompiledModel compiled_model_;

  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
};

template <typename ModelFunctor, typename Inputs, typename Outputs>
CompiledModelRunner<ModelFunctor, Inputs, Outputs>::CompiledModelRunner(
    Environment& env, Options& options, ModelFunctor model_func,
    bool build_model_now)
    : model_func_(model_func), env_(env), options_(options) {
  outputs_ = model_func_(inputs_);
  if (build_model_now) {
    std::vector<TensorTf> output_tensors;
    for (auto const& [name, tensor_ptr] : outputs_.tensors()) {
      tensor_ptr->SetName(name);
      output_tensors.push_back(*tensor_ptr);
    }
    ABSL_CHECK_OK(BuildModel(output_tensors));
  }
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status CompiledModelRunner<ModelFunctor, Inputs, Outputs>::BuildModel(
    const std::vector<TensorTf>& output_tensors) {
  absl::Status status = Save(output_tensors, model_buffer_);
  ABSL_CHECK_OK(status) << "Failed to serialize model to flatbuffer.";

  BufferRef<> model_buffer(model_buffer_.data(), model_buffer_.size());
  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_buffer, options_));

  LITERT_ASSIGN_OR_RETURN(input_buffers_,
                              compiled_model_.CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(output_buffers_,
                              compiled_model_.CreateOutputBuffers());
  return absl::OkStatus();
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status
CompiledModelRunner<ModelFunctor, Inputs, Outputs>::AddTensorsAsOutputs(
    const absl::flat_hash_map<GraphProbe::StableTensorId, std::string,
                              GraphProbe::StableTensorIdHash>& probe_tensors) {
  probed_tensors_ = probe_tensors;

  std::vector<TensorTf> original_output_tensors;
  for (auto const& [name, tensor_ptr] : outputs_.tensors()) {
    tensor_ptr->SetName(name);
    original_output_tensors.push_back(*tensor_ptr);
  }

  std::vector<TensorHandle> original_output_handles;
  original_output_handles.reserve(original_output_tensors.size());
  for (const auto& t : original_output_tensors) {
    original_output_handles.push_back(t);
  }

  LITERT_ASSIGN_OR_RETURN(auto execution_plan,
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

  std::vector<TensorTf> output_tensors;
  if (probe_tensors.empty()) {
    output_tensors = original_output_tensors;
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
            TensorTf original_tensor(tensor);
            TensorTf probed_tensor = Probe(original_tensor);
            probed_tensor.SetName(probe_it->second);
            output_tensors.push_back(probed_tensor);
          } else {
            LITERT_RETURN_IF_ERROR(
                graph::SetName(tensor, probe_it->second));
            output_tensors.push_back(TensorTf(tensor));
          }
        }
      }
    }
  }

  return BuildModel(output_tensors);
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status CompiledModelRunner<ModelFunctor, Inputs, Outputs>::SetInput(
    const std::string& name, const std::vector<float>& data) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (int i = 0; i < signature.InputNames().size(); ++i) {
    if (signature.InputNames()[i] == name) {
      LITERT_RETURN_IF_ERROR(
          input_buffers_[i].Write(absl::MakeConstSpan(data)));
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError(absl::StrCat("Input tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status CompiledModelRunner<ModelFunctor, Inputs, Outputs>::SetInput(
    const std::string& name, const std::vector<int8_t>& data) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (int i = 0; i < signature.InputNames().size(); ++i) {
    if (signature.InputNames()[i] == name) {
      LITERT_RETURN_IF_ERROR(
          input_buffers_[i].Write(absl::MakeConstSpan(data)));
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError(absl::StrCat("Input tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status CompiledModelRunner<ModelFunctor, Inputs, Outputs>::SetInput(
    const std::string& name, const std::vector<int32_t>& data) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (int i = 0; i < signature.InputNames().size(); ++i) {
    if (signature.InputNames()[i] == name) {
      LITERT_RETURN_IF_ERROR(
          input_buffers_[i].Write(absl::MakeConstSpan(data)));
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError(absl::StrCat("Input tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status CompiledModelRunner<ModelFunctor, Inputs, Outputs>::SetInput(
    const std::string& name, const TensorHandle& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));

  for (int i = 0; i < signature.InputNames().size(); ++i) {
    if (signature.InputNames()[i] == name) {
      auto buffer_or = tensor.GetBuffer();
      if (!buffer_or.ok()) {
        return absl::InvalidArgumentError("Tensor has no buffer");
      }
      Buffer& buffer = *buffer_or;
      auto* litert_buffer = dynamic_cast<LitertBuffer*>(&buffer);
      if (litert_buffer != nullptr) {
        auto dup_or = litert_buffer->tensor_buffer().Duplicate();
        if (!dup_or.HasValue()) {
          return absl::InternalError("Failed to duplicate TensorBuffer");
        }
        input_buffers_[i] = std::move(*dup_or);
      } else {
        auto locked_span = buffer.Lock();
        auto span = absl::MakeConstSpan(
            reinterpret_cast<const uint8_t*>(locked_span.data()),
            locked_span.size());
        auto res = input_buffers_[i].Write(span);
        if (!res.HasValue()) {
          return absl::InternalError("Failed to write input buffer");
        }
      }
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError(absl::StrCat("Input tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status CompiledModelRunner<ModelFunctor, Inputs, Outputs>::SetInput(
    const std::string& name, absl::Span<const uint8_t> data) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));

  for (int i = 0; i < signature.InputNames().size(); ++i) {
    if (signature.InputNames()[i] == name) {
      auto res = input_buffers_[i].Write(data);
      if (!res.HasValue()) {
        return absl::InternalError("Failed to write input buffer");
      }
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError(absl::StrCat("Input tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status CompiledModelRunner<ModelFunctor, Inputs, Outputs>::SetOutput(
    const std::string& name, const TensorHandle& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (int i = 0; i < signature.OutputNames().size(); ++i) {
    if (signature.OutputNames()[i] == name) {
      auto buffer_or = tensor.GetBuffer();
      if (!buffer_or.ok()) {
        return absl::InvalidArgumentError("Tensor has no buffer");
      }
      Buffer& buffer = *buffer_or;
      auto* litert_buffer = dynamic_cast<LitertBuffer*>(&buffer);
      if (litert_buffer != nullptr) {
        auto dup_or = litert_buffer->tensor_buffer().Duplicate();
        if (!dup_or.HasValue()) {
          return absl::InternalError("Failed to duplicate TensorBuffer");
        }
        output_buffers_[i] = std::move(*dup_or);
      } else {
        return absl::UnimplementedError(
            "SetOutput only supported for LitertBuffer");
      }
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError(absl::StrCat("Output tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status CompiledModelRunner<ModelFunctor, Inputs, Outputs>::SetInput(
    const std::string& name, const std::vector<bool>& data) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (int i = 0; i < signature.InputNames().size(); ++i) {
    if (signature.InputNames()[i] == name) {
      std::vector<uint8_t> temp_data(data.begin(), data.end());
      LITERT_RETURN_IF_ERROR(
          input_buffers_[i].Write(absl::MakeConstSpan(temp_data)));
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError(absl::StrCat("Input tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status CompiledModelRunner<ModelFunctor, Inputs, Outputs>::SetInput(
    const std::string& name, const std::vector<uint8_t>& data) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (int i = 0; i < signature.InputNames().size(); ++i) {
    if (signature.InputNames()[i] == name) {
      LITERT_RETURN_IF_ERROR(
          input_buffers_[i].Write(absl::MakeConstSpan(data)));
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError(absl::StrCat("Input tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::Status CompiledModelRunner<ModelFunctor, Inputs, Outputs>::Run() {
  LITERT_RETURN_IF_ERROR(
      compiled_model_.Run(input_buffers_, output_buffers_));
  return absl::OkStatus();
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::StatusOr<std::vector<float>>
CompiledModelRunner<ModelFunctor, Inputs, Outputs>::GetFloatOutput(
    const std::string& name) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (int i = 0; i < signature.OutputNames().size(); ++i) {
    if (signature.OutputNames()[i] == name) {
      LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type,
                                  compiled_model_.GetOutputTensorType(0, i));
      size_t num_elements = 1;
      for (int dim : ranked_tensor_type.Layout().Dimensions()) {
        num_elements *= dim;
      }
      std::vector<float> output_data(num_elements);
      LITERT_RETURN_IF_ERROR(
          output_buffers_[i].Read(absl::MakeSpan(output_data)));
      return output_data;
    }
  }
  return absl::NotFoundError(absl::StrCat("Output tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::StatusOr<std::vector<int32_t>>
CompiledModelRunner<ModelFunctor, Inputs, Outputs>::GetInt32Output(
    const std::string& name) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (int i = 0; i < signature.OutputNames().size(); ++i) {
    if (signature.OutputNames()[i] == name) {
      LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type,
                                  compiled_model_.GetOutputTensorType(0, i));
      size_t num_elements = 1;
      for (int dim : ranked_tensor_type.Layout().Dimensions()) {
        num_elements *= dim;
      }
      std::vector<int32_t> output_data(num_elements);
      LITERT_RETURN_IF_ERROR(
          output_buffers_[i].Read(absl::MakeSpan(output_data)));
      return output_data;
    }
  }
  return absl::NotFoundError(absl::StrCat("Output tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::StatusOr<std::vector<bool>>
CompiledModelRunner<ModelFunctor, Inputs, Outputs>::GetBoolOutput(
    const std::string& name) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (int i = 0; i < signature.OutputNames().size(); ++i) {
    if (signature.OutputNames()[i] == name) {
      LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type,
                                  compiled_model_.GetOutputTensorType(0, i));
      size_t num_elements = 1;
      for (int dim : ranked_tensor_type.Layout().Dimensions()) {
        num_elements *= dim;
      }
      std::vector<uint8_t> temp_data(num_elements);
      LITERT_RETURN_IF_ERROR(
          output_buffers_[i].Read(absl::MakeSpan(temp_data)));
      std::vector<bool> output_data(temp_data.begin(), temp_data.end());
      return output_data;
    }
  }
  return absl::NotFoundError(absl::StrCat("Output tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::StatusOr<TensorHandle>
CompiledModelRunner<ModelFunctor, Inputs, Outputs>::GetOutput(
    const std::string& name) {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (int i = 0; i < signature.OutputNames().size(); ++i) {
    if (signature.OutputNames()[i] == name) {
      LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type,
                                  compiled_model_.GetOutputTensorType(0, i));

      auto dup_or = output_buffers_[i].Duplicate();
      if (!dup_or.HasValue()) {
        return absl::InternalError("Failed to duplicate TensorBuffer");
      }

      auto litert_buffer = std::make_shared<LitertBuffer>(std::move(*dup_or));

      Type type = Type::kUnknown;
      switch (ranked_tensor_type.ElementType()) {
        case ElementType::Float32:
          type = Type::kFP32;
          break;
        case ElementType::Int32:
          type = Type::kI32;
          break;
        case ElementType::Int8:
          type = Type::kI8;
          break;
        default:
          break;
      }

      Shape shape;
      for (int dim : ranked_tensor_type.Layout().Dimensions()) {
        shape.push_back(dim);
      }

      TensorInit init;
      init.name = name;
      init.type = type;
      init.shape = std::move(shape);
      init.buffer = litert_buffer;

      return TensorHandle(init);
    }
  }
  return absl::NotFoundError(absl::StrCat("Output tensor not found: ", name));
}

template <typename ModelFunctor, typename Inputs, typename Outputs>
absl::StatusOr<std::vector<std::string>>
CompiledModelRunner<ModelFunctor, Inputs, Outputs>::GetInputNames() const {
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  std::vector<std::string> input_names;
  for (const auto& name : signature.InputNames()) {
    input_names.emplace_back(name);
  }
  return input_names;
}

}  // namespace tensor
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_COMPILED_MODEL_RUNNER_H_
