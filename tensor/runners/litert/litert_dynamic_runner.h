/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LITERT_DYNAMIC_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LITERT_DYNAMIC_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/tensor.h"

namespace litert {
namespace tensor {

class LitertDynamicRunner {
 public:
  static absl::StatusOr<LitertDynamicRunner> Create(
      Environment& env, const std::string& model_path, Options& options) {
    LitertDynamicRunner runner;
    LITERT_ASSIGN_OR_RETURN(runner.compiled_model_,
                            CompiledModel::Create(env, model_path, options));
    LITERT_RETURN_IF_ERROR(runner.InitializeBuffers());
    return runner;
  }

  static absl::StatusOr<LitertDynamicRunner> Create(
      Environment& env, absl::Span<const uint8_t> model_buffer,
      Options& options) {
    LitertDynamicRunner runner;
    BufferRef<uint8_t> buf_ref(model_buffer.data(), model_buffer.size());
    LITERT_ASSIGN_OR_RETURN(runner.compiled_model_,
                            CompiledModel::Create(env, buf_ref, options));
    LITERT_RETURN_IF_ERROR(runner.InitializeBuffers());
    return runner;
  }

  // Helper to initialize buffers for all signatures
  absl::Status InitializeBuffers() {
    LITERT_ASSIGN_OR_RETURN(auto keys, compiled_model_.GetSignatureKeys());
    if (keys.empty()) return absl::InternalError("No signatures found");
    default_signature_name_ = std::string(keys[0]);

    for (const auto& key : keys) {
      std::string key_str(key);
      LITERT_ASSIGN_OR_RETURN(auto in_buffers,
                              compiled_model_.CreateInputBuffers(key_str));
      signature_input_buffers_[key_str] = std::move(in_buffers);

      LITERT_ASSIGN_OR_RETURN(auto out_buffers,
                              compiled_model_.CreateOutputBuffers(key_str));
      signature_output_buffers_[key_str] = std::move(out_buffers);
    }
    return absl::OkStatus();
  }

  // Query input buffer index by name in a signature once at startup
  absl::StatusOr<size_t> GetInputIndex(const std::string& signature_name,
                                       const std::string& name) const {
    LITERT_ASSIGN_OR_RETURN(auto signature,
                            compiled_model_.FindSignature(signature_name));
    for (size_t i = 0; i < signature.InputNames().size(); ++i) {
      if (signature.InputNames()[i] == name) {
        return i;
      }
    }
    return absl::NotFoundError("Input tensor name not found in signature");
  }

  // Query output buffer index by name in a signature once at startup
  absl::StatusOr<size_t> GetOutputIndex(const std::string& signature_name,
                                        const std::string& name) const {
    LITERT_ASSIGN_OR_RETURN(auto signature,
                            compiled_model_.FindSignature(signature_name));
    for (size_t i = 0; i < signature.OutputNames().size(); ++i) {
      if (signature.OutputNames()[i] == name) {
        return i;
      }
    }
    return absl::NotFoundError("Output tensor name not found in signature");
  }

  // Non-signature overloads (default to first signature)
  absl::Status SetInput(const std::string& name, const TensorHandle& tensor) {
    return SetInput(default_signature_name_, name, tensor);
  }

  absl::Status SetInput(size_t index, const TensorHandle& tensor) {
    return SetInput(default_signature_name_, index, tensor);
  }

  absl::Status SetInput(const std::string& name,
                        absl::Span<const uint8_t> data) {
    return SetInput(default_signature_name_, name, data);
  }

  absl::Status SetInput(size_t index, absl::Span<const uint8_t> data) {
    return SetInput(default_signature_name_, index, data);
  }

  absl::Status Run() { return Run(default_signature_name_); }

  absl::StatusOr<TensorHandle> GetOutput(const std::string& name) {
    return GetOutput(default_signature_name_, name);
  }

  absl::StatusOr<TensorHandle> GetOutput(size_t index) {
    return GetOutput(default_signature_name_, index);
  }

  absl::StatusOr<TensorHandle> GetInput(const std::string& name) {
    return GetInput(default_signature_name_, name);
  }

  absl::StatusOr<TensorHandle> GetInput(size_t index) {
    return GetInput(default_signature_name_, index);
  }

  // Set input by signature and name
  absl::Status SetInput(const std::string& signature_name,
                        const std::string& name, const TensorHandle& tensor) {
    auto in_it = signature_input_buffers_.find(signature_name);
    if (in_it == signature_input_buffers_.end()) {
      return absl::NotFoundError("Signature not found");
    }

    LITERT_ASSIGN_OR_RETURN(auto signature,
                            compiled_model_.FindSignature(signature_name));

    for (size_t i = 0; i < signature.InputNames().size(); ++i) {
      if (signature.InputNames()[i] == name) {
        return SetInput(signature_name, i, tensor);
      }
    }
    return absl::NotFoundError("Input tensor not found");
  }

  // Set input by signature and index
  absl::Status SetInput(const std::string& signature_name, size_t index,
                        const TensorHandle& tensor) {
    auto in_it = signature_input_buffers_.find(signature_name);
    if (in_it == signature_input_buffers_.end()) {
      return absl::NotFoundError("Signature not found");
    }
    auto& input_buffers = in_it->second;

    if (index >= input_buffers.size())
      return absl::NotFoundError("Index out of bounds");

    LITERT_ASSIGN_OR_RETURN(Buffer & buffer, tensor.GetBuffer());
    auto litert_buffer_or = buffer.As<LitertBuffer>();
    if (litert_buffer_or.ok()) {
      LITERT_ASSIGN_OR_RETURN(input_buffers[index],
                              litert_buffer_or->tensor_buffer().Duplicate());
    } else {
      auto locked_span = buffer.Lock().As<const uint8_t>();
      LITERT_RETURN_IF_ERROR(
          input_buffers[index].Write(absl::Span<const uint8_t>(locked_span)));
    }
    return absl::OkStatus();
  }

  // Set input by signature and name with binary data
  absl::Status SetInput(const std::string& signature_name,
                        const std::string& name,
                        absl::Span<const uint8_t> data) {
    LITERT_ASSIGN_OR_RETURN(auto signature,
                            compiled_model_.FindSignature(signature_name));
    for (size_t i = 0; i < signature.InputNames().size(); ++i) {
      if (signature.InputNames()[i] == name) {
        return SetInput(signature_name, i, data);
      }
    }
    return absl::NotFoundError("Input tensor not found");
  }

  // Set input by signature and index with binary data
  absl::Status SetInput(const std::string& signature_name, size_t index,
                        absl::Span<const uint8_t> data) {
    auto in_it = signature_input_buffers_.find(signature_name);
    if (in_it == signature_input_buffers_.end()) {
      return absl::NotFoundError("Signature not found");
    }
    auto& input_buffers = in_it->second;

    if (index >= input_buffers.size())
      return absl::NotFoundError("Index out of bounds");
    auto res = input_buffers[index].Write(data);
    if (!res.HasValue()) {
      return absl::InternalError("Failed to write input buffer");
    }
    return absl::OkStatus();
  }

  // Run by signature
  absl::Status Run(const std::string& signature_name) {
    auto in_it = signature_input_buffers_.find(signature_name);
    auto out_it = signature_output_buffers_.find(signature_name);
    if (in_it == signature_input_buffers_.end() ||
        out_it == signature_output_buffers_.end()) {
      return absl::NotFoundError("Signature not found");
    }

    auto sig_index_or = compiled_model_.GetSignatureIndex(signature_name);
    if (!sig_index_or.HasValue()) {
      return absl::NotFoundError("Signature index not found in model");
    }

    auto status = compiled_model_.Run(sig_index_or.Value(), in_it->second,
                                      out_it->second);
    if (!status.HasValue()) {
      return absl::InternalError(status.Error().Message());
    }
    return absl::OkStatus();
  }

  // Get output by signature and name
  absl::StatusOr<TensorHandle> GetOutput(const std::string& signature_name,
                                         const std::string& name) {
    LITERT_ASSIGN_OR_RETURN(auto signature,
                            compiled_model_.FindSignature(signature_name));
    for (size_t i = 0; i < signature.OutputNames().size(); ++i) {
      if (signature.OutputNames()[i] == name) {
        return GetOutput(signature_name, i);
      }
    }
    return absl::NotFoundError("Output tensor not found");
  }

  // Get output by signature and index
  absl::StatusOr<TensorHandle> GetOutput(const std::string& signature_name,
                                         size_t index) {
    auto out_it = signature_output_buffers_.find(signature_name);
    if (out_it == signature_output_buffers_.end()) {
      return absl::NotFoundError("Signature not found");
    }
    auto& output_buffers = out_it->second;

    if (index >= output_buffers.size())
      return absl::NotFoundError("Index out of bounds");

    LITERT_ASSIGN_OR_RETURN(auto signature,
                            compiled_model_.FindSignature(signature_name));
    std::string name = "output";
    if (index < signature.OutputNames().size()) {
      name = signature.OutputNames()[index];
    }

    // Find signature index for CompiledModel API
    LITERT_ASSIGN_OR_RETURN(size_t sig_idx,
                            compiled_model_.GetSignatureIndex(signature_name));

    LITERT_ASSIGN_OR_RETURN(
        auto ranked_tensor_type,
        compiled_model_.GetOutputTensorType(sig_idx, index));

    auto dup_or = output_buffers[index].Duplicate();
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
      case ElementType::Bool:
        type = Type::kBOOL;
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

  // Get input by signature and name
  absl::StatusOr<TensorHandle> GetInput(const std::string& signature_name,
                                        const std::string& name) {
    LITERT_ASSIGN_OR_RETURN(auto signature,
                            compiled_model_.FindSignature(signature_name));
    for (size_t i = 0; i < signature.InputNames().size(); ++i) {
      if (signature.InputNames()[i] == name) {
        return GetInput(signature_name, i);
      }
    }
    return absl::NotFoundError("Input tensor not found");
  }

  // Get input by signature and index
  absl::StatusOr<TensorHandle> GetInput(const std::string& signature_name,
                                        size_t index) {
    auto in_it = signature_input_buffers_.find(signature_name);
    if (in_it == signature_input_buffers_.end()) {
      return absl::NotFoundError("Signature not found");
    }
    auto& input_buffers = in_it->second;

    if (index >= input_buffers.size())
      return absl::NotFoundError("Index out of bounds");

    LITERT_ASSIGN_OR_RETURN(auto signature,
                            compiled_model_.FindSignature(signature_name));
    std::string name = "input";
    if (index < signature.InputNames().size()) {
      name = signature.InputNames()[index];
    }

    LITERT_ASSIGN_OR_RETURN(size_t sig_idx,
                            compiled_model_.GetSignatureIndex(signature_name));

    LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type,
                            compiled_model_.GetInputTensorType(sig_idx, index));

    auto dup_or = input_buffers[index].Duplicate();
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
      case ElementType::Bool:
        type = Type::kBOOL;
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

  absl::StatusOr<uintptr_t> GetOutputWebGpuBuffer(
      const std::string& signature_name, const std::string& name) {
    auto out_it = signature_output_buffers_.find(signature_name);
    if (out_it == signature_output_buffers_.end()) {
      return absl::NotFoundError("Signature not found");
    }
    auto& output_buffers = out_it->second;

    LITERT_ASSIGN_OR_RETURN(auto signature,
                            compiled_model_.FindSignature(signature_name));
    for (size_t i = 0; i < signature.OutputNames().size(); ++i) {
      if (signature.OutputNames()[i] == name) {
        auto handle_or = output_buffers[i].GetWebGpuBuffer();
        if (handle_or.HasValue()) {
          return reinterpret_cast<uintptr_t>(*handle_or);
        }
        return absl::NotFoundError("Buffer is not a WebGPU buffer");
      }
    }
    return absl::NotFoundError("Output tensor not found");
  }

  absl::StatusOr<uintptr_t> GetInputWebGpuBuffer(
      const std::string& signature_name, const std::string& name) {
    auto in_it = signature_input_buffers_.find(signature_name);
    if (in_it == signature_input_buffers_.end()) {
      return absl::NotFoundError("Signature not found");
    }
    auto& input_buffers = in_it->second;

    LITERT_ASSIGN_OR_RETURN(auto signature,
                            compiled_model_.FindSignature(signature_name));
    for (size_t i = 0; i < signature.InputNames().size(); ++i) {
      if (signature.InputNames()[i] == name) {
        auto handle_or = input_buffers[i].GetWebGpuBuffer();
        if (handle_or.HasValue()) {
          return reinterpret_cast<uintptr_t>(*handle_or);
        }
        return absl::NotFoundError("Buffer is not a WebGPU buffer");
      }
    }
    return absl::NotFoundError("Input tensor not found");
  }

 private:
  LitertDynamicRunner() = default;
  CompiledModel compiled_model_;
  std::string default_signature_name_;
  absl::flat_hash_map<std::string, std::vector<TensorBuffer>>
      signature_input_buffers_;
  absl::flat_hash_map<std::string, std::vector<TensorBuffer>>
      signature_output_buffers_;
};

}  // namespace tensor
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LITERT_DYNAMIC_RUNNER_H_
