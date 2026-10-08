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

#include <algorithm>
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
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/litert/feedback_loop_config.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/tensor.h"

namespace litert {
namespace tensor {

class LitertDynamicRunner {
 public:
  static absl::StatusOr<LitertDynamicRunner> Create(
      Environment& env, const std::string& model_path, Options& options) {
    LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                            CompiledModel::Create(env, model_path, options));
    LitertDynamicRunner runner(std::move(compiled_model));
    LITERT_RETURN_IF_ERROR(runner.InitializeBuffers());
    return runner;
  }

  static absl::StatusOr<LitertDynamicRunner> Create(
      Environment& env, absl::Span<const uint8_t> model_buffer,
      Options& options) {
    BufferRef<uint8_t> buf_ref(model_buffer.data(), model_buffer.size());
    LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                            CompiledModel::Create(env, buf_ref, options));
    LitertDynamicRunner runner(std::move(compiled_model));
    LITERT_RETURN_IF_ERROR(runner.InitializeBuffers());
    return runner;
  }

  static absl::StatusOr<LitertDynamicRunner> Create(
      Environment& env, const std::string& model_path, Options& options,
      const std::vector<FeedbackLoopConfig>& feedback_loops) {
    auto gpu_options_or = options.GetGpuOptions();
    if (gpu_options_or.HasValue()) {
      LITERT_RETURN_IF_ERROR(gpu_options_or->EnableExternalTensorsMode(true));
      for (const auto& loop : feedback_loops) {
        LITERT_RETURN_IF_ERROR(
            gpu_options_or->AddExternalTensorPattern(loop.input_name.c_str()));
        LITERT_RETURN_IF_ERROR(
            gpu_options_or->AddExternalTensorPattern(loop.output_name.c_str()));
      }
    }

    LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                            CompiledModel::Create(env, model_path, options));
    LitertDynamicRunner runner(std::move(compiled_model));
    LITERT_RETURN_IF_ERROR(runner.InitializeBuffers());

    for (const auto& loop : feedback_loops) {
      LITERT_RETURN_IF_ERROR(
          runner.RegisterFeedbackLoop(loop.input_name, loop.output_name));
    }

    return runner;
  }

  static absl::StatusOr<LitertDynamicRunner> Create(
      Environment& env, absl::Span<const uint8_t> model_buffer,
      Options& options, const std::vector<FeedbackLoopConfig>& feedback_loops) {
    auto gpu_options_or = options.GetGpuOptions();
    if (gpu_options_or.HasValue()) {
      LITERT_RETURN_IF_ERROR(gpu_options_or->EnableExternalTensorsMode(true));
      for (const auto& loop : feedback_loops) {
        LITERT_RETURN_IF_ERROR(
            gpu_options_or->AddExternalTensorPattern(loop.input_name.c_str()));
        LITERT_RETURN_IF_ERROR(
            gpu_options_or->AddExternalTensorPattern(loop.output_name.c_str()));
      }
    }

    BufferRef<uint8_t> buf_ref(model_buffer.data(), model_buffer.size());
    LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                            CompiledModel::Create(env, buf_ref, options));
    LitertDynamicRunner runner(std::move(compiled_model));
    LITERT_RETURN_IF_ERROR(runner.InitializeBuffers());

    for (const auto& loop : feedback_loops) {
      LITERT_RETURN_IF_ERROR(
          runner.RegisterFeedbackLoop(loop.input_name, loop.output_name));
    }

    return runner;
  }

  // Helper to initialize buffers for all signatures
  absl::Status InitializeBuffers() {
    LITERT_ASSIGN_OR_RETURN(auto keys, compiled_model_.GetSignatureKeys());
    if (keys.empty()) return absl::InternalError("No signatures found");
    default_signature_name_ = std::string(keys[0]);

    for (const auto& key : keys) {
      std::string key_str(key);
      SignatureState state;
      LITERT_ASSIGN_OR_RETURN(state.index,
                              compiled_model_.GetSignatureIndex(key_str));
      LITERT_ASSIGN_OR_RETURN(state.input_buffers,
                              compiled_model_.CreateInputBuffers(key_str));
      LITERT_ASSIGN_OR_RETURN(state.output_buffers,
                              compiled_model_.CreateOutputBuffers(key_str));
      LITERT_ASSIGN_OR_RETURN(auto input_names,
                              compiled_model_.GetSignatureInputNames(key_str));
      LITERT_ASSIGN_OR_RETURN(auto output_names,
                              compiled_model_.GetSignatureOutputNames(key_str));
      state.input_names.reserve(input_names.size());
      for (const auto& name : input_names) {
        state.input_indices.emplace(std::string(name),
                                    state.input_names.size());
        state.input_names.emplace_back(name);
      }
      state.output_names.reserve(output_names.size());
      for (const auto& name : output_names) {
        state.output_indices.emplace(std::string(name),
                                     state.output_names.size());
        state.output_names.emplace_back(name);
      }
      state.resized_inputs.assign(state.input_buffers.size(), false);
      signatures_.emplace(std::move(key_str), std::move(state));
    }
    return absl::OkStatus();
  }

  // Resizes one input without allocating intermediate shapes. Bind all resized
  // inputs before Run(), which refreshes output shapes once for the signature.
  // Non-strict mode also permits reshaping models without shape signatures.
  // Feedback loops require fixed shapes and cannot be resized.
  absl::Status ResizeInput(const std::string& signature_name, size_t index,
                           absl::Span<const int> dimensions,
                           bool strict = true) {
    LITERT_ASSIGN_OR_RETURN(SignatureState * state,
                            FindSignatureState(signature_name));
    if (index >= state->input_buffers.size()) {
      return absl::NotFoundError("Index out of bounds");
    }
    LITERT_ASSIGN_OR_RETURN(
        auto layout,
        compiled_model_.GetInputTensorLayout(state->index, index));
    if (std::equal(dimensions.begin(), dimensions.end(),
                   layout.Dimensions().begin(), layout.Dimensions().end())) {
      return absl::OkStatus();
    }
    if (!state->feedback_loops.empty()) {
      return absl::FailedPreconditionError(
          "Resizing a signature with feedback loops is not supported");
    }
    if (strict) {
      LITERT_RETURN_IF_ERROR(
          compiled_model_.ResizeInputTensor(state->index, index, dimensions));
    } else {
      LITERT_RETURN_IF_ERROR(compiled_model_.ResizeInputTensorNonStrict(
          state->index, index, dimensions));
    }
    state->resized_inputs[index] = true;
    state->resized_outputs = true;
    return absl::OkStatus();
  }

  absl::Status ResizeInput(const std::string& signature_name,
                           const std::string& name,
                           absl::Span<const int> dimensions,
                           bool strict = true) {
    LITERT_ASSIGN_OR_RETURN(auto index, GetInputIndex(signature_name, name));
    return ResizeInput(signature_name, index, dimensions, strict);
  }

  absl::Status ResizeInput(const std::string& name,
                           absl::Span<const int> dimensions,
                           bool strict = true) {
    return ResizeInput(default_signature_name_, name, dimensions, strict);
  }

  // Retains a TensorBuffer without copying its contents. Its type and layout
  // must match the current input. For borrowed host memory, the caller retains
  // ownership and must keep the allocation alive through the last invocation
  // using it, including the alignment and padding required by the backend.
  absl::Status SetInputBuffer(const std::string& signature_name, size_t index,
                              TensorBuffer buffer) {
    LITERT_ASSIGN_OR_RETURN(SignatureState * state,
                            FindSignatureState(signature_name));
    if (index >= state->input_buffers.size()) {
      return absl::NotFoundError("Input index or signature not found");
    }
    LITERT_ASSIGN_OR_RETURN(
        auto type, compiled_model_.GetInputTensorType(state->index, index));
    LITERT_ASSIGN_OR_RETURN(
        auto layout,
        compiled_model_.GetInputTensorLayout(state->index, index));
    LITERT_ASSIGN_OR_RETURN(auto actual, buffer.TensorType());
    if (actual != RankedTensorType(type.ElementType(), std::move(layout))) {
      return absl::InvalidArgumentError("Input buffer type or shape mismatch");
    }
    state->input_buffers[index] = std::move(buffer);
    state->resized_inputs[index] = false;
    return absl::OkStatus();
  }

  absl::Status SetInputBuffer(const std::string& signature_name,
                              const std::string& name, TensorBuffer buffer) {
    LITERT_ASSIGN_OR_RETURN(auto index, GetInputIndex(signature_name, name));
    return SetInputBuffer(signature_name, index, std::move(buffer));
  }

  absl::Status SetInputBuffer(const std::string& name, TensorBuffer buffer) {
    return SetInputBuffer(default_signature_name_, name, std::move(buffer));
  }

  absl::Status RegisterFeedbackLoop(const std::string& input_name,
                                    const std::string& output_name) {
    return RegisterFeedbackLoop(default_signature_name_, input_name,
                                output_name);
  }

  absl::Status RegisterFeedbackLoop(const std::string& signature_name,
                                    const std::string& input_name,
                                    const std::string& output_name) {
    LITERT_ASSIGN_OR_RETURN(SignatureState * state,
                            FindSignatureState(signature_name));
    LITERT_ASSIGN_OR_RETURN(size_t input_idx,
                            GetInputIndex(*state, input_name));
    LITERT_ASSIGN_OR_RETURN(size_t output_idx,
                            GetOutputIndex(*state, output_name));

    FeedbackLoop loop;
    loop.input_index = input_idx;
    loop.output_index = output_idx;
    state->feedback_loops.push_back(loop);
    state->first_run = true;
    state->swapped = false;
    return absl::OkStatus();
  }

  absl::Status Reset() { return Reset(default_signature_name_); }

  absl::Status Reset(const std::string& signature_name) {
    LITERT_ASSIGN_OR_RETURN(SignatureState * state,
                            FindSignatureState(signature_name));
    state->first_run = true;
    if (state->swapped) {
      for (const auto& loop : state->feedback_loops) {
        std::swap(state->input_buffers[loop.input_index],
                  state->output_buffers[loop.output_index]);
      }
      state->swapped = false;
    }
    return absl::OkStatus();
  }

  // Query input buffer index by name in a signature once at startup
  absl::StatusOr<size_t> GetInputIndex(const std::string& signature_name,
                                       const std::string& name) const {
    LITERT_ASSIGN_OR_RETURN(const SignatureState* state,
                            FindSignatureState(signature_name));
    return GetInputIndex(*state, name);
  }

  // Query output buffer index by name in a signature once at startup
  absl::StatusOr<size_t> GetOutputIndex(const std::string& signature_name,
                                        const std::string& name) const {
    LITERT_ASSIGN_OR_RETURN(const SignatureState* state,
                            FindSignatureState(signature_name));
    return GetOutputIndex(*state, name);
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
    LITERT_ASSIGN_OR_RETURN(auto index, GetInputIndex(signature_name, name));
    return SetInput(signature_name, index, tensor);
  }

  // Set input by signature and index
  absl::Status SetInput(const std::string& signature_name, size_t index,
                        const TensorHandle& tensor) {
    LITERT_ASSIGN_OR_RETURN(SignatureState * state,
                            FindSignatureState(signature_name));
    if (index >= state->input_buffers.size())
      return absl::NotFoundError("Index out of bounds");

    LITERT_ASSIGN_OR_RETURN(Buffer & buffer, tensor.GetBuffer());
    auto litert_buffer_or = buffer.As<LitertBuffer>();
    if (litert_buffer_or.ok()) {
      LITERT_ASSIGN_OR_RETURN(auto duplicate,
                              litert_buffer_or->tensor_buffer().Duplicate());
      return SetInputBuffer(signature_name, index, std::move(duplicate));
    } else {
      LITERT_RETURN_IF_ERROR(EnsureInputBuffer(signature_name, *state, index));
      auto locked_span = buffer.Lock().As<const uint8_t>();
      LITERT_RETURN_IF_ERROR(state->input_buffers[index].Write(
          absl::Span<const uint8_t>(locked_span)));
    }
    return absl::OkStatus();
  }

  // Set input by signature and name with binary data
  absl::Status SetInput(const std::string& signature_name,
                        const std::string& name,
                        absl::Span<const uint8_t> data) {
    LITERT_ASSIGN_OR_RETURN(auto index, GetInputIndex(signature_name, name));
    return SetInput(signature_name, index, data);
  }

  // Set input by signature and index with binary data
  absl::Status SetInput(const std::string& signature_name, size_t index,
                        absl::Span<const uint8_t> data) {
    LITERT_ASSIGN_OR_RETURN(SignatureState * state,
                            FindSignatureState(signature_name));
    if (index >= state->input_buffers.size())
      return absl::NotFoundError("Index out of bounds");
    LITERT_RETURN_IF_ERROR(EnsureInputBuffer(signature_name, *state, index));
    auto res = state->input_buffers[index].Write(data);
    if (!res.HasValue()) {
      return absl::InternalError("Failed to write input buffer");
    }
    return absl::OkStatus();
  }

  // Set output buffer by signature and index
  absl::Status SetOutputBuffer(const std::string& signature_name, size_t index,
                               litert::TensorBuffer buffer) {
    LITERT_ASSIGN_OR_RETURN(SignatureState * state,
                            FindSignatureState(signature_name));
    if (index >= state->output_buffers.size()) {
      return absl::NotFoundError("Index out of bounds");
    }
    LITERT_ASSIGN_OR_RETURN(
        auto type, compiled_model_.GetOutputTensorType(state->index, index));
    LITERT_ASSIGN_OR_RETURN(
        auto layouts, compiled_model_.GetOutputTensorLayouts(state->index));
    LITERT_ASSIGN_OR_RETURN(auto actual, buffer.TensorType());
    if (actual !=
        RankedTensorType(type.ElementType(), std::move(layouts[index]))) {
      return absl::InvalidArgumentError("Output buffer type or shape mismatch");
    }
    state->output_buffers[index] = std::move(buffer);
    return absl::OkStatus();
  }

  // Set output buffer by signature and name
  absl::Status SetOutputBuffer(const std::string& signature_name,
                               const std::string& name,
                               litert::TensorBuffer buffer) {
    LITERT_ASSIGN_OR_RETURN(auto index, GetOutputIndex(signature_name, name));
    return SetOutputBuffer(signature_name, index, std::move(buffer));
  }

  absl::Status SetOutputBuffer(const std::string& name,
                               litert::TensorBuffer buffer) {
    return SetOutputBuffer(default_signature_name_, name, std::move(buffer));
  }

  absl::Status SetOutputBuffer(size_t index, litert::TensorBuffer buffer) {
    return SetOutputBuffer(default_signature_name_, index, std::move(buffer));
  }

  absl::Status SetOutput(const std::string& name, const TensorHandle& tensor) {
    LITERT_ASSIGN_OR_RETURN(Buffer & buffer, tensor.GetBuffer());
    auto litert_buffer_or = buffer.As<LitertBuffer>();
    if (!litert_buffer_or.ok()) {
      return absl::InvalidArgumentError("Tensor must be a LitertBuffer");
    }
    LITERT_ASSIGN_OR_RETURN(auto dup,
                            litert_buffer_or->tensor_buffer().Duplicate());
    return SetOutputBuffer(name, std::move(dup));
  }

  // Run by signature
  absl::Status Run(const std::string& signature_name) {
    LITERT_ASSIGN_OR_RETURN(SignatureState * state,
                            FindSignatureState(signature_name));

    if (std::find(state->resized_inputs.begin(), state->resized_inputs.end(),
                  true) != state->resized_inputs.end()) {
      return absl::FailedPreconditionError("Bind resized inputs before Run");
    }
    LITERT_RETURN_IF_ERROR(RefreshOutputBuffers(signature_name, *state));

    if (!state->first_run && !state->feedback_loops.empty()) {
      for (const auto& loop : state->feedback_loops) {
        std::swap(state->input_buffers[loop.input_index],
                  state->output_buffers[loop.output_index]);
      }
      state->swapped = !state->swapped;
    }
    state->first_run = false;

    auto status = compiled_model_.Run(state->index, state->input_buffers,
                                      state->output_buffers);
    if (!status.HasValue()) {
      return absl::InternalError(status.Error().Message());
    }
    return absl::OkStatus();
  }

  // Get output by signature and name
  absl::StatusOr<TensorHandle> GetOutput(const std::string& signature_name,
                                         const std::string& name) {
    LITERT_ASSIGN_OR_RETURN(auto index, GetOutputIndex(signature_name, name));
    return GetOutput(signature_name, index);
  }

  // Get output by signature and index
  absl::StatusOr<TensorHandle> GetOutput(const std::string& signature_name,
                                         size_t index) {
    LITERT_ASSIGN_OR_RETURN(SignatureState * state,
                            FindSignatureState(signature_name));
    if (index >= state->output_buffers.size())
      return absl::NotFoundError("Index out of bounds");

    const auto& name = state->output_names[index];

    LITERT_RETURN_IF_ERROR(RefreshOutputBuffers(signature_name, *state));
    LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type,
                            state->output_buffers[index].TensorType());

    auto dup_or = state->output_buffers[index].Duplicate();
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
    LITERT_ASSIGN_OR_RETURN(auto index, GetInputIndex(signature_name, name));
    return GetInput(signature_name, index);
  }

  // Get input by signature and index
  absl::StatusOr<TensorHandle> GetInput(const std::string& signature_name,
                                        size_t index) {
    LITERT_ASSIGN_OR_RETURN(SignatureState * state,
                            FindSignatureState(signature_name));
    if (index >= state->input_buffers.size())
      return absl::NotFoundError("Index out of bounds");

    const auto& name = state->input_names[index];

    LITERT_RETURN_IF_ERROR(EnsureInputBuffer(signature_name, *state, index));
    LITERT_ASSIGN_OR_RETURN(auto ranked_tensor_type,
                            state->input_buffers[index].TensorType());

    auto dup_or = state->input_buffers[index].Duplicate();
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
    LITERT_ASSIGN_OR_RETURN(const SignatureState* state,
                            FindSignatureState(signature_name));
    LITERT_ASSIGN_OR_RETURN(size_t index, GetOutputIndex(*state, name));
    auto handle_or = state->output_buffers[index].GetWebGpuBuffer();
    if (handle_or.HasValue()) {
      return reinterpret_cast<uintptr_t>(*handle_or);
    }
    return absl::NotFoundError("Buffer is not a WebGPU buffer");
  }

  absl::StatusOr<uintptr_t> GetInputWebGpuBuffer(
      const std::string& signature_name, const std::string& name) {
    LITERT_ASSIGN_OR_RETURN(const SignatureState* state,
                            FindSignatureState(signature_name));
    LITERT_ASSIGN_OR_RETURN(size_t index, GetInputIndex(*state, name));
    auto handle_or = state->input_buffers[index].GetWebGpuBuffer();
    if (handle_or.HasValue()) {
      return reinterpret_cast<uintptr_t>(*handle_or);
    }
    return absl::NotFoundError("Buffer is not a WebGPU buffer");
  }

 private:
  struct FeedbackLoop {
    size_t input_index;
    size_t output_index;
  };

  // Cache names and per-signature runtime state; shape/type queries continue to
  // use current buffers.
  struct SignatureState {
    size_t index = 0;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    absl::flat_hash_map<std::string, size_t> input_indices;
    absl::flat_hash_map<std::string, size_t> output_indices;
    std::vector<TensorBuffer> input_buffers;
    std::vector<TensorBuffer> output_buffers;
    std::vector<FeedbackLoop> feedback_loops;
    std::vector<bool> resized_inputs;
    bool resized_outputs = false;
    bool first_run = true;
    bool swapped = false;
  };

  absl::StatusOr<SignatureState*> FindSignatureState(
      const std::string& signature_name) {
    auto it = signatures_.find(signature_name);
    if (it == signatures_.end()) {
      return absl::NotFoundError("Signature not found");
    }
    return &it->second;
  }

  absl::StatusOr<const SignatureState*> FindSignatureState(
      const std::string& signature_name) const {
    auto it = signatures_.find(signature_name);
    if (it == signatures_.end()) {
      return absl::NotFoundError("Signature not found");
    }
    return &it->second;
  }

  static absl::StatusOr<size_t> GetInputIndex(const SignatureState& state,
                                              const std::string& name) {
    auto found = state.input_indices.find(name);
    if (found == state.input_indices.end()) {
      return absl::NotFoundError("Input tensor name not found in signature");
    }
    return found->second;
  }

  static absl::StatusOr<size_t> GetOutputIndex(const SignatureState& state,
                                               const std::string& name) {
    auto found = state.output_indices.find(name);
    if (found == state.output_indices.end()) {
      return absl::NotFoundError("Output tensor name not found in signature");
    }
    return found->second;
  }

  absl::Status EnsureInputBuffer(const std::string& signature_name,
                                 SignatureState& state, size_t index) {
    if (state.resized_inputs[index]) {
      LITERT_ASSIGN_OR_RETURN(
          auto buffer, compiled_model_.CreateInputBuffer(
                           signature_name, state.input_names[index]));
      state.input_buffers[index] = std::move(buffer);
      state.resized_inputs[index] = false;
    }
    return absl::OkStatus();
  }

  absl::Status RefreshOutputBuffers(const std::string& signature_name,
                                    SignatureState& state) {
    if (!state.resized_outputs) return absl::OkStatus();
    LITERT_ASSIGN_OR_RETURN(
        auto layouts, compiled_model_.GetOutputTensorLayouts(state.index));
    for (size_t i = 0; i < state.output_buffers.size(); ++i) {
      LITERT_ASSIGN_OR_RETURN(auto type, state.output_buffers[i].TensorType());
      if (type.Layout() == layouts[i]) continue;
      LITERT_ASSIGN_OR_RETURN(
          state.output_buffers[i],
          compiled_model_.CreateOutputBuffer(signature_name,
                                             state.output_names[i]));
    }
    state.resized_outputs = false;
    return absl::OkStatus();
  }

  explicit LitertDynamicRunner(CompiledModel compiled_model)
      : compiled_model_(std::move(compiled_model)) {}
  CompiledModel compiled_model_;
  std::string default_signature_name_;
  absl::flat_hash_map<std::string, SignatureState> signatures_;
};

}  // namespace tensor
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_RUNNERS_LITERT_LITERT_DYNAMIC_RUNNER_H_
