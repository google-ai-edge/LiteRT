// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CC_LITERT_COMPILED_MODEL_H_
#define ODML_LITERT_LITERT_CC_LITERT_COMPILED_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_profiler.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"

namespace mediapipe {
class InferenceRunnerLiteRt;
}  // namespace mediapipe

/// @file
/// @brief Defines the C++ wrapper for a compiled LiteRT model, providing a
/// high-level inference API.

namespace mediapipe {
class InferenceRunnerLiteRt;
}  // namespace mediapipe

namespace litert {

namespace benchmark {
class BenchmarkLiteRtModel;
}  // namespace benchmark

namespace compiled_model_wrapper {
class CompiledModelWrapper;
}  // namespace compiled_model_wrapper

namespace lm {
class EmbeddingLookupText;
class EndOfMultiModalEmbedding;
class FrontendModelWrapper;
class AudioLiteRtCompiledModelExecutor;
class LlmLiteRtNpuCompiledModelExecutor;
class VisionLiteRtCompiledModelExecutor;
class LlmLiteRtCompiledModelExecutorDynamic;
class LlmLiteRtCompiledModelExecutorStatic;
class LlmLiteRtCompiledModelExecutorBase;
}  // namespace lm

/// @brief A high-level inference API for LiteRT.
///
/// The `CompiledModel` is created by providing a model with compilation
/// options. Internally, it instantiates a runtime and applies delegates mapped
/// to the compilation options. It supports querying buffer requirements to
/// create input/output `TensorBuffer`s and invoking the model with them.
///
/// @par Example User Flow:
/// 1. Create a `CompiledModel`.
/// 2. Query the model's input/output requirements.
/// 3. Create input/output `TensorBuffer`s.
/// 4. Fill the input `TensorBuffer`s with input data.
/// 5. Invoke the model with the input/output `TensorBuffer`s.
/// 6. Evaluate the output `TensorBuffer`s.
class CompiledModel
    : public internal::Handle<LiteRtCompiledModel, LiteRtDestroyCompiledModel> {
 public:
  friend class ::mediapipe::InferenceRunnerLiteRt;
  friend class benchmark::BenchmarkLiteRtModel;
  friend class compiled_model_wrapper::CompiledModelWrapper;
  friend class lm::AudioLiteRtCompiledModelExecutor;
  friend class lm::EmbeddingLookupText;
  friend class lm::EndOfMultiModalEmbedding;
  friend class lm::FrontendModelWrapper;
  friend class lm::LlmLiteRtCompiledModelExecutorBase;
  friend class lm::LlmLiteRtCompiledModelExecutorDynamic;
  friend class lm::LlmLiteRtCompiledModelExecutorStatic;
  friend class lm::LlmLiteRtNpuCompiledModelExecutor;
  friend class lm::VisionLiteRtCompiledModelExecutor;

  CompiledModel() = default;

  /// @brief Creates a `CompiledModel` from a TFLite file.
  ///
  /// The model is loaded into memory, and the caller takes ownership of the
  /// returned `CompiledModel` object. The caller should keep the model alive
  /// until the `CompiledModel` is destroyed. The provided `compilation_options`
  /// are used for model compilation, and `hardware_accelerators` is used to
  /// select the accelerator, regardless of whether the model is AOT or JIT
  /// compiled.
  ///
  /// @note The provided environment must outlive the compiled model and any
  /// executions running on it.
  /// @note Even if the model is fully AOT-compiled for an NPU, you must
  /// specify the NPU accelerator in `hardware_accelerators` to use it
  /// properly.
  static Expected<CompiledModel> Create(litert::Environment& env,
                                        const std::string& model_filename,
                                        Options& compilation_options) {
    LITERT_RETURN_IF_ERROR(compilation_options.Build());
    LiteRtModel litert_model;
    if (auto status =
            LiteRtCreateModelFromFile(model_filename.c_str(), &litert_model);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to load model from file");
    }
    LiteRtCompiledModel compiled_model;
    if (auto res = LiteRtCreateCompiledModel(env.Get(), litert_model,
                                             compilation_options.Get(),
                                             &compiled_model);
        res != kLiteRtStatusOk) {
      LiteRtDestroyModel(litert_model);
      return Unexpected(res, "Failed to compile model");
    }
    return CompiledModel(litert_model, /*model_owned=*/OwnHandle::kYes,
                         compiled_model,
                         /*owned=*/OwnHandle::kYes);
  }

  /// @brief An overload of `Create` that takes a buffer reference to the model
  /// instead of a filename.
  static Expected<CompiledModel> Create(litert::Environment& env,
                                        BufferRef<uint8_t> model_buffer,
                                        Options& compilation_options) {
    LITERT_RETURN_IF_ERROR(compilation_options.Build());
    LiteRtModel litert_model;
    if (auto status = LiteRtCreateModelFromBuffer(
            model_buffer.Data(), model_buffer.Size(), &litert_model);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to load model from buffer");
    }
    LiteRtCompiledModel compiled_model;
    if (auto res = LiteRtCreateCompiledModel(env.Get(), litert_model,
                                             compilation_options.Get(),
                                             &compiled_model);
        res != kLiteRtStatusOk) {
      LiteRtDestroyModel(litert_model);
      return Unexpected(res, "Failed to compile model");
    }
    return CompiledModel(litert_model, /*model_owned=*/OwnHandle::kYes,
                         compiled_model,
                         /*owned=*/OwnHandle::kYes);
  }

  /// @brief A simplified version of `Create` that uses default compilation
  /// options.
  ///
  /// The provided hardware accelerator is used to select the target
  /// accelerator.
  /// @note This should be specified for both JIT and AOT compiled models.
  static Expected<CompiledModel> Create(
      litert::Environment& env, const std::string& model_filename,
      litert::HwAccelerators hardware_accelerators) {
    LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
    compilation_options.SetHardwareAccelerators(hardware_accelerators);
    return Create(env, model_filename, compilation_options);
  }

  /// @brief An overload of `Create` that takes a buffer reference to the model
  /// instead of a filename.
  static Expected<CompiledModel> Create(
      litert::Environment& env, BufferRef<uint8_t> model_buffer,
      litert::HwAccelerators hardware_accelerators) {
    LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
    compilation_options.SetHardwareAccelerators(
        static_cast<HwAccelerators>(hardware_accelerators));
    return Create(env, model_buffer, compilation_options);
  }

  /// @brief Gets input buffer requirements for the given signature and input
  /// name.
  Expected<TensorBufferRequirements> GetInputBufferRequirements(
      absl::string_view signature_name, absl::string_view input_name) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return GetInputBufferRequirements(signature_index, input_name);
  }

  /// @brief Returns the buffer requirements for the n-th input tensor.
  ///
  /// The returned `TensorBufferRequirements` is used to create the input
  /// tensor buffer.
  Expected<TensorBufferRequirements> GetInputBufferRequirements(
      size_t signature_index, size_t input_index) const {
    LiteRtTensorBufferRequirements buffer_requirements;
    LITERT_RETURN_IF_ERROR(LiteRtGetCompiledModelInputBufferRequirements(
        Get(), signature_index, input_index, &buffer_requirements));
    return TensorBufferRequirements::WrapCObject(buffer_requirements,
                                                 OwnHandle::kNo);
  }

  /// @brief An overload of `GetInputBufferRequirements` that takes an input
  /// tensor name.
  Expected<TensorBufferRequirements> GetInputBufferRequirements(
      size_t signature_index, absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(size_t input_index,
                            FindInputIndex(signature_index, input_name));
    return GetInputBufferRequirements(signature_index, input_index);
  }

  /// @brief Gets input buffer requirements of the default signature for the
  /// n-th input tensor.
  Expected<TensorBufferRequirements> GetInputBufferRequirements(
      size_t input_index) const {
    return GetInputBufferRequirements(/*signature_index=*/0, input_index);
  }

  /// @brief Gets input buffer requirements of the default signature for a given
  /// input name.
  Expected<TensorBufferRequirements> GetInputBufferRequirements(
      absl::string_view input_name) const {
    return GetInputBufferRequirements(/*signature_index=*/0, input_name);
  }

  /// @brief Gets output buffer requirements for the given signature and output
  /// name.
  Expected<TensorBufferRequirements> GetOutputBufferRequirements(
      absl::string_view signature_name, absl::string_view output_name) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return GetOutputBufferRequirements(signature_index, output_name);
  }

  /// @brief Returns the layouts of all output tensors for a given signature
  /// index.
  ///
  /// If `update_allocation` is `true`, the tensor allocations will be updated
  /// to the current state of the compiled model.
  Expected<std::vector<Layout>> GetOutputTensorLayouts(
      size_t signature_index, bool update_allocation = false) const {
    // get num tensors here
    LITERT_ASSIGN_OR_RETURN(auto output_names,
                            model_.GetSignatureOutputNames(signature_index));
    int num_tensors = output_names.size();
    std::vector<LiteRtLayout> litert_layout_vector(num_tensors);
    LITERT_RETURN_IF_ERROR(LiteRtGetCompiledModelOutputTensorLayouts(
        Get(), signature_index, num_tensors, litert_layout_vector.data(),
        update_allocation));

    // apply Layout() to each element within the litert_layout_vector
    std::vector<Layout> layout_vector;
    layout_vector.reserve(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
      layout_vector.push_back(Layout(litert_layout_vector[i]));
    }

    return layout_vector;
  }

  /// @brief Returns the buffer requirements for the given output tensor.
  ///
  /// The returned `TensorBufferRequirements` is used to create the output
  /// tensor buffer.
  Expected<TensorBufferRequirements> GetOutputBufferRequirements(
      size_t signature_index, size_t output_index) const {
    LiteRtTensorBufferRequirements buffer_requirements;
    LITERT_RETURN_IF_ERROR(LiteRtGetCompiledModelOutputBufferRequirements(
        Get(), signature_index, output_index, &buffer_requirements));
    return TensorBufferRequirements::WrapCObject(buffer_requirements,
                                                 OwnHandle::kNo);
  }

  /// @brief An overload of `GetOutputBufferRequirements` that takes an output
  /// tensor name.
  Expected<TensorBufferRequirements> GetOutputBufferRequirements(
      size_t signature_index, absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(size_t output_index,
                            FindOutputIndex(signature_index, output_name));
    return GetOutputBufferRequirements(signature_index, output_index);
  }

  /// @brief Gets input buffer requirements of the default signature for the
  /// n-th input tensor.
  Expected<TensorBufferRequirements> GetOutputBufferRequirements(
      size_t output_index) const {
    return GetOutputBufferRequirements(/*signature_index=*/0, output_index);
  }

  /// @brief Gets input buffer requirements of the default signature for a given
  /// input name.
  Expected<TensorBufferRequirements> GetOutputBufferRequirements(
      absl::string_view output_name) const {
    return GetOutputBufferRequirements(/*signature_index=*/0, output_name);
  }

  /// @brief Creates an input tensor buffer for the given signature and input
  /// name.
  Expected<TensorBuffer> CreateInputBuffer(absl::string_view signature_name,
                                           absl::string_view input_name) const {
    return CreateInputOutputBuffer(signature_name, input_name,
                                   /*is_input=*/true);
  }

  /// @brief Creates an input tensor buffer for the default signature and a
  /// given input name.
  Expected<TensorBuffer> CreateInputBuffer(absl::string_view input_name) const {
    return CreateInputOutputBuffer(/*signature_index=*/0, input_name,
                                   /*is_input=*/true);
  }

  /// @brief Creates an output tensor buffer for the given signature and output
  /// name.
  Expected<TensorBuffer> CreateOutputBuffer(
      absl::string_view signature_name, absl::string_view output_name) const {
    return CreateInputOutputBuffer(signature_name, output_name,
                                   /*is_input=*/false);
  }

  /// @brief Creates an output tensor buffer for the default signature and a
  /// given output name.
  Expected<TensorBuffer> CreateOutputBuffer(
      absl::string_view output_name) const {
    return CreateInputOutputBuffer(/*signature_index=*/0, output_name,
                                   /*is_input=*/false);
  }

  /// @brief A helper function to create input tensor buffers for a given
  /// signature.
  ///
  /// It uses `BufferRequirements` and `RankedTensorType` to create the input
  /// tensor buffers.
  Expected<std::vector<TensorBuffer>> CreateInputBuffers(
      absl::string_view signature_name) const {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return CreateInputOutputBuffers(signature_index, /*is_input=*/true);
  }

  /// @brief A helper function to create input tensor buffers for a given
  /// signature.
  ///
  /// It uses `BufferRequirements` and `RankedTensorType` to create the input
  /// tensor buffers.
  Expected<std::vector<TensorBuffer>> CreateInputBuffers(
      size_t signature_index) const {
    return CreateInputOutputBuffers(signature_index, /*is_input=*/true);
  }

  /// @brief A helper function to create input tensor buffers for the default
  /// signature.
  ///
  /// It uses `BufferRequirements` and `RankedTensorType` to create the input
  /// tensor buffers.
  Expected<std::vector<TensorBuffer>> CreateInputBuffers() const {
    return CreateInputOutputBuffers(/*signature_index=*/0, /*is_input=*/true);
  }

  /// @brief A helper function to create output tensor buffers for a given
  /// signature.
  ///
  /// It uses `BufferRequirements` and `RankedTensorType` to create the output
  /// tensor buffers.
  Expected<std::vector<TensorBuffer>> CreateOutputBuffers(
      absl::string_view signature_name) const {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return CreateOutputBuffers(signature_index);
  }

  /// @brief A helper function to create output tensor buffers for a given
  /// signature.
  ///
  /// It uses `BufferRequirements` and `RankedTensorType` to create the output
  /// tensor buffers.
  Expected<std::vector<TensorBuffer>> CreateOutputBuffers(
      size_t signature_index) const {
    return CreateInputOutputBuffers(signature_index, /*is_input=*/false);
  }

  /// @brief A helper function to create output tensor buffers for the default
  /// signature.
  ///
  /// It uses `BufferRequirements` and `RankedTensorType` to create the output
  /// tensor buffers.
  Expected<std::vector<TensorBuffer>> CreateOutputBuffers() const {
    return CreateInputOutputBuffers(/*signature_index=*/0, /*is_input=*/false);
  }

  /// @brief Runs the model for a given signature index synchronously with the
  /// provided input/output `TensorBuffer`s.
  Expected<void> Run(size_t signature_index,
                     absl::Span<const TensorBuffer> input_buffers,
                     absl::Span<const TensorBuffer> output_buffers) const {
    bool async = false;
    return RunHelper(signature_index, input_buffers, output_buffers, async);
  }

  /// @brief Runs the model for the default signature synchronously with the
  /// provided input/output `TensorBuffer`s.
  Expected<void> Run(absl::Span<const TensorBuffer> input_buffers,
                     absl::Span<const TensorBuffer> output_buffers) const {
    bool async = false;
    return RunHelper(/*signature_index=*/0, input_buffers, output_buffers,
                     async);
  }

  /// @brief Runs the model for a given signature index asynchronously, if
  /// possible, with the provided input/output `TensorBuffer`s.
  ///
  /// If asynchronous execution is possible, `async` will be set to `true`;
  /// otherwise, the function runs the model synchronously.
  Expected<void> RunAsync(size_t signature_index,
                          const std::vector<TensorBuffer>& input_buffers,
                          const std::vector<TensorBuffer>& output_buffers,
                          bool& async) const {
    async = true;
    return RunHelper(signature_index, input_buffers, output_buffers, async);
  }

  /// @brief Runs the model for the default signature asynchronously, if
  /// possible, with the provided input/output `TensorBuffer`s.
  ///
  /// If asynchronous execution is possible, `async` will be set to `true`;
  /// otherwise, the function runs the model synchronously.
  Expected<void> RunAsync(const std::vector<TensorBuffer>& input_buffers,
                          const std::vector<TensorBuffer>& output_buffers,
                          bool& async) const {
    async = true;
    return RunHelper(/*signature_index=*/0, input_buffers, output_buffers,
                     async);
  }

  /// @brief Runs the model for a given signature key synchronously with the
  /// provided input/output `TensorBuffer`s.
  Expected<void> Run(absl::string_view signature_key,
                     const std::vector<TensorBuffer>& input_buffers,
                     const std::vector<TensorBuffer>& output_buffers) const {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_key));
    return Run(signature_index, input_buffers, output_buffers);
  }

  /// @brief Runs the model for a given signature key asynchronously, if
  /// possible, with the provided input/output `TensorBuffer`s.
  ///
  /// If asynchronous execution is possible, `async` will be set to `true`;
  /// otherwise, the function runs the model synchronously.
  Expected<void> RunAsync(absl::string_view signature_key,
                          const std::vector<TensorBuffer>& input_buffers,
                          const std::vector<TensorBuffer>& output_buffers,
                          bool& async) const {
    async = true;
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_key));
    return RunAsync(signature_index, input_buffers, output_buffers, async);
  }

  /// @brief Runs the model for a given signature key synchronously with the
  /// provided input/output `TensorBuffer` map.
  ///
  /// If you have bound the input with external buffers through `Options`, you
  /// can skip providing those input buffers in the map.
  Expected<void> Run(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map)
      const {
    bool async = false;
    return RunMapHelper(signature_key, input_map, output_map, async);
  }

  /// @brief Runs the model for the default signature synchronously with the
  /// provided input/output `TensorBuffer` map.
  ///
  /// If you have bound the input with external buffers through `Options`, you
  /// can skip providing those input buffers in the map.
  Expected<void> Run(
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map)
      const {
    bool async = false;
    return RunMapWithIndexHelper(/*signature_index=*/0, input_map, output_map,
                                 async);
  }

  /// @brief Runs the model for a given signature key asynchronously, if
  /// possible, with the provided input/output `TensorBuffer` map.
  ///
  /// If asynchronous execution is possible, `async` will be set to `true`;
  /// otherwise, the function runs the model synchronously.
  Expected<void> RunAsync(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async) const {
    async = true;
    return RunMapHelper(signature_key, input_map, output_map, async);
  }

  /// @brief Returns `true` if the compiled model is fully accelerated with the
  /// given hardware accelerators.
  Expected<bool> IsFullyAccelerated();

  /// @brief Returns the profiler used by the compiled model.
  ///
  /// The returned `Profiler` does not own the underlying `LiteRtProfiler`.
  Expected<Profiler> GetProfiler() {
    LiteRtProfiler profiler = nullptr;
    LITERT_RETURN_IF_ERROR(LiteRtCompiledModelGetProfiler(Get(), &profiler));
    return Profiler(profiler, OwnHandle::kNo);
  };

  /// @brief Sets a callback function that will be called after every node/op
  /// during model execution to check if the execution should be cancelled.
  ///
  /// This behavior is defined here:
  /// tflite/core/subgraph.cc;l=1746-1750?q=tflite%20subgraph
  /// The callback should return `true` if execution should be cancelled.
  /// @note Use either this callback-based mechanism or the non-callback version
  /// (see below) with `EnableCancellation`/`Cancel`, but not both.
  void SetCancellationFunction(void* data,
                               bool (*check_cancelled_func)(void*)) {
    LiteRtSetCompiledModelCancellationFunction(Get(), data,
                                               check_cancelled_func);
  }

  /// @brief Sets a callback function for checking cancellation during
  /// execution.
  ///
  /// The callback will be called periodically during model execution. This is a
  /// C++-friendly version of `SetCancellationFunction`.
  void SetCancellationFunction(absl::AnyInvocable<bool()> check_cancelled_func);

  /// @brief Resizes the specified input tensor to support dynamic shapes.
  ///
  /// This function mirrors TFLite's `ResizeInputTensorStrict` API and requires
  /// the tensor signature to include dynamic dimensions. After calling this
  /// function, the compiled model will reallocate internal buffers as needed to
  /// accommodate the new tensor shape. For models that need relaxed validation,
  /// use `ResizeInputTensorNonStrict`.
  ///
  /// @note After resizing, previously obtained buffer requirements may be
  /// invalidated. Callers should re-query buffer requirements if needed.
  ///
  /// @param signature_index The index of the signature in the model.
  /// @param input_index The index of the input tensor in the signature.
  /// @param dims The new dimensions for the input tensor.
  /// @return Success if the resize operation completes successfully, or an
  /// error with an appropriate status code on failure.
  Expected<void> ResizeInputTensor(size_t signature_index, size_t input_index,
                                   absl::Span<const int> dims) {
    LITERT_RETURN_IF_ERROR(LiteRtCompiledModelResizeInputTensor(
        Get(), signature_index, input_index, dims.data(), dims.size()));
    return {};
  }

  /// @brief Resizes the specified input tensor by name for the given
  /// signature.
  Expected<void> ResizeInputTensor(size_t signature_index,
                                   absl::string_view input_name,
                                   absl::Span<const int> dims) {
    LITERT_ASSIGN_OR_RETURN(size_t input_index,
                            FindInputIndex(signature_index, input_name));
    return ResizeInputTensor(signature_index, input_index, dims);
  }

  /// @brief Resizes the specified input tensor by name for the given signature
  /// name.
  Expected<void> ResizeInputTensor(absl::string_view signature_name,
                                   absl::string_view input_name,
                                   absl::Span<const int> dims) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return ResizeInputTensor(signature_index, input_name, dims);
  }

  /// @brief Resizes the specified input tensor of the default signature by
  /// index.
  Expected<void> ResizeInputTensor(size_t input_index,
                                   absl::Span<const int> dims) {
    return ResizeInputTensor(/*signature_index=*/0, input_index, dims);
  }

  /// @brief Resizes the specified input tensor of the default signature by
  /// name.
  Expected<void> ResizeInputTensor(absl::string_view input_name,
                                   absl::Span<const int> dims) {
    return ResizeInputTensor(/*signature_index=*/0, input_name, dims);
  }

  /// @brief Non-strict variants mirror TFLite's `ResizeInputTensor` behavior,
  /// allowing arbitrary shape updates when backends support them.
  Expected<void> ResizeInputTensorNonStrict(size_t signature_index,
                                            size_t input_index,
                                            absl::Span<const int> dims) {
    LITERT_RETURN_IF_ERROR(LiteRtCompiledModelResizeInputTensorNonStrict(
        Get(), signature_index, input_index, dims.data(), dims.size()));
    return {};
  }

  /// @brief Resizes the specified input tensor by name for the given
  /// signature.
  Expected<void> ResizeInputTensorNonStrict(size_t signature_index,
                                            absl::string_view input_name,
                                            absl::Span<const int> dims) {
    LITERT_ASSIGN_OR_RETURN(size_t input_index,
                            FindInputIndex(signature_index, input_name));
    return ResizeInputTensorNonStrict(signature_index, input_index, dims);
  }

  /// @brief Resizes the specified input tensor by name for the given signature
  /// name.
  Expected<void> ResizeInputTensorNonStrict(absl::string_view signature_name,
                                            absl::string_view input_name,
                                            absl::Span<const int> dims) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return ResizeInputTensorNonStrict(signature_index, input_name, dims);
  }

  /// @brief Resizes the specified input tensor of the default signature by
  /// index.
  Expected<void> ResizeInputTensorNonStrict(size_t input_index,
                                            absl::Span<const int> dims) {
    return ResizeInputTensorNonStrict(/*signature_index=*/0, input_index, dims);
  }

  /// @brief Resizes the specified input tensor of the default signature by
  /// name.
  Expected<void> ResizeInputTensorNonStrict(absl::string_view input_name,
                                            absl::Span<const int> dims) {
    return ResizeInputTensorNonStrict(/*signature_index=*/0, input_name, dims);
  }

  /// @brief Reports an error to the compiled model's error reporter.
  ///
  /// Supports printf-style formatting for error messages.
  template <typename... Args>
  Expected<void> ReportError(const char* format, Args... args) const {
    LITERT_RETURN_IF_ERROR(
        LiteRtCompiledModelReportError(Get(), format, args...));
    return {};
  }

  /// @brief Clears all errors from the error reporter.
  /// @note This only works if the compiled model uses `BufferErrorReporter`,
  /// not `StderrReporter`.
  Expected<void> ClearErrors() const {
    LITERT_RETURN_IF_ERROR(LiteRtCompiledModelClearErrors(Get()));
    return {};
  }

  /// @brief Gets all error messages from the error reporter as a single
  /// string.
  ///
  /// @note This only works if the compiled model uses `BufferErrorReporter`,
  /// not `StderrReporter`. The C++ wrapper automatically manages memory using
  /// RAII.
  Expected<std::string> GetErrorMessages() const {
    char* error_messages = nullptr;
    LITERT_RETURN_IF_ERROR(
        LiteRtCompiledModelGetErrorMessages(Get(), &error_messages));

    // Use unique_ptr with custom deleter to ensure automatic cleanup
    std::unique_ptr<char, decltype(&std::free)> error_messages_ptr(
        error_messages, &std::free);

    if (!error_messages) {
      return std::string();
    }
    return std::string(error_messages);
  }

  //----------------------------------------------------------------------------
  // Underlying model accessors
  //----------------------------------------------------------------------------

  /// @brief Returns the number of signatures defined in the model.
  size_t GetNumSignatures() const { return model_.GetNumSignatures(); }

  /// @brief Returns the default signature key of the model.
  static absl::string_view DefaultSignatureKey() {
    return Model::DefaultSignatureKey();
  }

  /// @brief Returns the list of signature key names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureKeys() const {
    return model_.GetSignatureKeys();
  }

  /// @brief Returns the list of signatures defined in the model.
  Expected<std::vector<SimpleSignature>> GetSignatures() const {
    return model_.GetSignatures();
  }

  /// @brief Returns the signature at the given index.
  Expected<SimpleSignature> GetSignature(size_t signature_index) const {
    return model_.GetSignature(signature_index);
  }

  /// @brief Returns the signature index for a given signature key.
  ///
  /// Returns 0 if the signature key is empty.
  Expected<size_t> GetSignatureIndex(absl::string_view signature_key) const {
    return model_.GetSignatureIndex(signature_key);
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      size_t signature_index) const {
    return model_.GetSignatureInputNames(signature_index);
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames() const {
    return model_.GetSignatureInputNames();
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      absl::string_view signature_key) const {
    return model_.GetSignatureInputNames(signature_key);
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      size_t signature_index) const {
    return model_.GetSignatureOutputNames(signature_index);
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames() const {
    return model_.GetSignatureOutputNames();
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      absl::string_view signature_key) const {
    return model_.GetSignatureOutputNames(signature_key);
  }

  /// @brief Returns the tensor type for the n-th input tensor.
  Expected<RankedTensorType> GetInputTensorType(size_t signature_index,
                                                size_t input_index) const {
    return model_.GetInputTensorType(signature_index, input_index);
  }

  /// @brief Returns the tensor type for a given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      size_t signature_index, absl::string_view input_name) const {
    return model_.GetInputTensorType(signature_index, input_name);
  }

  /// @brief Returns the tensor type for a given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      absl::string_view signature_key, absl::string_view input_name) const {
    return model_.GetInputTensorType(signature_key, input_name);
  }

  /// @brief Gets the input tensor type of the default signature for a given
  /// input name.
  Expected<RankedTensorType> GetInputTensorType(
      absl::string_view input_name) const {
    return model_.GetInputTensorType(input_name);
  }

  /// @brief Returns the tensor type for the n-th output tensor.
  Expected<RankedTensorType> GetOutputTensorType(size_t signature_index,
                                                 size_t output_index) const {
    return model_.GetOutputTensorType(signature_index, output_index);
  }

  /// @brief Returns the tensor type for a given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      size_t signature_index, absl::string_view output_name) const {
    return model_.GetOutputTensorType(signature_index, output_name);
  }

  /// @brief Returns the tensor type for a given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view signature_key, absl::string_view output_name) const {
    return model_.GetOutputTensorType(signature_key, output_name);
  }

  /// @brief Gets the output tensor type of the default signature for a given
  /// output name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view output_name) const {
    return model_.GetOutputTensorType(output_name);
  }

  /// @internal
  /// @brief Wraps a `LiteRtCompiledModel` C object in a `CompiledModel` C++
  /// object.
  ///
  /// The `compiled_model` does not own the provided `litert_model`.
  /// @warning This is for internal use only.
  static CompiledModel WrapCObject(LiteRtModel litert_model,
                                   LiteRtCompiledModel compiled_model,
                                   OwnHandle owned) {
    return CompiledModel(litert_model, /*model_owned=*/OwnHandle::kNo,
                         compiled_model, owned);
  }

 protected:
  /// @internal
  /// @brief Creates a `CompiledModel` from a provided `litert::Model`.
  ///
  /// The model is loaded into memory, and the caller takes ownership of the
  /// returned `CompiledModel` object. The caller should keep the model alive
  /// until the `CompiledModel` is destroyed. The given `compilation_options`
  /// are used for model compilation, and `hardware_accelerators` selects the
  /// accelerator, regardless of whether the model is AOT or JIT compiled.
  ///
  /// @note The provided environment must outlive the compiled model and any
  /// executions running on it.
  /// @note Even if the model is fully AOT-compiled for an NPU, you must
  /// specify the NPU accelerator in `hardware_accelerators` to use it
  /// properly.
  static Expected<CompiledModel> Create(litert::Environment& env,
                                        const litert::Model& model,
                                        Options& compilation_options) {
    LITERT_RETURN_IF_ERROR(compilation_options.Build());
    LiteRtModel litert_model = model.Get();
    LiteRtCompiledModel compiled_model;
    LITERT_RETURN_IF_ERROR(LiteRtCreateCompiledModel(
        env.Get(), litert_model, compilation_options.Get(), &compiled_model));
    return CompiledModel(litert_model, /*model_owned=*/OwnHandle::kNo,
                         compiled_model,
                         /*owned=*/OwnHandle::kYes);
  }

  /// @internal
  /// @brief A simplified version of `Create` that uses default compilation
  /// options.
  ///
  /// The provided hardware accelerator is used to select the target
  /// accelerator.
  /// @note This should be specified for both JIT and AOT compiled models.
  static Expected<CompiledModel> Create(
      litert::Environment& env, const litert::Model& model,
      litert::HwAccelerators hardware_accelerators) {
    LITERT_ASSIGN_OR_RETURN(auto compilation_options, Options::Create());
    compilation_options.SetHardwareAccelerators(hardware_accelerators);
    return Create(env, model, compilation_options);
  }

  /// @brief Constructs a `CompiledModel` instance.
  ///
  /// @param model_owned Indicates whether the provided `litert_model` handle is
  /// owned by the `CompiledModel`.
  /// @param owned If `true`, the created object takes ownership of the
  /// `compiled_model` handle.
  explicit CompiledModel(LiteRtModel litert_model, OwnHandle model_owned,
                         LiteRtCompiledModel compiled_model, OwnHandle owned)
      : internal::Handle<LiteRtCompiledModel, LiteRtDestroyCompiledModel>(
            compiled_model, owned) {
    if (model_owned == OwnHandle::kYes) {
      model_ = Model::CreateFromOwnedHandle(litert_model);
    } else {
      model_ = Model::CreateFromNonOwnedHandle(litert_model);
    }
    LiteRtGetCompiledModelEnvironment(compiled_model, &env_);
  }

  static bool CheckCancelledWrapper(void* data);

  /// @brief Returns the signature input index for a given input tensor name.
  Expected<size_t> FindInputIndex(size_t signature_index,
                                  absl::string_view input_name) const;

  /// @brief Returns the signature output index for a given output tensor name.
  Expected<size_t> FindOutputIndex(size_t signature_index,
                                   absl::string_view output_name) const;

  /// @brief Creates a `TensorBuffer` with the given buffer requirements and
  /// tensor type.
  static Expected<TensorBuffer> CreateBufferImpl(
      const Environment& env,
      const TensorBufferRequirements& buffer_requirements,
      const RankedTensorType& tensor_type);

  /// @brief Creates a `TensorBuffer` for a given signature index and tensor
  /// name.
  Expected<TensorBuffer> CreateInputOutputBuffer(size_t signature_index,
                                                 absl::string_view tensor_name,
                                                 bool is_input) const;

  /// @brief Creates a `TensorBuffer` for a given signature and tensor name.
  Expected<TensorBuffer> CreateInputOutputBuffer(
      absl::string_view signature_name, absl::string_view tensor_name,
      bool is_input) const {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return CreateInputOutputBuffer(signature_index, tensor_name, is_input);
  }

  /// @brief Creates a vector of `TensorBuffer`s for a given signature
  /// subgraph.
  Expected<std::vector<TensorBuffer>> CreateInputOutputBuffers(
      size_t signature_index, bool is_input) const;

  /// @brief Returns the environment used to create this compiled model.
  ///
  /// The returned `Environment` does not own the underlying
  /// `LiteRtEnvironment`.
  Expected<Environment> GetEnvironment() const {
    return Environment::WrapCObject(env_, OwnHandle::kNo);
  }

  Expected<void> RunCApiHelper(LiteRtParamIndex signature_index,
                               size_t num_input_buffers,
                               LiteRtTensorBuffer* input_buffers,
                               size_t num_output_buffers,
                               LiteRtTensorBuffer* output_buffers,
                               bool& async) const;

  Expected<void> RunHelper(size_t signature_index,
                           absl::Span<const TensorBuffer> input_buffers,
                           absl::Span<const TensorBuffer> output_buffers,
                           bool& async) const;

  Expected<void> RunMapHelper(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async) const;

  Expected<void> RunMapWithIndexHelper(
      size_t signature_index,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async) const;

  LiteRtEnvironment env_;
  Model model_;
  absl::AnyInvocable<bool()> check_cancelled_func_;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_COMPILED_MODEL_H_
