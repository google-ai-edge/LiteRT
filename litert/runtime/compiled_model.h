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

#ifndef ODML_LITERT_LITERT_RUNTIME_COMPILED_MODEL_H_
#define ODML_LITERT_LITERT_RUNTIME_COMPILED_MODEL_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#if defined(LITERT_WITH_EXTERNAL_WEIGHT_LOADER)
#include "third_party/odml/litert/weight_loader/external_weight_loader_litert.h"
#endif  // defined(LITERT_WITH_EXTERNAL_WEIGHT_LOADER)
#if !defined(LITERT_DISABLE_NPU)
#include "litert/core/cache/compilation_cache.h"
#endif  // !defined(LITERT_DISABLE_NPU)
#include "litert/core/environment.h"
#include "litert/core/model/model.h"
#include "litert/runtime/accelerator.h"
#include "litert/runtime/custom_op_dispatcher.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/metrics.h"
#include "litert/runtime/profiler.h"
#include "litert/runtime/tensor_identifier.h"
#include "litert/runtime/tfl_utils.h"
#include "tflite/converter/allocation.h"
#include "tflite/core/api/error_reporter.h"
#include "tflite/delegates/utils/simple_opaque_delegate.h"
#include "tflite/interpreter.h"
#include "tflite/model_builder.h"

using TfLiteTensorIdentifier = litert::internal::TfLiteTensorIdentifier;
using TensorIdentifierHash = litert::internal::TensorIdentifierHash;
using TensorIdentifierEqual = litert::internal::TensorIdentifierEqual;

// The LiteRtCompiledModelT is internal implementation of CompiledModel C++ API.
class LiteRtCompiledModelT {
 public:
  using Ptr = std::unique_ptr<LiteRtCompiledModelT>;

  explicit LiteRtCompiledModelT(LiteRtEnvironmentT* env) : env_(env) {}
  ~LiteRtCompiledModelT() {
    // If the profiler is set, delete it here.
    if (profiler_ != nullptr) {
      delete profiler_;
      profiler_ = nullptr;
    }
  };

  // Creates a LiteRtCompiledModelT from a LiteRtModel object.
  // The model is loaded into memory and the caller takes ownership of the
  // returned object.
  static litert::Expected<Ptr> Create(
      LiteRtEnvironmentT* env, LiteRtModel model,
      LiteRtOptions jit_compilation_options = nullptr);

  // Returns the buffer requirements for the n-th input tensor. The returned
  // LiteRtTensorBufferRequirements is used to create the input tensor
  // buffer.
  litert::Expected<const LiteRtTensorBufferRequirementsT*>
  GetInputBufferRequirements(absl::string_view signature_key,
                             size_t input_index);

  // Returns the buffer requirements for the n-th input tensor using sigature
  // index. The returned LiteRtTensorBufferRequirements is used to create the
  // input tensor buffer.
  litert::Expected<const LiteRtTensorBufferRequirementsT*>
  GetInputBufferRequirements(size_t signature_index, size_t input_index) {
    if (signature_index >= signature_keys_.size()) {
      return litert::Unexpected(
          kLiteRtStatusErrorIndexOOB,
          "Signature index is out of range of signature keys");
    }
    return GetInputBufferRequirements(*signature_keys_[signature_index],
                                      input_index);
  }

  // Returns the buffer requirements for the n-th output tensor. The returned
  // LiteRtTensorBufferRequirements is used to create the output tensor
  // buffer.
  litert::Expected<const LiteRtTensorBufferRequirementsT*>
  GetOutputBufferRequirements(absl::string_view signature_key,
                              size_t output_index);

  // The same as GetOutputBufferRequirements() for C API.
  litert::Expected<LiteRtTensorBufferRequirements>
  GetOutputBufferRequirementsCApi(size_t signature_index, size_t output_index) {
    if (signature_index >= signature_keys_.size()) {
      return litert::Unexpected(
          kLiteRtStatusErrorIndexOOB,
          "Signature index is out of range of signature keys");
    }
    LITERT_ASSIGN_OR_RETURN(
        const LiteRtTensorBufferRequirementsT* requirements,
        GetOutputBufferRequirements(*signature_keys_[signature_index],
                                    output_index));
    return const_cast<LiteRtTensorBufferRequirements>(requirements);
  }

  // Returns the shapes of all output tensors for the given signature key.
  litert::Expected<void> GetOutputTensorShapes(
      absl::string_view signature_key, absl::Span<LiteRtLayout>& output_layouts,
      bool update_allocation = false);

  // Returns the shapes of all output tensors for the given signature index.
  litert::Expected<void> GetOutputTensorShapes(
      size_t signature_index, absl::Span<LiteRtLayout>& output_layouts,
      bool update_allocation = false) {
    if (signature_index >= signature_keys_.size()) {
      return litert::Unexpected(
          kLiteRtStatusErrorIndexOOB,
          "Signature index is out of range of signature keys");
    }
    return GetOutputTensorShapes(*signature_keys_[signature_index],
                                 output_layouts, update_allocation);
  }

  // Returns the layout for an input tensor identified by signature and index.
  litert::Expected<LiteRtLayout> GetInputTensorLayout(size_t signature_index,
                                                      size_t input_index);

  // Runs the model of the given signature with the provided input/output
  // litert::TensorBuffers. If parameter `async` is true, then the model is run
  // asynchronously, if possible. Upon returning, the function sets parameter
  // `async` to true if asynchronous execution was requested and possible,
  // otherwise it sets it to false.
  litert::Expected<void> Run(
      absl::string_view signature_key,
      const std::vector<LiteRtTensorBuffer>& input_buffers,
      const std::vector<LiteRtTensorBuffer>& output_buffers, bool& async);

  // The same as Run() for C API.
  litert::Expected<void> RunCApi(size_t signature_index,
                                 size_t num_input_buffers,
                                 const LiteRtTensorBuffer* input_buffers,
                                 size_t num_output_buffers,
                                 const LiteRtTensorBuffer* output_buffers,
                                 bool* async);

  litert::Expected<void> StartMetricsCollection(int detail_level);

  litert::Expected<LiteRtMetricsT> StopMetricsCollection();

  // Returns true if a non delegated operation is found in the interpreter.
  litert::Expected<bool> HasNonDelegatedOps();

  // Returns the environment associated with the compiled model.
  litert::Expected<LiteRtEnvironmentT*> GetEnvironment() { return env_; }

  // Returns the profiler used by the compiled model.
  litert::Expected<LiteRtProfilerT*> GetProfiler() { return profiler_; }

  // Resizes the specified input tensor to support dynamic shapes.
  litert::Expected<void> ResizeInputTensor(size_t signature_index,
                                           size_t input_index,
                                           absl::Span<const int> dims);

  // Returns the external buffer context which contains dispatch annotations.
  LiteRtExternalLiteRtBufferContextT* GetBufferContext() {
    return buffer_context_.get();
  }

  // Returns the number of signatures in the model.
  size_t GetNumSignatures() const { return signature_keys_.size(); }

  // Error reporter APIs

  // Reports an error. Thread-safe.
  void ReportError(const char* format, ...);

  // Clears all errors (only available in buffer mode)
  litert::Expected<void> ClearErrors();

  // Gets all error messages (only available in buffer mode)
  litert::Expected<std::string> GetErrorMessages();

  // Returns the TFLite interpreter associated with the compiled model.
  friend litert::Expected<::tflite::Interpreter*> GetInterpreter(
      LiteRtCompiledModelT* compiled_model);

  // Cancellation APIs

  // Enables cancellation for the compiled model. Once enabled, model execution
  // can be cancelled from any thread using Cancel().
  litert::Expected<void> EnableCancellation();

  // Sets a callback function for checking cancellation during execution.
  // The callback will be called periodically during model execution.
  void SetCancellationFunction(void* data, bool (*check_cancelled_func)(void*));

  // Sets a callback function for checking cancellation during execution.
  // The callback will be called periodically during model execution. This is a
  // C++-friendly version of SetCancellationFunction.
  void SetCancellationFunction(absl::AnyInvocable<bool()> check_cancelled_func);

  // Cancels an ongoing model execution. Can be called from any thread.
  // Returns an error if cancellation is not enabled.
  litert::Expected<void> Cancel();

 private:
  static bool CheckCancelledWrapper(void* data);
  // Helper function to automatically resize input tensor based on shape change
  static litert::Expected<bool> InputTensorNeedsResize(
      const TfLiteTensor* tensor, absl::Span<const int> new_shape);

  // Friend function to test InputTensorNeedsResize.
  friend litert::Expected<bool> InputTensorNeedsResize(
      LiteRtCompiledModelT* compiled_model, const TfLiteTensor* tensor,
      absl::Span<const int> new_shape);

  // A opaque delegate and its metrics collection functions.
  struct Delegate {
    std::unique_ptr<LiteRtDelegateWrapperT,
                    std::function<void(LiteRtDelegateWrapper)>>
        delegate;
    // NOLINTBEGIN(*-readability-class-member-naming)
    // Starts collection of HW-specific metrics at a specific level of detail.
    LiteRtStatus (*StartMetricsCollection)(LiteRtDelegateWrapper delegate,
                                           int detail_level);

    // Stops collection of HW-specific metrics and report the collected metrics.
    LiteRtStatus (*StopMetricsCollection)(LiteRtDelegateWrapper delegate,
                                          LiteRtMetricsT* metrics);
    // NOLINTEND(*-readability-class-member-naming)
  };

  // Initializes the internal TFLite interpreter and related objects.
  // This is called in the public Create*() methods.
  // The flatbuffer_model_ must be set before calling this method.
  litert::Expected<void> InitializeRuntime(
      LiteRtEnvironmentT* env, LiteRtHwAcceleratorSet hardware_accelerators,
      LiteRtOptions jit_compilation_options);

  // Handles any JIT compilation and initializes the flatbuffer_model_ and
  // related field within the compiled model.
  //
  // If no JIT compilation is requested, the compiled model will point to the
  // underlying tflite::Model* owned by the input litert model. The compiled
  // models alloc_ and model_buf_ will be nullptr as these are only relevant
  // when compiled model owns a flatbuffer.
  //
  // If JIT compilation is requested and compilation caching is enabled, the
  // compiled model will first check the cache for the compiled model. If the
  // model is found in the cache, the compiled model will load the model from
  // the cache and the JIT compilation will not occur. The alloc_ and
  // model_buf_ will be initialized based on the cached model.
  //
  // If JIT compilation does occur (either because compilation caching is
  // disabled or the model is not found in the cache), a new flatbuffer owned by
  // the compiled model will be serialized from the result of compilation. The
  // alloc_ and model_buf_ will be set for storage of the new flatbuffer.
  //
  // NOTE: JIT compilation invalidates the input litert model.
  // TODO: Design a better abstraction for optional ownership for flatbuffer,
  // consider caching JIT result.
  litert::Expected<void> InitializeModel(LiteRtModelT& model,
                                         LiteRtHwAcceleratorSet hw_accelerators,
                                         LiteRtOptions options,
                                         LiteRtEnvironmentT& env);

  // Returns the base address of the flatbuffer memory.
  //
  // If no JIT compilation has taken place, this points to flatbuffer memory
  // owned by the incoming litert model (litert models always owns their
  // flatbuffer memory until serialization).
  //
  // If JIT compilation has taken place, this points to the base address of the
  // a newly serialized flatbuffer which is owned by the compiled model (in
  // model_buf_);
  //
  // NOTE: This should never be nullptr after initialization.
  const char* GetModelBase() {
    if (fb_model_ == nullptr) {
      return nullptr;
    }

    // fb_model_->allocation is only null when the flatbuffer is built with
    // BuildFlatBufferFromModel, which is not currently in use in either
    // litert::LoadModel or LiteRtCompiledModelT::Create.
    const auto* alloc = fb_model_->allocation();
    if (alloc) {
      // NOTE: During JIT, alloc->base() == model_buf_.Data(), which is owned
      // by the compiled model. Otherwise, model_buf_.Data() is nullptr and
      // alloc->base() points a buffer owned by the incoming litert model.
      return reinterpret_cast<const char*>(alloc->base());
    }

    return nullptr;
  }

  // Returns the buffer requirements for the given tensor.
  litert::Expected<const LiteRtTensorBufferRequirementsT*>
  GetTensorBufferRequirements(const TfLiteTensor* tensor);

  // Returns the SignatureRunner for the given signature key.
  // If the signature key is not found, returns nullptr.
  tflite::SignatureRunner* GetSignatureRunner(absl::string_view signature_key);

  // Structure to track constant output tensors and their locked buffer
  // addresses
  struct ConstantOutputInfo {
    LiteRtTensorBuffer buffer;
    void* locked_address;
    const char* tensor_name;
    size_t data_size;
  };

  // Registers the TensorBuffer for the given tensor with the SignatureRunner.
  // If the TensorBuffer can be directly consumed as CPU Tensors, they'll be
  // locked and use it with CustomAllocation. The locked buffer is kept in the
  // `locked_buffers`. Caller is responsible for unlocking of these buffers.
  // If the TensorBuffer can be consumed by the delegate, then `tensor` will be
  // marked as non-CPU to avoid TFLite from allocating it.
  litert::Expected<void> RegisterBuffer(
      tflite::SignatureRunner* runner, TfLiteTensor* tensor,
      const char* tensor_name, LiteRtTensorBufferT* buffer, bool is_input,
      std::vector<LiteRtTensorBuffer>& locked_buffers,
      std::vector<ConstantOutputInfo>& constant_outputs);

  void RegisterDelegate(Delegate&& delegate) {
    delegates_.push_back(std::move(delegate));
  }

  // Checks the CPU Tensors and stores them in the `cpu_tensors_` set.
  void CheckCpuTensors();

#if !defined(LITERT_DISABLE_NPU)
  // Applies the plugins to the model and caches the compiled model if
  // compilation caching is enabled. Returns true if the compiled model is
  // initialized from the plugins, false otherwise.
  litert::Expected<bool> ApplyPluginsWithCaching(
      LiteRtModelT& model, LiteRtHwAcceleratorSet hw_accelerators,
      LiteRtOptionsT options, LiteRtEnvironmentT& env);

  // Tries to load the model from the cache. Returns true if the model is loaded
  // from the cache, false otherwise.
  bool TryLoadingFromCache(uint64_t model_hash);
#endif  // !defined(LITERT_DISABLE_NPU)

  // The environment associated with the compiled model.
  LiteRtEnvironmentT* env_;

  // NOTE: Any fields that must be destroyed after the TFL interpreter
  // is destroyed must be listed before field interp_.

  std::vector<Delegate> delegates_;
  std::vector<std::unique_ptr<litert::internal::CustomOpDispatcher>>
      custom_op_dispatchers_;

  // The TFL interpreter.
  std::unique_ptr<::tflite::Interpreter> interp_;

  // NOTE: List below TFL interpreter related objects used to run the
  // model. Note that these fields will be destroyed before the TFL interpreter
  // is destroyed.

  std::unique_ptr<::tflite::FlatBufferModel> fb_model_;
  litert::OwningBufferRef<uint8_t> model_buf_;
  std::vector<const std::string*> signature_keys_;
  // If JIT compilation hasn't happened, the flatbuffer fd belongs to the
  // incoming literal model. If JIT compilation has happened, the fd belongs to
  // a newly serialized flatbuffer owned by the compiled model. If the model is
  // loaded from the cache, the fd belongs to the cached flatbuffer.
  int fb_model_fd_ = -1;

#if !defined(LITERT_DISABLE_NPU)
  // The compilation cache used to store the compiled model. If the model is
  // found in the cache, the compiled model will be loaded from the cache.
  // Otherwise, the compiled model will be compiled and saved to the cache.
  std::optional<litert::internal::CompilationCache> compilation_cache_;
  std::optional<LiteRtModelT::Ptr> cached_model_;
#endif  // !defined(LITERT_DISABLE_NPU)

  // The buffer requirement maps for CPU buffers. For delegates with CPU
  // buffers, they don't register TensorBufferRequirements. Instead, the
  // CompiledModel creates the TensorBufferRequirements and stores them
  // in this map.
  absl::flat_hash_map<TfLiteTensorIdentifier, LiteRtTensorBufferRequirementsPtr,
                      TensorIdentifierHash, TensorIdentifierEqual>
      cpu_buffer_requirements_;

  // Map from signature key to SignatureRunner. This is used to lazy calling
  // GetSignatureRunner() which is expensive.
  absl::flat_hash_map<absl::string_view, tflite::SignatureRunner*>
      signature_runners_;

  // The ExternalLiteRtBufferContext used to register tensor buffers with
  // Delegates.
  // Note: The ExternalLiteRtBufferContext must be destroyed after the
  // Interpreter.
  std::unique_ptr<LiteRtExternalLiteRtBufferContextT> buffer_context_;

#if defined(LITERT_WITH_EXTERNAL_WEIGHT_LOADER)
  // The loader that manages external weight metadata and bindings.
  std::unique_ptr<weight_loader::WeightLoader> weight_loader_;
#endif  // defined(LITERT_WITH_EXTERNAL_WEIGHT_LOADER)

  // File system hints about the originating model location.
  std::optional<std::string> model_directory_;

  // The set of CPU Tensors. This is used to manage TensorBufferRequirements
  // for shared CPU Tensors.
  absl::flat_hash_set<TfLiteTensorIdentifier, TensorIdentifierHash,
                      TensorIdentifierEqual>
      cpu_tensors_;

  // The profiler used by the compiled model. This is used to forward the
  // profiler events to the TFLite interpreter.
  LiteRtProfilerT* profiler_ = nullptr;

  // The error reporter used by the compiled model
  std::unique_ptr<tflite::ErrorReporter> error_reporter_;

  // Cancellation support
  bool (*check_cancelled_func_)(void*) = nullptr;
  absl::AnyInvocable<bool()> check_cancelled_func_cpp_;
};

#endif  // ODML_LITERT_LITERT_RUNTIME_COMPILED_MODEL_H_
