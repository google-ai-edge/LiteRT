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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_COMPILED_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_COMPILED_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_compilation_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/core/environment.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_buffer.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"  // from @org_tensorflow
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"  // from @org_tensorflow
#include "tensorflow/lite/interpreter.h"  // from @org_tensorflow
#include "tensorflow/lite/model_builder.h"  // from @org_tensorflow

// The LiteRtCompiledModelT is internal implementation of CompiledModel C++ API.
class LiteRtCompiledModelT {
 public:
  using Ptr = std::unique_ptr<LiteRtCompiledModelT>;

  LiteRtCompiledModelT() = default;
  ~LiteRtCompiledModelT() = default;

  // Creates a LiteRtCompiledModelT from a LiteRtModel object.
  // The model is loaded into memory and the caller takes ownership of the
  // returned object.
  static litert::Expected<Ptr> Create(
      LiteRtEnvironmentT* env, LiteRtModel model,
      LiteRtCompilationOptions jit_compilation_options = nullptr);

  // Returns the buffer requirements for the n-th input tensor. The returned
  // LiteRtTensorBufferRequirements is used to create the input tensor
  // buffer.
  litert::Expected<LiteRtTensorBufferRequirements> GetInputBufferRequirements(
      absl::string_view signature_key, size_t input_index);

  // The same as GetInputBufferRequirements() for C API.
  litert::Expected<LiteRtTensorBufferRequirements>
  GetInputBufferRequirementsCApi(size_t signature_index, size_t input_index) {
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
  litert::Expected<LiteRtTensorBufferRequirements> GetOutputBufferRequirements(
      absl::string_view signature_key, size_t output_index);

  // The same as GetOutputBufferRequirements() for C API.
  litert::Expected<LiteRtTensorBufferRequirements>
  GetOutputBufferRequirementsCApi(size_t signature_index, size_t output_index) {
    if (signature_index >= signature_keys_.size()) {
      return litert::Unexpected(
          kLiteRtStatusErrorIndexOOB,
          "Signature index is out of range of signature keys");
    }
    return GetOutputBufferRequirements(*signature_keys_[signature_index],
                                       output_index);
  }

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
                                 LiteRtTensorBuffer* input_buffers,
                                 size_t num_output_buffers,
                                 LiteRtTensorBuffer* output_buffers,
                                 bool* async);

 private:
  // Initializes the internal TFLite interpreter and related objects.
  // This is called in the public Create*() methods.
  // The flatbuffer_model_ must be set before calling this method.
  litert::Expected<void> InitializeRuntime();

  // Handles any JIT compilation and intializes the flatbuffer_model_ and
  // related field within the compiled model.
  //
  // If no JIT compilation is requested, the compiled model will point to the
  // underlying tflite::Model* owned by the input litert model. The compiled
  // models alloc_ and model_buf_ will be nullptr as these are only relevant
  // when compiled model owns a flatbuffer.
  //
  // If JIT compilation does occur, a new flatbuffer owned by the compiled model
  // will be serialized from the result of compilation. The alloc_ and
  // model_buf_ will be set for storage of the new flatbuffer.
  //
  // NOTE: JIT compilation invalidates the input litert model.
  // TODO: Design a better abstraction for optional ownership for flatbuffer,
  // consider caching JIT result.
  litert::Expected<void> InitializeModel(LiteRtModelT& model,
                                         LiteRtHwAcceleratorSet hw_accelerators,
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
  litert::Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
      const TfLiteTensor* tensor);

  // Returns the SignatureRunner for the given signature key.
  // If the signature key is not found, returns nullptr.
  tflite::SignatureRunner* GetSignatureRunner(absl::string_view signature_key);

  // Registers the TensorBuffer for the given tensor with the SignatureRunner.
  // If the TensorBuffer can be directly consumed as CPU Tensors, they'll be
  // locked and use it with CustomAllocation. The locked buffer is kept in the
  // `locked_buffers`. Caller is responsible for unlocking of these buffers.
  // If the TensorBuffer can be consumed by the delegate, then `tensor` will be
  // marked as non-CPU to avoid TFLite from allocating it.
  litert::Expected<void> RegisterBuffer(
      tflite::SignatureRunner* runner, TfLiteTensor* tensor,
      const char* tensor_name, LiteRtTensorBuffer buffer, bool is_input,
      std::vector<LiteRtTensorBuffer>& locked_buffers);

  void RegisterDelegate(tflite::TfLiteOpaqueDelegateUniquePtr&& delegate) {
    delegates_.push_back(std::move(delegate));
  }

  // Checks the CPU Tensors and stores them in the `cpu_tensors_` set.
  void CheckCpuTensors();

  // Map from signature key to SignatureRunner. This is used to lazy calling
  // GetSignatureRunner() which is expensive.
  absl::flat_hash_map<absl::string_view, tflite::SignatureRunner*>
      signature_runners_;

  // The buffer requirement maps for CPU buffers. For delegates with CPU
  // buffers, they don't register TensorBufferRequirements. Instead, the
  // CompiledModel creates the TensorBufferRequirements and stores them
  // in this map.
  absl::flat_hash_map<const TfLiteTensor*, litert::TensorBufferRequirements>
      cpu_buffer_requirements_;

  // The Interpreter and related objects used to run the model.
  std::unique_ptr<::tflite::Interpreter> interp_;
  std::unique_ptr<::tflite::FlatBufferModel> fb_model_;
  litert::OwningBufferRef<uint8_t> model_buf_;
  std::vector<const std::string*> signature_keys_;

  // The ExternalLiteRtBufferContext used to register tensor buffers with
  // Delegates.
  // Note: The ExternalLiteRtBufferContext must be destroyed after the
  // Interpreter.
  std::unique_ptr<litert::internal::ExternalLiteRtBufferContext>
      buffer_context_;

  std::vector<tflite::TfLiteOpaqueDelegateUniquePtr> delegates_;

  // The set of CPU Tensors. This is used to manage TensorBufferRequirements
  // for shared CPU Tensors.
  absl::flat_hash_set<const void*> cpu_tensors_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_COMPILED_MODEL_H_
