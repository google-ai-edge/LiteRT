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
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"

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
class LlmLiteRtMtpDrafter;
}  // namespace lm

namespace internal::compiled_model_detail {

inline Expected<TensorBufferRequirements> ToTensorBufferRequirements(
    const internal::EnvironmentHolder& env,
    const LiteRtTensorBufferRequirements litert_requirements) {
  int num_types;
  LITERT_RETURN_IF_ERROR(
      env.runtime->GetNumTensorBufferRequirementsSupportedBufferTypes(
          litert_requirements, &num_types));
  std::vector<TensorBufferType> supported_types;
  supported_types.reserve(num_types);
  for (int i = 0; i < num_types; ++i) {
    LiteRtTensorBufferType type;
    LITERT_RETURN_IF_ERROR(
        env.runtime->GetTensorBufferRequirementsSupportedTensorBufferType(
            litert_requirements, i, &type));
    supported_types.push_back(static_cast<TensorBufferType>(type));
  }

  size_t buffer_size;
  LITERT_RETURN_IF_ERROR(env.runtime->GetTensorBufferRequirementsBufferSize(
      litert_requirements, &buffer_size));

  size_t alignment;
  LITERT_RETURN_IF_ERROR(env.runtime->GetTensorBufferRequirementsAlignment(
      litert_requirements, &alignment));

  int num_strides;
  const uint32_t* strides_ptr;
  LITERT_RETURN_IF_ERROR(env.runtime->GetTensorBufferRequirementsStrides(
      litert_requirements, &num_strides, &strides_ptr));
  std::vector<uint32_t> strides;
  if (num_strides > 0 && strides_ptr != nullptr) {
    strides.assign(strides_ptr, strides_ptr + num_strides);
  }

  if (num_strides == 0 || strides[0] == 0) {
    return TensorBufferRequirements::CreateWithAlignment(
        absl::MakeConstSpan(supported_types), buffer_size, alignment);
  }
  return TensorBufferRequirements::CreateWithAlignment(
      absl::MakeConstSpan(supported_types), buffer_size, alignment,
      absl::MakeConstSpan(strides));
}

inline absl::string_view FetchTensorName(const internal::EnvironmentHolder& env,
                                         LiteRtTensor tensor) {
  const char* name;
  LITERT_ABORT_IF_ERROR(env.runtime->GetTensorName(tensor, &name));
  return name;
}

inline std::uint32_t FetchTensorIndex(const internal::EnvironmentHolder& env,
                                      LiteRtTensor tensor) {
  std::uint32_t index;
  LITERT_ABORT_IF_ERROR(env.runtime->GetTensorIndex(tensor, &index));
  return index;
}

inline LiteRtTensorTypeId FetchTensorTypeId(
    const internal::EnvironmentHolder& env, LiteRtTensor tensor) {
  LiteRtTensorTypeId type_id;
  LITERT_ABORT_IF_ERROR(env.runtime->GetTensorTypeId(tensor, &type_id));
  return type_id;
}

inline std::variant<LiteRtUnrankedTensorType, litert::RankedTensorType>
FetchTensorType(const internal::EnvironmentHolder& env, LiteRtTensor tensor,
                LiteRtTensorTypeId type_id) {
  if (type_id == kLiteRtRankedTensorType) {
    LiteRtRankedTensorType ranked_tensor_type;
    LITERT_ABORT_IF_ERROR(
        env.runtime->GetRankedTensorType(tensor, &ranked_tensor_type));
    return litert::RankedTensorType(ranked_tensor_type);
  } else {
    LiteRtUnrankedTensorType unranked_tensor_type;
    LITERT_ABORT_IF_ERROR(
        env.runtime->GetUnrankedTensorType(tensor, &unranked_tensor_type));
    return unranked_tensor_type;
  }
}

inline LiteRtQuantizationTypeId FetchTensorQuantizationTypeId(
    const internal::EnvironmentHolder& env, LiteRtTensor tensor) {
  LiteRtQuantizationTypeId quantization_type_id;
  LITERT_ABORT_IF_ERROR(
      env.runtime->GetQuantizationTypeId(tensor, &quantization_type_id));
  return quantization_type_id;
}

inline LiteRtQuantizationPerTensor FetchTensorQuantizationPerTensor(
    const internal::EnvironmentHolder& env, LiteRtTensor tensor) {
  if (FetchTensorQuantizationTypeId(env, tensor) !=
      kLiteRtQuantizationPerTensor) {
    return {};
  }
  LiteRtQuantizationPerTensor per_tensor_quantization;
  LITERT_ABORT_IF_ERROR(
      env.runtime->GetPerTensorQuantization(tensor, &per_tensor_quantization));
  return per_tensor_quantization;
}

inline LiteRtQuantizationPerChannel FetchTensorQuantizationPerChannel(
    const internal::EnvironmentHolder& env, LiteRtTensor tensor) {
  if (FetchTensorQuantizationTypeId(env, tensor) !=
      kLiteRtQuantizationPerChannel) {
    return {};
  }
  LiteRtQuantizationPerChannel per_channel_quantization;
  LITERT_ABORT_IF_ERROR(env.runtime->GetPerChannelQuantization(
      tensor, &per_channel_quantization));
  return per_channel_quantization;
}

inline absl::string_view FetchSignatureKey(
    const internal::EnvironmentHolder& env, LiteRtSignature signature) {
  const char* key;
  LITERT_ABORT_IF_ERROR(env.runtime->GetSignatureKey(signature, &key));
  return key;
}

inline std::vector<absl::string_view> FetchSignatureInputNames(
    const internal::EnvironmentHolder& env, LiteRtSignature signature) {
  LiteRtParamIndex num_inputs;
  LITERT_ABORT_IF_ERROR(
      env.runtime->GetNumSignatureInputs(signature, &num_inputs));
  std::vector<absl::string_view> input_names;
  input_names.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const char* name;
    LITERT_ABORT_IF_ERROR(
        env.runtime->GetSignatureInputName(signature, i, &name));
    input_names.push_back(name);
  }
  return input_names;
}

inline std::vector<absl::string_view> FetchSignatureOutputNames(
    const internal::EnvironmentHolder& env, LiteRtSignature signature) {
  LiteRtParamIndex num_outputs;
  LITERT_ABORT_IF_ERROR(
      env.runtime->GetNumSignatureOutputs(signature, &num_outputs));
  std::vector<absl::string_view> output_names;
  output_names.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    const char* name;
    LITERT_ABORT_IF_ERROR(
        env.runtime->GetSignatureOutputName(signature, i, &name));
    output_names.push_back(name);
  }
  return output_names;
}

inline std::vector<std::unique_ptr<SimpleTensor>> FetchSignatureInputTensors(
    const internal::EnvironmentHolder& env, LiteRtSignature signature) {
  LiteRtParamIndex num_inputs;
  LITERT_ABORT_IF_ERROR(
      env.runtime->GetNumSignatureInputs(signature, &num_inputs));
  std::vector<std::unique_ptr<SimpleTensor>> input_tensors;
  input_tensors.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    LiteRtTensor tensor;
    LITERT_ABORT_IF_ERROR(
        env.runtime->GetSignatureInputTensorByIndex(signature, i, &tensor));
    input_tensors.push_back(std::make_unique<SimpleTensor>(
        FetchTensorIndex(env, tensor), FetchTensorName(env, tensor),
        FetchTensorTypeId(env, tensor),
        FetchTensorType(env, tensor, FetchTensorTypeId(env, tensor)),
        FetchTensorQuantizationTypeId(env, tensor),
        FetchTensorQuantizationPerTensor(env, tensor),
        FetchTensorQuantizationPerChannel(env, tensor)));
  }
  return input_tensors;
}

inline std::vector<std::unique_ptr<SimpleTensor>> FetchSignatureOutputTensors(
    const internal::EnvironmentHolder& env, LiteRtSignature signature) {
  LiteRtParamIndex num_outputs;
  LITERT_ABORT_IF_ERROR(
      env.runtime->GetNumSignatureOutputs(signature, &num_outputs));
  std::vector<std::unique_ptr<SimpleTensor>> output_tensors;
  output_tensors.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    LiteRtTensor tensor;
    LITERT_ABORT_IF_ERROR(
        env.runtime->GetSignatureOutputTensorByIndex(signature, i, &tensor));
    output_tensors.push_back(std::make_unique<SimpleTensor>(
        FetchTensorIndex(env, tensor), FetchTensorName(env, tensor),
        FetchTensorTypeId(env, tensor),
        FetchTensorType(env, tensor, FetchTensorTypeId(env, tensor)),
        FetchTensorQuantizationTypeId(env, tensor),
        FetchTensorQuantizationPerTensor(env, tensor),
        FetchTensorQuantizationPerChannel(env, tensor)));
  }
  return output_tensors;
}

}  // namespace internal::compiled_model_detail

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
class CompiledModel : public internal::BaseHandle<LiteRtCompiledModel> {
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
  friend class lm::LlmLiteRtMtpDrafter;

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
    auto env_holder = env.GetHolder();
    LITERT_ASSIGN_OR_RETURN(auto owned_options,
                            BuildOptions(compilation_options, env_holder));
    LiteRtModel litert_model;
    if (auto status = env_holder.runtime->CreateModelFromFile(
            model_filename.c_str(), &litert_model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from file");
    }
    LiteRtCompiledModel compiled_model;
    if (auto res = env_holder.runtime->CreateCompiledModel(
            env_holder.handle, litert_model, owned_options.get(),
            &compiled_model);
        res != kLiteRtStatusOk) {
      env_holder.runtime->DestroyModel(litert_model);
      return Unexpected(ToStatus(res), "Failed to compile model");
    }
    return CompiledModel(env_holder, litert_model,
                         /*model_owned=*/OwnHandle::kYes, compiled_model,
                         /*owned=*/OwnHandle::kYes, std::move(owned_options));
  }

  /// @brief An overload of `Create` that takes a buffer reference to the model
  /// instead of a filename.
  static Expected<CompiledModel> Create(litert::Environment& env,
                                        BufferRef<uint8_t> model_buffer,
                                        Options& compilation_options) {
    auto env_holder = env.GetHolder();
    LITERT_ASSIGN_OR_RETURN(
        auto owned_options,
        BuildOptions(std::move(compilation_options), env.GetHolder()));
    LiteRtModel litert_model;
    if (auto status = env_holder.runtime->CreateModelFromBuffer(
            model_buffer.Data(), model_buffer.Size(), &litert_model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from buffer");
    }
    LiteRtCompiledModel compiled_model;
    if (auto res = env_holder.runtime->CreateCompiledModel(
            env_holder.handle, litert_model, owned_options.get(),
            &compiled_model);
        res != kLiteRtStatusOk) {
      env_holder.runtime->DestroyModel(litert_model);
      return Unexpected(ToStatus(res), "Failed to compile model");
    }
    return CompiledModel(env_holder, litert_model,
                         /*model_owned=*/OwnHandle::kYes, compiled_model,
                         /*owned=*/OwnHandle::kYes, std::move(owned_options));
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
                            GetSignatureIndex(signature_name));
    return GetInputBufferRequirements(signature_index, input_name);
  }

  /// @brief Returns the buffer requirements for the n-th input tensor.
  ///
  /// The returned `TensorBufferRequirements` is used to create the input
  /// tensor buffer.
  Expected<TensorBufferRequirements> GetInputBufferRequirements(
      size_t signature_index, size_t input_index) const {
    LiteRtTensorBufferRequirements buffer_requirements;
    LITERT_RETURN_IF_ERROR(
        env_.runtime->GetCompiledModelInputBufferRequirements(
            Get(), signature_index, input_index, &buffer_requirements));
    return internal::compiled_model_detail::ToTensorBufferRequirements(
        env_, buffer_requirements);
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
                            GetSignatureIndex(signature_name));
    return GetOutputBufferRequirements(signature_index, output_name);
  }

  /// @brief Returns the layout of the given input tensor.
  ///
  /// This reflects the most recent shape requested via `ResizeInputTensor` or
  /// automatic resize during execution.
  Expected<Layout> GetInputTensorLayout(size_t signature_index,
                                        size_t input_index) const {
    LiteRtLayout input_layout;
    LITERT_RETURN_IF_ERROR(env_.runtime->GetCompiledModelInputTensorLayout(
        Get(), signature_index, input_index, &input_layout));
    return Layout(input_layout);
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
                            GetSignatureOutputNames(signature_index));
    int num_tensors = output_names.size();
    std::vector<LiteRtLayout> litert_layout_vector(num_tensors);
    LITERT_RETURN_IF_ERROR(env_.runtime->GetCompiledModelOutputTensorLayouts(
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
    LITERT_RETURN_IF_ERROR(
        env_.runtime->GetCompiledModelOutputBufferRequirements(
            Get(), signature_index, output_index, &buffer_requirements));
    return internal::compiled_model_detail::ToTensorBufferRequirements(
        env_, buffer_requirements);
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
                            GetSignatureIndex(signature_name));
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
                            GetSignatureIndex(signature_name));
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

  /// @brief Sets model-level default scheduling info.
  Expected<void> SetSchedulingInfo(
      const LiteRtSchedulingInfo& scheduling_info) const {
    auto status =
        env_.runtime->CompiledModelSetSchedulingInfo(Get(), &scheduling_info);
    if (status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to set scheduling info");
    }
    return {};
  }

  /// @brief Clears model-level default scheduling info.
  Expected<void> ClearSchedulingInfo() const {
    auto status = env_.runtime->CompiledModelSetSchedulingInfo(Get(), nullptr);
    if (status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to clear scheduling info");
    }
    return {};
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

  /// @brief Runs with per-run options for a given signature index.
  Expected<void> Run(size_t signature_index,
                     absl::Span<const TensorBuffer> input_buffers,
                     absl::Span<const TensorBuffer> output_buffers,
                     Options* run_options) const {
    LiteRtOptions options_handle = nullptr;
    internal::LiteRtOptionsPtr owned_options;

    if (run_options) {
      LITERT_ASSIGN_OR_RETURN(owned_options, BuildOptions(*run_options, env_));
      options_handle = owned_options.get();
    }
    bool async = false;
    return RunHelper(signature_index, input_buffers, output_buffers, async,
                     options_handle, nullptr);
  }

  /// @brief Runs with per-request scheduling info for a given signature index.
  Expected<void> Run(size_t signature_index,
                     absl::Span<const TensorBuffer> input_buffers,
                     absl::Span<const TensorBuffer> output_buffers,
                     const LiteRtSchedulingInfo& scheduling_info) const {
    bool async = false;
    return RunHelper(signature_index, input_buffers, output_buffers, async,
                     nullptr, &scheduling_info);
  }

  /// @brief Runs default signature with per-run options.
  Expected<void> Run(absl::Span<const TensorBuffer> input_buffers,
                     absl::Span<const TensorBuffer> output_buffers,
                     Options* run_options) const {
    LiteRtOptions options_handle = nullptr;
    internal::LiteRtOptionsPtr owned_options;

    if (run_options) {
      LITERT_ASSIGN_OR_RETURN(owned_options,
                              BuildOptions(std::move(*run_options), env_));
      options_handle = owned_options.get();
    }
    bool async = false;
    return RunHelper(/*signature_index=*/0, input_buffers, output_buffers,
                     async, options_handle, nullptr);
  }

  /// @brief Runs default signature with per-request scheduling info.
  Expected<void> Run(absl::Span<const TensorBuffer> input_buffers,
                     absl::Span<const TensorBuffer> output_buffers,
                     const LiteRtSchedulingInfo& scheduling_info) const {
    bool async = false;
    return RunHelper(/*signature_index=*/0, input_buffers, output_buffers,
                     async, nullptr, &scheduling_info);
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

  /// @brief Runs asynchronously with per-run options for a given signature.
  Expected<void> RunAsync(size_t signature_index,
                          const std::vector<TensorBuffer>& input_buffers,
                          const std::vector<TensorBuffer>& output_buffers,
                          bool& async, Options* run_options) const {
    LiteRtOptions options_handle = nullptr;
    internal::LiteRtOptionsPtr owned_options;

    if (run_options) {
      LITERT_ASSIGN_OR_RETURN(owned_options, BuildOptions(*run_options, env_));
      options_handle = owned_options.get();
    }
    async = true;
    return RunHelper(signature_index, input_buffers, output_buffers, async,
                     options_handle, nullptr);
  }

  /// @brief Runs asynchronously with per-request scheduling info for a given
  /// signature.
  Expected<void> RunAsync(size_t signature_index,
                          const std::vector<TensorBuffer>& input_buffers,
                          const std::vector<TensorBuffer>& output_buffers,
                          bool& async,
                          const LiteRtSchedulingInfo& scheduling_info) const {
    async = true;
    return RunHelper(signature_index, input_buffers, output_buffers, async,
                     nullptr, &scheduling_info);
  }

  /// @brief Runs default signature asynchronously with per-run options.
  Expected<void> RunAsync(const std::vector<TensorBuffer>& input_buffers,
                          const std::vector<TensorBuffer>& output_buffers,
                          bool& async, Options* run_options) const {
    LiteRtOptions options_handle = nullptr;
    internal::LiteRtOptionsPtr owned_options;

    if (run_options) {
      LITERT_ASSIGN_OR_RETURN(owned_options, BuildOptions(*run_options, env_));
      options_handle = owned_options.get();
    }
    async = true;
    return RunHelper(/*signature_index=*/0, input_buffers, output_buffers,
                     async, options_handle, nullptr);
  }

  /// @brief Runs default signature asynchronously with per-request scheduling
  /// info.
  Expected<void> RunAsync(const std::vector<TensorBuffer>& input_buffers,
                          const std::vector<TensorBuffer>& output_buffers,
                          bool& async,
                          const LiteRtSchedulingInfo& scheduling_info) const {
    async = true;
    return RunHelper(/*signature_index=*/0, input_buffers, output_buffers,
                     async, nullptr, &scheduling_info);
  }

  /// @brief Runs the model for a given signature key synchronously with the
  /// provided input/output `TensorBuffer`s.
  Expected<void> Run(absl::string_view signature_key,
                     const std::vector<TensorBuffer>& input_buffers,
                     const std::vector<TensorBuffer>& output_buffers) const {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            GetSignatureIndex(signature_key));
    return Run(signature_index, input_buffers, output_buffers);
  }

  /// @brief Runs by signature key with per-run options.
  Expected<void> Run(absl::string_view signature_key,
                     const std::vector<TensorBuffer>& input_buffers,
                     const std::vector<TensorBuffer>& output_buffers,
                     Options* run_options) const {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            GetSignatureIndex(signature_key));
    return Run(signature_index, input_buffers, output_buffers, run_options);
  }

  /// @brief Runs by signature key with per-request scheduling info.
  Expected<void> Run(absl::string_view signature_key,
                     const std::vector<TensorBuffer>& input_buffers,
                     const std::vector<TensorBuffer>& output_buffers,
                     const LiteRtSchedulingInfo& scheduling_info) const {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            GetSignatureIndex(signature_key));
    return Run(signature_index, input_buffers, output_buffers, scheduling_info);
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
                            GetSignatureIndex(signature_key));
    return RunAsync(signature_index, input_buffers, output_buffers, async);
  }

  /// @brief Runs by signature key asynchronously with per-run options.
  Expected<void> RunAsync(absl::string_view signature_key,
                          const std::vector<TensorBuffer>& input_buffers,
                          const std::vector<TensorBuffer>& output_buffers,
                          bool& async, Options* run_options) const {
    async = true;
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            GetSignatureIndex(signature_key));
    return RunAsync(signature_index, input_buffers, output_buffers, async,
                    run_options);
  }

  /// @brief Runs by signature key asynchronously with per-request scheduling
  /// info.
  Expected<void> RunAsync(absl::string_view signature_key,
                          const std::vector<TensorBuffer>& input_buffers,
                          const std::vector<TensorBuffer>& output_buffers,
                          bool& async,
                          const LiteRtSchedulingInfo& scheduling_info) const {
    async = true;
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            GetSignatureIndex(signature_key));
    return RunAsync(signature_index, input_buffers, output_buffers, async,
                    scheduling_info);
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

  /// @brief Runs by signature key with per-run options using named maps.
  Expected<void> Run(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      Options* run_options) const {
    LiteRtOptions options_handle = nullptr;
    internal::LiteRtOptionsPtr owned_options;

    if (run_options) {
      LITERT_ASSIGN_OR_RETURN(owned_options, BuildOptions(*run_options, env_));
      options_handle = owned_options.get();
    }
    bool async = false;
    return RunMapHelper(signature_key, input_map, output_map, async,
                        options_handle, nullptr);
  }

  /// @brief Runs by signature key with per-request scheduling info using named
  /// maps.
  Expected<void> Run(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      const LiteRtSchedulingInfo& scheduling_info) const {
    bool async = false;
    return RunMapHelper(signature_key, input_map, output_map, async, nullptr,
                        &scheduling_info);
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

  /// @brief Runs default signature with per-run options using named maps.
  Expected<void> Run(
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      Options* run_options) const {
    LiteRtOptions options_handle = nullptr;
    internal::LiteRtOptionsPtr owned_options;

    if (run_options) {
      LITERT_ASSIGN_OR_RETURN(owned_options, BuildOptions(*run_options, env_));
      options_handle = owned_options.get();
    }
    bool async = false;
    return RunMapWithIndexHelper(
        /*signature_index=*/0, input_map, output_map, async, options_handle,
        nullptr);
  }

  /// @brief Runs default signature with per-request scheduling info using
  /// named maps.
  Expected<void> Run(
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      const LiteRtSchedulingInfo& scheduling_info) const {
    bool async = false;
    return RunMapWithIndexHelper(/*signature_index=*/0, input_map, output_map,
                                 async, nullptr, &scheduling_info);
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

  /// @brief Runs by signature key asynchronously with per-run options using
  /// named maps.
  Expected<void> RunAsync(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async, Options* run_options) const {
    LiteRtOptions options_handle = nullptr;
    internal::LiteRtOptionsPtr owned_options;

    if (run_options) {
      LITERT_ASSIGN_OR_RETURN(owned_options, BuildOptions(*run_options, env_));
      options_handle = owned_options.get();
    }
    async = true;
    return RunMapHelper(signature_key, input_map, output_map, async,
                        options_handle, nullptr);
  }

  /// @brief Runs by signature key asynchronously with per-request scheduling
  /// info using named maps.
  Expected<void> RunAsync(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async, const LiteRtSchedulingInfo& scheduling_info) const {
    async = true;
    return RunMapHelper(signature_key, input_map, output_map, async, nullptr,
                        &scheduling_info);
  }

  /// @brief Returns `true` if the compiled model is fully accelerated with the
  /// given hardware accelerators.
  Expected<bool> IsFullyAccelerated() {
    bool fully_accelerated = false;
    LITERT_RETURN_IF_ERROR(env_.runtime->CompiledModelIsFullyAccelerated(
        Get(), &fully_accelerated));
    return fully_accelerated;
  }

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
    env_.runtime->SetCompiledModelCancellationFunction(Get(), data,
                                                       check_cancelled_func);
  }

  /// @brief Sets a callback function for checking cancellation during
  /// execution.
  ///
  /// The callback will be called periodically during model execution. This is a
  /// C++-friendly version of `SetCancellationFunction`.
  void SetCancellationFunction(
      absl::AnyInvocable<bool()> check_cancelled_func) {
    check_cancelled_func_ = std::move(check_cancelled_func);
    env_.runtime->SetCompiledModelCancellationFunction(Get(), this,
                                                       &CheckCancelledWrapper);
  }

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
    LITERT_RETURN_IF_ERROR(env_.runtime->CompiledModelResizeInputTensor(
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
                            GetSignatureIndex(signature_name));
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
    LITERT_RETURN_IF_ERROR(
        env_.runtime->CompiledModelResizeInputTensorNonStrict(
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
                            GetSignatureIndex(signature_name));
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

  /// @brief Sets a dispatch annotation on the compiled model.
  ///
  /// These annotations are propagated to dispatch graphs during model
  /// execution and provide runtime hints and metadata for hardware accelerator
  /// optimization.
  Expected<void> SetDispatchAnnotation(size_t signature_index,
                                       absl::string_view key,
                                       absl::string_view value) {
    const std::string key_string(key);
    const std::string value_string(value);
    LITERT_RETURN_IF_ERROR(env_.runtime->CompiledModelSetDispatchAnnotation(
        Get(), signature_index, key_string.c_str(), value_string.c_str()));
    return {};
  }

  /// @brief Gets a dispatch annotation from the compiled model.
  ///
  /// Returns `std::nullopt` if the key does not exist.
  Expected<std::optional<std::string>> GetDispatchAnnotation(
      size_t signature_index, absl::string_view key) {
    const std::string key_string(key);
    const char* value = nullptr;
    LITERT_RETURN_IF_ERROR(env_.runtime->CompiledModelGetDispatchAnnotation(
        Get(), signature_index, key_string.c_str(), &value));
    if (value == nullptr) {
      return std::optional<std::string>();
    }
    return std::optional<std::string>(std::string(value));
  }

  /// @brief Removes a dispatch annotation from the compiled model.
  ///
  /// This succeeds even if the key does not exist.
  Expected<void> RemoveDispatchAnnotation(size_t signature_index,
                                          absl::string_view key) {
    const std::string key_string(key);
    LITERT_RETURN_IF_ERROR(env_.runtime->CompiledModelRemoveDispatchAnnotation(
        Get(), signature_index, key_string.c_str()));
    return {};
  }

  /// @brief Overloaded version for the default signature (index 0).
  Expected<void> SetDispatchAnnotation(absl::string_view key,
                                       absl::string_view value) {
    return SetDispatchAnnotation(0, key, value);
  }

  /// @brief Overloaded version for the default signature (index 0).
  Expected<std::optional<std::string>> GetDispatchAnnotation(
      absl::string_view key) {
    return GetDispatchAnnotation(0, key);
  }

  /// @brief Overloaded version for the default signature (index 0).
  Expected<void> RemoveDispatchAnnotation(absl::string_view key) {
    return RemoveDispatchAnnotation(0, key);
  }

  /// @brief Overloaded version that takes a signature name instead of an
  /// index.
  Expected<void> SetDispatchAnnotation(absl::string_view signature_name,
                                       absl::string_view key,
                                       absl::string_view value) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            GetSignatureIndex(signature_name));
    return SetDispatchAnnotation(signature_index, key, value);
  }

  /// @brief Overloaded version that takes a signature name instead of an
  /// index.
  Expected<std::optional<std::string>> GetDispatchAnnotation(
      absl::string_view signature_name, absl::string_view key) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            GetSignatureIndex(signature_name));
    return GetDispatchAnnotation(signature_index, key);
  }

  /// @brief Overloaded version that takes a signature name instead of an
  /// index.
  Expected<void> RemoveDispatchAnnotation(absl::string_view signature_name,
                                          absl::string_view key) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            GetSignatureIndex(signature_name));
    return RemoveDispatchAnnotation(signature_index, key);
  }

  /// @brief Reports an error to the compiled model's error reporter.
  ///
  /// Supports printf-style formatting for error messages.
  template <typename... Args>
  Expected<void> ReportError(const char* format, Args&&... args) const {
    LITERT_RETURN_IF_ERROR(env_.runtime->CompiledModelReportError(
        Get(), format, std::forward<Args>(args)...));
    return {};
  }

  /// @brief Clears all errors from the error reporter.
  /// @note This only works if the compiled model uses `BufferErrorReporter`,
  /// not `StderrReporter`.
  Expected<void> ClearErrors() const {
    LITERT_RETURN_IF_ERROR(env_.runtime->CompiledModelClearErrors(Get()));
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
        env_.runtime->CompiledModelGetErrorMessages(Get(), &error_messages));

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

  /// @brief Returns the default signature key of the model.
  static absl::string_view DefaultSignatureKey() {
    return kDefaultSignatureKey;
  }

  /// @brief Returns the number of signatures defined in the model.
  size_t GetNumSignatures() const {
    LiteRtParamIndex num_signatures;
    LITERT_RETURN_IF_ERROR(
        env_.runtime->GetNumModelSignatures(model_.Get(), &num_signatures));
    return num_signatures;
  }

  /// @brief Returns the list of signature key names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureKeys() const {
    return GetSignatureKeysImpl();
  }

  /// @brief Returns the list of signatures defined in the model.
  Expected<std::vector<SimpleSignature>> GetSignatures() const {
    auto num_signatures = GetNumSignatures();
    std::vector<SimpleSignature> signatures;
    signatures.reserve(num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LITERT_ASSIGN_OR_RETURN(auto signature, GetSignature(i));
      signatures.push_back(std::move(signature));
    }
    return std::move(signatures);
  }

  /// @brief Returns the signature at the given index.
  Expected<SimpleSignature> GetSignature(size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    LITERT_RETURN_IF_ERROR(env_.runtime->GetModelSignature(
        model_.Get(), signature_index, &lite_rt_signature));
    return SimpleSignature(
        internal::compiled_model_detail::FetchSignatureKey(env_,
                                                           lite_rt_signature),
        internal::compiled_model_detail::FetchSignatureInputNames(
            env_, lite_rt_signature),
        internal::compiled_model_detail::FetchSignatureInputTensors(
            env_, lite_rt_signature),
        internal::compiled_model_detail::FetchSignatureOutputNames(
            env_, lite_rt_signature),
        internal::compiled_model_detail::FetchSignatureOutputTensors(
            env_, lite_rt_signature));
  }

  /// @brief Returns the signature index for a given signature key.
  ///
  /// Returns 0 if the signature key is empty.
  Expected<size_t> GetSignatureIndex(absl::string_view signature_key) const {
    if (signature_key.empty()) {
      return 0;
    }
    auto num_signatures = GetNumSignatures();
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      LITERT_RETURN_IF_ERROR(
          env_.runtime->GetModelSignature(model_.Get(), i, &lite_rt_signature));
      auto key = internal::compiled_model_detail::FetchSignatureKey(
          env_, lite_rt_signature);
      if (key == signature_key) {
        return i;
      }
    }
    return Unexpected(Status::kErrorNotFound, "Signature not found");
  }

  Expected<SimpleSignature> FindSignature(
      absl::string_view signature_key) const {
    LITERT_ASSIGN_OR_RETURN(auto index, GetSignatureIndex(signature_key));
    return GetSignature(index);
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      size_t signature_index) const {
    return GetSignatureInputNamesImpl(signature_index);
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames() const {
    return GetSignatureInputNames(/*signature_index=*/0);
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      absl::string_view signature_key) const {
    auto signature = FindSignature(signature_key);
    if (!signature) {
      return Unexpected(Status::kErrorNotFound, "Signature not found");
    }
    return signature->InputNames();
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      size_t signature_index) const {
    return GetSignatureOutputNamesImpl(signature_index);
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames() const {
    return GetSignatureOutputNames(/*signature_index=*/0);
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      absl::string_view signature_key) const {
    auto signature = FindSignature(signature_key);
    if (!signature) {
      return Unexpected(Status::kErrorNotFound, "Signature not found");
    }
    return signature->OutputNames();
  }

  /// @brief Returns the tensor type for the n-th input tensor.
  Expected<RankedTensorType> GetInputTensorType(size_t signature_index,
                                                size_t input_index) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            GetSignature(signature_index));
    return signature.InputTensorType(input_index);
  }

  /// @brief Returns the tensor type for a given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      size_t signature_index, absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            GetSignature(signature_index));
    return signature.InputTensorType(input_name);
  }

  /// @brief Returns the tensor type for a given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      absl::string_view signature_key, absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            FindSignature(signature_key));
    return signature.InputTensorType(input_name);
  }

  /// @brief Gets the input tensor type of the default signature for a given
  /// input name.
  Expected<RankedTensorType> GetInputTensorType(
      absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            GetSignature(/*signature_index=*/0));
    return signature.InputTensorType(input_name);
  }

  /// @brief Returns the tensor type for the n-th output tensor.
  Expected<RankedTensorType> GetOutputTensorType(size_t signature_index,
                                                 size_t output_index) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            GetSignature(signature_index));
    return signature.OutputTensorType(output_index);
  }

  /// @brief Returns the tensor type for a given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      size_t signature_index, absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            GetSignature(signature_index));
    return signature.OutputTensorType(output_name);
  }

  /// @brief Returns the tensor type for a given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view signature_key, absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            FindSignature(signature_key));
    return signature.OutputTensorType(output_name);
  }

  /// @brief Gets the output tensor type of the default signature for a given
  /// output name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            GetSignature(/*signature_index=*/0));
    return signature.OutputTensorType(output_name);
  }

 protected:
  /// @internal
  /// @brief Creates a `CompiledModel` from a provided `LiteRtModel`.
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
                                        const LiteRtModel litert_model,
                                        Options& compilation_options) {
    auto env_holder = env.GetHolder();
    LITERT_ASSIGN_OR_RETURN(
        auto owned_options,
        BuildOptions(std::move(compilation_options), env_holder));
    LiteRtCompiledModel compiled_model;
    LITERT_RETURN_IF_ERROR(env_holder.runtime->CreateCompiledModel(
        env_holder.handle, litert_model, owned_options.get(), &compiled_model));
    return CompiledModel(env_holder, litert_model,
                         /*model_owned=*/OwnHandle::kNo, compiled_model,
                         /*owned=*/OwnHandle::kYes, std::move(owned_options));
  }

  /// @internal
  /// @brief A simplified version of `Create` that uses default compilation
  /// options.
  ///
  /// The provided hardware accelerator is used to select the target
  /// accelerator.
  /// @note This should be specified for both JIT and AOT compiled models.
  static Expected<CompiledModel> Create(
      litert::Environment& env, const LiteRtModel litert_model,
      litert::HwAccelerators hardware_accelerators) {
    Options compilation_options;
    compilation_options.SetHardwareAccelerators(hardware_accelerators);
    return Create(env, litert_model, compilation_options);
  }

  /// @internal
  /// @brief Builds the options object and returns an owning handle.
  static Expected<internal::LiteRtOptionsPtr> BuildOptions(
      const Options& options, const internal::EnvironmentHolder& env) {
    LITERT_ASSIGN_OR_RETURN(
        auto options_handle,
        internal::LiteRtOptionsPtrBuilder::Build(options, env));
    return std::move(options_handle);
  }

  /// @brief Constructs a `CompiledModel` instance.
  ///
  /// @param model_owned Indicates whether the provided `litert_model` handle is
  /// owned by the `CompiledModel`.
  /// @param owned If `true`, the created object takes ownership of the
  /// `compiled_model` handle.
  explicit CompiledModel(internal::EnvironmentHolder& env,
                         LiteRtModel litert_model, OwnHandle model_owned,
                         LiteRtCompiledModel compiled_model, OwnHandle owned,
                         internal::LiteRtOptionsPtr options = {})
      : internal::BaseHandle<LiteRtCompiledModel>(
            compiled_model,
            [runtime = env.runtime](LiteRtCompiledModel compiled_model) {
              runtime->DestroyCompiledModel(compiled_model);
            },
            owned),
        env_(env),
        options_(std::move(options)) {
    if (model_owned == OwnHandle::kYes) {
      model_ = internal::BaseHandle<LiteRtModel>(
          litert_model,
          [runtime = env.runtime](LiteRtModel model) {
            runtime->DestroyModel(model);
          },
          OwnHandle::kYes);
    } else {
      model_ = internal::NonOwnedHandle<LiteRtModel>(litert_model);
    }
  }

  static bool CheckCancelledWrapper(void* data) {
    CompiledModel* model = static_cast<CompiledModel*>(data);
    if (model && model->check_cancelled_func_) {
      return model->check_cancelled_func_();
    }
    return false;
  }

  /// @brief Returns the signature input index for a given input tensor name.
  Expected<size_t> FindInputIndex(size_t signature_index,
                                  absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(const auto input_names,
                            GetSignatureInputNames(signature_index));
    auto it = absl::c_find(input_names, input_name);
    if (it != input_names.end()) {
      return std::distance(input_names.begin(), it);
    }
    return Unexpected(Status::kErrorNotFound, "Failed to find input");
  }

  /// @brief Returns the signature output index for a given output tensor
  /// name.
  Expected<size_t> FindOutputIndex(size_t signature_index,
                                   absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(const auto output_names,
                            GetSignatureOutputNames(signature_index));
    auto it = absl::c_find(output_names, output_name);
    if (it != output_names.end()) {
      return std::distance(output_names.begin(), it);
    }
    return Unexpected(Status::kErrorNotFound, "Failed to find output");
  }

  /// @brief Creates a `TensorBuffer` with the given buffer requirements and
  /// tensor type.
  static Expected<TensorBuffer> CreateBufferImpl(
      const Environment& env,
      const TensorBufferRequirements& buffer_requirements,
      const RankedTensorType& tensor_type) {
    return TensorBuffer::CreateManagedFromRequirements(env, tensor_type,
                                                       buffer_requirements);
  }

  /// @brief Creates a `TensorBuffer` for a given signature index and tensor
  /// name.
  Expected<TensorBuffer> CreateInputOutputBuffer(size_t signature_index,
                                                 absl::string_view tensor_name,
                                                 bool is_input) const {
    Expected<RankedTensorType> tensor_type_expected =
        is_input ? GetInputTensorType(signature_index, tensor_name)
                 : GetOutputTensorType(signature_index, tensor_name);
    LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, tensor_type_expected);
    LITERT_ASSIGN_OR_RETURN(auto env, GetEnvironment());
    if (is_input) {
      LITERT_ASSIGN_OR_RETURN(
          TensorBufferRequirements buffer_requirements,
          GetInputBufferRequirements(signature_index, tensor_name));
      LITERT_ASSIGN_OR_RETURN(size_t tensor_index,
                              FindInputIndex(signature_index, tensor_name));
      LiteRtLayout input_layout;
      if (env_.runtime->GetCompiledModelInputTensorLayout(
              Get(), signature_index, tensor_index, &input_layout) ==
          kLiteRtStatusOk) {
        Layout runtime_layout(input_layout);
        tensor_type = RankedTensorType(tensor_type.ElementType(),
                                       std::move(runtime_layout));
      }
      return CreateBufferImpl(env, buffer_requirements, tensor_type);
    } else {
      LITERT_ASSIGN_OR_RETURN(size_t tensor_index,
                              FindOutputIndex(signature_index, tensor_name));
      LITERT_ASSIGN_OR_RETURN(
          std::vector<Layout> runtime_layouts,
          GetOutputTensorLayouts(signature_index, /*update_allocation=*/true));
      tensor_type = RankedTensorType(tensor_type.ElementType(),
                                     std::move(runtime_layouts[tensor_index]));
      LITERT_ASSIGN_OR_RETURN(
          const TensorBufferRequirements& requirements,
          GetOutputBufferRequirements(signature_index, tensor_name));
      return CreateBufferImpl(env, requirements, tensor_type);
    }
  }

  /// @brief Creates a `TensorBuffer` for a given signature and tensor name.
  Expected<TensorBuffer> CreateInputOutputBuffer(
      absl::string_view signature_name, absl::string_view tensor_name,
      bool is_input) const {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            GetSignatureIndex(signature_name));
    return CreateInputOutputBuffer(signature_index, tensor_name, is_input);
  }

  /// @brief Creates a vector of `TensorBuffer`s for a given signature
  /// subgraph.
  Expected<std::vector<TensorBuffer>> CreateInputOutputBuffers(
      size_t signature_index, bool is_input) const {
    std::vector<TensorBuffer> tensor_buffers;
    Expected<std::vector<absl::string_view>> tensor_names;
    tensor_names = is_input ? GetSignatureInputNames(signature_index)
                            : GetSignatureOutputNames(signature_index);
    if (!tensor_names) {
      return tensor_names.Error();
    }
    tensor_buffers.reserve(tensor_names->size());

    for (int i = 0; i < tensor_names->size(); ++i) {
      LITERT_ASSIGN_OR_RETURN(
          TensorBuffer tensor_buffer,
          CreateInputOutputBuffer(signature_index, tensor_names->at(i),
                                  is_input));
      tensor_buffers.push_back(std::move(tensor_buffer));
    }

    return tensor_buffers;
  }

  /// @internal
  /// @brief Returns the environment used to create this compiled model.
  ///
  /// The returned `Environment` does not own the underlying
  /// `LiteRtEnvironment`.
  Expected<Environment> GetEnvironment() const {
    return Environment::WrapCObject(env_, OwnHandle::kNo);
  }

  // Compatibility overload that routes to the richer helper with default args.
  Expected<void> RunCApiHelper(LiteRtParamIndex signature_index,
                               size_t num_input_buffers,
                               LiteRtTensorBuffer* input_buffers,
                               size_t num_output_buffers,
                               LiteRtTensorBuffer* output_buffers,
                               bool& async) const {
    return RunCApiHelper(signature_index, num_input_buffers, input_buffers,
                         num_output_buffers, output_buffers, async,
                         /*run_options=*/nullptr,
                         /*scheduling_info=*/nullptr);
  }

  // Compatibility overload that routes to the richer helper with default args.
  Expected<void> RunCApiHelper(LiteRtParamIndex signature_index,
                               size_t num_input_buffers,
                               LiteRtTensorBuffer* input_buffers,
                               size_t num_output_buffers,
                               LiteRtTensorBuffer* output_buffers, bool& async,
                               LiteRtOptions run_options) const {
    return RunCApiHelper(signature_index, num_input_buffers, input_buffers,
                         num_output_buffers, output_buffers, async, run_options,
                         /*scheduling_info=*/nullptr);
  }

  Expected<void> RunCApiHelper(
      LiteRtParamIndex signature_index, size_t num_input_buffers,
      LiteRtTensorBuffer* input_buffers, size_t num_output_buffers,
      LiteRtTensorBuffer* output_buffers, bool& async,
      LiteRtOptions run_options,
      const LiteRtSchedulingInfo* scheduling_info) const {
    if (run_options != nullptr && scheduling_info != nullptr) {
      return Unexpected(
          Status::kErrorInvalidArgument,
          "Run options and scheduling info are mutually exclusive");
    }

    LiteRtStatus status;
    if (scheduling_info != nullptr) {
      status =
          async
              ? env_.runtime->RunCompiledModelAsyncWithSchedulingInfo(
                    Get(), signature_index, num_input_buffers, input_buffers,
                    num_output_buffers, output_buffers, &async, scheduling_info)
              : env_.runtime->RunCompiledModelWithSchedulingInfo(
                    Get(), signature_index, num_input_buffers, input_buffers,
                    num_output_buffers, output_buffers, scheduling_info);
    } else if (run_options != nullptr) {
      status =
          async ? env_.runtime->RunCompiledModelAsyncWithOptions(
                      Get(), signature_index, num_input_buffers, input_buffers,
                      num_output_buffers, output_buffers, &async, run_options)
                : env_.runtime->RunCompiledModelWithOptions(
                      Get(), signature_index, num_input_buffers, input_buffers,
                      num_output_buffers, output_buffers, run_options);
    } else {
      status =
          async ? env_.runtime->RunCompiledModelAsync(
                      Get(), signature_index, num_input_buffers, input_buffers,
                      num_output_buffers, output_buffers, &async)
                : env_.runtime->RunCompiledModel(
                      Get(), signature_index, num_input_buffers, input_buffers,
                      num_output_buffers, output_buffers);
    }
    if (status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status),
                        "Failed to invoke the compiled model");
    }
    return {};
  }

  // Compatibility overload that routes to the richer helper with default args.
  Expected<void> RunHelper(size_t signature_index,
                           absl::Span<const TensorBuffer> input_buffers,
                           absl::Span<const TensorBuffer> output_buffers,
                           bool& async) const {
    return RunHelper(signature_index, input_buffers, output_buffers, async,
                     /*run_options=*/nullptr,
                     /*scheduling_info=*/nullptr);
  }

  // Compatibility overload that routes to the richer helper with default args.
  Expected<void> RunHelper(size_t signature_index,
                           absl::Span<const TensorBuffer> input_buffers,
                           absl::Span<const TensorBuffer> output_buffers,
                           bool& async, LiteRtOptions run_options) const {
    return RunHelper(signature_index, input_buffers, output_buffers, async,
                     run_options,
                     /*scheduling_info=*/nullptr);
  }

  Expected<void> RunHelper(size_t signature_index,
                           absl::Span<const TensorBuffer> input_buffers,
                           absl::Span<const TensorBuffer> output_buffers,
                           bool& async, LiteRtOptions run_options,
                           const LiteRtSchedulingInfo* scheduling_info) const {
    auto input_buffers_ptr =
        std::make_unique<LiteRtTensorBuffer[]>(input_buffers.size());
    for (int i = 0; i < input_buffers.size(); ++i) {
      input_buffers_ptr[i] = input_buffers[i].Get();
    }
    auto output_buffers_ptr =
        std::make_unique<LiteRtTensorBuffer[]>(output_buffers.size());
    for (int i = 0; i < output_buffers.size(); ++i) {
      output_buffers_ptr[i] = output_buffers[i].Get();
    }
    return RunCApiHelper(signature_index, input_buffers.size(),
                         input_buffers_ptr.get(), output_buffers.size(),
                         output_buffers_ptr.get(), async, run_options,
                         scheduling_info);
  }

  // Compatibility overload that routes to the richer helper with default args.
  Expected<void> RunMapHelper(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async) const {
    return RunMapHelper(signature_key, input_map, output_map, async,
                        /*run_options=*/nullptr,
                        /*scheduling_info=*/nullptr);
  }

  // Compatibility overload that routes to the richer helper with default args.
  Expected<void> RunMapHelper(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async, LiteRtOptions run_options) const {
    auto signature_index = GetSignatureIndex(signature_key);
    return RunMapHelper(signature_key, input_map, output_map, async,
                        run_options,
                        /*scheduling_info=*/nullptr);
  }

  Expected<void> RunMapHelper(
      absl::string_view signature_key,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async, LiteRtOptions run_options,
      const LiteRtSchedulingInfo* scheduling_info) const {
    auto signature_index = GetSignatureIndex(signature_key);
    if (!signature_index) {
      return Unexpected(Status::kErrorNotFound,
                        "Failed to get signature_index");
    }
    return RunMapWithIndexHelper(*signature_index, input_map, output_map, async,
                                 run_options, scheduling_info);
  }

  // Compatibility overload that routes to the richer helper with default args.
  Expected<void> RunMapWithIndexHelper(
      size_t signature_index,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async) const {
    return RunMapWithIndexHelper(signature_index, input_map, output_map, async,
                                 /*run_options=*/nullptr,
                                 /*scheduling_info=*/nullptr);
  }

  // Compatibility overload that routes to the richer helper with default args.
  Expected<void> RunMapWithIndexHelper(
      size_t signature_index,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async, LiteRtOptions run_options) const {
    return RunMapWithIndexHelper(signature_index, input_map, output_map, async,
                                 run_options,
                                 /*scheduling_info=*/nullptr);
  }

  Expected<void> RunMapWithIndexHelper(
      size_t signature_index,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
      const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
      bool& async, LiteRtOptions run_options,
      const LiteRtSchedulingInfo* scheduling_info) const {
    LITERT_ASSIGN_OR_RETURN(auto input_names,
                            GetSignatureInputNames(signature_index));
    size_t num_inputs = input_names.size();
    auto input_buffers_ptr = std::make_unique<LiteRtTensorBuffer[]>(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      absl::string_view input_name = input_names[i];
      auto it = input_map.find(input_name);
      if (it == input_map.end()) {
        input_buffers_ptr[i] = nullptr;
        continue;
      }
      input_buffers_ptr[i] = it->second.Get();
    }
    LITERT_ASSIGN_OR_RETURN(auto output_names,
                            GetSignatureOutputNames(signature_index));
    size_t num_outputs = output_names.size();
    auto output_buffers_ptr =
        std::make_unique<LiteRtTensorBuffer[]>(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      absl::string_view output_name = output_names[i];
      auto it = output_map.find(output_name);
      if (it == output_map.end()) {
        return Unexpected(Status::kErrorNotFound,
                          "The given map is missing some output TensorBuffers");
      }
      output_buffers_ptr[i] = it->second.Get();
    }
    return RunCApiHelper(signature_index, num_inputs, input_buffers_ptr.get(),
                         num_outputs, output_buffers_ptr.get(), async,
                         run_options, scheduling_info);
  }

  Expected<std::vector<absl::string_view>> GetSignatureKeysImpl() const {
    size_t num_signatures = GetNumSignatures();
    std::vector<absl::string_view> signature_keys;
    signature_keys.reserve(num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      LITERT_RETURN_IF_ERROR(
          env_.runtime->GetModelSignature(model_.Get(), i, &lite_rt_signature));
      signature_keys.push_back(
          internal::compiled_model_detail::FetchSignatureKey(
              env_, lite_rt_signature));
    }
    return signature_keys;
  }
  Expected<std::vector<absl::string_view>> GetSignatureInputNamesImpl(
      size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    LITERT_RETURN_IF_ERROR(env_.runtime->GetModelSignature(
        model_.Get(), signature_index, &lite_rt_signature));
    return internal::compiled_model_detail::FetchSignatureInputNames(
        env_, lite_rt_signature);
  }
  Expected<std::vector<absl::string_view>> GetSignatureOutputNamesImpl(
      size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    LITERT_RETURN_IF_ERROR(env_.runtime->GetModelSignature(
        model_.Get(), signature_index, &lite_rt_signature));
    return internal::compiled_model_detail::FetchSignatureOutputNames(
        env_, lite_rt_signature);
  }

  internal::EnvironmentHolder env_;
  internal::LiteRtOptionsPtr options_;
  absl::AnyInvocable<bool()> check_cancelled_func_;
  internal::BaseHandle<LiteRtModel> model_;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_COMPILED_MODEL_H_
