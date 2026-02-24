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

#include "litert/cc/litert_compiled_model.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"

namespace litert {

namespace {

/// @brief Converts a `LiteRtTensorBufferRequirements` C object to a
/// `TensorBufferRequirements` C++ object.
Expected<TensorBufferRequirements> ToTensorBufferRequirements(
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

  int num_strides;
  const uint32_t* strides_ptr;
  LITERT_RETURN_IF_ERROR(env.runtime->GetTensorBufferRequirementsStrides(
      litert_requirements, &num_strides, &strides_ptr));
  std::vector<uint32_t> strides;
  if (num_strides > 0 && strides_ptr != nullptr) {
    strides.assign(strides_ptr, strides_ptr + num_strides);
  }

  size_t alignment;
  LITERT_RETURN_IF_ERROR(env.runtime->GetTensorBufferRequirementsAlignment(
      litert_requirements, &alignment));

  return TensorBufferRequirements::CreateWithAlignment(
      absl::MakeConstSpan(supported_types), buffer_size, alignment,
      absl::MakeConstSpan(strides));
}

absl::string_view FetchTensorName(const internal::EnvironmentHolder& env,
                                  LiteRtTensor tensor) {
  const char* name;
  LITERT_ABORT_IF_ERROR(env.runtime->GetTensorName(tensor, &name));
  return name;
}

std::uint32_t FetchTensorIndex(const internal::EnvironmentHolder& env,
                               LiteRtTensor tensor) {
  std::uint32_t index;
  LITERT_ABORT_IF_ERROR(env.runtime->GetTensorIndex(tensor, &index));
  return index;
}

LiteRtTensorTypeId FetchTensorTypeId(const internal::EnvironmentHolder& env,
                                     LiteRtTensor tensor) {
  LiteRtTensorTypeId type_id;
  LITERT_ABORT_IF_ERROR(env.runtime->GetTensorTypeId(tensor, &type_id));
  return type_id;
}

std::variant<LiteRtUnrankedTensorType, litert::RankedTensorType>
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

absl::string_view FetchSignatureKey(const internal::EnvironmentHolder& env,
                                    LiteRtSignature signature) {
  const char* key;
  LITERT_ABORT_IF_ERROR(env.runtime->GetSignatureKey(signature, &key));
  return key;
}

std::vector<absl::string_view> FetchSignatureInputNames(
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

std::vector<absl::string_view> FetchSignatureOutputNames(
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

std::vector<std::unique_ptr<SimpleTensor>> FetchSignatureInputTensors(
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
        FetchTensorType(env, tensor, FetchTensorTypeId(env, tensor))));
  }
  return input_tensors;
}

std::vector<std::unique_ptr<SimpleTensor>> FetchSignatureOutputTensors(
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
        FetchTensorType(env, tensor, FetchTensorTypeId(env, tensor))));
  }
  return output_tensors;
}

}  // namespace

Expected<std::vector<absl::string_view>> CompiledModel::GetSignatureKeys()
    const {
  size_t num_signatures = GetNumSignatures();
  std::vector<absl::string_view> signature_keys;
  signature_keys.reserve(num_signatures);
  for (int i = 0; i < num_signatures; ++i) {
    LiteRtSignature lite_rt_signature;
    LITERT_RETURN_IF_ERROR(
        env_.runtime->GetModelSignature(model_.Get(), i, &lite_rt_signature));
    signature_keys.push_back(FetchSignatureKey(env_, lite_rt_signature));
  }
  return signature_keys;
}

Expected<std::vector<SimpleSignature>> CompiledModel::GetSignatures() const {
  auto num_signatures = GetNumSignatures();
  std::vector<SimpleSignature> signatures;
  signatures.reserve(num_signatures);
  for (int i = 0; i < num_signatures; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto signature, GetSignature(i));
    signatures.push_back(std::move(signature));
  }
  return std::move(signatures);
}

Expected<SimpleSignature> CompiledModel::GetSignature(
    size_t signature_index) const {
  LiteRtSignature lite_rt_signature;
  LITERT_RETURN_IF_ERROR(env_.runtime->GetModelSignature(
      model_.Get(), signature_index, &lite_rt_signature));
  return SimpleSignature(FetchSignatureKey(env_, lite_rt_signature),
                         FetchSignatureInputNames(env_, lite_rt_signature),
                         FetchSignatureInputTensors(env_, lite_rt_signature),
                         FetchSignatureOutputNames(env_, lite_rt_signature),
                         FetchSignatureOutputTensors(env_, lite_rt_signature));
}

Expected<size_t> CompiledModel::GetSignatureIndex(
    absl::string_view signature_key) const {
  if (signature_key.empty()) {
    return 0;
  }
  auto num_signatures = GetNumSignatures();
  for (int i = 0; i < num_signatures; ++i) {
    LiteRtSignature lite_rt_signature;
    LITERT_RETURN_IF_ERROR(
        env_.runtime->GetModelSignature(model_.Get(), i, &lite_rt_signature));
    auto key = FetchSignatureKey(env_, lite_rt_signature);
    if (key == signature_key) {
      return i;
    }
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
}

Expected<size_t> CompiledModel::FindInputIndex(
    size_t signature_index, absl::string_view input_name) const {
  LITERT_ASSIGN_OR_RETURN(const auto input_names,
                          GetSignatureInputNames(signature_index));
  auto it = absl::c_find(input_names, input_name);
  if (it != input_names.end()) {
    return std::distance(input_names.begin(), it);
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find input");
}

Expected<std::vector<absl::string_view>> CompiledModel::GetSignatureInputNames(
    size_t signature_index) const {
  LiteRtSignature lite_rt_signature;
  LITERT_RETURN_IF_ERROR(env_.runtime->GetModelSignature(
      model_.Get(), signature_index, &lite_rt_signature));
  return FetchSignatureInputNames(env_, lite_rt_signature);
}

Expected<std::vector<absl::string_view>> CompiledModel::GetSignatureOutputNames(
    size_t signature_index) const {
  LiteRtSignature lite_rt_signature;
  LITERT_RETURN_IF_ERROR(env_.runtime->GetModelSignature(
      model_.Get(), signature_index, &lite_rt_signature));
  return FetchSignatureOutputNames(env_, lite_rt_signature);
}

Expected<SimpleSignature> CompiledModel::FindSignature(
    absl::string_view signature_key) const {
  LITERT_ASSIGN_OR_RETURN(auto index, GetSignatureIndex(signature_key));
  return GetSignature(index);
}

Expected<size_t> CompiledModel::FindOutputIndex(
    size_t signature_index, absl::string_view output_name) const {
  LITERT_ASSIGN_OR_RETURN(const auto output_names,
                          GetSignatureOutputNames(signature_index));
  auto it = absl::c_find(output_names, output_name);
  if (it != output_names.end()) {
    return std::distance(output_names.begin(), it);
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find output");
}

Expected<TensorBuffer> CompiledModel::CreateBufferImpl(
    const Environment& env, const TensorBufferRequirements& buffer_requirements,
    const RankedTensorType& tensor_type) {
  LITERT_ASSIGN_OR_RETURN(const std::vector<TensorBufferType>& supported_types,
                          buffer_requirements.SupportedTypes());
  if (supported_types.empty()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Input doesn't support any tensor buffer types");
  }
  // For simplicity we just pick the first supported tensor buffer type.
  TensorBufferType tensor_buffer_type = supported_types[0];
  LITERT_ASSIGN_OR_RETURN(size_t buffer_size, buffer_requirements.BufferSize());

  LITERT_ASSIGN_OR_RETURN(TensorBuffer buffer, TensorBuffer::CreateManaged(
                                                   env, tensor_buffer_type,
                                                   tensor_type, buffer_size));
  return buffer;
}

Expected<TensorBuffer> CompiledModel::CreateInputOutputBuffer(
    size_t signature_index, absl::string_view tensor_name,
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

Expected<std::vector<TensorBuffer>> CompiledModel::CreateInputOutputBuffers(
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

Expected<void> CompiledModel::RunCApiHelper(LiteRtParamIndex signature_index,
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

Expected<void> CompiledModel::RunCApiHelper(LiteRtParamIndex signature_index,
                                            size_t num_input_buffers,
                                            LiteRtTensorBuffer* input_buffers,
                                            size_t num_output_buffers,
                                            LiteRtTensorBuffer* output_buffers,
                                            bool& async,
                                            LiteRtOptions run_options) const {
  return RunCApiHelper(signature_index, num_input_buffers, input_buffers,
                       num_output_buffers, output_buffers, async, run_options,
                       /*scheduling_info=*/nullptr);
}

Expected<void> CompiledModel::RunCApiHelper(
    LiteRtParamIndex signature_index, size_t num_input_buffers,
    LiteRtTensorBuffer* input_buffers, size_t num_output_buffers,
    LiteRtTensorBuffer* output_buffers, bool& async, LiteRtOptions run_options,
    const LiteRtSchedulingInfo* scheduling_info) const {
  if (run_options != nullptr && scheduling_info != nullptr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Run options and scheduling info are mutually exclusive");
  }

  LiteRtStatus status;
  if (scheduling_info != nullptr) {
    status =
        async ? env_.runtime->RunCompiledModelAsyncWithSchedulingInfo(
                    Get(), signature_index, num_input_buffers, input_buffers,
                    num_output_buffers, output_buffers, &async, scheduling_info)
              : env_.runtime->RunCompiledModelWithSchedulingInfo(
                    Get(), signature_index, num_input_buffers, input_buffers,
                    num_output_buffers, output_buffers, scheduling_info);
  } else if (run_options != nullptr) {
    status = async
                 ? env_.runtime->RunCompiledModelAsyncWithOptions(
                       Get(), signature_index, num_input_buffers, input_buffers,
                       num_output_buffers, output_buffers, &async, run_options)
                 : env_.runtime->RunCompiledModelWithOptions(
                       Get(), signature_index, num_input_buffers, input_buffers,
                       num_output_buffers, output_buffers, run_options);
  } else {
    status = async
                 ? env_.runtime->RunCompiledModelAsync(
                       Get(), signature_index, num_input_buffers, input_buffers,
                       num_output_buffers, output_buffers, &async)
                 : env_.runtime->RunCompiledModel(
                       Get(), signature_index, num_input_buffers, input_buffers,
                       num_output_buffers, output_buffers);
  }
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to invoke the compiled model");
  }
  return {};
}

Expected<void> CompiledModel::RunHelper(
    size_t signature_index, absl::Span<const TensorBuffer> input_buffers,
    absl::Span<const TensorBuffer> output_buffers, bool& async) const {
  return RunHelper(signature_index, input_buffers, output_buffers, async,
                   /*run_options=*/nullptr,
                   /*scheduling_info=*/nullptr);
}

Expected<void> CompiledModel::RunHelper(
    size_t signature_index, absl::Span<const TensorBuffer> input_buffers,
    absl::Span<const TensorBuffer> output_buffers, bool& async,
    LiteRtOptions run_options) const {
  return RunHelper(signature_index, input_buffers, output_buffers, async,
                   run_options,
                   /*scheduling_info=*/nullptr);
}

Expected<void> CompiledModel::RunHelper(
    size_t signature_index, absl::Span<const TensorBuffer> input_buffers,
    absl::Span<const TensorBuffer> output_buffers, bool& async,
    LiteRtOptions run_options,
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

Expected<void> CompiledModel::RunMapHelper(
    absl::string_view signature_key,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
    bool& async) const {
  return RunMapHelper(signature_key, input_map, output_map, async,
                      /*run_options=*/nullptr,
                      /*scheduling_info=*/nullptr);
}

Expected<void> CompiledModel::RunMapHelper(
    absl::string_view signature_key,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
    bool& async, LiteRtOptions run_options) const {
  auto signature_index = GetSignatureIndex(signature_key);
  return RunMapHelper(signature_key, input_map, output_map, async, run_options,
                      /*scheduling_info=*/nullptr);
}

Expected<void> CompiledModel::RunMapHelper(
    absl::string_view signature_key,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
    bool& async, LiteRtOptions run_options,
    const LiteRtSchedulingInfo* scheduling_info) const {
  auto signature_index = GetSignatureIndex(signature_key);
  if (!signature_index) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get signature_index");
  }
  return RunMapWithIndexHelper(*signature_index, input_map, output_map, async,
                               run_options, scheduling_info);
}

Expected<void> CompiledModel::RunMapWithIndexHelper(
    size_t signature_index,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
    bool& async) const {
  return RunMapWithIndexHelper(signature_index, input_map, output_map, async,
                               /*run_options=*/nullptr,
                               /*scheduling_info=*/nullptr);
}

Expected<void> CompiledModel::RunMapWithIndexHelper(
    size_t signature_index,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
    bool& async, LiteRtOptions run_options) const {
  return RunMapWithIndexHelper(signature_index, input_map, output_map, async,
                               run_options,
                               /*scheduling_info=*/nullptr);
}

Expected<void> CompiledModel::RunMapWithIndexHelper(
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
    // if the input is not provided in the input map, we set it to nullptr.
    if (it == input_map.end()) {
      input_buffers_ptr[i] = nullptr;
      continue;
    }
    input_buffers_ptr[i] = it->second.Get();
  }
  LITERT_ASSIGN_OR_RETURN(auto output_names,
                          GetSignatureOutputNames(signature_index));
  size_t num_outputs = output_names.size();
  auto output_buffers_ptr = std::make_unique<LiteRtTensorBuffer[]>(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    absl::string_view output_name = output_names[i];
    auto it = output_map.find(output_name);
    if (it == output_map.end()) {
      return Unexpected(kLiteRtStatusErrorNotFound,
                        "The given map is missing some output TensorBuffers");
    }
    output_buffers_ptr[i] = it->second.Get();
  }
  return RunCApiHelper(signature_index, num_inputs, input_buffers_ptr.get(),
                       num_outputs, output_buffers_ptr.get(), async,
                       run_options, scheduling_info);
}

Expected<bool> CompiledModel::IsFullyAccelerated() {
  bool fully_accelerated = false;
  LITERT_RETURN_IF_ERROR(
      env_.runtime->CompiledModelIsFullyAccelerated(Get(), &fully_accelerated));
  return fully_accelerated;
}

bool CompiledModel::CheckCancelledWrapper(void* data) {
  CompiledModel* model = static_cast<CompiledModel*>(data);
  if (model && model->check_cancelled_func_) {
    return model->check_cancelled_func_();
  }
  return false;
}

void CompiledModel::SetCancellationFunction(
    absl::AnyInvocable<bool()> check_cancelled_func) {
  check_cancelled_func_ = std::move(check_cancelled_func);
  env_.runtime->SetCompiledModelCancellationFunction(Get(), this,
                                                     &CheckCancelledWrapper);
}

Expected<TensorBufferRequirements> CompiledModel::GetInputBufferRequirements(
    size_t signature_index, size_t input_index) const {
  LiteRtTensorBufferRequirements buffer_requirements;
  LITERT_RETURN_IF_ERROR(env_.runtime->GetCompiledModelInputBufferRequirements(
      Get(), signature_index, input_index, &buffer_requirements));
  return ToTensorBufferRequirements(env_, buffer_requirements);
}

Expected<TensorBufferRequirements> CompiledModel::GetOutputBufferRequirements(
    size_t signature_index, size_t output_index) const {
  LiteRtTensorBufferRequirements buffer_requirements;
  LITERT_RETURN_IF_ERROR(env_.runtime->GetCompiledModelOutputBufferRequirements(
      Get(), signature_index, output_index, &buffer_requirements));
  return ToTensorBufferRequirements(env_, buffer_requirements);
}

}  // namespace litert
