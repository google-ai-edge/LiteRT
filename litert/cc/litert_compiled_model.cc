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
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_tensor_buffer_types.h"

namespace litert {

Expected<size_t> CompiledModel::FindInputIndex(
    size_t signature_index, absl::string_view input_name) const {
  LITERT_ASSIGN_OR_RETURN(const auto input_names,
                          model_.GetSignatureInputNames(signature_index));
  auto it = absl::c_find(input_names, input_name);
  if (it != input_names.end()) {
    return std::distance(input_names.begin(), it);
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find input");
}

Expected<size_t> CompiledModel::FindOutputIndex(
    size_t signature_index, absl::string_view output_name) const {
  LITERT_ASSIGN_OR_RETURN(const auto output_names,
                          model_.GetSignatureOutputNames(signature_index));
  auto it = absl::c_find(output_names, output_name);
  if (it != output_names.end()) {
    return std::distance(output_names.begin(), it);
  }
  return Unexpected(kLiteRtStatusErrorNotFound, "Failed to find output");
}

Expected<TensorBuffer> CompiledModel::CreateBufferImpl(
    const Environment& env, const TensorBufferRequirements& buffer_requirements,
    const RankedTensorType& tensor_type) {
  LITERT_ASSIGN_OR_RETURN(
      const std::vector<TensorBufferType>& supported_types,
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
      is_input ? model_.GetInputTensorType(signature_index, tensor_name)
               : model_.GetOutputTensorType(signature_index, tensor_name);
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, tensor_type_expected);
  LITERT_ASSIGN_OR_RETURN(auto env, GetEnvironment());
  if (is_input) {
    LITERT_ASSIGN_OR_RETURN(
        TensorBufferRequirements buffer_requirements,
        GetInputBufferRequirements(signature_index, tensor_name));
    LITERT_ASSIGN_OR_RETURN(size_t tensor_index,
                            FindInputIndex(signature_index, tensor_name));
    LiteRtLayout input_layout;
    if (LiteRtGetCompiledModelInputTensorLayout(Get(), signature_index,
                                                tensor_index, &input_layout) ==
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
  tensor_names = is_input ? model_.GetSignatureInputNames(signature_index)
                          : model_.GetSignatureOutputNames(signature_index);
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
  LiteRtStatus status =
      async ? LiteRtRunCompiledModelAsync(
                  Get(), signature_index, num_input_buffers, input_buffers,
                  num_output_buffers, output_buffers, &async)
            : LiteRtRunCompiledModel(Get(), signature_index, num_input_buffers,
                                     input_buffers, num_output_buffers,
                                     output_buffers);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to invoke the compiled model");
  }
  return {};
}

Expected<void> CompiledModel::RunHelper(
    size_t signature_index, absl::Span<const TensorBuffer> input_buffers,
    absl::Span<const TensorBuffer> output_buffers, bool& async) const {
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
                       output_buffers_ptr.get(), async);
}

Expected<void> CompiledModel::RunMapHelper(
    absl::string_view signature_key,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
    bool& async) const {
  auto signature_index = model_.GetSignatureIndex(signature_key);
  if (!signature_index) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      "Failed to get signature_index");
  }
  return RunMapWithIndexHelper(*signature_index, input_map, output_map, async);
}

Expected<void> CompiledModel::RunMapWithIndexHelper(
    size_t signature_index,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& input_map,
    const absl::flat_hash_map<absl::string_view, TensorBuffer>& output_map,
    bool& async) const {
  LITERT_ASSIGN_OR_RETURN(auto input_names,
                          model_.GetSignatureInputNames(signature_index));
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
                          model_.GetSignatureOutputNames(signature_index));
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
                       num_outputs, output_buffers_ptr.get(), async);
}

Expected<bool> CompiledModel::IsFullyAccelerated() {
  bool fully_accelerated = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtCompiledModelIsFullyAccelerated(Get(), &fully_accelerated));
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
  LiteRtSetCompiledModelCancellationFunction(Get(), this,
                                             &CheckCancelledWrapper);
}

}  // namespace litert
