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
//
// Copyright 2024 Google LLC.

#include "litert/vendors/samsung/dispatch/litert_dispatch_invocation_context.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"

#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/samsung/dispatch/litert_dispatch_device_context.h"
namespace litert::samsung {

// Require continuous memory for tensor buffer. No strides.
Expected<LiteRtTensorBufferRequirements>
GetTensorBufferRequirements(const LiteRtRankedTensorType &tensor_type) {
  static constexpr std::array<const LiteRtTensorBufferType, 1>
      kSupportedTensorBufferTypes = {
          kLiteRtTensorBufferTypeDmaBuf,
      };

  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  if (auto status = LiteRtCreateTensorBufferRequirements(
          kSupportedTensorBufferTypes.size(),
          kSupportedTensorBufferTypes.data(), *buffer_size, /*num_strides=*/0,
          /*strides=*/nullptr, &requirements);
      status != kLiteRtStatusOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Not implemented");
  }

  return requirements;
}
} // namespace litert::samsung

LiteRtDispatchInvocationContextT::LiteRtDispatchInvocationContextT(
    const ::litert::samsung::EnnManager *enn_manager,
    LiteRtDispatchDeviceContext device_context, const EnnModelId &model_id,
    int num_inputs, int num_outputs)
    : enn_manager_(enn_manager), device_context_(device_context),
      model_id_(model_id), inputs_buf_(num_inputs, nullptr),
      outputs_buf_(num_outputs, nullptr) {
}

litert::Expected<LiteRtDispatchInvocationContextT::UniquePtr>
LiteRtDispatchInvocationContextT::Create(
    const ::litert::samsung::EnnManager *enn_manager,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer *exec_bytecode_buffer, const char *function_name,
    int num_inputs, int num_outputs) {
  EnnBufferPtr *tmp_buf_ptr;
  if (!enn_manager) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to get enn runtime.");
  }

  const void *exec_bytecode_ptr =
      static_cast<const uint8_t *>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  auto exec_bytecode_size = exec_bytecode_buffer->size;
  EnnModelId model_id;
  if (enn_manager->Api().EnnOpenModelFromMemory(
          reinterpret_cast<const char *>(exec_bytecode_ptr), exec_bytecode_size,
          &model_id) != ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to load model.");
  }

  NumberOfBuffersInfo buffer_info;
  if (enn_manager->Api().EnnGetBuffersInfo(model_id, &buffer_info) !=
      ENN_RET_SUCCESS) {
    enn_manager->Api().EnnCloseModel(model_id);
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to get buffers information");
  }
  if (buffer_info.n_in_buf != num_inputs ||
      buffer_info.n_out_buf != num_outputs) {
    enn_manager->Api().EnnCloseModel(model_id);
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Number of inputs/outputs is invalid");
  }

  if (enn_manager->Api().EnnAllocateAllBuffers(model_id, &tmp_buf_ptr,
                                               &buffer_info) != ENN_RET_SUCCESS) {
    enn_manager->Api().EnnCloseModel(model_id);
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "EnnAllocateAllBuffers Failed");
  }

  device_context->SetEnnCommittedBuffer(tmp_buf_ptr);

  return LiteRtDispatchInvocationContextT::UniquePtr(
      new LiteRtDispatchInvocationContextT(enn_manager, device_context,
                                           model_id, num_inputs, num_outputs));
}

LiteRtDispatchInvocationContextT::~LiteRtDispatchInvocationContextT() {
  enn_manager_->Api().EnnCloseModel(model_id_);
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType &tensor_type) {
  return litert::samsung::GetTensorBufferRequirements(tensor_type);
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType &tensor_type) {
  return litert::samsung::GetTensorBufferRequirements(tensor_type);
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_input_index < 0 || graph_input_index >= inputs_buf_.size()) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Invalid input index");
  }

  auto enn_buffer = device_context_->GetEnnBuffer(tensor_buffer_handle);
  if (!enn_buffer) {
    return enn_buffer.Error();
  }
  inputs_buf_[graph_input_index] = enn_buffer.Value();

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_output_index < 0 || graph_output_index >= outputs_buf_.size()) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Invalid input index");
  }

  auto enn_buffer = device_context_->GetEnnBuffer(tensor_buffer_handle);
  if (!enn_buffer) {
    return enn_buffer.Error();
  }
  outputs_buf_[graph_output_index] = enn_buffer.Value();

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::DetachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_input_index < 0 || graph_input_index >= inputs_buf_.size()) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Invalid input index");
  }
  inputs_buf_[graph_input_index] = nullptr;

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::DetachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_output_index < 0 || graph_output_index >= outputs_buf_.size()) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Invalid input index");
  }
  outputs_buf_[graph_output_index] = nullptr;

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::Invoke() {
  if (auto status = SetInputBuffers(); !status) {
    return status.Error();
  }

  if (enn_manager_->Api().EnnExecuteModel(model_id_) != ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure, "Fail to execute");
  }

  if (auto status = SetOutputBuffers(); !status) {
    return status.Error();
  }

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::SetInputBuffers() const {
  bool input_prepared =
      std::all_of(inputs_buf_.begin(), inputs_buf_.end(),
                  [](const EnnBufferPtr &val) { return val != nullptr; });
  bool output_prepared =
      std::all_of(outputs_buf_.begin(), outputs_buf_.end(),
                  [](const EnnBufferPtr &val) { return val != nullptr; });
  if (!input_prepared || !output_prepared) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Inputs/outputs not prepared.");
  }
  LITERT_ASSIGN_OR_RETURN(auto committed_buf, device_context_->GetEnnCommittedBuffer());

  for (int idx = 0; idx < inputs_buf_.size() ; idx++) {
    auto usr_buf = inputs_buf_.at(idx);
    auto target_buf = committed_buf[idx];
    memcpy(target_buf->va, usr_buf->va, usr_buf->size);
  }

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::SetOutputBuffers() const {
  LITERT_ASSIGN_OR_RETURN(auto committed_buf, device_context_->GetEnnCommittedBuffer());

  int _input_buf_size = inputs_buf_.size();
  for (int idx = 0; idx < outputs_buf_.size(); idx++) {
    auto usr_buf = outputs_buf_.at(idx);
    auto target_buf = committed_buf[idx + _input_buf_size];
    memcpy(usr_buf->va, target_buf->va, usr_buf->size);
  }
  return {};
}
