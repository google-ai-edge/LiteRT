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
// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/samsung/dispatch/litert_dispatch_invocation_context.h"

#include <unistd.h>

#include <cstdint>
#include <fstream>
#include <memory>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/samsung/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/samsung/dispatch/litert_weight_binary_manager.h"
#include "litert/vendors/samsung/schema/litert_samsung_header_generated.h"

namespace litert::samsung {

#define TO_MEM_OFFSET(header_offset, buffer_offset) \
  ((header_offset) + (buffer_offset))
#define TO_FILE_OFFSET(header_offset, buffer_offset, alloc_base_file_offset) \
  ((header_offset) + (buffer_offset) + (alloc_base_file_offset))

struct EnnModelHandle {
  EnnModelId id;
  const ::litert::samsung::EnnManager* manager;

  EnnModelHandle(EnnModelId id, const ::litert::samsung::EnnManager* manager)
      : id(id), manager(manager) {}
};

// Custom deleter for EnnModelHandle
struct EnnModelDeleter {
  void operator()(EnnModelHandle* handle) const {
    if (handle && handle->manager) {
      handle->manager->Api().EnnCloseModel(handle->id);
    }
    delete handle;
  }
};

using EnnModelPtr = std::unique_ptr<EnnModelHandle, EnnModelDeleter>;

static EnnModelPtr MakeEnnModelPtr(const ::litert::samsung::EnnManager* manager,
                                   EnnModelId model_id) {
  auto raw_ptr = std::make_unique<EnnModelHandle>(model_id, manager);
  return EnnModelPtr(raw_ptr.release(), EnnModelDeleter{});
}

// Structure to hold separated weight information
struct SeparatedWeightInfo {
  std::string signature;
  int64_t start_memory_offset;
  int64_t end_memory_offset;
  int64_t start_file_offset;
  int64_t end_file_offset;
};

// Structure to hold model loading information
struct ModelLoadInfo {
  bool has_valid_header = false;
  bool use_external_weights = false;
  const void* model_addr = nullptr;
  size_t model_size = 0;
  int fd = -1;
  uint32_t model_memory_offset = 0;
  uint32_t model_file_offset = 0;
  std::vector<SeparatedWeightInfo> separated_weights;
  std::vector<std::string> signatures;
};

static Expected<ModelLoadInfo> AnalyzeModelLoadStrategy(
    const LiteRtMemBuffer* exec_bytecode_buffer) {
  ModelLoadInfo info;

  const void* exec_bytecode_ptr =
      static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  size_t exec_bytecode_size = exec_bytecode_buffer->size;

  LITERT_LOG(LITERT_INFO, "Bytecode buffer: base_addr=%p, offset=%zu, size=%zu",
             exec_bytecode_buffer->base_addr, exec_bytecode_buffer->offset,
             exec_bytecode_size);
  LITERT_LOG(LITERT_INFO, "Calculated bytecode ptr: %p, size: %zu",
             exec_bytecode_ptr, exec_bytecode_size);

  info.fd = exec_bytecode_buffer->fd;

  auto* header_buf =
      litert::samsung::schema::GetLiteRTSamsungHeader(exec_bytecode_ptr);
  if (!header_buf) {
    LITERT_LOG(LITERT_INFO, "No valid Samsung header found - using old format");
    info.model_memory_offset =
        static_cast<uint32_t>(exec_bytecode_buffer->offset);
    info.model_file_offset =
        static_cast<uint32_t>(exec_bytecode_buffer->alloc_base_file_offset +
                              exec_bytecode_buffer->offset);
    info.model_size = exec_bytecode_size;
    return info;
  }

  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(exec_bytecode_ptr),
                                 exec_bytecode_size);

  bool fb_generated = header_buf->Verify(verifier);
  info.has_valid_header = fb_generated;

  if (!fb_generated) {
    LITERT_LOG(LITERT_INFO, "Header verification failed - using old format");
    info.model_memory_offset =
        static_cast<uint32_t>(exec_bytecode_buffer->offset);
    info.model_file_offset =
        static_cast<uint32_t>(exec_bytecode_buffer->alloc_base_file_offset +
                              exec_bytecode_buffer->offset);
    info.model_size = exec_bytecode_size;
    return info;
  }

  auto* dispatch_binary = header_buf->dispatch_binary();
  info.use_external_weights = dispatch_binary->use_external_weights();

  const auto* buf_section = dispatch_binary->buf();
  const auto* external_weights = dispatch_binary->external_weights();
  const auto* separated_weights = header_buf->separated_weights();

  LITERT_LOG(LITERT_INFO, "Schema version: %u", header_buf->version());
  LITERT_LOG(LITERT_VERBOSE,
             "Dispatch Binary Buffer: start_offset=%ld, end_offset=%ld",
             buf_section->start_offset(), buf_section->end_offset());
  LITERT_LOG(LITERT_VERBOSE, "Use External Weights: %s",
             info.use_external_weights ? "true" : "false");

  if (separated_weights) {
    for (int32_t i = 0; i < separated_weights->size(); i++) {
      const auto* weight = separated_weights->Get(i);
      SeparatedWeightInfo weight_info;
      weight_info.signature = weight->signature()->str();
      weight_info.start_memory_offset = TO_MEM_OFFSET(
          weight->buf()->start_offset(), exec_bytecode_buffer->offset);
      weight_info.end_memory_offset = TO_MEM_OFFSET(
          weight->buf()->end_offset(), exec_bytecode_buffer->offset);
      weight_info.start_file_offset = TO_FILE_OFFSET(
          weight->buf()->start_offset(), exec_bytecode_buffer->offset,
          exec_bytecode_buffer->alloc_base_file_offset);
      weight_info.end_file_offset = TO_FILE_OFFSET(
          weight->buf()->end_offset(), exec_bytecode_buffer->offset,
          exec_bytecode_buffer->alloc_base_file_offset);
      info.separated_weights.push_back(weight_info);
      LITERT_LOG(LITERT_SILENT,
                 "Separated Weight[%d]: signature=%s, offset=%ld-%ld", i,
                 weight_info.signature.c_str(), weight_info.start_file_offset,
                 weight_info.end_file_offset);
    }
    LITERT_LOG(LITERT_SILENT, "Total separated weights: %zu",
               info.separated_weights.size());
  }

  if (info.use_external_weights) {
    LITERT_LOG(LITERT_VERBOSE, "Valid header with external weights");
    info.model_memory_offset = static_cast<uint32_t>(TO_MEM_OFFSET(
        buf_section->start_offset(), exec_bytecode_buffer->offset));
    info.model_file_offset = static_cast<uint32_t>(TO_FILE_OFFSET(
        buf_section->start_offset(), exec_bytecode_buffer->offset,
        exec_bytecode_buffer->alloc_base_file_offset));
    info.model_size = buf_section->end_offset() - buf_section->start_offset();
    LITERT_LOG(LITERT_SILENT, "Model offset: %u, Model size: %zu",
               info.model_file_offset, info.model_size);
    for (int32_t i = 0; i < external_weights->size(); i++) {
      info.signatures.emplace_back(external_weights->Get(i)->str());
    }
  } else {
    LITERT_LOG(LITERT_VERBOSE, "Valid header without external weights");
    info.model_memory_offset = static_cast<uint32_t>(TO_MEM_OFFSET(
        buf_section->start_offset(), exec_bytecode_buffer->offset));
    info.model_file_offset = static_cast<uint32_t>(TO_FILE_OFFSET(
        buf_section->start_offset(), exec_bytecode_buffer->offset,
        exec_bytecode_buffer->alloc_base_file_offset));
    info.model_size = buf_section->end_offset() - buf_section->start_offset();
    LITERT_LOG(LITERT_SILENT, "Model offset: %u, Model size: %zu",
               info.model_file_offset, info.model_size);
  }
  return info;
}

// Require continuous memory for tensor buffer. No strides.
Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
    const LiteRtRuntimeContext* runtime_context,
    const LiteRtRankedTensorType& tensor_type) {
  static constexpr std::array<const LiteRtTensorBufferType, 1>
      kSupportedTensorBufferTypes = {
          kLiteRtTensorBufferTypeDmaBuf,
      };

  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  if (auto status = runtime_context->create_tensor_buffer_requirements(
          kSupportedTensorBufferTypes.size(),
          kSupportedTensorBufferTypes.data(), *buffer_size, /*num_strides=*/0,
          /*strides=*/nullptr, &requirements);
      status != kLiteRtStatusOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Not implemented");
  }

  return requirements;
}
}  // namespace litert::samsung

LiteRtDispatchInvocationContextT::LiteRtDispatchInvocationContextT(
    ::litert::samsung::EnnManager* enn_manager,
    LiteRtDispatchDeviceContext device_context, const EnnModelId& model_id,
    int num_inputs, int num_outputs)
    : enn_manager_(enn_manager),
      device_context_(device_context),
      model_id_(model_id),
      inputs_buf_(num_inputs, nullptr),
      outputs_buf_(num_outputs, nullptr) {}

litert::Expected<LiteRtDispatchInvocationContextT::UniquePtr>
LiteRtDispatchInvocationContextT::Create(
    ::litert::samsung::EnnManager* enn_manager,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs) {
  if (!enn_manager) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to get enn runtime.");
  }

  // Analyze model loading strategy
  LITERT_ASSIGN_OR_RETURN(
      auto load_info,
      litert::samsung::AnalyzeModelLoadStrategy(exec_bytecode_buffer));

  EnnModelId model_id;
  std::vector<std::string> signatures;

  // Load model based on condition
  if (load_info.has_valid_header && load_info.use_external_weights) {
    auto& weight_mgr =
        litert::samsung::WeightBinaryManager::GetInstance(enn_manager);
    std::vector<EnnBufferPtr> weight_buffers;

    for (size_t i = 0; i < load_info.signatures.size(); i++) {
      const auto& sig = load_info.signatures[i];

      int64_t offset = 0;
      size_t size = 0;

      // If separated_weights has the corresponding entry, use its offset/size
      if (!load_info.separated_weights.empty() &&
          i < load_info.separated_weights.size()) {
        if (load_info.fd >= 0) {
          offset = load_info.separated_weights[i].start_file_offset;
          size = load_info.separated_weights[i].end_file_offset -
                 load_info.separated_weights[i].start_file_offset;
        } else {
          offset = load_info.separated_weights[i].start_memory_offset;
          size = load_info.separated_weights[i].end_memory_offset -
                 load_info.separated_weights[i].start_memory_offset;
        }
      }
      litert::Expected<EnnBufferPtr> buffer;
      if (load_info.fd >= 0) {
        buffer = weight_mgr.Acquire(sig, load_info.fd, offset, size);
      } else {
        const void* addr =
            static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
            offset;
        buffer = weight_mgr.Acquire(sig, addr, size);
      }

      if (!buffer) {
        return buffer.Error();
      }
      weight_buffers.push_back(*buffer);
      signatures.push_back(sig);
    }
    if (load_info.fd >= 0) {
      if (enn_manager->Api().EnnOpenModelWithFileOpenFdWeight(
              load_info.fd, load_info.model_size, load_info.model_file_offset,
              &weight_buffers[0], weight_buffers.size(),
              &model_id) != ENN_RET_SUCCESS)
        return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to load model from fd with weights");
    } else {
      const void* model_addr =
          static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
          load_info.model_memory_offset;
      if (enn_manager->Api().EnnOpenModelFromMemoryWithWeight(
              reinterpret_cast<const char*>(model_addr), load_info.model_size,
              &weight_buffers[0], weight_buffers.size(),
              &model_id) != ENN_RET_SUCCESS)
        return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to load model from memory with weights");
    }
  } else {
    if (exec_bytecode_buffer->fd >= 0) {
      LITERT_LOG(LITERT_SILENT, "fd: %d, Model offset: %u, Model size: %zu",
                 exec_bytecode_buffer->fd, load_info.model_file_offset,
                 load_info.model_size);

      if (enn_manager->Api().EnnOpenModelWithFileOpenFd(
              exec_bytecode_buffer->fd,
              static_cast<uint32_t>(load_info.model_size),
              load_info.model_file_offset, &model_id) != ENN_RET_SUCCESS)
        return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                             "Fail to load model from fd.");
    } else {
      // Load from memory (fd not available)
      LITERT_LOG(LITERT_INFO, "Loading model from memory");

      const void* model_addr =
          static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
          load_info.model_memory_offset;
      size_t model_size = load_info.model_size;

      LITERT_LOG(LITERT_SILENT, "Model addr: %p, Model size: %zu", model_addr,
                 model_size);

      if (enn_manager->Api().EnnOpenModelFromMemory(
              reinterpret_cast<const char*>(model_addr), model_size,
              &model_id) != ENN_RET_SUCCESS)
        return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                             "Fail to load model from memory.");
    }
  }

  auto model_guard = MakeEnnModelPtr(enn_manager, model_id);

  NumberOfBuffersInfo buffer_info;
  if (enn_manager->Api().EnnGetBuffersInfo(model_id, &buffer_info) !=
      ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to get buffers information");
  }
  LITERT_LOG(LITERT_INFO, "Buffer info - inputs: %d, outputs: %d",
             buffer_info.n_in_buf, buffer_info.n_out_buf);

  if (buffer_info.n_in_buf != num_inputs ||
      buffer_info.n_out_buf != num_outputs) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Number of inputs/outputs is invalid");
  }

  // Release ownership since we're successfully creating the context
  model_guard.release();

  auto context = LiteRtDispatchInvocationContextT::UniquePtr(
      new LiteRtDispatchInvocationContextT(enn_manager, device_context,
                                           model_id, num_inputs, num_outputs));

  // Initialize commit caching state
  context->commit_state_ = CommitState::kUncommitted;
  context->attached_input_count_ = 0;
  context->attached_output_count_ = 0;

  // Set weight signatures for later release
  context->SetWeightSignatures(std::move(signatures));

  return context;
}

LiteRtDispatchInvocationContextT::~LiteRtDispatchInvocationContextT() {
  // If committed, uncommit and unset buffers
  if (commit_state_ == CommitState::kCommitted) {
    enn_manager_->Api().EnnBufferUncommit(model_id_);
    enn_manager_->Api().EnnUnsetBuffers(model_id_);
  }

  // Release weight buffers from global weight manager
  auto& weight_mgr =
      litert::samsung::WeightBinaryManager::GetInstance(enn_manager_);
  for (const auto& sig : weight_signatures_) {
    weight_mgr.Release(sig);
  }

  enn_manager_->Api().EnnCloseModel(model_id_);
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::TransitionToCommitted() {
  if (enn_manager_->Api().EnnBufferCommit(model_id_) != ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to commit buffer");
  }
  commit_state_ = CommitState::kCommitted;
  return {};
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::TransitionToUncommitted() {
  if (enn_manager_->Api().EnnBufferUncommit(model_id_) != ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to uncommit buffer");
  }
  commit_state_ = CommitState::kUncommitted;
  return {};
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::UpdateStateAfterAttach() {
  if (AreAllBuffersAttached()) {
    // Perform actual commit
    if (enn_manager_->Api().EnnBufferCommit(model_id_) != ENN_RET_SUCCESS) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "Fail to commit buffer");
    }
    commit_state_ = CommitState::kCommitted;
  } else {
    commit_state_ = CommitState::kPartialAttach;
  }
  return {};
}

void LiteRtDispatchInvocationContextT::UpdateStateAfterDetach() {
  commit_state_ = CommitState::kPartialAttach;
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  return litert::samsung::GetTensorBufferRequirements(device_context_->runtime_context(), tensor_type);
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  return litert::samsung::GetTensorBufferRequirements(device_context_->runtime_context(), tensor_type);
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_input_index < 0 || graph_input_index >= inputs_buf_.size()) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Invalid input index");
  }

  // Check if already attached
  if (inputs_buf_[graph_input_index] != nullptr) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Input already attached");
  }

  // If committed, uncommit first before modifying buffers
  if (commit_state_ == CommitState::kCommitted) {
    LITERT_RETURN_IF_ERROR(TransitionToUncommitted());
  }

  auto enn_buffer = device_context_->GetEnnBuffer(tensor_buffer_handle);
  if (!enn_buffer) {
    return enn_buffer.Error();
  }
  inputs_buf_[graph_input_index] = enn_buffer.Value();

  // Register buffer with ENN
  if (enn_manager_->Api().EnnSetBufferByIndex(
          model_id_, ENN_DIR_IN, graph_input_index, enn_buffer.Value()) !=
      ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to set input buffer");
  }

  attached_input_count_.fetch_add(1, std::memory_order_relaxed);
  LITERT_RETURN_IF_ERROR(UpdateStateAfterAttach());

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_output_index < 0 || graph_output_index >= outputs_buf_.size()) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Invalid output index");
  }

  // Check if already attached
  if (outputs_buf_[graph_output_index] != nullptr) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Output already attached");
  }

  // If committed, uncommit first before modifying buffers
  if (commit_state_ == CommitState::kCommitted) {
    LITERT_RETURN_IF_ERROR(TransitionToUncommitted());
  }

  auto enn_buffer = device_context_->GetEnnBuffer(tensor_buffer_handle);
  if (!enn_buffer) {
    return enn_buffer.Error();
  }
  outputs_buf_[graph_output_index] = enn_buffer.Value();

  // Register buffer with ENN
  if (enn_manager_->Api().EnnSetBufferByIndex(
          model_id_, ENN_DIR_OUT, graph_output_index, enn_buffer.Value()) !=
      ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to set output buffer");
  }

  attached_output_count_.fetch_add(1, std::memory_order_relaxed);
  LITERT_RETURN_IF_ERROR(UpdateStateAfterAttach());

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::DetachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_input_index < 0 || graph_input_index >= inputs_buf_.size()) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Invalid input index");
  }

  // If committed, uncommit first
  if (commit_state_ == CommitState::kCommitted) {
    LITERT_RETURN_IF_ERROR(TransitionToUncommitted());
  }

  inputs_buf_[graph_input_index] = nullptr;
  attached_input_count_.fetch_sub(1, std::memory_order_relaxed);
  UpdateStateAfterDetach();

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::DetachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_output_index < 0 || graph_output_index >= outputs_buf_.size()) {
    return litert::Error(kLiteRtStatusErrorInvalidArgument,
                         "Invalid output index");
  }

  // If committed, uncommit first
  if (commit_state_ == CommitState::kCommitted) {
    LITERT_RETURN_IF_ERROR(TransitionToUncommitted());
  }

  outputs_buf_[graph_output_index] = nullptr;
  attached_output_count_.fetch_sub(1, std::memory_order_relaxed);
  UpdateStateAfterDetach();

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::Invoke() {
  // Check if buffers are committed
  if (commit_state_ != CommitState::kCommitted) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Buffers not committed. Attach all buffers first.");
  }

  // Execute model (no memcpy needed)
  if (enn_manager_->Api().EnnExecuteModel(model_id_) != ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure, "Fail to execute");
  }

  return {};
}

void LiteRtDispatchInvocationContextT::SetSchedulingInfo(
    const LiteRtSchedulingInfo* scheduling_info) {
  if (scheduling_info != nullptr) {
    scheduling_info_ = *scheduling_info;
  } else {
    scheduling_info_.reset();
  }
}

// SetInputBuffers and SetOutputBuffers removed - no longer needed with commit
// caching
