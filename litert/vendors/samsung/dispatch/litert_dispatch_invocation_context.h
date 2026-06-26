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

#ifndef ODML_LITERT_VENDORS_SAMSUNG_DISPATCH_INVOCATION_CONTEXT_H_
#define ODML_LITERT_VENDORS_SAMSUNG_DISPATCH_INVOCATION_CONTEXT_H_

#include <atomic>
#include <optional>

#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/samsung/dispatch/enn_manager.h"

// Note: This class is NOT thread-safe.
// Caller must ensure that only one thread accesses a given instance at a time.
class LiteRtDispatchInvocationContextT {
 public:
  using UniquePtr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  static litert::Expected<UniquePtr> Create(
      ::litert::samsung::EnnManager* enn_manager,
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs);

  ~LiteRtDispatchInvocationContextT();

  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);

  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

  litert::Expected<void> AttachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> AttachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> DetachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> DetachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> Invoke();

  void SetSchedulingInfo(const LiteRtSchedulingInfo* scheduling_info);

 private:
  // Commit state for caching
  enum class CommitState {
    kUncommitted,    // Initial state or all buffers detached
    kCommitted,      // All buffers attached and committed
    kPartialAttach,  // Some buffers attached, waiting for commit
  };

  LiteRtDispatchInvocationContextT(::litert::samsung::EnnManager* enn_manager,
                                   LiteRtDispatchDeviceContext device_context,
                                   const EnnModelId& model_id, int num_inputs,
                                   int num_outputs);

  bool AreAllBuffersAttached() const {
    return attached_input_count_.load(std::memory_order_relaxed) ==
               inputs_buf_.size() &&
           attached_output_count_.load(std::memory_order_relaxed) ==
               outputs_buf_.size();
  }

  // Helper methods for state transitions
  litert::Expected<void> TransitionToCommitted();
  litert::Expected<void> TransitionToUncommitted();
  litert::Expected<void> UpdateStateAfterAttach();
  void UpdateStateAfterDetach();

  void SetWeightSignatures(std::vector<std::string> signatures) {
    weight_signatures_ = std::move(signatures);
  }

  ::litert::samsung::EnnManager* enn_manager_;
  LiteRtDispatchDeviceContext device_context_;
  std::optional<LiteRtSchedulingInfo> scheduling_info_;
  EnnModelId model_id_;
  std::vector<EnnBufferPtr> inputs_buf_;
  std::vector<EnnBufferPtr> outputs_buf_;
  std::vector<std::string> weight_signatures_;

  // Commit caching state
  CommitState commit_state_ = CommitState::kUncommitted;
  std::atomic<int> attached_input_count_{0};
  std::atomic<int> attached_output_count_{0};
};

#endif  // ODML_LITERT_VENDORS_SAMSUNG_DISPATCH_INVOCATION_CONTEXT_H_
