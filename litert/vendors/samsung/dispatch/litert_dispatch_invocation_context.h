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

#ifndef LITERT_VENDORS_SAMSUNG_DISPATCH_INVOCATION_CONTEXT_H_
#define LITERT_VENDORS_SAMSUNG_DISPATCH_INVOCATION_CONTEXT_H_

#include <optional>

#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/samsung/dispatch/enn_manager.h"

class LiteRtDispatchInvocationContextT {
 public:
  using UniquePtr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  static litert::Expected<UniquePtr> Create(
      const ::litert::samsung::EnnManager* enn_manager,
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

  void SetSchedulingInfo(const LiteRtSchedulingInfo* scheduling_info) {
    if (scheduling_info == nullptr) {
      scheduling_info_ = std::nullopt;
      return;
    }
    scheduling_info_ = *scheduling_info;
  }

  const LiteRtSchedulingInfo* GetSchedulingInfo() const {
    return scheduling_info_.has_value() ? &scheduling_info_.value() : nullptr;
  }

 private:
  LiteRtDispatchInvocationContextT(
      const ::litert::samsung::EnnManager* enn_manager,
      LiteRtDispatchDeviceContext device_context, const EnnModelId& model_id,
      int num_inputs, int num_outputs);

  litert::Expected<void> SetInputBuffers() const;
  litert::Expected<void> SetOutputBuffers() const;

  const ::litert::samsung::EnnManager* enn_manager_;
  LiteRtDispatchDeviceContext device_context_;
  std::optional<LiteRtSchedulingInfo> scheduling_info_;
  EnnModelId model_id_;
  std::vector<EnnBufferPtr> inputs_buf_;
  std::vector<EnnBufferPtr> outputs_buf_;
};

#endif  // LITERT_VENDORS_SAMSUNG_DISPATCH_INVOCATION_CONTEXT_H_
