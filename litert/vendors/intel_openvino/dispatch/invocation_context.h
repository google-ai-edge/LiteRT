// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/intel_openvino/dispatch/device_context.h"

class LiteRtDispatchDeviceContextT;

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  ~LiteRtDispatchInvocationContextT() = default;

  static litert::Expected<Ptr> Create(
      LiteRtDispatchDeviceContextT& device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs);

  litert::Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
      const LiteRtRankedTensorType& tensor_type);
  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);
  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

  litert::Expected<void> AttachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> AttachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle);
  litert::Expected<void> DetachInput(
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
    // Nothing to do.
    return {};
  }

  litert::Expected<void> DetachOutput(
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
    // Nothing to do.
    return {};
  }

  litert::Expected<void> Invoke();

 private:
  LiteRtDispatchInvocationContextT(ov::InferRequest& infer_request,
                                   LiteRtDispatchDeviceContextT& device_context,
                                   int num_inputs, int num_outputs)
      : device_context_(device_context), infer_request_(infer_request) {}
  LiteRtDispatchDeviceContextT& device_context_;
  ov::InferRequest infer_request_;
  // Timeout is in milliseconds
  static constexpr int kInferRequestTimeoutMs = 10000;
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
