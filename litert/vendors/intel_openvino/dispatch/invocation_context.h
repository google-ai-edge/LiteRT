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

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/intel_openvino/dispatch/device_context.h"
#if defined(__ANDROID__) && defined(ENABLE_NPU_HAL)
#include "litert/vendors/intel_openvino/dispatch/npu_hal_hook/npu_hal_hook.h"
#endif

class LiteRtDispatchDeviceContextT;

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;
  using IntelOpenVinoOptions = ::litert::intel_openvino::IntelOpenVinoOptions;

  ~LiteRtDispatchInvocationContextT() = default;

  static litert::Expected<Ptr> Create(
      LiteRtDispatchDeviceContextT& device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs,
      const IntelOpenVinoOptions* intel_openvino_opts);

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

  void SetSchedulingInfo(const LiteRtSchedulingInfo* scheduling_info) {
    if (scheduling_info == nullptr) {
      scheduling_info_ = std::nullopt;
    }
    scheduling_info_ = *scheduling_info;
  }

  const LiteRtSchedulingInfo* GetSchedulingInfo() const {
    return scheduling_info_.has_value() ? &scheduling_info_.value() : nullptr;
  }

 private:
  LiteRtDispatchInvocationContextT(ov::InferRequest& infer_request,
                                   LiteRtDispatchDeviceContextT& device_context,
                                   int num_inputs, int num_outputs,
                                   const void* exec_bytecode_ptr,
                                   size_t exec_bytecode_size,
                                   std::string device)
      : device_context_(device_context),
        infer_request_(infer_request),
        exec_bytecode_ptr_(exec_bytecode_ptr),
        exec_bytecode_size_(exec_bytecode_size),
        device_(std::move(device)) {}

  // Re-imports the model with the given NPU job priority and rebuilds the
  // infer request. ov::hint::model_priority is a compile-time hint, so a
  // priority change only takes effect when the model is (re-)imported with it.
  litert::Expected<void> ApplyModelPriority(int32_t job_priority);

  LiteRtDispatchDeviceContextT& device_context_;
  ov::InferRequest infer_request_;
  // Timeout is in milliseconds
  static constexpr int kInferRequestTimeoutMs = 10000;

  std::optional<LiteRtSchedulingInfo> scheduling_info_;
  // Non-owning view of the model bytecode. The backing memory is the loaded
  // (typically mmap'd) model buffer owned by the caller, which outlives this
  // invocation context. Used to re-import the model when the requested
  // priority changes. No copy is kept to avoid duplicating a large blob.
  const void* exec_bytecode_ptr_ = nullptr;
  size_t exec_bytecode_size_ = 0;
  std::string device_;
  // Priority currently baked into infer_request_, if any.
  std::optional<int32_t> applied_job_priority_;
  // Input/output tensors bound via AttachInput/AttachOutput, kept so they can
  // be re-bound after the infer request is rebuilt.
  std::map<int, ov::Tensor> input_tensors_;
  std::map<int, ov::Tensor> output_tensors_;
#if defined(__ANDROID__) && defined(ENABLE_NPU_HAL)
  npu_hal_context_t* ctx = nullptr;
#endif
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
