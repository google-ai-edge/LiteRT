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
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/qualcomm/context_binary_info.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

class LiteRtDispatchDeviceContextT;

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  ~LiteRtDispatchInvocationContextT() = default;

  static litert::Expected<Ptr> Create(
      litert::qnn::QnnManager& qnn_manager,
      LiteRtDispatchDeviceContextT& device_context,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name);

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

  litert::Expected<void> Execute();

  litert::Expected<void> Profile();

  Qnn_ContextHandle_t ContextHandle() { return context_handle_.get(); }

 private:
  LiteRtDispatchInvocationContextT(
      litert::qnn::QnnManager& qnn_manager,
      const litert::qnn::ContextBinaryInfo& context_binary_info,
      LiteRtDispatchDeviceContextT& device_context,
      litert::qnn::QnnManager::ContextHandle&& context_handle,
      Qnn_ProfileHandle_t profile_handle, int graph_index,
      Qnn_GraphHandle_t graph_handle);

  litert::Expected<void> AttachBuffer(
      Qnn_Tensor_t& tensor, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> DetachBuffer(
      Qnn_Tensor_t& tensor, LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void> ConvertToUint16(
      LiteRtTensorBufferHandle tensor_buffer_handle, size_t bytes);

  litert::Expected<void> ConvertToInt16(
      LiteRtTensorBufferHandle tensor_buffer_handle, size_t bytes);

  litert::Expected<void> WriteTensorTo(
      const std::filesystem::path& output_folder, ::qnn::TensorWrapper& tensor);

  litert::qnn::QnnManager& qnn_manager_;
  LiteRtDispatchDeviceContextT& device_context_;
  litert::qnn::QnnManager::ContextHandle context_handle_;
  Qnn_ProfileHandle_t profile_handle_;
  int graph_index_;
  Qnn_GraphHandle_t graph_handle_;
  std::vector<::qnn::TensorWrapper> inputs_;
  std::vector<::qnn::TensorWrapper> outputs_;
  std::vector<LiteRtTensorBufferHandle> input_buffer_handles_;
  std::vector<LiteRtTensorBufferHandle> output_buffer_handles_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_LITERT_DISPATCH_INVOCATION_CONTEXT_H_
