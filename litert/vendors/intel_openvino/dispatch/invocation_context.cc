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

#include "litert/vendors/intel_openvino/dispatch/invocation_context.h"

#include <chrono>  // NOLINT

#include "openvino/runtime/tensor.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/vendors/c/litert_dispatch.h"

litert::Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    LiteRtDispatchDeviceContextT& device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs) {
  const void* exec_bytecode_ptr =
      static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  auto exec_bytecode_size = exec_bytecode_buffer->size;

  std::string bytecode_buffer(reinterpret_cast<const char*>(exec_bytecode_ptr),
                              exec_bytecode_size);
  std::istringstream model_stream(bytecode_buffer);
  if (!model_stream) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to open model bytecode stream");
  }
  auto core = device_context.getCore();
  if (!core) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get OpenVINO core from device context");
  }
  // ov::CompiledModel compiled_model = core->import_model(model_stream, "NPU");
   static int graph_num = 0;
  std::string xml_path = "C:\\Workspace\\junwei\\gemma\\openvino_ir\\Partition_mod_" + std::to_string(graph_num) + ".xml";
  graph_num++;
  LITERT_LOG(LITERT_INFO, "Openvino InvocationContext Initialize with model %s",
             xml_path.c_str());
  ov::CompiledModel compiled_model = core->compile_model(xml_path, "NPU");
  auto infer_request = compiled_model.create_infer_request();
  LITERT_LOG(LITERT_INFO, "Openvino InvocationContext Initialize SUCCESS");
  // TODO: add support for loading cached model
  return Ptr(new LiteRtDispatchInvocationContextT(infer_request, device_context,
                                                  num_inputs, num_outputs));
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type) {
  LiteRtTensorBufferType supported_tensor_buffer_types[] = {
#if defined(LITERT_WINDOWS_OS)
      kLiteRtTensorBufferTypeHostMemory,
#else
      kLiteRtTensorBufferTypeAhwb,
      kLiteRtTensorBufferTypeDmaBuf,
#endif
  };

  int num_supported_tensor_buffer_types =
      sizeof(supported_tensor_buffer_types) /
      sizeof(supported_tensor_buffer_types[0]);

  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return litert::Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  auto status = LiteRtCreateTensorBufferRequirements(
      num_supported_tensor_buffer_types, supported_tensor_buffer_types,
      *buffer_size, 0, /*strides=*/nullptr, &requirements);
  if (status != kLiteRtStatusOk)
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to get buffer requirements");

  return requirements;
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
#if defined(LITERT_WINDOWS_OS)
  LITERT_ASSIGN_OR_RETURN(ov::intel_npu::level_zero::ZeroBufferTensor ov_tensor,
                          device_context_.getOvTensor(tensor_buffer_handle));
  input_tensor_buffer_handles_.push_back(tensor_buffer_handle);
#else
  LITERT_ASSIGN_OR_RETURN(ov::Tensor ov_tensor,
                          device_context_.getOvTensor(tensor_buffer_handle));
#endif
  // TODO: visit this if need to maintain graph indices for inputs and outputs
  // in dispatch_api
  infer_request_.set_input_tensor(graph_input_index, ov_tensor);
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
#if defined(LITERT_WINDOWS_OS)
  LITERT_ASSIGN_OR_RETURN(ov::intel_npu::level_zero::ZeroBufferTensor ov_tensor,
                          device_context_.getOvTensor(tensor_buffer_handle));
  output_tensor_buffer_handles_.push_back(tensor_buffer_handle);
#else
  LITERT_ASSIGN_OR_RETURN(ov::Tensor ov_tensor,
                          device_context_.getOvTensor(tensor_buffer_handle));
#endif
  // TODO: visit this if need to maintain graph indices for inputs and outputs
  // in dispatch_api
  infer_request_.set_output_tensor(graph_output_index, ov_tensor);
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::Invoke() {
  for (auto& tensor_buffer_handle : input_tensor_buffer_handles_) {
    LITERT_ASSIGN_OR_RETURN(LiteRtTensorBuffer tensor_buffer,
                          device_context_.getTensorBuffer(tensor_buffer_handle));

    size_t tensor_buffer_size;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size),
        litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to get tensor buffer size"));
    // LITERT_LOG(LITERT_ERROR, "========%d ", tensor_buffer_size);
    void* buffer_host_addr;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferHostMemory(tensor_buffer, &buffer_host_addr),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get HostMemory buffer"));

    LITERT_ASSIGN_OR_RETURN(ov::intel_npu::level_zero::ZeroBufferTensor ov_tensor,
                          device_context_.getOvTensor(tensor_buffer_handle));
    memcpy(ov_tensor.get(), buffer_host_addr, tensor_buffer_size);
  }
  infer_request_.start_async();
  if (!infer_request_.wait_for(
          std::chrono::milliseconds(kInferRequestTimeoutMs)))
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "Failed to execute inference request due to timeout");
  for (auto& tensor_buffer_handle : output_tensor_buffer_handles_) {
    LITERT_ASSIGN_OR_RETURN(LiteRtTensorBuffer tensor_buffer,
                          device_context_.getTensorBuffer(tensor_buffer_handle));

    size_t tensor_buffer_size;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size),
        litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to get tensor buffer size"));
    // LITERT_LOG(LITERT_ERROR, "========%d ", tensor_buffer_size);
    void* buffer_host_addr;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferHostMemory(tensor_buffer, &buffer_host_addr),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get HostMemory buffer"));

    LITERT_ASSIGN_OR_RETURN(ov::intel_npu::level_zero::ZeroBufferTensor ov_tensor,
                          device_context_.getOvTensor(tensor_buffer_handle));
    memcpy(buffer_host_addr, ov_tensor.get(), tensor_buffer_size);
  }
  return {};
}
