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

#include "litert/vendors/qualcomm/dispatch/litert_dispatch_invocation_context.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/qualcomm/context_binary_info.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "HTP/QnnHtpProfile.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnProfile.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

using litert::Expected;
using litert::Unexpected;
using litert::qnn::QnnManager;

std::string_view inline GetEventUnit(QnnProfile_EventUnit_t unit) {
  switch (unit) {
    case QNN_PROFILE_EVENTUNIT_MICROSEC:
      return "us";
    case QNN_PROFILE_EVENTUNIT_BYTES:
      return "bytes";
    case QNN_PROFILE_EVENTUNIT_CYCLES:
      return "cycles";
    case QNN_PROFILE_EVENTUNIT_COUNT:
      return "count";
    case QNN_PROFILE_EVENTUNIT_BACKEND:
    default:
      return "";
  }
}

LiteRtDispatchInvocationContextT::LiteRtDispatchInvocationContextT(
    litert::qnn::QnnManager& qnn_manager,
    const litert::qnn::ContextBinaryInfo& context_binary_info,
    LiteRtDispatchDeviceContextT& device_context,
    QnnManager::ContextHandle&& context_handle,
    Qnn_ProfileHandle_t profile_handle, int graph_index,
    Qnn_GraphHandle_t graph_handle)
    : qnn_manager_(qnn_manager),
      device_context_(device_context),
      context_handle_(std::move(context_handle)),
      profile_handle_(profile_handle),
      graph_index_(graph_index),
      graph_handle_(graph_handle),
      inputs_(context_binary_info.Graphs()[graph_index].Inputs()),
      outputs_(context_binary_info.Graphs()[graph_index].Outputs()) {
  input_buffer_handles_.resize(inputs_.size());

  std::vector<::qnn::TensorWrapper> real_outputs;
  std::vector<::qnn::TensorWrapper> dumped_outputs;
  // Dumped tensors are treated as output, move them to the end of outputs_
  std::for_each(
      outputs_.begin(), outputs_.end(),
      [&real_outputs, &dumped_outputs](const ::qnn::TensorWrapper& tensor) {
        tensor.IsMarkedDump() ? dumped_outputs.emplace_back(tensor)
                              : real_outputs.emplace_back(tensor);
      });

  output_buffer_handles_.resize(real_outputs.size());

  outputs_.clear();
  std::move(real_outputs.begin(), real_outputs.end(),
            std::back_inserter(outputs_));
  std::move(dumped_outputs.begin(), dumped_outputs.end(),
            std::back_inserter(outputs_));
}

Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    QnnManager& qnn, LiteRtDispatchDeviceContextT& device_context,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name) {
  auto exec_bytecode_ptr =
      static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  auto context_binary_info = litert::qnn::ContextBinaryInfo::Create(
      qnn, exec_bytecode_ptr, exec_bytecode_buffer->size);
  if (!context_binary_info) {
    return Unexpected(context_binary_info.Error());
  }

  const auto& graphs = context_binary_info->Graphs();
  if (graphs.empty()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "No graph found");
  }

  int graph_index = -1;
  // If the function_name is not specified and there is only one graph, then
  // take that graph.
  if (absl::string_view(function_name).empty() && graphs.size() == 1) {
    graph_index = 0;
    const auto& graph = graphs[graph_index];
    function_name = graph.Name().c_str();
  } else {
    for (auto i = 0; i < graphs.size(); ++i) {
      const auto& graph = graphs[i];
      if (graph.Name() == absl::string_view(function_name)) {
        graph_index = i;
        break;
      }
    }
  }
  if (graph_index < 0) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Function name not found");
  }

  auto configs = QnnManager::DefaultContextConfigs();

  auto profiling_level = qnn.GetOptions().GetProfiling();
  Qnn_ProfileHandle_t profile_handle = nullptr;
  if (profiling_level != ::qnn::Profiling::kOff) {
    if (auto status = qnn.Api()->profileCreate(
            qnn.BackendHandle(),
            static_cast<QnnProfile_Level_t>(profiling_level), &profile_handle);
        status != QNN_SUCCESS) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to create profile handle");
    }
  }

  auto context_handle = qnn.CreateContextHandle(
      configs,
      absl::MakeSpan(static_cast<const uint8_t*>(exec_bytecode_ptr),
                     exec_bytecode_buffer->size),
      profile_handle);
  if (!context_handle) {
    return Unexpected(context_handle.Error());
  }

  Qnn_GraphHandle_t graph_handle;
  if (auto status = qnn.Api()->graphRetrieve(context_handle->get(),
                                             function_name, &graph_handle);
      status != QNN_SUCCESS) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to retrieve graph");
  }

  return Ptr(new LiteRtDispatchInvocationContextT(
      qnn, std::move(*context_binary_info), device_context,
      std::move(*context_handle), profile_handle, graph_index, graph_handle));
}

namespace {

Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type) {
  if (tensor_type.layout.has_strides) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported by QNN");
  }

  static constexpr std::array<const LiteRtTensorBufferType, 2>
      kSupportedTensorBufferTypes = {
          kLiteRtTensorBufferTypeFastRpc,
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

}  // namespace

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_input_index < 0 || graph_input_index >= inputs_.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_input_index");
  }

  auto& tensor = inputs_[graph_input_index];
  input_buffer_handles_[graph_input_index] = tensor_buffer_handle;
  return AttachBuffer(tensor.GetQnnTensor(), tensor_buffer_handle);
}

Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (graph_output_index < 0 || graph_output_index >= outputs_.size()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Invalid graph_output_index");
  }

  auto& tensor = outputs_[graph_output_index];
  output_buffer_handles_[graph_output_index] = tensor_buffer_handle;
  return AttachBuffer(tensor.GetQnnTensor(), tensor_buffer_handle);
}

Expected<void> LiteRtDispatchInvocationContextT::DetachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto& tensor = inputs_[graph_input_index];
  LITERT_RETURN_IF_ERROR(
      DetachBuffer(tensor.GetQnnTensor(), tensor_buffer_handle));
  input_buffer_handles_[graph_input_index] = -1;
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::DetachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto& tensor = outputs_[graph_output_index];
  LITERT_RETURN_IF_ERROR(
      DetachBuffer(tensor.GetQnnTensor(), tensor_buffer_handle));
  output_buffer_handles_[graph_output_index] = -1;
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::AttachBuffer(
    Qnn_Tensor_t& tensor, LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);
  if (!tensor_buffer) {
    return Unexpected(tensor_buffer.Error());
  }

  auto mem_handle = device_context_.GetMemHandle(tensor_buffer_handle, tensor);
  if (!mem_handle) {
    return Unexpected(mem_handle.Error());
  }

  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    tensor.v1.memHandle = *mem_handle;

  } else if (tensor.version == QNN_TENSOR_VERSION_2) {
    if (tensor.v2.isDynamicDimensions != nullptr) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Dynamic dimensions not yet supported");
    }
    tensor.v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    tensor.v2.memHandle = *mem_handle;

  } else {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unsupported QNN tensor version");
  }
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::DetachBuffer(
    Qnn_Tensor_t& tensor, LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_RETURN_IF_ERROR(
      device_context_.UnregisterTensorBuffer(tensor_buffer_handle, tensor));
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::Execute() {
  for (int i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i].IsQuantU16()) {
      ConvertToUint16(input_buffer_handles_[i], inputs_[i].GetTensorBytes());
    }
  }
  const size_t num_ins = inputs_.size();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, inputs, num_ins, QNN_TENSOR_INIT);
  for (size_t i = 0; i < num_ins; ++i) {
    *(inputs + i) = inputs_.at(i).GetQnnTensor();
  }

  const size_t num_outs = outputs_.size();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, outputs, num_outs, QNN_TENSOR_INIT);
  for (size_t i = 0; i < num_outs; ++i) {
    if (outputs_.at(i).IsMarkedDump()) {
      // need to manage its own buffer since LiteRT only allocates and attaches
      // buffer for real graph in/out
      outputs_.at(i).AllocateOutputTensorBuffer();
    }
    *(outputs + i) = outputs_.at(i).GetQnnTensor();
  }

  if (auto status = qnn_manager_.Api()->graphExecute(
          graph_handle_, inputs, num_ins, outputs, num_outs, profile_handle_,
          /*signalHandle=*/nullptr);
      status != QNN_SUCCESS) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to execute graph");
  }

  if (profile_handle_ != nullptr) {
    LITERT_RETURN_IF_ERROR(Profile());
  }

  for (int i = 0; i < outputs_.size(); ++i) {
    if (outputs_[i].IsQuantU16()) {
      ConvertToInt16(output_buffer_handles_[i], outputs_[i].GetTensorBytes());
    }
  }
  // TODO (chunhsue-qti): pass folder as option
  std::string dump_folder = "/data/local/tmp/dumped_tensors/";
  for (int i = 0; i < outputs_.size(); ++i) {
    if (outputs_.at(i).IsMarkedDump()) {
      auto status = WriteTensorTo(dump_folder, outputs_[i]);
      if (!status) {
        LITERT_LOG(LITERT_ERROR, "Failed to dump tensor id: %d", i);
      }
    }
  }

  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::ConvertToUint16(
    LiteRtTensorBufferHandle tensor_buffer_handle, size_t bytes) {
  auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);
  if (!tensor_buffer) {
    return Unexpected(tensor_buffer.Error());
  }
  void* mem_addr;
  if (auto status = LiteRtLockTensorBuffer(
          *tensor_buffer, &mem_addr, kLiteRtTensorBufferLockModeReadWrite);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to lock the tensor buffer");
  }
  auto int16_data = absl::MakeSpan(static_cast<const std::int16_t*>(mem_addr),
                                   bytes / sizeof(std::int16_t));
  std::vector<std::uint16_t> uint16_data;
  qnn::ConvertDataFromInt16toUInt16(int16_data, uint16_data);
  std::memcpy(mem_addr, uint16_data.data(), bytes);
  if (auto status = LiteRtUnlockTensorBuffer(*tensor_buffer);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to unlock the tensor buffer");
  }
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::ConvertToInt16(
    LiteRtTensorBufferHandle tensor_buffer_handle, size_t bytes) {
  auto tensor_buffer = device_context_.GetTensorBuffer(tensor_buffer_handle);
  if (!tensor_buffer) {
    return Unexpected(tensor_buffer.Error());
  }
  void* mem_addr;
  if (auto status = LiteRtLockTensorBuffer(
          *tensor_buffer, &mem_addr, kLiteRtTensorBufferLockModeReadWrite);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to lock the tensor buffer");
  }
  auto uint16_data = absl::MakeSpan(static_cast<const std::uint16_t*>(mem_addr),
                                    bytes / sizeof(std::uint16_t));
  std::vector<std::int16_t> int16_data;
  qnn::ConvertDataFromUInt16toInt16(uint16_data, int16_data);
  std::memcpy(mem_addr, int16_data.data(), bytes);
  if (auto status = LiteRtUnlockTensorBuffer(*tensor_buffer);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to unlock the tensor buffer");
  }
  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::Profile() {
  // TODO: Implement a viewer class to beautify the format of profiling output
  std::stringstream data_ss;
  data_ss << "\nExecute Stats:\n"
          << "----------------" << std::endl;
  const QnnProfile_EventId_t* events_ptr = nullptr;
  const QnnProfile_EventId_t* sub_events_ptr = nullptr;
  std::uint32_t num_events = 0;
  std::uint32_t num_sub_events = 0;

  if (auto status = qnn_manager_.Api()->profileGetEvents(
          profile_handle_, &events_ptr, &num_events);
      status != QNN_SUCCESS) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to get the profile events");
  }

  QnnProfile_EventData_t event_data;
  for (std::uint32_t i = 0; i < num_events; ++i) {
    if (auto status =
            qnn_manager_.Api()->profileGetEventData(events_ptr[i], &event_data);
        status != QNN_SUCCESS) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to get the event data from profiling result");
    }
    data_ss << "    " << event_data.identifier << ": " << event_data.value
            << " " << GetEventUnit(event_data.unit) << std::endl;

    // Check the sub events only related to graph execution time
    if (event_data.type ==
        QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_CYCLE) {
      if (auto status = qnn_manager_.Api()->profileGetSubEvents(
              events_ptr[i], &sub_events_ptr, &num_sub_events);
          status != QNN_SUCCESS) {
        return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                          "Failed to get the event data from profiling result");
      }

      QnnProfile_EventData_t sub_event_data;
      for (std::uint32_t j = 0; j < num_sub_events; ++j) {
        if (auto status = qnn_manager_.Api()->profileGetEventData(
                sub_events_ptr[j], &sub_event_data);
            status != QNN_SUCCESS) {
          return Unexpected(
              kLiteRtStatusErrorRuntimeFailure,
              "Failed to get the sub event data from profiling result");
        }
        if (sub_event_data.type == QNN_PROFILE_EVENTTYPE_NODE &&
            (sub_event_data.unit == QNN_PROFILE_EVENTUNIT_MICROSEC ||
             sub_event_data.unit == QNN_PROFILE_EVENTUNIT_CYCLES)) {
          data_ss << "        " << sub_event_data.identifier << ": "
                  << sub_event_data.value << " "
                  << GetEventUnit(sub_event_data.unit) << std::endl;
        }
      }
    }
  }

  LITERT_LOG(LITERT_INFO, "%s", data_ss.str().c_str());

  return {};
}

Expected<void> LiteRtDispatchInvocationContextT::WriteTensorTo(
    const std::filesystem::path& output_folder, ::qnn::TensorWrapper& tensor) {
  qnn::CreateDirectoryRecursive(output_folder);
  std::filesystem::path output_path =
      output_folder / (tensor.GetName() + ".raw");
  std::ofstream fout(output_path, std::ios::binary);
  if (fout.fail()) {
    LITERT_LOG(LITERT_ERROR, "Failed to write dumped tensor");
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }
  fout.write(static_cast<const char*>(tensor.GetQnnTensor().v2.clientBuf.data),
             tensor.GetQnnTensor().v2.clientBuf.dataSize);
  std::filesystem::path quant_param_path =
      output_folder / (tensor.GetName() + ".csv");
  std::ofstream quant_file(quant_param_path);
  float scale = 1;
  int32_t zero_point = 0;
  auto quant_param = tensor.GetQuantParams();
  if (std::holds_alternative<qnn::ScaleOffsetQuantizeParamsWrapper>(
          quant_param)) {
    scale =
        std::get<qnn::ScaleOffsetQuantizeParamsWrapper>(quant_param).GetScale();
    zero_point = std::get<qnn::ScaleOffsetQuantizeParamsWrapper>(quant_param)
                     .GetZeroPoint();
  }
  std::stringstream quant_ss;
  quant_ss << scale << "," << zero_point << "\n";
  quant_file << quant_ss.str();
  quant_file.close();
  return {};
}
