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

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstddef>
#include <cstring>
#include <exception>
#include <ios>
#include <istream>
#include <streambuf>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/intel_openvino/bytecode_header.h"

namespace {

// This class is copied from the OpenVINO codebase with minor modifications
// for Google C++ Style Guide compliance. It wraps a pre-allocated memory
// buffer to provide a std::streambuf interface, enabling zero-copy stream
// reading.
//
// TODO(b/449624371): Remove SharedStreamBuffer once OpenVINO provides a
// public equivalent.
class SharedStreamBuffer : public std::streambuf {
 public:
  SharedStreamBuffer(const char* data, size_t size)
      : data_(data), size_(size), offset_(0) {}
  explicit SharedStreamBuffer(const void* data, size_t size)
      : SharedStreamBuffer(reinterpret_cast<const char*>(data), size) {}

 protected:
  // override std::streambuf methods
  std::streamsize xsgetn(char* s, std::streamsize count) override {
    auto real_count = std::min<std::streamsize>(size_ - offset_, count);
    std::memcpy(s, data_ + offset_, real_count);
    offset_ += real_count;
    return real_count;
  }

  int_type underflow() override {
    return (size_ == offset_) ? traits_type::eof()
                              : traits_type::to_int_type(*(data_ + offset_));
  }

  int_type uflow() override {
    return (size_ == offset_) ? traits_type::eof()
                              : traits_type::to_int_type(*(data_ + offset_++));
  }

  std::streamsize showmanyc() override { return size_ - offset_; }

  pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
    return seekoff(pos, std::ios_base::beg, which);
  }

  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which) override {
    if (which != std::ios_base::in) {
      return pos_type(off_type(-1));
    }

    size_t new_offset;
    switch (dir) {
      case std::ios_base::beg:
        new_offset = off;
        break;
      case std::ios_base::cur:
        new_offset = offset_ + off;
        break;
      case std::ios_base::end:
        new_offset = size_ + off;
        break;
      default:
        return pos_type(off_type(-1));
    }

    // Check bounds
    if (new_offset > size_) {
      return pos_type(off_type(-1));
    }

    offset_ = new_offset;
    return pos_type(offset_);
  }

  // Non-virtual overload with default argument for backward compatibility
  pos_type seekoff(off_type off, std::ios_base::seekdir dir) {
    return seekoff(off, dir, std::ios_base::in);
  }

 private:
  const char* data_;
  const size_t size_;
  size_t offset_;
};

}  // namespace

litert::Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    LiteRtDispatchDeviceContextT& device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    const IntelOpenVinoOptions* intel_openvino_opts) {
  const void* exec_bytecode_ptr =
      static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  auto exec_bytecode_size = exec_bytecode_buffer->size;

  // If the compiler embedded a self-describing header, honor the device
  // recorded there.  Per-partition bytecode produced by the LiteRT OpenVINO
  // compiler plugin always carries this header, so each partition can be
  // dispatched to its own target device (NPU/CPU/GPU) even when the
  // model-wide options request a different default.
  std::string device = "NPU";  // Default device
  LiteRtIntelOpenVinoGraphBackend embedded_graph_backend =
      kLiteRtIntelOpenVinoGraphBackendNPU;
  size_t payload_offset = 0;
  bool device_from_header = litert::openvino::TryParseBytecodeHeader(
      exec_bytecode_ptr, exec_bytecode_size, &embedded_graph_backend,
      &payload_offset);
  if (device_from_header) {
    device = litert::openvino::GraphBackendToString(embedded_graph_backend);
    exec_bytecode_ptr =
        static_cast<const uint8_t*>(exec_bytecode_ptr) + payload_offset;
    exec_bytecode_size -= payload_offset;
    LITERT_LOG(LITERT_INFO, "Dispatch: using device '%s' from bytecode header",
               device.c_str());
  } else {
    LITERT_LOG(LITERT_INFO,
               "Dispatch: no bytecode header found, defaulting to '%s'",
               device.c_str());
  }

  // Validate that the requested device is actually available on this system
  // before setting the device.
  auto core = device_context.getCore();
  if (!core) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get OpenVINO core from device context");
  }
  {
    const std::vector<std::string>& available_devices =
        OpenVINOSharedCore::GetInstance()->GetAvailableDevices();

    auto matches = [&device](const std::string& name) {
      if (name == device) return true;
      auto dot = name.find('.');
      return dot != std::string::npos && name.substr(0, dot) == device;
    };

    if (std::none_of(available_devices.begin(), available_devices.end(),
                     matches)) {
      std::string available_list;
      for (const auto& d : available_devices) {
        if (!available_list.empty()) available_list += ", ";
        available_list += d;
      }
      LITERT_LOG(LITERT_ERROR,
                 "Dispatch: requested OpenVINO device '%s' is not available. "
                 "Available devices: [%s]",
                 device.c_str(), available_list.c_str());
      return litert::Error(
          kLiteRtStatusErrorRuntimeFailure,
          "Requested OpenVINO device is not available on this system");
    }
  }
  LITERT_LOG(LITERT_INFO, "Using Intel OpenVINO device: %s", device.c_str());

  // Forward configs_map entries to OV Core. Runtime-only properties such
  // as PERF_COUNT must be set here (at import_model time) — they are not
  // baked into the exported NPU blob by the AOT compile path.
  ov::AnyMap import_configs;
  bool perf_count_enabled = false;
  const int num_options = intel_openvino_opts != nullptr ?
                          intel_openvino_opts->GetNumConfigsMapOptions() : 0;
  for (int i = 0; i < num_options; ++i) {
    auto [key, value] = intel_openvino_opts->GetConfigsMapOption(i);
    if (key.empty()) continue;
    if (key == "PERF_COUNT" && (value == "YES" || value == "true")) {
      perf_count_enabled = true;
    }

    import_configs[key] = value;
    LITERT_LOG(LITERT_INFO, "Dispatch: OpenVINO config '%s'='%s'", key.c_str(),
               value.c_str());
  }

  OpenVINOSharedCore::GetInstance()->SetDevice(device);

  if (!exec_bytecode_ptr || exec_bytecode_size == 0) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Empty bytecode buffer");
  }
  SharedStreamBuffer membuf(static_cast<const char*>(exec_bytecode_ptr),
                            exec_bytecode_size);
  std::istream model_stream(&membuf);
  if (!model_stream) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to open model bytecode stream");
  }
  ov::CompiledModel compiled_model;
  try {
    compiled_model = core->import_model(model_stream, device, import_configs);
  } catch (const std::exception& e) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure, e.what());
  }

  auto infer_request = compiled_model.create_infer_request();
  LITERT_LOG(LITERT_INFO, "Openvino InvocationContext Initialize SUCCESS");
  // TODO: add support for loading cached model
  return Ptr(new LiteRtDispatchInvocationContextT(infer_request, device_context,
                                                  num_inputs, num_outputs,
                                                  perf_count_enabled));
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type) {
  LiteRtTensorBufferType supported_tensor_buffer_types[] = {
      kLiteRtTensorBufferTypeOpenVINOTensorBuffer,
      // OpenVINO RemoteTensor doesn't support copy-free AHWB buffer. Until
      // it's supported, we use DMA-BUF.
      kLiteRtTensorBufferTypeDmaBuf,
      kLiteRtTensorBufferTypeAhwb,
  };

  int num_supported_tensor_buffer_types =
      sizeof(supported_tensor_buffer_types) /
      sizeof(supported_tensor_buffer_types[0]);
  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return litert::Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  auto status =
      device_context_.runtime_context()->create_tensor_buffer_requirements(
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
  LITERT_ASSIGN_OR_RETURN(ov::Tensor ov_tensor,
                          device_context_.getOVTensor(tensor_buffer_handle));
  // TODO: visit this if need to maintain graph indices for inputs and outputs
  // in dispatch_api
  infer_request_.set_input_tensor(graph_input_index, ov_tensor);
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_ASSIGN_OR_RETURN(ov::Tensor ov_tensor,
                          device_context_.getOVTensor(tensor_buffer_handle));
  // TODO: visit this if need to maintain graph indices for inputs and outputs
  // in dispatch_api
  infer_request_.set_output_tensor(graph_output_index, ov_tensor);
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::Invoke() {
  infer_request_.start_async();
  if (!infer_request_.wait_for(
          std::chrono::milliseconds(kInferRequestTimeoutMs)))
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "Failed to execute inference request due to timeout");

  if (perf_count_enabled_) {
    try {
      (void)infer_request_.get_profiling_info();
    } catch (const std::exception& e) {
      LITERT_LOG(LITERT_WARNING, "Failed to read OpenVINO profiling info: %s",
                 e.what());
    }
  }
  return {};
}
