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

#include "openvino/core/any.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/c/internal/litert_logging.h"
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
  std::string device = "NPU";  // Default device
  if (intel_openvino_opts) {
    const auto& intel_opts = *intel_openvino_opts;
    // Configure device type
    auto device_type = intel_opts.GetDeviceType();
    switch (device_type) {
      case kLiteRtIntelOpenVinoDeviceTypeCPU:
        device = "CPU";
        break;
      case kLiteRtIntelOpenVinoDeviceTypeGPU:
        device = "GPU";
        break;
      case kLiteRtIntelOpenVinoDeviceTypeNPU:
        device = "NPU";
        break;
      case kLiteRtIntelOpenVinoDeviceTypeAUTO:
        return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                             "AUTO device type is not supported");
    }
  }
  LITERT_LOG(LITERT_INFO, "Using Intel OpenVINO device: %s", device.c_str());

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
  auto core = device_context.getCore();
  if (!core) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get OpenVINO core from device context");
  }
  ov::CompiledModel compiled_model;
  try {
    compiled_model = core->import_model(model_stream, device);
  } catch (const std::exception& e) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure, e.what());
  }

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
  return {};
}
