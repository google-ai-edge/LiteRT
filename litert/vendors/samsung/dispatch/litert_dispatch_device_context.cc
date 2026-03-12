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


#include "litert/vendors/samsung/dispatch/litert_dispatch_device_context.h"

LiteRtDispatchDeviceContextT::LiteRtDispatchDeviceContextT(
    const litert::samsung::EnnManager *enn_manager)
    : enn_manager_(enn_manager), tensor_buffer_registry_(enn_manager) {}

litert::Expected<LiteRtDispatchDeviceContextT::UniquePtr>
LiteRtDispatchDeviceContextT::Create(
    const litert::samsung::EnnManager *enn_manager) {
  return LiteRtDispatchDeviceContextT::UniquePtr(
      new LiteRtDispatchDeviceContextT(enn_manager));
}

litert::Expected<LiteRtTensorBufferHandle>
LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer) {
  LITERT_LOG(LITERT_DEBUG, "Registering tensor buffer %p", tensor_buffer);
  LiteRtTensorBufferType tensor_buffer_type;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferType(tensor_buffer, &tensor_buffer_type));

  if (tensor_buffer_type != kLiteRtTensorBufferTypeDmaBuf) {
    return litert::Error(
        kLiteRtStatusErrorUnsupported,
        absl::StrFormat("Unsupported buffer type %d", tensor_buffer_type));
  }

  size_t tensor_buffer_size;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size));

  size_t tensor_buffer_offset;
  if (auto status =
          LiteRtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset);
      status != kLiteRtStatusOk) {
    return litert::Unexpected(status, "Failed to get tensor buffer offset");
  }

  LiteRtRankedTensorType tensor_type;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type));

  if (tensor_type.layout.has_strides) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Tensor strides are not supported");
  }

  int fd;
  void *addr;

  switch (tensor_buffer_type) {
  case kLiteRtTensorBufferTypeDmaBuf:
#if LITERT_HAS_DMABUF_SUPPORT
    if (auto status =
            LiteRtGetTensorBufferDmaBufBuffer(tensor_buffer, &addr, &fd);
        status != kLiteRtStatusOk) {
      return litert::Error(status, "Failed to get DMA-BUF");
    }
#else
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "DMA-BUF is not supported on this platform");
#endif // LITERT_HAS_DMABUF_SUPPORT
    break;

  default:
    LITERT_LOG(LITERT_ERROR, "Unsupported buffer type: %d", tensor_buffer_type);
    return litert::Unexpected(kLiteRtStatusErrorUnsupported);
  }
  EnnBufferPtr enn_buffer_p;
  if (enn_manager_->Api().EnnCreateBufferFromFdWithOffset(
          fd, tensor_buffer_size, tensor_buffer_offset, &enn_buffer_p) !=
      ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to create buffer from fd");
  }

  return tensor_buffer_registry_.Register(enn_buffer_p);
}

LiteRtDispatchDeviceContextT::EnnBufferRegistry::~EnnBufferRegistry() {}

LiteRtTensorBufferHandle
LiteRtDispatchDeviceContextT::EnnBufferRegistry::Register(
    EnnBufferPtr enn_buffer) {
  int assign_handle = -1;
  for (auto i = 0; i < buffers_.size(); ++i) {
    if (buffers_[i] == nullptr) {
      assign_handle = i;
      break;
    }
  }
  if (assign_handle < 0) {
    assign_handle = buffers_.size();
    buffers_.push_back(EnnBufferPtr{nullptr});
  }
  std::swap(buffers_[assign_handle], enn_buffer);

  return assign_handle;
}

litert::Expected<void>
LiteRtDispatchDeviceContextT::EnnBufferRegistry::Unregister(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto buffer = Find(tensor_buffer_handle);
  if (!buffer) {
    return buffer.Error();
  }

  if (enn_manager_->Api().EnnReleaseBuffer(buffer.Value()) != ENN_RET_SUCCESS) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Fail to release buffer");
  };
  buffers_[tensor_buffer_handle] = nullptr;
  return {};
}

litert::Expected<EnnBufferPtr>
LiteRtDispatchDeviceContextT::EnnBufferRegistry::Find(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (tensor_buffer_handle >= buffers_.size() ||
      buffers_[tensor_buffer_handle] == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Invalid tensor buffer handle");
  }
  return buffers_[tensor_buffer_handle];
}
