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

#include "litert/vendors/intel_openvino/dispatch/device_context.h"

#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#if __ANDROID__
#include <android/hardware_buffer.h>
#endif  // __ANDROID__

#include <string.h>

#include <cstdint>
#include <vector>

#if LITERT_HAS_AHWB_SUPPORT
#include <sys/socket.h>
#include <unistd.h>
#endif  // LITERT_HAS_AHWB_SUPPORT

#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/remote_context.hpp>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/vendors/intel_openvino/utils.h"

litert::Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create() {
  return Ptr(new LiteRtDispatchDeviceContextT());
}

#if LITERT_HAS_AHWB_SUPPORT
litert::Expected<int> GetFdFromUnixHandle(AHardwareBuffer *ahwb) {
  int socks[2];
  if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, socks) != 0) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create socket pair");
  }

  auto socket_cleaup = absl::Cleanup([&socks] {
    close(socks[0]);
    close(socks[1]);
  });

  if (AHardwareBuffer_sendHandleToUnixSocket(ahwb, socks[0]) != 0) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to send handle to unix socket");
  }
  // Receives a fd(an int) over the unix socket, sets up control buffer to
  // receive an int
  char payload_byte;
  struct iovec io = {.iov_base = &payload_byte,
                     .iov_len = sizeof(payload_byte)};

  // Buffer for receiving fd
  char control_buf[CMSG_SPACE(sizeof(int))];

  struct msghdr msg = {.msg_iov = &io,
                       .msg_iovlen = 1,
                       .msg_control = control_buf,
                       .msg_controllen = sizeof(control_buf)};

  if (recvmsg(socks[1], &msg, 0) < 0) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to receive socket message");
  }
  int fd = -1;
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
  }

  return fd;
}
#endif  // LITERT_HAS_AHWB_SUPPORT

litert::Expected<LiteRtTensorBufferHandle>
LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer) {
  LiteRtTensorBufferType tensor_buffer_type;

  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferType(tensor_buffer, &tensor_buffer_type),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer type"));

  size_t tensor_buffer_size;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer size"));

  size_t tensor_buffer_offset;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer offset"));

  LiteRtRankedTensorType tensor_type;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer's type"));
  LITERT_RETURN_IF_ERROR(
      !tensor_type.layout.has_strides,
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Tensor strides are not supported"));

  ov::element::Type ov_element_type =
      litert::openvino::MapLiteTypeToOV(tensor_type.element_type);
  switch (tensor_buffer_type) {
    case kLiteRtTensorBufferTypeHostMemory: {
      void* buffer_host_addr;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferHostMemory(tensor_buffer, &buffer_host_addr),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get HostMemory buffer"));

      auto context = core_->get_default_context("NPU")
                         .as<ov::intel_npu::level_zero::ZeroContext>();
      std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
      for (int i = 0; i < ov_shape_vec.size(); i++)
        ov_shape_vec[i] = tensor_type.layout.dimensions[i];

      auto remote_tensor = context.create_l0_host_tensor(
          ov_element_type, ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()});
      memcpy(remote_tensor.get(), buffer_host_addr, tensor_buffer_size);
      tensor_handle_map_.emplace((LiteRtTensorBufferHandle)next_handle_,
                                 remote_tensor);
      return next_handle_++;
    }
    case kLiteRtTensorBufferTypeDmaBuf: {
#if LITERT_HAS_DMABUF_SUPPORT
      int buffer_fd;
      void *buffer_host_addr;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferDmaBufBuffer(tensor_buffer, &buffer_host_addr,
                                            &buffer_fd),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get DMA-BUF buffer"));

      auto mmap_handle = mmap(NULL, tensor_buffer_size, PROT_WRITE | PROT_READ,
                              MAP_SHARED, buffer_fd, tensor_buffer_offset);

      if (mmap_handle == MAP_FAILED)
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "MMAP failed for tensor buffer");

      auto context = core_->get_default_context("NPU")
                         .as<ov::intel_npu::level_zero::ZeroContext>();
      std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
      for (int i = 0; i < ov_shape_vec.size(); i++)
        ov_shape_vec[i] = tensor_type.layout.dimensions[i];

      // TODO: change f32 to ov_element_type fetched from TensorType
      auto remote_tensor = context.create_tensor(
          ov_element_type, ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()},
          buffer_fd);
      tensor_handle_map_.emplace((LiteRtTensorBufferHandle)next_handle_,
                                 remote_tensor);
      return next_handle_++;

#else
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "DmaBuf support is missing on this platform");
#endif  // LRT_HAS_DMABUF_SUPPORT
      break;
    }

    case kLiteRtTensorBufferTypeAhwb: {
#if LITERT_HAS_AHWB_SUPPORT
      AHardwareBuffer *ahwb;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferAhwb(tensor_buffer, &ahwb),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get LiteRT Tensor Buffer for AHWB"));

      auto fd_exp = GetFdFromUnixHandle(ahwb);
      int fd = fd_exp.Value();
      LITERT_RETURN_IF_ERROR(
          fd != -1, litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                       "Failed to get FD from unix handle"));

      std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
      for (int i = 0; i < ov_shape_vec.size(); i++)
        ov_shape_vec[i] = tensor_type.layout.dimensions[i];
      auto context = core_->get_default_context("NPU")
                         .as<ov::intel_npu::level_zero::ZeroContext>();
      void* buffer = mmap(nullptr, tensor_buffer_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, tensor_buffer_offset);
      ov::Tensor ov_tensor(ov_element_type,
                           ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()},
                           buffer);
      tensor_handle_map_.emplace((LiteRtTensorBufferHandle)next_handle_,
                                 ov_tensor);
      return next_handle_++;

#else
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "AHWB support is missing on this platform");
#endif  // LITERT_HAS_AHWB_SUPPORT
      break;
    }

    default:
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Unsupported tensor buffer type");
  }
}

litert::Expected<void> LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto it = tensor_handle_map_.find(tensor_buffer_handle);
  if (it != tensor_handle_map_.end()) {
    tensor_handle_map_.erase(tensor_buffer_handle);
  } else {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to Unregister Tensor Buffer");
  }

  return {};
}
