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

#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#if __ANDROID__
#include <android/hardware_buffer.h>
#endif  // __ANDROID__

#include <string.h>

#include <cstdint>
#include <vector>

#include "litert/c/litert_layout.h"

#if LITERT_HAS_AHWB_SUPPORT
#include <sys/socket.h>
#include <unistd.h>
#endif  // LITERT_HAS_AHWB_SUPPORT

#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/remote_context.hpp>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/vendors/intel_openvino/dispatch/openvino_tensor_buffer.h"
#include "litert/vendors/intel_openvino/utils.h"

#if LITERT_HAS_AHWB_SUPPORT || LITERT_HAS_DMABUF_SUPPORT
#include <limits>
#include "openvino/core/type/element_type.hpp"
#endif

litert::Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create(
    const LiteRtRuntimeContext* runtime_context) {
  return Ptr(new LiteRtDispatchDeviceContextT(runtime_context));
}

#if LITERT_HAS_AHWB_SUPPORT
// Extracts the backing dma-buf fd of an AHardwareBuffer by serializing the
// buffer's native handle over a socketpair and reading it back with
// recvmsg(SCM_RIGHTS). Ownership of the returned fd transfers to the caller,
// which must close() it.
//
// An AHardwareBuffer handle can carry more than one fd: for a BLOB buffer the
// kernel installs the backing dma-buf plus a secondary metadata/reserved-region
// fd, and other formats can carry more. This shapes the function in two ways:
//   1. The control buffer is sized for a generous upper bound on the fd count
//      so the kernel never truncates the array. Any fds that do not fit are
//      dropped by the kernel before we ever see them and can no longer be
//      closed.
//   2. The caller only needs the first fd (the dma-buf), but every fd the
//      kernel installed is a live descriptor owned by this process. The
//      function keeps the first and closes all the rest, regardless of how many
//      the handle carried, so this process leaves none open.
litert::Expected<int> GetFdFromUnixHandle(AHardwareBuffer* ahwb) {
  int socks[2];
  if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, socks) != 0) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create socket pair");
  }

  auto socket_cleanup = absl::Cleanup([&socks] {
    close(socks[0]);
    close(socks[1]);
  });

#if defined(__ANDROID__)
#if __ANDROID_API__ >= 26
  if (AHardwareBuffer_sendHandleToUnixSocket(ahwb, socks[0]) != 0) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to send handle to unix socket");
  }
#else
  return litert::Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      "AHardwareBuffer_sendHandleToUnixSocket requires API level 26");
#endif  // __ANDROID_API__ >= 26
#else
  return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                            "AHWB is not supported on this platform");
#endif  // defined(__ANDROID__)

  // A single dummy payload byte travels alongside the ancillary fd array; the
  // sender writes one byte together with the SCM_RIGHTS control message.
  char payload_byte = 0;
  struct iovec io = {.iov_base = &payload_byte,
                     .iov_len = sizeof(payload_byte)};

  // Control buffer sized for a generous upper bound on the number of fds a
  // single handle can carry, so the kernel never truncates the fd array.
  constexpr int kMaxHandleFds = 128;
  char control_buf[CMSG_SPACE(sizeof(int) * kMaxHandleFds)];
  memset(control_buf, 0, sizeof(control_buf));

  struct msghdr msg = {.msg_iov = &io,
                       .msg_iovlen = 1,
                       .msg_control = control_buf,
                       .msg_controllen = sizeof(control_buf)};

  if (recvmsg(socks[1], &msg, 0) < 0) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to receive socket message");
  }

  // Keep the first fd (the dma-buf handed to the importer) and close every
  // other fd the kernel installed, so this process leaves none open.
  int dma_buf_fd = -1;
  for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr;
       cmsg = CMSG_NXTHDR(&msg, cmsg)) {
    if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
      continue;
    }
    const int num_fds =
        static_cast<int>((cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int));
    const int* fds = reinterpret_cast<const int*>(CMSG_DATA(cmsg));
    for (int i = 0; i < num_fds; ++i) {
      if (dma_buf_fd < 0) {
        dma_buf_fd = fds[i];  // Keep the first fd: the dma-buf.
      } else {
        close(fds[i]);  // Extra fds are not needed by the caller.
      }
    }
  }

  if (dma_buf_fd < 0) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Socket message carried no dma-buf fd");
  }
  return dma_buf_fd;
}
#endif  // LITERT_HAS_AHWB_SUPPORT

namespace {

#if LITERT_HAS_AHWB_SUPPORT || LITERT_HAS_DMABUF_SUPPORT
// Builds an OpenVINO shape from an untrusted LiteRtRankedTensorType and
// validates that the resulting tensor fits within the host buffer of
// `buffer_size` bytes.
litert::Expected<std::vector<int32_t>> BuildAndValidateShape(
    const LiteRtRankedTensorType& tensor_type, size_t element_bit_width,
    size_t buffer_size) {
  const auto& layout = tensor_type.layout;

  // `rank` is a 7-bit field, but guard explicitly against the array bound so a
  // crafted value can never index past `dimensions[LITERT_TENSOR_MAX_RANK]`.
  if (layout.rank > LITERT_TENSOR_MAX_RANK) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Tensor rank exceeds maximum supported rank");
  }

  if (element_bit_width == 0) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Unsupported/unknown tensor element type");
  }

  std::vector<int32_t> ov_shape_vec(layout.rank);
  // Number of elements implied by the declared shape, computed in 64-bit with
  // overflow checks before it is multiplied by the element size.
  uint64_t num_elements = 1;
  for (uint32_t i = 0; i < layout.rank; ++i) {
    const int32_t dim = layout.dimensions[i];
    if (dim <= 0) {
      return litert::Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          "Dynamic/negative/zero tensor dimensions are not supported");
    }
    ov_shape_vec[i] = dim;
    if (num_elements != 0 &&
        static_cast<uint64_t>(dim) >
            std::numeric_limits<uint64_t>::max() / num_elements) {
      return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                                "Tensor element count overflow");
    }
    num_elements *= static_cast<uint64_t>(dim);
  }

  if (num_elements != 0 &&
      static_cast<uint64_t>(element_bit_width) >
          std::numeric_limits<uint64_t>::max() / num_elements) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Tensor byte size overflow");
  }
  const uint64_t required_bits = num_elements * element_bit_width;
  const uint64_t required_bytes = (required_bits + 7) / 8;

  if (required_bytes > buffer_size) {
    return litert::Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Declared tensor shape exceeds the mapped buffer size");
  }

  return ov_shape_vec;
}
#endif  // LITERT_HAS_AHWB_SUPPORT || LITERT_HAS_DMABUF_SUPPORT

}  // namespace

litert::Expected<LiteRtTensorBufferHandle>
LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer) {
  LiteRtTensorBufferType tensor_buffer_type;

  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_type(tensor_buffer,
                                               &tensor_buffer_type),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer type"));

  size_t tensor_buffer_size;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_size(tensor_buffer,
                                               &tensor_buffer_size),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer size"));

  size_t tensor_buffer_offset;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_offset(tensor_buffer,
                                                 &tensor_buffer_offset),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer offset"));

  LiteRtRankedTensorType tensor_type;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_tensor_type(tensor_buffer,
                                                      &tensor_type),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get tensor buffer's type"));
  LITERT_RETURN_IF_ERROR(
      !tensor_type.layout.has_strides,
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                         "Tensor strides are not supported"));

  switch (tensor_buffer_type) {
    case kLiteRtTensorBufferTypeOpenVINOTensorBuffer: {
      HwMemoryHandle hw_memory_handle;
      LITERT_RETURN_IF_ERROR(
          runtime_context_->get_tensor_buffer_custom_tensor_buffer_handle(
              tensor_buffer, &hw_memory_handle),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get OpenVINO tensor buffer."));
      OpenVinoTensorBuffer* custom_tensor_buffer =
          reinterpret_cast<OpenVinoTensorBuffer*>(hw_memory_handle);

      LITERT_ASSIGN_OR_RETURN(auto openvino_tensor,
                              custom_tensor_buffer->GetOVTensor());
      return InsertTensor(RegisteredTensor{.tensor = openvino_tensor});
    }
    case kLiteRtTensorBufferTypeDmaBuf: {
#if LITERT_HAS_DMABUF_SUPPORT
      ov::element::Type ov_element_type =
          litert::openvino::MapLiteTypeToOV(tensor_type.element_type);
      int buffer_fd;
      void* buffer_host_addr;
      LITERT_RETURN_IF_ERROR(
          runtime_context_->get_tensor_buffer_dma_buf_buffer(
              tensor_buffer, &buffer_host_addr, &buffer_fd),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get DMA-BUF buffer"));
      auto context = getCore()
                         ->get_default_context("NPU")
                         .as<ov::intel_npu::level_zero::ZeroContext>();
      LITERT_ASSIGN_OR_RETURN(
          std::vector<int32_t> ov_shape_vec,
          BuildAndValidateShape(tensor_type, ov_element_type.bitwidth(),
                                tensor_buffer_size));

      auto remote_tensor = context.create_tensor(
          ov_element_type, ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()},
          buffer_fd);
      return InsertTensor(RegisteredTensor{.tensor = remote_tensor});

#else
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "DmaBuf support is missing on this platform");
#endif  // LRT_HAS_DMABUF_SUPPORT
      break;
    }

    case kLiteRtTensorBufferTypeAhwb: {
#if LITERT_HAS_AHWB_SUPPORT
      ov::element::Type ov_element_type =
          litert::openvino::MapLiteTypeToOV(tensor_type.element_type);
      AHardwareBuffer* ahwb;
      LITERT_RETURN_IF_ERROR(
          runtime_context_->get_tensor_buffer_ahwb(tensor_buffer, &ahwb),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get LiteRT Tensor Buffer for AHWB"));

      LITERT_ASSIGN_OR_RETURN(
          std::vector<int32_t> ov_shape_vec,
          BuildAndValidateShape(tensor_type, ov_element_type.bitwidth(),
                                tensor_buffer_size));

      // Borrow the AHardwareBuffer's backing dma-buf fd. GetFdFromUnixHandle
      // returns a single fd owned by this scope; it is closed once mapped.
      LITERT_ASSIGN_OR_RETURN(int fd, GetFdFromUnixHandle(ahwb));

      auto context = getCore()
                         ->get_default_context("NPU")
                         .as<ov::intel_npu::level_zero::ZeroContext>();
      // The mmap'd region is independent of the fd once mapped, so we close the
      // fd as soon as the mapping is established.
      void* buffer = mmap(nullptr, tensor_buffer_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, tensor_buffer_offset);
      close(fd);
      if (buffer == MAP_FAILED) {
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "MMAP failed for tensor buffer");
      }
      CleanupAction cleanup([buffer, tensor_buffer_size]() {
        munmap(buffer, tensor_buffer_size);
      });
      ov::Tensor ov_tensor(ov_element_type,
                           ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()},
                           buffer);
      return InsertTensor(
          RegisteredTensor{.tensor = ov_tensor, .cleanup = std::move(cleanup)});

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
  absl::MutexLock lock(&tensor_handle_mutex_);
  auto it = tensor_handle_map_.find(tensor_buffer_handle);
  if (it != tensor_handle_map_.end()) {
    tensor_handle_map_.erase(tensor_buffer_handle);
  } else {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to Unregister Tensor Buffer");
  }

  return {};
}
