// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/intel_openvino/dispatch/device_context.h"
#if __ANDROID__
#include <android/hardware_buffer.h>
#endif  // __ANDROID__

#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/remote_context.hpp>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"

litert::Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create(ov::Core core) {
  return Ptr(new LiteRtDispatchDeviceContextT(core));
}

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

  int buffer_fd;

  switch (tensor_buffer_type) {
    case kLiteRtTensorBufferTypeDmaBuf:
#if LITERT_HAS_DMABUF_SUPPORT
      void *buffer_host_addr;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferDmaBufBuffer(tensor_buffer, &buffer_host_addr,
                                            &buffer_fd),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get DMA-BUF buffer"));
#else
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "DmaBuf support is missing on this platform");
#endif  // LRT_HAS_DMABUF_SUPPORT
      break;
    case kLiteRtTensorBufferTypeAhwb:
#if LITERT_HAS_AHWB_SUPPORT
      AHardwareBuffer *ahwb;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetTensorBufferAhwb(tensor_buffer, &ahwb),
          litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                             "Failed to get LiteRT Tensor Buffer for AHWB"));
      AHardwareBuffer_acquire(ahwb);
      if (ahwb == NULL)
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "Failed to acquire AHWB");

        // TODO: Change below APIs to use AHardwareBuffer_lock
        /*    const native_handle_t *buffer_handle =
           AHardwareBuffer_getNativeHandle(ahwb); if (buffer_handle == nullptr)
                return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                          "Native handle for AHWB is NULL");
            if (buffer_handle->numFds != 1)
                return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                          "AHWB cannot be mapped");

            buffer_fd = buffer_handle->data[0];
            if (buffer_fd == -1)
                return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                          "Failed to get file descriptor for
           AHWB buffer");
         */

#else
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "AHWB support is missing on this platform");
#endif  // LITERT_HAS_AHWB_SUPPORT
      break;

    default:
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Unsupported tensor buffer type");
  }

  auto mmap_handle = mmap(NULL, tensor_buffer_size, PROT_WRITE | PROT_READ,
                          MAP_SHARED, buffer_fd, tensor_buffer_offset);

  LITERT_RETURN_IF_ERROR(mmap_handle == MAP_FAILED,
                         litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                            "MMAP failed for tensor buffer"));

  auto context = core_.get_default_context("NPU")
                     .as<ov::intel_npu::level_zero::ZeroContext>();
  std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
  for (int i = 0; i < ov_shape_vec.size(); i++)
    ov_shape_vec[i] = tensor_type.layout.dimensions[i];

  // TODO: change f32 to ov_element_type fetched from TensorType
  auto remote_tensor = context.create_tensor(
      ov::element::f32, ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()},
      buffer_fd);
  tensor_handle_map_.emplace((LiteRtTensorBufferHandle)mmap_handle,
                             remote_tensor);

  return (uint64_t)mmap_handle;
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
