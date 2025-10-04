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

#include "litert/vendors/qualcomm/dispatch/litert_dispatch_device_context.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "HTP/QnnHtpMem.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnMem.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

using litert::Expected;
using litert::Unexpected;
using litert::qnn::QnnManager;

Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create(QnnManager& qnn) {
  return Ptr(new LiteRtDispatchDeviceContextT(qnn));
}

Expected<LiteRtTensorBuffer> LiteRtDispatchDeviceContextT::GetTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
  if (!registry_entry) {
    return Unexpected(registry_entry.Error());
  }

  return (*registry_entry)->tensor_buffer;
}

Expected<Qnn_MemHandle_t> LiteRtDispatchDeviceContextT::GetMemHandle(
    LiteRtTensorBufferHandle tensor_buffer_handle, const Qnn_Tensor_t& tensor) {
  auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
  if (!registry_entry) {
    return Unexpected(registry_entry.Error());
  }

  if (!(*registry_entry)->qnn_mem_handle) {
    auto qnn_mem_handle =
        RegisterTensorBuffer((*registry_entry)->tensor_buffer, tensor);
    if (!qnn_mem_handle) {
      return Unexpected(qnn_mem_handle.Error());
    }
    (*registry_entry)->qnn_mem_handle = *qnn_mem_handle;
  }

  return (*registry_entry)->qnn_mem_handle;
}

Expected<Qnn_MemHandle_t> LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer, const Qnn_Tensor_t& tensor) {
  LITERT_LOG(LITERT_DEBUG, "Registering tensor buffer %p", tensor_buffer);
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status =
          LiteRtGetTensorBufferType(tensor_buffer, &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer type");
  }

  size_t tensor_buffer_size;
  if (auto status =
          LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer size");
  }

  size_t tensor_buffer_offset;
  if (auto status =
          LiteRtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer offset");
  }

  LiteRtRankedTensorType tensor_type;
  if (auto status =
          LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer's type");
  }

  auto element_type =
      static_cast<enum litert::ElementType>(tensor_type.element_type);
  Qnn_DataType_t tensor_data_type;
  if (auto status = LegalizeElementType(element_type, &tensor_data_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to legalize datatype");
  }

  uint32_t tensor_rank = tensor_type.layout.rank;
  uint32_t* tensor_dimensions = reinterpret_cast<uint32_t*>(
      const_cast<int32_t*>(tensor_type.layout.dimensions));
  if (tensor_type.layout.has_strides) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported by QNN");
  }

  void* buffer_host_addr;
  int buffer_fd;
  (void)buffer_host_addr;

  switch (tensor_buffer_type) {
    case kLiteRtTensorBufferTypeFastRpc:
#if LITERT_HAS_FASTRPC_SUPPORT
      if (auto status = LiteRtGetTensorBufferFastRpcBuffer(
              tensor_buffer, &buffer_host_addr, &buffer_fd);
          status != kLiteRtStatusOk) {
        return Unexpected(status, "Failed to get FastRPC buffer");
      }
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "FastRPC support is missing on this platform");
#endif  // LRT_HAS_FASTRPC_SUPPORT
      break;

    case kLiteRtTensorBufferTypeDmaBuf:
#if LITERT_HAS_DMABUF_SUPPORT
      if (auto status = LiteRtGetTensorBufferDmaBufBuffer(
              tensor_buffer, &buffer_host_addr, &buffer_fd);
          status != kLiteRtStatusOk) {
        return Unexpected(status, "Failed to get DMA-BUF buffer");
      }
#else
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "DmaBuf support is missing on this platform");
#endif  // LRT_HAS_DMABUF_SUPPORT
      break;

    default:
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Unsupported tensor buffer type");
  }

  QnnMemHtp_Descriptor_t mem_htp_descriptor = {};
  mem_htp_descriptor.type = QNN_HTP_MEM_SHARED_BUFFER;
  mem_htp_descriptor.size = tensor_buffer_size;
  mem_htp_descriptor.sharedBufferConfig =
      QnnHtpMem_SharedBufferConfig_t{buffer_fd, tensor_buffer_offset};

  Qnn_MemDescriptor_t mem_descriptor = {};
  // QNN does not support 0-dimensional tensors.
  std::array<uint32_t, 1> dim{1};
  if (tensor_rank == 0) {
    mem_descriptor.memShape = {1, dim.data(), nullptr};
  } else {
    mem_descriptor.memShape = {tensor_rank, tensor_dimensions, nullptr};
  }
  mem_descriptor.dataType = tensor_data_type;
  mem_descriptor.memType = QNN_MEM_TYPE_CUSTOM;
  mem_descriptor.customInfo = &mem_htp_descriptor;

  if (invocation_context_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Missing invocation context");
  }

  Qnn_ContextHandle_t context_handle = invocation_context_->ContextHandle();

  Qnn_MemHandle_t mem_handle = nullptr;
  if (auto status = qnn_manager_.Api()->memRegister(
          context_handle, &mem_descriptor, 1UL, &mem_handle);
      status != QNN_SUCCESS) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Failed to register tensor buffer, QNN error code: %d",
                        status));
  }

  if (!mem_handle) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to register buffer: null mem_handle");
  }

  return mem_handle;
}

litert::Expected<void> LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle, const Qnn_Tensor_t& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_buffer,
                          GetTensorBuffer(tensor_buffer_handle));
  LITERT_LOG(LITERT_DEBUG, "Unregistering tensor buffer %p", tensor_buffer);
  LITERT_RETURN_IF_ERROR(
      tensor_buffer_registry_.Unregister(tensor_buffer_handle));
  LITERT_ASSIGN_OR_RETURN(auto mem_handle,
                          GetMemHandle(tensor_buffer_handle, tensor));
  if (auto status = qnn_manager_.Api()->memDeRegister(&mem_handle, 1UL);
      status != QNN_SUCCESS) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat(
            "Failed to unregister tensor buffer, QNN error code: %d", status));
  }
  return {};
}
