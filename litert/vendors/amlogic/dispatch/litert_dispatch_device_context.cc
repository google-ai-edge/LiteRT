// Copyright 2026 Amlogic, Inc.
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

#include "litert/vendors/amlogic/dispatch/litert_dispatch_device_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"

using litert::Expected;
using litert::Unexpected;

LiteRtDispatchDeviceContextT::~LiteRtDispatchDeviceContextT() = default;

Expected<LiteRtDispatchDeviceContextT::Ptr> LiteRtDispatchDeviceContextT::Create() {
  LITERT_LOG(LITERT_DEBUG, "[AML] LiteRtDispatchDeviceContextT Enter");
  return Ptr(new LiteRtDispatchDeviceContextT());
}

Expected<LiteRtTensorBuffer> LiteRtDispatchDeviceContextT::GetTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_LOG(LITERT_DEBUG, "[AML] GetTensorBuffer Enter");
  auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
  if (!registry_entry) {
    return Unexpected(registry_entry.Error());
  }

  return (*registry_entry)->tensor_buffer;
}

// Expected<void *> LiteRtDispatchDeviceContextT::GetMemHandle(
//     LiteRtTensorBufferHandle tensor_buffer_handle)
// {
//   auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
//   if (!registry_entry)
//   {
//     return Unexpected(registry_entry.Error());
//   }

//   return (*registry_entry)->tensor_buffer->data;
// }

Expected<LiteRtTensorBufferHandle> LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer) {
  LITERT_LOG(LITERT_DEBUG, "Registering tensor buffer %p", tensor_buffer);
  LiteRtTensorBufferType tensor_buffer_type;
  if (auto status =
          LiteRtGetTensorBufferType(tensor_buffer, &tensor_buffer_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer type");
  }
  LiteRtRankedTensorType tensor_type;
  if (auto status =
          LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type);
      status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to get tensor buffer's type");
  }

  if (tensor_type.layout.has_strides) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported");
  }

  void *buffer_host_addr;
  (void)buffer_host_addr;

  switch (tensor_buffer_type) {
  case kLiteRtTensorBufferTypeHostMemory:
    if (auto status =
            LiteRtGetTensorBufferHostMemory(tensor_buffer, &buffer_host_addr);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to get host memory buffer");
    }
    break;

  case kLiteRtTensorBufferTypeFastRpc:
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "FastRPC tensor buffers are not supported by AML");

  case kLiteRtTensorBufferTypeDmaBuf:
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "DMA-BUF tensor buffers are not supported by AML");

  default:
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unsupported tensor buffer type");
  }

  return tensor_buffer_registry_.Register(
      TensorBufferRegistryEntry(tensor_buffer));
}

litert::Expected<void> LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_buffer,
                          GetTensorBuffer(tensor_buffer_handle));
  LITERT_LOG(LITERT_DEBUG, "Unregistering tensor buffer %p", tensor_buffer);
  LITERT_RETURN_IF_ERROR(
      tensor_buffer_registry_.Unregister(tensor_buffer_handle));
  return {};
}
