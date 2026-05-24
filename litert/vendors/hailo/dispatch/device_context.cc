#include "litert/vendors/hailo/dispatch/device_context.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"

litert::Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create(const LiteRtRuntimeContext* runtime_context) {
  if (runtime_context == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument, "Null runtime context");
  }

  // Create Hailo Virtual Device.
  auto vdevice_exp = hailort::VDevice::create();
  if (!vdevice_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Hailo VDevice: status = %d", vdevice_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to create Hailo VDevice");
  }

  auto vdevice = std::move(vdevice_exp.value());
  return Ptr(new LiteRtDispatchDeviceContextT(runtime_context, std::move(vdevice)));
}

litert::Expected<LiteRtTensorBufferHandle>
LiteRtDispatchDeviceContextT::RegisterTensorBuffer(LiteRtTensorBuffer tensor_buffer) {
  LiteRtTensorBufferType tensor_buffer_type;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_type(tensor_buffer, &tensor_buffer_type),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to get tensor buffer type"));

  if (tensor_buffer_type != kLiteRtTensorBufferTypeHostMemory) {
    LITERT_LOG(LITERT_ERROR, "Hailo NPU driver currently only supports host CPU buffers.");
    return litert::Unexpected(kLiteRtStatusErrorUnsupported, "Unsupported tensor buffer type");
  }

  size_t size;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_size(tensor_buffer, &size),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to get tensor buffer size"));

  void* host_memory_addr = nullptr;
  LITERT_RETURN_IF_ERROR(
      runtime_context_->get_tensor_buffer_host_memory(tensor_buffer, &host_memory_addr),
      litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to get host memory address"));

  LiteRtTensorBufferHandle handle = (LiteRtTensorBufferHandle)next_handle_++;
  registered_buffers_[handle] = RegisteredBuffer{.host_memory_addr = host_memory_addr, .size = size};

  return handle;
}

litert::Expected<void>
LiteRtDispatchDeviceContextT::UnregisterTensorBuffer(LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto it = registered_buffers_.find(tensor_buffer_handle);
  if (it == registered_buffers_.end()) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument, "Buffer handle not registered");
  }
  registered_buffers_.erase(it);
  return {};
}
