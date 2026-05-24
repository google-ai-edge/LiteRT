#ifndef ODML_LITERT_LITERT_VENDORS_HAILO_DISPATCH_DEVICE_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_HAILO_DISPATCH_DEVICE_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"

#include "hailo/hailort.hpp"

class LiteRtDispatchDeviceContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchDeviceContextT>;

  ~LiteRtDispatchDeviceContextT() = default;

  static litert::Expected<Ptr> Create(const LiteRtRuntimeContext* runtime_context);

  litert::Expected<LiteRtTensorBufferHandle> RegisterTensorBuffer(LiteRtTensorBuffer tensor_buffer);
  litert::Expected<void> UnregisterTensorBuffer(LiteRtTensorBufferHandle tensor_buffer_handle);

  litert::Expected<void*> GetHostMemoryAddress(LiteRtTensorBufferHandle handle) const {
    auto it = registered_buffers_.find(handle);
    if (it != registered_buffers_.end()) {
      return it->second.host_memory_addr;
    }
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Buffer handle not registered");
  }

  const LiteRtRuntimeContext* runtime_context() const { return runtime_context_; }
  hailort::VDevice& vdevice() { return *vdevice_; }

 private:
  struct RegisteredBuffer {
    void* host_memory_addr;
    size_t size;
  };

  explicit LiteRtDispatchDeviceContextT(
      const LiteRtRuntimeContext* runtime_context,
      std::unique_ptr<hailort::VDevice> vdevice)
      : runtime_context_(runtime_context),
        vdevice_(std::move(vdevice)),
        next_handle_(1) {}

  const LiteRtRuntimeContext* runtime_context_;
  std::unique_ptr<hailort::VDevice> vdevice_;
  std::unordered_map<LiteRtTensorBufferHandle, RegisteredBuffer> registered_buffers_;
  uint64_t next_handle_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_HAILO_DISPATCH_DEVICE_CONTEXT_H_
