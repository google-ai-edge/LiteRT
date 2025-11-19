# Tensor Registration API

## C API (litert/c/internal/litert_tensor_buffer_registry.h):

```
LiteRtStatus LiteRtRegisterTensorBufferHandlers(
    LiteRtTensorBufferType buffer_type,     // e.g., kLiteRtTensorBufferTypeWebGpuBuffer
    CreateCustomTensorBuffer create_func,    // Your custom buffer creation function
    DestroyCustomTensorBuffer destroy_func,  // Your custom destruction function
    LockCustomTensorBuffer lock_func,        // Your custom lock function
    UnlockCustomTensorBuffer unlock_func     // Your custom unlock function
);
```

## C++ API (litert/runtime/tensor_buffer_registry.h):

```
litert::Expected<void> RegisterHandlers(
    LiteRtTensorBufferType buffer_type,
    const CustomTensorBufferHandlers& handlers
);

```

## How It Enables Custom Buffer Creation

When one registers handlers for a buffer type, the system will use the custom
create function whenever someone requests that buffer type:

## Example from tests (litert/runtime/tensor_buffer_registry_test.cc):

```
// Define your custom creation function
LiteRtStatus CreateMyCustomTensorBuffer(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, size_t packed_bytes,
    HwMemoryInfoPtr* hw_memory_info) {
  // Your custom buffer creation logic here
  auto memory_info = new CustomHwMemoryInfo{.bytes = bytes};
  memory_info->memory_handle = /* your hardware handle */;
  *hw_memory_info = memory_info;
  return kLiteRtStatusOk;
}

// Register it
registry.RegisterHandlers(kLiteRtTensorBufferTypeWebGpuBuffer, {
    .create_func = CreateMyCustomTensorBuffer,
    .destroy_func = DestroyMyCustomTensorBuffer,
    .lock_func = LockMyCustomTensorBuffer,
    .unlock_func = UnlockMyCustomTensorBuffer
});
```

The Complete Flow

1. Registration: Call LiteRtRegisterTensorBufferHandlers with your custom
   functions
1. Storage: Registry stores handlers in map: buffer_type → {create, destroy,
   lock, unlock}
1. Creation Request: User calls LiteRtCreateManagedTensorBuffer(buffer_type,
   ...)
1. Handler Lookup: CustomBuffer::Alloc retrieves your registered handlers
   (litert/runtime/custom_buffer.cc)
1. Invocation: Your create_func is called to create the buffer
   (custom_buffer.cc)
1. Usage: All operations (lock/unlock/destroy) use your registered handlers
