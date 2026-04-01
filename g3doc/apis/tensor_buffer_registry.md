# Tensor Registration API

## Introduction

The primary goal of the Tensor Registration API is to provide a way for vendors
to extend the LiteRT TensorBuffer API to support custom hardware memory
allocation and operations natively.

> [!NOTE] Recent refactoring efforts have decoupled the API slightly. The goal
> of this refactoring was to limit direct LiteRT C API usage (such as reading
> environment options directly) from Accelerator or Dispatch API
> implementations. Now, accelerators can manage memory allocations and extract
> required resources (like device contexts and command queues) via direct
> parameters passed by the runtime.

## C API (litert/c/internal/litert_tensor_buffer_registry.h):

```c
LiteRtStatus LiteRtRegisterTensorBufferHandlers(
    LiteRtEnvironment env,                   // The LiteRT environment
    LiteRtTensorBufferType buffer_type,      // e.g., kLiteRtTensorBufferTypeWebGpuBuffer
    CreateCustomTensorBuffer create_func,    // Your custom buffer creation function
    DestroyCustomTensorBuffer destroy_func,  // Your custom destruction function
    LockCustomTensorBuffer lock_func,        // Your custom lock function
    UnlockCustomTensorBuffer unlock_func,    // Your custom unlock function
    ClearCustomTensorBuffer clear_func,      // Optional custom clear function
    ImportCustomTensorBuffer import_func,    // Optional custom import function
    LiteRtEnvOptionTag device_tag,           // Tag to read device_id from EnvironmentOptions
    LiteRtEnvOptionTag queue_tag             // Tag to read queue_id from EnvironmentOptions
);
```


## How It Enables Custom Buffer Creation

When one registers handlers for a buffer type, the system will use the custom
create function whenever someone requests that buffer type:

**Usage of `device_tag` and `queue_tag`:** The underlying buffer allocation
(especially on GPUs) often requires a device handle (like an OpenCL context or
WebGPU device) and a command queue. Instead of having buffer handlers directly
fetch `LiteRtEnvironmentOptions`, the `TensorBufferRegistry` extracts these
resources for you using `device_tag` and `queue_tag`. During creation or import,
the runtime looks up these tags in `LiteRtEnvironmentOptions`, retrieves the
values as handles, and passes them directly as `LiteRtGpuDeviceId device_id` and
`LiteRtGpuQueueId queue_id` over to your `create_func` and `import_func`. This
isolates the environment lookup logic from the handlers and limits their
reliance on the broader C API.

## Example from tests (litert/runtime/tensor_buffer_registry_test.cc):

```cpp
// Define your custom creation function
LiteRtStatus CreateMyCustomTensorBuffer(
    LiteRtGpuDeviceId device_id, LiteRtGpuQueueId queue_id,
    const LiteRtRankedTensorType* tensor_type,
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
    .unlock_func = UnlockMyCustomTensorBuffer,
    .clear_func = ClearMyCustomTensorBuffer,
    .import_func = ImportMyCustomTensorBuffer,
    .device_tag = kLiteRtEnvOptionTagWebGpuDevice,
    .queue_tag = kLiteRtEnvOptionTagWebGpuQueue,
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
