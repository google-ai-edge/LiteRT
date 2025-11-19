# LiteRT Next C API

This folder contains C APIs of LiteRT Next.

The C API is LiteRT's most fundamental API and is primarily used to implement
the user-facing C++ API. The C API was developed to provide ABI stability.
This means that when the LiteRT runtime is distributed as
`libLiteRt.so`, it helps ensure that user applications remain
compatible and run smoothly even when using different versions of the runtime.

NOTE: C API is not the recommended option for general application developments.
User should use C++ API which is easier to use.

## C API naming guide

Function names follow this pattern:

- LiteRtCreate`<Object>`
  - e.g., `LiteRtCreateTensorBuffer(&tensor_buffer)`
- LiteRtDestroy`<Object>`
  - e.g., `LiteRtDestroyTensorBuffer(tensor_buffer)`
- LiteRt`<Get/Set><Object><Field>`
  - e.g., `LiteRtGetTensorBufferSize(tensor_buffer, &size)`
- LiteRt`<Verb><Object>`
  - e.g., `LiteRtLockTensorBuffer(tensor_buffer, &host_addr)`

Container accessors follow this pattern:

- LiteRtGetNum`<Object><Container>`
  - e.g., `LiteRtGetNumTensorProduces(tensor, &num_producers)`
- LiteRtGet`<Object><Item>`
  - e.g., `LiteRtGetTensorProducer(tensor, producer_index, &producer)`

Query function names follow this pattern:

- LiteRt`<Object><Has|Supports><Feature>`
  - e.g. `LiteRtEnvironmentHasGpuEnvironment(env, &has_gpu_env)`

Objects are managed through opaque handles, which are defined with an
`LITERT_DEFINE_HANDLE(ObjectHandle)` macro in [litert_common.h](litert_common.h).

- `LITERT_DEFINE_HANDLE(LiteRtTensorBuffer)`
  - `LiteRtTensorBuffer` defined as `typedef struct LiteRtTensorBufferT*`
- `LiteRtCreateTensorBuffer(LiteRtTensorBuffer* tensor_buffer)`
- `LiteRtDestroyTensorBuffer(LiteRtTensorBuffer tensor_buffer)`
  - Note the absence of `*` when passing a tensor_buffer

