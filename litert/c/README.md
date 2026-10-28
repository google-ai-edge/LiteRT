# LiteRT Next C API

This folder contains LiteRT "Compiled Model" C APIs.

The C API is LiteRT's most fundamental API and is primarily used to implement
the user-facing C++ API. The C API was developed to provide ABI stability.
This means that when the LiteRT runtime is distributed as
`libLiteRt.so`, it helps ensure that user applications remain
compatible and run smoothly even when using different versions of the runtime.

NOTE: the C API is not the recommended option for general application
development. Most developers should use the C++ API which is easier to use.

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
For instructions on adding a new C API function and updating the runtime method
table, symbol lists, and C++ `RuntimeProxy`, see [`../C_API.md`](../C_API.md).

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

## Types in C API: Opaque Types vs Concrete Types

For comprehensive guidelines, see [C_API.md](../C_API.md).

### Opaque Types (Recommended)

Objects are managed through opaque handles, which are defined with an
`LITERT_DEFINE_HANDLE(ObjectHandle)` macro in [litert_common.h](litert_common.h).

- `LITERT_DEFINE_HANDLE(LiteRtTensorBuffer)`
  - `LiteRtTensorBuffer` defined as `typedef struct LiteRtTensorBufferT*`
- `LiteRtCreateTensorBuffer(LiteRtTensorBuffer* tensor_buffer)`
- `LiteRtDestroyTensorBuffer(LiteRtTensorBuffer tensor_buffer)`
  - Note the absence of `*` when passing a tensor_buffer

### Concrete Types (When Unavoidable)

By default, all C API types should be opaque. However, for lightweight value
types (e.g. `LiteRtLayout`, `LiteRtAny`, `LiteRtEnvOption`) or C callback tables
(e.g. `LiteRtOpResolverCApi`), concrete structs may be used.

When creating a concrete struct:

1. Document with `/// @note This concrete type is part of the public API and is ABI stable.`
2. Add compile-time assertions for `sizeof` and `offsetof` under `#if defined(__cplusplus) && defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8`.
3. Use pure C types only (no C++ standard library types).
4. Treat the struct layout as frozen once published.

