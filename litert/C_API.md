# Adding a New LiteRT Runtime C API

This guide lists the files, design conventions, and steps required when
adding a new function or type to the LiteRT Runtime C API and exposing it
through the C++ API.

## Architecture Overview

LiteRT's public C++ API (`litert/cc/`) does not call C API functions directly.
Instead, it routes calls through `litert::internal::RuntimeProxy`
([`cc/internal/litert_runtime_proxy.h`](cc/internal/litert_runtime_proxy.h)),
which invokes function pointers stored in `LiteRtRuntimeCApiStruct`
([`c/internal/litert_runtime_c_api.h`](c/internal/litert_runtime_c_api.h)).

This indirection provides binary backward and forward compatibility when the
C++ client SDK is paired with different versions of the LiteRT runtime
shared library (`libLiteRt.so`) or an externally provided system runtime (such
as LiteRT in Google Play services).

## Type Design: Opaque Types vs. Concrete Types

### Opaque Types (Default and Recommended)

By default, the LiteRT C API manages objects through **opaque handles**
(`LITERT_DEFINE_HANDLE` in [`c/litert_common.h`](c/litert_common.h)):

- **Information Hiding**: Callers cannot inspect or modify internal struct
  members directly.
- **Seamless ABI Stability**: The underlying internal struct
  (`LiteRtTensorBufferT`, `LiteRtEnvironmentT`, etc.) can add, reorder, or
  remove fields without altering the client's binary compatibility or memory
  size.
- **Explicit Lifecycle**: Memory allocation and destruction are strictly paired
  and encapsulated through `LiteRtCreate...` and `LiteRtDestroy...`.

### Concrete Types (Guidelines for Inevitable Exceptions)

While opaque handles are strongly preferred, in certain scenarios exposing
a **concrete struct** in public C headers is necessary or beneficial:

1. **Lightweight Value Types / Plain Old Data (POD)**: Types passed directly by
   value or stored in contiguous arrays where pointer indirection or individual
   heap allocations impose unacceptable overhead (e.g., `LiteRtLayout`,
   `LiteRtAny`, `LiteRtEnvOption`, `LiteRtRankedTensorType`).
2. **C Callback / Function Pointer Tables**: Function pointer tables passed
   across runtime boundaries to avoid multi-hop runtime object allocations
   (e.g., `LiteRtCustomOpKernel`).

When a concrete struct is required in the public C API, the following rules
**must** be strictly followed:

#### Rule 1: Document Public ABI Stability

Explicitly document the struct with the standard `@note` annotation:

```c
/// @note This concrete type is part of the public API and is ABI stable.
```

#### Rule 2: Compile-Time Size and Offset Invariants

To prevent accidental layout shifts, struct padding mismatches, or
compiler-dependent alignment discrepancies across toolchains, every concrete
struct must enforce compile-time size and member offset checks for standard
64-bit platforms:

```c
#if defined(__cplusplus) && defined(__SIZEOF_POINTER__) && \
    __SIZEOF_POINTER__ == 8
static_assert(sizeof(MyConcreteType) == <EXPECTED_SIZE>,
              "MyConcreteType size mismatch");
static_assert(offsetof(MyConcreteType, field_name) == <EXPECTED_OFFSET>,
              "MyConcreteType field_name offset mismatch");
#endif  // __cplusplus
```

#### Rule 3: Pure C Compatibility (No C++ Types)

- Concrete structs must only use standard C types (`stdint.h`, `stddef.h`,
  `stdbool.h`).
- Never include C++ types (e.g., `std::vector`, `std::string`, `std::optional`,
  `std::unique_ptr`, or non-trivial classes).
- Be careful with cross-platform bitfield packing and union layouts (MSVC and
  Clang handle adjacent bitfield coalescing differently).

#### Rule 4: Layout Immutability

Once published in a public release, the binary layout of a concrete struct is
frozen. Fields cannot be reordered, resized, or removed without breaking ABI
stability. If future expansion is required, use established patterns:

- **Append-only function tables**: Like `LiteRtRuntimeCApiStruct`, append new
  entries only at the end and bump the ABI version.
- **Leading size or version field**: Include a `uint32_t struct_size` or
  `LiteRtApiVersion version` as the first field so the runtime can safely
  handle different struct versions passed across the boundary.
- **Prefer opaque types**: If a struct is expected to evolve frequently with
  new configuration fields, define it as an opaque type with accessor functions
  instead of a concrete struct.

## Step-by-Step Checklist

### 1. Declare and Implement the C API Function

*   **Header (`c/litert_<module>.h`)**: Declare the function following the
    naming conventions in [`c/README.md`](c/README.md) (e.g.,
    `LiteRtCreate<Object>`, `LiteRtDestroy<Object>`,
    `LiteRtGet<Object><Field>`). Functions should return `LiteRtStatus` (or
    `void` for `Destroy` functions).
*   **Implementation (`c/litert_<module>.cc`)**: Implement the function and
    validate pointer arguments, returning `kLiteRtStatusErrorInvalidArgument` on
    null pointers.

### 2. Append to `LiteRtRuntimeCApiStruct` and Bump ABI Version

In [`c/internal/litert_runtime_c_api.h`](c/internal/litert_runtime_c_api.h):

1.  **Increment `LITERT_RUNTIME_ABI_VERSION`**:
    *   `LITERT_RUNTIME_ABI_VERSION` follows Semantic Versioning (`"X.Y.Z"`, see
        https://semver.org).
    *   Increment the **minor** version number (for example, from `"1.2.0"` to
        `"1.3.0"`) on **every change** that appends new C API functions to
        `LiteRtRuntimeCApiStruct`.
2.  **Append Function Pointer to `LiteRtRuntimeCApiStruct`**:

    *   Add the new function pointer at the **very end** of
        `LiteRtRuntimeCApiStruct`.
    *   **Never** delete, reorder, or change the signature of existing entries
        in `LiteRtRuntimeCApiStruct`, as doing so shifts struct field offsets
        and breaks binary compatibility with existing binaries.
    *   Include a comment above the new field indicating its source header and
        the ABI version in which it was added:

    ```c
    // litert_<module>.h: LiteRt<FunctionName>
    // This API was added in version 1.2.0.
    LiteRtStatus (*litert_<function_name>)(...);
    ```

### 3. Populate the Built-in Runtime Method Table

In
[`c/internal/litert_runtime_api_export.cc`](c/internal/litert_runtime_api_export.cc),
register the new function in `kLiteRtRuntimeBuiltin`:

```cpp
.litert_<function_name> = LiteRt<FunctionName>,
```

### 4. Update Exported Symbol Lists

Update the shared library symbol export lists so the new C API symbol is
exported across platforms:

*   **Linux / Android (`libLiteRt.so`)**: Add `LiteRt<FunctionName>` to
    [`c/litert_runtime_c_api_so_symbols.txt`](c/litert_runtime_c_api_so_symbols.txt).
    This list is verified by `litert_runtime_c_api_so_symbol_test`.
*   **Windows (`.dll`)**: Add `LiteRt<FunctionName>` to
    [`c/windows_exported_symbols.def`](c/windows_exported_symbols.def).

### 5. Add Version-Guarded Method to `RuntimeProxy`

In [`cc/internal/litert_runtime_proxy.h`](cc/internal/litert_runtime_proxy.h),
add a wrapper method to `litert::internal::RuntimeProxy` using
`LITERT_PROXY_METHOD_STATUS_MIN_VERSION(major, minor, patch, method, ...)` with
the ABI version in which the C API was introduced:

```cpp
LiteRtStatus <MethodName>(...) {
  LITERT_PROXY_METHOD_STATUS_MIN_VERSION(
      1, 2, 0, litert_<function_name>, ...);
}
```

When a client calls this method against an older runtime whose ABI version is
less than the specified minimum version, `RuntimeProxy` returns
`kLiteRtStatusErrorUnsupported` without reading past the end of the older
runtime's `LiteRtRuntimeCApiStruct`.

### 6. Expose in the C++ API and Add Tests

*   **C++ Wrapper (`cc/litert_<module>.h`)**: Call the new `RuntimeProxy` method
    from the corresponding C++ wrapper class (such as `Environment` or
    `CompiledModel`).
*   **Unit Tests**:
    *   Add C API unit tests in `c/litert_<module>_test.cc`.
    *   Add C++ API unit tests in `cc/litert_<module>_test.cc`, including
        testing version gating via the
        `RuntimeProxy(runtime_c_api, abi_version)` constructor overload to
        verify that older runtime ABI versions return
        `kLiteRtStatusErrorUnsupported`.
    *   Verify the symbol export test passes:
        `bazel test //litert/c:litert_runtime_c_api_so_symbol_test`.
