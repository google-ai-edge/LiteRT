# Adding a New LiteRT Runtime C API

This guide lists the files and steps required when adding a new function to the
LiteRT Runtime C API and exposing it through the C++ API.

## Architecture Overview

LiteRT's public C++ API (`litert/cc/`) does not call C API functions directly.
Instead, it routes calls through `litert::internal::RuntimeProxy`
([`cc/internal/litert_runtime_proxy.h`](cc/internal/litert_runtime_proxy.h)),
which invokes function pointers stored in `LiteRtRuntimeCApiStruct`
([`c/internal/litert_runtime_c_api.h`](c/internal/litert_runtime_c_api.h)).

This indirection provides binary backward and forward compatibility when the C++
client SDK is paired with different versions of the LiteRT runtime shared
library (`libLiteRt.so`) or an externally provided system runtime (such as
LiteRT in Google Play services).

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

### 2. Append to `LiteRtRuntimeCApiStruct` and Increment `LITERT_RUNTIME_ABI_VERSION`

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
