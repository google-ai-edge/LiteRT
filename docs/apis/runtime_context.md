# LiteRT Runtime Decoupling with RuntimeContext

## Background

Hardware accelerator components previously relied heavily on the core LiteRT
runtime library (`libLiteRt.so`) to access executing state.

This architectural setup mandated that delegates pull in large runtime
dependencies and dynamic symbols to operate correctly, creating an unwanted
reverse dependency where discrete backend components (such as GPU Accelerators,
Dispatch APIs) relied on the core executing framework.

While this documentation focuses on the GPU Accelerator as the initial
implementation, **Dispatch APIs** and **Compiler Plugins** suffer from the same
reverse dependency issues. This document serves as a guide for decoupling them
as well.

Note: **Compiler Plugins** use a dedicated `LiteRtCompilerContext` for
compile-time operations, while **Accelerators** and **Dispatch APIs** use
`LiteRtRuntimeContext`.

## Goal

Decouple Accelerator components from the LiteRT Runtime C API (`libLiteRt.so`)
and standard LiteRt Runtime APIs (`litert/runtime/*`) by utilizing
`RuntimeContext` and opaque pointer abstractions to pass execution state. This
effectively eliminates the dependency on `libLiteRt.so` for extension plugins.

## Refactored Custom Tensor Buffer Handlers

Custom tensor buffer handlers have been refactored so they no longer depend on
LiteRT C APIs.

As an implementation detail, memory buffer handlers across all target backends
no longer require a `LiteRtEnvironment` parameter to derive underlying device
and command queue primitives:

-   **Refactoring Signatures:** We removed the `LiteRtEnvironment` arguments
    from all custom tensor buffer function signatures (e.g., Create, Lock,
    Unlock, Clear, Destroy) within specific backend wrapper implementations.
-   **Opaque Injection:** Function parameters representing the structural
    execution scope (e.g., `device_id` and `queue_id`) have been fully converted
    to opaque `void*` pointers. By relying on generic `void*` arguments,
    backend-specific handlers can receive their required memory allocations and
    queue handlers directly from the instantiating caller without needing to
    query standard runtime APIs.

## Introduction to RuntimeContext

`RuntimeContext` is fundamentally designed as a virtual function table (a C
struct housing function pointers) that encapsulates the subset of LiteRT C APIs
required by Accelerators.

### Structure Overview

The core interface, `LiteRtRuntimeContext` (defined dynamically via headers),
acts as a structural adapter containing pointers for operations including:

-   **Tensor Buffers:** Creating, locking, unlocking, and parsing memory
    primitive handles.
-   **External Buffers:** Registering and unregistering external buffer
    contexts.
-   **Environment Context:** Parsing arbitrary opaque options and verifying GPU
    runtime environments.
-   **Event Synchronization:** Creating, polling, and waiting on custom
    execution events (e.g., `LiteRtClEvent`).

Note: Both `LiteRtRuntimeContext` and `LiteRtCompilerContext` are ABI-stable,
versioned function tables. See
[ABI Stability and Extending Context Tables](#abi-stability-and-extending-context-tables)
for rules on adding and calling new functions safely.

### Decoupling Mechanism

Instead of an Accelerator component forcing symbol resolution against
`libLiteRt.so` at compile or load time, the host populates and supplies a
`LiteRtRuntimeContext` struct instance. The external components then seamlessly
invokes the runtime APIs securely through the struct's function pointers,
achieving total decoupling from the compiled dynamic library API.

## Architecture Diagram

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

```mermaid
graph TD
    classDef core fill:#d4e157,stroke:#333,stroke-width:2px;
    classDef header fill:#4fc3f7,stroke:#333,stroke-width:2px;
    classDef extension fill:#ffb74d,stroke:#333,stroke-width:2px;
    classDef removed fill:#ef9a9a,stroke:#f44336,stroke-width:2px,stroke-dasharray: 5 5;
    subgraph Extension Ecosystem
        Accelerator["GPU Accelerator"]:::extension
        Dispatch["Dispatch API"]:::extension
        Compiler["Compiler Plugin"]:::extension
    end
    subgraph Abstraction Layer
        ContextHeader["litert_runtime_context_header<br/>(Header-only)"]:::header
    end
    subgraph Core Runtime Implementation
        ContextImpl["litert_runtime_context<br/>(Implementation)"]:::core
        Environment["Environment"]:::core
        CompiledModel["CompiledModel"]:::core
        TensorBuffer["TensorBuffer"]:::core
        LibLiteRt["libLiteRt.so"]:::core
    end
    Accelerator -->|Includes| ContextHeader
    Dispatch -->|Includes| ContextHeader
    Compiler -->|Includes| ContextHeader
    ContextImpl -.->|Implements| ContextHeader
    ContextImpl -->|Uses| LibLiteRt
    Environment -->|Bundles| ContextImpl
    CompiledModel -->|Bundles| ContextImpl

    TensorBuffer -.->|Must be linked with to satisfy RuntimeContext| Environment
    Accelerator -.-x|REMOVED DEPENDENCY| LibLiteRt
```

## Transitioning to RuntimeContext

To guarantee that Accelerators can operate without compiling against the Runtime
C API:

-   **Dependency Inversion via `RuntimeContext`:** Instead of an accelerator
    invoking standard runtime APIs to extract the active environment variables,
    the necessary low-level execution context primitives are provided explicitly
    via the abstraction interface.
-   **Build Target Decoupling:** `//litert/c:litert_runtime_c_api_shared_lib`
    has been completely purged from the BUILD dependencies of the accelerator
    modules.
-   **Target Cleanup:** We removed all `_runtimecapi` targets since all
    components no longer use LiteRT C APIs. This allows them to cleanly compile
    using semantic header boundaries.

## RuntimeContext Targets

`RuntimeContext` is broken down into two separate targets to cleanly enforce
decoupling and prevent cyclic dependencies:

1.  `litert_runtime_context_header`

    -   **Type:** Header-only target.
    -   **Usage:** This target is used by the **Accelerator** and **Dispatch
        API**. **CompilerPlugin** uses `LiteRtCompilerContext` (defined in
        `litert_compiler_context_header`).

2.  `litert_runtime_context`

    -   **Type:** Implementation target.
    -   **Usage:** This target contains the actual LiteRT C API implementation
        and is bundled into the core CompiledModel and Environment components.

### Implementation Caveat: TensorBuffer Circular Dependencies

To break a known circular dependency, the core TensorBuffer component does
**not** link against the `litert_runtime_context` implementation.

**Client Usage Impact:** If a client wishes to use the TensorBuffer C API
(`//litert/c:litert_tensor_buffer`), they **must** explicitly link the
Environment dependency (`//litert/c:litert_environment`) to satisfy the
underlying runtime context requirements.

## ABI Stability and Extending Context Tables

Both `LiteRtRuntimeContext` (for accelerators, delegates, and Dispatch API
implementations) and `LiteRtCompilerContext` (for Compiler Plugins and C++
compiler wrappers in `litert/compiler/cc/`) cross shared library boundaries
with the core LiteRT library (`libLiteRt.so`). Because a plugin or accelerator
built against newer headers may run against an older LiteRT binary (or vice
versa), both structs embed `LiteRtAbiHeader`
(`litert/c/internal/litert_abi_header.h`) at byte offset `0` to guarantee
binary compatibility.

### Adding a New Function to `LiteRtRuntimeContext` or `LiteRtCompilerContext`

1.  **Append Only:** Add the new function pointer at the **very end** of
    `struct LiteRtRuntimeContext` (`litert/c/internal/litert_runtime_context.h`)
    or `struct LiteRtCompilerContext`
    (`litert/c/internal/litert_compiler_context.h`) with a version comment
    (e.g., `// Added in version 1.3.0`). Never reorder, remove, or modify
    existing function pointer fields.
2.  **Update Layout Assertions:** Update `static_assert(sizeof(...) == ...)` and
    add a `static_assert(offsetof(..., <new_fn>) == ...)` entry in the
    corresponding context header.
3.  **Populate and Bump Minor Version:** Assign the implementation in
    `LrtGetRuntimeContext()` (`litert/c/internal/litert_runtime_context.cc`) or
    `LrtGetCompilerContext()` (`litert/c/internal/litert_compiler_context.cc`)
    and increment `abi_header.minor_version` (while keeping `major_version`
    unchanged for append-only additions).

### Calling Newly Added Functions (`LITERT_ABI_HAS_API` Required)

Any code calling a function added to `LiteRtRuntimeContext` or
`LiteRtCompilerContext` after version `1.0.0` **must** guard the call with
`LITERT_ABI_HAS_API` (from `//litert/c/internal:litert_abi_header`) instead of
checking `ctx->new_fn != nullptr` directly.

*   **Why direct null checks are unsafe:** When an older runtime or compiler
    host (whose `abi_header.struct_size` is smaller) passes a
    `LiteRtRuntimeContext*` or `LiteRtCompilerContext*` to a delegate, dispatch
    library, or compiler plugin compiled against newer headers, dereferencing
    `ctx->new_fn` reads past the end of the allocated struct (out-of-bounds
    read).
*   **What `LITERT_ABI_HAS_API` checks:**
    `LITERT_ABI_HAS_API(ctx, /*req_major=*/1, new_fn)` verifies in order that
    `ctx` is non-null, `abi_header.major_version == 1`,
    `abi_header.struct_size` is large enough to contain `new_fn`, and
    `ctx->new_fn != NULL`.

```cpp
#include "litert/c/internal/litert_abi_header.h"

// Example with LiteRtRuntimeContext:
if (LITERT_ABI_HAS_API(runtime_context, /*req_major=*/1,
                       create_tensor_buffer_requirements_with_alignment)) {
  return runtime_context->create_tensor_buffer_requirements_with_alignment(
      num_types, types, buffer_size, num_strides, strides, alignment,
      out_requirements);
}
return runtime_context->create_tensor_buffer_requirements(
    num_types, types, buffer_size, num_strides, strides, out_requirements);

// Example with LiteRtCompilerContext (e.g., in a plugin or litert/compiler/cc):
if (!LITERT_ABI_HAS_API(compiler_context, /*req_major=*/1,
                        get_conv_3d_transpose_padding_option)) {
  return kLiteRtStatusErrorUnsupported;
}
return compiler_context->get_conv_3d_transpose_padding_option(op, &padding);
```

When adding unit tests for a newly added `LiteRtRuntimeContext` or
`LiteRtCompilerContext` function, test both:

1.  The function pointer being `nullptr` on a full-sized context.
2.  A truncated `abi_header.struct_size = offsetof(<ContextStruct>, <new_fn>)`
    where the function pointer slot in memory remains non-null, confirming
    `LITERT_ABI_HAS_API` prevents reading or calling the out-of-bounds slot.
