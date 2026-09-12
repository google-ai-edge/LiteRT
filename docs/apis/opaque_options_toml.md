# LiteRT Opaque Options (TOML) Design & Refactoring Guide

## 1. Background & Motivation

In LiteRT, a significant portion of the internal C API surface was traditionally
dedicated to handling options (e.g., `LiteRtRuntimeOptions`,
`LiteRtCompilerOptions`, `LiteRtGpuOptions`). Historically, each options field
required its own C API getter and setter, leading to a sprawling API surface and
tight coupling across the runtime.

To solve this and ensure ABI stability without maintaining an exponentially
expanding API surface, LiteRT has transitioned to using **Opaque Options**
serialized via **TOML**.

### Why TOML?

-   **Minimal and schema-less**: Unlike Protocol Buffers or FlatBuffers, TOML
    doesn't require compiling a standalone schema.
-   **Lightweight Parser**: Instead of compiling a heavy third-party dependency
    into the core library, LiteRT utilizes its own minimal customized TOML
    parser (`litert/core/litert_toml_parser.cc`).
-   **Data Types**: TOML supports all the necessary LiteRT field types natively:
    booleans, integers, strings, arrays, and even pointers (serialized
    dynamically as integers).

## 2. Architecture: Producer-Consumer Separation

The foundational concept behind Opaque Options is separating options
construction (Producer) from the options parsing (Consumer):

-   **Producer (Client/App)**: Uses the C API/C++ wrapper to construct,
    validate, and serialize options into a TOML string payload. The C API may
    not contain parsing logic.
-   **Consumer (Runtime/Plugin/Compiler)**: Receives the serialized opaque
    options payload globally, parses the TOML context string into an internal
    C++ struct, and consumes the values inside the driver/runtime space. Note:
    This internal struct could be the exact same object type used on the
    Producer side (e.g., `LrtGpuOptions`), or it could be a completely separate
    runtime structure (e.g., `LiteRtRuntimeOptionsT`). It is a consumer-side
    decision, and LiteRT utilizes both approaches.

## 3. Step-by-Step Refactoring Guide

This section is a guide for refactoring newly introduced or existing standard
options into the TOML-based opaque options structure.

### 3.1: Define the C API (`litert/c/options/`)

The C API should provide the opaque struct namespace and ABI-safe modifiers.

-   Lowercase old APIs using the `Lrt` prefix (e.g., `typedef struct
    LrtRuntimeOptions LrtRuntimeOptions;`). The `Lrt` prefix indicates that
    these functions are not part of the core LiteRT C APIs, but belong to a
    totally independent library. Exported tools should be removed from scripts
    like `windows_exported_symbols.def`.
-   Declare `LrtCreate...`, `LrtDestroy...`.
-   Declare `LrtCreateOpaque...` which natively serializes the active context to
    `LiteRtOpaqueOptions`.
-   Internally (in `.cc`), leverage `std::optional` to track dynamically set
    fields. Serialize by aggregating strings (e.g., via `std::stringstream`),
    returning the final payload securely using
    `litert::internal::MakeCStringPayload`.

### 3.2: Implement the C++ API Wrapper (`litert/cc/options/`)

Provide a safe, strictly RAII-style C++ wrapper surrounding the targeted C API.

-   Retain a `std::unique_ptr<LrtOptionsType, Deleter>` wrapper to manage C
    object lifetimes.
-   Intercept setters/getters directly through native C API endpoints.
-   Ensure the wrapper does **not** rely on `litert_opaque_options.h` or possess
    `CreateOpaqueOptions()`.
-   Client serialization flows rely on extracting the native context through
    `.Get()` and manually applying `LrtCreateOpaque...`.

### 3.3: Utilizing the TOML Parser (`litert/core/`)

-   Utilize the basic `litert::internal::ParseToml` parser to hook standard
    key-value conversions targeting callback triggers.
-   Convert context mapping safely: Since the internal parser organically strips
    quotes from string values, directly evaluate using `std::string(value)`.
    Context reliant on arrays (e.g., `["a", "b"]`) will not inherently strip
    tokens natively; utilize `ParseTomlStringArray`.
