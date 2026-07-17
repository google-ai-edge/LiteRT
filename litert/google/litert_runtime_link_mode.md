# Linking LiteRT Runtime Options

This document describes how to link the LiteRT runtime for both C++ and Kotlin
clients under different usage scenarios (such as standalone applications vs. GMS
Core integration).

--------------------------------------------------------------------------------

## Overview of Linking Modes

LiteRT supports multiple runtime linking configurations depending on client
requirements:

*   **Static Linking (`static`)**: The LiteRT runtime implementation is compiled
    directly into the client binary.
*   **Dynamic Linking (`dynamic`)**: The LiteRT runtime is loaded from a shared
    library (`libLiteRt.so`).
*   **Header-only / External (`none`)**: Only LiteRT API headers/interfaces are
    linked into the binary. The runtime implementation is provided externally at
    runtime (e.g., loaded dynamically by GMS Core).

In Bazel/Blaze builds, the linking mode for standalone targets is controlled by
the build flag:

```bash
--define=litert_runtime_link_mode=<mode>
```

Where `<mode>` can be `static`, `dynamic`, or `none`. If unspecified, the
default is `static`.

--------------------------------------------------------------------------------

## C++ Clients

### 1. Standalone Clients

Standalone C++ clients should depend on the `litert_environment` target:

```bzl
cc_binary(
    name = "my_standalone_app",
    srcs = ["main.cc"],
    deps = [
        "//litert/cc:litert_environment",
    ],
)
```

Use the `--define=litert_runtime_link_mode=<mode>` flag during build to select
the linking strategy:

*   **Static Link (Default):**

    ```bash
    blaze build //path/to:my_standalone_app
    # or explicitly:
    blaze build --define=litert_runtime_link_mode=static //path/to:my_standalone_app
    ```

*   **Dynamic Link:**

    ```bash
    blaze build --define=litert_runtime_link_mode=dynamic //path/to:my_standalone_app
    ```

### 2. GMS Core Clients

GMS Core C++ clients (or clients where the runtime is provided externally)
should depend on `litert_environment_api`:

```bzl
cc_library(
    name = "my_gmscore_native_lib",
    srcs = ["gmscore_client.cc"],
    deps = [
        "//litert/cc:litert_environment_api",
    ],
)
```

`litert_environment_api` provides the C++ API headers without bundling the
LiteRT runtime implementation, allowing GMS Core to dynamically bind or provide
the runtime implementation at execution time.

--------------------------------------------------------------------------------

## Kotlin / Android Clients

### 1. Standalone Clients

Standalone Android/Kotlin clients should depend on the `litert` target in
`//litert/kotlin`:

```bzl
android_binary(
    name = "my_standalone_android_app",
    manifest = "AndroidManifest.xml",
    deps = [
        "//litert/kotlin:litert",
    ],
)
```

The `:litert` target bundles both the LiteRT Kotlin/Java API and the native
runtime implementation (`libLiteRt.so`).

NOTE: `static` link is not support for Kotlin clients for now. If it's needed,
a different target could be provided.

### 2. GMS Core Clients

GMS Core (or applications using an externally provided LiteRT runtime) should
depend on `litert_api`:

```bzl
android_library(
    name = "my_gmscore_feature_lib",
    srcs = glob(["*.kt"]),
    deps = [
        "//litert/kotlin:litert_api",
    ],
)
```

The `:litert_api` target provides the LiteRT API interface and JNI bindings
without bundling the native LiteRT runtime shared library (`libLiteRt.so`).

--------------------------------------------------------------------------------

## Summary Table

Client Type    | Target Language | Bazel Dependency                                             | Link Control Flag
:------------- | :-------------- | :----------------------------------------------------------- | :----------------
**Standalone** | C++             | `//litert/cc:litert_environment`     | `--define=litert_runtime_link_mode=[static\|dynamic]`
**GMS Core**   | C++             | `//litert/cc:litert_environment_api` | N/A (Header / API only)
**Standalone** | Kotlin          | `//litert/kotlin:litert`             | N/A (dynamic link only)
**GMS Core**   | Kotlin          | `//litert/kotlin:litert_api`         | N/A (API only)
