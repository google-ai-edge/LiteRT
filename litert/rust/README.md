# Rust Binding

## Build procedure

Cargo uses the prebuilt [LiteRT C++ Binary SDK](https://ai.google.dev/edge/litert/next/cpp_sdk). The SDK version is defined in `build/build.rs`.

## Dependencies

The build script depends on a few tools. It checks that the tools are installed
by trying to run them and fails if they are not found:

* **Clang**: Needed for bindgen.
* **CMake**: Used to build C++ LiteRT SDK.

## Features

* `async_support` - Enables the async API for the LiteRT runtime. Third-party async runtimes ([tokio](https://tokio.rs/), etc.) are supported.
* `metal` - Enables Metal acceleration for the LiteRT runtime on macOS.
* `opengl` - Enables OpenGL acceleration for the LiteRT runtime.
* `webgpu` - Enables WebGPU acceleration for the LiteRT runtime.
For more information about acceleratoirs see [documentation](https://developers.google.com/edge/litert/next/cpp_sdk#prebuilt-gpu-accelerators).
