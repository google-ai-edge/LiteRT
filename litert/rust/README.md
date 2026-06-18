# Rust Binding

## Build procedure

The cargo uses uses the prebuilt [LiteRT C++ Binary SDK](https://ai.google.dev/edge/litert/next/cpp_sdk). Always the latest available
version is downloaded.

## Dependencies

The build script depends on a few tools, it checks that the tools are installed
by trying to run them and fails if it can't:

* Clang - is needed for bindgen
* Cmake - is used to build C++ LiteRT SDK.