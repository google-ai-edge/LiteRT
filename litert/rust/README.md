# Rust Binding

## Build procedure

If LiteRT sources are already installed, set the environment variable
`RUST_LITERT_SOURCE_DIR`. If the variable is not set, a copy of sources will be
downloaded from Github.

If LiteRT binary runtime is available, set the environment variable
`RUST_LITERT_RUNTIME_LIBRARY_DIR`. If the variable is not set, the build script
will download pre-build binaries or, if the feature `build_runtime_with_docker`
is enabled, use Docker to build it from sources.

## Dependencies

The build script depends on a few tools, it checks that the tools are installed
by trying to run them and fails if it can't:

* Clang - is needed for bindgen
* Docker - only used if the script builds Runtime binary library.