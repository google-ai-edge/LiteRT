# Source-Built LiteRT C++ API CMake Example

This example shows the source-vendored CMake flow for an app developer who
wants one dependency for the LiteRT C++ headers and the source-built LiteRT
runtime shared library:

```cmake
target_link_libraries(my_app PRIVATE litert_cc_api_with_dynamic_runtime)
```

## Files

`CMakeLists.txt` is the downstream consumer project. It vendors the LiteRT
source tree with `add_subdirectory`:

```cmake
set(LITERT_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../.."
    CACHE PATH "Path to the LiteRT source tree")
set(LITERT_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/litert_build")

add_subdirectory("${LITERT_SOURCE_DIR}" "${LITERT_BUILD_DIR}" EXCLUDE_FROM_ALL)
```

It then links only the convenience target:

```cmake
add_executable(litert_source_build_cc_api_run_model
    source_build_cc_api_run_model.cc
)

target_link_libraries(litert_source_build_cc_api_run_model
    PRIVATE
        litert_cc_api_with_dynamic_runtime
)
```

`source_build_cc_api_run_model.cc` is the sample app. The CPU default is:

```text
litert/test/testdata/simple_add_dynamic_shape.tflite
```

That model has two dynamic float inputs, so the sample exercises
`CompiledModel::ResizeInputTensor`, input/output metadata APIs, buffer
requirement APIs, `TensorBuffer::Write`, `TensorBuffer::Read`, `Run`,
`RunAsync`, `TensorBuffer::Duplicate`, and the named-map `Run` overloads.

The GPU default is:

```text
litert/test/testdata/mobilenet_v2_1.0_224.tflite
```

## Build The CPU Example

From the repository root:

```bash
cmake -S litert/vendors/examples/cmake_example/source_build_cc_api \
  -B /tmp/litert_source_build_cc_api_example \
  -DLITERT_ENABLE_GPU=OFF \
  -DLITERT_ENABLE_NPU=OFF

cmake --build /tmp/litert_source_build_cc_api_example \
  --target litert_source_build_cc_api_run_model -j
```

Run the default checked-in CPU model:

```bash
/tmp/litert_source_build_cc_api_example/litert_source_build_cc_api_run_model \
  --iterations=2 \
  --print_tensors \
  --sample_size=6
```

Use a different model:

```bash
/tmp/litert_source_build_cc_api_example/litert_source_build_cc_api_run_model \
  --model=/absolute/path/to/model.tflite \
  --resize_inputs=none
```

When no `--model` is passed, the sample uses `--resize_inputs=1,128,4` for the
CPU default `simple_add_dynamic_shape.tflite`. For explicit models, the default
is no resize; pass `--resize_inputs=d0,d1,...` only for dynamic-shape models.

## Build The GPU Example

This example mirrors that SDK flow for the accelerator artifact, but keeps the
CMake intentionally small. When `LITERT_ENABLE_GPU=ON`, CMake picks the
documented prebuilt accelerator for the current platform and always downloads
the `latest` artifact during configure:

```text
https://storage.googleapis.com/litert/binaries/latest/<platform>/<accelerator-library>
```

The post-build step copies the accelerator next to the source-built `libLiteRt`
runtime after building the example executable.

Use a GPU-enabled build directory:

```bash
cmake -S litert/vendors/examples/cmake_example/source_build_cc_api \
  -B /tmp/litert_source_build_cc_api_gpu_example \
  -DLITERT_ENABLE_GPU=ON \
  -DLITERT_ENABLE_NPU=OFF

cmake --build /tmp/litert_source_build_cc_api_gpu_example \
  --target litert_source_build_cc_api_run_model -j
```

On macOS arm64, the post-build step stages:

```text
/tmp/litert_source_build_cc_api_gpu_example/litert_build/c/libLiteRtMetalAccelerator.dylib
```

The example sets `--runtime_library_dir` to the build output directory by
default. In the GPU build shown above that is:

```text
/tmp/litert_source_build_cc_api_gpu_example/litert_build/c
```

Run with Metal:

```bash
/tmp/litert_source_build_cc_api_gpu_example/litert_source_build_cc_api_run_model \
  --accelerator=gpu,cpu \
  --gpu_backend=metal \
  --gpu_precision=fp16
```

Because no `--model` is passed, that command uses the static GPU default
`mobilenet_v2_1.0_224.tflite`.

## Convenience Target

It's defined in `litert/cc/CMakeLists.txt`:

```cmake
add_library(litert_cc_api_headers INTERFACE)

target_compile_features(litert_cc_api_headers INTERFACE cxx_std_20)

target_link_libraries(litert_cc_api_headers
    INTERFACE
        absl::algorithm_container
        absl::any_invocable
        absl::cleanup
        absl::flat_hash_map
        absl::status
        absl::statusor
        absl::span
        absl::str_format
        absl::strings
        absl::log
        absl::die_if_null
)

add_library(litert_cc_api_with_dynamic_runtime INTERFACE)

target_link_libraries(litert_cc_api_with_dynamic_runtime
    INTERFACE
        litert_cc_api_headers
        litert_runtime_c_api_shared_lib
)
```

`litert_cc_api_headers` is header-only usage requirements: include paths,
C++20, and public header dependencies.

`litert_cc_api_with_dynamic_runtime` adds the runtime shared library target.
When an executable links this interface target, CMake has a real target
dependency on `litert_runtime_c_api_shared_lib`, so building the executable also
builds and links the LiteRT runtime shared library.

The runtime shared library target is defined in `litert/c/CMakeLists.txt`:

```cmake
add_library(litert_runtime_c_api_shared_lib SHARED empty.cc)
set_target_properties(litert_runtime_c_api_shared_lib PROPERTIES
    OUTPUT_NAME "LiteRt"
    POSITION_INDEPENDENT_CODE ON
)
```

That produces `libLiteRt.so` on Linux/Android and `libLiteRt.dylib` on macOS.



## Summary

For a real app, LiteRT can live anywhere. The important part is that the app
adds the LiteRT `litert/` directory to the same CMake build graph before
linking the convenience target:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_litert_app LANGUAGES CXX C)

set(LITERT_SOURCE_DIR "/path/to/LiteRT/litert" CACHE PATH "")
set(TFLITE_SOURCE_DIR "/path/to/LiteRT/tflite" CACHE PATH "")

add_subdirectory("${LITERT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/litert")

add_executable(my_litert_app main.cc)
target_link_libraries(my_litert_app PRIVATE litert_cc_api_with_dynamic_runtime)
```

