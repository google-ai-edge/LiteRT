# Repository Guidelines

This repo is LiteRT's source code. LiteRT (short for Lite Runtime), formerly known as TensorFlow Lite, is Google's
high-performance runtime for on-device AI. You can find ready-to-run LiteRT
models for a wide range of ML/AI tasks, or convert and run TensorFlow, PyTorch,
and JAX models to the TFLite format using the AI Edge conversion and
optimization tools. 

## Project Structure & Module Organization
- `litert/`: Core LiteRT runtime, compiler, and C/C++ APIs.
  - litert/build_common:  Bazel-specific, common internal build utilities (lrt_buiddefs.bzl)
  - litert/c: .h and associated .cc files for stable C APIs (ABI stable) to implement C++
  APIs
  - litert/cc: .h and associated .cc files for public/stable C++ APIs for app developers
  (not ABI stable). The C++ APIs are wrappers around the C APIs mentioned above.
  - litert/core: Private code and APIs shared across compiler and runtime (e.g., schema.fbs),
  defined inside lrt::internal namespace
  - litert/integration_test: Integration tests
  - litert/js: JavaScript language bindings
  - litert/kotlin: Kotlin language bindings, which depends on the public C++ API mentioned above.
  - litert/python: Python language bindings
  - litert/runtime: Private code and APIs specific to the runtime, defined inside lrt:::internal
  namespace.
  - litert/samples: Sample codes
  - litert/tools: Various tools (Benchmark, Accuracy evaluation, Apply Compiler Plugin CLI …)
  - litert/vendors: Code specific to SoC-vendors
- `tflite/`: Legacy TensorFlow Lite sources, delegates, tools, and examples.
- `g3doc/`: Product docs and build instructions.
- `docker_build/`: Docker-based build scripts and images.
- `third_party/`: Vendored dependencies and build rules.
- Tests live next to code as `*_test.cc` (C++/gtest) and `*_test.py` (Python),
  with fixtures in `litert/test/testdata/` and `tflite/testdata/`.

## Build, Test, and Development Commands

The project supports two different build systems: bazel and cmake. The CMake one is relatively new and immature. 
Please prefer to use bazel unless you are asked to improve the cmake scripts. Whenever using bazel, please start 
bazel by using bazelisk which is a launcher for launching a particular version of bazel. Before running bazel or 
bazelisk, change the current dir to the top level source dir and use the following command to create necessary folders
```bash
mkdir -p .bazelisk-cache .cache .bazel-output
```

Then you may run bazel in a way like:
```bash
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" bazelisk --output_base="$PWD/.bazel-output" build ...
```

So all the intermediate build files will be saved here so that AI agents can easily access(without breaking sandbox rules).

To build the core library for the host environment, you may build the "//litert/cc:litert_api_with_dynamic_runtime" bazel target. 
For example:

```bash
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" bazelisk --output_base="$PWD/.bazel-output" build -c dbg //litert/cc:litert_api_with_dynamic_runtime
```

To build it on macOS with XCode, we also need to add `--xcode_version=26.2 --repo_env=XCODE_VERSION=26.2 --action_env=DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` to every build commmand.


`//litert/tools:benchmark_model` is an important tool that can be used for running benchmarks.

- `./docker_build/build_with_docker.sh`: Build via the hermetic Docker image.
- `bazel build //litert/cc:litert_api_with_dynamic_runtime`: Build the core LiteRT API.
- `bazel build //litert/tools:all`: Build CLI tools (e.g., benchmarks).
- `bazel test //litert/...`: Run LiteRT unit tests (gtest).
- `cmake --preset default` then `cmake --build cmake_build -j`: Host build via
  CMake (see `g3doc/instructions/CMAKE_BUILD_INSTRUCTIONS.md`).

## Coding Style & Naming Conventions
- C++ uses Google Coding Style and `gnu++17` (see `.bazelrc`).
- Follow existing file-local style; use `clang-format` only where configs exist
  (e.g., `tflite/converter/quantization/ir/.clang-format`).
- File names are `snake_case.cc/.h`; tests use `*_test.cc` or `*_test.py`.
- New files must include the Apache 2.0 license header (see
  `CONTRIBUTING.md`).

## Testing Guidelines
- C++ tests use GoogleTest (`<gtest/gtest.h>`); keep tests near the code they
  cover.
- Prefer targeted test targets over full tree runs when iterating.
- Add or update test data under `litert/test/testdata/` or `tflite/testdata/`
  when behavior changes depend on fixtures.

## Commit & Pull Request Guidelines
- Commit messages are short, sentence-case summaries (see recent `git log`).
- A signed CLA is required before merging (see `CONTRIBUTING.md`).
- PRs are reviewed, CI must pass, and commits are typically squashed on merge.
- Include a clear description, testing notes, and link relevant issues when
  applicable.

## Security & Support
- Report vulnerabilities via `SECURITY.md`; avoid filing sensitive issues in
  public trackers.
