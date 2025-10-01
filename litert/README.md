# LiteRT Overview

LiteRT (short for Lite Runtime), formerly known as TensorFlow Lite, is Google's
high-performance runtime for on-device AI. You can find ready-to-run LiteRT
models for a wide range of ML/AI tasks, or convert and run TensorFlow, PyTorch,
and JAX models to the TFLite format using the AI Edge conversion and
optimization tools.

-   See the documentation: https://ai.google.dev/edge/litert

## Directory Map

- build_common/

  Bazel-specific, common internal build utilities (lrt_buiddefs.bzl)

- c/

  .h and associated .cc files for stable C APIs (ABI stable) to implement C++
  APIs

- cc/

  .h and associated .cc files for public/stable C++ APIs for app developers
  (not ABI stable)

- compiler/

  Private code and APIs specific to the compiler, defined inside lrt::internal
  namespace. Eventual destination for the internal code of the TFL
  converter/compiler

- core/

  Private code and APIs shared across compiler and runtime (e.g., schema.fbs),
  defined inside lrt::internal namespace

- integration_test/

  Integration tests

- js/

  JavaScript language bindings

- kotlin/

  Kotlin language bindings

- python/

  Python language bindings

- runtime/

  Private code and APIs specific to the runtime, defined inside lrt:::internal
  namespace.

- samples/

  Sample codes

- tools/

  Various tools (Benchmark, Accuracy evaluation, Apply Compiler Plugin CLI …)

- vendors/

  Code specific to SoC-vendors
