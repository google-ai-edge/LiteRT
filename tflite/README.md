> [!IMPORTANT]
> Maintenance Mode Notice: TensorFlow Lite packages and `tensorflow/lite/` are
> in maintenance mode and only receive critical security and stability updates.
> All active on-device ML development, optimizations, and new features have
> transitioned to **LiteRT**.
>
>
> For all on-device ML work, use [**LiteRT**](https://github.com/google-ai-edge/litert).
>
>LiteRT includes:
> * The modern `CompiledModel` API and legacy `Interpreter` API
> * Unified Hardware Acceleration (NPU/GPU)
> * Powers LiteRT-LM - Google's open-source inference framework designed to run
> Large Language Models (LLMs) and multi-modal models locally on edge devices.

> **Repository:** [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT)
> **Documentation:** [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)
> **Building from Source:** If you need to build from source, follow the instructions at [https://developers.google.com/edge/litert/build/cmake_litert](https://developers.google.com/edge/litert/build/cmake_litert).

# TensorFlow Lite (runtime dependency of LiteRT)

This directory contains the subset of TensorFlow Lite that LiteRT depends on:

-   The interpreter runtime (`core/`, `kernels/`, `c/`, `delegates/`, ...) that
    backs the LiteRT compiled model runtime.
-   The converter (`converter/`) used by the LiteRT converter.
-   The Java bindings (`java/`) and delegate plugins used to build the LiteRT
    Maven artifacts, and the Python interpreter bits packaged into the
    `ai-edge-litert` wheel.
-   Supporting tooling (`tools/`, `testing/`, `schema/`, `profiling/`) needed
    to build and test the above.

It is kept here so that LiteRT can be built in the open-source environment.
Standalone TensorFlow Lite deliverables that are not needed by LiteRT (legacy
docs, examples, iOS/ObjC/Swift pods, TOCO, benchmark apps, etc.) have been
removed. New development should happen in LiteRT
(https://github.com/google-ai-edge/LiteRT); see the documentation at
https://ai.google.dev/edge/litert.
