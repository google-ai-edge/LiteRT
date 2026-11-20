# LiteRT Compiler

Work in progress. Standalone StableHLO -> LiteRT (`.tflite`) compiler, with no
TensorFlow dependency.

## Layout

Planned layout; directories are added as the fork lands.

| Directory     | Contents                                                     |
| ------------- | ------------------------------------------------------------ |
| `ir/`         | TFL dialect, types, traits, and quantization IR.             |
| `frontend/`   | StableHLO import and preparation passes.                     |
| `transforms/` | TFL-level passes and quantization helpers.                   |
| `legalize/`   | StableHLO/CHLO -> TFL legalization.                          |
| `export/`     | TFL -> `.tflite` flatbuffer export.                          |
| `pipeline/`   | Conversion pipeline, options, and debug instrumentation.     |
| `utils/`      | Shared utilities.                                            |
| `python/`     | Python bindings.                                             |
| `tools/`      | `litert-compiler-opt` and other developer tools.             |
| `backends/`   | Reserved for partitioning and vendor AOT integration.        |

## Conventions

*   C++ code lives in the `litert::compiler` namespace (the TFL dialect in
    `litert::compiler::tfl`).
*   Registered pass flags use the `litert-` prefix.
*   The dialect namespace and op names stay `tfl` / `tfl.*`, so the produced
    `.tflite` files are unchanged.
