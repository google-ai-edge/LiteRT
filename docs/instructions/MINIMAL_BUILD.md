# Building a Minimal LiteRT Runtime

## Overview

By default, a binary that links the LiteRT `CompiledModel` API pulls in every
backend and every kernel: GPU and NPU support, the XNNPACK CPU accelerator, all
TFLite built-in kernels, and all reference kernels. That is convenient, but a
small, fixed model on a size-sensitive target (embedded Linux, Web/WASM, apps
with a size cap) needs much less.

This guide covers:

*   The build flags that control what goes into the LiteRT runtime.
*   How to build the common CPU configurations:
    *   **CPU with XNNPACK** (the default CPU path).
    *   **CPU with built-in ops only** (no XNNPACK).
    *   **CPU with selective built-in ops** (no XNNPACK, only the ops used by
        the model).
*   How the runtime `kernel_mode` option relates to, and differs from, the
    build flags.
*   Typical binary sizes on a Linux ARM32 target, and known limitations.

## Build Flags

| Flag                                                                     | Effect                                       |
| ------------------------------------------------------------------------ | -------------------------------------------- |
| `--//litert/build_common:build_include=cpu_only` | Leaves out GPU and NPU support (compiler     |
:                                                                          : plugin, compilation cache, dispatch, GPU     :
:                                                                          : environment). The default is `gpu,npu`.      :
| `--//litert/build_common:cpu_backend=builtin`    | Leaves out the XNNPACK CPU accelerator and   |
:                                                                          : delegate. CPU execution uses the TFLite      :
:                                                                          : built-in kernels. The default is `xnnpack`.  :
| `--//litert/build_common:cpu_backend=selective`  | Leaves out the XNNPACK CPU accelerator,      |
:                                                                          : built-in kernels, and reference kernels      :
:                                                                          : (`LITERT_DISABLE_CPU` and                    :
:                                                                          : `LITERT_NO_BUILTIN_OPS`). The application    :
:                                                                          : provides a selective op resolver via         :
:                                                                          : `litert\:\:tflite_support\:\:SetOpResolver`. :
| `--define=litert_builtin_ops=false`                                      | Leaves out the TFLite built-in kernels and   |
:                                                                          : uses a stub op resolver instead              :
:                                                                          : (`LITERT_NO_BUILTIN_OPS`). Every op must     :
:                                                                          : then be handled by an accelerator (XNNPACK   :
:                                                                          : on CPU) or a user-provided op resolver.      :

These flags are handled by `select()`s in:

*   `//litert/runtime:compiled_model`
*   `//litert/runtime/accelerators:cpu_registry`
*   `//litert/build_common:build_config_header`

> **Note:** `cpu_backend=builtin` replaces the older
> `--define=tflite_with_xnnpack=false`, which is still accepted. Both select
> the `litert_disable_cpu` config setting, which defines `LITERT_DISABLE_CPU`.
> Despite that name, CPU execution is **not** disabled; only the XNNPACK
> accelerator is removed.

## CPU Build Configurations

### Option A: CPU with XNNPACK (default CPU path)

```sh
bazel build -c opt \
  --//litert/build_common:build_include=cpu_only \
  //your/package:your_target
```

*   **Linked:** XNNPACK accelerator and delegate, all built-in kernels (used as
    a fallback), and all reference kernels.
*   **Runtime behavior:** Nodes that XNNPACK supports are delegated to XNNPACK.
    Everything else runs on built-in kernels.
*   **When to use:** Larger or compute-heavy models where XNNPACK performance
    matters.

### Option B: CPU with built-in ops only (no XNNPACK)

```sh
bazel build -c opt \
  --//litert/build_common:build_include=cpu_only \
  --//litert/build_common:cpu_backend=builtin \
  //your/package:your_target
```

*   **Linked:** Built-in and reference kernels. No XNNPACK.
*   **Runtime behavior:** Every op runs on the TFLite built-in (optimized)
    kernels. No CPU accelerator is registered. Create the model with
    `HwAccelerators::kCpu` as usual.
*   **When to use:** Small models where XNNPACK's code size (~1.2 MB on ARM32)
    outweighs its speedup. This matches what a TFLite `Interpreter` without
    XNNPACK does.

### Option C: XNNPACK only (no built-in ops)

```sh
bazel build -c opt \
  --//litert/build_common:build_include=cpu_only \
  --define=litert_builtin_ops=false \
  //your/package:your_target
```

*   **Linked:** XNNPACK only. No built-in kernels.
*   **Runtime behavior:** Only works if **every** node in the model is
    delegated to XNNPACK. If any node is not supported by XNNPACK, the model
    fails to run.
*   **When to use:** Models that XNNPACK is known to fully support.

### Option D: CPU with selective built-in ops (no XNNPACK, model-specific ops)

Generate a `RegisterSelectedOps` function for your model(s) using TFLite's
`gen_selected_ops` rule in your `BUILD` file:

```python
load("//litert/build_common:special_rule.bzl", "litert_linkopts")
load("//third_party/tensorflow/lite:build_def.bzl", "gen_selected_ops")

gen_selected_ops(
    name = "my_model_selected_ops",
    model = ["my_model.tflite"],
)

cc_binary(
    name = "my_app",
    srcs = [
        "my_app.cc",
        ":my_model_selected_ops",
    ],
    linkopts = litert_linkopts(),
    linkstatic = 1,
    deps = [
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/tflite_support/op_resolver",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite:mutable_op_resolver",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
    ],
)
```

Populate a `tflite::MutableOpResolver` with `RegisterSelectedOps` and pass it to
`litert::tflite_support::SetOpResolver`:

```cpp
#include "litert/tflite_support/op_resolver/op_resolver.h"
#include "third_party/tensorflow/lite/mutable_op_resolver.h"

void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);

// ...
tflite::MutableOpResolver resolver;
RegisterSelectedOps(&resolver);
LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
LITERT_RETURN_IF_ERROR(
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu));
LITERT_RETURN_IF_ERROR(
    litert::tflite_support::SetOpResolver(options, &resolver));
LITERT_ASSIGN_OR_RETURN(
    auto compiled_model,
    litert::CompiledModel::Create(env, model_path, options));
```

Build with `cpu_backend=selective`:

```sh
bazel build -c opt \
  --linkopt=-Wl,--gc-sections \
  --//litert/build_common:build_include=cpu_only \
  --//litert/build_common:cpu_backend=selective \
  //your/package:my_app
```

*   **Linked:** Only the built-in kernels referenced by `RegisterSelectedOps`.
    No XNNPACK, no unused built-in kernels, and no reference kernels.
*   **Runtime behavior:** Every op in the model runs on its registered TFLite
    built-in (or custom) kernel.
*   **When to use:** Fixed-model deployments where binary size is critical.

## Per-Target Build Configuration (Without Command-Line Flags)

Instead of passing `--build_include` and `--cpu_backend` on the command line,
you can configure them per target in your `BUILD` file using the rules and
macros in `//litert/build_common:litert_build_defs.bzl`:

| Rule / Macro                 | Use Case                                     |
| ---------------------------- | -------------------------------------------- |
| `litert_cc_binary`           | Drop-in wrapper around `cc_binary` that      |
:                              : transitions the binary and its dependencies  :
:                              : to `build_include` and `cpu_backend`.        :
| `litert_cc_test`             | Drop-in wrapper around `cc_test` that        |
:                              : transitions the test and its dependencies to :
:                              : `build_include` and `cpu_backend`.           :
| `litert_configured_target` / | Transitions an existing executable or test   |
: `litert_configured_test`     : `target` to `build_include` and              :
:                              : `cpu_backend`.                               :
| `litert_cc_library`           | Defines a `cc_library` or transitions a list |
:                              : of `deps` (`CcInfo`) to `build_include` and  :
:                              : `cpu_backend`. Use for static-link targets   :
:                              : such as WebAssembly (`wasm_js_library`).     :

### Example: `litert_cc_binary` / `litert_cc_test`

```python
load(
    "//litert/build_common:litert_build_defs.bzl",
    "litert_cc_binary",
    "litert_cc_test",
)
load("//litert/build_common:special_rule.bzl", "litert_linkopts")

litert_cc_binary(
    name = "my_app",
    srcs = [
        "my_app.cc",
        ":my_model_selected_ops",
    ],
    build_include = "cpu_only",
    cpu_backend = "selective",
    linkopts = litert_linkopts(),
    linkstatic = 1,
    deps = [
        "//litert/cc:litert_compiled_model",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/tflite_support/op_resolver",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite:mutable_op_resolver",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
    ],
)

litert_cc_test(
    name = "my_app_test",
    srcs = ["my_app_test.cc"],
    build_include = "cpu_only",
    cpu_backend = "selective",
    deps = [":my_app_lib"],
)
```

### Example: `litert_cc_library` (for WebAssembly / `wasm_js_library`)

```python
load(
    "//litert/build_common:litert_build_defs.bzl",
    "litert_cc_library",
)

litert_cc_library(
    name = "my_model_runner_cpu_only",
    build_include = "cpu_only",
    cpu_backend = "selective",
    deps = [":my_model_runner"],
)

cc_library(
    name = "my_wasm_bindings",
    srcs = ["my_wasm_bindings.cc"],
    deps = [
        ":my_model_runner_cpu_only",
        "//third_party/emscripten:embind",
    ],
    alwayslink = 1,
)
```

## Runtime `kernel_mode` vs. Build Flags

`CpuOptions::SetKernelMode()` chooses at runtime which of the **already
linked** kernels to use:

| `kernel_mode` | Behavior |
| ------------- | -------- |
| `kLiteRtCpuKernelModeDelegate` (default) | Applies the XNNPACK delegate. Unsupported nodes fall back to built-in kernels. |
| `kLiteRtCpuKernelModeBuiltin` | Skips XNNPACK and uses built-in kernels only. |
| `kLiteRtCpuKernelModeReference` | Uses `BuiltinRefOpResolver` (reference kernels). |

The runtime option does **not** change binary size. For example, building
Option A and running with `kLiteRtCpuKernelModeBuiltin` still ships the whole
XNNPACK library. Use the build flags above to actually remove code.

## Typical Sizes (Linux ARM32)

Reference target: Linux on 32-bit ARM (ARMv7-A, hard float, `armhf`), built
with `-c opt`. With the open source Bazel setup, use `--config=elinux_armhf`.

Test program: `//litert/test:minimal_compiled_model`.

Sizes are the loadable ELF sections (code + read-only data + data, excluding
symbol tables and `.bss`).

Configuration                                             | Loadable size
--------------------------------------------------------- | -------------
Default (`gpu,npu`, XNNPACK, all ops)                     | 5.45 MB
Option A: `cpu_only` + XNNPACK                            | 5.11 MB
Option B: `cpu_only` + `cpu_backend=builtin`              | 3.87 MB
Option C: `cpu_only` + XNNPACK only                       | 2.68 MB
Option D: `cpu_only` + `cpu_backend=selective` (18 ops)   | 2.18 MB
Reference: TFLite `Interpreter` + selective op resolver\* | 0.74 MB

\* A separate test program that runs the same model with the TFLite
`Interpreter` API and registers only the 18 op types the model uses.

## Known Limitations

*   **XNNPACK is linked as a whole.** The XNNPACK delegate references every
    XNNPACK subgraph operator, so XNNPACK cannot be trimmed per model.

## Measuring Binary Size

*   For native binaries, compare the loadable sections (`size -A`, or
    [Bloaty](https://github.com/google/bloaty)) of an unstripped `-c opt`
    build. Symbol tables in "stripped" outputs can otherwise hide the real
    difference.
*   To find out why a library is linked, use
    `bazel cquery "somepath(//your:target, //suspicious:library)"` with the
    same flags as the build.
*   For WASM, the `.wasm.map` source map that Emscripten produces maps every
    code byte to a source file. Summing byte ranges per source path gives a
    per-library breakdown.
*   Always compare configurations with the same compilation mode, target CPU,
    allocator, and post-link optimization.
