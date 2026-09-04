# Python AOT Compilation (pip)

This page walks through compiling a `.tflite` model for the Qualcomm HTP (NPU)
backend on an **x86 Linux host** using pip-installed packages, with **no LiteRT
source build (Bazel/CMake) required**. It is the fastest way to produce a
compiled model or a QNN `.dlc` when you only need ahead-of-time (**AOT**) model
compilation.

> Throughout this page, "compile" means compiling the **model** (lowering the
> `.tflite` graph into a QNN context binary), not building the LiteRT source
> code.

**Scope:** this flow covers **AOT model compilation on x86 Linux**, plus
**host-side CPU verification**. It does **not** run the Qualcomm NPU. NPU
execution happens on device with `run_model` (see
[Run the model](#run-the-model) and [HTP_INSTRUCTIONS.md](./HTP_INSTRUCTIONS.md)).

Two ways to compile are covered, followed by how to run the result:

| Method | Interface | Section |
| ------ | --------- | ------- |
| **1** | `ai_edge_litert.aot` high-level Python API | [Method 1: Python API](#method-1-compile-with-the-python-api-aot_compile) |
| **2** | `apply_plugin_main` command-line tool | [Method 2: CLI](#method-2-compile-with-the-cli-apply_plugin_main) |
| **Run** | CPU on host (Python) / NPU on device (`run_model`) | [Run the model](#run-the-model) |

--------------------------------------------------------------------------------

## Prerequisites

*   **x86-64 Linux** host. Compilation is Linux-x86 only; on other platforms the
    SDK package raises `NotImplementedError`.
*   **Python 3.10+** in a fresh virtual environment. The packages are published
    per-Python-version, so match your interpreter.
*   **~6 GB free disk.** The Qualcomm SDK package unpacks to ~5.5 GB (it bundles
    the full QAIRT/QNN runtime for every target).
*   **Network access to PyPI.** The Qualcomm SDK package builds from an sdist and
    can take several minutes to install.

--------------------------------------------------------------------------------

## Install

Create a virtual environment and install the two packages:

```bash
python -m venv litert-converter
source litert-converter/bin/activate

# The LiteRT tooling: compiler CLI, compiler plugin, and Python AOT API.
pip install ai-edge-litert

# The Qualcomm QAIRT/QNN SDK libraries (large; builds from sdist).
pip install ai-edge-litert-sdk-qualcomm
```

Two packages are needed because they are complementary:

*   **`ai-edge-litert`** provides the compiler front end: the `apply_plugin_main`
    tool, the Qualcomm compiler plugin `.so`, and the `ai_edge_litert.aot` Python
    API.
*   **`ai-edge-litert-sdk-qualcomm`** provides the QNN backend libraries
    (`libQnnHtp.so`, `libQnnIr.so`, `libQnnSystem.so`, …) that the compiler
    plugin loads at runtime. It is the QAIRT SDK repackaged as a wheel (QAIRT
    2.47.0 / QNN backend API 2.18.0 at the time of writing).

--------------------------------------------------------------------------------

## What you get

After installing, the artifacts that matter live under your venv's
`site-packages/` directory:

| Package | Artifact | Path (under `site-packages/`) |
| ------- | -------- | ----------------------------- |
| ai-edge-litert | Compile CLI | `ai_edge_litert/tools/apply_plugin_main` |
| ai-edge-litert | Compiler plugin | `ai_edge_litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so` |
| ai-edge-litert | Python AOT API | `ai_edge_litert/aot/aot_compile.py` |
| ai-edge-litert-sdk-qualcomm | QNN host libraries (for compile) | `ai_edge_litert_sdk_qualcomm/data/lib/x86_64-linux-clang/` |
| ai-edge-litert-sdk-qualcomm | On-device / other-target libraries | `ai_edge_litert_sdk_qualcomm/data/lib/{aarch64-android, hexagon-v*, …}/` |

The compiler plugin dynamically links the QNN libraries, so the SDK package's
`x86_64-linux-clang` directory must be on `LD_LIBRARY_PATH` when you compile
with the CLI (the [Method 2](#method-2-compile-with-the-cli-apply_plugin_main)
command sets this up; the [Method 1](#method-1-compile-with-the-python-api-aot_compile)
Python API handles it for you). You can resolve that directory programmatically:

```python
import ai_edge_litert_sdk_qualcomm as q
print(q.path_to_sdk_libs())   # .../ai_edge_litert_sdk_qualcomm/data/lib/x86_64-linux-clang
```

The `ai-edge-litert-sdk-qualcomm` package also ships the runtime libraries for
on-device targets (`aarch64-android`, `hexagon-v*`, etc.), which you can reuse
when deploying the compiled model to a device.

--------------------------------------------------------------------------------

## Variables

The commands below refer to the following `${}` variables. Configure them for
your environment before running any step.

Variable                 | Description
------------------------ | -----------
`${SOURCE_MODEL_PATH}`   | Path to the input `.tflite` model (e.g. `model.tflite`).
`${COMPILED_MODEL_PATH}` | Path for the AOT-compiled output `.tflite` model (e.g. `model_compiled.tflite`).
`${SOC_MODEL}`           | Target SoC, e.g. `SM8850` (Snapdragon 8 Gen 5), `SM8750` (8 Elite), `SM8650` (8 Gen 3). See [supported_soc.csv](../supported_soc.csv).

--------------------------------------------------------------------------------

## Method 1: Compile with the Python API (`aot_compile`)

A pure-Python call that performs the backend pre-processing and plugin
application for you, driven from a `Target`.

```python
from ai_edge_litert.aot import aot_compile as aot
from ai_edge_litert.aot.vendors.qualcomm import target as qualcomm

result = aot.aot_compile(
    "model.tflite",                       # ${SOURCE_MODEL_PATH}
    output_dir="compiled_models",
    target=qualcomm.Target(soc_model=qualcomm.SocModel.SM8850),
)
print(result)
```

The compiled model is written to
`compiled_models/<name>_Qualcomm_<SOC>_apply_plugin.tflite` (for the example
above, `model_Qualcomm_SM8850_apply_plugin.tflite`).

`aot_compile` is defined in the `ai_edge_litert.aot.aot_compile` submodule (it is
not re-exported from `ai_edge_litert.aot`), so import it directly as shown.

Supported `SocModel` values: `SA8255`, `SA8295`, `SM8350`, `SM8450`, `SM8475`,
`SM8550`, `SM8650`, `SM8750`, `SM8845`, `SM8850`, and `ALL` (compile for every
registered SoC). See [supported_soc.csv](../supported_soc.csv) for the full
device list.

--------------------------------------------------------------------------------

## Method 2: Compile with the CLI (`apply_plugin_main`)

If you prefer a shell command over Python, or need the `.dlc` dump below, call
the `apply_plugin_main` tool directly. It produces the same compiled `.tflite`
as [Method 1](#method-1-compile-with-the-python-api-aot_compile).

The command needs two directories on `LD_LIBRARY_PATH`: the `ai_edge_litert`
package directory (for `libLiteRt.so`) and the Qualcomm SDK host-lib directory.
The snippet below resolves both from Python so it is copy-pasteable:

```bash
LITERT_DIR=$(python -c "import ai_edge_litert, os; print(os.path.dirname(ai_edge_litert.__file__))")
SDK_LIBS=$(python -c "import ai_edge_litert_sdk_qualcomm as q; print(q.path_to_sdk_libs())")

LD_LIBRARY_PATH="${LITERT_DIR}:${SDK_LIBS}" \
"${LITERT_DIR}/tools/apply_plugin_main" \
    --cmd apply \
    --libs "${LITERT_DIR}/vendors/qualcomm/compiler/" \
    --soc_model ${SOC_MODEL} \
    --soc_manufacturer Qualcomm \
    --model ${SOURCE_MODEL_PATH} \
    -o ${COMPILED_MODEL_PATH}
```

> Note: `LD_LIBRARY_PATH` entries are separated by `:` on Linux (not `;`).

This writes the compiled model to `${COMPILED_MODEL_PATH}`, ready to push to a
device (see [Run the model](#run-the-model)).

### Dump the QNN graph as a `.dlc`

Add `--qualcomm_dlc_dir <dir>` to also emit one `.dlc` per compiled partition
into `<dir>`:

```bash
LD_LIBRARY_PATH="${LITERT_DIR}:${SDK_LIBS}" \
"${LITERT_DIR}/tools/apply_plugin_main" \
    --cmd apply \
    --libs "${LITERT_DIR}/vendors/qualcomm/compiler/" \
    --soc_model ${SOC_MODEL} \
    --soc_manufacturer Qualcomm \
    --model ${SOURCE_MODEL_PATH} \
    -o ${COMPILED_MODEL_PATH} \
    --qualcomm_dlc_dir .
```

The partition graphs are written as `qnn_partition_0.dlc`,
`qnn_partition_1.dlc`, … in the given directory.

--------------------------------------------------------------------------------

## Run the model

> **The Qualcomm NPU cannot be executed from this x86 pip flow.** NPU execution
> requires the vendor dispatch library (`libLiteRtDispatch_Qualcomm.so`) and NPU
> hardware, neither of which exists on an x86 host. The `ai-edge-litert-sdk-qualcomm`
> package ships no x86 Qualcomm dispatch library. On the host you can only run
> **CPU** (and GPU where supported); NPU execution happens **on device**.

The compiled model runs on the **NPU on device**. Optionally, you can also run
the **original** model on **CPU on the host** to get a reference output, which
mirrors how `npu_numerics_check` validates a model:

| Where | What to run | How |
| ----- | ----------- | --- |
| **Device (aarch64)** | the **compiled** `.tflite` on **NPU** | `run_model`, see [HTP_INSTRUCTIONS.md → Run on device (Android)](./HTP_INSTRUCTIONS.md#run-on-device-android) |
| **Host (x86), optional** | the **original** `.tflite` on **CPU** | Python API below, produces the CPU reference output |

### Run on device (NPU)

This is the main path for executing the model you just compiled. Push the
**compiled** `${COMPILED_MODEL_PATH}` together with the on-device runtime
libraries and `run_model` to the target, then execute with the NPU accelerator.
The QNN device libraries (`aarch64-android`, `hexagon-v*`, …) are available under
`ai_edge_litert_sdk_qualcomm/data/lib/` from the SDK package, so you can reuse
them instead of a separate QAIRT SDK download. Follow
[HTP_INSTRUCTIONS.md → Run on device (Android)](./HTP_INSTRUCTIONS.md#run-on-device-android).

### Verify on the host (CPU) with Python (optional)

To sanity-check numerics without a device, run the **original** `.tflite` (not
the compiled one) on CPU to get a reference result. The compiled `.tflite`
embeds an NPU context binary and will fail to execute on the host CPU; it is
meant for on-device NPU dispatch.

Using the `CompiledModel` API (the execution counterpart to the compile step):

```python
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator

# Load the ORIGINAL .tflite for CPU execution (NPU is device-only).
model = CompiledModel.from_file(
    "model.tflite", hardware_accel=HardwareAccelerator.CPU
)

# Allocate the input/output TensorBuffers for signature index 0.
inputs = model.create_input_buffers(0)
outputs = model.create_output_buffers(0)

# Fill inputs, run, read outputs.
inputs[0].write(np.array([[1, 2], [3, 4]], dtype=np.float32))
model.run_by_index(0, inputs, outputs)
print(outputs[0].read(4, np.float32))
```

Or with the classic `Interpreter` API:

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
interpreter.set_tensor(inp["index"], np.array([[1, 2], [3, 4]], dtype=np.float32))
interpreter.invoke()
print(interpreter.get_tensor(out["index"]))
```

`HardwareAccelerator` is a bit flag (`CPU=1`, `GPU=2`, `NPU=4`). `NPU` exists in
the enum but will fail on the host for the reason above; use it only in an
on-device build that has the Qualcomm dispatch library.

--------------------------------------------------------------------------------

## Next steps

*   **Tune compilation.** Both methods accept the Qualcomm compile-time options
    (`--qualcomm_*` on the CLI, or config kwargs to `aot_compile`). See
    [OPTIONS_REFERENCE.md](./OPTIONS_REFERENCE.md) for the full list.
*   **Inspect the QNN graph.** Use the `.dlc` files from
    [Method 2](#dump-the-qnn-graph-as-a-dlc) with the QAIRT native tools. See
    [LITERT_QNN_NATIVE_RUN.md](./LITERT_QNN_NATIVE_RUN.md).
