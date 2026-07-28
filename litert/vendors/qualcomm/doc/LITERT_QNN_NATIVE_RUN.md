# LiteRT Compile + QNN Native Execution

This page walks through compiling a `.tflite` model with LiteRT into a Qualcomm
**DLC** (Deep Learning Container) and then running that graph **natively** with
the QAIRT native tools (`qnn-context-binary-generator`, `qnn-net-run`) on the
x86 host and on an Android (aarch64) device. Use it when you want to run a model
through the QNN native path but still benefit from LiteRT's compile-time graph
optimizations: LiteRT lowers the `.tflite` to a `.dlc`, and from there the
standard QNN tools take over.

```mermaid
flowchart LR
    M[".tflite"] -->|"1.1 apply_plugin_main<br/>--qualcomm_dlc_dir<br/>(LiteRT)"| D["qnn_partition_&lt;N&gt;.dlc"]
    D -.->|"3.2 qnn-net-run --dlc_path<br/>(QAIRT, online prepare)"| R["Result_*/<br/>output tensors"]
    D -->|"2.1 qnn-context-binary-generator<br/>(QAIRT, offline prepare)"| B["qnn_partition_&lt;N&gt;.bin<br/>context binary"]
    M -->|"1.2 apply_plugin_main<br/>(LiteRT)"| C["compiled .tflite"]
    C -. "2.2 extract_bytecode<br/>(LiteRT)" .-> B
    B -->|"3.1 qnn-net-run --retrieve_context<br/>(QAIRT)"| R
```

| Step | Tool | Owner | Input | Output | Notes |
| ---- | ---- | ----- | ----- | ------ | ----- |
| 1.1 | `apply_plugin_main --qualcomm_dlc_dir` | LiteRT | `.tflite` | `qnn_partition_<N>.dlc` | Compiles the graph into a `.dlc`. |
| 1.2 | `apply_plugin_main` | LiteRT | `.tflite` | compiled `.tflite` | Compiles the graph into a LiteRT `.tflite` with the QNN context binary embedded. Feeds step 2.2. |
| 2.1 | `qnn-context-binary-generator` | QAIRT | `.dlc` (from step 1.1) | `qnn_partition_<N>.bin` (context binary) | Offline prepare from a `.dlc`. |
| 2.2 | `extract_bytecode` | LiteRT | compiled `.tflite` (from step 1.2) | `qnn_partition_<N>.bin` (context binary) | Extracts the embedded context binary. Skips step 1.1/2.1 if you already have a LiteRT-compiled `.tflite`. |
| 3.1 | `qnn-net-run --retrieve_context` | QAIRT | context binary (from step 2) | `Result_*/` (output tensors) | Run the pre-built context binary (solid path). |
| 3.2 | `qnn-net-run --dlc_path` | QAIRT | `.dlc` (from step 1.1) | `Result_*/` (output tensors) | Prepare the `.dlc` online (dotted path). |

> 💡 Once LiteRT has compiled the `.dlc`, the whole QNN/QAIRT toolchain is open
> to you: the `.dlc` is a standard Qualcomm artifact, so in principle every QNN
> feature applies, including any backend (CPU / GPU / HTP / DSP), quantization,
> graph inspection, context-binary caching, profiling, weight sharing, and
> custom op packages. This page covers only the run path above. For everything
> else the QNN tools can do with a `.dlc`, see the official QNN documentation:
> <https://docs.qualcomm.com/doc/80-63442-10/topic/index_QNN.html>.

--------------------------------------------------------------------------------

## Prerequisites

Before you start, make sure the toolchain in
[PREREQUISITES.md](./PREREQUISITES.md) is installed and the QNN concepts and
libraries in [QAIRT_SDK.md](./QAIRT_SDK.md) are understood.

The commands below refer to the following `${}` variables. Configure them for
your environment before running any step.

Variable                | Description
----------------------- | -----------
`${LITERT}`             | The path of the LiteRT source code.
`${QAIRT}`              | The root of the unzipped QAIRT SDK (holds `bin/`, `lib/`). For example `${LITERT}/third_party/qairt/<version>` (the SDK folder is named after its version, e.g. `2.47.0.xxxxxx`. Symlink it to a stable name yourself if you prefer).
`${SOURCE_MODEL_PATH}`  | The input `.tflite` model to compile.
`${X86_HOST_ARTIFACT_FOLDER}` | Host working directory that holds everything produced on x86: the `.dlc` files (step 1.1), the LiteRT-compiled `.tflite` (step 1.2), the extracted / generated context binaries, and the raw input tensors and input list.
`${SOC_MODEL}`          | Target SoC, e.g. `SM8850`.
`${HTP_ARCH}`           | HTP architecture of `${SOC_MODEL}`. Used in two casings: uppercase in QNN library names (`${HTP_ARCH}` = `V81`) and lowercase in the `hexagon-vXX` directory name (`${HEXAGON_ARCH}` = `v81`). See [QAIRT_SDK.md → Hexagon Arch](./QAIRT_SDK.md#hexagon-arch) for how to look this up.
`${ANDROID_TEST_FOLDER}` | Working directory on the device, e.g. `/data/local/tmp/qnn_native`. Only needed for the [Android device run](#android-device-aarch64).

> 💡 LiteRT names each compiled partition `qnn_partition_0`, `qnn_partition_1`,
> … The examples below use `qnn_partition_0` (a single-partition model).
> Substitute the relevant partition name if your model splits into several.

--------------------------------------------------------------------------------

## 1. Compile the .tflite with LiteRT

LiteRT compiles the `.tflite` in one of two ways, matching the two context-binary
sources in [§2](#2-build-a-context-binary-on-the-x86-host). Both invoke the same
`apply_plugin_main`. The only difference is whether `--qualcomm_dlc_dir` is set.
You only need the one that feeds the §2 path you plan to use.

Build the tool and the plugin once (used by both):

```bash
cd ${LITERT}

bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/tools:apply_plugin_main
bazel build -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so

export LD_LIBRARY_PATH=${QAIRT}/lib/x86_64-linux-clang
```

> **`--libs` must point at `bazel-bin/`, not the source tree** — the plugin `.so`
> lives under `bazel-bin/litert/vendors/qualcomm/compiler/` after the build
> above. The commands below use the explicit `${LITERT}/bazel-bin/...` path for
> that reason.

### 1.1 Compile to a .dlc (`--qualcomm_dlc_dir`)

`--qualcomm_dlc_dir=<dir>` makes LiteRT switch to the QNN **IR Backend** at
compile time and serialize each composed graph to a `.dlc` in that directory.
The flag takes a **directory**. If empty (the default), the feature is off. Use
this to feed the `qnn-context-binary-generator` path ([§2.1](#21-from-a-dlc-with-qnn-context-binary-generator)).

```bash
bazel-bin/litert/tools/apply_plugin_main \
  --cmd apply \
  --model ${SOURCE_MODEL_PATH} \
  --soc_manufacturer Qualcomm --soc_model ${SOC_MODEL} \
  --libs ${LITERT}/bazel-bin/litert/vendors/qualcomm/compiler \
  -o ${X86_HOST_ARTIFACT_FOLDER}/ir_backend_out.tflite \
  --qualcomm_dlc_dir ${X86_HOST_ARTIFACT_FOLDER}
```

This writes **one `qnn_partition_<N>.dlc` per partition** into
`${X86_HOST_ARTIFACT_FOLDER}`. A model that LiteRT splits into multiple NPU
partitions yields multiple `.dlc` files (one per QNN graph). A single-partition
model yields just `qnn_partition_0.dlc`. Run / inspect each `.dlc` separately in
the steps below.

> ⚠️ **Backend override side effect.** When `--qualcomm_dlc_dir` is non-empty, the
> compiler forces the backend to the IR Backend (a warning is logged). The
> resulting `-o` `.tflite` reflects IR-Backend semantics, **not** HTP/CPU/GPU
> execution. This flag is for artifact extraction, not for producing a
> deployable model.

### 1.2 Compile to a LiteRT `.tflite`

Without `--qualcomm_dlc_dir`, `apply_plugin_main` produces a normal LiteRT
compile output: a `.tflite` that **embeds the QNN context binary** inside its
dispatch op(s). Use this to feed the `extract_bytecode` path
([§2.2](#22-from-a-litert-compiled-tflite-with-extract_bytecode)).

```bash
bazel-bin/litert/tools/apply_plugin_main \
  --cmd apply \
  --model ${SOURCE_MODEL_PATH} \
  --soc_manufacturer Qualcomm --soc_model ${SOC_MODEL} \
  --libs ${LITERT}/bazel-bin/litert/vendors/qualcomm/compiler \
  -o ${X86_HOST_ARTIFACT_FOLDER}/compiled.tflite
```

This is the standard HTP compile flow. See
[HTP_INSTRUCTIONS.md](./HTP_INSTRUCTIONS.md) for the full walkthrough (including
the CMake build and on-device options). The `-o` `compiled.tflite` is what §2.2 reads.

--------------------------------------------------------------------------------

## 2. Build a context binary (on the x86 host)

For HTP you typically pre-build a **context binary** once on the x86 host and
run it on device. This moves the (slow) graph-prepare step off the device's
critical path. There are two ways to produce the same `.bin`, both run on the
x86 host:

| Source you have | Tool | Owner | When to use |
| --------------- | ---- | ----- | ----------- |
| `.dlc` from step 1.1 | `qnn-context-binary-generator` | QAIRT | You went through step 1.1 specifically to emit a `.dlc`. |
| A LiteRT-compiled `.tflite` (step 1.2, no `--qualcomm_dlc_dir`) | `extract_bytecode` | LiteRT | You already have a LiteRT-compiled model and want the embedded context binary, no need to recompile. |

### 2.1 From a .dlc with qnn-context-binary-generator

Use this when you produced a `.dlc` in step 1.1.

```bash
${QAIRT}/bin/x86_64-linux-clang/qnn-context-binary-generator \
  --backend     ${QAIRT}/lib/x86_64-linux-clang/libQnnHtp.so \
  --model       ${QAIRT}/lib/x86_64-linux-clang/libQnnModelDlc.so \
  --dlc_path    ${X86_HOST_ARTIFACT_FOLDER}/qnn_partition_0.dlc \
  --binary_file qnn_partition_0
```

`--backend` is the QNN backend the context is prepared for (`libQnnHtp.so`).
`--model libQnnModelDlc.so` is the generic **DLC loader** library (required to
read a `.dlc`, not your graph). The output lands at
`./output/qnn_partition_0.bin` by default.

You can skip this step entirely and let `qnn-net-run` prepare the `.dlc` online
(see the x86 host run in §3), but for device HTP runs the pre-built context
binary is the common path.

### 2.2 From a LiteRT-compiled .tflite with extract_bytecode

Use this when you already have a `.tflite` **compiled by LiteRT**
([step 1.2](#12-compile-to-a-litert-tflite),
`apply_plugin_main` output, no `--qualcomm_dlc_dir`). Such a `.tflite` already
embeds the QNN context binary inside its dispatch op, so steps 1.1 / 2.1 are not
needed. The tool writes one `<name>_<index>.bin` per dispatch op into
`${X86_HOST_ARTIFACT_FOLDER}`. Feed each `.bin` to `qnn-net-run
--retrieve_context` exactly as in §3.

Build and run with Bazel:

```bash
cd ${LITERT}

bazel run -c opt --cxxopt=--std=c++17 --nocheck_visibility //litert/tools:extract_bytecode -- \
  --model_path ${X86_HOST_ARTIFACT_FOLDER}/compiled.tflite \
  --output_dir ${X86_HOST_ARTIFACT_FOLDER}
```

Or with CMake (configure once, then build):

```bash
cd ${LITERT}/litert

# Configure once (skip if cmake_build/ already configured).
cmake -S . -B cmake_build

cmake --build cmake_build --target extract_bytecode -j8

cmake_build/tools/extract_bytecode \
  --model_path ${X86_HOST_ARTIFACT_FOLDER}/compiled.tflite \
  --output_dir ${X86_HOST_ARTIFACT_FOLDER}
```

--------------------------------------------------------------------------------

## 3. Run natively with qnn-net-run

`qnn-net-run` loads the graph in two ways: from a `.dlc` (online prepare) or
from a pre-built context binary (offline prepare). **Both work on both x86 and
Android.** The choice is about *when* the graph is prepared, not *where* it
runs. The x86 and device sections below are example environments, not a
restriction.

| Input flag | Prepare | Typical use |
| ---------- | ------- | ----------- |
| `--dlc_path <dlc>` (with `--model libQnnModelDlc.so`) | Online, at load | Convenient on the host |
| `--retrieve_context <bin>` | Offline (step 2), pre-built | On-device HTP, to skip the slow prepare at run |

Also pass these so raw tensors keep the graph's **native** dtype. Without them
`qnn-net-run` reads inputs as float32 and writes float32 outputs, silently
misreading a quantized (`uint8` / `int8`) graph.

| Flag | Effect |
| ---- | ------ |
| `--use_native_input_files` | Read input `.raw` in the graph's native dtype |
| `--use_native_output_files` | Write output `.raw` in the graph's native dtype |

### x86 host

Use `libQnnHtp.so` for the HTP x86 reference, or `libQnnCpu.so` for a pure-CPU
reference run. This example prepares the `.dlc` online. To run a pre-built
context binary instead, replace `--model ...` and `--dlc_path ...` with
`--retrieve_context ./output/qnn_partition_0.bin` (the binary from step 2.1).

```bash
${QAIRT}/bin/x86_64-linux-clang/qnn-net-run \
  --backend    ${QAIRT}/lib/x86_64-linux-clang/libQnnHtp.so \
  --model      ${QAIRT}/lib/x86_64-linux-clang/libQnnModelDlc.so \
  --dlc_path   ${X86_HOST_ARTIFACT_FOLDER}/qnn_partition_0.dlc \
  --input_list ${X86_HOST_ARTIFACT_FOLDER}/input_list.txt \
  --output_dir ${X86_HOST_ARTIFACT_FOLDER}/qnn_out \
  --use_native_input_files --use_native_output_files
```

Outputs are written under `--output_dir` (default `./output`), one `Result_N/`
per inference line in the input list. See [§4](#4-inputs-and-outputs) for the
input / output file formats.

### Android device (aarch64)

The example uses **`${HTP_ARCH}` = `V81`** (SM8850). Substitute the values for
your SoC from the Prerequisites table. It runs the pre-built context binary from
step 2. To prepare the `.dlc` online on device instead, push the `.dlc` and swap
`--retrieve_context ./qnn_partition_0.bin` for
`--model ./libQnnModelDlc.so --dlc_path ./qnn_partition_0.dlc`.

```bash
adb shell "mkdir -p ${ANDROID_TEST_FOLDER}"

# Tool + DLC loader + backend (ARM side)
adb push ${QAIRT}/bin/aarch64-android/qnn-net-run                  ${ANDROID_TEST_FOLDER}/
adb push ${QAIRT}/lib/aarch64-android/libQnnModelDlc.so            ${ANDROID_TEST_FOLDER}/
adb push ${QAIRT}/lib/aarch64-android/libQnnHtp.so                 ${ANDROID_TEST_FOLDER}/
adb push ${QAIRT}/lib/aarch64-android/libQnnHtp${HTP_ARCH}Stub.so  ${ANDROID_TEST_FOLDER}/
adb push ${QAIRT}/lib/aarch64-android/libQnnHtpPrepare.so          ${ANDROID_TEST_FOLDER}/
adb push ${QAIRT}/lib/aarch64-android/libQnnSystem.so              ${ANDROID_TEST_FOLDER}/

# DSP-side skel (unsigned build for a dev device)
adb push ${QAIRT}/lib/hexagon-${HEXAGON_ARCH}/unsigned/libQnnHtp${HTP_ARCH}Skel.so ${ANDROID_TEST_FOLDER}/

# Artifacts: context binary (from step 2) + inputs
adb push ./output/qnn_partition_0.bin        ${ANDROID_TEST_FOLDER}/
adb push ${X86_HOST_ARTIFACT_FOLDER}/input0.raw        ${ANDROID_TEST_FOLDER}/
adb push ${X86_HOST_ARTIFACT_FOLDER}/input_list.txt    ${ANDROID_TEST_FOLDER}/

adb shell "
  cd ${ANDROID_TEST_FOLDER}
  LD_LIBRARY_PATH=${ANDROID_TEST_FOLDER} ADSP_LIBRARY_PATH=${ANDROID_TEST_FOLDER} \
  ./qnn-net-run \
    --backend          ./libQnnHtp.so \
    --retrieve_context ./qnn_partition_0.bin \
    --input_list       ./input_list.txt \
    --output_dir       ./qnn_out \
    --use_native_input_files --use_native_output_files"

adb pull ${ANDROID_TEST_FOLDER}/qnn_out ./qnn_out_device
```

> ⚠️ `LD_LIBRARY_PATH` must include the ARM-side libs, and `ADSP_LIBRARY_PATH`
> must include the dir holding the Hexagon skel. The `input_list.txt` pushed to
> the device must contain **on-device** paths (e.g.
> `${ANDROID_TEST_FOLDER}/input0.raw`), not host paths.

--------------------------------------------------------------------------------

## 4. Inputs and outputs

**`--input_list`** is plain text, one inference per line. Each line lists the
raw input file(s) for that inference, space-separated, one entry per graph
input. Use **absolute paths** (relative paths resolve against the tool's working
dir, which differs on device).

```
/abs/path/inputs/input0_0.raw
/abs/path/inputs/input0_1.raw
```

Each `.raw` is the raw little-endian bytes of one input tensor in the graph's
expected dtype and layout (e.g. quantized `uint8` / `int8`, or `float32`). This
is the format the native-dtype flags in [§3](#3-run-natively-with-qnn-net-run)
expect. Inspect the expected I/O with `qairt-dlc-info`
([§5](#5-inspecting-a-dlc-and-further-reading)) to get each tensor's exact
shape and dtype, then write a matching `.raw`. For example, for a
`float32 [1,64,128]` input:

```bash
python3 -c "import numpy as np; \
  np.random.rand(1,64,128).astype(np.float32).tofile('${X86_HOST_ARTIFACT_FOLDER}/input0.raw')"
echo ${X86_HOST_ARTIFACT_FOLDER}/input0.raw > ${X86_HOST_ARTIFACT_FOLDER}/input_list.txt
```

`qnn-net-run` writes one subdirectory per inference under `--output_dir`
(`Result_0/`, `Result_1/`, …), each holding one `.raw` per output tensor (named
after the graph output, e.g. `StatefulPartitionedCall_0.raw`).

--------------------------------------------------------------------------------

## 5. Inspecting a .dlc and further reading

The `qairt-dlc-*` inspection tools are Python and need the SDK's Python modules
on `PYTHONPATH` (otherwise they fail with `No module named 'qti'`):

```bash
export PYTHONPATH=${QAIRT}/lib/python:$PYTHONPATH

qairt-dlc-info    -i ${X86_HOST_ARTIFACT_FOLDER}/qnn_partition_0.dlc   # graph structure, tensor shapes,
                                               # dtypes, quantization encodings,
                                               # and the network input/output list
qairt-dlc-to-json -i ${X86_HOST_ARTIFACT_FOLDER}/qnn_partition_0.dlc   # dump the .dlc to JSON
```

Use the input/output table `qairt-dlc-info` prints to get each input's exact
name, dimensions, and dtype before writing the `.raw` files in
[§4](#4-inputs-and-outputs).

For deeper QNN-side detail, the QAIRT SDK docs (under
`${QAIRT}/docs/QAIRT-Docs/`) include the "Utilizing DLCs" tutorial
(`QNN/general/tutorial5.html`) and the full tool reference
(`QNN/general/tools.html`).

> The newer QAIRT wrappers `qairt-net-run` / `qairt-dlc-prepare` cover the same
> flow (see `QAIRT-API/migration-guide.html`). This page uses the `qnn-*` tools,
> which are the most widely documented.
