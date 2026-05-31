# LiteRT Qualcomm Backend — Debug Features

This directory contains the LiteRT vendor backend for Qualcomm SoCs. It uses
the **Qualcomm AI Runtime (QAIRT) SDK** — also referred to as **QNN** — under
the hood. The compiler plugin lives under `compiler/` and the runtime dispatch
implementation under `dispatch/`. Backend-specific options are declared in
`litert/c/options/litert_qualcomm_options.h` and exposed as `--qualcomm_*` CLI
flags via `litert/tools/flags/vendors/qualcomm_flags.h`.

LiteRT exposes three compile-time debug knobs that map directly to QNN native
artifacts. They are independent and can be combined in a single compile run.
Each one takes an output **directory** path; if the directory is empty (the
default), the feature is off.

| Option flag | Artifact | C++ setter |
|---|---|---|
| `--qualcomm_saver_output_dir=<dir>` | `saver_output.c` + `params.bin` | `SetSaverOutputDir()` |
| `--qualcomm_ir_json_dir=<dir>` | `<graph>.json` | `SetIrJsonDir()` |
| `--qualcomm_dlc_dir=<dir>` | `<graph>.dlc` | `SetDlcDir()` |

All three are wired through `apply_plugin_main` (the AOT compile entry point).
A typical compile invocation looks like:

```bash
bazel build //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so

bazel run //litert/tools:apply_plugin_main -- \
  --cmd apply \
  --model <input.tflite> \
  --soc_manufacturer Qualcomm --soc_model SM8650 \
  --libs litert/vendors/qualcomm/compiler \
  -o <output.tflite> \
  <one or more --qualcomm_*_dir ... flags from the table above>
```

The `bazel build` of `qnn_compiler_plugin_so` is required before
`apply_plugin_main` can load the Qualcomm compiler plugin via `--libs`.

The sections below describe each feature, how to enable it, and how the
artifact relates to the QAIRT SDK. For deeper information on the QNN-side
artifacts and tools, refer to the QAIRT SDK documentation under
`third_party/qairt/latest/docs/QAIRT-Docs/QNN/`.

--------------------------------------------------------------------------------

# Features

## Saver

QNN ships a special debug **Saver Backend** (`libQnnSaver.so`) that records
every QNN API call instead of executing the graph. Setting
`--qualcomm_saver_output_dir` makes LiteRT load the Saver Backend during
compile (in place of the normal HTP/CPU/GPU/DSP backend) and ask QNN to dump
its recording into the directory you provide.

Enable it:

```bash
bazel build //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so

bazel run //litert/tools:apply_plugin_main -- \
  --cmd apply \
  --model <input.tflite> \
  --soc_manufacturer Qualcomm --soc_model SM8650 \
  --libs litert/vendors/qualcomm/compiler \
  -o <output.tflite> \
  --qualcomm_saver_output_dir /tmp/qnn_saver
```

You will get two files in the directory:

- `saver_output.c` — every QNN API call rendered as C source.
- `params.bin` — all input/output/parameter tensor bytes.

These can be compiled and replayed against any real QNN backend (CPU, HTP,
GPU, …) for offline reproduction or cross-backend diffing. Because the Saver
Backend does not execute the graph, timings and accuracy numbers from a Saver
run are not meaningful — this mode is debug-only.

> **QNN relationship.** The artifacts and replay flow are entirely standard
> QNN Saver. See `general/saver/saver_backend.html` and
> `general/saver/saver_tutorial.html` in the QAIRT SDK documentation for the
> full description and replay instructions.

--------------------------------------------------------------------------------

## IR JSON

A LiteRT-internal debug dump. After LiteRT finishes composing the QNN graph
(adding all ops and tensors via the QNN API), it can serialize the in-memory
graph to a human-readable JSON file. This is the fastest way to eyeball the
QNN graph LiteRT actually builds.

Enable it:

```bash
bazel build //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so

bazel run //litert/tools:apply_plugin_main -- \
  --cmd apply \
  --model <input.tflite> \
  --soc_manufacturer Qualcomm --soc_model SM8650 \
  --libs litert/vendors/qualcomm/compiler \
  -o <output.tflite> \
  --qualcomm_ir_json_dir /tmp/qnn_ir_json
```

You will get one `<graph_name>.json` per partition, listing the QNN ops,
tensors, and their attributes/encodings.

> **QNN relationship.** The JSON format itself is **not** part of QNN — it is
> produced by LiteRT (`core/dump/dump_graph.cc`) for debugging only. The
> *content* describes the QNN graph (op types, tensor shapes, quantization
> parameters), so the QAIRT SDK documentation under `general/operations.html`
> and the QNN op definition references are useful when interpreting the dump.

--------------------------------------------------------------------------------

## DLC

DLC (Deep Learning Container) is Qualcomm's serialized model format. QNN
provides a dedicated **IR Backend** (`libQnnIr.so`) whose only job is to
serialize a composed graph into DLC. Setting `--qualcomm_dlc_dir` causes
LiteRT to switch to the IR Backend at compile time and ask it to write a
`.dlc` file per partition.

Enable it:

```bash
bazel build //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so

bazel run //litert/tools:apply_plugin_main -- \
  --cmd apply \
  --model <input.tflite> \
  --soc_manufacturer Qualcomm --soc_model SM8650 \
  --libs litert/vendors/qualcomm/compiler \
  -o <output.tflite> \
  --qualcomm_dlc_dir /tmp/qnn_dlc
```

You will get one `<graph_name>.dlc` per partition.

> ⚠️ **Side effect.** When `--qualcomm_dlc_dir` is non-empty, the compiler
> automatically overrides the backend to the IR Backend (a warning is
> logged). The resulting compiled `.tflite` therefore reflects IR-Backend
> semantics, not HTP/CPU/GPU execution. Use this flag for artifact extraction,
> not for producing a deployable model.

> **QNN relationship.** The output DLC files are standard QNN/SNPE artifacts
> and can be consumed by QNN native tools (for example, `qnn-net-run` and
> `qnn-context-binary-generator` accept `--dlc_path`, and Qualcomm Neural
> Processing SDK provides `dlc-info` / `dlc-viewer` for inspection). See
> `general/overview.html`, `general/tools.html`, and the "Utilizing DLCs"
> tutorial (`general/tutorial5.html`) in the QAIRT SDK documentation.
