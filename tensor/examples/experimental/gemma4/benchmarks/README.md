<!--
Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Optional Gemma4 CPU performance reproduction

This package compares the Tensor API LiteRT Gemma4 example with the CPU XNNPACK
executor in the **latest official [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)**.
Use LiteRT-LM's own dependency revisions. The comparison measures both current
stacks; it does not promise the historical numbers in the example README or
isolate the effect of graph layout from changes to their runtime dependencies.

The package includes a raw-token LiteRT-LM client, a fixed 1024-prompt/64-decode
workload, a prompt sweep generator, Android deployment and capture, and strict
result validation. The client is an added Bazel package: it uses the official
executor without modifying its implementation. YNNPACK and speculative decoding
are disabled explicitly. Normal prefill signature selection is the primary
baseline; selecting only `prefill_128` is an optional second experiment.

## Inputs and tools

Use a Linux host with Python 3, Bazelisk, adb, and an Android SDK/NDK. Bundle
preparation also needs NumPy and FlatBuffers. Building is independent of model
access; obtain the published model under its applicable license before running.
Set these variables to your own locations:

- `LITERT_SRC`: this branch's source directory.
- `LM_SRC`: a separate official LiteRT-LM checkout at current `main`.
- `WORK`: a new directory for binaries, prepared weights, and captures.
- `MODEL`: the original published E2B `.litertlm` file.
- `ANDROID_SDK`, `ANDROID_NDK`: installed SDK and NDK directories.
- `REMOTE`: a writable directory on the Android device.
- `MASK`: a hexadecimal CPU affinity mask for the cores being compared.

Use absolute paths for these variables because the commands change directories.
The scripts accept an optional `--serial` to select an adb device. Choose a mask
appropriate for that device and the same thread count for both executors; no
particular core numbering is assumed.

```sh
git clone https://github.com/google-ai-edge/LiteRT-LM.git "$LM_SRC"
BENCH="$LITERT_SRC/tensor/examples/experimental/gemma4/benchmarks"
mkdir -p "$WORK/bin"
```

If reusing a checkout, update it to current official `main` first. Avoid local
runtime/dependency patches or repository overrides for the official baseline.
The capture records both Git revisions, tracked-dirty status, binary and model
hashes, adapter source hash, and effective run options. Keep the build commands
with the results if you change the commands below. Builds from a dirty source
are identified but cannot be reconstructed from a commit ID alone.

A known E2B source for the supplied fixture is
[`litert-community/gemma-4-E2B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/616f4124e6ff216292f16e7f73ff33b5ba9a4dd4),
file `gemma-4-E2B-it.litertlm`, at model revision
`616f4124e6ff216292f16e7f73ff33b5ba9a4dd4`:

- Size: `2583085056` bytes.
- SHA-256: `ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42`.

This model revision is an input reference, **not a LiteRT-LM runtime pin**. A newer
compatible E2B bundle can be used by preparing weights and exporting the graph
again. Both executors must use coefficients from the same source file. The
orchestrator checks that relationship and verifies prepared-file hashes on the
host and device. The exporter supports E2B; this workflow does not claim E4B
support.

## Prepare and build

First follow [bundle preparation and source validation](../REPRODUCING.md#optional-published-bundle-preparation).
For a complete invocation, after installing the Python dependencies:

```sh
cd "$LITERT_SRC"
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //tensor/examples/experimental/gemma4:export_model @flatbuffers//:flatc
cp bazel-bin/tensor/examples/experimental/gemma4/export_model "$WORK/bin/export_model"
mkdir -p "$WORK/schema"
bazel-bin/external/flatbuffers/flatc --python -o "$WORK/schema" \
  tflite/converter/schema/schema.fbs
python3 tensor/examples/experimental/gemma4/tools/prepare_bundle.py \
  --model "$MODEL" --schema-dir "$WORK/schema" --output-dir "$WORK/prepared"
"$WORK/bin/export_model" --bundle_dir "$WORK/prepared/weights" \
  --model_path "$WORK/model.tflite" --prefill_chunk_rows 128
```

Inspect `prepared/audit/export-validation.json` before proceeding. Do not run
preparation under Python `-O`, which disables its assertions. The current graph
uses the original static activation scales; do not label its measurements as
those of earlier dynamic-INT4 authoring experiments.

Build the Android Tensor API runner from this branch. These commands use NDK 28,
API 35, and build tools 35.0.0; adjust the version flags together if using a
different installation.

```sh
cd "$LITERT_SRC"
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 --config=android_arm64 \
  --repo_env=ANDROID_SDK_HOME="$ANDROID_SDK" \
  --repo_env=ANDROID_NDK_HOME="$ANDROID_NDK" \
  --repo_env=ANDROID_NDK_VERSION=28 --repo_env=ANDROID_SDK_API_LEVEL=35 \
  --repo_env=ANDROID_NDK_API_LEVEL=35 --repo_env=ANDROID_BUILD_TOOLS_VERSION=35.0.0 \
  --define=litert_runtime_link_mode=static \
  //tensor/examples/experimental/gemma4:runner
cp bazel-bin/tensor/examples/experimental/gemma4/runner "$WORK/bin/runner.android"
```

Create a new package in LiteRT-LM. `BUILD.litert_lm` is deliberately named as a
template here so LiteRT does not try to load LiteRT-LM workspace labels.

```sh
mkdir "$LM_SRC/gemma4_benchmark"
cp "$BENCH/litert_lm_harness/adapter.cc" "$LM_SRC/gemma4_benchmark/adapter.cc"
cp "$BENCH/litert_lm_harness/BUILD.litert_lm" "$LM_SRC/gemma4_benchmark/BUILD"
cd "$LM_SRC"
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 --config=android_arm64 \
  --repo_env=ANDROID_SDK_HOME="$ANDROID_SDK" \
  --repo_env=ANDROID_NDK_HOME="$ANDROID_NDK" \
  --repo_env=ANDROID_NDK_VERSION=28 --repo_env=ANDROID_SDK_API_LEVEL=35 \
  --repo_env=ANDROID_NDK_API_LEVEL=35 --repo_env=ANDROID_BUILD_TOOLS_VERSION=35.0.0 \
  --define=litert_runtime_link_mode=static \
  --define=DISABLE_HUGGINGFACE_TOKENIZER=1 \
  --define=DISABLE_SENTENCEPIECE_TOKENIZER=1 \
  //gemma4_benchmark:adapter
cp bazel-bin/gemma4_benchmark/adapter "$WORK/bin/litertlm.android"
```

The tokenizer flags only remove unused dependencies: both runners consume token
IDs directly. The adapter uses internal LiteRT-LM APIs, so future upstream API
changes may require adapting this small client. Keep such changes with the
reproduction package and rebuild; the orchestrator verifies the copied adapter
matches its source.

## Run the matched workload

The following shell function makes it easier to keep all common options the same
across experiments. Use a new output directory each time. The default fixture
has 1024 prompt tokens and 64 forced decode inputs. Step zero predicts after the
prompt, followed by 64 timed decode steps; there are 65 logits rows per session.

```sh
run_compare() {
  python3 "$BENCH/run_phone.py" \
    --litert-src "$LITERT_SRC" --lm-src "$LM_SRC" \
    --runner-bin "$WORK/bin/runner.android" --lm-bin "$WORK/bin/litertlm.android" \
    --model "$WORK/model.tflite" --source-model "$MODEL" \
    --bundle-dir "$WORK/prepared/weights" --remote "$REMOTE" \
    --mask "$MASK" --threads 4 --warmups 1 --runs 3 --blocks 2 \
    --thermal-mode gate "$@"
}
run_compare --capacities 8448 --output "$WORK/plan" --dry-run
run_compare --capacities 8448 --output "$WORK/matched"
python3 "$BENCH/summarize.py" --input "$WORK/matched" \
  --output "$WORK/matched-summary.json" --markdown "$WORK/matched-summary.md"
```

The dry run checks local inputs and writes a plan without invoking adb. Device
runs prime persistent weight caches outside measurement, then launch one process
per case/capacity/backend/block. Blocks alternate order (LiteRT, LiteRT-LM;
LiteRT-LM, LiteRT). Within each process, one warmup and three measured sessions
reuse the executor and reset the sequence. This gives six measured sessions per
configuration with the defaults. Setup, reset, priming and file I/O are outside
latency measurements. This is a warm CPU comparison, not a cold-start benchmark.

Thermal gating requires two consecutive CPU/skin readings at or below the limits
(default 45/34 degrees C), no active CPU cooling device, and a cooldown. Adjust
`--cpu-max`, `--skin-max`, and `--cooldown` for the device. If Android does not
expose these readings, explicitly use `--thermal-mode record`; `off` disables
sampling. A timeout or failed run aborts the campaign. After a timeout, verify the
remote process has ended before starting another campaign. Keep charging state,
background load and power mode stable; record mode alone does not establish
thermal comparability. The JSON includes thermal aggregates per process.

## Vary one factor at a time

Prompt length at fixed cache capacity:

```sh
python3 "$BENCH/make_cases.py" --prompt-lengths 1 128 1024 4096 \
  --decode-tokens 64 --output "$WORK/prompt-fixtures"
run_compare --cases "$WORK/prompt-fixtures/cases.tsv" --capacities 8448 \
  --output "$WORK/prompt-sweep"
```

Configured capacity at a fixed 1024-token prompt:

```sh
run_compare --capacities 2048 8448 16896 --output "$WORK/capacity-sweep"
```

The driver skips cases whose prompt plus continuation exceeds capacity and
records each skip. A capacity must also accommodate the model's selected prefill
signatures: very small capacities can fail LiteRT-LM tensor preparation even
when the token schedule fits. The suggested sweep avoids that issue for the
referenced bundle. Use the prompt sweep to study work that grows with active
context and the capacity sweep to study allocation or work tied to reserved
capacity. They help prioritize profiling; they do not by themselves attribute a
cost to a specific operator or memory copy.

Optional fixed 128-row LiteRT-LM signature selection:

```sh
run_compare --capacities 8448 --lm-prefill prefill_128 \
  --output "$WORK/lm-prefill128"
```

This uses `SetSelectedSignatures({"decode", "prefill_128"})`. It does not rewrite
the model file. Compare this result separately with the primary `auto` baseline.
For an untimed diagnostic run, the adapter accepts `--trace_signatures true` and
prints every graph invocation. Its manifest then marks timings invalid for
benchmarking, and the summarizer rejects them. Do not enable tracing in timed
captures.

To change Tensor API chunk rows, re-export with `--prefill_chunk_rows` and record
that value with `run_phone.py --prefill-rows`. For dynamic rows or explicit
shapes, export with `--experimental_dynamic_prefill` and/or
`--experimental_explicit_dynamic_shapes`, and pass `--dynamic-prefill` and/or
`--strict-shapes` respectively to the driver. A compact-INT2 preparation also
requires `--experimental_compact_int2` when exporting and `--compact-int2` when
capturing. These settings are recorded declarations; the driver does not infer
export options from a FlatBuffer. Use a new exported model and capture for each
experiment.

## Interpret and share results

- **First-logits latency** adds Tensor API prefix time to step-zero token time.
  LiteRT-LM's equivalent includes `Prefill` and the first `DecodeLogits` call.
- **Decode latency** is the mean of steps 1 through 64 within each session,
  reported as a median and range across sessions. Tokens/second is 1000 divided
  by the median session mean. Validation and full-vocabulary greedy argmax are
  included in both token timers; tokenization, sampling and text rendering are
  excluded.
- **Request latency** adds first-logits time and all forced decode steps within
  each session before taking the median. Setup and reset are excluded.
- **Peak RSS** is a process high-water mark including setup and warmup, not live
  KV bytes. Each configuration uses separate processes to avoid inheriting a
  larger case's high-water mark. The Markdown table shows the maximum process;
  JSON includes the range, median and paired ratio of process peaks.
- Ratios in JSON are LiteRT-LM divided by Tensor API. Positive reduction
  percentages mean lower Tensor API latency or RSS. Read the per-block results
  and thermal data before interpreting small differences as improvements.

The supplied fixture repeats an existing Wikitext-derived token stream after a
single BOS (token 2); its continuation is forced. This synthetic repeated-text
workload avoids sampling drift and makes the work identical. The original
tokenizer SHA-256 is
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`;
no tokenizer is needed at execution time. They are not a language-quality evaluation. The
adapter checks finite logits and the exact consumed-token history; summaries
also show argmax agreement without treating agreement as a quality guarantee.
Use the separate small-model tests and full-model accuracy evaluation for
correctness claims.

`summarize.py` rejects missing/duplicate steps, failed or partial captures,
changed fixtures, mismatched LM settings, and tracing-enabled runs. It includes
both source revisions, input identities, options and per-process measurements.
Share the summary JSON/Markdown and `workload.tsv` with the build commands.
Reports omit host paths, adb serials, Git authors, remotes and device fingerprints.
Raw `process.log` files and preparation manifests can contain paths; review them
separately before sharing. Do not commit generated runs, caches, weights or
binaries.

## Test the optional tooling

These tests need no model, device, NumPy or FlatBuffers:

```sh
cd "$LITERT_SRC"
python3 -m unittest discover \
  -s tensor/examples/experimental/gemma4/benchmarks -p '*_test.py' -v
```

The same suite is available as the Bazel target
`//tensor/examples/experimental/gemma4/benchmarks:benchmark_test`.

They cover fixture identity, shared prompt prefixes, alternating run order,
capacity skips, metric boundaries, malformed captures, and exclusion of private
paths and transport selectors. The C++ example's required regression tests remain
in the separate [unit-test targets](../REPRODUCING.md#unit-tests-without-a-checkpoint).

Validation when adding this package used official LiteRT-LM commit
`c0713a680878f57ca4e270525b9df5c60f2b6665` with its normal dependencies:
Linux and Android ARM64 adapter builds, an Android ARM64 Tensor API runner
build, host execution of the full 1024/64
fixture with warmup/reset, and a traced host regression run confirming only
`prefill_128` and `decode` when that option is selected. The tooling tests pass
under Bazel. These are compatibility checks, not Android performance results;
run the campaign on the target device to obtain current measurements.
