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

# Gemma4 LiteRT reproduction

## Unit tests without a checkpoint

From the repository root:

```sh
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" test -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //tensor/backends/tflite:tflite_flatbuffer_conversion_test \
  //tensor/examples/experimental/gemma4:bundle_loader_test \
  //tensor/examples/experimental/gemma4:graph_ops_test \
  //tensor/examples/experimental/gemma4:model_builder_test \
  //tensor/examples/experimental/gemma4:kv_bank_test
```

The tests cover:

- Explicit quantize/dequantize versus numeric casts, nonzero per-channel zero
  points, and scope lifetime across independently authored signatures.
- INT4/INT8 static FC scales, rounding and saturation; primitive RMSNorm; GeGLU;
  repeated dynamic RoPE resizing; full-matrix PLE requantization before splitting.
- A small six-layer model with local/global attention and two shared consumers.
  Chunked prefill is compared with token-by-token decode across partial chunks,
  sliding-window boundaries, reset, and four combinations of dynamic rows and
  explicit shape signatures. The test requests full CPU delegation and depends
  on the production authoring library, so test-only CI compiles it.
- Invalid tokens, cache capacity exhaustion, KV transactions, signature pruning,
  malformed manifests, architecture mismatches, owner ordering, and KV metadata.

These are numerical and contract tests on synthetic inputs. They do not establish
full-checkpoint accuracy, E4B graph-authoring support, or device performance.

## Optional published-bundle preparation

The Python helpers in `tools/` extract weights from an existing published
`.litertlm` bundle. They do not convert a source framework checkpoint. Keep this
material separate from the C++ unit tests when choosing what to upstream.
The complete helper dependency set is included: inventory, KV tracing and
summary, export, independent validation, model specifications, and the LiteRT-LM
header schema. The generated TFLite Python schema is created locally below.

Install NumPy and FlatBuffers in your chosen Python environment. Build:

```sh
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //tensor/examples/experimental/gemma4:export_model \
  //tensor/examples/experimental/gemma4:runner @flatbuffers//:flatc
```

Set `MODEL` to an existing published E2B `.litertlm` file, `SCHEMA_DIR` to a
schema output directory, and `PREPARED` to a new directory:

```sh
mkdir -p "$SCHEMA_DIR"
bazel-bin/external/flatbuffers/flatc --python -o "$SCHEMA_DIR" \
  tflite/converter/schema/schema.fbs
python3 tensor/examples/experimental/gemma4/tools/prepare_bundle.py \
  --model "$MODEL" --schema-dir "$SCHEMA_DIR" --output-dir "$PREPARED"
```

Run Python without `-O`: the helpers use assertions for source validation.
Preparation creates `audit/` and `weights/`, then validates exported tensor
codes, scales, constants, and hashes against the source. Review
`audit/export-validation.json` before export. The C++ loader validates layout
and metadata; it does not replace this source coefficient validation.
No model weights or generated model binaries are included in this repository.

New extractions use manifest schema 2, which validates the E2B/E4B architecture.
Legacy schema 1 requires the original E2B source identity. The C++ exporter
below only supports E2B. Although the preparation helpers accept
`--model-variant=e4b --compact-int2`, executing E4B also requires an externally
authored compatible model; that exporter is not part of this material.

## Export and smoke execution

Set `EXPORTED` to a new `.tflite` path and `RESULTS` to a new directory:

```sh
bazel-bin/tensor/examples/experimental/gemma4/export_model \
  --bundle_dir "$PREPARED/weights" --model_path "$EXPORTED"
bazel-bin/tensor/examples/experimental/gemma4/runner \
  --bundle_dir "$PREPARED/weights" --model_path "$EXPORTED" \
  --cases_file tensor/examples/experimental/gemma4/fixtures/smoke.tsv \
  --output_dir "$RESULTS" --cache_capacity 128 --num_threads 1 \
  --warmup_runs 0 --measured_runs 1 --dump_outputs
```

The exporter calls `AddGemma4Signatures`; it does not carry a second copy of the
model graph. `--prefill_chunk_rows` selects a power of two from 32 through 1024
(default 128). To exercise variable query rows and strict input resizing, supply
both `--experimental_dynamic_prefill` and
`--experimental_explicit_dynamic_shapes` to **both** exporter and runner. For
compact INT2 preparation, also pass `--experimental_compact_int2` to both.

Each TSV row contains a unique case ID, comma-separated prompt token IDs, and
comma-separated forced decode inputs (`-` means none). Step zero consumes the
last prompt token; prefill consumes the preceding tokens. The small fixtures
cover empty history and a short partial chunk, and deliberately use fixed token
IDs so a generated-token difference cannot change the later workload.
For a longer correctness run, replace `smoke.tsv` with `regressions.tsv` and
use `--cache_capacity 576`. It crosses 32-token alignment, 128-row prefill and
512-token sliding-window boundaries, advances the local-cache view, and ends
with a short request after a long one to exercise reset. Use a new results
directory for every run.

The runner writes logits and committed KV dumps plus `timings.jsonl`. Dumping is
for correctness checks. For latency runs omit `--dump_outputs`, select threads
and affinity explicitly, and use warmups and repeated measurements. Add prefix
time to the first decode step to obtain first-logits latency. Use identical
inputs, model coefficients, quantization mode, thread count and thermal state
when comparing runs. Record the source revision and all experimental flags.
The packed-weight cache defaults to `$EXPORTED.xnnpack_cache`; use a separate
cache path for a different model export.

## Optional current-runtime performance comparison

See [the benchmark package](benchmarks/README.md) for the official LiteRT-LM
CPU adapter, Android build and deployment commands, fixed workloads, prompt and
capacity sweeps, and validated summaries. It uses current official LiteRT-LM
dependencies and reports new measurements rather than requiring historical
performance numbers.
