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

# LiteRT-LM CPU comparison adapter

[adapter.cc](adapter.cc) drives LiteRT-LM's CPU `LlmExecutor` with the same
raw-token fixtures as the native runner. It preserves the source used for the
September 15, 2026 XNNPACK/YNNPACK comparison, including first-token accounting,
runtime reuse and output checks. Its SHA256 is
`e85be79b91dd9048849ec386207ccd0d72b8f847a569c7e4e85ab6512d04bafb`.
The standard LiteRT-LM CLI does not implement this exact forced-token protocol.
Use the [comparison recipe](../../REPRODUCE.md) for the model, fixtures, native
build and device commands.

## Source and dependency contract

Copy this package into a separate LiteRT-LM checkout at
`2d044a3376ebd063752661d8da2704ec5a77c32f`. Its dependencies need both supplied
patches:

| File | Purpose |
| --- | --- |
| [BUILD.litert_lm](BUILD.litert_lm) | Becomes `local_gemma4_benchmark/BUILD` inside LiteRT-LM. It is deliberately not a LiteRT Bazel package. |
| [litert_lm_dependency.patch](litert_lm_dependency.patch) | The previously built compatibility update to LiteRT-LM's `WORKSPACE` and `.bazelrc`: TensorFlow `5c0b7a5946f0f485e3a532b2a00e03f42a6e14c1`, matching repository initialization and dependency aliases. |
| [litert_lm_cache_audit.patch](litert_lm_cache_audit.patch) | Adds the const `LitertState::PrimaryBankForLocalAudit()` accessor needed to compile the optional cache-shape audit. It reads a cloned cache after correctness timers; performance runs disable that audit. |

The build overrides `@litert` with this LiteRT checkout. Its upstream base is
`adae5c349ccfa928b7c4c32ceff4592d4b0b28b1`, which already contains the complete
LiteRT PR #9918 dependency update. No additional local LiteRT delegate,
compiler or XNNPACK patch is required. Overriding `@litert` alone is insufficient:
the outer LiteRT-LM `WORKSPACE` still selects TensorFlow and shared dependencies,
which is why its separate compatibility patch is included.

That TensorFlow pin selects XNNPACK
`d89ef6669a14db203b3b7935b1b3862cb63fb6df` (including its YNNPACK sources) and
KleidiAI `dce86647385ab2638aa5abebcb652f3e4271970d`. No XNNPACK repository
override is part of this baseline. These pins differ from the native
standalone dependency set.

The historical capture used LiteRT
`08cc9adcade8da5d996e3f68e184a5498482fa19` with PR #9918 applied. This recipe
rebuilds against the newer LiteRT source in this branch; intervening runtime
changes mean it is a new baseline build, not a byte-for-byte recreation of the
old executable. Record both Git revisions, local changes and executable hashes
with new results. The patches preserve the prior benchmark's CPU executor
implementation; they do not change inference arithmetic.

## Copy and build

Set `LITERT_SRC` to this LiteRT checkout and `LM_SRC` to a fresh, isolated
LiteRT-LM checkout at the revision above. Start from the LiteRT repository root:

```bash
set -euo pipefail
LITERT_SRC="$PWD"
LM_SRC=/absolute/path/to/isolated/LiteRT-LM
LM_TOOLS="$LITERT_SRC/tensor/examples/gemma4/native/tools/litert_lm"
test "$(git -C "$LM_SRC" rev-parse HEAD)" = \
  2d044a3376ebd063752661d8da2704ec5a77c32f
git -C "$LM_SRC" apply --check "$LM_TOOLS/litert_lm_dependency.patch"
git -C "$LM_SRC" apply --check "$LM_TOOLS/litert_lm_cache_audit.patch"
git -C "$LM_SRC" apply "$LM_TOOLS/litert_lm_dependency.patch"
git -C "$LM_SRC" apply "$LM_TOOLS/litert_lm_cache_audit.patch"
mkdir "$LM_SRC/local_gemma4_benchmark"
cp "$LM_TOOLS/adapter.cc" "$LM_SRC/local_gemma4_benchmark/adapter.cc"
cp "$LM_TOOLS/BUILD.litert_lm" "$LM_SRC/local_gemma4_benchmark/BUILD"
```

The Android command retains the previous baseline's NDK `28.2.13676358`, API
31, SDK 35 and build-tools 35.0.0 configuration. It builds both CPU delegates
into one executable, allowing the runtime flag to select the baseline. These
toolchain settings differ from the native standalone recipe; record them when
interpreting the comparison.

```bash
cd "$LM_SRC"
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
bazelisk --output_base="$PWD/.bazel-output" build -c opt --jobs=4 \
  --config=android_arm64 \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  --repo_env=ANDROID_SDK_HOME="$ANDROID_HOME" \
  --repo_env=ANDROID_NDK_HOME="$ANDROID_HOME/ndk/28.2.13676358" \
  --repo_env=ANDROID_NDK_VERSION=28 \
  --repo_env=ANDROID_SDK_API_LEVEL=35 \
  --repo_env=ANDROID_NDK_API_LEVEL=31 \
  --repo_env=ANDROID_BUILD_TOOLS_VERSION=35.0.0 \
  --override_repository=litert="$LITERT_SRC" \
  --define=litert_runtime_link_mode=static \
  --define=litert_link_capi_so=false \
  --define=DISABLE_HUGGINGFACE_TOKENIZER=1 \
  --define=DISABLE_SENTENCEPIECE_TOKENIZER=1 \
  --define=litert_enable_ynnpack=true \
  //local_gemma4_benchmark:adapter
LM_ADAPTER="$LM_SRC/bazel-bin/local_gemma4_benchmark/adapter"
sha256sum "$LM_ADAPTER"
cd "$LITERT_SRC"
```

The adapter uses C++20, as configured by this LiteRT-LM revision's Android
build. Disabling the tokenizers is appropriate because all inputs are token
IDs. No separate `libLiteRt.so` is deployed with this static CPU executable.

## Runtime settings and results

With the executable, original `.litertlm` model and fixture already on the
device, the XNNPACK control uses this command in the device shell. Set the
variables to device paths; `OUTPUT_DIR` must not exist yet.

```bash
taskset "$CPU_MASK" "$LM_ADAPTER" \
  --model_path "$MODEL" --cases_file "$CASES_FILE" \
  --output_dir "$OUTPUT_DIR" --num_threads 4 --max_num_tokens 8448 \
  --reuse_runtimes true --warmup_runs 1 --measured_runs 3 \
  --dump_full_logits false --enable_ynnpack false --enable_profiling false
```

Use `--enable_ynnpack=true` in a separate output directory for the optional
YNNPACK run. The adapter defaults to YNNPACK enabled and logit dumps enabled,
so both options must be explicit for the XNNPACK performance baseline. Keep
delegate logs alongside results to confirm actual delegation; a requested
option alone is not evidence that all model operators were delegated.

Engine metadata settings are applied before executor creation. The driver
verifies Gemma4's disabled delegate clustering and the `32003 -> 8448`
environment rewrite for context dimensions. KV dtype remains the published
bundle's INT8. Prefill signatures are selected by LiteRT-LM itself: the saved
128-, 1024- and 4096-token cases used one 128-row, one 1024-row and four 1024-row
prefill calls respectively, with the last prompt token handled separately.
The configured capacity does not force a 8448-row prefill signature.

Pass zero includes `Prefill()` followed by empty-input `DecodeLogits()` for
the pending last prompt token and first complete-vocabulary prediction.
The following 64 passes each consume one forced token. A reused executor's
`Reset()` time is included in prefill `elapsed_ms`; executor creation is
reported separately. Input-buffer creation, output locking, argmax and finite
checks are timed. History verification, logit dumps and JSON output are outside
these intervals. Warmup and measured runs use the same executor with resets.

Each measured case produces `CASE.run_000.json` through `CASE.run_002.json`;
warmup files are separate. `manifest.json` is updated during progress and
`run.json` is written at successful completion. Check `status=completed` and
`timings_valid_for_benchmark=true`. Compute prefill seconds from
`passes[0].elapsed_ms / 1000`, and decode tokens/s from
`64000 / sum(pass.elapsed_ms for pass in passes[1:])`, separately per repetition
before taking medians. Use `elapsed_ms` for both runners; their `forward_ms`
fields cover different internal boundaries.

`--enable_profiling=true` is for diagnostic captures and marks timings invalid
for benchmarking. `--dump_full_logits=true` saves complete logit rows for the
first measured repetition and audits the first case's cloned cache afterward.
The cache audit checks 30 actual INT8 buffers at the requested capacity; normal
performance captures do not clone cache contents.
