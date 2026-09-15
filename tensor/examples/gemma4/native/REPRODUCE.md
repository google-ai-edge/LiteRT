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

# Compare native Gemma4 E2B with LiteRT-LM CPU

This guide builds the native Tensor API runner and compares it with LiteRT-LM
CPU using the same Gemma4 E2B coefficients, INT8 KV quantization, token histories,
and session capacity. It assumes familiarity with building LiteRT-LM,
connecting Android devices, and deploying files with adb.

The primary baseline is **LiteRT-LM CPU with XNNPACK**. YNNPACK is an optional,
separately labeled baseline. Use the supplied fixed-token adapter so both
runners measure the same phases. These workloads measure model execution;
tokenization, chat templates, sampling and EOS stopping are outside the test.

## Build the native runner

From the LiteRT repository root, with CMake 3.28+, Ninja and Clang:

```bash
cmake -S tensor/standalone -B .native-tensor-build/host -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build .native-tensor-build/host -j 3
ctest --test-dir .native-tensor-build/host --output-on-failure -j 1
```

The host tests use small synthetic inputs. For the Android benchmark, assume
`ANDROID_HOME` points to an SDK with NDK `27.3.13750724`:

```bash
cmake -S tensor/standalone -B .native-tensor-build/android -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_HOME/ndk/27.3.13750724/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
  -DANDROID_STL=c++_static
cmake --build .native-tensor-build/android --target gemma4_native_runner -j 3
```

The executable is `.native-tensor-build/android/bin/gemma4_native_runner`.
The build downloads its dependencies from pinned archives, including XNNPACK
`bf3ee43b63070284f85a4a882f298b6ca5273f01`, and uses `-ffp-contract=off` for
Tensor API C++ sources. This preserves the native benchmark's NDK, API level,
optimization level and dependency configuration. No external source checkout
or previously built archive is needed. See the
[standalone guide](../../../standalone/README.md) for other targets and Bazel.

## Prepare LiteRT-LM and the model

Build the supplied [LiteRT-LM comparison adapter](tools/litert_lm/README.md)
in your LiteRT-LM checkout with the documented runtime dependency and audit
patch. Its source calls the CPU executor directly, consumes the checked-in
TSV histories, and records inclusive first-logit and forced-decode timings.
The package instructions identify the required local changes and build target;
use your usual Android LiteRT-LM build setup. A normal text-generation CLI has
different input and timing semantics and does not reproduce this protocol.

The model is
[`litert-community/gemma-4-E2B-it-litert-lm`](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/616f4124e6ff216292f16e7f73ff33b5ba9a4dd4),
revision `616f4124e6ff216292f16e7f73ff33b5ba9a4dd4`, file
`gemma-4-E2B-it.litertlm`. Its size is 2,583,085,056 bytes and SHA256 is
`ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42`.

```bash
MODEL_DIR="$PWD/.native-tensor-build/model"
uvx hf download litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --revision 616f4124e6ff216292f16e7f73ff33b5ba9a4dd4 \
  --local-dir "$MODEL_DIR"
MODEL="$MODEL_DIR/gemma-4-E2B-it.litertlm"
sha256sum "$MODEL"
```

LiteRT-LM reads this file directly. For native execution, follow the
[export workflow](tools/README.md#export-workflow) to produce and validate a
`MATCHED_BUNDLE` directory from the same file. The tools require Python with
NumPy and FlatBuffers, plus `flatc` to generate TFLite bindings from this
checkout. The earlier export used `flatc` 25.9.23, NumPy 2.4.2 and FlatBuffers
Python 25.12.19. The matching `.litertlm` container parser is included.
Reuse an existing verified export of this exact bundle when available.

Deploy both executables, the original model, the validated native export, and
[fixtures](fixtures) to the chosen phone. The native export's `manifest.json`
must be directly inside its bundle directory. Preserve model and executable
hashes with the results. No GGUF or raw compressed-tensors checkpoint is used
in this comparison.

## Match the execution settings

| Setting | Native runner | LiteRT-LM XNNPACK baseline |
| --- | --- | --- |
| Model weights | Export of the pinned bundle, including original compact INT2 | The same pinned `.litertlm` bundle |
| KV quantization | Published INT8 codes/scales in token-major owner buffers | Published INT8 codes/scales in the model's layout |
| Allocated capacity | 8448 | 8448, applied through executor settings and the model shape rewrite |
| CPU threads | 4 | 4 |
| Prompt lengths | 128, 1024, 4096, including BOS | Identical raw IDs |
| Continuation | 64 fixed input IDs | Identical 64 IDs |
| Prefill | Reusable 128-row prefix stages; final prompt token uses decode | Model's 128/1024 signatures and normal chunk selection; initial `DecodeLogits` completes the prompt |
| Repetitions | One warmup, three measured sessions, reused runtimes | One warmup, three measured sessions, reused executor |
| Preparation | Shared packed weights, worker pool and workspace | Engine CPU settings, memory cache and normal executor resources |
| Diagnostics during timing | Logit/cache dumps, traces and memory reporting disabled | Logit dumps and profiling disabled |

Capacity is a session capability, not the prompt length. Both runners support
8448 tokens in these measurements, while the native runner restricts attention
to active prefixes/windows. Setting each runtime to a different capacity would
change the comparison. The LiteRT-LM adapter applies the model's `32003 → 8448`
environment rewrite; `SetMaxNumTokens` by itself is insufficient. Its separate
full-logit smoke mode also verifies all 30 actual bound INT8 cache buffers.

Use the same CPU affinity on each phone for both runners. Earlier four-thread
captures used `f0` on TECNO LJ9 and `1e0` on Pixel 8. Select an appropriate mask
for other devices. Run one benchmark process per phone, record thermal/load
conditions, and keep normal device scheduling and thermal controls enabled.

## Run 128, 1024 and 4096 tokens

Set these variables to the deployed files and an available device. All device
paths below should be absolute and contain no shell-special characters.
`DEVICE_RESULTS` is a parent directory; each individual run directory must be
new. The local results layout is consumed by the summarizer below.

```bash
set -euo pipefail
ADB="$ANDROID_HOME/platform-tools/adb"
SERIAL=DEVICE_SERIAL
CPU_MASK=DEVICE_CPU_MASK
DEVICE_NATIVE=/data/local/tmp/gemma4/native-runner
DEVICE_LM=/data/local/tmp/gemma4/litert-lm-adapter
DEVICE_MODEL=/data/local/tmp/gemma4/gemma-4-E2B-it.litertlm
DEVICE_BUNDLE=/data/local/tmp/gemma4/matched-bundle
DEVICE_FIXTURES=/data/local/tmp/gemma4/fixtures
DEVICE_RESULTS=/data/local/tmp/gemma4/results
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RESULTS="$PWD/.native-tensor-build/comparison-$RUN_ID"
mkdir -p "$RESULTS/native" "$RESULTS/litert-lm"
"$ADB" -s "$SERIAL" shell mkdir -p "$DEVICE_RESULTS"

for LENGTH in 128 1024 4096; do
  ORDER="native litert-lm"
  if [ "$LENGTH" = 1024 ]; then ORDER="litert-lm native"; fi
  for BACKEND in $ORDER; do
    OUTPUT="$DEVICE_RESULTS/$RUN_ID-$BACKEND-$LENGTH"
    CASES="$DEVICE_FIXTURES/performance_${LENGTH}_64.tsv"
    if [ "$BACKEND" = native ]; then
      "$ADB" -s "$SERIAL" shell \
        "taskset $CPU_MASK $DEVICE_NATIVE \
          --bundle_dir=$DEVICE_BUNDLE --cases_file=$CASES --output_dir=$OUTPUT \
          --num_threads=4 --weight_cache=true --cache_capacity=8448 \
          --prefill_chunk_rows=128 --kv_alignment=32 \
          --preserve_static_int2=true --share_workspace=true \
          --reuse_runtimes=true --warmup_runs=1 --measured_runs=3 \
          --memory_report=false --fixed_attention_extent=false \
          --dump_full_logits=false --dump_cache=false --trace_position=-1"
    else
      "$ADB" -s "$SERIAL" shell \
        "taskset $CPU_MASK $DEVICE_LM \
          --model_path $DEVICE_MODEL --cases_file $CASES --output_dir $OUTPUT \
          --num_threads 4 --max_num_tokens 8448 \
          --reuse_runtimes true --warmup_runs 1 --measured_runs 3 \
          --dump_full_logits false --enable_ynnpack false --enable_profiling false"
    fi
    "$ADB" -s "$SERIAL" pull "$OUTPUT" "$RESULTS/$BACKEND/$LENGTH"
    sleep 30
  done
done
```

The pause is a minimum cooldown, not proof of matched temperature. Repeat with
reversed backend order when results vary materially, and retain every complete
session. A separate phone can run its own campaign concurrently. Avoid running
these performance commands on a busy development host.

**Pass `--enable_ynnpack false` explicitly.** The supplied LiteRT-LM adapter's
default is true. To measure YNNPACK separately, build it into LiteRT-LM as
explained in the adapter guide, change that flag to true, and use a separate
results root. YNNPACK uses different attention arithmetic in the recorded
runtime; keep its speed and numerical validation separate from XNNPACK.

## Summarize and interpret the results

After pulling the completed captures, run the offline summarizer:

```bash
python3 tensor/examples/gemma4/native/tools/summarize_performance.py \
  "$RESULTS" --output "$RESULTS/summary.json"
```

Each measured session records one prefill pass and 64 decode passes. Prefill
seconds are `passes[0].elapsed_ms / 1000`. Decode tokens/s are
`1000 * 64 / sum(pass.elapsed_ms for pass in passes[1:])`. Compute these per
session, then take the median of the three sessions. Keep the individual
samples to expose variance; do not average instantaneous token rates.

Prefill includes session reset or fresh native KV allocation/zeroing, the
entire prompt, the first full-vocabulary prediction, and its finite/argmax
checks. LiteRT-LM reaches that point with `Prefill` plus its first empty-input
`DecodeLogits`; native reserves the final prompt token for decode. The next
64 input tokens are teacher-forced in both runners. Input preparation,
synchronous execution, output access and finite/argmax checks are included.
Static model loading, runtime creation and static weight packing are outside
these intervals; dynamic KV/BMM preparation remains inside timed execution.
Output serialization and diagnostic dumps are outside the timers. Use
`elapsed_ms` consistently rather than substituting narrower stage timings.

A native speedup at fixed capacity includes graph and cache-layout effects,
including active attention extents and less padded attention work. Compact
weights and runtime reuse are enabled in both runners and are not uniquely
native features. The comparison also uses different executor integrations and
compiler/dependency configurations. These timings alone cannot assign a percentage of
the difference to any one optimization. Lowering the capacity to fit one
prompt is a separate experiment and must change both runners consistently.

For a correctness check, run `capacity_smoke_8_1.tsv` and
`native_boundary_4096_2.tsv` separately with zero warmups, one measured session,
and full-logit dumps enabled. Compare all vocabulary entries using
[compare_live.py](tools/compare_live.py) with `--no-cache --require-bitwise`.
Keep those diagnostic captures outside the performance results directories.

## Earlier reference measurements

These medians were recorded before this branch was created, using the protocol
above and the earlier runtime revisions described in the adapter guide. They
are comparison points, not new measurements of this branch or fixed acceptance
thresholds. The native implementation and pinned standalone configuration are
preserved; a newer LiteRT-LM build must be identified separately in results.

| Phone | Prompt | LiteRT-LM XNNPACK first logits, s | Native first logits, s | LiteRT-LM XNNPACK decode, tokens/s | Native decode, tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| TECNO LJ9 | 128 | 1.233 | 0.377 | 13.45 | 27.93 |
| TECNO LJ9 | 1024 | 9.972 | 3.344 | 13.26 | 23.86 |
| TECNO LJ9 | 4096 | 42.258 | 17.329 | 11.29 | 18.51 |
| Pixel 8 | 128 | 1.451 | 0.573 | 12.30 | 22.88 |
| Pixel 8 | 1024 | 13.402 | 4.118 | 12.09 | 22.14 |
| Pixel 8 | 4096 | 65.506 | 20.421 | 9.37 | 13.47 |

Three sessions have limited precision. For example, the recorded Pixel 8
4096-token native decode rates ranged from 12.47 to 16.66 tokens/s. Preserve
that spread when comparing a new measurement, and investigate configuration,
temperature and competing work before attributing a change to the source.
