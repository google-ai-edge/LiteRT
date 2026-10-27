# Evaluate an ImageNet model with LiteRT

Use `imagenet_eval_runner` to measure top-1/top-5 accuracy or compare a model
against a CPU reference on identical images. It supports FP32, INT8 and UINT8
classification I/O, NCHW/NHWC inputs, and CPU, MLDrift GPU and vendor NPU
backends. Image decoding, resize, center crop and normalization run in C++.

The examples below start with **three images**. Run host commands from the
LiteRT repository root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Bazelisk first.

## 1. Download ImageNet validation data

Accept the access conditions on the
[ILSVRC/imagenet-1k dataset page](https://huggingface.co/datasets/ILSVRC/imagenet-1k),
then authenticate with an account that has access:

```sh
uvx hf auth login
```

Download one validation shard for the smoke test. A shard contains many images
and can still be hundreds of megabytes; `--max_samples=3` later limits evaluation,
not the download size.

```sh
uvx hf download ILSVRC/imagenet-1k \
  data/validation-00000-of-00014.parquet \
  --repo-type dataset \
  --revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5 \
  --local-dir eval-data/hf-imagenet
```

For the full validation split, replace that command with:

```sh
uvx hf download ILSVRC/imagenet-1k \
  --repo-type dataset \
  --revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5 \
  --include 'data/validation-*.parquet' \
  --local-dir eval-data/hf-imagenet
```

The filter excludes training and test shards. Add `--dry-run` to inspect the
files and download sizes first. The revision pins the dataset layout used in
these examples. See the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
for download and authentication options.

## 2. Extract JPEGs and create the manifest

The evaluator reads image files rather than Parquet. This command extracts the
first three images from the downloaded shards and retains their class labels.
It copies the original encoded bytes without decoding or resizing them:

```sh
uv run --no-project --python 3.13 --with pyarrow python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

shards = sorted(Path("eval-data/hf-imagenet/data").glob("validation-*.parquet"))
if not shards:
    raise SystemExit("No validation shards found; download them first")
output = Path("eval-data/imagenet")
(output / "images").mkdir(parents=True, exist_ok=True)
limit = 3  # Set to 0 to extract every row in all downloaded validation shards.
count = 0
with (output / "manifest.tsv").open("w") as manifest:
    manifest.write("id\tlabel\tpath\n")
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(
                batch_size=32, columns=["image", "label"]):
            for row in batch.to_pylist():
                encoded = row["image"]["bytes"]
                label = int(row["label"])
                if not encoded or not 0 <= label < 1000:
                    raise ValueError("Expected encoded ImageNet image and label 0..999")
                relative = f"images/{count:08d}.JPEG"
                (output / relative).write_bytes(encoded)
                manifest.write(f"{count}\t{label}\t{relative}\n")
                count += 1
                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break
        if limit and count >= limit:
            break
print(f"Wrote {count} images and {output / 'manifest.tsv'}")
PY
```

For existing JPEGs, create a TSV with the same three columns: unique sample ID,
zero-based ImageNet class label, and image path relative to `--image_root`.
The optional header must be exactly `id<TAB>label<TAB>path`. Use validation
labels; unlabeled test images cannot produce an accuracy result.

## 3. Download a model and specify preprocessing

This example uses a pinned ResNet18 export with both FP32 and int8 variants:

```sh
uvx hf download litert-community/resnet18 \
  resnet18.tflite resnet18_int8_channelwise.tflite \
  --revision 7eabef429f15b4fcd4db3e225d81bd0fd7860c4f \
  --local-dir eval-data/models

cat > eval-data/resnet18.json <<'JSON'
{
  "profile": "torchvision", "interpolation": "bilinear", "antialias": true,
  "resize": 256, "crop_h": 224, "crop_w": 224,
  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
}
JSON
```

Use the preprocessing associated with **your checkpoint**. Other models may
require a different resize, interpolation, crop or normalization. The evaluator
requires one batch-one, three-channel image input and one classification vector;
the output class ordering must match the manifest labels.

## 4. Build and run on CPU

```sh
mkdir -p .cache .bazelisk-cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //litert/tools:imagenet_eval_runner

bazel-bin/litert/tools/imagenet_eval_runner \
  --model=eval-data/models/resnet18.tflite \
  --accelerator=cpu --num_threads=4 --xnnpack_only=1 \
  --manifest=eval-data/imagenet/manifest.tsv --image_root=eval-data/imagenet \
  --preprocessing_config=eval-data/resnet18.json --max_samples=3 \
  --out=eval-data/cpu.jsonl --summary_json=eval-data/cpu.json
```

To compare against CPU reference kernels in the same run, add:

```sh
  --baseline_model=eval-data/models/resnet18.tflite \
  --baseline_accelerator=cpu --baseline_num_threads=1 \
  --require_output_parity=1 --atol=1e-4 --rtol=1e-3
```

The baseline uses reference kernels without CPU delegates. Omitting
`--baseline_accelerator=cpu` preserves the default `inherit` behavior: both
models use equivalent backend settings. A different `--baseline_model` can
compare two model artifacts with compatible input/output shapes.

## 5. Run on an Android CPU or MLDrift GPU

Build the Android ARM64 evaluator using the SDK/NDK instructions in the
[Android guide](../../g3doc/instructions/IMAGENET_ANDROID_EVAL.md). The normal
Android target links MLDrift's OpenCL/OpenGL accelerator. Copy the Android
executable to `eval-data/imagenet_eval_runner_android` before switching back to a
host build, since Bazel's `bazel-bin` symlink changes with build configuration.

Select the phone explicitly and push only the prepared fixture:

```sh
ANDROID_SERIAL=YOUR_ADB_SERIAL
adb -s "$ANDROID_SERIAL" shell mkdir -p /data/local/tmp/litert-imagenet
adb -s "$ANDROID_SERIAL" push eval-data/imagenet_eval_runner_android \
  /data/local/tmp/litert-imagenet/imagenet_eval_runner
adb -s "$ANDROID_SERIAL" push eval-data/models/resnet18.tflite \
  eval-data/resnet18.json /data/local/tmp/litert-imagenet/
adb -s "$ANDROID_SERIAL" push eval-data/imagenet \
  /data/local/tmp/litert-imagenet/
adb -s "$ANDROID_SERIAL" shell chmod +x /data/local/tmp/litert-imagenet/imagenet_eval_runner

adb -s "$ANDROID_SERIAL" shell 'cd /data/local/tmp/litert-imagenet && \
  ./imagenet_eval_runner \
    --model=resnet18.tflite --baseline_model=resnet18.tflite \
    --baseline_accelerator=cpu --baseline_num_threads=1 \
    --accelerator=gpu --gpu_backend=opencl --gpu_use_ir_model=true \
    --gpu_precision=fp32 --require_full_acceleration=1 \
    --require_output_parity=1 --atol=1e-4 --rtol=1e-3 \
    --manifest=imagenet/manifest.tsv --image_root=imagenet \
    --preprocessing_config=resnet18.json --max_samples=3 \
    --out=gpu.jsonl --summary_json=gpu.json'

adb -s "$ANDROID_SERIAL" pull /data/local/tmp/litert-imagenet/gpu.json eval-data/
adb -s "$ANDROID_SERIAL" pull /data/local/tmp/litert-imagenet/gpu.jsonl eval-data/
```

For CPU evaluation on the phone, replace the GPU settings with
`--accelerator=cpu --num_threads=4 --xnnpack_only=1` and use different output
filenames. Keep the CPU-baseline options if you want reference comparison.
For dynamic-runtime deployment, see the Android guide; deploy ABI-compatible
runtime and accelerator libraries and set `LD_LIBRARY_PATH`.

## 6. Select Arm Vulkan ML

Arm Vulkan ML uses LiteRT's **NPU plugin interface**, not the MLDrift GPU path.
Build its compiler and dispatch libraries, then run on a supported Vulkan ML
device or an explicitly configured host emulator:

```sh
bazel-bin/litert/tools/imagenet_eval_runner \
  --model=eval-data/models/resnet18_int8_channelwise.tflite \
  --baseline_model=eval-data/models/resnet18_int8_channelwise.tflite \
  --baseline_accelerator=cpu --baseline_num_threads=1 \
  --accelerator=npu --arm_vulkan_ml_enable_jit=1 \
  --compiler_plugin_library_dir=/path/to/compiler-library-directory \
  --dispatch_library_dir=/path/to/dispatch-library-directory \
  --require_full_acceleration=1 \
  --manifest=eval-data/imagenet/manifest.tsv --image_root=eval-data/imagenet \
  --preprocessing_config=eval-data/resnet18.json --max_samples=3 \
  --out=eval-data/arm.jsonl --summary_json=eval-data/arm.json
```

Follow the [backend setup](../vendors/arm_vulkan_ml/README.md) for device
requirements and emulation. Merely having an Arm CPU or Mali GPU does not prove
support. The evaluator does not enable emulation or CPU fallback automatically.

## 7. Interpret results and expand the sample count

Require **exit code zero**, `"status": "passed"` and the expected `"samples"`
count. The JSON summary includes:

- `candidate_accuracy` and `candidate_top5`: fractions in `[0, 1]`.
- For paired runs: baseline accuracy, prediction agreement and score errors.
- Candidate/baseline execution settings and delegation metrics.
- Model, executable, input and loaded-library hashes; device properties on Android.

`--require_full_acceleration=1` prevents a partially delegated GPU/NPU candidate
from being mistaken for full backend execution. An explicit CPU-reference
baseline is exempt. Preserve logs to confirm MLDrift/OpenCL initialization or
vendor dispatch loading; after Arm JIT, one runtime DISPATCH op can represent
many original operators.

`--require_output_parity=1` enforces the configured score tolerance, not matching
top-1 predictions. Quantized outputs are compared after dequantization. Choose
quantized-model tolerances explicitly: four output codes correspond to an
absolute tolerance of `4 * output_scale` with `--rtol=0`. For the pinned int8
ResNet18 above, this is `--atol=0.6821842193603516`. Do not reuse that numeric
value for a different artifact.

Three images are a smoke test, not a meaningful ImageNet accuracy estimate.
For a larger evaluation, extract more images and set `--max_samples=N`.
`--start_sample=N` skips that many manifest rows; `--max_samples=0` evaluates all
remaining rows. Full validation requires downloading **all validation shards**,
extracting with `limit = 0`, and selecting the entire resulting manifest.

## Dynamic runtime builds

To load a separately built LiteRT runtime, build the dynamic target:

```sh
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  --define=litert_runtime_link_mode=dynamic \
  //litert/tools:imagenet_eval_runner_dynamic_runtime
```

Use `imagenet_eval_runner_dynamic_runtime` in place of the normal executable.
The dynamic target requires the runtime-link-mode define. To compare with a
Python runtime, set `LD_LIBRARY_PATH` to that wheel's library directory as well
as passing `--runtime_library_dir` and `--dispatch_library_dir`. The directory
options alone do not choose the ELF loader's core runtime. Headers and runtime
must be ABI compatible. The Linux/Android summary records the actual mapped
LiteRT library paths and SHA-256 hashes.

The Linux dynamic target uses `DT_RUNPATH` so `LD_LIBRARY_PATH` can override
Bazel's bundled runtime even when launching directly from `bazel-bin`.

For CPU settings, the evaluator emits the current `cpu_options` payload plus
the older `xnnpack` and `cpu_delegate` aliases. Older runtime builds otherwise
silently ignore a newer SDK's thread/kernel settings. Each compiled model owns
independent payloads. The summary records these identifiers and, on Linux, the
process thread count after compilation; this is diagnostic evidence, not a
claim that every process thread runs inference.

## Preprocessing details

Quantized I/O requires finite, positive per-tensor scales. Inputs are rounded
and saturated; output scores are dequantized before comparison. Each image is
decoded and preprocessed once, and both models receive the same normalized
input tensor, converted to each model's input type.

Alternatively use `--interpolation=bilinear|bicubic` with the resize, crop,
antialias, mean and std flags. A preprocessing JSON file overrides these flags.

The implementation truncates the resized long side, uses ties-to-even crop
offsets, supports center-crop zero padding, and performs separable bilinear or
bicubic interpolation with uint8 rounding between passes. Antialiasing widens
the downsampling filter. Keep it enabled for PIL checkpoint conventions.
STB JPEG decoding and resize rounding can differ from Pillow/libjpeg; this
implementation does not promise byte-identical tensors to Pillow.

The optional JSON `profile` field must be `torchvision`. Input hashes identify tensors actually used by both models;
they do not imply equality to historical Pillow-based runs or bitwise inference
determinism across runtimes and hardware.

## Output files and diagnostics

JSONL, `--inputs_out` and `--scores_out` outputs remain available.
Raw outputs contain candidate FP32 scores and normalized FP32 inputs in selected
row order. Paired JSONL adds baseline top-k and per-row score errors. The summary
records model/binary/input hashes, runtime settings, completed sample count,
labels/predictions, top-1/top-5 deltas, prediction agreement and score errors.
Ties use ascending class index. Default comparison is
`abs(candidate-baseline) <= 1e-4 + 1e-3*abs(baseline)`.

`--compiler_plugin_library_dir` sets the compiler search directory for vendor
plugins. `--arm_vulkan_ml_enable_jit` defaults to zero and requires `npu` in the
candidate accelerator selection. It sets the Arm compiler option only when
explicitly requested. Runtime and plugin binaries must be ABI compatible.

`--baseline_accelerator=inherit` is the default and preserves the previous
behavior: both models use equivalent execution settings, with separately owned
options. `--baseline_accelerator=cpu` instead selects CPU reference kernels,
disables CPU delegates and clears vendor/GPU settings for the baseline. Its
thread count defaults to one. `--baseline_model` is required; it may be the same
path as `--model`. The core runtime remains shared by the process.

With an explicit CPU baseline, `--require_full_acceleration=1` checks the
candidate only. With an inherited baseline it checks both models, as before.
The summary adds `candidate_execution`, `baseline_execution`,
`candidate_delegation` and `baseline_delegation`. Delegation counts describe the
runtime graph; after Arm JIT, a whole source graph may be represented by one
DISPATCH op. They must not be interpreted as source operator counts.

The specialized Arm qualification harness also checks integer-code mean error
and model-instance lifetime behavior. These checks are separate from the shared
evaluator's score-parity gate.

For host emulation, configure Arm's emulation layers and Lavapipe explicitly,
then set `LITERT_ARM_VULKAN_ML_ALLOW_EMULATION=1`. The evaluator never enables
emulation or CPU fallback automatically. Native devices must advertise the
backend's required Vulkan features and instruction-set profiles.

Linux/Android library provenance includes dynamically loaded compiler and
dispatch libraries. Android summaries also contain model, serial, OS version
and SoC properties in `android_device`. A statically linked accelerator is
identified by the evaluator binary hash and registration logs, not a fabricated
shared-library path. Retain initialization logs to verify the actual GPU API.
