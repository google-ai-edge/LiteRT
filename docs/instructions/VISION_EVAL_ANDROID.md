<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# Evaluate vision models on Android

Build `litert_vision_eval` for ARM64 and select CPU, MLDrift GPU, or a vendor NPU
with the same options used on the host. The
[evaluator guide](../../litert/tools/vision_eval/README.md) describes dataset
preparation, protobuf configs, dense tasks, baselines, and resume behavior.
The commands below assume you already use adb and select a device with `-s` when
several phones are connected.

## Build

From the repository root, select your installed SDK and NDK:

```sh
mkdir -p .cache .bazelisk-cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  --repo_env=ANDROID_SDK_HOME=/path/to/android_sdk \
  --repo_env=ANDROID_NDK_HOME=/path/to/android_sdk/ndk/28.2.13676358 \
  --repo_env=ANDROID_NDK_VERSION=28 \
  --repo_env=ANDROID_SDK_API_LEVEL=35 \
  --repo_env=ANDROID_NDK_API_LEVEL=35 \
  --repo_env=ANDROID_BUILD_TOOLS_VERSION=35.0.0 \
  --config=android_arm64 \
  //litert/tools:litert_vision_eval
```

The normal target includes MLDrift's OpenCL/OpenGL accelerator dependencies.
For a pinned runtime, build `//litert/tools:litert_vision_eval_dynamic_runtime`
with `--define=litert_runtime_link_mode=dynamic`, and deploy its compatible
`libLiteRt.so` and accelerator libraries together. The dynamic executable allows
`LD_LIBRARY_PATH` to override its bundled runtime.

## Classification on CPU and GPU

After following the evaluator guide to prepare three ImageNet samples and a
model, deploy them under one directory:

```sh
adb shell mkdir -p /data/local/tmp/vision-eval
adb push bazel-bin/litert/tools/litert_vision_eval /data/local/tmp/vision-eval/
adb push eval-data/imagenet /data/local/tmp/vision-eval/
adb push eval-data/models/resnet18.tflite /data/local/tmp/vision-eval/model.tflite
adb shell chmod +x /data/local/tmp/vision-eval/litert_vision_eval
```

Use checkpoint-appropriate preprocessing. For the ResNet18 example:

```sh
adb shell 'cd /data/local/tmp/vision-eval && ./litert_vision_eval \
  --model=model.tflite --manifest=imagenet/manifest.tsv \
  --output_dir=cpu-results --accelerator=cpu --num_threads=4 \
  --xnnpack_only=true --resize=256 --max_samples=3'
```

Deploy the build's matching `libLiteRtClGlAccelerator.so`, then select MLDrift:

```sh
adb shell 'cd /data/local/tmp/vision-eval && LD_LIBRARY_PATH=$PWD \
  ./litert_vision_eval --model=model.tflite --manifest=imagenet/manifest.tsv \
  --output_dir=gpu-results --accelerator=gpu --gpu_backend=opencl \
  --gpu_precision=fp32 --gpu_use_ir_model=true \
  --require_full_acceleration=true --resize=256 --max_samples=3 \
  --baseline_model=model.tflite --baseline_accelerator=cpu'
```

Full acceleration is a check on the model/device combination. The summary records
actual delegation; a partial GPU or NPU execution fails when
`require_full_acceleration` is set. Numerical parity gates and classification
prediction agreement are separately selectable.

## Vendor NPU

Deploy the vendor's compatible dispatch library and a model compiled for the
phone, or the matching compiler plugin for supported on-device compilation:

```sh
adb shell 'cd /data/local/tmp/vision-eval && LD_LIBRARY_PATH=$PWD \
  ./litert_vision_eval --model=model_npu.tflite --accelerator=npu \
  --dispatch_library_dir=$PWD --compiler_plugin_library_dir=$PWD \
  --manifest=imagenet/manifest.tsv --output_dir=npu-results \
  --resize=256 --max_samples=3 --require_full_acceleration=true \
  --baseline_model=model.tflite --baseline_accelerator=cpu'
```

For the Arm Vulkan ML plugin, enable JIT explicitly with
`--arm_vulkan_ml_enable_jit=true` when using its compiler and dispatch libraries.
That option requires the selected accelerators to include `npu`. Other vendors
can use the same evaluation and scoring paths without it.

Prepared dense-task datasets use protobuf manifests and raw files. Deploy the
prepared directory without changing the relative paths and pass its manifest.
A config protobuf may replace the scalar flags in any example.

Pull the result directory to inspect `summary.txtpb` and the binary sample
checkpoints. Repeat the identical command to resume on the same device; use a new
output directory when changing models, backend settings, or binaries.
