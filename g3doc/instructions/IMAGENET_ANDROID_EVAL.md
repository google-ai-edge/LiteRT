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

# ImageNet evaluation on Android

The shared [evaluator](../../litert/tools/imagenet_eval_runner.md) supports FP32,
INT8 and UINT8 classification I/O, NCHW/NHWC layouts, native preprocessing, and
CPU/GPU/NPU selection. CPU-reference comparison uses the same preprocessed
image for both models. These instructions run only three images.

## Build

From the worktree root, use the documented Android SDK/NDK and local caches:

```sh
mkdir -p .bazelisk-cache .cache .arm-vulkan-preflight/bazel-android
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
bazelisk --output_base="$PWD/.arm-vulkan-preflight/bazel-android" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  --repo_env=ANDROID_SDK_HOME=/data/bt/os/android_sdk \
  --repo_env=ANDROID_NDK_HOME=/data/bt/os/android_sdk/ndk/28.2.13676358 \
  --repo_env=ANDROID_NDK_VERSION=28 \
  --repo_env=ANDROID_SDK_API_LEVEL=35 --repo_env=ANDROID_NDK_API_LEVEL=35 \
  --repo_env=ANDROID_BUILD_TOOLS_VERSION=35.0.0 --config=android_arm64 \
  //litert/tools:imagenet_eval_runner
```

The normal Android target links the MLDrift OpenCL/OpenGL accelerator. For the
dynamic-runtime variant, add `--define=litert_runtime_link_mode=dynamic` and
build these targets with the same configuration:

- `//litert/tools:imagenet_eval_runner_dynamic_runtime`
- `//litert/c:litert_runtime_c_api_so`
- `//litert/runtime/accelerators/gpu:ml_drift_cl_gl_accelerator_so`

Deploy the resulting `libLiteRt.so` and `libLiteRtClGlAccelerator.so` together.
Use libraries built against the same SDK; do not accidentally select a runtime
or GPU library from another wheel. Set `LD_LIBRARY_PATH` as well as the runtime
and dispatch directory flags for the dynamic executable.

## Deploy a bounded fixture

Select the phone explicitly with `adb -s SERIAL`. Use a fresh directory below
`/data/local/tmp`, and push the executable, model, checkpoint preprocessing JSON,
a three-row manifest and only its three JPEGs. Preserve manifest sample IDs and
relative image paths. `tools/export_imagenet_val_jpegs.py` can export original
JPEGs when a local dataset cache is available; exporting the full dataset is
not required for a smoke check.

Make the executable runnable with `chmod +x`. Run from the deployment directory
so paths in the following examples resolve there. Set the serial and deployment
path for the actual device; do not rely on ADB's implicit default device.

## CPU and MLDrift GPU comparison

CPU/XNNPACK against one-thread CPU reference:

```sh
./imagenet_eval_runner \
  --model=model.tflite --baseline_model=model.tflite \
  --accelerator=cpu --num_threads=4 --xnnpack_only=1 \
  --baseline_accelerator=cpu --baseline_num_threads=1 \
  --manifest=manifest.tsv --image_root=. \
  --preprocessing_config=checkpoint.json --max_samples=3 \
  --out=cpu.jsonl --summary_json=cpu.json
```

MLDrift OpenCL FP32 against CPU reference, using an FP32 model:

```sh
./imagenet_eval_runner \
  --model=model_fp32.tflite --baseline_model=model_fp32.tflite \
  --accelerator=gpu --gpu_backend=opencl --gpu_use_ir_model=true \
  --gpu_precision=fp32 --require_full_acceleration=1 \
  --baseline_accelerator=cpu --baseline_num_threads=1 \
  --require_output_parity=1 --atol=1e-4 --rtol=1e-3 \
  --manifest=manifest.tsv --image_root=. \
  --preprocessing_config=checkpoint.json --max_samples=3 \
  --out=gpu.jsonl --summary_json=gpu.json
```

For the dynamic target, substitute its executable name and add
`--runtime_library_dir=dynamic --dispatch_library_dir=dynamic` with
`LD_LIBRARY_PATH=$PWD/dynamic`. That directory must contain the two matching
libraries above.

Quantized models require explicitly chosen score tolerances; one output code
corresponds to the output scale. CPU delegates and reference kernels can round
differently. Record the differences rather than applying FP32 tolerances or
claiming exact integer agreement.

## Verify execution

Require exit code zero, `status: passed` and `samples: 3`. Check the candidate's
full-acceleration result, delegation metrics and GPU initialization logs; CPU
fallback is not a passing GPU test. Confirm the requested IR-model setting,
OpenCL initialization and the built MLDrift accelerator. Dynamic runs must
record the expected loaded library paths and hashes. Static runs identify their
accelerator through the executable hash and registration logs.

Pull summaries, JSONL rows and logs. `android_device` records device properties;
retain the ADB serial used to launch the test as well. Small sample accuracy
values and inference times are diagnostic, not dataset accuracy or performance
claims.

Arm Vulkan ML uses `--accelerator=npu` plus its explicit compiler/JIT/dispatch
options. It must not be confused with MLDrift GPU. Other vendor NPU workflows
retain their own model-compilation and dispatch requirements.
