# Building LiteRT GPU Accelerator from Source

This guide walks open-source developers through building the LiteRT GPU
Accelerator shared library (`libLiteRtClGlAccelerator.so`) from source using
Bazel and [ML Drift](https://github.com/google-ai-edge/ml-drift).

## Prerequisites

Before starting, ensure your build machine has the following tools installed:

- **Host OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Bazel** (or [Bazelisk](https://github.com/bazelbuild/bazelisk))
- **Git**
- **C++ Compiler**: Clang 16+ or GCC 11+ with C++17 support
- **Python**: 3.10+
- **Android SDK & NDK**: Android NDK r26b (`26.3.11579264`) or r21+ recommended
  for Android builds

## Step 1: Clone the LiteRT Repository

```bash
git clone https://github.com/google-ai-edge/LiteRT.git
cd LiteRT
```

> [!NOTE]
> By default, Bazel automatically fetches the pinned version of
> [`@ml_drift`](https://github.com/google-ai-edge/ml-drift) configured in
> `WORKSPACE`. You do not need to clone `ml-drift` separately unless you want to
> build against local `ml-drift` changes (see
> [Building with a Local `ml-drift` Checkout](#optional-building-with-a-local-ml-drift-checkout)).

## Step 2: Build the Android GPU Accelerator

Set `ANDROID_NDK_HOME` to your Android NDK installation path and run Bazel to
build the OpenCL + OpenGL GPU Accelerator (`ml_drift_cl_gl_accelerator_so`):

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk-r26b

bazel build -c opt \
  --config=android_arm64 \
  --action_env=ANDROID_NDK_HOME=$ANDROID_NDK_HOME \
  //litert/runtime/accelerators/gpu:ml_drift_cl_gl_accelerator_so
```

## Step 3: Locate Build Artifacts & Deploy to Device

Once the build completes, the generated shared library is located under
`bazel-bin`:

```text
bazel-bin/litert/runtime/accelerators/gpu/libLiteRtClGlAccelerator.so
```

Push the library to your Android device alongside your test binary or
application:

```bash
adb push bazel-bin/litert/runtime/accelerators/gpu/libLiteRtClGlAccelerator.so /data/local/tmp/
```

## Optional: Building with a Local `ml-drift` Checkout

If you are developing or testing local modifications to
[ML Drift](https://github.com/google-ai-edge/ml-drift), clone `ml-drift` and pass
`--override_repository=ml_drift=<path>` to Bazel:

```bash
mkdir -p ~/workspace
cd ~/workspace

# Clone both repositories
git clone https://github.com/google-ai-edge/ml-drift.git
git clone https://github.com/google-ai-edge/LiteRT.git

cd ~/workspace/LiteRT
export ANDROID_NDK_HOME=/path/to/android-ndk-r26b

bazel build -c opt \
  --config=android_arm64 \
  --action_env=ANDROID_NDK_HOME=$ANDROID_NDK_HOME \
  --override_repository=ml_drift=$HOME/workspace/ml-drift \
  //litert/runtime/accelerators/gpu:ml_drift_cl_gl_accelerator_so
```

> [!TIP]
> The `--override_repository=ml_drift=...` flag instructs Bazel to use your
> local `ml-drift` source tree instead of downloading the pinned archive from
> GitHub.

## Troubleshooting

| Error | Cause | Solution |
| :--- | :--- | :--- |
| `ANDROID_NDK_HOME` not found / toolchain error | `ANDROID_NDK_HOME` is unset or points to an invalid path | Export `ANDROID_NDK_HOME=/path/to/ndk` and pass `--action_env=ANDROID_NDK_HOME=$ANDROID_NDK_HOME`. |
| `no such package '@@ml_drift//...'` | Invalid path in `--override_repository` | Verify `$HOME/workspace/ml-drift` exists and contains `WORKSPACE` or `MODULE.bazel`. |
| `No MODULE.bazel or WORKSPACE file found` | Wrong directory passed to `--override_repository` | Pass the absolute path to the root directory of `ml-drift`. |

## License

Copyright 2026 Google LLC. Licensed under Apache License 2.0.
