# Gemma3 LiteRT Example

This directory contains several experimental Gemma3 entry points. The supported
OSS entry point documented here is `litert_main`, which uses the LiteRT compiled
model path and the platform LiteRT GPU accelerator prebuilt.

The `xnnpack_main`, `opencl_main`, and `webgpu_main` entry points are not covered
by these instructions.

## Inputs

The executable expects model weights and a SentencePiece tokenizer model:

```sh
--weights_path=/path/to/model.safetensors
--tokenizer_path=/path/to/tokenizer.model
```

Useful runtime flags:

```sh
--prompt="Hello, world!"
--max_tokens=50
--accelerator=gpu
```

## Build With Bazel

From the repository root:

```sh
bazel build //tensor/examples/gemma3:litert_main
```

The host executable is produced at:

```sh
bazel-bin/tensor/examples/gemma3/litert_main
```

Run it from the repository root with the GPU accelerator directory on the dynamic
loader path. For macOS arm64:

```sh
DYLD_LIBRARY_PATH=$PWD/bazel-bin/tensor/examples/gemma3/litert_main.runfiles/litert_prebuilts/macos_arm64 \
  bazel-bin/tensor/examples/gemma3/litert_main \
  --weights_path=/path/to/model.safetensors \
  --tokenizer_path=/path/to/tokenizer.model \
  --prompt="Hello, world!" \
  --max_tokens=50 \
  --accelerator=gpu
```

On Linux, use `LD_LIBRARY_PATH` and the matching
`linux_x86_64` or `linux_arm64` runfiles directory instead.

For CPU execution with the provided quantized safetensor model, dequantize
weights while building the temporary TFLite model:

```sh
bazel-bin/tensor/examples/gemma3/litert_main \
  --weights_path=/path/to/model.safetensors \
  --tokenizer_path=/path/to/tokenizer.model \
  --prompt="Hello, world!" \
  --max_tokens=1 \
  --accelerator=cpu \
  --weight_mode=float
```

The Bazel target declares the platform GPU accelerator prebuilt through
`litert_gpu_accelerator_prebuilts()`. For manual packaging or deployment, locate
the selected prebuilt with:

```sh
find bazel-bin -name 'libLiteRt*Accelerator*' -print
```

## Build With CMake

From the repository root:

```sh
cmake -S litert -B cmake_build/gemma3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLITERT_ENABLE_GPU=ON \
  -DLITERT_BUILD_TESTS=OFF
cmake --build cmake_build/gemma3 --target litert_main
```

The CMake target is `EXCLUDE_FROM_ALL`, so build `litert_main` explicitly. The
host executable is produced at:

```sh
cmake_build/gemma3/tensor/examples/gemma3/litert_main
```

The CMake build downloads the matching LiteRT GPU accelerator prebuilt and
copies it next to `litert_main` after the executable links. For example, on
macOS arm64 the copied accelerator is:

```sh
cmake_build/gemma3/tensor/examples/gemma3/libLiteRtMetalAccelerator.dylib
```

Run the host binary with:

```sh
DYLD_LIBRARY_PATH=$PWD/cmake_build/gemma3/tensor/examples/gemma3 \
  cmake_build/gemma3/tensor/examples/gemma3/litert_main \
  --weights_path=/path/to/model.safetensors \
  --tokenizer_path=/path/to/tokenizer.model \
  --prompt="Hello, world!" \
  --max_tokens=50 \
  --accelerator=gpu
```

On Linux, use `LD_LIBRARY_PATH` instead of `DYLD_LIBRARY_PATH`.
Use the same `--accelerator=cpu --weight_mode=float` flags for a CMake CPU run.

## Android Cross Compilation

Android GPU execution uses the arm64 LiteRT OpenCL/GL accelerator prebuilt:

```text
libLiteRtClGlAccelerator.so
```

### Bazel Android Build

Install an Android NDK and make it visible to Bazel, for example:

```sh
export ANDROID_NDK_HOME=/path/to/android-ndk
```

Then build the arm64 Android target from the repository root:

```sh
bazel build --config=android_arm64 //tensor/examples/gemma3:litert_main
```

The relevant Android config is defined in the repository `.bazelrc`; it sets
`--cpu=arm64-v8a`, `--fat_apk_cpu=arm64-v8a`, and the TensorFlow Android arm64
platform.

When deploying manually, push the generated executable and the Android GPU
accelerator to the same directory on the device. The `litert_main` binary links
LiteRT statically; it does not need `libLiteRt.so`.

```sh
DEVICE_DIR=/data/local/tmp/litert_gemma3_bazel
ASSET_DIR=/data/local/tmp/litert_gemma3_assets
RUNFILES=bazel-bin/tensor/examples/gemma3/litert_main.runfiles
GPU_ACCELERATOR_SO=$(find "$RUNFILES" -path '*/android_arm64/libLiteRtClGlAccelerator.so' -print -quit)

adb shell "rm -rf $DEVICE_DIR && mkdir -p $DEVICE_DIR $ASSET_DIR"
adb push bazel-bin/tensor/examples/gemma3/litert_main "$DEVICE_DIR/"
adb push "$GPU_ACCELERATOR_SO" "$DEVICE_DIR/"
adb push /path/to/model.safetensors "$ASSET_DIR/"
adb push /path/to/tokenizer.model "$ASSET_DIR/"

adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=. ./litert_main \
  --weights_path=$ASSET_DIR/model.safetensors \
  --tokenizer_path=$ASSET_DIR/tokenizer.model \
  --prompt='Hello, world!' \
  --max_tokens=50 \
  --accelerator=gpu"
```

For Android CPU execution, omit the accelerator `.so` from deployment if you do
not need GPU, and run with `--accelerator=cpu --weight_mode=float`.

### CMake Android Build

Install an Android NDK and a host `protoc`, then point CMake at the NDK
toolchain file. The host `protoc` must be compatible with the Protobuf headers
used by the CMake build; this checkout currently fetches Protobuf 3.21.9, so a
newer package-manager `protoc` may generate incompatible C++.
If CMake is configured for Android without a host `protoc`, the Gemma3
`litert_main` target is skipped so other examples in the same build tree can
still build.

```sh
export ANDROID_NDK_HOME=/path/to/android-ndk
export LITERT_HOST_PROTOC=/path/to/host/protoc

cmake -S litert -B cmake_build/gemma3_android_arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLITERT_ENABLE_GPU=ON \
  -DLITERT_HOST_PROTOC="$LITERT_HOST_PROTOC" \
  -DLITERT_BUILD_TESTS=OFF

cmake --build cmake_build/gemma3_android_arm64 --target litert_main
```

The Android CMake build downloads `android_arm64/libLiteRtClGlAccelerator.so`
and copies it next to `litert_main`.

Typical manual deployment:

```sh
adb shell 'rm -rf /data/local/tmp/gemma3 && mkdir -p /data/local/tmp/gemma3'
adb push cmake_build/gemma3_android_arm64/tensor/examples/gemma3/litert_main \
  /data/local/tmp/gemma3/
adb push cmake_build/gemma3_android_arm64/tensor/examples/gemma3/libLiteRtClGlAccelerator.so \
  /data/local/tmp/gemma3/
adb push /path/to/model.safetensors /data/local/tmp/gemma3/
adb push /path/to/tokenizer.model /data/local/tmp/gemma3/

adb shell 'cd /data/local/tmp/gemma3 && \
  chmod +x litert_main && \
  LD_LIBRARY_PATH=. ./litert_main \
    --weights_path=model.safetensors \
    --tokenizer_path=tokenizer.model \
    --prompt="Hello, world!" \
    --max_tokens=50 \
    --accelerator=gpu'
```

For CMake Android CPU execution, use the same deployment pattern but run with
`--accelerator=cpu --weight_mode=float`; the GPU accelerator `.so` is not needed
for CPU-only execution.

