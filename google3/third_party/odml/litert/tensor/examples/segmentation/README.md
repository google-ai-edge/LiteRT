# LiteRT Segmentation Pipeline Modernization

This directory contains a modernized implementation of the LiteRT segmentation
pipeline, leveraging the high-level **Tensor API** to simplify implementation
and enable seamless hardware acceleration.

## Key Flow of the Example

The pipeline performs the following steps to process an image and produce a
segmented blend:

1.  **Image Loading & Pre-processing:**
    *   Loads a JPEG image using `ImageUtils`.
    *   Resizes and normalizes the image floats to fit the expected model inputs (`256x256x3`).
2.  **Model Execution:**
    *   Instantiates `CompiledModelRunner` targeting the pre-compiled `.tflite` segmentation model.
    *   Executes inference optionally backed by OpenCL **GPU** hardware acceleration or failing over to custom CPU parameters.
3.  **Post-processing via Tensor API:**
    *   Identifies winning classes for each pixel via `ArgMax`.
    *   Applies `Reshape` and 1D `Gather` workarounds to compliant dimensions to force indices to handle restrictive GPU device delegate operations natively.
    *   Scales colors by blending factors via isolated `Mul` and `Add` operations.
4.  **Verification & Saving:**
    *   Saves the final blended image directly as a PNG file.
    *   Cross-checks the rendered pixels against a baseline reference golden image securely to verify hardware precision accuracy!

## Main Benefits of the Tensor API Rewrite

Rewriting the segmentation example to use the modern LiteRT Tensor API provides
several key advantages over legacy TFLite interpreter methods:

*   **Declarative Graph Operations:** Expresses post-processing steps (like
    `ArgMax`, `Gather`, `Reshape`, `Add`) as clean, readable C++ functions
    instead of writing manual, error-prone nested loops over raw pointer
    calculations!
*   **Abstracted Buffer Lifecycles:** `CompiledModelRunner` and
    `TensorBuffer::CreateManaged` automate complex memory ownership details and
    lifecycle hooks cleanly under the hood.
*   **Name-based I/O Binding:** Instead of tracking input/output indices
    manually (which break easily when model shapes or nodes change), tensors are
    connected by their graph names.
*   **Hardware Agnostic Setup:** Setting up edge operations on custom delegates
    is abstracted away via structured `Options` trees, preventing messy manual
    driver wiring.

## Running on Device

The commands below run the same Tensor API pipeline on either the host machine or
an Android device. Run them from the repository root.

### Bazel on Host

```bash
bazel build //tensor/examples/segmentation:segmentation_example

DYLD_LIBRARY_PATH=$PWD/bazel-bin/tensor/examples/segmentation/segmentation_example.runfiles/litert_prebuilts/macos_arm64 \
  bazel-bin/tensor/examples/segmentation/segmentation_example \
  --image_path=tensor/examples/segmentation/image.jpeg \
  --core_model_path=tensor/examples/segmentation/selfie_multiclass_256x256.tflite \
  --output_dir=/tmp
```

This writes `/tmp/segmented_output.png`. The Bazel target includes the platform
GPU accelerator in runfiles through `litert_gpu_accelerator_prebuilts()`. On
Linux, use `LD_LIBRARY_PATH` and the matching `linux_x86_64` or `linux_arm64`
runfiles directory instead.
For CPU execution, pass `--accelerator=cpu`; the example defaults to GPU.

### CMake on Host

```bash
cmake -S litert -B cmake_build/segmentation_host \
  -DCMAKE_BUILD_TYPE=Release \
  -DLITERT_ENABLE_GPU=ON

cmake --build cmake_build/segmentation_host \
  --target litert_tensor_segmentation_example \
  --parallel

DYLD_LIBRARY_PATH=$PWD/cmake_build/segmentation_host/tensor/examples/segmentation \
  cmake_build/segmentation_host/tensor/examples/segmentation/litert_tensor_segmentation_example \
  --image_path=tensor/examples/segmentation/image.jpeg \
  --core_model_path=tensor/examples/segmentation/selfie_multiclass_256x256.tflite \
  --output_dir=/tmp
```

This also writes `/tmp/segmented_output.png`. With `LITERT_ENABLE_GPU=ON`, CMake
downloads the host GPU accelerator and copies it next to the executable. On
Linux, use `LD_LIBRARY_PATH` instead of `DYLD_LIBRARY_PATH`.
For CPU execution, pass `--accelerator=cpu`.

### Bazel on Android

Build the Android ARM64 binary first. Adjust the SDK and NDK paths for your
machine:

```bash
ANDROID_SDK_HOME=$HOME/Library/Android/sdk \
ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/28.0.13004108 \
ANDROID_SDK_API_LEVEL=35 \
ANDROID_NDK_API_LEVEL=26 \
ANDROID_BUILD_TOOLS_VERSION=35.0.1 \
ANDROID_NDK_VERSION=28 \
bazel build --config=android_arm64 \
  //tensor/examples/segmentation:segmentation_example
```

Push the binary, model, image, and Android GPU accelerator to the device. The
example binary links LiteRT statically; it does not need `libLiteRt.so`.
For CPU-only execution, the accelerator `.so` can be omitted and the run command
can use `--accelerator=cpu`.

```bash
DEVICE_DIR=/data/local/tmp/litert_segmentation
RUNFILES=bazel-bin/tensor/examples/segmentation/segmentation_example.runfiles
GPU_ACCELERATOR_SO=$(find "$RUNFILES" -path '*/android_arm64/libLiteRtClGlAccelerator.so' -print -quit)

adb shell "rm -rf $DEVICE_DIR && mkdir -p $DEVICE_DIR"
adb push bazel-bin/tensor/examples/segmentation/segmentation_example "$DEVICE_DIR/"
adb push tensor/examples/segmentation/image.jpeg "$DEVICE_DIR/"
adb push tensor/examples/segmentation/selfie_multiclass_256x256.tflite "$DEVICE_DIR/"
adb push "$GPU_ACCELERATOR_SO" "$DEVICE_DIR/"
```

Run it on device and pull the rendered output back to the host:

```bash
adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=. ./segmentation_example \
  --image_path=image.jpeg \
  --core_model_path=selfie_multiclass_256x256.tflite \
  --output_dir=."

adb pull "$DEVICE_DIR/segmented_output.png" ./android_output.png
```

### CMake on Android

Configure CMake with the Android NDK toolchain. The Android CMake build uses
`nativewindow`, so set the minimum Android API to 26 or newer:

```bash
export ANDROID_SDK_HOME=$HOME/Library/Android/sdk
export ANDROID_NDK_HOME=$ANDROID_SDK_HOME/ndk/28.0.13004108

cmake -S litert -B cmake_build/segmentation_android_arm64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLITERT_ENABLE_GPU=ON \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DLITERT_HOST_C_COMPILER=/usr/bin/clang \
  -DLITERT_HOST_CXX_COMPILER=/usr/bin/clang++

cmake --build cmake_build/segmentation_android_arm64 \
  --target litert_tensor_segmentation_example \
  --parallel
```

Push the CMake-built artifacts and assets:

```bash
DEVICE_DIR=/data/local/tmp/litert_segmentation

adb shell "rm -rf $DEVICE_DIR && mkdir -p $DEVICE_DIR"
adb push cmake_build/segmentation_android_arm64/tensor/examples/segmentation/litert_tensor_segmentation_example "$DEVICE_DIR/"
adb push cmake_build/segmentation_android_arm64/tensor/examples/segmentation/libLiteRtClGlAccelerator.so "$DEVICE_DIR/"
adb push tensor/examples/segmentation/image.jpeg "$DEVICE_DIR/"
adb push tensor/examples/segmentation/selfie_multiclass_256x256.tflite "$DEVICE_DIR/"
```

For CPU-only execution, the accelerator `.so` can be omitted and the run command
can use `--accelerator=cpu`.

Run it on device:

```bash
adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=. ./litert_tensor_segmentation_example \
  --image_path=image.jpeg \
  --core_model_path=selfie_multiclass_256x256.tflite \
  --output_dir=."

adb pull "$DEVICE_DIR/segmented_output.png" ./android_output.png
```

## Performance Measurements

Performance was measured on a **Samsung S25 (SM-S938U)**. The end-to-end
execution time was measured from input ingestion to final block synchronization
across four different configurations:

| Mode | End-to-End Execution Time | Description |
| :--- | :--- | :--- |
| **`pure_gpu`** | **10 ms** | **Zero-Copy**. All stages run on GPU with direct buffer sharing. |
| **`pure_gpu_copy`** | **11 ms** | GPU processing, but forces a redundant `memcpy` between stages. |
| **`cpu_gpu_buffer`** | **12 ms** | CPU pre-processing, GPU inference (involves CPU cache flush mapping). |
| **`pure_cpu`** | **83 ms** | All stages run on the CPU. |

### Key Takeaways

- **Zero-Copy Benefit:** Removing the CPU/GPU boundary copies and redundant memory operations eliminated latency overhead, achieving the absolute hardware floor of **10 ms**.
- **Hardware Acceleration:** Moving to a fully GPU accelerated pipeline achieved roughly an **8x speedup** end-to-end compared to baseline CPU execution (which required 83 ms).
- **Asynchrony Advantage:** Leaving execution asynchronous by avoiding mid-pipeline CPU locks allowed the OpenCL queue to execute smoothly.
