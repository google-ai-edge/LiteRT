# LiteRT Android CPU/GPU/NPU Image Segmentation

## This project demonstrates:

1.  Preprocessing an input image: resizing to 256x256, and normalizing pixel
values to the [-1, 1] range (stored in a float OpenGL buffer).
2.  Performing image segmentation on the preprocessed image using the async API of
LiteRT, generating 6 masks.
3.  Assigning a unique color to each mask and blending these colored masks onto
the original (un-preprocessed) input image.

It's a command-line tool intended to be run via Android ADB.
The C++ code is organized into `ImageUtils`, `ImageProcessor` utility classes, and execution layers separated by hardware target (`main_cpu.cc`, `main_gpu.cc`, `main_npu.cc`).

## Image Processing Workflow:

1.  Load one input image (as bytes) into memory.
2.  Create an OpenGL texture for the input image (from byte data).
3.  **Preprocessing for Segmentation:**
    *   The input image texture is passed to
        `ImageProcessor::PreprocessInputForSegmentation`.
    *   This step resizes the image to 256x256, and normalizes its pixel values
        to the [-1, 1] range, outputting a new preprocessed **float** OpenGL
        buffer.
4.  **Segmentation:**

    *   The preprocessed float buffer (256x256) is passed to a `litert::CompiledModel` instance.
    *    A `CompiledModel` is initialized with an accelerator preference
        **(CPU, GPU, or NPU)** via `litert::Environment` options.
    *   The `CompiledModel` reads buffer data (as floats) to perform validation.
    *   For **GPU**, the preprocessed buffer is created with 4-channel aligned,
        in order to be compatible with the GPU accelerator. This allows the
        buffer to be directly bound as model input and executed in an **async** fashion (`compiled_model.RunAsync(...)`). Since the result targets an OpenCL-OpenGL Shader Storage Buffer Object (SSBO), the CPU is not blocked downloading buffers. 
    *   For **NPU** and **CPU**, the model buffer will be downloaded to CPU memory and
        executed in sync mode (`compiled_model.Run(...)`).
    *   The model generates 6 (256x256) segmentation masks for different classes.

5.  **Coloring and Blending Masks:**

    *   An OpenGL SSBO (Shader Storage Buffer Object) is created for each of the 6 (256x256) single-channel
        float masks.
    *   A predefined set of 6 RGBA colors is used.
    *   The `ImageProcessor` blends the original input image (not the
        preprocessed one) with these 6 colored masks. The masks are implicitly
        scaled to match the original image dimensions during blending by the
        shader.
    *   The final blended image is saved to disk (`output_segmented.png`).

## Prerequisites

1.  **clang or gcc**: Installed.
2.  **Android NDK and SDK**: Installed. (Tested with NDK=25c, SDK=34)
3.  **Blaze**: Installed.
4.  **ADB**: Installed and in PATH.
5.  **LiteRT**: [LiteRT libraries](https://github.com/google-ai-edge/LiteRT).

### Build Instructions

Configure the build tools (if not inside an active Google3 workspace):
```bash
./configure
# default python
# default python lib path
# N to ROCm support
# N to CUDA support
# Best tested with clang (tested with 18.1.3)
# default opt flags
# configure ./WORKSPACE for Android builds (y)
# Min Android NDK level (at least 26)
# configure path to sdk
# specify Android SDK API level (tested with 34)
# specify Android build tools (tested with 34.0.0)
```

Build the CPU and GPU standalone executables using Blaze:
```bash
blaze build //litert/samples/async_segmentation:async_segmentation_cpu --config=android_arm64
blaze build //litert/samples/async_segmentation:async_segmentation_gpu --config=android_arm64
```

Build the specific custom NPU executable using Blaze (Qualcomm and MediaTek targets directly map to pre-imported `third_party/` libraries):

```bash
# For Qualcomm NPU Build (uses existing //third_party/qairt dependencies)
blaze build //litert/samples/async_segmentation:async_segmentation_npu \
  --config=android_arm64 \
  --nocheck_visibility

# For MediaTek NPU Build (No extra SDK required)
blaze build //litert/samples/async_segmentation:async_segmentation_npu_mtk \
  --config=android_arm64 \
  --nocheck_visibility
```

> [!NOTE]
> The `--nocheck_visibility` flag is recommended because some upstream LiteRT targets have restricted visibility defaults that may conflict with external usage.

### Running the Executables
After building, use the local `deploy_and_run_on_android.sh` script to deploy and run the executables on your Android ADB device.
```bash
# For CPU
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=cpu --phone=s25 blaze-bin/
# For GPU (defaults to OpenCL)
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=gpu --phone=s25 blaze-bin/
# For GPU (OpenGL backend)
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=gpu --backend=opengl --phone=s25 blaze-bin/
# For Qualcomm NPU with an ahead-of-time (AOT) compiled model
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=npu --phone=s25 blaze-bin/
# For Qualcomm NPU with just-in-time (JIT) compilation of the model
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=npu --phone=s25 --jit blaze-bin/

# For MediaTek APU (dim9400 chipset)
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=npu --phone=dim9400 --jit blaze-bin/
```
The script pulls the generated final output image mask blending from the device `tmp` folder. Look for `output_segmented.png` in your current directory.

### Performance

*Performance measured on Samsung S25 Ultra (Qualcomm) and MediaTek Dimensity 9400.*

| Processor             | Execution Type                 | Time (ms) |
| :-------------------- | :----------------------------- | :-------- |
| CPU                   | Sync Exec                      | 116       |
| GPU                   | Sync Exec                      | 35        |
| GPU                   | Async Exec + 0-copy buffer     | 17        |
| NPU                   | Sync Exec (AOT)                | 17        |
| NPU                   | Sync Exec (JIT)                | 28        |
| MediaTek APU          | Sync Exec (JIT)                | 9         |
