# LiteRT Android CPU/GPU/NPU Image Segmentation
__Disclaimer:__

*LiteRT NPU acceleration is only available through an Early Access Program. If
you are not already enrolled, [sign up](http://forms.gle/CoH4jpLwxiEYvDvF6).
See [NPU acceleration instruction](https://ai.google.dev/edge/litert/next/eap/npu) for more information about compiling NPU
models and setup NPU runtime.*

## This project demonstrates:

1.  Preprocessing an input image: resizing to 256x256, and normalizing pixel
values to the [-1, 1] range (stored in a float OpenGL buffer).
2.  Performing image segmentation on the preprocessed image using async API of
LiteRT, generating 6 masks.
3.  Assigning a unique color to each mask and blending these colored masks onto
the original (un-preprocessed) input image.

It's a command-line tool intended to be run via android ADB.
The C++ code is organized into `ImageUtils` and `ImageProcessor`, and
`SegmentationModel` classes.

## Image Processing Workflow:

1.  Load one input image (as bytes).
2.  Create an OpenGL texture for the input image (from byte data).
3.  **Preprocessing for Segmentation:**
    *   The input image texture is passed to
        `ImageProcessor::preprocessInputForSegmentation`.
    *   This step resizes the image to 256x256, and normalizes its pixel values
        to the [-1, 1] range, outputting a new preprocessed **float** OpenGL
        buffer.
4.  **Segmentation:**

    *   The preprocessed float buffer (256x256) is passed to
        `SegmentationModel`.
    *   The `SegmentationModel` is initialized with an accelerator preference
        **(CPU, GPU, or NPU)**.
    *   `SegmentationModel` reads buffer data (as floats).
    *   For **GPU**, the preprocessed buffer is created with 4-channel aligned,
        in order to be compatible with the GPU accelerator. This will allow the
        buffer to be directly bind as model input and executed in async fashion.
    *   For **NPU** and **CPU**, the buffer will be downloaded to CPU and
        executed in sync model.
    *   It loads the segmentation model from model directory, and creates a
        `LiteRT` `CompiledModel` instance. It binds the input buffer to the
        model and generates 6 (256x256)segmentation masks for different classes.
    *   User can specify three different accelerators (gpu/npu/cpu) for
        executing the model.
    *   User can specify whether `SegmentationModel` should use GL buffers
        (`--use_gl_buffers`) for input and output buffers, eliminating the need
        to create new OpenGL buffers. This enables zero-copy between pre and
        post processing. Your target device must support OpenCL-OpenGL buffer
        sharing (`cl_khr_gl_sharing`) to use this feature.
        * Supported on (not limited to): Samsung Galaxy S24/S25

5.  **Coloring and Blending Masks:**

    *   OpenGL buffer are created for each of the 6 (256x256) single-channel
        byte masks.
    *   A predefined set of 6 RGBA colors is used.
    *   The `ImageProcessor` blends the original input image (not the
        preprocessed one) with these 6 colored masks. The masks are implicitly
        scaled to match the original image dimensions during blending by the
        shader.
    *   The final blended image is saved.

## Prerequisites

1.  **clang or gcc**: Installed.
2.  **Android NDK and SDK**: Installed. (Tested with NDK=25c, SDK=34)
3.  **Bazel**: Installed.
4.  **ADB**: Installed and in PATH.
5.  **LiteRT**: [LiteRT libraries](https://github.com/google-ai-edge/LiteRT).

### Build Instructions

All commands should be run from the root of the LiteRT repository.

Configure the build tools:
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

```bash
bazel build //litert/samples/async_segmentation:async_segmentation_cpu --config=android_arm64
bazel build //litert/samples/async_segmentation:async_segmentation_gpu --config=android_arm64
bazel build //litert/samples/async_segmentation:async_segmentation_npu --config=android_arm64
```

### Running the Executables
After building, use the `deploy_and_run_on_android.sh` script to deploy and run the executables.
```bash
# For CPU
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=cpu --phone=s25 bazel-bin/
# For GPU
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=gpu --phone=s25 bazel-bin/
# For GPU with GL buffers
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=gpu --use_gl_buffers --phone=s25 bazel-bin/
# For NPU with an ahead-of-time compiled model
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=npu --phone=s25 bazel-bin/
# For NPU with just-in-time (jit) compilation of the model
./litert/samples/async_segmentation/deploy_and_run_on_android.sh --accelerator=npu --phone=s25 --jit bazel-bin/
```
The output image `output_segmented.png` will be pulled from the device and saved in the current directory.

### Performance

*Performance measured on Samsung S25 Ultra, includes both pre/post processing.*

| Processor             | Execution Type                 | Time (ms) |
| :-------------------- | :----------------------------- | :-------- |
| CPU                   | Sync Exec                      | 116       |
| GPU                   | Sync Exec                      | 35        |
| GPU                   | Async Exec + 0-copy buffer     | 17        |
| NPU                   | Sync Exec (AOT)                | 17        |
| NPU                   | Sync Exec (JIT)                | 28        |

