
# LiteRT Android CPU/GPU/NPU Image Segmentation

This project demonstrates:

1.  Preprocessing an input image: resizing to 256x256, and normalizing pixel
values to the [-1, 1] range (stored in a float OpenGL buffer).
2.  Performing image segmentation on the preprocessed image using async API of
LiteRT, generating 6 masks.
3.  Assigning a unique color to each mask and blending these colored masks onto
the original (un-preprocessed) input image.

It's a command-line tool intended to be run via android ADB.
The C++ code is organized into `ImageUtils`, `ImageProcessor`, and
`SegmentationModel` classes.

**Image Processing Workflow:**

1.  Load one input image (as bytes).
2.  Create an OpenGL texture for the input image (from byte data).
3.  **Preprocessing for Segmentation:**
    * The input image texture is passed to `ImageProcessor::preprocessInputForSegmentation`.
    * This step resizes the image to 256x256, and normalizes its pixel values to
    the [-1, 1] range, outputting a new preprocessed **float** OpenGL buffer.
4.  **Segmentation:**
    * The preprocessed float buffer (256x256) is passed to `SegmentationModel`.
    * The `SegmentationModel` is initialized with an accelerator preference
      (CPU, GPU, or NPU).
    * `SegmentationModel` reads buffer data (as floats).
    * It loads the segmentation model from model directory, and creates a
      `LiteRT` `CompiledModel` instance. It binds the input buffer to the
      model and generates 6 (256x256)segmentation masks for different classes.
    * User can specify three different accelerators (gpu/npu/cpu) for executing the model.

5.  **Coloring and Blending Masks:**
    * OpenGL buffer are created for each of the 6 (256x256) single-channel byte masks.
    * A predefined set of 6 RGBA colors is used.
    * The `ImageProcessor` blends the original input image (not the preprocessed
      one) with these 6 colored masks. The masks are implicitly scaled to match
      the original image dimensions during blending by the shader.
    * The final blended image is saved.

## Prerequisites

1.  **Android NDK**: Installed and configured for Bazel.
2.  **Bazel**: Installed.
3.  **ADB**: Installed and in PATH.
4.  **LiteRT**: LiteRT libraries and headers for Android.

## Setup Instructions

## WORKSPACE Setup (for Open-Source Bazel)
Edit `WORKSPACE` to point to your NDK if not using `ANDROID_NDK_HOME`.

## Build Instructions
Follow instructions for open-source Bazel.
Example (open-source):
```bash
bazel build //litert/samples/async_segmentation --config=android_arm64
```

## Running the Executable
Use the `deploy_and_run_on_android.sh` script. Review and edit paths within the
script first.
```bash
chmod +x deploy_and_run_on_android.sh
./deploy_and_run_on_android.sh --accelerator=gpu bazel-bin/
```

Check for `output_segmented.png` on the device.
