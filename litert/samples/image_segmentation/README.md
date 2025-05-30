# LiteRT Image Segmentation Samples

This directory contains Android image segmentation samples demonstrating how to
use LiteRT (Google's new runtime for TensorFlow Lite) with different hardware
accelerators. The samples perform multi-class image segmentation to identify and
classify objects in images.

## Overview

Image segmentation is a computer vision task that involves partitioning an image
into multiple segments or regions, where each pixel is assigned to a specific
class. These samples demonstrate real-time segmentation on Android devices using
optimized neural network models.

## Available Implementations

### 1. kotlin_cpu_gpu

A standard implementation supporting CPU and GPU acceleration for broad device
compatibility.

**Features:**

-   CPU and GPU delegate support

-   Real-time camera and gallery image segmentation

-   21-class segmentation (person, car, bicycle, etc.)

-   Compatible with a wide range of Android devices

**Performance on Samsung S25 ULtra:**

-   CPU: 120-140ms per frame

-   GPU: 40-50ms per frame

### 2. kotlin_npu

An advanced implementation with Neural Processing Unit (NPU) support for
significantly faster inference on compatible devices.

**Features:**

-   CPU, GPU, and NPU delegate support

-   Optimized for Qualcomm and MediaTek NPU hardware

-   Same 21-class segmentation as CPU/GPU version

-   Requires enrollment in the
    [Early Access Program](forms.gle/CoH4jpLwxiEYvDvF6)

**Performance on Samsung S25 Ultra:**

-   CPU: 120-140ms per frame

-   GPU: 40-50ms per frame

-   NPU: 6-12ms per frame (10-20x faster than CPU!)

## Technical Details

### Model Architecture

-   Input: 256x256x3 RGB image

-   Output: 256x256x6 segmentation mask

-   Model format: TensorFlow Lite (.tflite)

### Key Dependencies

-   LiteRT (com.google.ai.edge.litert): 2.0.0-alpha

-   Android CameraX: For camera functionality

-   Jetpack Compose: For modern UI

-   Kotlin Coroutines: For asynchronous operations

### Architecture Components

-   **ImageSegmentationHelper**: Core segmentation logic and model management

-   **MainActivity**: UI orchestration with camera/gallery tabs

-   **MainViewModel**: State management and data flow

-   **TensorUtils**: Utilities for tensor inspection and debugging

## Contributing

When contributing to these samples:

1.  Follow existing code style and patterns

2.  Test on multiple devices and accelerators

3.  Update documentation for any new features

4.  Include performance metrics for optimizations
