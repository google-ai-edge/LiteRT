# LiteRT Tensor WASM Demos

This directory contains demo applications demonstrating high-performance
machine learning inference and simulations in the browser using LiteRT WASM
with WebGPU acceleration.

## Demos

### 1. Segmentation Demo (`segmentation_demo.html`)

- **Description**: Captures video from the webcam, runs a 3-stage
pipeline (pre-processing, core segmentation model, post-processing), and renders
the result with alpha blending.
- **Optimizations**: Uses direct GPU copies between stages and `writeBuffer`
for inputs.
- **Model**: `selfie_multiclass_256x256.tflite`

### 2. Mandelbrot Set (`mandelbrot_demo.html`)

- **Description**: An animated and interactive visualization of the Mandelbrot
set.
- **Key Features**: Interactive zoom (click on canvas to zoom in).
- **Backend**: Runs on CPU for reliable animation.

### 3. Conway's Game of Life (`game_of_life_demo.html`)

- **Description**: A simulation of Conway's Game of Life using
`conv2d` operations.
- **Backend**: High performance on GPU (WebGPU) using direct GPU buffer
updates (`writeBuffer`).

### 4. Interactive Playground (`playground.html`)

- **Description**: An interactive code sandbox where users can write and execute JavaScript code using the LiteRT WASM API.
- **Key Features**: Includes snippets for basic math, reductions, broadcasting, static graphs, and eager mode.

### 5. Segmentation (WebGPU C++ Pipeline) (`segmentation_webgpu_demo.html`)

- **Description**: A segmentation demo that uses a C++ pipeline defined in `segmentation_webgpu_wasm.cc` with custom shaders.
- **Backend**: WebGPU accelerated via WASM.

### 6. Gemma 3 270M Demo (`gemma3/gemma3_demo.html`)

- **Description**: A demo running the Gemma 3 270M model.
- **Backend**: WebGPU accelerated with CPU fallback support.

## How to Run

You can run any of the demos directly using Blaze, which will build the
target and start a local development server:

```bash
blaze run //third_party/odml/litert/tensor/wasm:playground
blaze run //third_party/odml/litert/tensor/wasm:segmentation_demo
blaze run //third_party/odml/litert/tensor/wasm:mandelbrot_demo
blaze run //third_party/odml/litert/tensor/wasm:game_of_life_demo
blaze run //third_party/odml/litert/tensor/wasm:segmentation_webgpu_demo
blaze run //third_party/odml/litert/tensor/wasm:gemma3_demo
```

Follow the instructions in the terminal to open the URL in your browser.
