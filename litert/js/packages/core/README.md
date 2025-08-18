# @litertjs/core

Core web runtime for [LiteRT](https://ai.google.dev/edge/litert) ([previously
TFLite](https://developers.googleblog.com/en/tensorflow-lite-is-now-litert/)),
Google's open-source high performance runtime for on-device AI.

This package provides the core functionality to load and run arbitrary LiteRT
(`.tflite`) models directly in the browser, with support for WebGPU and CPU
acceleration.

This is the primary package for LiteRT.js. For integration with TensorFlow.js,
see
[`@litertjs/tfjs-interop`](https://www.npmjs.com/package/@litertjs/tfjs-interop).
To test and benchmark your models, check out the
[`@litertjs/model-tester`](https://www.npmjs.com/package/@litertjs/model-tester).

## Features

-   Load and run a LiteRT (.tflite) model.
-   Run with WebGPU acceleration on supported browsers.
-   Run with ([XNNPack](https://github.com/google/XNNPACK))-accelerated CPU
    kernels on any browser.
-   Slot into existing [TensorFlow.js](https://github.com/tensorflow/tfjs)
    pipelines as a replacement for [TFJS Graph
    Models](https://js.tensorflow.org/api/latest/#loadGraphModel). See
    [`@litertjs/tfjs-interop`](https://www.npmjs.com/package/@litertjs/tfjs-interop)
    package for details.

## Usage

For a complete guide, see our docs at
[ai.google.dev/edge/litert/web](https://ai.google.dev/edge/litert/web).

The following code snippet loads LiteRT.js, loads a MobileNetV2 model, and runs
it on a sample input tensor.

```typescript
import {loadLiteRt, loadAndCompile, Tensor} from '@litertjs/core';

// Initialize LiteRT.js's Wasm files.
// These files are located in `node_modules/@litertjs/core/wasm/`
// and need to be served by your web server.
await loadLiteRt('/path/to/wasm/directory/');

const model = await loadAndCompile(
  '/path/to/your/model/torchvision_mobilenet_v2.tflite',
  {accelerator: 'webgpu'},
);

// Create an input tensor.
const inputTypedArray = new Float32Array(1 * 3 * 224 * 224);
const inputTensor = new Tensor(inputTypedArray, [1, 3, 224, 224]);

// Move the tensor to the GPU for a WebGPU-accelerated model.
const gpuTensor = await inputTensor.moveTo('webgpu');

// Run the model.
const results = model.run(gpuTensor);

// All tensors that are not moved to another backend with `moveTo` must
// eventually be freed with `.delete()`.
gpuTensor.delete();

// Move the result back to the CPU to read the data.
const result = results[0].moveTo('wasm');
console.log(result.toTypedArray());

// Clean up the result tensor.
result.delete();
```