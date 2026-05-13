# @litertjs/core

Core web runtime for [LiteRT](https://ai.google.dev/edge/litert) ([previously
TFLite](https://developers.googleblog.com/en/tensorflow-lite-is-now-litert/)),
Google's open-source high performance runtime for on-device AI.

This package provides the core functionality to load and run arbitrary LiteRT
(`.tflite`) models directly in the browser, with robust support for WebGPU
acceleration, WebNN for dedicated hardware acceleration (such as NPUs), and
optimal CPU execution.

This is the primary package for LiteRT.js. For integration with TensorFlow.js,
see
[`@litertjs/tfjs-interop`](https://www.npmjs.com/package/@litertjs/tfjs-interop).
To test and benchmark your models, check out the
[`@litertjs/model-tester`](https://www.npmjs.com/package/@litertjs/model-tester).

## Features

-   **Comprehensive Model Coverage**: Load and reliably run standardized LiteRT
(`.tflite`) models directly in the web browser.
-   **WebGPU Acceleration**: Leverage WebGPU to unlock high-performance GPU
processing across modern desktops and mobile browsers.
-   **WebNN Integration**: Harness the emerging WebNN API to target dedicated
neural processing units (NPUs) or specialized hardware accelerators, enabling
highly power-efficient and ultra-low latency inference.
-   **High-Performance CPU Inference**: Fallback to CPU kernels accelerated by [XNNPack](https://github.com/google/XNNPACK) running in WebAssembly, ensuring dependable, high-speed performance on any browser architecture.
-   **Seamless TensorFlow.js Integration**: Slot into existing [TensorFlow.js](https://github.com/tensorflow/tfjs)
    pipelines as a replacement for [TFJS Graph Models](https://js.tensorflow.org/api/latest/#loadGraphModel). See the
    [`@litertjs/tfjs-interop`](https://www.npmjs.com/package/@litertjs/tfjs-interop) package for integration guidelines and types.

## Usage

For a comprehensive guide covering model conversion, debugging, and advanced
pipeline development, see our official documentation at
[ai.google.dev/edge/litert/web](https://ai.google.dev/edge/litert/web).

The following code snippet demonstrates how to initialize the LiteRT.js runtime,
load a MobileNetV2 model, prepare inputs, and execute inference utilizing
available accelerators.

```typescript
import {loadLiteRt, loadAndCompile, Tensor} from '@litertjs/core';

// 1. Initialize LiteRT.js's Wasm runtime components.
// These files are located in `node_modules/@litertjs/core/wasm/`
// and need to be served statically by your web server.
// alternatively they can be loaded via
// https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/
await loadLiteRt('/path/to/wasm/directory/');

// 2. Load and compile the model.
// Determine the appropriate accelerator based on device support:
// - 'wasm' acts as a high-performance CPU executor.
// - 'webgpu' targets integrated or dedicated graphics acceleration.
// - 'webnn' [experimental] targets integrated NPUs or OS-level acceleration.
// Note: In the case where a model cannot be fully delegated to the specified
// accelerator, LiteRT will fall back to wasm execution, as it has the widest
// model coverage & operator support.
const model = await loadAndCompile(
  '/path/to/your/model/torchvision_mobilenet_v2.tflite',
  {accelerator: 'webgpu'}, // Alternatively, use 'webnn' or 'wasm'
  // If using 'webnn', the 'jspi' option must be passed as true when calling
  // loadLiteRt() in step 1
);

// 3. Construct the input tensor.
// Dimensions map tightly to the underlying PyTorch or TensorFlow constraints,
// but can vary depending on the model conversion process.
// e.g. tensor is of the form (batch, channel, height, width)
const inputTypedArray = new Float32Array(1 * 3 * 224 * 224);
/*
  This comment is meant to represent some code which populates inputTypedArray
  with the data intended for inference.
*/
const inputTensor = new Tensor(inputTypedArray, [1, 3, 224, 224]);

// 4. Run the model
const results = await model.run(gpuTensor);

// 5. Clean up Tensors
// LiteRT.js uses manual memory management, so all tensors must be manually
// deleted when they are no longer needed.
gpuTensor.delete();

// 6. Read the model's output
const result = await results[0].moveTo('wasm');
console.log('Prediction buffer:', result.toTypedArray());

// 7. Clean up result Tensors.
results.delete();
result.delete();
```
