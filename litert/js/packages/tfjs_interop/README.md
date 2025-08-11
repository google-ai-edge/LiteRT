# @litertjs/tfjs-interop

Utility package for using
[@litertjs/core](https://www.npmjs.com/package/@litertjs/core) with
[TensorFlow.js](https://www.tensorflow.org/js).

This package provides helper functions to allow seamless interoperability
between the LiteRT.js and TensorFlow.js libraries. You can use it to run a
LiteRT model using TFJS tensors as inputs and receiving TFJS tensors as outputs,
making it easy to integrate LiteRT.js into an existing TFJS pipeline.

## Prerequisites

This package has peer dependencies on `@litertjs/core`, `@tensorflow/tfjs`, and
`@tensorflow/tfjs-backend-webgpu`. You must have these installed in your
project.

```bash
npm install @litertjs/core @litertjs/tfjs-interop @tensorflow/tfjs @tensorflow/tfjs-backend-webgpu
```

## Usage

For a complete guide, see our [Get Started section on
ai.google.dev](https://ai.google.dev/edge/litert/web/get_started).

### Setup

Before you can run a model, you must initialize both TensorFlow.js and
LiteRT.js. To enable efficient GPU tensor conversion, you must also configure
LiteRT.js to use the same WebGPU device as the TFJS WebGPU backend.

```typescript
import {loadLiteRt, liteRt} from '@litertjs/core';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

// Initialize TFJS WebGPU backend
await tf.setBackend('webgpu');

// Initialize LiteRT.js's Wasm files.
// These files are located in `node_modules/@litertjs/core/wasm/`
// and need to be served by your web server.
await loadLiteRt('/path/to/wasm/directory/');

// Make LiteRT use the same GPU device as TFJS for efficient tensor conversion.
// This must be run before loading a LiteRT model.
const backend = tf.backend() as WebGPUBackend;
liteRt.setWebGpuDevice(backend.device);
```

### Running a Model with TFJS Tensors

Once set up, you can use the `runWithTfjsTensors` function to wrap a LiteRT
`model.run` call. This function handles the conversion of TFJS input tensors to
LiteRT tensors and converts the LiteRT output tensors back into TFJS tensors.

```typescript
// Assumes the prior setup code has already been run.
import {runWithTfjsTensors} from '@litertjs/tfjs-interop';
import {loadAndCompile} from '@litertjs/core';
import * as tf from '@tensorflow/tfjs';

const model = await loadAndCompile(
  '/path/to/your/model/torchvision_mobilenet_v2.tflite',
  {accelerator: 'webgpu'}, // or 'wasm' for CPU.
);

// You can inspect the model's expected inputs and outputs.
console.log(model.getInputDetails());
console.log(model.getOutputDetails());

// Create a random TFJS tensor for input.
const input = tf.randomUniform([1, 3, 224, 224]);

// `runWithTfjsTensors` accepts a single tensor, an array of tensors,
// or a map of tensors by name.

// 1. Pass a single tensor
let results = runWithTfjsTensors(model, input);

// The result is an array of TFJS tensors.
await results[0].data();
results[0].print();
results[0].dispose();

// 2. Pass an array of tensors
results = runWithTfjsTensors(model, [input]);
await results[0].data();
results[0].print();
results[0].dispose();

// 3. Pass a map of inputs by name.
// Find the input tensor's name from `model.getInputDetails()`:
let resultsObject = runWithTfjsTensors(model, {
  'serving_default_args_0:0': input,
});

// The output is a Record<string, tf.Tensor>
// Find the output name from `model.getOutputDetails()`:
const result = resultsObject['StatefulPartitionedCall:0'];
await result.data();
result.print();
result.dispose();

// You can also run a specific model signature. Find available
// signatures from `model.signatures`.
console.log(model.signatures); // e.g., { 'serving_default': SignatureRunner }

// Pass the signature name as the second argument.
results = runWithTfjsTensors(model, 'serving_default', input);
await results[0].data();
results[0].print();
results[0].dispose();

// Or pass the signature object directly.
const signature = model.signatures['serving_default'];
results = runWithTfjsTensors(signature, input);
await results[0].data();
results[0].print();
results[0].dispose();
```