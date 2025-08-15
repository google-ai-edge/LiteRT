# LiteRT.js

Web runtime for [LiteRT](https://ai.google.dev/edge/litert) ([previously
TFLite](https://developers.googleblog.com/en/tensorflow-lite-is-now-litert/)),
Google's open-source high performance runtime for on-device AI.

LiteRT.js runs arbitrary LiteRT (.tfite) models that can perform a variety of
tasks, such as image segmentation, text classification, pose estimation, and
more, directly in the browser. You can find pretrained models on
[Kaggle](https://www.kaggle.com/models?framework=tfLite) or
[HuggingFace](https://huggingface.co/models?library=tflite&sort=trending), or
you can [convert your own from
PyTorch](https://github.com/google-ai-edge/ai-edge-torch).

## Features

-   Load and run a LiteRT (.tflite) model.
-   Run with WebGPU acceleration on supported browsers.
-   Run with ([XNNPack](https://github.com/google/XNNPACK))-accelerated CPU
    kernels on any browser.
-   Slot into existing [TensorFlow.js](https://github.com/tensorflow/tfjs)
    pipelines as a replacement for [TFJS Graph
    Models](https://js.tensorflow.org/api/latest/#loadGraphModel).

### Limitations

-   Large models may fail to load, since WebAssembly CPU memory for the runtime
is limited to 2GB.
-   Not all models are supported on WebGPU. You may need to replace some operations or
change some data types to get them working. The [Model Tester](#model-tester) can help check if a
    model will run on GPU and if the outputs match CPU.
-   LiteRT.js is just one piece of an ML pipeline -- it runs ML models but
    doesn't implement pre- and post-processing; those are left to the user to
    either implement themselves or obtain from libraries like TensorFlow.js. If
    you're interested in complete model pipelines, see [MediaPipe's Suite of Web
    Solutions](https://mediapipe-studio.webapps.google.com/home).

## Usage

For a complete guide, see our docs at
[ai.google.dev/edge/litert/web](https://ai.google.dev/edge/litert/web).

The following code snippet loads LiteRT.js, loads the MobileNetV2 model, and
runs it on a fake input image tensor. 

```typescript
import {loadLiteRt, loadAndCompile} from '@litertjs/core';

// Initialize LiteRT.js's Wasm files
// Wasm files are located in `node_modules/@litertjs/core/wasm/`.
await loadLiteRt('your/path/to/wasm');

const model = await loadAndCompile(
  '/torchvision_mobilenet_v2.tflite',
  {accelerator: 'webgpu'},
);

const inputTypedArray = new Float32Array(1*3*224*224);
const inputTensor = new Tensor(inputTypedArray, [1, 3, 224, 224]);
const gpuTensor = await inputTensor.moveTo('webgpu');

const results = model.run(gpuTensor);
gpuTensor.delete();

const result = results[0].moveTo('cpu');
console.log(result.toTypedArray());
result.delete();
```

## Usage with TFJS

For a complete guide, see our [Get Started section on
ai.google.dev](https://ai.google.dev/edge/litert/web/get_started).

LiteRT.js integrates well with existing TensorFlow.js pipelines. It requires
some setup code to use the same WebGPU device, after which tensors can be
converted between LiteRT.js and TFJS.

```typescript
import {loadLiteRt} from '@litertjs/core';
import {
  runWithTfjsTensors,
} from '@litertjs/tfjs-interop';

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

// Initialize TFJS WebGPU backend
await tf.setBackend('webgpu');

// Initialize LiteRT.js's Wasm files
// Wasm files are located in `node_modules/@litertjs/core/wasm/`.
await loadLiteRt('your/path/to/wasm');

// Make LiteRt use the same GPU device as TFJS (for tensor conversion)
const backend = tf.backend() as WebGPUBackend;
// Make sure to run this before loading a LiteRT model.
liteRt.setWebGpuDevice(backend.device);
```

Use the utility package, `@litertjs/tfjs-interop`, to convert between TFJS and
LiteRT tensors. The easiest way to do so is with the `runWithTfjsTensors`
function that wraps a LiteRT `model.run` call to use TFJS inputs and outputs:

```typescript
// Assumes the prior setup code has already been run.
import {runWithTfjsTensors} from '@litertjs/tfjs-interop';
import {loadAndCompile} from '@litertjs/core';

const model = await loadAndCompile(
  '/torchvision_mobilenet_v2.tflite',
  {accelerator: 'webgpu'}, // or 'wasm' for CPU.
);

// Show the input shapes that the model expects. You can also use the
// LiteRT Model Tester.
console.log(model.getInputDetails());
console.log(model.getOutputDetails());

// In this case, we know the model expects a 1x3x224x224 tensor.
// Example random input.
const input = tf.randomUniform([1, 3, 224, 224]);

// The model can be run in a few different ways. These are all equivalent to
// each other.

// You can pass a single tensor to the model
let results = runWithTfjsTensors(model, input);

// Results[0] contains the single output tensor from running the model.
await results[0].data();
results[0].print();
results[0].dispose();

// You can also pass an array of inputs
results = runWithTfjsTensors(model, [input]);

// The output type is still an array, as in the single tensor case.
await results[0].data();
results[0].print();
results[0].dispose();

// You can also pass inputs by name by passing an Object of inputs.
// Find the input tensor's name from `model.getInputDetails()`:
let resultsObject = runWithTfjsTensors(model, {
  'serving_default_args_0:0': input,
});

// The outputs are a `Record<string, litert.Tensor>`
// Find the output name from `model.getOutputDetails()`:
const result = resultsObject['StatefulPartitionedCall:0'];
await result.data();
result.print();
result.dispose();

// All of these methods of calling a model also work when calling a signature
// of a model. You can find the signatures a model contains by checking
// `model.signatures` or using the LiteRT Model Tester.
//
// (The base signature of a model is not in this Record. Just use the model
// object itself).

console.log(model.signatures); // { 'serving_default': SignatureRunner }

results = runWithTfjsTensors(model, 'serving_default', input);
await results[0].data();
results[0].print();
results[0].dispose();

// You can also pass the signature directly if you prefer.
results = runWithTfjsTensors(model.signatures['serving_default'], input);
await results[0].data();
results[0].print();
results[0].dispose();
```

## Model Tester

LiteRT.js provides a model tester webapp that can check if a `.tflite` model is
supported and benchmark it on CPU / GPU. You can install it with `npm i
@litertjs/model-tester` and run it with `npx model-tester`.

## Development

Run `npm install` and then `npm run build` in the same directory as this README
file to build everything. Then, to serve individual demos, `cd` to their
directory under `demos/` or `apps/` and run `npm run dev`.

To run the tests, run `npm run build && npm run test` in this directory. Then,
visit [http://localhost:8888](http://localhost:8888) to run the tests in your
browser. Your browser must support WebGPU (Chromium recommended).
