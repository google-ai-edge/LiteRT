# LiteRT Tensor API WASM Binding Documentation

This document describes the public API for the LiteRT Tensor API WebAssembly
binding.

## Initialization

To use the LiteRT WASM binding, you need to load the generated Emscripten module
and the wrapper.

```html
<script src="tensor_wasm_internal.js"></script>
<script type="module">
  import { createLiteRT } from './tensor_wasm.js';

  const litert = await createLiteRT({
    preinitializedWebGPUDevice: device // Optional, for WebGPU support
  });
</script>
```

### `createLiteRT(options?: object): Promise<LiteRtWasmModule>`

Loads the LiteRT WASM module and returns an augmented module instance.

*   `options`: Options passed to the Emscripten module creator.

## Enums

### `TensorType`

Supported tensor data types.

*   `UNKNOWN`
*   `BOOL`
*   `I8`
*   `I32`
*   `FP32`

### `HwAccelerators`

Supported hardware accelerators.

*   `NONE`
*   `CPU`
*   `GPU`

## Tensor Creation

### `litert.createTensor(options: object): Tensor`

Creates a new Tensor with the specified options.

*   `options`:
    *   `name` (string, optional): Name of the tensor.
    *   `type` (string | TensorType, optional): Data type (e.g., 'FP32', 'I32').
    *   `shape` (number[], optional): Shape dimensions.

Example:
```javascript
const tensor = litert.createTensor({
  name: "input",
  type: "FP32",
  shape: [1, 256, 256, 3]
});
```

### `litert.createTensorWithData(data: TypedArray, type: string | TensorType, shape: number[], name?: string): Tensor`

Creates a new Tensor initialized with the provided data.

*   `data`: A typed array containing the initial data.
*   `type`: Data type (e.g., 'FP32', 'I32').
*   `shape`: Shape dimensions.
*   `name` (optional): Name of the tensor.

## Tensor Interface

The `Tensor` object represents a multi-dimensional array.

### Attributes and Accessors

*   `getName(): string`: Returns the name of the tensor.
*   `getType(): TensorType`: Returns the data type.
*   `getShape(): number[]`: Returns the shape dimensions.
*   `getData(): Promise<TypedArray>`: Returns a copy of the tensor data.
*   `getMutableData(): Promise<TypedArray>`: Returns a view of the WASM memory for direct manipulation.
*   `getWebGpuBuffer(): number`: Returns the WebGPU buffer handle (ID).
*   `setName(name: string): void`: Sets the name.
*   `setType(type: TensorType): void`: Sets the type.
*   `setShape(shape: number[]): void`: Sets the shape. Accepts standard JS arrays.
*   `setData(data: TypedArray): void`: Copies data from a JS typed array into the tensor.

## JIT Compilation Mode

LiteRT WASM supports a high-performance JIT (Just-In-Time) Compilation Mode. This allows you to write standard, eagerly structured JavaScript functions containing tensor operations, which are compiled into optimized hardware-accelerated execution runners behind the scenes.

### `litert.jit(func: Function, options?: object): Function`

Traces a standard JavaScript function and JIT-compiles it into an optimized Graph Runner.

*   `func`: The JavaScript function representing the operations: `(...inputs) => outputTensor`.
*   `options` (optional): Configuration mapping (e.g., `accelerators`).
*   **Returns**: A compiled asynchronous function with the same signature: `async (...inputs) => outputTensor`.

Example:
```javascript
const customPipeline = litert.jit((img, scale, bias) => {
  const resized = img.resizeBilinear([256, 256], false, false);
  return resized.mul(scale).add(bias).relu();
}, { accelerators: litert.HwAccelerators.GPU });

// First call: Traces, AOT compiles GraphRunner, and runs.
// Subsequent calls: Skips compile, binds arguments zero-copy, dispatches immediately.
const output = await customPipeline(actualImage, scaleTensor, biasTensor);
```

### `litert.jitMulti(sigConfigs: object, options?: object): object`

Compiles multiple related, state-sharing subgraphs into a single multi-signature Graph Runner. 

*   `sigConfigs`: A dictionary mapping signature names to configurations:
    *   If configuration is a **Function**: Defer compilation (requires calling `.compile(sampleInputs)` manually before execution).
    *   If configuration is an **Object**: Supply configuration upfront:
        *   `func`: The subgraph execution function.
        *   `inputs`: Array of expected input descriptions: `[{ type: TensorType | string, shape: number[] }]`.
*   `options` (optional): Configuration mapping (e.g., `accelerators`).
*   **Returns**: An object containing the compiled asynchronous functions as callable methods, plus a `.compile(sampleInputs)` helper method.

Example (KV-Cache State Sharing):
```javascript
const kvCache = litert.createTensor({ type: 'FP32', shape: [1, 32, 128, 128] });

const model = litert.jitMulti({
  prefill: {
    func: (tokens) => {
      kvCache.setData(...); // updates shared cache in closure
      return tokens.mul(10);
    },
    inputs: [{ type: 'FP32', shape: [1, 32] }]
  },
  decode: {
    func: (nextToken) => {
      return nextToken.add(kvCache.sum([2])); // reads shared cache
    },
    inputs: [{ type: 'FP32', shape: [1, 1] }]
  }
}, { accelerators: litert.HwAccelerators.GPU });

// Direct fast-path dispatching
const logits = await model.prefill(actualPrefillTokens);
```

## Eager Execution Mode

LiteRT WASM supports an intuitive Eager Execution Mode that evaluates tensor
operations instantly without requiring explicit runner compilation.

> [!WARNING]
> **Performance Impact**: Eager execution is designed for rapid prototyping,
debugging, and interactive exploration. It significantly reduces performance
compared to executing a complete computation graph with a graph runner
(e.g., `createGraphRunner` or `createModelRunner`). Eager mode compiles and
dispatches small subgraphs on the fly for individual operations, preventing
graph-level optimizations like operator fusion, buffer sharing, and static
memory planning, while introducing repeated runtime overhead. For production
workloads, always construct your complete computation graph upfront and execute
it using a graph runner.

### `litert.setEagerMode(enable: boolean): void`

Enables or disables automatic eager execution mode globally.

### `tensor.evaluate(): Promise<void>`

Instantly evaluates and executes this tensor's producer graph in eager mode.
Useful when global eager mode is disabled.

## Operations

Tensors support chaining of operations. Most operations return a new `Tensor`
representing the result.

### Unary & Activation Operations

*   `abs()`: Computes the absolute value of each element.
*   `relu()`: Computes rectified linear: `max(x, 0)`.
*   `relu6()`: Computes rectified linear 6: `min(max(x, 0), 6)`.
*   `elu()`: Computes exponential linear: `exp(x) - 1` if x < 0, `x` otherwise.
*   `hardSwish()`: Computes hard swish activation.
*   `logSoftmax()`: Computes log softmax.
*   `logistic()`: Computes sigmoid/logistic: `1 / (1 + exp(-x))`.
*   `neg()`: Computes numerical negative: `-x`.
*   `sqrt()`: Computes square root.
*   `cos()`: Computes cosine.
*   `sin()`: Computes sine.
*   `exp()`: Computes exponential: `e^x`.
*   `log()`: Computes natural logarithm.
*   `ceil()`: Computes ceiling: smallest integer >= x.
*   `floor()`: Computes floor: largest integer <= x.
*   `sign()`: Computes sign: -1 if x < 0, 0 if x == 0, 1 if x > 0.
*   `round()`: Rounds to nearest integer.
*   `logicalNot()`: Computes logical NOT (for boolean tensors).

Example:
```javascript
const input = litert.createTensor({ type: 'FP32', shape: [1, 5] });
input.setData(new Float32Array([-1, 0, 1, 2, 3]));

const absOutput = input.abs();
// absOutput data will be [1, 0, 1, 2, 3]

const reluOutput = input.relu();
// reluOutput data will be [0, 0, 1, 2, 3]
```

### Binary & Logic Operations

*   `add(other: Tensor)`: Adds elements of two tensors.
*   `sub(other: Tensor)`: Subtracts elements of two tensors.
*   `mul(other: Tensor)`: Multiplies elements of two tensors.
*   `div(other: Tensor)`: Divides elements of two tensors.
*   `pow(other: Tensor)`: Raises elements of first tensor to power of elements of second tensor.
*   `minimum(other: Tensor)`: Computes element-wise minimum.
*   `maximum(other: Tensor)`: Computes element-wise maximum.
*   `less(other: Tensor)`: Returns boolean tensor where first < second.
*   `greater(other: Tensor)`: Returns boolean tensor where first > second.
*   `lessEqual(other: Tensor)`: Returns boolean tensor where first <= second.
*   `greaterEqual(other: Tensor)`: Returns boolean tensor where first >= second.
*   `equal(other: Tensor)`: Returns boolean tensor where first == second.
*   `notEqual(other: Tensor)`: Returns boolean tensor where first != second.
*   `logicalAnd(other: Tensor)`: Computes logical AND.
*   `logicalOr(other: Tensor)`: Computes logical OR.
*   `floorDiv(other: Tensor)`: Computes floor division.
*   `floorMod(other: Tensor)`: Computes floor modulo.

Example:
```javascript
const a = litert.createTensor({ type: 'FP32', shape: [1, 3] });
a.setData(new Float32Array([1, 2, 3]));

const b = litert.createTensor({ type: 'FP32', shape: [1, 3] });
b.setData(new Float32Array([4, 5, 6]));

const sum = a.add(b);
// sum data will be [5, 7, 9]

const isLess = a.less(b);
// isLess data will be a boolean tensor
```

### Reduction & Shaping Operations

*   `sum(axes: number[], keepDims: boolean)`: Computes the sum of elements across dimensions specified in `axes`.
*   `reduceMax(axes: number[], keepDims: boolean)`: Computes the maximum of elements across dimensions specified in `axes`.
*   `mean(axes: number[], keepDims: boolean)`: Computes the mean of elements across dimensions specified in `axes`.
*   `reshape(shape: number[])`: Returns a tensor with a new shape but same data.
*   `squeeze(dims: number[])`: Removes dimensions of size 1 from the shape.
*   `expandDims(axis: number)`: Inserts a dimension of size 1 at the specified `axis`.

Example:
```javascript
const input = litert.createTensor({ type: 'FP32', shape: [2, 3] });
input.setData(new Float32Array([1, 2, 3, 4, 5, 6]));

const sumAll = input.sum([0, 1], false);
// sumAll data will be [21]

const sumRow = input.sum([1], true);
// sumRow shape will be [2, 1]

const reshaped = input.reshape([6]);
// reshaped shape will be [6]
```

### Spatial & Vision Operations

*   `resizeBilinear(size: number[], alignCorners: boolean, halfPixelCenters: boolean)`: Resizes image using bilinear interpolation to target `size` `[height, width]`.
*   `resizeNearestNeighbor(size: number[], alignCorners: boolean, halfPixelCenters: boolean)`: Resizes image using nearest neighbor interpolation.
*   `gather(indices: Tensor, axis: number)`: Gathers slices from `self` along `axis` according to `indices`.
*   `argMax(axis: number)`: Returns the indices of the maximum values along `axis`.

Example:
```javascript
const image = litert.createTensor({ type: 'FP32', shape: [1, 2, 2, 1] });
image.setData(new Float32Array([1, 2, 3, 4]));

const resized = image.resizeBilinear([4, 4], false, false);
// resized shape will be [1, 4, 4, 1]

const winningClasses = image.argMax(-1);
// winningClasses returns indices of max values along last axis
```

## Runners

### `createGraphRunner(inputs: object, outputs: object, accelerators?: number | HwAccelerators): LiteRTRunner`

Creates a runner for a defined graph of operations.

*   `inputs`: Map of string names to input Tensors.
*   `outputs`: Map of string names to output Tensors.
*   `accelerators` (optional): Bitmask or `HwAccelerators` enum value. Defaults to both CPU and GPU enabled.

Example:
```javascript
const rawImage = litert.createTensor({ name: "raw_image", type: "FP32", shape: [1, 512, 512, 3] });
const scaleTensor = litert.createTensor({ name: "scale", type: "FP32", shape: [1] });
const offsetTensor = litert.createTensor({ name: "offset", type: "FP32", shape: [1] });

const resized = rawImage.resizeBilinear([256, 256], false, false);
const preprocessed = resized.mul(scaleTensor).add(offsetTensor);

const preRunner = litert.createGraphRunner(
    {"raw_image": rawImage, "scale": scaleTensor, "offset": offsetTensor},
    {"normalized_image": preprocessed}
);

// Execute the graph
await preRunner.run();
const outputTensor = preRunner.getOutput("normalized_image");
```

### `createModelRunner(buffer: Uint8Array, accelerators?: number | HwAccelerators): LiteRTRunner`

Creates a runner for a TFLite model loaded from a buffer.

*   `buffer`: The model bytes.
*   `accelerators` (optional): Bitmask or `HwAccelerators` enum value. Defaults to both CPU and GPU enabled.

Example:
```javascript
const response = await fetch('model.tflite');
const modelBytes = await response.arrayBuffer();

const coreRunner = litert.createModelRunner(
    new Uint8Array(modelBytes),
    litert.HwAccelerators.CPU | litert.HwAccelerators.GPU
);

// Get input by index using the unified method
const inputTensor = coreRunner.getInput(0);
// Set input data...

await coreRunner.run();
// Get output by index using the unified method
const outputTensor = coreRunner.getOutput(0);
```

### `createMultiSignatureRunner(signatures: object, accelerators?: number | HwAccelerators): Promise<LiteRTRunner>`

Creates a runner for a model with multiple signatures, built from graph outputs.

*   `signatures`: A map where keys are signature names, and values are objects containing an `outputs` array of `Tensor` handles.
*   `accelerators` (optional): Bitmask or `HwAccelerators` enum value. Defaults to GPU WebGPU.

Example:
```javascript
const signatures = {
  "prefill": {
    outputs: [prefillGraph.topk_values, prefillGraph.topk_indices]
  },
  "decode": {
    outputs: [decodeGraph.topk_values, decodeGraph.topk_indices]
  }
};

const runner = await litert.createMultiSignatureRunner(signatures, litert.HwAccelerators.GPU_WEBGPU);
```

### Runner Interface

#### `LiteRTRunner`

*   `run(): Promise<boolean>`: Executes the model or graph.
*   `getInput(nameOrIndex: string | number): Tensor`: Gets input tensor by name or index. Access by index is only supported by model runners.
*   `getOutput(nameOrIndex: string | number): Tensor`: Gets output tensor by name or index. Access by index is only supported by model runners.
*   `setInput(name: string, tensor: Tensor): boolean`: Sets input tensor.
*   `setInputBinary(name: string, array: Uint8Array): boolean`: Sets input data directly.
*   `delete(): void`: Deallocates the runner.

#### `LiteRtMultiSignatureRunner`

*   `run(signatureName: string): Promise<boolean>`: Executes the model for the specified signature.
*   `getInput(signatureName: string, name: string): Tensor`: Gets input tensor by signature and name.
*   `getOutput(signatureName: string, name: string): Tensor`: Gets output tensor by signature and name.
*   `setInput(signatureName: string, name: string, tensor: Tensor): boolean`: Sets input tensor for the specified signature.
*   `setInputBinary(signatureName: string, name: string, array: Uint8Array): boolean`: Sets input data directly for the specified signature.
*   `getOutputWebGpuBuffer(signatureName: string, name: string): number`: Returns the WebGPU buffer handle for the specified output signature and name.
*   `getInputWebGpuBuffer(signatureName: string, name: string): number`: Returns the WebGPU buffer handle for the specified input signature and name.
*   `delete(): void`: Deallocates the runner.

## Chaining Models and Graphs

LiteRT allows you to chain multiple runners together, for example, to combine
pre-processing, inference, and post-processing.

### CPU Data Transfer (Fallback)

The simplest way to chain runners is to read data from the output of one runner
and write it to the input of the next using CPU memory.

```javascript
// Execute first runner
await runner1.run();

// Get output tensor and its data
const outputTensor = runner1.getOutput("output_name");
const data = await outputTensor.getData();

// Get input tensor of second runner and set data
const inputTensor = runner2.getInput("input_name");
inputTensor.setData(data);

// Execute second runner
await runner2.run();
```

### Zero-Copy WebGPU Chaining and Buffer Sharing

For maximum performance when using WebGPU, you can avoid copying data between
JS and WASM heap by sharing or copying GPU buffers directly.

#### Option 1: Direct GPU Copy
If you have access to the underlying `GPUBuffer` objects, you can perform a
GPU-to-GPU copy:

```javascript
// Execute first runner
await runner1.run();

const srcTensor = runner1.getOutput("output_name");
const dstTensor = runner2.getInput("input_name");

// Get raw WebGPU buffer handles (integers)
const srcBufferId = srcTensor.getWebGpuBuffer();
const dstBufferId = dstTensor.getWebGpuBuffer();

// Map handles to JS GPUBuffer objects
const srcBuffer = litert.WebGPU.getJsObject(srcBufferId);
const dstBuffer = litert.WebGPU.getJsObject(dstBufferId);

// Perform GPU-to-GPU copy using WebGPU API
const commandEncoder = device.createCommandEncoder();
commandEncoder.copyBufferToBuffer(srcBuffer, 0, dstBuffer, 0, sizeInBytes);
device.queue.submit([commandEncoder.finish()]);

// Execute second runner
await runner2.run();
```

#### Option 2: Buffer Sharing via `setInput`
A more idiomatic way in LiteRT WASM is to pass the output `Tensor` of one runner
directly as the input to another runner using `setInput`. This shares the
underlying WebGPU buffer without any copying:

```javascript
const outputTensor = runner1.getOutput("output_name");
// Shares the WebGPU buffer directly!
runner2.setInput("input_name", outputTensor);
await runner2.run();
```
This reduces the transfer time to absolute zero.
