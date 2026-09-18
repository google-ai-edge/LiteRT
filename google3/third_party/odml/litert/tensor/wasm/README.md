# LiteRT Tensor WebAssembly Bindings

This directory contains the WebAssembly bindings for the LiteRT Tensor API,
enabling high-performance machine learning inference in the browser with WebGPU
support.

## Features

-   **WebGPU Support**: Tensors can be backed by WebGPU buffers, allowing for GPU-accelerated inference.
-   **JSPI (JavaScript Promise Integration)**: Supports asynchronous execution and waiting for GPU work.
-   **Zero-Copy Optimization**: APIs to expose raw WebGPU buffer handles to JavaScript, enabling direct GPU-to-GPU copies between models and avoiding slow CPU readbacks.

## API Reference

### Core Classes

#### `Tensor`
Wraps a LiteRT Tensor.

-   `getName()`: Get tensor name.
-   `getType()`: Get tensor data type.
-   `getShape()`: Get tensor shape as an array.
-   `getData()`: Get tensor data (async, returns a copy).
-   `getMutableData()`: Get mutable tensor data (async, returns a copy).
-   `getWebGpuBuffer()`: Get raw WebGPU buffer handle (integer ID).
-   `setName(name)`: Set tensor name.
-   `setType(type)`: Set tensor type.
-   `setShape(shape)`: Set tensor shape.
-   `evaluate()`: Instantly evaluate and execute this tensor's producer graph in
      eager mode (async).

*Note: Many operations are available as methods on `Tensor` (e.g., `abs()`,
`relu6()`, `argMax()`, etc.).*

#### `LambdaModelRunner`
Runner for static models defined via functional graph.

-   `run()`: Execute the model (async).
-   `setInput(name, tensor)`: Set input tensor.
-   `setInputBinary(name, array)`: Set input data from a typed array.
-   `getInput(name)`: Get input tensor handle.
-   `getOutput(name)`: Get output tensor handle.

#### `LitertDynamicRunner`
Runner for dynamic models loaded from TFLite buffer.

-   `run()`: Execute the model (async).
-   `getInput(name)` / `getInputByIndex(index)`: Get input tensor handle.
-   `getOutput(name)` / `getOutputByIndex(index)`: Get output tensor handle.
-   `setInput(name, tensor)`: Set input tensor.
-   `setInputBinary(name, array)`: Set input data from a typed array.

### Global Functions

-   `createStaticLambdaRunner(inputs, outputs)`: Create a static runner.
-   `createDynamicRunnerFromBuffer(buffer, accelerators)`: Create a dynamic runner from a TFLite model buffer.
-   `createMultiSignatureRunner(signatures, accelerator)`: Create a multi-signature runner from a set of graph outputs.
-   `setEagerMode(enable)`: Enable or disable automatic eager execution mode globally.

## Eager Execution Mode

> [!WARNING]
> **Performance Impact**: Eager execution is designed for rapid prototyping,
debugging, and interactive exploration. It significantly reduces performance
compared to executing a complete computation graph with a graph runner
(e.g., `createGraphRunner` or `createModelRunner`). Eager mode compiles and
dispatches small subgraphs on the fly for individual operations, which
prevents graph-level optimizations like operator fusion and static memory
planning, while introducing repeated runtime overhead. For production
deployment, always compile your end-to-end graph into a runner.

LiteRT WASM supports an intuitive **Eager Execution Mode** that evaluates tensor
operations instantly without requiring explicit runner compilation or graph
management.

### Enabling Eager Mode
You can enable eager mode globally at the top of your script:
```javascript
litert.setEagerMode(true);
```

### Usage Example
Once enabled, mathematical operations automatically compile and execute their
underlying subgraphs on the fly:

```javascript
const shape = [2, 2];
const t1 = litert.createTensorWithData([1.0, 2.0, 3.0, 4.0], litert.TensorType.FP32, shape, "t1");
const t2 = litert.createTensorWithData([10.0, 20.0, 30.0, 40.0], litert.TensorType.FP32, shape, "t2");

// Executes instantly on CPU or WebGPU! No runner needed.
const result = t1.add(t2);

const data = await result.getData();
console.log(data); // Float32Array([11, 22, 33, 44])
```

### Manual Evaluation (`evaluate()`)
If `setEagerMode` is false (default static graph mode), you can still evaluate
any specific tensor node on demand using `evaluate()`:
```javascript
const lazyNode = t1.mul(t2);
await lazyNode.evaluate(); // Compiles and runs just this node's subgraph!
```

## API Extensions

We added the following extensions to support high-performance pipelines:

### `TensorHandle::getWebGpuBuffer()`

Returns the integer ID of the internal `WGPUBuffer`. In JavaScript, you can
retrieve the actual `GPUBuffer` object using:

```javascript
const gpuBufferId = tensor.getWebGpuBuffer();
// Assuming 'litert' is your module instance
const gpuBuffer = litert.WebGPU.getJsObject(gpuBufferId);
```

### `LambdaModelRunner::getInput(name)`

Allows retrieving the `TensorHandle` for a specific input of a
`LambdaModelRunner` (static runner). This was previously only available for
`DynamicRunner`.

## Performance Tips

### Prefer Graph Runners Over Eager Mode
While Eager Execution Mode is convenient for debugging and step-by-step experimentation, it reduces performance significantly. Executing operations eagerly compiles small subgraphs on the fly and misses out on critical graph-level optimizations such as operator fusion, buffer sharing, and static memory allocation. For production workloads, always construct your complete computation graph upfront and execute it using `createGraphRunner` or `createModelRunner`.

### Zero-Copy WebGPU Transfers and Buffer Sharing

To avoid the high overhead of copying data between JS and WASM heap, use direct GPU copies or buffer sharing when passing data between models:

#### Option 1: Direct GPU Copy
If you have access to the underlying `GPUBuffer` objects, you can copy data directly on the GPU:
```javascript
const commandEncoder = device.createCommandEncoder();
commandEncoder.copyBufferToBuffer(srcGpuBuffer, 0, dstGpuBuffer, 0, size);
device.queue.submit([commandEncoder.finish()]);
```

#### Option 2: Buffer Sharing via `setInput`
A more idiomatic way in LiteRT WASM is to pass the output `TensorHandle` of one runner directly as the input to another runner using `setInput`. This shares the underlying WebGPU buffer without any copying:
```javascript
const outputTensor = runnerA.getOutput("output_name");
runnerB.setInput("input_name", outputTensor);
```
This reduces the transfer time to absolute zero.
