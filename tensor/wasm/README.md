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
-   `jit(func, options)`: Creates a JIT-compiled mathematical pipeline from a standard JS function.
-   `jitMulti(sigConfigs, options)`: Compiles multiple state-sharing subgraphs into a single multi-signature runner.

## JIT Compilation Mode

LiteRT WASM supports a high-performance **JIT (Just-In-Time) Compilation Mode** via `litert.jit()` and `litert.jitMulti()`. This allows you to write standard, readable mathematical JavaScript functions that look like normal eager code, but are compiled into optimized execution pipelines behind the scenes.

### How JIT Works
1. **Lazy Tracing (First Execution)**: The JIT decorator intercepts the first call, creates placeholder "tracers" matching the argument shapes/types, executes your function under the hood to trace the DAG, and compiles an optimized `createGraphRunner`.
2. **Cached Fast-Path Execution (Subsequent Runs)**: On all subsequent invocations, compilation is skipped entirely. The arguments are dynamically bound zero-copy to the compiled runner, executing immediately on the hardware accelerator.

### 1. Single JIT Pipeline (`jit()`)

Ideal for stateless, straight-line execution flows like pre-processing:

```javascript
// Define and wrap a function in litert.jit()
const preprocess = litert.jit((img, scale, bias) => {
  // Traces Bilinear Resizing, Multiplication, and Addition
  const resized = img.resizeBilinear([256, 256], false, false);
  return resized.mul(scale).add(bias).relu();
}, { accelerators: litert.HwAccelerators.GPU });

// Execute dynamically (Zero-copy buffer sharing)
const outputTensor = await preprocess(imageTensor, scaleTensor, biasTensor);
```

### 2. Multi-Signature State Sharing (`jitMulti()`)

Ideal for state-sharing pipelines, such as Large Language Models with persistent key-value caches. By defining shared state variables inside JavaScript closures, both subgraphs share the same persistent GPU buffers:

```javascript
// Shared KV-Cache buffer allocated once on the GPU
const kvCache = litert.createTensor({ type: 'FP32', shape: [1, 32, 128, 128] });

const gemma3 = litert.jitMulti({
  prefill: {
    func: (inputTokens) => {
      // Traces writing to the shared kvCache
      kvCache.setData(...);
      return inputTokens.mul(10);
    },
    inputs: [{ type: 'FP32', shape: [1, 32] }] // Expected input signature
  },
  decode: {
    func: (nextToken) => {
      // Traces reading from the shared kvCache
      return nextToken.add(kvCache.sum([2]));
    },
    inputs: [{ type: 'FP32', shape: [1, 1] }]
  }
}, { accelerators: litert.HwAccelerators.GPU });

// Call the compiled endpoints directly
const prefillLogits = await gemma3.prefill(actualPrefillTokens);
const nextLogits    = await gemma3.decode(actualNextToken);
```

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
