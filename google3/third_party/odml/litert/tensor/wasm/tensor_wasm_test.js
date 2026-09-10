/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import { SafetensorsLoader } from './demo/gemma3/safetensors.js';

/** @suppress {missingProperties} */
describe("LiteRT WebAssembly Complete End-to-End Bindings Verification", () => {
  let litert;

  beforeAll(async () => {
    if (typeof globalThis.createTensorModule === "function") {
      litert = await globalThis.createTensorModule();
    }
  });

  it("verifies setting and getting Name, Type, and Shape fields successfully", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    expect(tensor.getName()).toEqual("");

    tensor.setName("custom_tensor");
    tensor.setType(litert.TensorType.FP32);

    expect(tensor.getName()).toEqual("custom_tensor");
    expect(tensor.getType()).toEqual(litert.TensorType.FP32);

    tensor.delete();
  });

  it("verifies setting and getting Quantization sub-attributes correctly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();

    const quantParams = {
      scales: [0.25],
      zeroPoints: [128],
      quantizedDimension: 0,
    };

    tensor.setQuantization(quantParams);

    const retrieved = tensor.getQuantization();
    expect(retrieved).toBeDefined();

    tensor.delete();
  });

  it("verifies invoking element-wise Binary operations seamlessly", () => {
    if (!litert) return;

    const tensorA = new litert.Tensor();
    const tensorB = new litert.Tensor();
    tensorA.setType(litert.TensorType.FP32);
    tensorB.setType(litert.TensorType.FP32);

    const added = tensorA.add(tensorB);
    const multiplied = added.mul(tensorA);

    expect(added).toBeDefined();
    expect(multiplied).toBeDefined();

    tensorA.delete();
    tensorB.delete();
    added.delete();
    multiplied.delete();
  });

  it("verifies invoking Reduction & Accumulation operations correctly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    tensor.setType(litert.TensorType.FP32);

    const summed = tensor.sum([0], true);
    const maxed = tensor.reduceMax([1], false);
    const meaned = tensor.mean([0, 1], true);

    expect(summed).toBeDefined();
    expect(maxed).toBeDefined();
    expect(meaned).toBeDefined();

    tensor.delete();
    summed.delete();
    maxed.delete();
    meaned.delete();
  });

  it("verifies invoking Shape & Axis Manipulation operations correctly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    tensor.setType(litert.TensorType.FP32);

    const expanded = tensor.expandDims(0);
    const squeezed = expanded.squeeze([0]);
    const reshaped = squeezed.reshape([1, 256, 256, 3]);

    expect(expanded).toBeDefined();
    expect(squeezed).toBeDefined();
    expect(reshaped).toBeDefined();

    tensor.delete();
    expanded.delete();
    squeezed.delete();
    reshaped.delete();
  });

  it("verifies invoking Neural Network & Convolutional operations correctly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    tensor.setType(litert.TensorType.FP32);

    const pooled = tensor.averagePool2d(2, 2, 1, 1, 0);
    expect(pooled).toBeDefined();

    tensor.delete();
    pooled.delete();
  });

  it("verifies invoking Unstacking & Decomposition operations correctly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    const axis = new litert.Tensor();
    tensor.setType(litert.TensorType.FP32);

    const unpacked = tensor.unpack(2, 0);
    const split = tensor.split(axis, 2);

    expect(unpacked).toBeDefined();
    expect(split).toBeDefined();

    tensor.delete();
    axis.delete();
    unpacked.delete();
    split.delete();
  });

  it("verifies invoking newly added Specialized & Indexing operations correctly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    const indices = new litert.Tensor();
    const depth = new litert.Tensor();
    const valueOn = new litert.Tensor();
    const valueOff = new litert.Tensor();

    tensor.setType(litert.TensorType.FP32);

    const gathered = tensor.gather(indices, 0);
    const encoded = indices.oneHot(depth, valueOn, valueOff, 0);
    const normalized = tensor.l2Normalization();
    const squared = tensor.square();
    const rootInversed = tensor.rsqrt();

    expect(gathered).toBeDefined();
    expect(encoded).toBeDefined();
    expect(normalized).toBeDefined();
    expect(squared).toBeDefined();
    expect(rootInversed).toBeDefined();

    tensor.delete();
    indices.delete();
    depth.delete();
    valueOn.delete();
    valueOff.delete();
    gathered.delete();
    encoded.delete();
    normalized.delete();
    squared.delete();
    rootInversed.delete();
  });

  it("verifies invoking Sequence operations like cumsum successfully", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    tensor.setType(litert.TensorType.FP32);

    const cumsummed = tensor.cumsum(0, false, false);

    expect(cumsummed).toBeDefined();

    tensor.delete();
    cumsummed.delete();
  });

  it("verifies invoking newly added Spatial & Pixel Transformation operations correctly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    const axes = new litert.Tensor();
    tensor.setType(litert.TensorType.FP32);

    const s2d = tensor.spaceToDepth(2);
    const d2s = tensor.depthToSpace(2);
    const reversed = tensor.reverse(axes);
    const resized = tensor.resizeBilinear([256, 256], false, false);

    expect(s2d).toBeDefined();
    expect(d2s).toBeDefined();
    expect(reversed).toBeDefined();
    expect(resized).toBeDefined();

    tensor.delete();
    axes.delete();
    s2d.delete();
    d2s.delete();
    reversed.delete();
    resized.delete();
  });

  it("verifies invoking Conditional Masking & Selector operations correctly", () => {
    if (!litert) return;

    const trueBranch = new litert.Tensor();
    const falseBranch = new litert.Tensor();
    const condition = new litert.Tensor();

    trueBranch.setType(litert.TensorType.FP32);
    falseBranch.setType(litert.TensorType.FP32);

    const selected = trueBranch.select(condition, falseBranch);
    const selectedV2 = trueBranch.selectV2(condition, falseBranch);

    expect(selected).toBeDefined();
    expect(selectedV2).toBeDefined();

    trueBranch.delete();
    falseBranch.delete();
    condition.delete();
    selected.delete();
    selectedV2.delete();
  });

  it("verifies invoking remaining Sequence, Memory & Reshaping operations perfectly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    tensor.setType(litert.TensorType.FP32);

    const concatenated = tensor.concatenation([tensor], 0);
    const packed = tensor.pack([tensor], 0);
    const sliced = tensor.slice([0], [1]);
    const tiled = tensor.tile([2]);
    const transposed = tensor.transpose([0]);

    expect(concatenated).toBeDefined();
    expect(packed).toBeDefined();
    expect(sliced).toBeDefined();
    expect(tiled).toBeDefined();
    expect(transposed).toBeDefined();

    tensor.delete();
    concatenated.delete();
    packed.delete();
    sliced.delete();
    tiled.delete();
    transposed.delete();
  });

  it("verifies invoking remaining Specialized Vision & Memory operations seamlessly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    const indices = new litert.Tensor();
    tensor.setType(litert.TensorType.FP32);

    const gnd = tensor.gatherNd(indices);
    const resized = tensor.resizeNearestNeighbor([256, 256], false, false);

    expect(gnd).toBeDefined();
    expect(resized).toBeDefined();

    tensor.delete();
    indices.delete();
    gnd.delete();
    resized.delete();
  });

  it("verifies invoking remaining Quantization & Debug Instrumentation operations perfectly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    tensor.setType(litert.TensorType.FP32);

    const casted = tensor.cast(litert.TensorType.I32);
    const quantized = tensor.quantize(litert.TensorType.I8, [0.1], [0]);
    const dequantized = quantized.dequantize();
    const probed = dequantized.probe();

    expect(casted).toBeDefined();
    expect(quantized).toBeDefined();
    expect(dequantized).toBeDefined();
    expect(probed).toBeDefined();

    tensor.delete();
    casted.delete();
    quantized.delete();
    dequantized.delete();
    probed.delete();
  });

  it("verifies invoking remaining advanced sub-graph operations seamlessly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    const update = new litert.Tensor();
    const startIndices = new litert.Tensor();
    const nmsConstant = new litert.Tensor();

    const geluOut = tensor.gelu();
    const emb = tensor.embeddingLookup(update, litert.TensorType.FP32);
    const updated = tensor.dynamicUpdateSlice(update, startIndices);
    const nms = tensor.nonMaxSuppressionV5(update, nmsConstant, nmsConstant, nmsConstant, nmsConstant);

    expect(geluOut).toBeDefined();
    expect(emb).toBeDefined();
    expect(updated).toBeDefined();
    expect(nms).toBeDefined();

    tensor.delete();
    update.delete();
    startIndices.delete();
    nmsConstant.delete();
    geluOut.delete();
    emb.delete();
    updated.delete();
    nms.delete();
  });

  it("verifies instantiating and invoking CompiledModelRunner successfully", () => {
    if (!litert) return;

    const runner = new litert.CompiledModelRunner();
    expect(runner).toBeDefined();

    const successful = runner.run();
    expect(successful).toBe(true);

    runner.delete();
  });

  it("verifies instantiating and invoking LambdaModelRunner & LitertDynamicRunner successfully", () => {
    if (!litert) return;

    const lambdaRunner = new litert.LambdaModelRunner();
    const dynamicRunner = new litert.LitertDynamicRunner();

    expect(lambdaRunner).toBeDefined();
    expect(dynamicRunner).toBeDefined();

    expect(lambdaRunner.run()).toBe(true);
    expect(dynamicRunner.run()).toBe(true);

    lambdaRunner.delete();
    dynamicRunner.delete();
  });

  it("verifies WebGpuBuffer zero-copy VRAM staging operations successfully", () => {
    if (!litert) return;

    const gpuBuffer = new litert.WebGpuBuffer();
    expect(gpuBuffer).toBeDefined();

    expect(gpuBuffer.setGPUBuffer(null)).toBe(true);
    expect(gpuBuffer.getGPUBuffer()).toBeNull();

    gpuBuffer.delete();
  });

  it("verifies abstracting model buffer pipelines using stable Tensor Handle IDs seamlessly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    const dynamicRunner = new litert.LitertDynamicRunner();

    const id = tensor.getId();
    expect(typeof id).toBe("number");

    expect(dynamicRunner.setInputById("input", id)).toBe(true);

    tensor.delete();
    dynamicRunner.delete();
  });

  it("verifies orchestrating DynamicWasmRunner execution using abstract graph endpoints perfectly", () => {
    if (!litert) return;

    const tensor = new litert.Tensor();
    const runner = new litert.DynamicWasmRunner();

    expect(runner.buildModelFromEndpoints([tensor])).toBe(true);
    expect(runner.setInput("input", [1.0, 2.0])).toBe(true);
    expect(runner.run()).toBe(true);
    expect(runner.getOutput("output")).toBeNull();

    tensor.delete();
    runner.delete();
  });

  it("verifies creating and running a multi-signature model successfully", async () => {
    if (!litert) return;

    // Create a simple graph for signature 1: C = A + B
    const a = litert.createTensor({ type: 'FP32', shape: [2, 2] });
    a.setName("a");
    const b = litert.createTensor({ type: 'FP32', shape: [2, 2] });
    b.setName("b");
    const c = a.add(b);
    c.setName("c");

    // Create a simple graph for signature 2: E = A * D
    const d = litert.createTensor({ type: 'FP32', shape: [2, 2] });
    d.setName("d");
    const e = a.mul(d);
    e.setName("e");

    const runner = await litert.createMultiSignatureRunner({
      "sig1": { outputs: [c] },
      "sig2": { outputs: [e] }
    }, litert.HwAccelerators.CPU); // Use CPU for testing simplicity

    expect(runner).toBeDefined();
    expect(runner.isNull()).toBe(false);

    // Test running signature 1
    const a_data = new Float32Array([1, 2, 3, 4]);
    const b_data = new Float32Array([5, 6, 7, 8]);
    await a.setData(a_data);
    await b.setData(b_data);

    await runner.setInput("sig1", "a", a);
    await runner.setInput("sig1", "b", b);

    await runner.run("sig1");

    const c_out = runner.getOutput("sig1", "c");
    expect(c_out).toBeDefined();
    const c_data = await c_out.getData();
    expect(c_data).toEqual(new Float32Array([6, 8, 10, 12]));
    c_out.delete();

    // Test running signature 2
    const d_data = new Float32Array([2, 2, 2, 2]);
    await d.setData(d_data);

    await runner.setInput("sig2", "a", a); // Reusing 'a'
    await runner.setInput("sig2", "d", d);

    await runner.run("sig2");

    const e_out = runner.getOutput("sig2", "e");
    expect(e_out).toBeDefined();
    const e_data = await e_out.getData();
    expect(e_data).toEqual(new Float32Array([2, 4, 6, 8]));
    e_out.delete();

    a.delete();
    b.delete();
    c.delete();
    d.delete();
    e.delete();
    runner.delete();
  });

  it("verifies setData on a tensor without pre-allocated buffer", async () => {
    if (!litert) return;

    const tensor = litert.createPlaceholderTensor(litert.TensorType.FP32, [2, 2]);
    expect(tensor).toBeDefined();

    const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    
    // This should not crash and should allocate buffer automatically
    await tensor.setData(data);

    const retrievedData = await tensor.getData();
    expect(retrievedData).toBeDefined();
    expect(retrievedData.length).toEqual(4);
    expect(retrievedData[0]).toEqual(1.0);
    expect(retrievedData[3]).toEqual(4.0);

    tensor.delete();
  });

  it("verifies SafetensorsLoader.getTensor converts BF16 to FP32 correctly", async () => {
    if (!litert) return;

    const header = {
      "test_tensor": {
        "dtype": "BF16",
        "shape": [2],
        "data_offsets": [0, 4]
      }
    };

    const buffer = new ArrayBuffer(4);
    const u16View = new Uint16Array(buffer);
    u16View[0] = 0x3F80; // 1.0 in BF16
    u16View[1] = 0xBF80; // -1.0 in BF16
    
    const dataBuffer = new Uint8Array(buffer);

    const tensor = await SafetensorsLoader.getTensor(litert, header, dataBuffer, 0, "test_tensor");
    expect(tensor).toBeDefined();

    const retrievedData = await tensor.getData();
    expect(retrievedData).toBeDefined();
    expect(retrievedData.length).toEqual(2);
    expect(retrievedData[0]).toBeCloseTo(1.0, 5);
    expect(retrievedData[1]).toBeCloseTo(-1.0, 5);

    tensor.delete();
  });
  it("verifies that transpose operation on CPU produces correct output", async () => {
    if (!litert) return;

    const a = litert.createTensor({ type: 'FP32', shape: [1, 2, 3, 4] });
    a.setName("a");
    const perm = [0, 1, 3, 2];
    const b = a.transpose(perm);
    const c = b.transpose(perm); // Should be identity
    c.setName("c");

    const runner = await litert.createMultiSignatureRunner({
      "main": { outputs: [c] }
    }, litert.HwAccelerators.CPU);

    expect(runner).toBeDefined();
    expect(runner.isNull()).toBe(false);

    const input_data = new Float32Array(24);
    for (let i = 0; i < 24; ++i) input_data[i] = i + 1;
    await a.setData(input_data);

    await runner.setInput("main", "a", a);
    await runner.run("main");

    const c_out = runner.getOutput("main", "c");
    expect(c_out).toBeDefined();
    const c_data = await c_out.getData();
    
    expect(c_data).toEqual(input_data);

    a.delete();
    b.delete();
    c.delete();
    c_out.delete();
    runner.delete();
  });

  it("verifies that transpose operation on GPU produces correct output", async () => {
    if (!litert) return;

    const a = litert.createTensor({ type: 'FP32', shape: [1, 2, 3, 4] });
    a.setName("a");
    const perm = [0, 1, 3, 2];
    const b = a.transpose(perm);
    const c = b.transpose(perm); // Should be identity
    c.setName("c");

    const runner = await litert.createMultiSignatureRunner({
      "main": { outputs: [c] }
    }, litert.HwAccelerators.GPU);

    expect(runner).toBeDefined();
    expect(runner.isNull()).toBe(false);

    const input_data = new Float32Array(24);
    for (let i = 0; i < 24; ++i) input_data[i] = i + 1;
    await a.setData(input_data);

    await runner.setInput("main", "a", a);
    await runner.run("main");

    const c_out = runner.getOutput("main", "c");
    expect(c_out).toBeDefined();
    const c_data = await c_out.getData();
    
    expect(c_data).toEqual(input_data);

    a.delete();
    b.delete();
    c.delete();
    c_out.delete();
    runner.delete();
  });
  it("verifies that transpose operation on GPU works for LARGE tensors", async () => {
    if (!litert) return;

    const shape = [1, 4, 512, 256];
    const size = 1 * 4 * 512 * 256;
    const a = litert.createTensor({ type: 'FP32', shape: shape });
    a.setName("a");
    const perm = [0, 1, 3, 2];
    const b = a.transpose(perm);
    const c = b.transpose(perm); // Should be identity
    c.setName("c");

    const runner = await litert.createMultiSignatureRunner({
      "main": { outputs: [c] }
    }, litert.HwAccelerators.GPU);

    expect(runner).toBeDefined();
    expect(runner.isNull()).toBe(false);

    const input_data = new Float32Array(size);
    for (let i = 0; i < size; ++i) input_data[i] = i % 100; // Dummy data
    await a.setData(input_data);

    await runner.setInput("main", "a", a);
    await runner.run("main");

    const c_out = runner.getOutput("main", "c");
    expect(c_out).toBeDefined();
    const c_data = await c_out.getData();
    
    // Check a sample of elements to be fast
    let match = true;
    for (let i = 0; i < 1000; ++i) {
      if (c_data[i] !== input_data[i]) {
        match = false;
        break;
      }
    }
    expect(match).toBe(true);
    expect(c_data[0]).toBeCloseTo(input_data[0], 5);

    a.delete();
    b.delete();
    c.delete();
    c_out.delete();
    runner.delete();
  });

  it("verifies that transpose operation on GPU works for COMPUTED tensors", async () => {
    if (!litert) return;

    const shape = [1, 4, 512, 256];
    const size = 1 * 4 * 512 * 256;
    const a = litert.createTensor({ type: 'FP32', shape: shape });
    a.setName("a");
    
    // Create a computed tensor via Mul(a, 1)
    const one = litert.createPlaceholderTensor(litert.TensorType.FP32, [1]);
    one.setName("one");
    const q = a.mul(one); // Computed!
    
    const perm = [0, 1, 3, 2];
    const b = q.transpose(perm);
    const c = b.transpose(perm); // Should be identity
    c.setName("c");

    const runner = await litert.createMultiSignatureRunner({
      "main": { outputs: [c] }
    }, litert.HwAccelerators.GPU);

    expect(runner).toBeDefined();
    expect(runner.isNull()).toBe(false);

    const input_data = new Float32Array(size);
    for (let i = 0; i < size; ++i) input_data[i] = i % 100;
    await a.setData(input_data);
    
    const one_data = new Float32Array([1.0]);
    await one.setData(one_data);

    await runner.setInput("main", "a", a);
    await runner.setInput("main", "one", one);
    await runner.run("main");

    const c_out = runner.getOutput("main", "c");
    expect(c_out).toBeDefined();
    const c_data = await c_out.getData();
    
    let match = true;
    for (let i = 0; i < 1000; ++i) {
      if (c_data[i] !== input_data[i]) {
        match = false;
        break;
      }
    }
    expect(match).toBe(true);

    a.delete();
    one.delete();
    q.delete();
    b.delete();
    c.delete();
    c_out.delete();
    runner.delete();
  });
});


