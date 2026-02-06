/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {CompiledModel, Environment, LiteRt, loadAndCompile, loadLiteRt, type LoadLiteRtOptions, supportsFeature, Tensor, TensorBufferType, type TypedArray, unloadLiteRt} from '@litertjs/core';
// Placeholder for internal dependency on trusted resource url

describe('LiteRt', () => {
  let liteRt: LiteRt;

  async function resetLiteRt(
      loadFromDirectory = false, options: LoadLiteRtOptions = {}) {
    unloadLiteRt();
    if (loadFromDirectory) {
      liteRt = await loadLiteRt('/wasm', options);
    } else {
      liteRt = await loadLiteRt(
          trustedResourceUrl`/wasm/litert_wasm_internal.js`, options);
    }
  }

  beforeAll(async () => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 600_000;
  });

  it('loads the WASM module from its js file', async () => {
    await resetLiteRt();
    expect(liteRt).toBeDefined();
  });

  it('loads the WASM module from a directory', async () => {
    await resetLiteRt(/* loadFromDirectory= */ true);
    expect(liteRt).toBeDefined();
  });

  it('reloads the WASM module', async () => {
    await resetLiteRt();
    expect(liteRt).toBeDefined();
  });

  describe('threaded wasm', () => {
    it('loads the threaded WASM module', async () => {
      try {
        await resetLiteRt(/* loadFromDirectory= */ true, {threads: true});
        expect(liteRt).toBeDefined();
      } finally {
        await resetLiteRt();
      }
    });

    it('uses `navigator.hardwareConcurrency` threads by default', async () => {
      await resetLiteRt(/* loadFromDirectory= */ true, {threads: true});
      const model = await loadAndCompile('/testdata/add_10x10.tflite');
      expect(model).toBeDefined();
      expect(model.options.cpuOptions.numThreads)
          .toBe(navigator.hardwareConcurrency);
      model.delete();
    });

    it('can set the number of threads', async () => {
      await resetLiteRt(/* loadFromDirectory= */ true, {threads: true});
      const model = await loadAndCompile('/testdata/add_10x10.tflite', {
        cpuOptions: {numThreads: 3},
      });
      expect(model).toBeDefined();
      expect(model.options.cpuOptions.numThreads).toBe(3);
      model.delete();
    });
  });

  describe('jspi wasm', () => {
    it('loads the JSPI Wasm module', async () => {
      if (await supportsFeature('jspi')) {
        try {
          await resetLiteRt(/* loadFromDirectory= */ true, {jspi: true});
          expect(liteRt).toBeDefined();
        } finally {
          await resetLiteRt();
        }
      } else {
        pending('This browser does not support JSPI');
      }
    });

    it('throws an error if JSPI is not supported', async () => {
      if (await supportsFeature('jspi')) {
        pending('JSPI is supported in this browser');
      } else {
        await expectAsync(resetLiteRt(/* loadFromDirectory= */ true, {
          jspi: true
        })).toBeRejectedWithError('JSPI is not supported');
      }
    });
  });
  it('setDefaultEnvironment() sets the default environment', async () => {
    await resetLiteRt();
    const environment = new Environment({webGpuDevice: null});
    liteRt.setDefaultEnvironment(environment);
    expect(liteRt.getDefaultEnvironment()).toBe(environment);
  });

  it('setWebGpuDevice() sets the WebGPU device', async () => {
    await resetLiteRt();
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('No GPU adapter found.');
    }
    const device = await adapter.requestDevice();
    liteRt.setWebGpuDevice(device);
    expect(liteRt.getWebGpuDevice()).toBe(device);
  });

  describe('loadAndCompile', () => {
    // Some of these tests that intentionally cause C++ exceptions leave
    // LiteRT's wasm in a bad state. Reset it after each test to prevent
    // cascading failures.
    beforeEach(resetLiteRt);
    const modelPath = '/testdata/add_10x10.tflite';  // A small test model.
    const model2Path = '/testdata/multi_signature_model.tflite';

    it('loads from a Uint8Array', async () => {
      const modelData = await fetch(modelPath);
      const model =
          await loadAndCompile(new Uint8Array(await modelData.arrayBuffer()));
      expect(model).toBeDefined();
      model.delete();
    });

    it('loads from a string URL', async () => {
      const model = await loadAndCompile(modelPath);
      expect(model).toBeDefined();
      model.delete();
    });

    it('loads from a URL object', async () => {
      const modelUrl = new URL(`${location.origin}${modelPath}`);
      const model = await loadAndCompile(modelUrl);
      expect(model).toBeDefined();
      model.delete();
    });

    it('loads from a ReadableStreamDefaultReader', async () => {
      const modelData = await fetch(modelPath);
      const modelReader = modelData.body!.getReader();
      const model = await loadAndCompile(modelReader);
      expect(model).toBeDefined();
      model.delete();
    });

    it('throws when loading a huge model from a ReadableStreamDefaultReader',
       async () => {
         // TODO(msoulanille): Remove this test when we support loading large
         // models.
         const fakeModel = new ReadableStream<Uint8Array>({
           start(controller) {
             controller.enqueue(new Uint8Array(1.5e9));
             controller.enqueue(new Uint8Array(1.5e9));
             controller.close();
           }
         });

         await expectAsync(loadAndCompile(fakeModel.getReader()))
             .toBeRejectedWithError(/Model is too large/);
       });

    it('throws an error when given a bad model', async () => {
      const badModel =
          new TextEncoder().encode('****BADM and some extra data here.');

      await expectAsync(loadAndCompile(badModel))
          .toBeRejectedWithError(/Failed to load model from buffer/);
    });

    it('loads two models', async () => {
      const [model1, model2] = await Promise.all([
        loadAndCompile(modelPath),
        loadAndCompile(model2Path),
      ]);
      expect(model1).toBeDefined();
      expect(model2).toBeDefined();
      model1.delete();
      model2.delete();
    });

    it('loads with compileOptions with undefined accelerator', async () => {
      const model = await loadAndCompile(modelPath, {accelerator: undefined});
      expect(model).toBeDefined();
      model.delete();
    });

    it('loads with compileOptions with wasm accelerator', async () => {
      const model = await loadAndCompile(modelPath, {accelerator: 'wasm'});
      expect(model).toBeDefined();
      model.delete();
    });

    it('defaults to webgpu if unspecified and a WebGPU device is available',
       async () => {
         const adapter = await navigator.gpu.requestAdapter();
         if (!adapter) {
           throw new Error('No GPU adapter found.');
         }
         const device = await adapter.requestDevice();
         liteRt.setWebGpuDevice(device);
         const model = await loadAndCompile(modelPath, {
           environment: new Environment({webGpuDevice: device}),
         });
         expect(model.options.accelerator).toBe('webgpu');
         model.delete();
       });

    it('defaults to wasm if unspecified and no WebGPU device is available',
       async () => {
         const model = await loadAndCompile(modelPath, {
           environment: new Environment({webGpuDevice: null}),
         });
         expect(model.options.accelerator).toBe('wasm');
         model.delete();
       });

    it('throws an error if WebGPU is requested but no device is available',
       async () => {
         await expectAsync(loadAndCompile(modelPath, {
           environment: new Environment({webGpuDevice: null}),
           accelerator: 'webgpu',
         })).toBeRejectedWithError(/no WebGPU device is set/);
       });

    it('throws an error with compileOptions with unsupported accelerator type',
       async () => {
         await expectAsync(loadAndCompile(modelPath, {
           // this test is designed to throw an error, using any to
           // contruct a mock CompileOptions object.
           // tslint:disable-next-line:no-any
           accelerator: 'unsupported' as any
         })).toBeRejectedWithError(/Invalid accelerator: unsupported/);
       });
  });

  describe('input / output details', () => {
    let multiSignatureModel: CompiledModel;

    beforeAll(async () => {
      await resetLiteRt(true, {threads: false});
      multiSignatureModel =
          await loadAndCompile('/testdata/multi_signature_model.tflite');
    });

    afterAll(() => {
      multiSignatureModel.delete();
    });

    it('gets input details about the default signature', async () => {
      const inputDetails = multiSignatureModel.getInputDetails();
      expect(inputDetails).toEqual([
        {
          name: 'b',
          index: 0,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
          supportedBufferTypes:
              new Set([TensorBufferType.WEB_GPU_BUFFER_PACKED]),
        },
        {
          name: 'a',
          index: 1,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
          supportedBufferTypes:
              new Set([TensorBufferType.WEB_GPU_BUFFER_PACKED]),
        },
      ]);
    });

    it('gets output details about the default signature', async () => {
      const outputDetails = multiSignatureModel.getOutputDetails();
      expect(outputDetails).toEqual([
        {
          name: 'output',
          index: 0,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
          supportedBufferTypes:
              new Set([TensorBufferType.WEB_GPU_BUFFER_PACKED]),
        },
      ]);
    });

    it('gets input details about a specific signature', async () => {
      const inputDetails =
          multiSignatureModel.signatures['mul'].getInputDetails();
      expect(inputDetails).toEqual([
        {
          name: 'b',
          index: 0,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
          supportedBufferTypes:
              new Set([TensorBufferType.WEB_GPU_BUFFER_PACKED]),
        },
        {
          name: 'a',
          index: 1,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
          supportedBufferTypes:
              new Set([TensorBufferType.WEB_GPU_BUFFER_PACKED]),
        },
      ]);
    });

    it('gets output details about a specific signature', async () => {
      const outputDetails =
          multiSignatureModel.signatures['mul'].getOutputDetails();
      expect(outputDetails).toEqual([
        {
          name: 'output',
          index: 0,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
          supportedBufferTypes:
              new Set([TensorBufferType.WEB_GPU_BUFFER_PACKED]),
        },
      ]);
    });
  });

  describe('tensors', () => {
    it('creates a tensor from a Float32Array', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const tensor = new Tensor(data);
      expect(tensor.type.dtype).toEqual('float32');
      expect(tensor.type.layout.dimensions).toEqual([3]);
      expect(await tensor.data()).toEqual(data);
    });

    it('creates a tensor from a Int32Array', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Int32Array([1, 2, 3, 4, 5]);
      const tensor = new Tensor(data);
      expect(tensor.type.dtype).toEqual('int32');
      expect(tensor.type.layout.dimensions).toEqual([5]);
      expect(await tensor.data()).toEqual(data);
    });

    it('creates a tensor from a Uint8Array', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Uint8Array([1, 2, 3, 4, 5]);
      const tensor = new Tensor(data);
      expect(tensor.type.dtype).toEqual('uint8');
      expect(tensor.type.layout.dimensions).toEqual([5]);
      expect(await tensor.data()).toEqual(data);
    });

    it('creates a tensor from a GPUBuffer', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);

      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter!.requestDevice();
      liteRt.setWebGpuDevice(device);

      const gpuBuffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(gpuBuffer.getMappedRange()).set(data);
      gpuBuffer.unmap();

      const tensor =
          new Tensor(gpuBuffer, [3], 'float32', liteRt.getDefaultEnvironment());
      expect(tensor.type.dtype).toEqual('float32');
      expect(tensor.type.layout.dimensions).toEqual([3]);
      expect(await tensor.data()).toEqual(data);
      expect(tensor.bufferType).toEqual(TensorBufferType.WEB_GPU_BUFFER_PACKED);
      expect(tensor.accelerator).toEqual('webgpu');
      tensor.delete();
      gpuBuffer.destroy();
    });

    it('calls the onDelete callback when deleted', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const onDelete = jasmine.createSpy('onDelete');
      const tensor = new Tensor(data, undefined, undefined, onDelete);
      tensor.delete();
      expect(onDelete).toHaveBeenCalled();
    });

    it('environment is the default when not provided', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const tensor = new Tensor(data);
      expect(tensor.environment).toBe(liteRt.getDefaultEnvironment());
    });

    it('environment is the provided one when provided', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const environment = new Environment({webGpuDevice: null});
      const tensor = new Tensor(data, undefined, environment);
      expect(tensor.environment).toBe(environment);
    });

    it('copies a CPU tensor to a float32 WebGPU tensor and back', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const cpuTensor = new Tensor(data);
      expect(cpuTensor.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
      expect(cpuTensor.accelerator).toEqual('wasm');
      const gpuTensor = await cpuTensor.copyTo('webgpu');
      expect(gpuTensor.bufferType)
          .toEqual(TensorBufferType.WEB_GPU_BUFFER_PACKED);
      expect(gpuTensor.accelerator).toEqual('webgpu');

      // Copy back to CPU.
      const cpuTensor2 = await gpuTensor.copyTo('wasm');
      expect(cpuTensor2.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
      expect(cpuTensor2.accelerator).toEqual('wasm');
      // Copy back again to verify that gpuTensor is still valid after one copy.
      const cpuTensor3 = await gpuTensor.copyTo('wasm');
      expect(cpuTensor3.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
      expect(cpuTensor3.accelerator).toEqual('wasm');

      // Source tensor is still valid after copy.
      expect(await cpuTensor.data()).toEqual(data);
      // Copy actually copies the data.
      expect(await cpuTensor2.data()).toEqual(data);
      expect(await cpuTensor3.data()).toEqual(data);
      cpuTensor.delete();
      gpuTensor.delete();
      cpuTensor2.delete();
      cpuTensor3.delete();
    });

    it('can copy to a different environment', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const cpuTensor = new Tensor(data);

      // Create a new WebGPU device.
      const newAdapter = await navigator.gpu.requestAdapter();
      if (!newAdapter) {
        throw new Error('No GPU adapter found.');
      }
      const newDevice = await newAdapter.requestDevice();
      const newEnv = new Environment({webGpuDevice: newDevice});

      const gpuTensor = await cpuTensor.copyTo('webgpu', {environment: newEnv});
      expect(gpuTensor.environment).toBe(newEnv);
      expect(gpuTensor.bufferType)
          .toEqual(TensorBufferType.WEB_GPU_BUFFER_PACKED);
      expect(gpuTensor.accelerator).toEqual('webgpu');

      const cpuTensor2 = await gpuTensor.copyTo('wasm');
      expect(await cpuTensor2.data()).toEqual(data);
      expect(cpuTensor2.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
      expect(cpuTensor2.accelerator).toEqual('wasm');

      gpuTensor.delete();
      cpuTensor.delete();
      cpuTensor2.delete();
    });

    it('copying to WebGPU throws if the environment does not have a WebGPU device',
       async () => {
         await resetLiteRt(true, {threads: false});
         const data = new Float32Array([1.234, 2.345, 3.456]);
         const cpuTensor = new Tensor(data);
         const badEnv = new Environment({webGpuDevice: null});
         await expectAsync(cpuTensor.copyTo('webgpu', {
           environment: badEnv
         })).toBeRejectedWithError(/No WebGPU device is available/);
         cpuTensor.delete();
       });

    it('can specify copy destination by TensorBufferType', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const cpuTensor = new Tensor(data);
      const gpuTensor =
          await cpuTensor.copyTo(TensorBufferType.WEB_GPU_BUFFER_PACKED);
      expect(gpuTensor.bufferType)
          .toEqual(TensorBufferType.WEB_GPU_BUFFER_PACKED);
      expect(gpuTensor.accelerator).toEqual('webgpu');
      gpuTensor.delete();
      cpuTensor.delete();
    });

    it('throws an error if copying to an unsupported destination', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const cpuTensor = new Tensor(data);
      await expectAsync(
          cpuTensor.copyTo('unsupported' as unknown as TensorBufferType))
          .toBeRejectedWithError(
              /Unknown destination 'unsupported' for copying or moving/);
      cpuTensor.delete();
    });

    it('moves a CPU tensor to a float32 WebGPU tensor and back', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const cpuTensor = new Tensor(data);
      expect(cpuTensor.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
      expect(cpuTensor.accelerator).toEqual('wasm');
      const gpuTensor = await cpuTensor.moveTo('webgpu');
      expect(gpuTensor.bufferType)
          .toEqual(TensorBufferType.WEB_GPU_BUFFER_PACKED);
      expect(gpuTensor.accelerator).toEqual('webgpu');
      const cpuTensor2 = await gpuTensor.moveTo('wasm');
      expect(cpuTensor2.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
      expect(cpuTensor2.accelerator).toEqual('wasm');
      // Source tensors are not valid after move.
      expect(cpuTensor.deleted).toBeTrue();
      expect(gpuTensor.deleted).toBeTrue();
      // Move actually moves the data.
      expect(await cpuTensor2.data()).toEqual(data);
      cpuTensor2.delete();
    });

    it('data() rejects if the tensor is deleted', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const tensor = new Tensor(data);
      tensor.delete();
      await expectAsync(tensor.data())
          .toBeRejectedWithError('Tensor is deleted and cannot be used.');
    });

    it('toGpuBuffer() returns the underlying GPUBuffer', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);

      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter!.requestDevice();
      liteRt.setWebGpuDevice(device);

      const tensor = new Tensor(data);
      const gpuTensor = await tensor.moveTo('webgpu');

      const gpuBuffer = gpuTensor.toGpuBuffer();
      expect(gpuBuffer).toBeInstanceOf(GPUBuffer);
      expect(gpuBuffer.size).toBeGreaterThanOrEqual(data.byteLength);

      const readBuffer = device.createBuffer({
        size: gpuBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(gpuBuffer, 0, readBuffer, 0, gpuBuffer.size);
      device.queue.submit([encoder.finish()]);

      await readBuffer.mapAsync(GPUMapMode.READ);
      const resultData =
          new Float32Array(readBuffer.getMappedRange(), 0, data.length);
      expect(resultData).toEqual(data);

      readBuffer.unmap();
      readBuffer.destroy();
      // gpuTensor.delete();
    });

    it('toGpuBuffer() throws if the tensor is not on WebGPU', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const tensor = new Tensor(data);
      expect(() => tensor.toGpuBuffer())
          .toThrowError(
              'Cannot convert a Tensor with non-WebGPU memory to a GPUBuffer.');
      tensor.delete();
    });

    it('deletes the WebGPU buffer when the tensor is deleted', async () => {
      await resetLiteRt(true, {threads: false});
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const tensor = new Tensor(data);
      const gpuTensor = await tensor.moveTo('webgpu');
      const gpuBuffer = gpuTensor.toGpuBuffer();
      gpuBuffer.label = 'the test buffer';
      expect(gpuBuffer).toBeInstanceOf(GPUBuffer);
      gpuTensor.delete();
      await expectAsync(
          checkBufferIsUsable(tensor.environment.webGpuDevice!, gpuBuffer))
          .toBeRejectedWithError(/the test buffer.*destroyed/);
    });

    it('liteRtWasm.WebGPU.Internals.jsObjects no longer contains the buffer pointer when the tensor is deleted',
       async () => {
         await resetLiteRt(true, {threads: false});
         const data = new Float32Array([1.234, 2.345, 3.456]);
         const tensor = new Tensor(data);
         const gpuTensor = await tensor.moveTo('webgpu');
         const bufferPtr = gpuTensor.liteRtTensorBuffer.getWebGpuBuffer();

         // This is a private property of the WebGPU module, but we need to
         // access it to test that the buffer is removed from the map when the
         // tensor is deleted.
         // tslint:disable-next-line:no-any
         const jsObjects =
             (liteRt.liteRtWasm.WebGPU as any).Internals.jsObjects;
         expect(jsObjects[bufferPtr]).toBe(gpuTensor.toGpuBuffer());
         gpuTensor.delete();
         expect(jsObjects[bufferPtr]).toBeUndefined();
       });

    for (const dimensionCount of [1, 2, 3, 4] as const) {
      describe(`with ${dimensionCount} dimensions`, () => {
        const batchDims = new Array(dimensionCount - 1).fill(1);
        it('copies a Float32 tensor from CPU to WebGPU and back', async () => {
          await resetLiteRt(true, {threads: false});
          const data = new Float32Array([1.234, 2.345, 3.456]);
          const cpuTensor = new Tensor(data, [...batchDims, 3]);
          expect(cpuTensor.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
          expect(cpuTensor.accelerator).toEqual('wasm');
          const gpuTensor = await cpuTensor.copyTo('webgpu');
          expect(gpuTensor.bufferType)
              .toEqual(TensorBufferType.WEB_GPU_BUFFER_PACKED);
          expect(gpuTensor.accelerator).toEqual('webgpu');
          const cpuTensor2 = await gpuTensor.copyTo('wasm');
          expect(cpuTensor2.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
          expect(cpuTensor2.accelerator).toEqual('wasm');
          expect(await cpuTensor2.data()).toEqual(await cpuTensor.data());
          expect(cpuTensor.deleted).toBeFalse();
          expect(gpuTensor.deleted).toBeFalse();
          cpuTensor.delete();
          gpuTensor.delete();
          cpuTensor2.delete();
        });

        it('copies an Int32 tensor from CPU to WebGPU and back', async () => {
          await resetLiteRt(true, {threads: false});
          const data = new Int32Array([1, 2, 3, 2147483647]);
          const cpuTensor = new Tensor(data, [...batchDims, 4]);
          expect(cpuTensor.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
          expect(cpuTensor.accelerator).toEqual('wasm');
          const gpuTensor = await cpuTensor.copyTo('webgpu');
          expect(gpuTensor.bufferType)
              .toEqual(TensorBufferType.WEB_GPU_BUFFER_PACKED);
          expect(gpuTensor.accelerator).toEqual('webgpu');
          const cpuTensor2 = await gpuTensor.copyTo('wasm');
          expect(cpuTensor2.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
          expect(cpuTensor2.accelerator).toEqual('wasm');
          expect(await cpuTensor2.data()).toEqual(await cpuTensor.data());
          expect(cpuTensor.deleted).toBeFalse();
          expect(gpuTensor.deleted).toBeFalse();
          cpuTensor.delete();
          gpuTensor.delete();
          cpuTensor2.delete();
        });

        it('moves a Float32 tensor from CPU to WebGPU and back', async () => {
          await resetLiteRt(true, {threads: false});
          const data = new Float32Array([1.234, 2.345, 3.456]);
          const cpuTensor = new Tensor(data, [...batchDims, 3]);
          expect(cpuTensor.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
          expect(cpuTensor.accelerator).toEqual('wasm');
          const gpuTensor = await cpuTensor.moveTo('webgpu');
          expect(gpuTensor.bufferType)
              .toEqual(TensorBufferType.WEB_GPU_BUFFER_PACKED);
          expect(gpuTensor.accelerator).toEqual('webgpu');
          const cpuTensor2 = await gpuTensor.moveTo('wasm');
          expect(cpuTensor2.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
          expect(cpuTensor2.accelerator).toEqual('wasm');
          expect(await cpuTensor2.data()).toEqual(data);
          expect(cpuTensor.deleted).toBeTrue();
          expect(gpuTensor.deleted).toBeTrue();
          cpuTensor2.delete();
        });

        it('moves an Int32 tensor from CPU to WebGPU and back', async () => {
          await resetLiteRt(true, {threads: false});
          const data = new Int32Array([1, 2, 3, 2147483647]);
          const cpuTensor = new Tensor(data, [...batchDims, 4]);
          expect(cpuTensor.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
          expect(cpuTensor.accelerator).toEqual('wasm');
          const gpuTensor = await cpuTensor.moveTo('webgpu');
          expect(gpuTensor.bufferType)
              .toEqual(TensorBufferType.WEB_GPU_BUFFER_PACKED);
          expect(gpuTensor.accelerator).toEqual('webgpu');
          const cpuTensor2 = await gpuTensor.moveTo('wasm');
          expect(cpuTensor2.bufferType).toEqual(TensorBufferType.HOST_MEMORY);
          expect(cpuTensor2.accelerator).toEqual('wasm');
          expect(await cpuTensor2.data()).toEqual(data);
          expect(cpuTensor.deleted).toBeTrue();
          expect(gpuTensor.deleted).toBeTrue();
          cpuTensor2.delete();
        });
      });
    }
  });

  describe('run', () => {
    let identity: Tensor;
    let range: Tensor;

    beforeAll(async () => {
      await resetLiteRt();
      // 10x10 identity matrix
      const identityArray = new Float32Array(100);
      for (let i = 0; i < 10; i++) {
        identityArray[i * 10 + i] = 1.0;
      }
      identity = new Tensor(identityArray, [10, 10]);

      // Range from 0 to 99
      const rangeArray = new Float32Array(100);
      for (let i = 0; i < 100; i++) {
        rangeArray[i] = i;
      }
      range = new Tensor(rangeArray, [10, 10]);
    });

    afterAll(() => {
      identity.delete();
      range.delete();
    });

    it('copies wasm inputs to webgpu when running a webgpu model', async () => {
      const model = await loadAndCompile(
          '/testdata/multi_signature_model.tflite', {accelerator: 'webgpu'});
      const identityCpu = await identity.copyTo('wasm');
      const rangeCpu = await range.copyTo('wasm');
      const outputs = await model.run([identityCpu, rangeCpu]);
      expect(outputs.length).toBe(1);
      const sum = await addTensors(identity, range);
      expect(await outputs[0].data()).toEqual(sum);
      outputs[0].delete();
      model.delete();
      identityCpu.delete();
      rangeCpu.delete();
    });

    it('copies webgpu inputs to wasm when running a wasm model', async () => {
      const model = await loadAndCompile(
          '/testdata/multi_signature_model.tflite', {accelerator: 'wasm'});
      const identityWebGpu = await identity.copyTo('webgpu');
      const rangeWebGpu = await range.copyTo('webgpu');
      const outputs = await model.run([identityWebGpu, rangeWebGpu]);
      expect(outputs.length).toBe(1);
      const sum = await addTensors(identity, range);
      expect(await outputs[0].data()).toEqual(sum);
      outputs[0].delete();
      model.delete();
      identityWebGpu.delete();
      rangeWebGpu.delete();
    });

    for (const accelerator of ['wasm', 'webgpu'] as const) {
      describe(accelerator, () => {
        let mobilenetModel: CompiledModel;
        let multiSignatureModel: CompiledModel;
        let mixedInputModel: CompiledModel;

        beforeAll(async () => {
          mobilenetModel = await loadAndCompile(
              '/testdata/torchvision_mobilenet_v2.tflite', {accelerator});
          multiSignatureModel = await loadAndCompile(
              '/testdata/multi_signature_model.tflite', {accelerator});
          mixedInputModel = await loadAndCompile(
              '/testdata/mixed_input_model.tflite', {accelerator});
        });

        afterAll(() => {
          mobilenetModel.delete();
          multiSignatureModel.delete();
          mixedInputModel.delete();
        });

        it(`model.options.accelerator is ${accelerator}`, () => {
          expect(mobilenetModel.options.accelerator).toBe(accelerator);
          expect(multiSignatureModel.options.accelerator).toBe(accelerator);
          expect(mixedInputModel.options.accelerator).toBe(accelerator);
        });

        it('runs multiple signatures of a model', async () => {
          // "add" signature
          const sum = await addTensors(identity, range);
          const addOutputs =
              await multiSignatureModel.run('add', {'a': identity, 'b': range});
          expect(Object.keys(addOutputs)).toEqual(['output']);
          const output = addOutputs['output'];
          expect(await output.data()).toEqual(sum);
          output.delete();

          // "sub" signature
          const diff = await subTensors(identity, range);
          const subOutputs =
              await multiSignatureModel.run('sub', {'a': identity, 'b': range});
          expect(Object.keys(subOutputs)).toEqual(['output']);
          const subOutput = subOutputs['output'];
          expect(await subOutput.data()).toEqual(diff);
          subOutput.delete();

          // "mul" signature
          const product = await mulTensors(identity, range);
          const mulOutputs =
              await multiSignatureModel.run('mul', {'a': identity, 'b': range});
          expect(Object.keys(mulOutputs)).toEqual(['output']);
          const mulOutput = mulOutputs['output'];
          expect(await mulOutput.data()).toEqual(product);
          mulOutput.delete();
        });

        it('runs a model multiple times', async () => {
          const fakeInput =
              await (new Tensor(new Float32Array(1 * 3 * 224 * 224), [
                1, 3, 224, 224
              ])).moveTo(accelerator);

          const outputPromises = [];
          for (let i = 0; i < 10; ++i) {
            // The model should not crash when run multiple times.
            outputPromises.push(mobilenetModel.run({'args_0': fakeInput}));
          }
          const outputs = await Promise.all(outputPromises);
          const outputsData = await Promise.all(
              outputs.map(output => output['output_0'].data()));

          // All the outputs should be the same.
          for (let i = 1; i < outputsData.length; ++i) {
            expect(outputsData[i]).toEqual(outputsData[0]);
          }
          fakeInput.delete();
          for (const output of outputs) {
            output['output_0'].delete();
          }
        });

        it('runs a model with mixed input types', async () => {
          const int32InputData = new Int32Array(10);
          for (let i = 0; i < 10; i++) int32InputData[i] = i;
          const int32Input =
              await (new Tensor(int32InputData, [1, 10])).moveTo(accelerator);

          const float32InputData = new Float32Array(10);
          for (let i = 0; i < 10; i++) float32InputData[i] = i / 10;
          const float32Input =
              await (new Tensor(float32InputData, [1, 10])).moveTo(accelerator);

          const outputs = await mixedInputModel.run({
            'int_input': int32Input,
            'float_input': float32Input,
          });

          const outputData = await outputs['Identity'].data();
          const expectedOutputData = new Float32Array(10);
          for (let i = 0; i < 10; i++) {
            expectedOutputData[i] =
                (int32InputData[i] + float32InputData[i]) * 2;
          }
          expect(outputData).toEqual(expectedOutputData);
          int32Input.delete();
          float32Input.delete();
          outputs['Identity'].delete();
        });

        it('`run(Tensor)` returns `Tensor[]`', async () => {
          const fakeInput =
              await (new Tensor(new Float32Array(1 * 3 * 224 * 224), [
                1, 3, 224, 224
              ])).moveTo(accelerator);

          const outputs = await mobilenetModel.run(fakeInput);
          const expectedOutputs =
              await mobilenetModel.run({'args_0': fakeInput});

          expect(outputs.length).toBe(1);
          expect(outputs[0]).toBeInstanceOf(Tensor);
          expect(await outputs[0].data())
              .toEqual(await expectedOutputs['output_0'].data());
          fakeInput.delete();
          outputs[0].delete();
          expectedOutputs['output_0'].delete();
        });

        it('`run([Tensor, ...])` returns `Tensor[]`', async () => {
          const outputs = await multiSignatureModel.run([identity, range]);
          expect(outputs.length).toBe(1);
          const sum = await addTensors(identity, range);
          expect(outputs[0]).toBeInstanceOf(Tensor);
          expect(await outputs[0].data()).toEqual(sum);
          outputs[0].delete();
        });

        it('`run({\'input\': Tensor, ...})` returns `{\'output\': Tensor, ...}`',
           async () => {
             const outputs =
                 await multiSignatureModel.run({'a': identity, 'b': range});
             const sum = await addTensors(identity, range);
             expect(await outputs['output'].data()).toEqual(sum);
             expect(outputs['output']).toBeInstanceOf(Tensor);
             outputs['output'].delete();
           });

        it('`run(\'signature\', Tensor)` returns `Tensor[]`', async () => {
          const fakeInput =
              await (new Tensor(new Float32Array(1 * 3 * 224 * 224), [
                1, 3, 224, 224
              ])).moveTo(accelerator);
          const outputs =
              await mobilenetModel.run('serving_default', fakeInput);
          const expectedOutputs =
              await mobilenetModel.run({'args_0': fakeInput});
          expect(outputs.length).toBe(1);
          expect(await outputs[0].data())
              .toEqual(await expectedOutputs['output_0'].data());
          fakeInput.delete();
          outputs[0].delete();
          expectedOutputs['output_0'].delete();
        });

        it('`run(\'signature\', [Tensor, ...])` returns `Tensor[]`',
           async () => {
             const outputs =
                 await multiSignatureModel.run('add', [identity, range]);
             expect(outputs.length).toBe(1);
             const sum = await addTensors(identity, range);
             expect(outputs[0]).toBeInstanceOf(Tensor);
             expect(await outputs[0].data()).toEqual(sum);
             outputs[0].delete();
           });

        describe('custom environments', () => {
          let customEnv: Environment;
          let modelWithCustomEnv: CompiledModel;
          let identityCustom: Tensor;
          let rangeCustom: Tensor;

          beforeEach(async () => {
            customEnv = new Environment({webGpuDevice: null});
            modelWithCustomEnv = await loadAndCompile(
                '/testdata/multi_signature_model.tflite',
                {environment: customEnv});
            identityCustom =
                new Tensor(await identity.data(), [10, 10], customEnv);
            rangeCustom = new Tensor(await range.data(), [10, 10], customEnv);
          });

          afterEach(() => {
            identityCustom.delete();
            rangeCustom.delete();
            modelWithCustomEnv.delete();
            customEnv.delete();
          });

          it('model.options.environment is the custom environment', () => {
            expect(modelWithCustomEnv.options.environment).toBe(customEnv);
          });

          it('returned tensors share the same environment as the model',
             async () => {
               const outputs =
                   await modelWithCustomEnv.run([identityCustom, rangeCustom]);
               expect(outputs[0].environment).toBe(customEnv);
               expect(outputs[0].environment)
                   .not.toBe(liteRt.getDefaultEnvironment());
               outputs[0].delete();
             });

          it('returned tensors share the same environment as the signature',
             async () => {
               const outputs = await modelWithCustomEnv.signatures['add'].run(
                   [identityCustom, rangeCustom]);
               expect(outputs[0].environment).toBe(customEnv);
               expect(outputs[0].environment)
                   .not.toBe(liteRt.getDefaultEnvironment());
               outputs[0].delete();
             });
        });

        it('`run([Tensor])` throws if the model expects more tensors',
           async () => {
             await expectAsync(multiSignatureModel.run([identity]))
                 .toBeRejectedWithError(/called with 1.*expects 2/);
           });

        it('throws an error if no input is provided', async () => {
          // TS compiler will prevent this, but JS may call it so test it
          // anyway.
          // tslint:disable-next-line:no-any
          await expectAsync((mobilenetModel as any).run('serving_default'))
              .toBeRejectedWithError(
                  /No input provided for signature serving_default/);
        });

        it('throws an error if the tensor is of the wrong type', async () => {
          const fakeInput =
              new Tensor(new Int32Array(1 * 3 * 224 * 224), [1, 3, 224, 224]);
          await expectAsync(mobilenetModel.run({'args_0': fakeInput}))
              .toBeRejectedWithError(
                  'TensorBuffer ranked tensor type Int32[1, 3, 224, 224] does not match expected ranked tensor type Float32[1, 3, 224, 224]');
          fakeInput.delete();
        });

        it('throws an error if the tensor is of the wrong shape', async () => {
          const fakeInput =
              new Tensor(new Float32Array(1 * 3 * 224 * 225), [1, 3, 224, 225]);
          await expectAsync(mobilenetModel.run({'args_0': fakeInput}))
              .toBeRejectedWithError(
                  'TensorBuffer ranked tensor type Float32[1, 3, 224, 225] does not match expected ranked tensor type Float32[1, 3, 224, 224]');
          fakeInput.delete();
        });

        it('throws an error if the signature does not exist', async () => {
          await expectAsync(multiSignatureModel.run('bad_signature', [
            identity, range
          ])).toBeRejectedWithError(/No signature named bad_signature/);
        });

        it('throws an error if the signature does not have all the array inputs',
           async () => {
             await expectAsync(multiSignatureModel.run('add', [identity]))
                 .toBeRejectedWithError(/called with 1.*expects 2/);
           });

        it('throws an error if the signature does not have all the object inputs',
           async () => {
             await expectAsync(multiSignatureModel.run('add', {'a': identity}))
                 .toBeRejectedWithError(
                     /called with input record that is missing input b/);
           });

        it('throws an error if the model has been deleted', async () => {
          const deletableModel = await loadAndCompile(
              '/testdata/multi_signature_model.tflite', {accelerator});
          deletableModel.delete();
          await expectAsync(deletableModel.run([identity, range]))
              .toBeRejectedWithError(
                  Error, /CompiledModel is deleted and cannot be used./);
        });

        it('throws an error if a signature\'s model has been deleted', async () => {
          const deletableModel = await loadAndCompile(
              '/testdata/multi_signature_model.tflite', {accelerator});
          const signature = deletableModel.signatures['add'];
          deletableModel.delete();
          await expectAsync(signature.run([identity, range]))
              .toBeRejectedWithError(
                  Error,
                  /CompiledModelSignatureRunner is deleted and cannot be used./);
        });

        it('supports 1D, 2D, 3D, and 4D input tensors', async () => {
          const model = await loadAndCompile(
              '/testdata/add_1d_2d_3d_4d.tflite', {accelerator});
          const x = new Tensor(new Float32Array(4).fill(1.0), [4]);

          const y = new Tensor(new Float32Array(12).fill(1.0), [3, 4]);
          const z = new Tensor(new Float32Array(24).fill(1.0), [2, 3, 4]);
          const w = new Tensor(new Float32Array(24).fill(1.0), [1, 2, 3, 4]);
          const inputs = {'x': x, 'y': y, 'z': z, 'w': w};

          const outputs = await model.run(inputs);
          expect(await outputs['Identity'].data())
              .toEqual(new Float32Array(24).fill(4.0));

          x.delete();
          y.delete();
          z.delete();
          w.delete();
          outputs['Identity'].delete();
          model.delete();
        });

        it('supports 1D tensors with last dim not divisible by 4', async () => {
          const model =
              await loadAndCompile('/testdata/add_c1_c7.tflite', {accelerator});
          const x = new Tensor(new Float32Array([1.0]), [1]);
          const y = new Tensor(new Float32Array(7).fill(1.0), [7]);
          const inputs = {'x': x, 'y': y};

          const outputs = await model.run(inputs);
          expect(await outputs['Identity'].data())
              .toEqual(new Float32Array(7).fill(2.0));
          x.delete();
          y.delete();
          outputs['Identity'].delete();
          model.delete();
        });

        it('supports 2D tensors with last dim not divisible by 4', async () => {
          const model =
              await loadAndCompile('/testdata/add_10x10.tflite', {accelerator});

          const identityData = new Float32Array(100);
          for (let i = 0; i < 10; i++) {
            for (let j = 0; j < 10; j++) {
              identityData[i * 10 + j] = 1.0;
            }
          }
          const a = new Tensor(identityData, [10, 10]);

          const rangeData = new Float32Array(100);
          for (let i = 0; i < 100; i++) {
            rangeData[i] = i;
          }
          const b = new Tensor(rangeData, [10, 10]);

          const outputs = await model.run({'a': a, 'b': b});
          const outputData = await outputs['Identity'].data();

          const expectedData = new Float32Array(100);
          for (let i = 0; i < 100; i++) {
            expectedData[i] = identityData[i] + rangeData[i];
          }
          expect(outputData).toEqual(expectedData);
          a.delete();
          b.delete();
          outputs['Identity'].delete();
          model.delete();
        });
      });
    }
  });

  describe('delegate compatibility', () => {
    let model: CompiledModel;
    let a: Tensor;
    let b: Tensor;
    let expectedOutput: Float32Array;

    beforeAll(async () => {
      await resetLiteRt();
      model = await loadAndCompile(
          '/testdata/delegate_compatibility_test.tflite',
          {accelerator: 'webgpu'});

      const aData = new Float32Array(100).fill(0);
      for (let i = 0; i < 10; ++i) aData[i * 10 + i] = 1;
      a = new Tensor(aData, [10, 10]);

      const bData = new Float32Array(100);
      for (let i = 0; i < 100; ++i) bData[i] = i;
      b = new Tensor(bData, [10, 10]);
      expectedOutput = bData;
    });

    afterAll(() => {
      model.delete();
      a.delete();
      b.delete();
    });

    it('loads the model', async () => {
      expect(model).toBeDefined();
    });

    describe('matmul', () => {
      it('works with 10x10 inputs', async () => {
        const outputs = await model.run('matmul_10x10', {
          'a': a,
          'b': b,
        });
        expect(await outputs['output'].data()).toEqual(expectedOutput);
        outputs['output'].delete();
      });

      it('works with 1x10x10 inputs', async () => {
        const a1x10x10 = new Tensor(await a.data(), [1, 10, 10]);
        const b1x10x10 = new Tensor(await b.data(), [1, 10, 10]);
        const outputs = await model.run('matmul_1x10x10', {
          'a': a1x10x10,
          'b': b1x10x10,
        });
        expect(await outputs['output'].data()).toEqual(expectedOutput);
        a1x10x10.delete();
        b1x10x10.delete();
        outputs['output'].delete();
      });

      it('works with 1x1x10x10 inputs', async () => {
        const a1x1x10x10 = new Tensor(await a.data(), [1, 1, 10, 10]);
        const b1x1x10x10 = new Tensor(await b.data(), [1, 1, 10, 10]);
        const outputs = await model.run('matmul_1x1x10x10', {
          'a': a1x1x10x10,
          'b': b1x1x10x10,
        });
        expect(await outputs['output'].data()).toEqual(expectedOutput);
        a1x1x10x10.delete();
        b1x1x10x10.delete();
        outputs['output'].delete();
      });
    });
  });
});

async function addTensors(a: Tensor, b: Tensor): Promise<TypedArray> {
  const aArray = await a.data();
  const bArray = await b.data();
  const result = new Float32Array(aArray.length);
  for (let i = 0; i < aArray.length; i++) {
    result[i] = aArray[i] + bArray[i];
  }
  return result;
}

async function subTensors(a: Tensor, b: Tensor): Promise<TypedArray> {
  const aArray = await a.data();
  const bArray = await b.data();
  const result = new Float32Array(aArray.length);
  for (let i = 0; i < aArray.length; i++) {
    result[i] = aArray[i] - bArray[i];
  }
  return result;
}

async function mulTensors(a: Tensor, b: Tensor): Promise<TypedArray> {
  const aArray = await a.data();
  const bArray = await b.data();
  const result = new Float32Array(aArray.length);
  for (let i = 0; i < aArray.length; i++) {
    result[i] = aArray[i] * bArray[i];
  }
  return result;
}

async function checkBufferIsUsable(
    device: GPUDevice, buffer: GPUBuffer): Promise<void> {
  device.pushErrorScope('validation');
  const tempBuffer =
      device.createBuffer({size: 4, usage: GPUBufferUsage.COPY_DST});
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, tempBuffer, 0, 0);
  device.queue.submit([encoder.finish()]);
  tempBuffer.destroy();

  const error = await device.popErrorScope();
  if (error) {
    // Wrap in an error message that jasmine can read.
    throw new Error(error.message);
  }
}
