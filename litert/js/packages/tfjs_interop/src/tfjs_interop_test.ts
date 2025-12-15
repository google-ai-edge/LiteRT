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

import '@tensorflow/tfjs-backend-cpu';

import * as litert from '@litertjs/core_litert';
import {litertToTfjs, runWithTfjsTensors, TensorConversionError, tfjsToLitert} from '@litertjs/tfjs-interop';
import {WebGPUBackend} from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// Placeholder for internal dependency on trusted resource url

describe('TFJS Interop', () => {
  async function resetLiteRt() {
    litert.unloadLiteRt();
    await litert.loadLiteRt('/wasm');

    // Share the same WebGPU device with TFJS.
    await tf.ready();


    const device = await litert.getWebGpuDevice();
    if (!device) {
      throw new Error('No WebGPU device is available.');
    }

    const adapterInfo = device.adapterInfo;
    tf.removeBackend('webgpu');
    tf.registerBackend('webgpu', () => new WebGPUBackend(device, adapterInfo));
    await tf.setBackend('webgpu');
  }

  beforeAll(async () => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 10000;
    await resetLiteRt();
  });

  describe('tfjsToLitert', () => {
    it('throws when LiteRT and TFJS are using different WebGPU devices',
       async () => {
         // Create a new WebGPU device.
         const newAdapter = await navigator.gpu.requestAdapter();
         if (!newAdapter) {
           throw new Error('No GPU adapter found.');
         }
         const newDevice = await newAdapter.requestDevice();
         const newEnv = new litert.Environment({webGpuDevice: newDevice});


         const tfjsGpuTensor = tf.range(0, 10, 1, 'float32').pow(2);
         expect(() => tfjsToLitert(tfjsGpuTensor, newEnv))
             .toThrowError(
                 TensorConversionError,
                 /To convert from TFJS to LiteRT, you must use an environment that has the same WebGPU device as the TFJS backend/);
       });

    it('copies to a HOST_MEMORY tensor when requested (if TFJS has the tensor cached in CPU memory)',
       async () => {
         const tfjsTensor = makeFloat32TfjsTensor([10, 10]);
         await tfjsTensor.data();  // Cache the tensor in CPU memory.

         const litertTensor = tfjsToLitert(
             tfjsTensor, litert.getDefaultEnvironment(),
             litert.TensorBufferType.HOST_MEMORY);
         expect(litertTensor.bufferType)
             .toEqual(litert.TensorBufferType.HOST_MEMORY);
         expect(litertTensor.accelerator).toEqual('wasm');
         expect(await litertTensor.data()).toEqual(await tfjsTensor.data());
         tfjsTensor.dispose();
         litertTensor.delete();
       });

    it('copies to a WEB_GPU_BUFFER_PACKED tensor even when HOST_MEMORY is requested if the TFJS tensor is not cached in CPU memory',
       async () => {
         const tfjsTensor = makeFloat32TfjsTensor([10, 10]);
         const litertTensor = tfjsToLitert(
             tfjsTensor, litert.getDefaultEnvironment(),
             litert.TensorBufferType.HOST_MEMORY);
         expect(litertTensor.bufferType)
             .toEqual(litert.TensorBufferType.WEB_GPU_BUFFER_PACKED);
         expect(litertTensor.accelerator).toEqual('webgpu');
         expect(await litertTensor.data()).toEqual(await tfjsTensor.data());
         tfjsTensor.dispose();
         litertTensor.delete();
       });


    for (const dimensionCount of [1, 2, 3, 4] as const) {
      describe(`with ${dimensionCount} dimensions`, () => {
        const shape = new Array(dimensionCount).fill(3);

        it('converts a TFJS Float32 CPU tensor to a LiteRT CPU tensor',
           async () => {
             await tf.setBackend('cpu');
             try {
               const tfjsTensor = makeFloat32TfjsTensor(shape);
               const litertTensor = tfjsToLitert(tfjsTensor);
               expect(litertTensor.bufferType)
                   .toEqual(litert.TensorBufferType.HOST_MEMORY);
               expect(litertTensor.accelerator).toEqual('wasm');
               expect(await litertTensor.data())
                   .toEqual(await tfjsTensor.data());
               tfjsTensor.dispose();
               litertTensor.delete();
             } finally {
               await resetLiteRt();
             }
           });

        it('converts a TFJS Float32 WebGPU tensor to a LiteRT WebGPU tensor',
           async () => {
             // .pow(2) is a non-trivial op that will trigger WebGPU upload.
             const tfjsTensor = makeFloat32TfjsTensor(shape);
             const litertTensor = tfjsToLitert(tfjsTensor);
             expect(litertTensor.bufferType)
                 .toEqual(litert.TensorBufferType.WEB_GPU_BUFFER_PACKED);
             expect(litertTensor.accelerator).toEqual('webgpu');
             tfjsTensor.dispose();
             litertTensor.delete();
           });

        it('converts a TFJS Int32 CPU tensor to a LiteRT CPU tensor',
           async () => {
             await tf.setBackend('cpu');
             try {
               const tfjsTensor = makeInt32TfjsTensor(shape);
               const litertTensor = tfjsToLitert(tfjsTensor);
               expect(litertTensor.bufferType)
                   .toEqual(litert.TensorBufferType.HOST_MEMORY);
               expect(litertTensor.accelerator).toEqual('wasm');
               expect(await litertTensor.data())
                   .toEqual(await tfjsTensor.data());
               tfjsTensor.dispose();
               litertTensor.delete();
             } finally {
               await resetLiteRt();
             }
           });

        it('converts a TFJS Int32 WebGPU tensor to a LiteRT WebGPU tensor',
           async () => {
             const tfjsTensor = makeInt32TfjsTensor(shape);
             const litertTensor = tfjsToLitert(tfjsTensor);
             expect(litertTensor.bufferType)
                 .toEqual(litert.TensorBufferType.WEB_GPU_BUFFER_PACKED);
             expect(litertTensor.accelerator).toEqual('webgpu');
             tfjsTensor.dispose();
             litertTensor.delete();
           });
      });
    }
  });

  describe('litertToTfjs', () => {
    it('throws when converting a WebGPU tensor without the TFJS WebGPU backend',
       async () => {
         await tf.setBackend('cpu');
         try {
           const litertTensor =
               await makeFloat32LiteRtTensor([1, 2, 3]).moveTo('webgpu');
           expect(() => litertToTfjs(litertTensor))
               .toThrowError(
                   TensorConversionError,
                   /LiteRT WebGPU tensors can only be converted to TFJS WebGPU tensors/);
           litertTensor.delete();
         } finally {
           await resetLiteRt();
         }
       });


    for (const dimensionCount of [1, 2, 3, 4] as const) {
      describe(`with ${dimensionCount} dimensions`, () => {
        const shape = new Array(dimensionCount).fill(3);

        it('converts a LiteRT Float32 CPU tensor to a TFJS Float32 CPU tensor',
           async () => {
             await tf.setBackend('cpu');
             try {
               const litertTensor = makeFloat32LiteRtTensor(shape);
               const tfjsTensor = litertToTfjs(litertTensor);
               expect(tfjsTensor.dtype).toEqual('float32');
               expect(tfjsTensor.shape).toEqual(shape);
               expect(await tfjsTensor.data())
                   .toEqual(await litertTensor.data());
               tfjsTensor.dispose();
               litertTensor.delete();
             } finally {
               await resetLiteRt();
             }
           });

        it('converts a LiteRT Float32 WebGPU tensor to a TFJS Float32 WebGPU tensor',
           async () => {
             const litertTensor =
                 await makeFloat32LiteRtTensor(shape).moveTo('webgpu');
             const tfjsTensor = litertToTfjs(litertTensor);
             expect(tfjsTensor.dtype).toEqual('float32');
             expect(tfjsTensor.shape).toEqual(shape);
             expect(await tfjsTensor.data()).toEqual(await litertTensor.data());
             tfjsTensor.dispose();
             litertTensor.delete();
           });

        it('converts a LiteRT Int32 CPU tensor to a TFJS Int32 CPU tensor',
           async () => {
             await tf.setBackend('cpu');
             try {
               const litertTensor = makeInt32LiteRtTensor(shape);
               const tfjsTensor = litertToTfjs(litertTensor);
               expect(tfjsTensor.dtype).toEqual('int32');
               expect(tfjsTensor.shape).toEqual(shape);
               expect(await tfjsTensor.data())
                   .toEqual(await litertTensor.data());
               tfjsTensor.dispose();
               litertTensor.delete();
             } finally {
               await resetLiteRt();
             }
           });

        it('converts a LiteRT Int32 WebGPU tensor to a TFJS Int32 WebGPU tensor',
           async () => {
             const litertTensor =
                 await makeInt32LiteRtTensor(shape).moveTo('webgpu');
             const tfjsTensor = litertToTfjs(litertTensor);
             expect(tfjsTensor.dtype).toEqual('int32');
             expect(tfjsTensor.shape).toEqual(shape);
             expect(await tfjsTensor.data()).toEqual(await litertTensor.data());
             tfjsTensor.dispose();
             litertTensor.delete();
           });
      });
    }
  });

  describe('runWithTfjsTensors', () => {
    beforeAll(async () => {
      await resetLiteRt();
    });

    it('can run a wasm model when TFJS WebGPU input tensors are not on CPU',
       async () => {
         const modelPath = '/testdata/add_10x10.tflite';
         const model =
             await litert.loadAndCompile(modelPath, {accelerator: 'wasm'});
         const tfjsGpuTensor = tf.tidy(
             () => tf.range(0, 100, 1, 'float32').reshape([10, 10]).pow(2));
         const outputs =
             await runWithTfjsTensors(model, [tfjsGpuTensor, tfjsGpuTensor]);
         expect(outputs.length).toBe(1);
         const expectedOutput = tfjsGpuTensor.add(tfjsGpuTensor);
         expect(await outputs[0].data()).toEqual(await expectedOutput.data());
         tfjsGpuTensor.dispose();
         expectedOutput.dispose();
         model.delete();
       });

    it('can run a webgpu model when TFJS inputs are on CPU', async () => {
      await tf.setBackend('cpu');
      try {
        const modelPath = '/testdata/add_10x10.tflite';
        const model =
            await litert.loadAndCompile(modelPath, {accelerator: 'webgpu'});
        const tfjsCpuTensor = tf.tidy(
            () => tf.range(0, 100, 1, 'float32').reshape([10, 10]).pow(2));

        // Now that inputs have been created on CPU, switch to WebGPU so we can
        // store the models WebGPU outputs.
        await tf.setBackend('webgpu');

        const outputs =
            await runWithTfjsTensors(model, [tfjsCpuTensor, tfjsCpuTensor]);
        expect(outputs.length).toBe(1);
        const expectedOutput = tfjsCpuTensor.add(tfjsCpuTensor);
        expect(await outputs[0].data()).toEqual(await expectedOutput.data());
        tfjsCpuTensor.dispose();
        expectedOutput.dispose();
        model.delete();
      } finally {
        await resetLiteRt();
      }
    });

    for (const accelerator of ['webgpu', 'wasm'] as const) {
      describe(accelerator, () => {
        let model: litert.CompiledModel;
        let inputs: Record<string, tf.Tensor>;
        let expectedOutput: tf.Tensor;

        beforeAll(async () => {
          try {
            model = await litert.loadAndCompile(
                '/testdata/multi_signature_model.tflite', {accelerator});
            inputs = {
              'a': tf.diag(tf.ones([10], 'float32')),
              'b': tf.range(0, 100, 1, 'float32').reshape([10, 10]),
            };
            expectedOutput = inputs['a'].add(inputs['b']);
            await inputs['a'].data();
            await inputs['b'].data();
          } catch (e) {
            console.error(e);
            throw e;
          }
        });

        afterAll(() => {
          model.delete();
          for (const value of Object.values(inputs)) {
            value.dispose();
          }
          expectedOutput.dispose();
        });

        it('runs a model with TFJS inputs and outputs', async () => {
          const outputs = await runWithTfjsTensors(model, inputs);

          expect(await outputs['output'].data())
              .toEqual(await expectedOutput.data());
        });

        it('runs a signature with TFJS inputs and outputs', async () => {
          const outputs = await runWithTfjsTensors(model, 'add', inputs);

          expect(await outputs['output'].data())
              .toEqual(await expectedOutput.data());
        });

        it('runs with `run([Tensor...]) calling style', async () => {
          const outputs =
              await runWithTfjsTensors(model, [inputs['a'], inputs['b']]);

          expect(await outputs[0].data()).toEqual(await expectedOutput.data());
        });

        it('runs a signature that\'s passed directly to `run`', async () => {
          const outputs = await runWithTfjsTensors(model.signatures['add'], [
            inputs['a'],
            inputs['b'],
          ]);

          expect(await outputs[0].data()).toEqual(await expectedOutput.data());
        });
      });
    }
  });
});

function makeFloat32TfjsTensor(shape: number[]): tf.Tensor {
  return tf.tidy(() => {
    const size = shape.reduce((a, b) => a * b, 1);
    return tf.range(0, size, 1, 'float32').div(10).reshape(shape).pow(2);
  });
}

function makeInt32TfjsTensor(shape: number[]): tf.Tensor {
  return tf.tidy(() => {
    const size = shape.reduce((a, b) => a * b, 1);
    return tf.range(0, size, 1, 'int32').reshape(shape).pow(2);
  });
}

function makeFloat32LiteRtTensor(shape: number[]): litert.Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = i / 10;
  }
  return new litert.Tensor(data, shape);
}

function makeInt32LiteRtTensor(shape: number[]): litert.Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Int32Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = i;
  }
  return new litert.Tensor(data, shape);
}
