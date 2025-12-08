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

import '@tensorflow/tfjs-backend-webgpu'; // DO NOT REMOVE: Requried for side effects.

import {CompiledModel, LiteRt, loadAndCompile, loadLiteRt, unloadLiteRt} from '@litertjs/core';
import {litertToTfjs, runWithTfjsTensors, TensorConversionError, tfjsToLitert} from '@litertjs/tfjs-interop';
import {type WebGPUBackend} from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// Placeholder for internal dependency on trusted resource url

describe('TFJS Interop', () => {
  let liteRt: LiteRt;

  async function resetLiteRt() {
    unloadLiteRt();
    liteRt = await loadLiteRt('/wasm');

    // Share the same WebGPU device with TFJS.
    await tf.ready();
    const backend = tf.backend() as WebGPUBackend;
    const device = backend.device;
    // TF.js AdapterInfo doesn't match GPUAdapterInfo, so we fudge it a bit for
    // now. However, once all Chrome versions we test on have migrated to
    // supporting just `device.adapterInfo`, then this workaround will no longer
    // be needed. So we use this for testing on Linux-dev, but publicly our API
    // will assume users are using just `setWebGpuDevice(device)` for TF.js
    // integrations. TODO: Remove once this workaround is obsolete.
    const adapterInfo = Object.assign(
                            {
                              device,
                              description: '',
                              __brand: '',
                            },
                            backend.adapterInfo) as unknown as GPUAdapterInfo;
    liteRt.setWebGpuDevice(device, adapterInfo);
  }

  beforeAll(async () => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 10000;
    try {
      await tf.setBackend('webgpu');
      await resetLiteRt();
    } catch (e) {
      console.error('!!!beforeAll failed!!!', e);
      throw e;
    }
  });

  describe('tfjsToLitert', () => {
    it('throws when TFJS WebGPU tensor data is not on CPU', async () => {
      tf.tidy(() => {
        const tfjsGpuTensor = tf.range(0, 10, 1, 'float32').pow(2);
        expect(() => tfjsToLitert(tfjsGpuTensor, 'wasm'))
            .toThrowError(
                TensorConversionError,
                /TFJS tensor data is on WebGPU but not on CPU/);
      });
    });
  });

  describe('tfjsToLitert and litertToTfjs', () => {
    for (const dimensionCount of [1, 2, 3, 4] as const) {
      describe(`with ${dimensionCount} dimensions`, () => {
        const batchDims = new Array(dimensionCount - 1).fill(1);
        const floatShape = [...batchDims, 3];
        const intShape = [...batchDims, 4];

        it('converts Float32 tensors between TFJS and LiteRT WASM',
           async () => {
             const tfjsTensor =
                 tf.tensor(new Float32Array([1.234, 2.345, 3.456]), floatShape);
             await tfjsTensor.data();
             const litertTensor = tfjsToLitert(tfjsTensor, 'wasm');
             const tfjsTensor2 = litertToTfjs(litertTensor);
             expect(await tfjsTensor2.data()).toEqual(await tfjsTensor.data());
             tfjsTensor.dispose();
             litertTensor.delete();
             tfjsTensor2.dispose();
           });

        it('converts Float32 tensors between TFJS and LiteRT WebGPU',
           async () => {
             const tfjsTensor =
                 tf.tensor(new Float32Array([1.234, 2.345, 3.456]), floatShape);
             await tfjsTensor.data();
             const litertTensor = tfjsToLitert(tfjsTensor, 'webgpu');
             const tfjsTensor2 = litertToTfjs(litertTensor);
             expect(await tfjsTensor2.data()).toEqual(await tfjsTensor.data());
             tfjsTensor.dispose();
             litertTensor.delete();
             tfjsTensor2.dispose();
           });

        it('converts Int32 tensors between TFJS and LiteRT WASM', async () => {
          const tfjsTensor = tf.tensor(
              new Int32Array([1, 2, 3, 2147483647]), intShape, 'int32');
          await tfjsTensor.data();
          const litertTensor = tfjsToLitert(tfjsTensor, 'wasm');
          const tfjsTensor2 = litertToTfjs(litertTensor);
          expect(await tfjsTensor2.data()).toEqual(await tfjsTensor.data());
          tfjsTensor.dispose();
          litertTensor.delete();
          tfjsTensor2.dispose();
        });

        it('converts Int32 tensors between TFJS and LiteRT WebGPU',
           async () => {
             const tfjsTensor = tf.tensor(
                 new Int32Array([1, 2, 3, 2147483647]), intShape, 'int32');
             await tfjsTensor.data();
             const litertTensor = tfjsToLitert(tfjsTensor, 'webgpu');
             const tfjsTensor2 = litertToTfjs(litertTensor);
             expect(await tfjsTensor2.data()).toEqual(await tfjsTensor.data());
             tfjsTensor.dispose();
             litertTensor.delete();
             tfjsTensor2.dispose();
           });
      });
    }
  });

  describe('runWithTfjsTensors', () => {
    it('throws when TFJS WebGPU input tensors are not on CPU', async () => {
      const modelPath = '/testdata/add_10x10.tflite';
      const model = await loadAndCompile(modelPath, {accelerator: 'wasm'});
      const tfjsGpuTensor = tf.range(0, 10, 1, 'float32').pow(2);
      expect(() => runWithTfjsTensors(model, [tfjsGpuTensor, tfjsGpuTensor]))
          .toThrowError(
              TensorConversionError,
              /TFJS tensor data is on WebGPU but not on CPU/);
    });

    for (const accelerator of ['webgpu', 'wasm'] as const) {
      describe(accelerator, () => {
        let model: CompiledModel;
        let inputs: Record<string, tf.Tensor>;
        let expectedOutput: tf.Tensor;

        beforeAll(async () => {
          try {
            model = await loadAndCompile(
                '/testdata/multi_signature_model.tflite', {accelerator});
            inputs = {
              'add_a:0': tf.diag(tf.ones([10], 'float32')),
              'add_b:0': tf.range(0, 100, 1, 'float32').reshape([10, 10]),
            };
            expectedOutput = inputs['add_a:0'].add(inputs['add_b:0']);
            await inputs['add_a:0'].data();
            await inputs['add_b:0'].data();
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
          const outputs = runWithTfjsTensors(model, inputs);

          expect(await outputs['PartitionedCall:0'].data())
              .toEqual(await expectedOutput.data());
        });

        it('runs a signature with TFJS inputs and outputs', async () => {
          const outputs = runWithTfjsTensors(model, 'add', inputs);

          expect(await outputs['PartitionedCall:0'].data())
              .toEqual(await expectedOutput.data());
        });

        it('runs with `run([Tensor...]) calling style', async () => {
          const outputs =
              runWithTfjsTensors(model, [inputs['add_a:0'], inputs['add_b:0']]);

          expect(await outputs[0].data()).toEqual(await expectedOutput.data());
        });

        it('runs a signature that\'s passed directly to `run`', async () => {
          const outputs = runWithTfjsTensors(model.signatures['add'], [
            inputs['add_a:0'],
            inputs['add_b:0'],
          ]);

          expect(await outputs[0].data()).toEqual(await expectedOutput.data());
        });
      });
    }
  });
});
