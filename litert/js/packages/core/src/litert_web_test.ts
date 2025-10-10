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

import {CompiledModel, ErrorReporter, getAdapterInfo, getGlobalLiteRt, getGlobalLiteRtPromise, getWebGpuDevice, isWebGPUSupported, LiteRt, loadAndCompile, loadLiteRt, type LoadLiteRtOptions, setErrorReporter, setWebGpuDevice, Tensor, TensorTypeError, unloadLiteRt} from '@litertjs/core';
import {litertToTfjs, runWithTfjsTensors, TensorConversionError, tfjsToLitert} from '@litertjs/tfjs-interop';
import {type WebGPUBackend} from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// Placeholder for internal dependency on trusted resource url

describe('LiteRt', () => {
  let liteRt: LiteRt;
  let device: GPUDevice;

  async function resetLiteRt(
      loadFromDirectory = false, {threads = false}: LoadLiteRtOptions = {}) {
    unloadLiteRt();
    if (loadFromDirectory) {
      liteRt = await loadLiteRt('/wasm', {threads});
    } else {
      liteRt =
          await loadLiteRt('/wasm/litert_wasm_internal.js');
    }

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
      // Log GPU info for debugging missing GPU issues in CI.
      console.log('GPU:', navigator.gpu);
      console.log('Adapter: ', await navigator.gpu.requestAdapter());
      console.log('setting webgpu backend');
      await tf.setBackend('webgpu');
      console.log('webgpu backend set');

      // We don't reset LiteRt on every test because it's expensive.
      console.log('resetting liteRt');
      await resetLiteRt();
      console.log('liteRt reset');
    } catch (e) {
      console.error('!!!beforeAll failed!!!', e);
      throw e;
    }
  });

  it('loads the WASM module from its js file', async () => {
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

  it('loads the threaded WASM module', async () => {
    try {
      await resetLiteRt(/* loadFromDirectory= */ true, {threads: true});
      expect(liteRt).toBeDefined();
    } finally {
      await resetLiteRt();
    }
  });

  it('can create its own WebGPU device', async () => {
    // TODO: b/434244321 - Move tests that affect global state to a different
    // describe block instead of using try/finally.
    try {
      unloadLiteRt();
      liteRt = await loadLiteRt('/wasm');

      // Load a model to create the WebGPU device.
      const model = await loadAndCompile(
          '/testdata/add_10x10.tflite', {accelerator: 'webgpu'});
      model.delete();
      expect(await getWebGpuDevice()).toBeDefined();
    } finally {
      await resetLiteRt();
    }
  });

  describe('loadAndCompile', () => {
    // Some of these tests that intentionally cause C++ exceptions leave
    // LiteRT's wasm in a bad state. Reset it after each test to prevent
    // cascading failures.
    afterEach(resetLiteRt);
    const modelPath = '/testdata/add_10x10.tflite';  // A small test model.

    it('loads from a Uint8Array', async () => {
      const modelData = await fetch(modelPath);
      const model = await loadAndCompile(
          new Uint8Array(await modelData.arrayBuffer()),
          {accelerator: 'webgpu'});
      expect(model).toBeDefined();
      model.delete();
    });

    it('loads from a string URL', async () => {
      const model = await loadAndCompile(modelPath, {accelerator: 'webgpu'});
      expect(model).toBeDefined();
      model.delete();
    });

    it('loads from a URL object', async () => {
      const modelUrl = new URL(`${location.origin}${modelPath}`);
      const model = await loadAndCompile(modelUrl, {accelerator: 'webgpu'});
      expect(model).toBeDefined();
      model.delete();
    });

    it('loads from a ReadableStreamDefaultReader', async () => {
      const modelData = await fetch(modelPath);
      const modelReader = modelData.body!.getReader();
      const model = await loadAndCompile(modelReader, {accelerator: 'webgpu'});
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

         await expectAsync(loadAndCompile(fakeModel.getReader(), {
           accelerator: 'webgpu'
         })).toBeRejectedWithError(/Model is too large/);
       });

    it('throws an error when given a bad model', async () => {
      const badModel =
          new TextEncoder().encode('****BADM and some extra data here.');

      await expectAsync(loadAndCompile(badModel, {
        accelerator: 'webgpu'
      })).toBeRejectedWithError(/Failed to build interpreter/);
    });
  });

  describe('input / output details', () => {
    let multiSignatureModel: CompiledModel;

    beforeAll(async () => {
      multiSignatureModel = await loadAndCompile(
          '/testdata/multi_signature_model.tflite', {accelerator: 'webgpu'});
    });

    afterAll(() => {
      multiSignatureModel.delete();
    });

    it('gets input details about the default signature', async () => {
      const inputDetails = multiSignatureModel.getInputDetails();
      expect(inputDetails).toEqual([
        {
          name: 'add_a:0',
          index: 0,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
        },
        {
          name: 'add_b:0',
          index: 1,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
        },
      ]);
    });

    it('gets output details about the default signature', async () => {
      const outputDetails = multiSignatureModel.getOutputDetails();
      expect(outputDetails).toEqual([
        {
          name: 'PartitionedCall:0',
          index: 0,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
        },
      ]);
    });

    it('gets input details about a specific signature', async () => {
      const inputDetails =
          multiSignatureModel.signatures['mul'].getInputDetails();
      expect(inputDetails).toEqual([
        {
          name: 'mul_a:0',
          index: 0,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
        },
        {
          name: 'mul_b:0',
          index: 1,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
        },
      ]);
    });

    it('gets output details about a specific signature', async () => {
      const outputDetails =
          multiSignatureModel.signatures['mul'].getOutputDetails();
      expect(outputDetails).toEqual([
        {
          name: 'PartitionedCall_1:0',
          index: 0,
          dtype: 'float32',
          shape: new Int32Array([10, 10]),
        },
      ]);
    });
  });

  describe('run', () => {
    for (const accelerator of ['webgpu', 'wasm'] as const) {
      describe(accelerator, () => {
        let mobilenetModel: CompiledModel;
        let multiSignatureModel: CompiledModel;
        let mixedInputModel: CompiledModel;

        let identityTfjs: tf.Tensor;
        let rangeTfjs: tf.Tensor;
        let identity: Tensor;
        let range: Tensor;

        beforeAll(async () => {
          mobilenetModel = await loadAndCompile(
              '/testdata/torchvision_mobilenet_v2.tflite', {accelerator});

          multiSignatureModel = await loadAndCompile(
              '/testdata/multi_signature_model.tflite', {accelerator});

          mixedInputModel = await loadAndCompile(
              '/testdata/mixed_input_model.tflite', {accelerator});

          identityTfjs = tf.diag(tf.ones([10], 'float32'));
          await identityTfjs.data();
          identity = tfjsToLitert(identityTfjs, accelerator);

          rangeTfjs = tf.range(0, 100, 1, 'float32').reshape([10, 10]);
          await rangeTfjs.data();
          range = tfjsToLitert(rangeTfjs, accelerator);
        });

        afterAll(() => {
          mobilenetModel.delete();
          multiSignatureModel.delete();
          identityTfjs.dispose();
          identity.delete();
          rangeTfjs.dispose();
          range.delete();
        });

        it('runs a model multiple times', async () => {
          const fakeInput = tf.zeros([1, 3, 224, 224], 'float32');

          const outputs = [];
          for (let i = 0; i < 10; ++i) {
            // The model should not crash when run multiple times.
            outputs.push(runWithTfjsTensors(mobilenetModel, {
              'serving_default_args_0:0': fakeInput,
            }));
          }

          const outputsData = await Promise.all(outputs.map(
              output => output['StatefulPartitionedCall:0'].data()));

          // All the outputs should be the same.
          for (let i = 1; i < outputsData.length; ++i) {
            expect(outputsData[i]).toEqual(outputsData[0]);
          }
        });

        it('runs a model with mixed input types', async () => {
          const int32Input = tf.range(0, 10, 1, 'int32').reshape([1, 10]);
          const float32Input = tf.range(0, 10, 1, 'float32').reshape([1, 10]);

          const outputs = runWithTfjsTensors(mixedInputModel, {
            'int_input': int32Input,
            'float_input': float32Input,
          });

          const outputData = await outputs['Identity'].data();
          const expectedOutput = tf.tidy(() => {
            return tf.add(int32Input, float32Input).mul(2);
          });
          expect(outputData).toEqual(await expectedOutput.data());
          int32Input.dispose();
          float32Input.dispose();
          expectedOutput.dispose();
          outputs['Identity'].dispose();
        });

        it('gives meaningful errors on shape mismatch', async () => {
          const badShape = tf.zeros([1, 3, 224, 225], 'float32');
          expect(() => runWithTfjsTensors(mobilenetModel, {
                   'serving_default_args_0:0': badShape,
                 }))
              .toThrowError(
                  /Source .*1, 3, 224, 225.*dest input.*1, 3, 224, 224|Input.*1, 3, 224, 225.*expects.*1, 3, 224, 224/);
        });

        it('output tensors have their accelerator set correctly', async () => {
          const fakeInputTfjs = tf.zeros([1, 3, 224, 224], 'float32');
          const fakeInput = tfjsToLitert(fakeInputTfjs, accelerator);

          const outputs = mobilenetModel.run({
            'serving_default_args_0:0': fakeInput,
          });

          expect(outputs['StatefulPartitionedCall:0']).toBeDefined();
          expect(outputs['StatefulPartitionedCall:0'].accelerator)
              .toEqual(accelerator);
        });

        it('runs multiple signatures of a model', async () => {
          // "add" signature
          const addOutputs = multiSignatureModel.run(
              'add', {'add_a:0': identity, 'add_b:0': range});
          const addOutputsAsTfjs =
              litertToTfjs(addOutputs['PartitionedCall:0']);
          expect(await addOutputsAsTfjs.data())
              .toEqual(await rangeTfjs.add(identityTfjs).data());

          // "sub" signature
          const subOutputs = multiSignatureModel.run(
              'sub', {'sub_a:0': identity, 'sub_b:0': range});
          const subOutputsAsTfjs =
              litertToTfjs(subOutputs['PartitionedCall_2:0']);
          expect(await subOutputsAsTfjs.data())
              .toEqual(await identityTfjs.sub(rangeTfjs).data());

          // "mul" signature
          const mulOutputs = multiSignatureModel.run(
              'mul', {'mul_a:0': identity, 'mul_b:0': range});
          const mulOutputsAsTfjs =
              litertToTfjs(mulOutputs['PartitionedCall_1:0']);
          expect(await mulOutputsAsTfjs.data())
              .toEqual(await identityTfjs.mul(rangeTfjs).data());
        });

        it('`run(Tensor)` returns `Tensor[]`', async () => {
          const fakeInput = tf.zeros([1, 3, 224, 224], 'float32');
          const fakeInputAsLiteRt = tfjsToLitert(fakeInput, accelerator);

          const outputs = mobilenetModel.run(fakeInputAsLiteRt);
          const expectedOutputs = mobilenetModel.run({
            'serving_default_args_0:0': fakeInputAsLiteRt,
          });

          const outputAsTfjs = litertToTfjs(outputs[0]);
          const expectedOutputAsTfjs =
              litertToTfjs(expectedOutputs['StatefulPartitionedCall:0']);

          expect(await outputAsTfjs.data())
              .toEqual(await expectedOutputAsTfjs.data());
        });

        it('`run([Tensor, ...])` returns `Tensor[]`', async () => {
          const outputs = multiSignatureModel.run([identity, range]);

          const outputAsTfjs = litertToTfjs(outputs[0]);
          expect(await outputAsTfjs.data())
              .toEqual(await tf.add(identityTfjs, rangeTfjs).data());
        });

        it('`run({\'input\': Tensor, ...})` returns `{\'output\': Tensor, ...}`',
           async () => {
             const outputs = multiSignatureModel.run(
                 {'add_a:0': identity, 'add_b:0': range});

             const outputAsTfjs = litertToTfjs(outputs['PartitionedCall:0']);
             expect(await outputAsTfjs.data())
                 .toEqual(await tf.add(identityTfjs, rangeTfjs).data());
           });

        it('`run(\'signature\', Tensor)` returns `Tensor[]`', async () => {
          const fakeInput = tf.zeros([1, 3, 224, 224], 'float32');
          const fakeInputAsLiteRt = tfjsToLitert(fakeInput, accelerator);

          const outputs =
              mobilenetModel.run('serving_default', fakeInputAsLiteRt);
          const expectedOutputs = mobilenetModel.run({
            'serving_default_args_0:0': fakeInputAsLiteRt,
          });

          const outputAsTfjs = litertToTfjs(outputs[0]);
          const expectedOutputAsTfjs =
              litertToTfjs(expectedOutputs['StatefulPartitionedCall:0']);

          expect(await outputAsTfjs.data())
              .toEqual(await expectedOutputAsTfjs.data());
        });

        it('`run(\'signature\', [Tensor, ...])` returns `Tensor[]`',
           async () => {
             const outputs = multiSignatureModel.run('add', [identity, range]);

             const outputAsTfjs = litertToTfjs(outputs[0]);
             expect(await outputAsTfjs.data())
                 .toEqual(await tf.add(identityTfjs, rangeTfjs).data());
           });

        it('`run(\'signature\', {\'input\': ...})` returns `{\'output\': ...}`',
           async () => {
             const outputs = multiSignatureModel.run(
                 'add', {'add_a:0': identity, 'add_b:0': range});

             const outputAsTfjs = litertToTfjs(outputs['PartitionedCall:0']);
             expect(await outputAsTfjs.data())
                 .toEqual(await tf.add(identityTfjs, rangeTfjs).data());
           });

        it('`run(Tensor)` throws if the model expects more tensors', () => {
          expect(() => multiSignatureModel.run([identity]))
              .toThrowError(/called with 1.*expects 2/);
        });

        it('throws an error if the tensor is of the wrong type', async () => {
          const fakeInput = tf.zeros([1, 3, 224, 224], 'int32');
          const fakeInputAsLiteRt = tfjsToLitert(fakeInput, accelerator);
          expect(() => mobilenetModel.run(fakeInputAsLiteRt))
              .toThrowError(
                  TensorTypeError,
                  /Input.*serving_default_args_0:0.*position 0.*has type int32.*signature expects float32/);
        });

        it('throws an error if the model has been deleted', async () => {
          const fakeInput = tf.zeros([10, 10], 'float32');
          const fakeInputAsLiteRt = tfjsToLitert(fakeInput, accelerator);
          const deletableModel = await loadAndCompile(
              '/testdata/multi_signature_model.tflite', {accelerator});
          deletableModel.delete();
          expect(() => deletableModel.run(fakeInputAsLiteRt))
              .toThrowError(
                  Error, /Model has been deleted. Please reload the model./);
        });

        it('throws an error if the signature has been deleted', async () => {
          const fakeInput = tf.zeros([10, 10], 'float32');
          const fakeInputAsLiteRt = tfjsToLitert(fakeInput, accelerator);
          const deletableModel = await loadAndCompile(
              '/testdata/multi_signature_model.tflite', {accelerator});
          deletableModel.signatures['add'].delete();
          expect(() => deletableModel.run('add', fakeInputAsLiteRt))
              .toThrowError(
                  Error,
                  /Signature has been deleted. Please reload the model./);
        });
      });
    }
  });

  it('supports custom error reporting', async () => {
    const loggedErrors: string[] = [];
    const customErrorReporter: ErrorReporter = (message: string) => {
      loggedErrors.push(message);
      return new Error(`Custom error: ${message}`);
    };

    setErrorReporter(customErrorReporter);

    // Create an error by loading a bad model.
    const badModel =
        new TextEncoder().encode('****BADM and some extra data here.');

    await expectAsync(loadAndCompile(badModel, {
      accelerator: 'webgpu',
    })).toBeRejectedWithError(/Custom error: /);
    expect(loggedErrors).toEqual(['Failed to build interpreter.']);

    // Reset LiteRt because we changed the error reporter.
    await resetLiteRt();
  });

  it('calls multiple signatures of a model', async () => {
    const modelPath = '/testdata/multi_signature_model.tflite';
    const model = await loadAndCompile(modelPath, {accelerator: 'webgpu'});
    expect(model).toBeDefined();

    const identity = tf.diag(tf.ones([10], 'float32'));
    const a = tfjsToLitert(identity);

    const range = tf.range(0, 100, 1, 'float32').reshape([10, 10]);
    const b = tfjsToLitert(range);

    // "add" signature
    const addOutputs = model.run('add', {'add_a:0': a, 'add_b:0': b});
    const addOutputsAsTfjs = litertToTfjs(addOutputs['PartitionedCall:0']);
    expect(await addOutputsAsTfjs.data())
        .toEqual(await range.add(identity).data());

    // "sub" signature
    const subOutputs = model.run('sub', {'sub_a:0': a, 'sub_b:0': b});
    const subOutputsAsTfjs = litertToTfjs(subOutputs['PartitionedCall_2:0']);
    expect(await subOutputsAsTfjs.data())
        .toEqual(await identity.sub(range).data());

    // "mul" signature
    const mulOutputs = model.run('mul', {'mul_a:0': a, 'mul_b:0': b});
    const mulOutputsAsTfjs = litertToTfjs(mulOutputs['PartitionedCall_1:0']);
    expect(await mulOutputsAsTfjs.data())
        .toEqual(await identity.mul(range).data());

    model.delete();
  });

  it('sets WebGPU device', async () => {
    if (!isWebGPUSupported()) {
      throw new Error('This browser does not support WebGPU.');
    }

    // We reset LiteRt state because otherwise we could throw an error when
    // changing WebGPU device, depending on whether or not `loadAndCompile` has
    // been called yet.
    await resetLiteRt();

    const oldDevice = await getWebGpuDevice();
    const oldAdapterInfo = await getAdapterInfo();
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('No GPU adapter found.');
    }
    const adapterInfo = adapter.info;
    const device = await adapter.requestDevice();

    // TODO: Remove adapterInfo from the api, as the latest GPUDevice type
    // should have adapterInfo.
    setWebGpuDevice(device, adapterInfo);
    await expectAsync(getWebGpuDevice()).toBeResolvedTo(device);
    await expectAsync(getAdapterInfo()).toBeResolvedTo(adapterInfo);
    setWebGpuDevice(oldDevice, oldAdapterInfo);
  });

  it('supports 1D, 2D, 3D, and 4D input tensors', async () => {
    const modelPath = '/testdata/add_1d_2d_3d_4d.tflite';
    const model = await loadAndCompile(modelPath, {accelerator: 'webgpu'});
    expect(model).toBeDefined();

    const x = tf.ones([4], 'float32');
    const y = tf.ones([3, 4], 'float32');
    const z = tf.ones([2, 3, 4], 'float32');
    const w = tf.ones([1, 2, 3, 4], 'float32');
    const inputs = {
      'x': tfjsToLitert(x),
      'y': tfjsToLitert(y),
      'z': tfjsToLitert(z),
      'w': tfjsToLitert(w),
    };

    // "add" signature
    const outputs = model.run(inputs);
    const outputAsLiteRt = outputs['Identity'];
    const outputAsTfjs = litertToTfjs(outputAsLiteRt);
    expect(await outputAsTfjs.data())
        .toEqual(await x.add(y).add(z).add(w).data());

    model.delete();
  });

  it('supports 1D tfjs tensors with the last dimension not divisible by 4',
     async () => {
       const modelPath = '/testdata/add_c1_c7.tflite';
       const model = await loadAndCompile(modelPath, {accelerator: 'webgpu'});

       const x = tf.ones([1], 'float32');
       const y = tf.ones([7], 'float32');
       const inputs = {
         'x': tfjsToLitert(x),
         'y': tfjsToLitert(y),
       };

       const outputs = model.run(inputs);
       const outputAsLiteRt = outputs['Identity'];
       const outputAsTfjs = litertToTfjs(outputAsLiteRt);
       const outputData = await outputAsTfjs.data();
       const expectedData = await x.add(y).data();

       expect(outputData).toEqual(expectedData);

       model.delete();
     });

  it('supports 2D tfjs tensors with the last dimension not divisible by 4',
     async () => {
       const modelPath = '/testdata/add_10x10.tflite';
       const model = await loadAndCompile(modelPath, {accelerator: 'webgpu'});

       const identity = tf.diag(tf.ones([10], 'float32'));
       const a = tfjsToLitert(identity);

       const range = tf.range(0, 100, 1, 'float32').reshape([10, 10]);
       const b = tfjsToLitert(range);
       const outputs = model.run({'a': a, 'b': b});

       const outputAsLiteRt = outputs['Identity'];
       const outputAsTfjs = litertToTfjs(outputAsLiteRt);
       const outputData = await outputAsTfjs.data();
       const expectedData = await identity.add(range).data();

       expect(outputData).toEqual(expectedData);

       model.delete();
     });


  describe('TFJS Interop', () => {
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
            const outputs = runWithTfjsTensors(
                model, [inputs['add_a:0'], inputs['add_b:0']]);

            expect(await outputs[0].data())
                .toEqual(await expectedOutput.data());
          });

          it('runs a signature that\'s passed directly to `run`', async () => {
            const outputs = runWithTfjsTensors(model.signatures['add'], [
              inputs['add_a:0'],
              inputs['add_b:0'],
            ]);

            expect(await outputs[0].data())
                .toEqual(await expectedOutput.data());
          });
        });
      }
    });
  });

  describe('tensor conversion functions', () => {
    for (const dimensionCount of [1, 2, 3, 4] as const) {
      describe(`with ${dimensionCount} dimensions`, () => {
        const batchDims = new Array(dimensionCount - 1).fill(1);

        it('copies a Float32 tensor from CPU to WebGPU and back', async () => {
          const cpuTensor = Tensor.fromTypedArray(
              new Float32Array([1.234, 2.345, 3.456]), [...batchDims, 3]);
          const gpuTensor = await cpuTensor.copyTo('webgpu');
          const cpuTensor2 = await gpuTensor.copyTo('wasm');
          expect(cpuTensor2.toTypedArray()).toEqual(cpuTensor.toTypedArray());
          expect(cpuTensor.deleted).toBeFalse();
          expect(gpuTensor.deleted).toBeFalse();
        });

        it('copies an Int32 tensor from CPU to WebGPU and back', async () => {
          const cpuTensor = Tensor.fromTypedArray(
              new Int32Array([1, 2, 3, 2147483647]), [...batchDims, 4]);
          const gpuTensor = await cpuTensor.copyTo('webgpu');
          const cpuTensor2 = await gpuTensor.copyTo('wasm');
          expect(cpuTensor2.toTypedArray()).toEqual(cpuTensor.toTypedArray());
          expect(cpuTensor.deleted).toBeFalse();
          expect(gpuTensor.deleted).toBeFalse();
        });

        it('moves a Float32 tensor from CPU to WebGPU and back', async () => {
          const data = new Float32Array([1.234, 2.345, 3.456]);
          const cpuTensor = Tensor.fromTypedArray(data, [...batchDims, 3]);
          const gpuTensor = await cpuTensor.moveTo('webgpu');
          const cpuTensor2 = await gpuTensor.moveTo('wasm');
          expect(cpuTensor2.toTypedArray()).toEqual(data);
          expect(cpuTensor.deleted).toBeTrue();
          expect(gpuTensor.deleted).toBeTrue();
        });

        it('moves an Int32 tensor from CPU to WebGPU and back', async () => {
          const data = new Int32Array([1, 2, 3, 2147483647]);
          const cpuTensor = Tensor.fromTypedArray(data, [...batchDims, 4]);
          const gpuTensor = await cpuTensor.moveTo('webgpu');
          const cpuTensor2 = await gpuTensor.moveTo('wasm');
          expect(cpuTensor2.toTypedArray()).toEqual(data);
          expect(cpuTensor.deleted).toBeTrue();
          expect(gpuTensor.deleted).toBeTrue();
        });
      });
    }

    it('supports 1D tensors with the last dimension not divisible by 4',
       async () => {
         const modelPath = '/testdata/add_c1_c7.tflite';
         const model = await loadAndCompile(modelPath, {accelerator: 'webgpu'});

         const x = new Float32Array([1.0]);
         const y = new Float32Array(new Array(7).fill(1.0));

         const inputs = {
           'x': await Tensor.fromTypedArray(x, [1]).moveTo('webgpu'),
           'y': await Tensor.fromTypedArray(y, [7]).moveTo('webgpu'),
         };

         const outputs = model.run(inputs);
         const output = outputs['Identity'];
         const outputCpu = await output.moveTo('wasm');
         const outputData = outputCpu.toTypedArray();

         const expectedData = new Float32Array(new Array(7).fill(2.0));
         expect(outputData).toEqual(expectedData);

         model.delete();
       });

    it('supports 2D tensors with the last dimension not divisible by 4',
       async () => {
         const modelPath = '/testdata/add_10x10.tflite';
         const model = await loadAndCompile(modelPath, {accelerator: 'webgpu'});

         const identity = new Float32Array(100);
         for (let i = 0; i < 10; i++) {
           for (let j = 0; j < 10; j++) {
             identity[i * 10 + j] = 1.0;
           }
         }
         const a =
             await Tensor.fromTypedArray(identity, [10, 10]).moveTo('webgpu');

         const range = new Float32Array(100);
         for (let i = 0; i < 100; i++) {
           range[i] = i;
         }
         const b =
             await Tensor.fromTypedArray(range, [10, 10]).moveTo('webgpu');

         const outputs = model.run({'a': a, 'b': b});
         const output = outputs['Identity'];
         const outputCpu = await output.moveTo('wasm');
         const outputData = await outputCpu.toTypedArray();

         const expectedData = new Float32Array(100);
         for (let i = 0; i < 100; i++) {
           expectedData[i] = identity[i] + range[i];
         }

         expect(outputData).toEqual(expectedData);

         model.delete();
       });

    it('can pass a TypedArray to the Tensor constructor', async () => {
      const data = new Float32Array([1.234, 2.345, 3.456]);
      const cpuTensor = new Tensor(data);
      expect(cpuTensor.type.dtype).toEqual('float32');
      expect(cpuTensor.type.layout.dimensions).toEqual([3]);
      expect(cpuTensor.accelerator).toEqual('wasm');
      expect(cpuTensor.toTypedArray()).toEqual(data);
    });

    it('can pass a TypedArray to the Tensor constructor with a shape',
       async () => {
         const data = new Int32Array([1, 2, 3, 4, 5, 6, 7, 8]);
         const cpuTensor = new Tensor(data, [2, 1, 2, 2]);
         expect(cpuTensor.type.dtype).toEqual('int32');
         expect(cpuTensor.type.layout.dimensions).toEqual([2, 1, 2, 2]);
         expect(cpuTensor.accelerator).toEqual('wasm');
         expect(cpuTensor.toTypedArray()).toEqual(data);
       });
  });

  describe('delegate compatibility', () => {
    let model: CompiledModel;

    beforeAll(async () => {
      model = await loadAndCompile(
          '/testdata/delegate_compatibility_test.tflite',
          {accelerator: 'webgpu'});
    });

    afterAll(() => {
      model.delete();
    });

    it('loads the model', async () => {
      expect(model).toBeDefined();
    });

    describe('matmul', () => {
      let a: tf.Tensor;
      let b: tf.Tensor;
      let expectedOutput: tf.Tensor;

      beforeAll(async () => {
        await tf.ready();
        a = tf.diag(tf.ones([10], 'float32'));
        b = tf.range(0, 100, 1, 'float32').reshape([10, 10]);
        expectedOutput = a.matMul(b);
      });

      afterAll(() => {
        a.dispose();
        b.dispose();
        expectedOutput.dispose();
      });

      it('works with 10x10 inputs', async () => {
        const outputs = runWithTfjsTensors(model, 'matmul_10x10', {
          'matmul_10x10_a:0': a.reshape([10, 10]),
          'matmul_10x10_b:0': b.reshape([10, 10]),
        });
        expect(await outputs['PartitionedCall:0'].data())
            .toEqual(await expectedOutput.data());
      });

      it('works with 1x10x10 inputs', async () => {
        const outputs = runWithTfjsTensors(model, 'matmul_1x10x10', {
          'matmul_1x10x10_a:0': a.reshape([1, 10, 10]),
          'matmul_1x10x10_b:0': b.reshape([1, 10, 10]),
        });
        expect(await outputs['PartitionedCall_1:0'].data())
            .toEqual(await expectedOutput.data());
      });

      it('works with 1x1x10x10 inputs', async () => {
        const outputs = runWithTfjsTensors(model, 'matmul_1x1x10x10', {
          'matmul_1x1x10x10_a:0': a.reshape([1, 1, 10, 10]),
          'matmul_1x1x10x10_b:0': b.reshape([1, 1, 10, 10]),
        });
        expect(await outputs['PartitionedCall_2:0'].data())
            .toEqual(await expectedOutput.data());
      });
    });
  });

  describe('getGlobalLiteRtPromise', () => {
    afterEach(async () => {
      await resetLiteRt();
    });

    it('returns undefined if LiteRt is not loaded', async () => {
      unloadLiteRt();
      expect(getGlobalLiteRtPromise()).toBeUndefined();
    });

    it('returns a promise that resolves to the loaded LiteRt instance',
       async () => {
         unloadLiteRt();
         const liteRtPromise = loadLiteRt('/wasm/');
         await expectAsync(liteRtPromise).toBeResolved();
         expect(getGlobalLiteRt()).toEqual(await liteRtPromise);
       });
  });
});
