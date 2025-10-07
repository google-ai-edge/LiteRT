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

import '@tensorflow/tfjs-backend-webgpu'; // Side-effect import for webgpu backend.
import '@tensorflow/tfjs-backend-cpu'; // CPU backend is needed if WebGPU is not available.

import {runWithTfjsTensors} from '@litertjs/tfjs-interop';
import {CompiledModel, loadAndCompile, loadLiteRt, SignatureRunner, getWebGpuDevice} from '@litertjs/core';
import {WebGPUBackend} from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';
// Placeholder for internal dependency on trusted resource url

import {ConsoleMessage} from './console_renderer';
import {BenchmarkSample, compareResults, just, LITERT_WASM_CPU, LITERT_WASM_GPU, Maybe, ModelResult, ModelRunner, RunResult, SerializableTensor, toMaybe} from './model_runner';

/**
 * A promise that resolves when LiteRt is loaded.
 */
export const liteRtPromise = (async () => {
  // Some older WebGPU implementations do not make adapterInfo available on
  // the device, so we need to construct a new TFJS WebGPU backend here in
  // order to have access to the adapterInfo.
  // TODO: b/434057579 - Remove this once all WebGPU implementations have
  //   an adapterInfo property on the device.

  // TODO: b/445746846 - Add an option for threads / no threads.
  try {
    await loadLiteRt('./wasm/', {threads: true});
    console.log('LiteRt loaded with threads');
  } catch (e) {
    await loadLiteRt('./wasm/');
    console.log('LiteRt loaded without threads');
  }
  try {
    const device = await getWebGpuDevice();
    const adapterInfo = (device as unknown as {adapterInfo: GPUAdapterInfo}).adapterInfo;
    tf.removeBackend('webgpu');
    tf.registerBackend('webgpu', () => new WebGPUBackend(device, adapterInfo));

    await tf.setBackend('webgpu');
  } catch (e) {
    console.warn('WebGPU failed to load. Will only run on CPU.');
    await tf.setBackend('cpu');
    console.error(e);
  }
})();

/**
 * A model runner that runs LiteRT models.
 */
export class LiteRtModelRunner implements ModelRunner {
  constructor(
      private gpuModel: Maybe<CompiledModel>,
      private cpuModel: Maybe<CompiledModel>,
      private readConsole?: () => ConsoleMessage[]) {}

  static async load(data: Uint8Array, readConsole?: () => ConsoleMessage[]):
      Promise<LiteRtModelRunner> {
    await liteRtPromise;
    const gpuModel =
        await toMaybe(() => loadAndCompile(data, {accelerator: 'webgpu'}));
    const cpuModel =
        await toMaybe(() => loadAndCompile(data, {accelerator: 'wasm'}));

    if (gpuModel.error && cpuModel.error) {
      console.error('Both GPU and CPU models failed to load');
    }

    return new LiteRtModelRunner(gpuModel, cpuModel, readConsole);
  }

  getSignatures() {
    const signatures: Array<{
      name: string;
      id?: string;
      signature?: CompiledModel | SignatureRunner;
    }> =
        [
          {
            name: 'Base Signature',
            id: undefined,
          },
        ];

    if (!this.gpuModel.value) {
      return signatures;
    }

    signatures[0].signature = this.gpuModel.value;

    for (const [id, signature] of Object.entries(
             this.gpuModel.value.signatures)) {
      signatures.push({
        name: id,
        id,
        signature,
      });
    }
    return signatures;
  }

  private async runModel(
      model: CompiledModel|SignatureRunner,
      fakeInputs: Record<string, tf.Tensor>,
      benchmarkRunCount: number): Promise<ModelResult> {
    const partialResult: Partial<ModelResult> = {};

    const results = runWithTfjsTensors(model, fakeInputs);

    partialResult.tensors = {record: {}};
    for (const [name, tensor] of Object.entries(results)) {
      partialResult.tensors.record[name] = await serializeTfjsTensor(tensor);
    }

    if (benchmarkRunCount === 0) {
      return partialResult as ModelResult;
    }

    const benchmarkSamples: BenchmarkSample[] = [];
    for (let i = 0; i < benchmarkRunCount; i++) {
      const start = performance.now();
      console.log(`Benchmarking run ${i + 1} of ${benchmarkRunCount}`);
      const results = runWithTfjsTensors(model, fakeInputs);
      for (const result of Object.values(results)) {
        await result.data();  // Ensure data is synced to CPU
        result.dispose();
      }
      const end = performance.now();
      benchmarkSamples.push({latency: end - start});
    }

    partialResult.benchmark = {
      samples: benchmarkSamples,
    };

    return partialResult as ModelResult;
  }

  async run(signatureName?: string, benchmarkRunCount = 0): Promise<RunResult> {
    const gpuSignature = getSignature(this.gpuModel, signatureName);
    const cpuSignature = getSignature(this.cpuModel, signatureName);

    let fakeInputs: Record<string, tf.Tensor>;
    try {
      if (!gpuSignature.value) {
        if (!cpuSignature.value) {
          console.error(gpuSignature.error);
          console.error(cpuSignature.error);
          console.error(`GPU and CPU failed to load`);
          return {
            results: {
              [LITERT_WASM_CPU]: cpuSignature,
              [LITERT_WASM_GPU]: gpuSignature,
            },
            consoleMessages: this.readConsole?.(),
          };
        }
        fakeInputs = makeFakeInputs(cpuSignature.value);
      } else {
        fakeInputs = makeFakeInputs(gpuSignature.value);
      }
    } catch (e) {
      console.error(e);
      const error = e instanceof Error ? (e.stack ?? e.message) : String(e);
      return {
        results: {
          [LITERT_WASM_CPU]: {error},
          [LITERT_WASM_GPU]: {error},
        },
        consoleMessages: this.readConsole?.(),
      };
    }

    let gpuResult: Maybe<ModelResult>;
    let cpuResult: Maybe<ModelResult>;

    if (gpuSignature.value) {
      gpuResult = await toMaybe(
          () => this.runModel(
              gpuSignature.value!, fakeInputs, benchmarkRunCount));
    } else {
      gpuResult = gpuSignature;  // Pass forward the error
    }

    if (cpuSignature.value) {
      // Ensure the fake inputs are synced to CPU before running the model.
      await Promise.all(Object.values(fakeInputs).map(tensor => tensor.data()));
      cpuResult = await toMaybe(
          () => this.runModel(
              cpuSignature.value!, fakeInputs, benchmarkRunCount));
    } else {
      cpuResult = cpuSignature;  // Pass forward the error
    }

    if (cpuResult.value && gpuResult.value) {
      gpuResult.value.meanSquaredError =
          compareResults(cpuResult.value, gpuResult.value);
    }

    return {
      results: {
        [LITERT_WASM_CPU]: cpuResult,
        [LITERT_WASM_GPU]: gpuResult,
      },
      consoleMessages: this.readConsole?.(),
    };
  }
}

interface TensorMeta {
  shape: number[]|Int32Array;
  dtype: tf.DataType;
}

/**
 * Make a fake input tensor.
 */
function makeFakeInput({shape, dtype}: TensorMeta): tf.Tensor {
  if (dtype === 'int32') {
    return tf.randomUniformInt([...shape], 0, 10);
  } else {
    return tf.randomUniform([...shape], 0, 1);
  }
}

/**
 * Make fake inputs for a model.
 */
function makeFakeInputs(model: CompiledModel|SignatureRunner) {
  return Object.fromEntries(model.getInputDetails().map(
      details => [details.name, makeFakeInput(details)]));
}

/**
 * Serializes a tf.Tensor to a SerializableTensor.
 */
async function serializeTfjsTensor(tensor: tf.Tensor):
    Promise<SerializableTensor> {
  return {
    data: await tensor.data(),
    dtype: tensor.dtype,
    shape: tensor.shape,
  };
}

function getSignature(model: Maybe<CompiledModel>, signatureName?: string):
    Maybe<CompiledModel|SignatureRunner> {
  if (!model.value) {
    return model;  // Pass forward the error
  }
  if (signatureName) {
    if (model.value.signatures[signatureName]) {
      return just(model.value.signatures[signatureName]);
    } else {
      return {error: `Signature ${signatureName} not found`};
    }
  } else {
    return just(model.value);
  }
}
