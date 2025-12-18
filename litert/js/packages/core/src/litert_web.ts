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

import {WasmModule} from '@litertjs/wasm-utils';

import {CompiledModel} from './compiled_model';
import {Environment} from './environment';
import {getGlobalLiteRt} from './global_litert';
import {readableStreamDefaultReaderToUint8Array, urlToUint8Array} from './load_utils';
import {Model} from './model';
import {CompileOptions} from './model_types';
import {Deletable, LiteRtWasm} from './wasm_binding_types';

/**
 * Check if the browser supports WebGPU.
 */
export function isWebGPUSupported(): boolean {
  return !!(
      typeof globalThis !== 'undefined' && (globalThis.navigator) &&
      (globalThis.navigator.gpu));
}

// These functions wrap the global LiteRT instance's methods and make it easier
// to call them.

/**
 * Get or create the default environment.
 */
export function getDefaultEnvironment() {
  return getGlobalLiteRt().getDefaultEnvironment();
}

/**
 * Loads and compiles a LiteRt model.
 *
 * @param model The model data. This can be a string (the model url), a URL
 *     object, a Uint8Array (the model bytes), or a
 *     ReadableStreamDefaultReader (for streaming model loading).
 * @param compileOptions The options for compiling the model. This includes
 *     the accelerator to use ('webgpu' or 'wasm') and the WebGPU device
 *     (for direct GPU model inputs / outputs).
 * @return A promise that resolves to the CompiledModel.
 */
export function loadAndCompile(
    model: string|URL|Uint8Array|ReadableStreamDefaultReader,
    compileOptions?: CompileOptions,
    ): Promise<CompiledModel> {
  return getGlobalLiteRt().loadAndCompile(model, compileOptions);
}

/**
 * Get the WebGPU device used by the default environment.
 */
export function getWebGpuDevice(): GPUDevice|null {
  return getGlobalLiteRt().getWebGpuDevice();
}

/**
 * Create a new default environment with the given WebGPU device.
 */
export function setWebGpuDevice(device: GPUDevice) {
  getGlobalLiteRt().setWebGpuDevice(device);
}

/**
 * Run LiteRt models in the browser.
 */
export class LiteRt {
  readonly liteRtWasm: LiteRtWasm;
  private defaultEnvironment?: Environment;
  private readonly objectsToDelete = new Set<Deletable>();

  constructor(wasmModule: WasmModule) {
    this.liteRtWasm = wasmModule as LiteRtWasm;
    this.liteRtWasm.setupLogging();
  }

  setDefaultEnvironment(environment: Environment) {
    this.defaultEnvironment = environment;
  }

  getDefaultEnvironment(): Environment {
    if (!this.defaultEnvironment) {
      throw new Error('Default environment is not set.');
    }
    return this.defaultEnvironment;
  }

  setWebGpuDevice(device: GPUDevice) {
    // Environment is immutable, so we need to create a new one.
    const oldEnvironment = this.getDefaultEnvironment();
    this.setDefaultEnvironment(new Environment({
      ...oldEnvironment.options,
      webGpuDevice: device,
    }));
  }

  getWebGpuDevice(): GPUDevice|null {
    return this.getDefaultEnvironment().webGpuDevice;
  }

  /**
   * Loads and compiles a LiteRt model.
   *
   * @param model The model data. This can be a string (the model url), a URL
   *     object, a Uint8Array (the model bytes), or a
   *     ReadableStreamDefaultReader (for streaming model loading).
   * @param compileOptions The options for compiling the model. This includes
   *     the accelerator to use ('webgpu' or 'wasm') and the WebGPU device
   *     (for direct GPU model inputs / outputs).
   * @returns A promise that resolves to the CompiledModel.
   */
  async loadAndCompile(
      model: string|URL|Uint8Array|ReadableStreamDefaultReader<Uint8Array>,
      compileOptions: CompileOptions = {}): Promise<CompiledModel> {
    let modelData: Uint8Array;
    if (typeof model === 'string' || model instanceof URL) {
      modelData = await urlToUint8Array(model);
    } else if (model instanceof Uint8Array) {
      modelData = model;
    } else if (model instanceof ReadableStreamDefaultReader) {
      modelData = await readableStreamDefaultReaderToUint8Array(model);
    } else {
      throw new Error('Unsupported model type.');
    }

    // Assign defaults
    const environment =
        compileOptions.environment ?? this.getDefaultEnvironment();
    const accelerator = compileOptions.accelerator ??
        (environment.webGpuDevice ? 'webgpu' : 'wasm');

    if (accelerator === 'webgpu' && !environment.webGpuDevice) {
      throw new Error(
          'WebGPU was requested but no WebGPU device is set in the ' +
          'environment.');
    }

    const cpuOptions = compileOptions.cpuOptions ??
        {numThreads: this.liteRtWasm.getThreadCount()};

    const filledCompileOptions: Required<CompileOptions> = {
      environment,
      accelerator,
      cpuOptions,
    };

    const ptr = this.liteRtWasm._malloc(modelData.byteLength);
    this.liteRtWasm.HEAPU8.set(modelData, ptr);
    const wasmModel = this.liteRtWasm.loadModel(
        filledCompileOptions.environment.liteRtEnvironment, ptr,
        modelData.byteLength);
    const wasmCompiledModel = this.liteRtWasm.compileModel(
        filledCompileOptions.environment.liteRtEnvironment, wasmModel,
        filledCompileOptions);
    const loadedModel = new Model(wasmModel, () => {
      this.liteRtWasm._free(ptr);
    });
    const compiledModel = new CompiledModel(
        loadedModel, wasmCompiledModel, filledCompileOptions, () => {
          // When deleted, remove the compiled model from the set of objects to
          // delete.
          this.objectsToDelete.delete(compiledModel);
        });

    // Currently, compiledModel will delete the loadedModel when it is deleted.
    this.objectsToDelete.add(compiledModel);
    return compiledModel;
  }

  delete() {
    for (const object of this.objectsToDelete) {
      object.delete();
    }
  }
}
