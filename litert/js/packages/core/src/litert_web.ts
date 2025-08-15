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
import {Accelerator} from './constants';
import {CpuSignatureRunner} from './cpu_signature_runner';
import {getGlobalLiteRt} from './global_litert';
import {ConverterFactory} from './gpu_conversion';
import {GpuSignatureRunner} from './gpu_signature_runner';
import {Box, GpuErrorReporter, isWebGPUSupported, popErrorScopes, pushErrorScopes} from './gpu_utils';
import type {GPUDeviceWithAdapterInfo} from './gpu_utils';
import {ErrorReporter, LiteRtWasm} from './wasm_binding_types';

// These functions wrap the global LiteRT instance's methods and make it easier
// to call them.

/**
 * WebGPU features that LiteRT.js can use for improved performance.
 */
const DESIRED_WEBGPU_FEATURES = [
  'shader-f16',
  'subgroups' as GPUFeatureName,      // In origin trial
  'subgroups-f16' as GPUFeatureName,  // In origin trial
] satisfies GPUFeatureName[];

/**
 * Set the error reporter for the global LiteRT instance.
 */
export function setErrorReporter(errorReporter: ErrorReporter) {
  getGlobalLiteRt().setErrorReporter(errorReporter);
}

/**
 * Set the WebGPU error reporter for the global LiteRT instance.
 */
export function setGpuErrorReporter(errorReporter: GpuErrorReporter) {
  getGlobalLiteRt().setGpuErrorReporter(errorReporter);
}

/**
 * Set the WebGPU device for the global LiteRT instance.
 */
export function setWebGpuDevice(
    device: GPUDevice, adapterInfo?: GPUAdapterInfo) {
  getGlobalLiteRt().setWebGpuDevice(device, adapterInfo);
}

/**
 * Get the WebGPU device for the global LiteRT instance.
 */
export function getWebGpuDevice(): Promise<GPUDevice> {
  return getGlobalLiteRt().getWebGpuDevice();
}

/**
 * Get the WebGPU adapter info for the global LiteRT instance.
 */
export function getAdapterInfo(): Promise<GPUAdapterInfo> {
  return getGlobalLiteRt().getAdapterInfo();
}

/**
 * Loads a LiteRt model.
 *
 * @param model The model data. This can be a string (the model url), a URL
 *     object, a Uint8Array (the model bytes), or a
 *     ReadableStreamDefaultReader (for streaming model loading).
 * @param compileOptions The options for compiling the model. This includes
 *     the accelerator to use ('webgpu' or 'wasm') and the WebGPU device
 *     (for direct GPU model inputs / outputs).
 * @returns A promise that resolves to the CompiledModel.
 */
export function loadAndCompile(
    model: string|URL|Uint8Array|ReadableStreamDefaultReader,
    compileOptions: CompileOptions,
    ): Promise<CompiledModel> {
  return getGlobalLiteRt().loadAndCompile(model, compileOptions);
}

/**
 * The options for compiling a LiteRt model.
 */
export interface CompileOptions {
  accelerator: Accelerator;
}

/**
 * Run LiteRt models in the browser.
 */
export class LiteRt {
  readonly liteRtWasm: LiteRtWasm;
  private device?: GPUDeviceWithAdapterInfo;
  // Boxed so it can be passed as a reference to the signatures and updated
  // later.
  private readonly gpuErrorReporter: Box<GpuErrorReporter> = {
    val: (error, callsite) => {
      console.error('GPU error:', error, 'at:', callsite);
    },
  };
  private loadAndCompileWebGpuWasCalled = false;
  private loadedModels = new Set<CompiledModel>();
  private converterFactory?: ConverterFactory;

  constructor(wasmModule: WasmModule) {
    this.liteRtWasm = wasmModule as LiteRtWasm;
    if (!this.liteRtWasm.loadAndCompileWebGpu) {
      throw new Error('loadAndCompileWebGpu is not defined.');
    }
    this.liteRtWasm.setupLogging();
  }

  private pushErrorScopes() {
    if (!this.device) {
      throw new Error('No GPU device provided.');
    }
    pushErrorScopes(this.device);
  }

  private popErrorScopes(callsite: string) {
    if (!this.device) {
      throw new Error('No GPU device provided.');
    }
    popErrorScopes(this.device, callsite, this.gpuErrorReporter.val);
  }

  private static async urlToUint8Array(url: string|URL): Promise<Uint8Array> {
    // TODO(msoulanille): Streaming support for model loading once C++ supports
    // it.
    const response = await fetch(url);
    return new Uint8Array(await response.arrayBuffer());
  }

  private static async readableStreamToUint8Array(
      reader: ReadableStreamDefaultReader<Uint8Array>): Promise<Uint8Array> {
    // TODO(msoulanille): Rewrite this when we support streaming directly to
    // WebGPU memory.
    let byteOffset = 0;
    let array = new Uint8Array(1024 /* arbitrary starting size */);
    const MAX_ARRAY_SIZE = 2e9;  // Chrome gets flaky with sizes > 2GB.

    // Collecting all the chunks and then copying them would be easier, but this
    // is more memory efficient.
    while (true) {
      const {done, value} = await reader.read();
      if (value) {
        if (array.byteLength < byteOffset + value.byteLength) {
          if (byteOffset + value.byteLength > MAX_ARRAY_SIZE) {
            throw new Error(`Model is too large (> ${MAX_ARRAY_SIZE} bytes`);
          }

          // Allocate more space, but double the size to avoid reallocating too
          // often.
          // Note: This will not work with huge models since we store everything
          // in one ArrayBuffer, but more things will need to be refactored for
          // those anyway.
          const newArray = new Uint8Array(Math.min(
              MAX_ARRAY_SIZE,
              Math.max(array.byteLength, value.byteLength) * 2));
          newArray.set(array);
          array = newArray;
        }
        array.set(value, byteOffset);
        byteOffset += value.byteLength;
      }
      if (done) {
        break;
      }
    }

    // Resize to the exact byte length. Could use `.subarray`, but we'd like to
    // avoid keeping the extra bytes allocated.
    return array.slice(0, byteOffset);
  }

  /**
   * Initialize the WebGPU device for LiteRT.
   */
  private async initializeDefaultWebGpuDevice() {
    if (this.device) {
      console.warn('WebGPU device is already initialized.');
      return;
    }
    if (!isWebGPUSupported()) {
      throw new Error('This browser does not support WebGPU.');
    }
    const adapterDescriptor: GPURequestAdapterOptions = {
      powerPreference: 'high-performance',
    };
    const adapter = await navigator.gpu.requestAdapter(adapterDescriptor);
    if (!adapter) {
      throw new Error('No GPU adapter found.');
    }
    const adapterInfo = adapter.info;
    const requiredLimits = {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage:
          adapter.limits.maxStorageBuffersPerShaderStage,
      maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
    };

    const requiredFeatures: GPUFeatureName[] = [];
    for (const feature of DESIRED_WEBGPU_FEATURES) {
      if (adapter.features.has(feature)) {
        requiredFeatures.push(feature);
      }
    }
    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits,
    });

    // TODO: Remove adapterInfo from the api, as the latest GPUDevice type
    // should have adapterInfo.
    this.setWebGpuDevice(device, adapterInfo);
  }
  /**
   * Set the error reporter for LiteRt.
   */
  setErrorReporter(errorReporter: ErrorReporter) {
    this.liteRtWasm.setErrorReporter(errorReporter);
  }

  /**
   * Set the WebGPU error reporter for LiteRt.
   */
  setGpuErrorReporter(errorReporter: GpuErrorReporter) {
    this.gpuErrorReporter.val = errorReporter;
  }

  /**
   * Set the WebGPU device and adapter info for LiteRT.
   */
  // TODO: Remove adapterInfo from the api, as the latest GPUDevice type should
  // have adapterInfo.
  setWebGpuDevice(device: GPUDevice, adapterInfo?: GPUAdapterInfo) {
    if (this.loadAndCompileWebGpuWasCalled) {
      throw new Error(
          'The WebGPU device cannot be set after loading a WebGPU model.');
    }

    // TODO: P0, for EAP, Handle GPUDevice error events.
    this.device = device as GPUDeviceWithAdapterInfo;
    // Depending on version of WebGPU/browser, some devices will have
    // readonly adapterInfo, and some will not have this property at all.
    if (!this.device.adapterInfo) {
      if (!adapterInfo) {
        throw new Error(
            'The device does not have adapter info, so adapterInfo must be provided.');
      }
      this.device.adapterInfo = adapterInfo;
    }
    this.liteRtWasm.preinitializedWebGPUDevice = this.device;
  }

  /**
   * Get the WebGPU device that LiteRT is using. If the device is not set,
   * initialize it.
   */
  async getWebGpuDevice(): Promise<GPUDevice> {
    if (!this.device) {
      await this.initializeDefaultWebGpuDevice();
    }
    return this.device!;
  }

  /**
   * Get the WebGPU adapter info that LiteRT is using. If the WebGPU device is
   * not set, initialize it.
   */
  async getAdapterInfo(): Promise<GPUAdapterInfo> {
    if (!this.device) {
      await this.initializeDefaultWebGpuDevice();
    }
    return this.device!.adapterInfo!;
  }

  /**
   * Loads a LiteRt model.
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
      compileOptions: CompileOptions): Promise<CompiledModel> {
    // TODO: make `compileOptions` parameter optional.
    // Note: This will definitely be async in the future and will likely change.
    let modelData: Uint8Array;
    if (typeof model === 'string' || model instanceof URL) {
      modelData = await LiteRt.urlToUint8Array(model);
    } else if (model instanceof Uint8Array) {
      modelData = model;
    } else if (model instanceof ReadableStreamDefaultReader) {
      modelData = await LiteRt.readableStreamToUint8Array(model);
    } else {
      throw new Error('Unsupported model type.');
    }

    const ptr = this.liteRtWasm._malloc(modelData.byteLength);
    this.liteRtWasm.HEAPU8.set(modelData, ptr);
    let compiledModel: CompiledModel;
    const onDelete = () => {
      this.liteRtWasm._free(ptr);
      this.loadedModels.delete(compiledModel);
    };
    // Pipe WebGPU device into the Wasm module.
    if (compileOptions.accelerator === 'webgpu') {
      if (!this.liteRtWasm.preinitializedWebGPUDevice) {
        await this.initializeDefaultWebGpuDevice();
      }

      this.pushErrorScopes();
      this.loadAndCompileWebGpuWasCalled = true;
      const liteRtInterpreter =
          this.liteRtWasm.loadAndCompileWebGpu(ptr, modelData.byteLength);
      this.popErrorScopes('loadAndCompile');
      compiledModel =
          new CompiledModel(liteRtInterpreter, (signatureRunnerWrapper) => {
            if (!this.device) {
              throw new Error('No GPU device provided.');
            }
            return new GpuSignatureRunner(
                signatureRunnerWrapper, this.device, this.gpuErrorReporter);
          }, onDelete);
    } else {
      const liteRtInterpreter =
          this.liteRtWasm.loadAndCompileCpu(ptr, modelData.byteLength);
      compiledModel =
          new CompiledModel(liteRtInterpreter, (signatureRunnerWrapper) => {
            return new CpuSignatureRunner(signatureRunnerWrapper);
          }, onDelete);
    }

    this.loadedModels.add(compiledModel);
    return compiledModel;
  }

  /**
   * Gets or creates a ConverterFactory for our tensor converters.
   */
  getConverterFactory(): ConverterFactory {
    if (!this.converterFactory) {
      this.converterFactory =
          new ConverterFactory(this.liteRtWasm, this.gpuErrorReporter);
    }
    return this.converterFactory;
  }

  /**
   * Delete the LiteRt wasm module and all loaded models.
   */
  delete() {
    for (const model of this.loadedModels) {
      model.delete();
    }  // models automatically remove themselves from the loadedModels set.
  }
}
