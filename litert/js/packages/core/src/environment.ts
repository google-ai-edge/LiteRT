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

import {getGlobalLiteRt} from './global_litert';
import {Deletable, LiteRtEnvironment} from './wasm_binding_types';

/**
 * Options for creating a LiteRT.js Environment.
 *
 * If a property is omitted, it may be set to a default value. If a property is
 * explicitly set to `null`, no default value will be used.
 */
export interface EnvironmentOptions {
  webGpuDevice?: GPUDevice|null;
}

/**
 * WebGPU features that LiteRT.js can use for improved performance.
 */
const DESIRED_WEBGPU_FEATURES = [
  'shader-f16',
  'subgroups' as GPUFeatureName,      // In origin trial
  'subgroups-f16' as GPUFeatureName,  // In origin trial
] satisfies GPUFeatureName[];

/**
 * A type that has an environment property.
 */
export interface WithEnvironment {
  readonly environment: Environment;
}

/**
 * An environment for LiteRT.js.
 */
export class Environment implements Deletable, EnvironmentOptions {
  readonly liteRtEnvironment: LiteRtEnvironment;

  constructor(readonly options: Required<EnvironmentOptions>) {
    this.liteRtEnvironment =
        getGlobalLiteRt().liteRtWasm.LiteRtEnvironment.create(
            options.webGpuDevice);
  }

  static async create(options: EnvironmentOptions = {}): Promise<Environment> {
    let webGpuDevice: GPUDevice|null = null;
    if ('webGpuDevice' in options) {
      // Create a default webgpu device only if `webGpuDevice` is not a key in
      // the options. This allows the caller to pass in a custom webgpu device
      // and allows them to disable WebGPU entirely if they wish (e.g. not
      // supported on their platform).
      if (options.webGpuDevice) {
        webGpuDevice = options.webGpuDevice;
      }
    } else {
      // User did not specify a webgpu device or disable WebGPU, so create a
      // default one.
      try {
        webGpuDevice = await createDefaultWebGpuDevice();
      } catch (e) {
        console.warn('Failed to create default WebGPU device:', e);
      }
    }

    return new Environment({
      ...options,
      webGpuDevice,
    });
  }

  get webGpuDevice(): GPUDevice|null {
    return this.options.webGpuDevice;
  }

  delete() {
    this.liteRtEnvironment.delete();
  }
}

async function createDefaultWebGpuDevice(): Promise<GPUDevice> {
  const adapterDescriptor: GPURequestAdapterOptions = {
    powerPreference: 'high-performance',
  };
  const adapter = await navigator.gpu.requestAdapter(adapterDescriptor);
  if (!adapter) {
    throw new Error('No GPU adapter found.');
  }

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

  return await adapter.requestDevice({
    requiredFeatures,
    requiredLimits,
  });
}
