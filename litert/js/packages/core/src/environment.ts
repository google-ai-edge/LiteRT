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

import {createDefaultWebGpuDevice} from '@litertjs/webgpu-utils';

import {getGlobalLiteRt} from './global_litert';
import type {LiteRt} from './litert_web';
import {Deletable, LiteRtEnvironment, unwrapWasmResult} from './wasm_binding_types';

/**
 * Options for creating a LiteRT.js Environment.
 *
 * If a property is omitted, it may be set to a default value. If a property is
 * explicitly set to `null`, no default value will be used.
 *
 * @property webGpuDevice A custom WebGPU device to use, or null to disable
 *     WebGPU.
 * @property cpuOnly If true, disables WebGPU and initializes a CPU-only
 *     environment without querying WebGPU adapters.
 * @property identifyGpuWithWebGl Whether to briefly create a WebGL context to
 *     identify the GPU when the default WebGPU device's adapter info is too
 *     generic to pick the right kernels (for example on Apple Silicon Macs in
 *     Chrome). Defaults to true. Only applies to the default device; the probe
 *     runs at most once per page, but takes one of the page's limited WebGL
 *     context slots while it runs.
 */
export interface EnvironmentOptions {
  webGpuDevice?: GPUDevice|null;
  cpuOnly?: boolean;
  identifyGpuWithWebGl?: boolean;
}

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
  readonly options: EnvironmentOptions;

  constructor(
      options: EnvironmentOptions = {},
      liteRt?: LiteRt) {
    const wasm = (liteRt ?? getGlobalLiteRt()).liteRtWasm;
    const hasWebGpuSupport = wasm.hasWebGpuSupport?.() ?? true;
    const webGpuDevice =
        hasWebGpuSupport ? (options.webGpuDevice ?? null) : null;
    this.options = {
      ...options,
      webGpuDevice,
    };
    this.liteRtEnvironment =
        unwrapWasmResult(wasm.LiteRtEnvironment.create(webGpuDevice));
  }

  static async create(
      options: EnvironmentOptions = {},
      liteRt?: LiteRt): Promise<Environment> {
    const rt = liteRt ?? getGlobalLiteRt();
    const hasWebGpuSupport = rt.liteRtWasm.hasWebGpuSupport?.() ?? true;

    let webGpuDevice: GPUDevice|null = null;
    if (options.cpuOnly || options.webGpuDevice === null || !hasWebGpuSupport) {
      webGpuDevice = null;
    } else if (options.webGpuDevice !== undefined) {
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
        webGpuDevice = await createDefaultWebGpuDevice(
            {identifyGpuWithWebGl: options.identifyGpuWithWebGl});
      } catch (e) {
        console.warn('Failed to create default WebGPU device:', e);
      }
    }

    return new Environment({
      ...options,
      webGpuDevice,
      cpuOnly: webGpuDevice === null,
    }, rt);
  }

  /**
   * Helper to create a CPU-only environment without attempting to query or
   * initialize WebGPU.
   */
  static createCpuOnly(liteRt?: LiteRt): Promise<Environment> {
    return Environment.create({cpuOnly: true, webGpuDevice: null}, liteRt);
  }

  get webGpuDevice(): GPUDevice|null {
    return this.options.webGpuDevice ?? null;
  }

  get cpuOnly(): boolean {
    return this.webGpuDevice === null;
  }

  delete() {
    this.liteRtEnvironment.delete();
  }
}
