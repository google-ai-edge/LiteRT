/**
 * Copyright 2026 Google LLC
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
 *
 * @fileoverview Shared WebGPU device creation for LiteRT.js and LiteRT-LM.js.
 */

import {fixDeviceInfoWithWebGl, type FixDeviceInfoWithWebGlOptions} from './gpu_adapter_info';

/** Options for `createDefaultWebGpuDevice`. */
export type CreateDefaultWebGpuDeviceOptions = FixDeviceInfoWithWebGlOptions;

/** WebGPU features that LiteRT.js and LiteRT-LM.js use when available. */
const DESIRED_WEBGPU_FEATURES: readonly GPUFeatureName[] = [
  'shader-f16',
  'subgroups' as GPUFeatureName,
];

/**
 * Requests a high-performance WebGPU adapter and creates a `GPUDevice` with
 * the adapter's maximum supported buffer/texture limits and any supported
 * desired features (`shader-f16`, `subgroups`), then calls
 * `fixDeviceInfoWithWebGl` so Wasm runtimes see a specific Apple GPU name.
 */
export async function createDefaultWebGpuDevice(
    options: CreateDefaultWebGpuDeviceOptions = {}): Promise<GPUDevice> {
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

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits,
  });
  // Browsers report bucketed GPU names; report the specific GPU (if found) in
  // `device.adapterInfo`, which the Wasm module reads to tune its kernels.
  fixDeviceInfoWithWebGl(device, options);
  return device;
}
