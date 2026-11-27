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
 */

import {createDefaultWebGpuDevice} from './webgpu_device';

const ANGLE_M5_MAX_RENDERER =
    'ANGLE (Apple, ANGLE Metal Renderer: Apple M5 Max, Unspecified Version)';

function createFakeDevice(
    vendor = 'apple', architecture = 'metal-3', device = '',
    description = ''): GPUDevice {
  return {
    adapterInfo: {vendor, architecture, device, description},
  } as unknown as GPUDevice;
}

function createFakeGpu(
    device: GPUDevice, features: readonly string[] = ['shader-f16']) {
  const requestDevice =
      jasmine.createSpy('requestDevice').and.resolveTo(device);
  const fakeAdapter = {
    limits: {
      maxBufferSize: 1024,
      maxStorageBufferBindingSize: 512,
      maxStorageBuffersPerShaderStage: 8,
      maxTextureDimension2D: 8192,
    },
    features: new Set<string>(features),
    requestDevice,
  };
  const requestAdapter =
      jasmine.createSpy('requestAdapter').and.resolveTo(fakeAdapter);
  return {gpu: {requestAdapter}, requestAdapter, requestDevice};
}

describe('createDefaultWebGpuDevice', () => {
  let originalGpuDescriptor: PropertyDescriptor|undefined;
  let originalOffscreenCanvasDescriptor: PropertyDescriptor|undefined;
  let getContextSpy: jasmine.Spy;

  beforeEach(() => {
    originalGpuDescriptor = Object.getOwnPropertyDescriptor(navigator, 'gpu');
    originalOffscreenCanvasDescriptor =
        Object.getOwnPropertyDescriptor(globalThis, 'OffscreenCanvas');

    const fakeGl = {
      RENDERER: 0x1F01,
      getParameter: (pname: number) =>
          pname === 0x1F01 ? ANGLE_M5_MAX_RENDERER : null,
      getExtension: () => null,
    };
    getContextSpy = jasmine.createSpy('getContext').and.returnValue(fakeGl);
    class FakeOffscreenCanvas {
      getContext = getContextSpy;
    }
    Object.defineProperty(globalThis, 'OffscreenCanvas', {
      value: FakeOffscreenCanvas,
      configurable: true,
    });
  });

  afterEach(() => {
    if (originalGpuDescriptor) {
      Object.defineProperty(navigator, 'gpu', originalGpuDescriptor);
    } else {
      delete (navigator as unknown as {gpu?: unknown}).gpu;
    }
    if (originalOffscreenCanvasDescriptor) {
      Object.defineProperty(
          globalThis, 'OffscreenCanvas', originalOffscreenCanvasDescriptor);
    } else {
      delete (globalThis as unknown as {OffscreenCanvas?: unknown})
          .OffscreenCanvas;
    }
  });

  it('requests limits, supported features, and identifies Apple GPUs by default',
     async () => {
       const fakeDevice = createFakeDevice();
       const {gpu, requestAdapter, requestDevice} = createFakeGpu(
           fakeDevice, ['shader-f16', 'subgroups', 'unused-feature']);
       Object.defineProperty(
           navigator, 'gpu', {value: gpu, configurable: true});

       const device = await createDefaultWebGpuDevice();

       expect(device).toBe(fakeDevice);
       expect(requestAdapter)
           .toHaveBeenCalledWith({powerPreference: 'high-performance'});
       expect(requestDevice).toHaveBeenCalledWith({
         requiredFeatures: ['shader-f16', 'subgroups'],
         requiredLimits: {
           maxBufferSize: 1024,
           maxStorageBufferBindingSize: 512,
           maxStorageBuffersPerShaderStage: 8,
           maxTextureDimension2D: 8192,
         },
       });
       expect(device.adapterInfo.device).toBe('Apple M5 Max');
       expect(device.adapterInfo.description).toBe('Apple M5 Max');
     });

  it('skips WebGL probing when identifyGpuWithWebGl is false', async () => {
    const fakeDevice = createFakeDevice();
    const {gpu, requestDevice} = createFakeGpu(fakeDevice, []);
    Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});

    const device =
        await createDefaultWebGpuDevice({identifyGpuWithWebGl: false});

    expect(device).toBe(fakeDevice);
    expect(requestDevice).toHaveBeenCalledWith({
      requiredFeatures: [],
      requiredLimits: {
        maxBufferSize: 1024,
        maxStorageBufferBindingSize: 512,
        maxStorageBuffersPerShaderStage: 8,
        maxTextureDimension2D: 8192,
      },
    });
    expect(device.adapterInfo.device).toBe('');
    expect(device.adapterInfo.description).toBe('');
    expect(getContextSpy).not.toHaveBeenCalled();
  });

  it('throws when no GPU adapter is found', async () => {
    const gpu = {
      requestAdapter: jasmine.createSpy('requestAdapter').and.resolveTo(null),
    };
    Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});

    await expectAsync(createDefaultWebGpuDevice())
        .toBeRejectedWithError('No GPU adapter found.');
  });
});
