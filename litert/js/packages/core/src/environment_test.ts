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

import {WasmModule} from '@litertjs/wasm-utils';

import {Environment} from './environment';
import {getGlobalLiteRt, getGlobalLiteRtPromise, hasGlobalLiteRt, setGlobalLiteRt, setGlobalLiteRtPromise} from './global_litert';
import type {LiteRt} from './litert_web';
import {initFromWasmModule, unloadLiteRt} from './load_litert';

const ANGLE_M5_MAX_RENDERER =
    'ANGLE (Apple, ANGLE Metal Renderer: Apple M5 Max, Unspecified Version)';

function createFakeDevice(
    vendor = 'apple', architecture = 'metal-3', device = '',
    description = ''): GPUDevice {
  return {
    adapterInfo: {vendor, architecture, device, description},
  } as unknown as GPUDevice;
}

function createFakeGpu(device: GPUDevice) {
  const requestDevice =
      jasmine.createSpy('requestDevice').and.resolveTo(device);
  const fakeAdapter = {
    limits: {
      maxBufferSize: 1024,
      maxStorageBufferBindingSize: 1024,
      maxStorageBuffersPerShaderStage: 8,
      maxTextureDimension2D: 8192,
    },
    features: new Set<string>(['shader-f16']),
    requestDevice,
  };
  const requestAdapter =
      jasmine.createSpy('requestAdapter').and.resolveTo(fakeAdapter);
  return {gpu: {requestAdapter}, requestAdapter, requestDevice};
}

function createFakeWasmModule(): WasmModule {
  return {
    setupLogging: jasmine.createSpy('setupLogging'),
    hasWebGpuSupport: () => true,
    LiteRtEnvironment: {
      create: jasmine.createSpy('create').and.returnValue({delete: () => {}}),
    },
  } as unknown as WasmModule;
}

function createFakeLiteRt(): LiteRt {
  return {liteRtWasm: createFakeWasmModule()} as unknown as LiteRt;
}

describe('Environment.create', () => {
  let originalGpuDescriptor: PropertyDescriptor|undefined;
  let getContextSpy: jasmine.Spy;

  beforeEach(() => {
    originalGpuDescriptor = Object.getOwnPropertyDescriptor(navigator, 'gpu');

    const fakeGl = {
      RENDERER: 0x1F01,
      getParameter: (pname: number) =>
          pname === 0x1F01 ? ANGLE_M5_MAX_RENDERER : null,
      getExtension: () => null,
    };
    getContextSpy =
        spyOn(OffscreenCanvas.prototype, 'getContext')
            .and.returnValue(fakeGl as unknown as WebGL2RenderingContext);
  });

  afterEach(() => {
    if (originalGpuDescriptor) {
      Object.defineProperty(navigator, 'gpu', originalGpuDescriptor);
    } else {
      delete (navigator as unknown as {gpu?: unknown}).gpu;
    }
  });

  it('identifies a bucketed Apple GPU with WebGL by default', async () => {
    const device = createFakeDevice();
    const {gpu} = createFakeGpu(device);
    Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});

    const env = await Environment.create({}, createFakeLiteRt());

    expect(env.webGpuDevice).toBe(device);
    expect(env.webGpuDevice!.adapterInfo.device).toBe('Apple M5 Max');
    expect(env.webGpuDevice!.adapterInfo.description).toBe('Apple M5 Max');
    expect(env.webGpuDevice!.adapterInfo.vendor).toBe('apple');
    expect(env.webGpuDevice!.adapterInfo.architecture).toBe('metal-3');
  });

  it('skips WebGL probing when identifyGpuWithWebGl is false', async () => {
    const device = createFakeDevice();
    const {gpu} = createFakeGpu(device);
    Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});

    const env = await Environment.create(
        {identifyGpuWithWebGl: false}, createFakeLiteRt());

    expect(env.webGpuDevice).toBe(device);
    expect(env.webGpuDevice!.adapterInfo.device).toBe('');
    expect(env.webGpuDevice!.adapterInfo.description).toBe('');
    expect(getContextSpy).not.toHaveBeenCalled();
  });

  it('leaves a caller-provided WebGPU device untouched', async () => {
    const customDevice = createFakeDevice();
    const {gpu, requestAdapter} = createFakeGpu(createFakeDevice());
    Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});

    const env = await Environment.create(
        {webGpuDevice: customDevice}, createFakeLiteRt());

    expect(env.webGpuDevice).toBe(customDevice);
    expect(customDevice.adapterInfo.device).toBe('');
    expect(requestAdapter).not.toHaveBeenCalled();
    expect(getContextSpy).not.toHaveBeenCalled();
  });
});

describe('initFromWasmModule', () => {
  let originalGpuDescriptor: PropertyDescriptor|undefined;
  let previousGlobal: LiteRt|undefined;
  let previousGlobalPromise: Promise<LiteRt>|undefined;
  let getContextSpy: jasmine.Spy;

  beforeEach(() => {
    originalGpuDescriptor = Object.getOwnPropertyDescriptor(navigator, 'gpu');
    previousGlobal = hasGlobalLiteRt() ? getGlobalLiteRt() : undefined;
    previousGlobalPromise = getGlobalLiteRtPromise();
    setGlobalLiteRt(undefined);
    setGlobalLiteRtPromise(undefined);

    const fakeGl = {
      RENDERER: 0x1F01,
      getParameter: (pname: number) =>
          pname === 0x1F01 ? ANGLE_M5_MAX_RENDERER : null,
      getExtension: () => null,
    };
    getContextSpy =
        spyOn(OffscreenCanvas.prototype, 'getContext')
            .and.returnValue(fakeGl as unknown as WebGL2RenderingContext);
  });

  afterEach(() => {
    unloadLiteRt();
    setGlobalLiteRt(previousGlobal);
    setGlobalLiteRtPromise(previousGlobalPromise);
    if (originalGpuDescriptor) {
      Object.defineProperty(navigator, 'gpu', originalGpuDescriptor);
    } else {
      delete (navigator as unknown as {gpu?: unknown}).gpu;
    }
  });

  it('identifies a bucketed Apple GPU with WebGL by default', async () => {
    const device = createFakeDevice();
    const {gpu} = createFakeGpu(device);
    Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});

    const liteRt = await initFromWasmModule(createFakeWasmModule());

    const webGpuDevice = liteRt.getWebGpuDevice()!;
    expect(webGpuDevice).toBe(device);
    expect(webGpuDevice.adapterInfo.device).toBe('Apple M5 Max');
    expect(webGpuDevice.adapterInfo.description).toBe('Apple M5 Max');
  });

  it('forwards identifyGpuWithWebGl: false to Environment.create', async () => {
    const device = createFakeDevice();
    const {gpu} = createFakeGpu(device);
    Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});

    const liteRt = await initFromWasmModule(
        createFakeWasmModule(), {identifyGpuWithWebGl: false});

    const webGpuDevice = liteRt.getWebGpuDevice()!;
    expect(webGpuDevice).toBe(device);
    expect(webGpuDevice.adapterInfo.device).toBe('');
    expect(webGpuDevice.adapterInfo.description).toBe('');
    expect(getContextSpy).not.toHaveBeenCalled();
  });
});
