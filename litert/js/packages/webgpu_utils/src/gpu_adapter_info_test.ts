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

import {classifyFormFactor, fixDeviceInfoWithWebGl, GpuPlatform, HANDHELD_APPLE_GPU_HINT, overrideAdapterInfoName, parseAppleGpuName, readWebGlRenderer, WebGlRendererSource} from './gpu_adapter_info';

const MAC_USER_AGENT =
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
    '(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36';
const IPHONE_USER_AGENT =
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) ' +
    'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 ' +
    'Safari/604.1';
const WINDOWS_USER_AGENT =
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
    '(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36';

function angleMetalRenderer(gpu: string): string {
  return `ANGLE (Apple, ANGLE Metal Renderer: ${gpu}, Unspecified Version)`;
}

declare interface AdapterInfoFields {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

const adapterInfoFields = new WeakMap<object, AdapterInfoFields>();

/**
 * A stand-in for the browser's GPUAdapterInfo. Like the real one, its getters
 * throw a TypeError when called with any `this` but the object itself.
 */
class FakeAdapterInfo {
  constructor(
      vendor: string, architecture: string, device = '', description = '') {
    adapterInfoFields.set(this, {vendor, architecture, device, description});
  }

  private fields(): AdapterInfoFields {
    const fields = adapterInfoFields.get(this);
    if (!fields) throw new TypeError('Illegal invocation');
    return fields;
  }

  get vendor() {
    return this.fields().vendor;
  }
  get architecture() {
    return this.fields().architecture;
  }
  get device() {
    return this.fields().device;
  }
  get description() {
    return this.fields().description;
  }
  get subgroupMinSize() {
    return this.fields() && 32;
  }
  get subgroupMaxSize() {
    return this.fields() && 64;
  }
  get isFallbackAdapter() {
    return !this.fields();
  }
  describe() {
    return `${this.vendor} ${this.architecture}`;
  }
}

function fakeDevice(info: FakeAdapterInfo|undefined): GPUDevice {
  return {adapterInfo: info} as unknown as GPUDevice;
}

function fakePlatform(
    renderer: string, userAgent = MAC_USER_AGENT, maxTouchPoints = 0) {
  return {
    probeWebGlRenderer: jasmine.createSpy('probeWebGlRenderer')
                            .and.returnValue(renderer),
    userAgent,
    maxTouchPoints,
  } satisfies GpuPlatform;
}

function fakeWebGl(parameters: Record<number, unknown>, hasDebugInfo = true):
    WebGlRendererSource&{getExtension: jasmine.Spy} {
  return {
    RENDERER: 1,
    getParameter: (pname: number) => parameters[pname],
    getExtension: jasmine.createSpy('getExtension')
                      .and.callFake(
                          (name: string) =>
                              name === 'WEBGL_debug_renderer_info' &&
                                  hasDebugInfo ?
                                  {UNMASKED_RENDERER_WEBGL: 2} :
                                  null),
  };
}

describe('parseAppleGpuName', () => {
  it('extracts the GPU from ANGLE Metal renderer strings', () => {
    for (const gpu
             of ['Apple M1', 'Apple M1 Pro', 'Apple M1 Max', 'Apple M1 Ultra',
                 'Apple M2', 'Apple M2 Ultra', 'Apple M3 Pro', 'Apple M3 Max',
                 'Apple M4', 'Apple M4 Max', 'Apple M5', 'Apple M5 Pro',
                 'Apple M5 Max', 'Apple M5 Ultra']) {
      expect(parseAppleGpuName(angleMetalRenderer(gpu))).toBe(gpu);
    }
  });

  it('extracts the GPU from other renderer strings', () => {
    expect(parseAppleGpuName('ANGLE (Apple, Apple M1 Pro, OpenGL 4.1)'))
        .toBe('Apple M1 Pro');
    expect(parseAppleGpuName('Apple M1, or similar')).toBe('Apple M1');
    expect(parseAppleGpuName('apple a17 pro')).toBe('Apple A17 Pro');
  });

  it('returns undefined when no Apple GPU is named', () => {
    expect(parseAppleGpuName('')).toBeUndefined();
    expect(parseAppleGpuName('Apple GPU')).toBeUndefined();
    expect(parseAppleGpuName('WebKit WebGL')).toBeUndefined();
    expect(parseAppleGpuName('apple apple')).toBeUndefined();
    expect(parseAppleGpuName('ANGLE (NVIDIA, NVIDIA GeForce RTX 4090)'))
        .toBeUndefined();
  });
});

describe('classifyFormFactor', () => {
  it('classifies user agents', () => {
    expect(classifyFormFactor(MAC_USER_AGENT, 0)).toBe('desktop');
    expect(classifyFormFactor(IPHONE_USER_AGENT, 5)).toBe('phone');
    expect(classifyFormFactor(
               'Mozilla/5.0 (iPod touch; CPU iPhone OS 15_0 like Mac OS X)', 5))
        .toBe('phone');
    expect(classifyFormFactor(
               'Mozilla/5.0 (iPad; CPU OS 18_6 like Mac OS X)', 5))
        .toBe('tablet');
    expect(classifyFormFactor(WINDOWS_USER_AGENT, 0)).toBe('unknown');
    expect(classifyFormFactor('', 0)).toBe('unknown');
  });

  it('treats a Mac user agent with touch support as an iPad', () => {
    expect(classifyFormFactor(MAC_USER_AGENT, 5)).toBe('tablet');
  });
});

describe('readWebGlRenderer', () => {
  it('uses RENDERER when it is not masked', () => {
    const gl = fakeWebGl({1: 'Apple M1, or similar', 2: 'unused'});

    expect(readWebGlRenderer(gl)).toBe('Apple M1, or similar');
    expect(gl.getExtension).not.toHaveBeenCalled();
  });

  it('uses the debug extension when RENDERER is masked', () => {
    const renderer = angleMetalRenderer('Apple M5 Max');

    expect(readWebGlRenderer(fakeWebGl({1: 'WebKit WebGL', 2: renderer})))
        .toBe(renderer);
    expect(readWebGlRenderer(fakeWebGl({1: '', 2: renderer}))).toBe(renderer);
  });

  it('returns an empty string when nothing is available', () => {
    expect(readWebGlRenderer(fakeWebGl({1: 'WebKit WebGL'}, false))).toBe('');
    expect(readWebGlRenderer(fakeWebGl({1: 'WebKit WebGL', 2: 7}))).toBe('');
  });
});

describe('overrideAdapterInfoName', () => {
  it('reports the name and passes other fields through', () => {
    const device = fakeDevice(new FakeAdapterInfo('apple', 'metal-3'));

    overrideAdapterInfoName(device, 'Apple M5 Max');

    const info = device.adapterInfo as unknown as FakeAdapterInfo;
    expect(info.device).toBe('Apple M5 Max');
    expect(info.description).toBe('Apple M5 Max');
    expect(info.vendor).toBe('apple');
    expect(info.architecture).toBe('metal-3');
    expect(info.subgroupMinSize).toBe(32);
    expect(info.subgroupMaxSize).toBe(64);
    expect(info.isFallbackAdapter).toBeFalse();
    expect(info.describe()).toBe('apple metal-3');
  });
});

describe('fixDeviceInfoWithWebGl', () => {
  it('reports the WebGL renderer GPU for Chromium Apple adapters', () => {
    const device = fakeDevice(new FakeAdapterInfo('apple', 'metal-3'));
    const platform = fakePlatform(angleMetalRenderer('Apple M5 Max'));

    fixDeviceInfoWithWebGl(device, {}, platform);

    expect(device.adapterInfo.device).toBe('Apple M5 Max');
    expect(device.adapterInfo.description).toBe('Apple M5 Max');
    expect(device.adapterInfo.architecture).toBe('metal-3');
    expect(platform.probeWebGlRenderer).toHaveBeenCalledTimes(1);
  });

  it('probes "common-3" adapters too', () => {
    const device = fakeDevice(new FakeAdapterInfo('apple', 'common-3'));

    fixDeviceInfoWithWebGl(
        device, {}, fakePlatform(angleMetalRenderer('Apple M2')));

    expect(device.adapterInfo.device).toBe('Apple M2');
  });

  it('does not probe when disabled', () => {
    const info = new FakeAdapterInfo('apple', 'metal-3');
    const device = fakeDevice(info);
    const platform = fakePlatform(angleMetalRenderer('Apple M5 Max'));

    fixDeviceInfoWithWebGl(device, {identifyGpuWithWebGl: false}, platform);

    expect(platform.probeWebGlRenderer).not.toHaveBeenCalled();
    expect(device.adapterInfo as unknown).toBe(info);
  });

  it('leaves adapter info that already names the GPU alone', () => {
    const info =
        new FakeAdapterInfo('apple', 'metal-3', '', 'Apple M3 Pro (Metal)');
    const device = fakeDevice(info);
    const platform = fakePlatform(angleMetalRenderer('Apple M5 Max'));

    fixDeviceInfoWithWebGl(device, {}, platform);

    expect(platform.probeWebGlRenderer).not.toHaveBeenCalled();
    expect(device.adapterInfo as unknown).toBe(info);
  });

  it('leaves Macs alone when the renderer is unusable', () => {
    for (const renderer of ['', 'WebKit WebGL', 'Apple GPU']) {
      const info = new FakeAdapterInfo('apple', 'metal-3');
      const device = fakeDevice(info);

      fixDeviceInfoWithWebGl(device, {}, fakePlatform(renderer));

      expect(device.adapterInfo as unknown).toBe(info);
    }
  });

  it('does not probe Safari, and hints an A15 on iPhones and iPads', () => {
    const safariInfo = () =>
        new FakeAdapterInfo('apple', 'apple', 'apple', 'apple');
    const mac = fakeDevice(safariInfo());
    const iphone = fakeDevice(safariInfo());
    const ipad = fakeDevice(safariInfo());
    const macPlatform = fakePlatform(angleMetalRenderer('Apple M5 Max'));

    fixDeviceInfoWithWebGl(mac, {}, macPlatform);
    fixDeviceInfoWithWebGl(iphone, {}, fakePlatform('', IPHONE_USER_AGENT, 5));
    fixDeviceInfoWithWebGl(ipad, {}, fakePlatform('', MAC_USER_AGENT, 5));

    expect(macPlatform.probeWebGlRenderer).not.toHaveBeenCalled();
    expect(mac.adapterInfo.device).toBe('apple');
    expect(iphone.adapterInfo.device).toBe(HANDHELD_APPLE_GPU_HINT);
    expect(ipad.adapterInfo.description).toBe(HANDHELD_APPLE_GPU_HINT);
    expect(ipad.adapterInfo.vendor).toBe('apple');
  });

  it('ignores other GPUs and browsers without adapterInfo', () => {
    const info = new FakeAdapterInfo('nvidia', 'ada');
    const device = fakeDevice(info);
    const platform = fakePlatform(angleMetalRenderer('Apple M5 Max'));

    fixDeviceInfoWithWebGl(device, {}, platform);
    fixDeviceInfoWithWebGl(fakeDevice(undefined), {}, platform);

    expect(platform.probeWebGlRenderer).not.toHaveBeenCalled();
    expect(device.adapterInfo as unknown).toBe(info);
  });
});
