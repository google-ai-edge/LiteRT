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
 * @fileoverview Reports a more specific GPU in a WebGPU device's adapter info.
 *
 * Browsers bucket `GPUAdapterInfo`. Chrome reports every Apple Silicon GPU (M1
 * through M5) as vendor "apple", architecture "metal-3" (or "common-3"), with
 * an empty device and description, and Safari reports "apple" in every field.
 * The Wasm side (ML Drift's `IdentifyGpu`, in
 * third_party/ml_drift/webgpu/gpu_identification.h) picks kernel tunings per
 * GPU from `device.adapterInfo`, so this module fills in the adapter info's
 * `device` and `description` with the GPU's name when it can find one.
 *
 * Only Apple GPUs are handled today. LiteRT.js and LiteRT-LM.js both call
 * `fixDeviceInfoWithWebGl` on the WebGPU devices they create, so that the Wasm
 * side never needs to touch WebGL or canvases itself.
 */

/** Options for `fixDeviceInfoWithWebGl`. */
export declare interface FixDeviceInfoWithWebGlOptions {
  /**
   * Whether to create a short-lived WebGL context to read the GPU renderer
   * name. Defaults to true.
   *
   * The probe runs at most once per page and only in Chromium-based browsers,
   * but it takes one of the page's limited WebGL context slots while it runs.
   */
  identifyGpuWithWebGl?: boolean;
}

/**
 * The GPU reported for iPhones and iPads whose GPU is unknown. The A15
 * (5 GPU cores) is a conservative choice: iPhones that run WebGPU have A13 to
 * A19 chips, and iPads span A- and M-series chips.
 */
export const HANDHELD_APPLE_GPU_HINT = 'Apple A15';

/** The kind of device a browser runs on, as told by the user agent. */
export type FormFactor = 'desktop'|'phone'|'tablet'|'unknown';

/** The subset of `WebGLRenderingContext` that `readWebGlRenderer` uses. */
export declare interface WebGlRendererSource {
  readonly RENDERER: number;
  getParameter(pname: number): unknown;
  getExtension(name: string): unknown;
}

const APPLE_GPU_PATTERN = /\bapple ([am])(\d{1,2})(?: (pro|max|ultra))?\b/i;

/**
 * Returns the Apple GPU name in `text` in canonical form (for example "Apple M5
 * Max" for the ANGLE renderer "ANGLE (Apple, ANGLE Metal Renderer: Apple M5
 * Max, Unspecified Version)"), or undefined if `text` names no Apple GPU.
 */
export function parseAppleGpuName(text: string): string|undefined {
  const match = APPLE_GPU_PATTERN.exec(text);
  if (!match) return undefined;
  const [, series, generation, tier] = match;
  const tierName =
      tier ? ` ${tier[0].toUpperCase()}${tier.slice(1).toLowerCase()}` : '';
  return `Apple ${series.toUpperCase()}${generation}${tierName}`;
}

/**
 * Classifies the device from `navigator.userAgent` and
 * `navigator.maxTouchPoints`. iPadOS reports a Mac user agent by default, so a
 * Mac user agent with touch support counts as a tablet.
 */
export function classifyFormFactor(
    userAgent: string, maxTouchPoints: number): FormFactor {
  // iPhone and iPad user agents also contain "like Mac OS X".
  if (/iPhone|iPod/i.test(userAgent)) return 'phone';
  if (/iPad/i.test(userAgent)) return 'tablet';
  if (/Macintosh|Mac OS X/i.test(userAgent)) {
    return maxTouchPoints > 0 ? 'tablet' : 'desktop';
  }
  return 'unknown';
}

/**
 * Returns the renderer name reported by a WebGL context, or an empty string.
 *
 * Reads `RENDERER` first, and only falls back to the
 * `WEBGL_debug_renderer_info` extension when `RENDERER` is masked (Chrome
 * reports "WebKit WebGL"). Firefox already returns a sanitized renderer name in
 * `RENDERER` and logs a deprecation warning for the extension.
 */
export function readWebGlRenderer(gl: WebGlRendererSource): string {
  const renderer = gl.getParameter(gl.RENDERER);
  if (typeof renderer === 'string' && renderer !== '' &&
      renderer.toLowerCase() !== 'webkit webgl') {
    return renderer;
  }
  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info') as
          {UNMASKED_RENDERER_WEBGL: number} |
      null;
  if (!debugInfo) return '';
  const unmasked = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
  return typeof unmasked === 'string' ? unmasked : '';
}

const WEBGL_CONTEXT_ATTRIBUTES: WebGLContextAttributes = {
  alpha: false,
  depth: false,
  stencil: false,
  antialias: false,
  failIfMajorPerformanceCaveat: true,
};

function getWebGlContext(canvas: OffscreenCanvas|HTMLCanvasElement):
    WebGLRenderingContext|WebGL2RenderingContext|null {
  return (canvas.getContext('webgl2', WEBGL_CONTEXT_ATTRIBUTES) ??
          canvas.getContext('webgl', WEBGL_CONTEXT_ATTRIBUTES)) as
      WebGLRenderingContext |
      WebGL2RenderingContext | null;
}

let cachedWebGlRenderer: string|undefined;

/**
 * Returns the GPU renderer name from a throwaway WebGL context, or an empty
 * string if WebGL is unavailable. Never throws.
 *
 * The context has no extra buffers, refuses software rendering, and is
 * released right away. The result is cached because it doesn't change during
 * the lifetime of the page.
 */
export function probeWebGlRenderer(): string {
  if (cachedWebGlRenderer !== undefined) return cachedWebGlRenderer;
  let renderer = '';
  try {
    let gl: WebGLRenderingContext|WebGL2RenderingContext|null = null;
    if (typeof OffscreenCanvas !== 'undefined') {
      gl = getWebGlContext(new OffscreenCanvas(1, 1));
    }
    if (!gl && typeof document !== 'undefined') {
      const canvas = document.createElement('canvas');
      canvas.width = 1;
      canvas.height = 1;
      gl = getWebGlContext(canvas);
    }
    if (gl) {
      renderer = readWebGlRenderer(gl);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    }
  } catch {
    renderer = '';
  }
  cachedWebGlRenderer = renderer;
  return renderer;
}

/**
 * Makes `device.adapterInfo` report `gpuName` as its `device` and
 * `description`. Every other field (vendor, architecture, subgroup sizes, ...)
 * still comes from the browser's `GPUAdapterInfo`.
 */
export function overrideAdapterInfoName(device: GPUDevice, gpuName: string) {
  const info = device.adapterInfo;
  const wrappedInfo = new Proxy(info, {
    get(target, property) {
      if (property === 'device' || property === 'description') {
        return gpuName;
      }
      // Use the real object as the receiver: GPUAdapterInfo's getters throw
      // "Illegal invocation" for any other `this`.
      const value: unknown = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  Object.defineProperty(device, 'adapterInfo', {
    value: wrappedInfo,
    configurable: true,
    enumerable: true,
  });
}

/** Platform queries used by `fixDeviceInfoWithWebGl`, faked in tests. */
export declare interface GpuPlatform {
  probeWebGlRenderer(): string;
  userAgent: string;
  maxTouchPoints: number;
}

function getBrowserPlatform(): GpuPlatform {
  const nav = typeof navigator === 'undefined' ? undefined : navigator;
  return {
    probeWebGlRenderer,
    userAgent: nav?.userAgent ?? '',
    maxTouchPoints: nav?.maxTouchPoints ?? 0,
  };
}

/**
 * Reports a more specific GPU in `device.adapterInfo` when the browser only
 * reports a bucketed Apple GPU. Does nothing for other GPUs, when the adapter
 * info already names the GPU (for example with Chrome's
 * chrome://flags/#enable-webgpu-developer-features), or in browsers without
 * `GPUDevice.adapterInfo`.
 *
 * The GPU is taken from, in order:
 *  1. For Chromium's Apple adapters (architecture "metal-3" or "common-3")
 *     only, the WebGL renderer name, unless `identifyGpuWithWebGl` is false.
 *     Safari's WebGL renderer is always "Apple GPU", so it is not probed.
 *  2. A form-factor hint, not a detected GPU: `HANDHELD_APPLE_GPU_HINT` on
 *     iPhones, iPods and iPads. Otherwise the adapter info is left alone, and
 *     ML Drift assumes an M1.
 *
 * Call this right after `GPUAdapter.requestDevice()`, before handing the device
 * to Wasm. It changes `device.adapterInfo` in place (as an own property of
 * `device`), so every reader of the device's adapter info, including
 * Emscripten's `wgpuDeviceGetAdapterInfo`, sees the result.
 */
export function fixDeviceInfoWithWebGl(
    device: GPUDevice, options: FixDeviceInfoWithWebGlOptions = {},
    platform: GpuPlatform = getBrowserPlatform()) {
  const info = device.adapterInfo as GPUAdapterInfo | undefined;
  if (!info || info.vendor.toLowerCase() !== 'apple') return;
  if (parseAppleGpuName(`${info.device} ${info.description}`)) return;

  let gpuName: string|undefined;
  const isChromiumAdapter =
      info.architecture === 'metal-3' || info.architecture === 'common-3';
  if (isChromiumAdapter && (options.identifyGpuWithWebGl ?? true)) {
    gpuName = parseAppleGpuName(platform.probeWebGlRenderer());
  }
  if (!gpuName) {
    const formFactor =
        classifyFormFactor(platform.userAgent, platform.maxTouchPoints);
    if (formFactor === 'phone' || formFactor === 'tablet') {
      gpuName = HANDHELD_APPLE_GPU_HINT;
    }
  }
  if (gpuName) {
    overrideAdapterInfoName(device, gpuName);
  }
}
