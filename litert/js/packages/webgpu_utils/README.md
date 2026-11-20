# @litertjs/webgpu-utils

WebGPU helpers shared by @litertjs/core (LiteRT.js) and LiteRT-LM.js. It is not
intended to be used directly.

- `createDefaultWebGpuDevice(options)` requests a high-performance WebGPU
  adapter and device with the limits and features LiteRT/LiteRT-LM use, and
  calls `fixDeviceInfoWithWebGl`.
- `fixDeviceInfoWithWebGl(device)` fills in `device.adapterInfo.device` and
  `description` with the GPU's name when the browser only reports a bucketed GPU
  (for example every Apple Silicon GPU in Chrome). The Wasm runtimes read the
  GPU from `device.adapterInfo` to tune their kernels.
