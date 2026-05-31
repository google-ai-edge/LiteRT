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

import {CompiledModel} from './compiled_model';
import {getGlobalLiteRt} from './global_litert';
import {Model} from './model';
import {CompileOptions} from './model_types';

/**
 * A global lock to ensure only one model is being compiled at a time.
 * This is necessary because the C++ WebGPU delegate uses a global callback
 * for streamed weight loading.
 */
let compilationLock: Promise<void> = Promise.resolve();

interface WeightRequest {
  id: number;
  wgpuBufferId: number;
  offset: number;
  length: number;
}

/**
 * Loads a model from a flatbuffer and streams its weights from a separate
 * source directly into WebGPU buffers.
 *
 * @param modelData The model flatbuffer data.
 * @param weightsStream A stream of the weights file.
 * @param compileOptions Options for compilation.
 * @returns A promise resolving to the compiled model.
 */
export async function loadModelAndWeights(
    modelData: Uint8Array,
    weightsStream: ReadableStream<Uint8Array>,
    compileOptions: CompileOptions = {},
    ): Promise<CompiledModel> {
  const liteRt = getGlobalLiteRt();
  const wasm = liteRt.liteRtWasm;
  const env = compileOptions.environment ?? liteRt.getDefaultEnvironment();

  if (!env.webGpuDevice) {
    throw new Error(
        'WebGPU device is required for streamed loading in this implementation.');
  }

  // Wait for the compilation lock
  const myTurn = compilationLock;
  let resolveLock: () => void;
  compilationLock = new Promise((resolve) => {
    resolveLock = resolve;
  });
  await myTurn;

  try {
    const originalCallback = wasm.getStreamWeightsCallback();

    // Register the callback for this specific loading session
    wasm.registerStreamWeightsCallback(async (
        tflIds: Int32Array,
        wgpuBufferIds: Uint32Array,
        offsets: Float64Array,
        lengths: Float64Array,
        ) => {
      // Sort the requests by offset so we can read the stream sequentially
      const requests: WeightRequest[] = [];
      for (let i = 0; i < tflIds.length; i++) {
        requests.push({
          id: tflIds[i],
          wgpuBufferId: wgpuBufferIds[i],
          offset: offsets[i],
          length: lengths[i],
        });
      }
      requests.sort((a, b) => a.offset - b.offset);

      const reader = weightsStream.getReader();
      let streamOffset = 0;
      let buffer = new Uint8Array(0);
      let reqIndex = 0;

      try {
        while (reqIndex < requests.length) {
          const req = requests[reqIndex];
          const relStart = req.offset - streamOffset;
          const relEnd = relStart + req.length;
          if (relEnd <= buffer.length) {
            // We have enough data to fulfill the current tensor request
            if (relStart < 0) {
              throw new Error(
                  `Stream logic error: weight starts before current buffer (req.offset=${
                      req.offset}, streamOffset=${streamOffset}).`);
            }

            let weightData = buffer.subarray(relStart, relEnd);
            // WebGPU writeBuffer requires 4-byte alignment for the data size.
            // If it's not aligned, we create a padded copy.
            if (weightData.byteLength % 4 !== 0) {
              const paddedSize = (weightData.byteLength + 3) & ~3;
              const paddedData = new Uint8Array(paddedSize);
              paddedData.set(weightData);
              weightData = paddedData;
            }

            const gpuBuffer = wasm.WebGPU.getJsObject(req.wgpuBufferId);
            if (!gpuBuffer) {
              throw new Error(
                  `Failed to find GPUBuffer for ID: ${req.wgpuBufferId}`);
            }
            env.webGpuDevice!.queue.writeBuffer(gpuBuffer, 0, weightData);
            reqIndex++;
          } else {
            // Need more data from the stream
            const {done, value} = await reader.read();
            if (done) {
              throw new Error(
                  `Stream ended before all weights were loaded.`);
            }
            const newBuffer = new Uint8Array(buffer.length + value.length);
            newBuffer.set(buffer);
            newBuffer.set(value, buffer.length);
            buffer = newBuffer;
          }

          // Compact buffer to free up memory
          if (reqIndex < requests.length) {
            const nextStartRel = requests[reqIndex].offset - streamOffset;
            const bytesToDiscard = Math.min(nextStartRel, buffer.length);
            // Compact if we have at least 1MB of processed data, or if we can
            // discard the entire buffer
            if (bytesToDiscard > 1024 * 1024 ||
                bytesToDiscard === buffer.length) {
              buffer = buffer.slice(bytesToDiscard);
              streamOffset += bytesToDiscard;
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    });

    try {
      // Load Model
      const ptr = wasm._malloc(modelData.byteLength);
      wasm.HEAPU8.set(modelData, ptr);
      const wasmModel =
          wasm.loadModel(env.liteRtEnvironment, ptr, modelData.byteLength);

      const loadedModel = new Model(wasmModel, () => {
        wasm._free(ptr);
      });

      const fullOptions: Required<CompileOptions> = {
        environment: env,
        accelerator: compileOptions.accelerator ??
            (env.webGpuDevice ? 'webgpu' : 'wasm'),
        cpuOptions:
            compileOptions.cpuOptions ?? {numThreads: wasm.getThreadCount()},
        gpuOptions: compileOptions.gpuOptions ?? {},
        webNNOptions: compileOptions.webNNOptions ?? {},
      };
      // This compilation call triggers the WebGPU delegate which will
      // eventually invoke our callback.
      const wasmCompiledModel = await wasm.compileModel(
          env.liteRtEnvironment, wasmModel, fullOptions);

      const compiledModel = new CompiledModel(
          loadedModel, wasmCompiledModel, fullOptions, () => {
            liteRt._unregisterObjectForDeletion(compiledModel);
          });
      liteRt._registerObjectForDeletion(compiledModel);
      return compiledModel;
    } finally {
      // Restore the original callback (likely undefined)
      wasm.registerStreamWeightsCallback(originalCallback);
    }
  } finally {
    resolveLock!();
  }
}

