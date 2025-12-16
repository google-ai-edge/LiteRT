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

import {Environment} from './environment';
import {getGlobalLiteRt} from './global_litert';
import {Tensor} from './tensor';
import {LiteRtTensorBuffer, TensorBufferType} from './wasm_binding_types';

/**
 * Copies a tensor from host memory to host memory.
 *
 * @param cpuTensor The tensor to copy.
 * @param options Optional parameters.
 *     - environment: The environment to use for the destination tensor.
 *       If not provided, the source tensor's environment will be used.
 * @returns A promise that resolves to the copied tensor.
 */
export async function copyHostMemoryToHostMemory(
    cpuTensor: Tensor,
    options: {
      environment?: Environment,
    } = {},
    ): Promise<Tensor> {
  const environment = options.environment ?? cpuTensor.environment;
  const liteRtWasm = getGlobalLiteRt().liteRtWasm;
  const srcTensorBuffer = cpuTensor.liteRtTensorBuffer;
  const bufferType = srcTensorBuffer.bufferType();
  if (bufferType.value !== TensorBufferType.HOST_MEMORY) {
    throw new Error(
        'Source tensor is not in host memory. Cannot copy to host memory.');
  }

  const srcTensorMemoryPtr = srcTensorBuffer.lock(
      liteRtWasm.LiteRtTensorBufferLockMode.READ,
  );
  let destTensorBuffer: LiteRtTensorBuffer|undefined;
  try {
    destTensorBuffer = liteRtWasm.LiteRtTensorBuffer.createManaged(
        environment.liteRtEnvironment,
        liteRtWasm.LiteRtTensorBufferType.HOST_MEMORY,
        srcTensorBuffer.tensorType(),
        srcTensorBuffer.size(),
    );

    const destMemoryPointer = destTensorBuffer.lock(
        liteRtWasm.LiteRtTensorBufferLockMode.WRITE,
    );
    try {
      const srcTensorMemoryView = new Uint8Array(
          liteRtWasm.HEAPU8.buffer,
          srcTensorMemoryPtr,
          srcTensorBuffer.size(),
      );
      liteRtWasm.HEAPU8.set(srcTensorMemoryView, destMemoryPointer);
    } finally {
      destTensorBuffer.unlock();
    }
  } finally {
    srcTensorBuffer.unlock();
  }
  if (!destTensorBuffer) {
    throw new Error('Failed to create destination tensor buffer.');
  }
  return new Tensor(destTensorBuffer, environment);
}
