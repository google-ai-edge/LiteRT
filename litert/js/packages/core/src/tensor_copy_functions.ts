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

import {copyHostMemoryToHostMemory} from './cpu_copy_functions';
import {cpuTensorToGpuTensor, gpuTensorToCpuTensor} from './gpu_copy_functions';
import {CopyOptions, Tensor, TensorCopyFn} from './tensor';
import {TensorBufferType} from './wasm_binding_types';

function makeMoveTo(copyTo: TensorCopyFn): TensorCopyFn {
  return async (tensor: Tensor, options?: CopyOptions) => {
    const result = await copyTo(tensor, options);
    tensor.delete();
    return result;
  };
}

/**
 * Registers functions to copy tensors between the CPU and WebGPU accelerators.
 */
export function registerCopyFunctions() {
  Tensor.copyFunctions.set(TensorBufferType.HOST_MEMORY, new Map([
                             [
                               TensorBufferType.HOST_MEMORY, {
                                 copyTo: copyHostMemoryToHostMemory,
                                 // There might be a more efficient way to move
                                 // from CPU to CPU.
                                 moveTo: makeMoveTo(copyHostMemoryToHostMemory),
                               }
                             ],
                             [
                               TensorBufferType.WEB_GPU_BUFFER_PACKED, {
                                 copyTo: cpuTensorToGpuTensor,
                                 moveTo: makeMoveTo(cpuTensorToGpuTensor),
                               }
                             ],
                           ]));

  Tensor.copyFunctions.set(TensorBufferType.WEB_GPU_BUFFER_PACKED, new Map([
                             [
                               TensorBufferType.HOST_MEMORY, {
                                 copyTo: gpuTensorToCpuTensor,
                                 moveTo: makeMoveTo(gpuTensorToCpuTensor),
                               }
                             ],
                           ]));
}
