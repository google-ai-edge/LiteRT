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

import {cpuTensorToGpuTensor, gpuTensorToCpuTensor} from './gpu_utils';
import {Tensor} from './tensor';

/**
 * Registers functions to copy tensors between the CPU and WebGPU accelerators.
 */
export function registerCopyFunctions() {
  Tensor.copyFunctions['wasm'] = {
    'webgpu': {
      copyTo: cpuTensorToGpuTensor,
      moveTo: async (tensor: Tensor) => {
        const gpuTensor = await cpuTensorToGpuTensor(tensor);
        tensor.delete();
        return gpuTensor;
      },
    },
  };
  Tensor.copyFunctions['webgpu'] = {
    'wasm': {
      copyTo: gpuTensorToCpuTensor,
      moveTo: async (tensor: Tensor) => {
        const cpuTensor = await gpuTensorToCpuTensor(tensor);
        tensor.delete();
        return cpuTensor;
      }
    },
  };
}
