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

import {TensorBufferType} from './wasm_binding_types';

/**
 * The accelerators that LiteRt Web supports.
 */
export const ACCELERATORS = ['webgpu', 'wasm'] as const;
/**
 * The type for accelerators that LiteRt Web supports.
 */
export type Accelerator = (typeof ACCELERATORS)[number];

/**
 * The default tensor buffer type for each accelerator when copying to that
 * accelerator.
 */
export const AcceleratorDefaultTensorBufferType = {
  'webgpu': TensorBufferType.WEB_GPU_BUFFER_PACKED,
  'wasm': TensorBufferType.HOST_MEMORY,
};

/**
 * The accelerator name for each tensor buffer type.
 */
export const TensorBufferTypeToAccelerator = {
  [TensorBufferType.HOST_MEMORY]: 'wasm',
  [TensorBufferType.WEB_GPU_BUFFER]: 'webgpu',
  [TensorBufferType.WEB_GPU_BUFFER_FP16]: 'webgpu',
  [TensorBufferType.WEB_GPU_BUFFER_PACKED]: 'webgpu',
} as const;
