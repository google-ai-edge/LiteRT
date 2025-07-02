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

import {Dimensions, DTYPE_TO_ARRAY_TYPE} from './constants';
import {getGlobalLiteRt} from './global_litert';
import {Tensor} from './tensor';
import {CpuTensorReference} from './wasm_binding_types';

/**
 * A box for passing a reference around. This is useful for changing the value
 * of the reference without having to update all the references.
 */
export interface Box<T> {
  val: T;
}

/**
 * Check if the browser supports WebGPU.
 */
export function isWebGPUSupported(): boolean {
  return !!(
      typeof globalThis !== 'undefined' && (globalThis.navigator) &&
      (globalThis.navigator.gpu));
}

/**
 * A callback for reporting GPU errors.
 */
export type GpuErrorReporter = (error: GPUError, callsite: string) => void;

const ERROR_SCOPES = ['internal', 'out-of-memory', 'validation'] as const;

/**
 * Push error scopes for the given device.
 */
export function pushErrorScopes(device: GPUDevice) {
  for (const scopeType of ERROR_SCOPES) {
    device.pushErrorScope(scopeType);
  }
}

/**
 * Pop error scopes for the given device.
 */
export function popErrorScopes(
    device: GPUDevice, callsite: string, reportError: GpuErrorReporter) {
  for (let i = 0; i < ERROR_SCOPES.length; ++i) {
    device.popErrorScope().then(error => {
      if (error) {
        reportError(error, callsite);
      }
    });
  }
}

/**
 * A GPUDevice with the adapterInfo property.
 */
export declare interface GPUDeviceWithAdapterInfo extends GPUDevice {
  // TODO: b/423997093 - Update WebGPU types so we don't have to do this.
  adapterInfo: GPUAdapterInfo;
}

/**
 * Converts a LiteRT tensor shape to a WebGPU tensor shape.
 */
export function getBhwcShapeFromInputShape(shape: Dimensions):
    [number, number, number, number] {
  const shape4d: [number, number, number, number] = [1, 1, 1, 1];
  switch (shape.length) {
    case 1:
      shape4d[3] = shape[0];
      break;
    case 2:
      shape4d[3] = shape[1];
      shape4d[2] = shape[0];
      break;
    case 3:
      shape4d[3] = shape[2];
      shape4d[2] = shape[1];
      shape4d[1] = shape[0];
      break;
    case 4:
      shape4d[3] = shape[3];
      shape4d[2] = shape[2];
      shape4d[1] = shape[1];
      shape4d[0] = shape[0];
      break;
    default:
      // TODO: Support higher rank tensors for WebGPU inference, once ML Drift
      // supports it.
      // ML Drift currently only supports 1D~4D tensors for the converted TFLite
      // model inference. LiteRT-Web won't be able to support higher rank
      // tensors for WebGPU accelerator until ML Drift supports it.
      throw new Error(
          'Only 1D~4D tensors are supported, but got shape: ' +
          shape.toString() + '.');
  }
  return shape4d;
}

/**
 * Internal helper function for converting a GPU backed tensor to a CPU backed
 * tensor.
 */
export async function gpuTensorToCpuTensor(gpuTensor: Tensor): Promise<Tensor> {
  const device = await getGlobalLiteRt().getWebGpuDevice();

  // Convert from MLDrift format to a more standard format.
  const converter = getGlobalLiteRt().getConverterFactory().makeConverterToTfjs(
      gpuTensor.reference);
  const buffer: GPUBuffer = converter.convertToTfjs(gpuTensor.reference);

  // Create a staging buffer we can read from.
  const stagingBuffer = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false,
  });

  // Copy the GPU buffer to the staging buffer.
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, buffer.size);
  device.queue.submit([commandEncoder.finish()]);

  // Map the staging buffer to an array buffer.
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const mappedBuffer = stagingBuffer.getMappedRange();
  const mappedArray = new Uint8Array(mappedBuffer);

  // Set the data on the CPU tensor.
  const cpuTensorConstructor = getGlobalLiteRt().liteRtWasm.CpuTensor;
  const cpuTensorReference = new cpuTensorConstructor(mappedArray.byteLength);
  cpuTensorReference.data().set(mappedArray);

  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return new Tensor({
    type: gpuTensor.type,
    accelerator: 'wasm',
    reference: cpuTensorReference
  });
}


/**
 * Internal helper function for converting a CPU backed tensor to a WebGPU
 * backed tensor.
 */
export async function cpuTensorToGpuTensor(
    cpuTensor: Tensor,
    ): Promise<Tensor> {
  // TODO: b/431839967 - Avoid the two staging buffers and make this more
  // efficient.
  // TODO: Make this synchronous.
  const device = await getGlobalLiteRt().getWebGpuDevice();

  const cpuTensorData = (cpuTensor.reference as CpuTensorReference).data();
  const typedArrayConstructor = DTYPE_TO_ARRAY_TYPE[cpuTensor.type.dtype];
  const typedArray = new typedArrayConstructor(
      // Cast is needed to avoid 'SharedArrayBuffer' in the type.
      cpuTensorData.buffer as ArrayBuffer,
      cpuTensorData.byteOffset,
      cpuTensorData.length,
  );

  // Create a staging buffer we can write to.
  const stagingBuffer = device.createBuffer({
    size: typedArray.byteLength,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  const mappedBuffer = await stagingBuffer.getMappedRange();

  if (typedArray instanceof Float32Array) {
    const mappedArray = new Float32Array(mappedBuffer);
    mappedArray.set(typedArray);
  } else if (typedArray instanceof Int32Array) {
    const mappedArray = new Int32Array(mappedBuffer);
    mappedArray.set(typedArray);
  } else {
    throw new Error(
        'Unsupported typed array type: ' +
        (typedArray as Uint8Array).constructor.name);
  }
  stagingBuffer.unmap();

  // Copy the staging buffer to a temporary buffer that MLDrift can read from.
  const tempBuffer = device.createBuffer({
    size: stagingBuffer.size,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST |
        GPUBufferUsage.STORAGE,
  });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
      stagingBuffer, 0, tempBuffer, 0, stagingBuffer.size);
  device.queue.submit([commandEncoder.finish()]);
  stagingBuffer.destroy();

  // Copy the temporary buffer to a new WebGPU Tensor.
  const mlDriftShape =
      getBhwcShapeFromInputShape(cpuTensor.type.layout.dimensions);
  const converter =
      getGlobalLiteRt().getConverterFactory().makeConverterFromTfjs(
          cpuTensor.type.dtype, ...mlDriftShape);
  // This submits the command queue, so it's safe to destroy the staging
  // buffer after.
  const gpuTensorReference = converter.convertFromTfjs(tempBuffer);
  tempBuffer.destroy();

  return new Tensor({
    type: cpuTensor.type,
    accelerator: 'webgpu',
    reference: gpuTensorReference,
  });
}
