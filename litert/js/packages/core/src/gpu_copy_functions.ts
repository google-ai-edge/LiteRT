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

import {getDataType} from './datatypes';
import {Environment} from './environment';
import {getGlobalLiteRt} from './global_litert';
import {Tensor} from './tensor';

/**
 * Internal helper function for converting a CPU backed tensor to a WebGPU
 * backed tensor.
 *
 * @param cpuTensor The CPU backed tensor to convert.
 * @param options Optional parameters.
 *     - environment: The environment to use for the WebGPU tensor.
 *       If not provided, the CPU tensor's environment will be used.
 *     - bufferDataType: The data type to use for the WebGPU buffer.
 * @returns A promise that resolves to the WebGPU backed tensor.
 */
export async function cpuTensorToGpuTensor(
    cpuTensor: Tensor,
    options: {
      environment?: Environment,
    } = {},
    ): Promise<Tensor> {
  // TODO: b/431839967 - Avoid the two staging buffers and make this more
  // efficient.
  const environment = options.environment ?? cpuTensor.environment;
  const device = environment.webGpuDevice;
  if (!device) {
    throw new Error(
        'No WebGPU device is available. Did you forget to pass a ' +
        'destination environment that has a WebGPU device?');
  }
  const liteRtWasm = getGlobalLiteRt().liteRtWasm;

  const byteLength = cpuTensor.liteRtTensorBuffer.size();
  // The WebGPU buffer size must be a multiple of 4.
  //   1. Add 3 to ensure that size is larger than the smallest multiple of 4
  //      that can hold the data.
  //   2. Bitwise AND with 0b1...100 to 'floor' the value to the nearest
  //      multiple of 4.
  const paddedByteLength = (byteLength + 3) & ~0b11;

  // Create a staging buffer we can write to.
  const stagingBuffer = device.createBuffer({
    size: paddedByteLength,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  const mappedBuffer = await stagingBuffer.getMappedRange();

  const mappedArray = new Uint8Array(mappedBuffer);
  const cpuMemoryPtr = cpuTensor.liteRtTensorBuffer.lock(
      liteRtWasm.LiteRtTensorBufferLockMode.READ,
  );
  try {
    const cpuMemoryView = new Uint8Array(
        liteRtWasm.HEAPU8.buffer,
        cpuMemoryPtr,
        cpuTensor.liteRtTensorBuffer.size(),
    );
    mappedArray.set(cpuMemoryView);
  } finally {
    cpuTensor.liteRtTensorBuffer.unlock();
  }

  stagingBuffer.unmap();

  // Copy the staging buffer to a buffer that MLDrift can read from.
  // Is this necessary?
  const buffer = device.createBuffer({
    size: paddedByteLength,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST |
        GPUBufferUsage.STORAGE,
  });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
      stagingBuffer, 0, buffer, 0, paddedByteLength);
  device.queue.submit([commandEncoder.finish()]);
  stagingBuffer.destroy();

  return new Tensor(
      buffer,
      cpuTensor.type.layout.dimensions,
      cpuTensor.type.dtype,
      environment,
      () => {
        buffer.destroy();
      },
  );
}

/**
 * Internal helper function for converting a WebGPU backed tensor to a CPU
 * backed tensor.
 *
 * @param gpuTensor The WebGPU backed tensor to convert.
 * @param options Optional parameters.
 *     - environment: The environment to use for the CPU tensor. If
 *       not provided, the WebGPU tensor's environment will be used.
 * @returns A promise that resolves to the CPU backed tensor.
 */
export async function gpuTensorToCpuTensor(
    gpuTensor: Tensor,
    options: {
      environment?: Environment,
    } = {},
    ): Promise<Tensor> {
  const environment = options.environment ?? gpuTensor.environment;
  // We use the input tensor's WebGPU device, not the destination environment's
  // WebGPU device, because the input tensor's WebGPU device is the one that
  // was used to create the WebGPU buffer.
  const device = gpuTensor.environment.webGpuDevice;
  if (!device) {
    throw new Error(
        'No WebGPU device is available. Does the source tensor have a WebGPU ' +
        'device?');
  }

  const liteRtWasm = getGlobalLiteRt().liteRtWasm;

  const tensorBuffer = gpuTensor.liteRtTensorBuffer;
  const bufferType = tensorBuffer.bufferType();
  if (bufferType !== liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER_PACKED) {
    throw new Error(`Cannot convert a tensor with a non-WebGPU buffer type ${
        bufferType} to a CPU tensor.`);
  }

  const gpuBuffer = liteRtWasm.WebGPU.getJsObject(
      tensorBuffer.getWebGpuBuffer(),
  );

  const byteOffset = tensorBuffer.offset();

  // Get the number of elements in the tensor.
  const tensorType = tensorBuffer.tensorType();
  const layout = tensorType.layout();
  const numElements = layout.numElements();
  const arrayConstructor =
      getDataType(tensorType.elementType().value).typedArrayConstructor;
  layout.delete();
  tensorType.delete();

  // Copy the buffer to a new one that is mappable if it's not already.
  let mappableBuffer = gpuBuffer;
  let cleanupBuffer =
      () => {};  // No-op unless we create a new mappable buffer.
  if (!(gpuBuffer.usage & GPUBufferUsage.MAP_READ)) {
    mappableBuffer = device.createBuffer({
      size: gpuBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    cleanupBuffer = () => {
      mappableBuffer.destroy();
    };
    const commandEncoder = device.createCommandEncoder();
    // Maybe we should only copy the portion we're going to map?
    commandEncoder.copyBufferToBuffer(
        gpuBuffer, 0, mappableBuffer, 0, gpuBuffer.size);
    device.queue.submit([commandEncoder.finish()]);
  }

  // Map the GPU buffer to a typed array of the correct data type.
  // NOTE: We assume that data is stored in a packed format with no padding,
  // with each element taking up a number of bytes equal to the element's size.
  // If we ever support other formats in C++, we'll need to add additional
  // conversion logic here.
  await mappableBuffer.mapAsync(GPUMapMode.READ);
  const mappedBuffer = mappableBuffer.getMappedRange();

  const mappedArray =
      new arrayConstructor(mappedBuffer, byteOffset, numElements);

  // Copy the mapped array into a CPU tensor.
  const cpuTensor =
      new Tensor(mappedArray, gpuTensor.type.layout.dimensions, environment);

  // Tensor constructor copies the data, so unmapping is okay.
  mappableBuffer.unmap();
  cleanupBuffer();

  return cpuTensor;
}
