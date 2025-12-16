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

import {CompiledModel, type DType, getDefaultEnvironment, SignatureRunner, Tensor, TensorBufferType, TensorDetails} from '@litertjs/core_litert';
import {type WebGPUBackend} from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';

const WEBGPU_DEVICE_MISMATCH_ERROR_MESSAGE =
    'To convert from TFJS to LiteRT, you must use an environment ' +
    'that has the same WebGPU device as the TFJS backend.\n\n' +
    'Since LiteRT.js sets some WebGPU options by default, the recommended ' +
    'way to share a WebGPU device is to let LiteRT.js create the device when ' +
    'it sets up the default environment ' +
    'and then create a new TFJS WebGPU backnend using that device:\n\n' +
    '```javascript\n' +
    '    const device = await getWebGpuDevice(); // from LiteRT.js\n' +
    '    const adapterInfo = device.adapterInfo;\n' +
    '    tf.removeBackend(\'webgpu\');\n' +
    '    tf.registerBackend(\'webgpu\', () => new WebGPUBackend(device, adapterInfo));\n\n' +
    '    await tf.setBackend(\'webgpu\');\n' +
    '    ```';

function isWebGpuBackend(): boolean {
  // We can't use `instanceof WebGPUBackend` because then we would have to
  // import the actual WebGPU backend rather than just its type.
  return tf.getBackend().includes('webgpu');
}

/**
 * Error thrown when a tensor can not be converted between TFJS and LiteRT.
 */
export class TensorConversionError extends Error {
  tensor?: string|number;
  private originalMessage: string;

  constructor(message: string) {
    super(message);
    this.originalMessage = message;
  }

  setTensor(tensor: string|number) {
    this.message = `For tensor ${tensor}: ${this.originalMessage}`;
    this.tensor = tensor;
  }
}

function tfjsDtypeToLiteRt(tfjsDtype: tf.DataType): DType {
  switch (tfjsDtype) {
    case 'float32':
      return 'float32';
    case 'int32':
      return 'int32';
    default:
      throw new Error(
          `Unsupported type: ${tfjsDtype}. You may need to cast ` +
          'to int32 or float32.');
  }
}

function liteRtDtypeToTfjs(liteRtDtype: DType): tf.DataType {
  switch (liteRtDtype) {
    case 'float32':
      return 'float32';
    case 'int32':
      return 'int32';
    default:
      throw new Error(`Unsupported type: ${liteRtDtype}.`);
  }
}

/**
 * Convert a TFJS tensor to a LiteRT tensor.
 *
 * The TFJS tensor is copied to whichever LiteRT buffer type is most efficient
 * to copy to.
 * - For TFJS tensors on WebGPU, TensorBufferType.WEB_GPU_BUFFER_PACKED is
 *   used (if the TFJS tensor has been uploaded to WebGPU).
 * - Otherwise, TensorBufferType.HOST_MEMORY is used.
 *
 * The user can specify a preferred buffer type for the LiteRT tensor to be
 * created with, but this does not guarantee that the tensor will be created
 * with that buffer type. For example, if the TFJS tensor is on WebGPU and not
 * cached in CPU memory, the WebGPU buffer type will be used even if
 * TensorBufferType.HOST_MEMORY is requested.
 */
export function tfjsToLitert(
    tfjsTensor: tf.Tensor, environment = getDefaultEnvironment(),
    preferredBufferType?: TensorBufferType): Tensor {
  if (isWebGpuBackend()) {
    const backend = tf.backend() as WebGPUBackend;
    const tensorMapValue = backend.tensorMap.get(tfjsTensor.dataId);

    // Sometimes, the user does not want a WebGPU tensor even when the data is
    // on WebGPU (e.g., they're running a CPU model). In this case, we should
    // return a CPU tensor if we can do so efficiently, i.e., if TFJS has the
    // tensor cached in CPU memory.
    const shouldReturnCpuTensor = tensorMapValue?.values &&
        preferredBufferType === TensorBufferType.HOST_MEMORY;
    if (tensorMapValue?.resource && !shouldReturnCpuTensor) {
      // Then the data is on WebGPU, so create a new LiteRT WebGPU /
      // WEB_GPU_BUFFER_PACKED tensor.

      if (backend.device !== environment.webGpuDevice) {
        throw new TensorConversionError(WEBGPU_DEVICE_MISMATCH_ERROR_MESSAGE);
      }

      const gpuData = tfjsTensor.dataToGPU();
      const buffer = gpuData.buffer;
      if (!buffer) {
        throw new TensorConversionError(
            'TFJS tensor did not have a GPU buffer.');
      }

      const dtype = tfjsDtypeToLiteRt(tfjsTensor.dtype);
      return new Tensor(buffer, tfjsTensor.shape, dtype, environment, () => {
        // Return the tfjs GPU data back to the pool.
        gpuData.tensorRef.dispose();
      });
    }
  }

  // Otherwise, copy the data to a CPU / HOST_MEMORY tensor.
  const tfjsData = tfjsTensor.dataSync();
  // dtype is handled implicitly by the TypeArray type returned by
  // dataSync.
  return new Tensor(tfjsData, tfjsTensor.shape, environment);
}

/**
 * Convert a LiteRT tensor to a TFJS tensor.
 *
 * The tensor is copied to the respective TFJS backend.
 * - Wasm CPU tensors will be copied to the TFJS tensor's CPU cache. This
 * is supported on all TFJS backends.
 * - WebGPU tensors will be copied to the TFJS WebGPU backend. This is
 * only supported on the TFJS WebGPU backend.
 */
export function litertToTfjs(tensor: Tensor): tf.Tensor {
  switch (tensor.bufferType) {
    case TensorBufferType.HOST_MEMORY:
      return litertToTfjsCpu(tensor);
    case TensorBufferType.WEB_GPU_BUFFER_PACKED:
      return litertToTfjsWebGpu(tensor);
    default:
      throw new Error('Unsupported accelerator: ' + tensor.accelerator);
  }
}

function litertToTfjsCpu(tensor: Tensor): tf.Tensor {
  const typedArray = tensor.toTypedArray();
  const tfjsDataType = liteRtDtypeToTfjs(tensor.type.dtype);

  return tf.tensor(
      typedArray, [...tensor.type.layout.dimensions], tfjsDataType);
}

function litertToTfjsWebGpu(tensor: Tensor): tf.Tensor {
  if (!isWebGpuBackend()) {
    throw new TensorConversionError(
        'LiteRT WebGPU tensors can only be converted to TFJS WebGPU tensors, ' +
        `but the TFJS backend is ${
            tf.getBackend()}. If you want to convert to a CPU ` +
        'TFJS tensor, please first move (or copy) the LiteRT tensor to CPU ' +
        'with `tensor.moveTo(\'wasm\')` (or `tensor.copyTo(\'wasm\')`).');
  }

  const backend = tf.backend() as WebGPUBackend;
  const device = backend.device;
  if (device !== tensor.environment.webGpuDevice) {
    throw new TensorConversionError(WEBGPU_DEVICE_MISMATCH_ERROR_MESSAGE);
  }

  const litertBuffer = tensor.toGpuBuffer();
  const requiredSizeInBytes = tensor.liteRtTensorBuffer.size();

  let buffer: GPUBuffer;
  if (litertBuffer.size === requiredSizeInBytes) {
    buffer = litertBuffer;
  } else {
    // If the LiteRT buffer is larger than needed, we need to copy it to a
    // new buffer with the correct size, because TFJS does not support
    // buffer views.
    buffer = device.createBuffer({
      size: requiredSizeInBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
    });
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
        litertBuffer, 0, buffer, 0, requiredSizeInBytes);
    device.queue.submit([commandEncoder.finish()]);
  }

  // We do not need to manually make a copy of the WebGPU buffer from
  // LiteRT. TFJS will automatically copy the buffer when creating the
  // tensor (unless we set `zeroCopy: true`, which we don't).
  // https://github.com/tensorflow/tfjs/blob/tfjs-v4.22.0/tfjs-backend-webgpu/src/backend_webgpu.ts#L584-L586

  const tfjsDataType = liteRtDtypeToTfjs(tensor.type.dtype);
  return tf.tensor({buffer}, [...tensor.type.layout.dimensions], tfjsDataType);
}

/** Container types for tensors passed to / from the model. */
type Container<T> = T|Record<string, T>|T[];

// clang-format off
/**
 * Maps the type for a Container of one value type to a type for a Container of
 * another value type.
 */
type MapContainer<T extends Container<unknown>, NewVal> = T extends Array<infer _> ? NewVal[]
    : T extends Record<string, infer _> ? Record<string, NewVal>
    : NewVal;
// clang-format on

/**
 * Map a Container of one value type to the same Container type of another
 * value type.
 *
 * This is similar to a functor.
 *
 * @param t The Container to map.
 * @param f The function to map each value in the Container.
 * @param isA The function to check if the value is of type A.
 */
function mapOnContainer<A, T extends Container<A>, B>(
    t: T, f: (a: A, keyOrIndex?: string|number) => B,
    isA: (val: unknown) => val is A): MapContainer<T, B> {
  if (isA(t)) {
    return f(t) as MapContainer<T, B>;
  } else if (Array.isArray(t)) {
    return t.map(f) as MapContainer<T, B>;
  } else {
    return Object.fromEntries(Object.entries(t as Record<string, A>)
                                  .map(([key, a]) => [key, f(a, key)])) as
        MapContainer<T, B>;
  }
}

/**
 * Map a Container of TFJS tensors to the same Container type but with new
 * values from f.
 *
 * @param t The Container to map.
 * @param f The function to map each value in the Container.
 */
function mapOnTfjs<T extends Container<tf.Tensor>, NewVal>(
    t: T, f: (tfjsTensor: tf.Tensor, keyOrIndex?: string|number) => NewVal):
    MapContainer<T, NewVal> {
  return mapOnContainer(t, f, (val) => val instanceof tf.Tensor);
}

/**
 * Map a Container of LiteRT tensors to the same Container type but with
 * new values from f.
 *
 * @param t The Container to map.
 * @param f The function to map each value in the Container.
 */
function mapOnLiteRt<T extends Container<Tensor>, NewVal>(
    t: T, f: (liteRtTensor: Tensor, keyOrIndex?: string|number) => NewVal):
    MapContainer<T, NewVal> {
  return mapOnContainer(t, f, (val) => val instanceof Tensor);
}

/**
 * Run a LiteRT CompiledModel with TFJS inputs and outputs.
 *
 * If a signature name is not provided, the default signature will be
 * used.
 *
 * When calling a Wasm CPU LiteRT model with WebGPU TFJS tensors, please
 * first call `await tensor.data()` on each tensor to more efficiently
 * copy the tensors to CPU.
 */
// Without signature name
export function runWithTfjsTensors(
    model: CompiledModel|SignatureRunner,
    input: tf.Tensor|tf.Tensor[]): Promise<tf.Tensor[]>;
export function runWithTfjsTensors(
    model: CompiledModel|SignatureRunner,
    input: Record<string, tf.Tensor>): Promise<Record<string, tf.Tensor>>;
// With signature name
export function runWithTfjsTensors(
    model: CompiledModel, signature: string,
    input: tf.Tensor|tf.Tensor[]): Promise<tf.Tensor[]>;
export function runWithTfjsTensors(
    model: CompiledModel, signature: string,
    input: Record<string, tf.Tensor>): Promise<Record<string, tf.Tensor>>;
// The following signatures are needed since TS won't automatically
// distribute a union type across the above function signatures.
// https://github.com/microsoft/TypeScript/issues/14107
export function runWithTfjsTensors(
    model: CompiledModel|SignatureRunner,
    input: tf.Tensor|tf.Tensor[]|
    Record<string, tf.Tensor>): Promise<tf.Tensor[]|Record<string, tf.Tensor>>;
export function runWithTfjsTensors(
    model: CompiledModel, signature: string,
    input: tf.Tensor|tf.Tensor[]|
    Record<string, tf.Tensor>): Promise<tf.Tensor[]|Record<string, tf.Tensor>>;
export async function runWithTfjsTensors(
    model: CompiledModel|SignatureRunner,
    inputOrSignatureName: string|tf.Tensor|tf.Tensor[]|
    Record<string, tf.Tensor>,
    maybeInputOrLiteRt?: tf.Tensor|tf.Tensor[]|
    Record<string, tf.Tensor>): Promise<tf.Tensor[]|Record<string, tf.Tensor>> {
  let signature: string|undefined;
  let tfjsInputs: Record<string, tf.Tensor>|tf.Tensor[]|tf.Tensor;
  if (typeof inputOrSignatureName === 'string') {
    signature = inputOrSignatureName;
    tfjsInputs = maybeInputOrLiteRt as Container<tf.Tensor>;
  } else {
    tfjsInputs = inputOrSignatureName;
  }
  const litertInputDetails = model.getInputDetails();

  // Convert TFJS inputs to LiteRT tensors.
  const inputs = mapOnTfjs(tfjsInputs, (tfjsTensor, keyOrIndex) => {
    try {
      let inputDetails: TensorDetails|undefined;
      if (typeof keyOrIndex === 'number') {
        inputDetails = litertInputDetails[keyOrIndex];
      } else if (typeof keyOrIndex === 'string') {
        inputDetails =
            litertInputDetails.find(details => details.name === keyOrIndex);
      }

      // If the user runs a CPU model with a WebGPU TFJS tensor that is already
      // cached in TFJS CPU memory, we should create a CPU tensor from that
      // cache rather than a WebGPU tensor (which `model.run` would then
      // have to copy back to CPU).
      const preferredBufferType =
          inputDetails?.supportedBufferTypes.values().next().value;

      return tfjsToLitert(
          tfjsTensor, model.options.environment, preferredBufferType);
    } catch (e) {
      if (e instanceof TensorConversionError && keyOrIndex !== undefined) {
        e.setTensor(keyOrIndex);
      }
      throw e;
    }
  });

  // Run the model.
  let outputs: Record<string, Tensor>|Tensor[];
  if (signature) {
    if (!(model instanceof CompiledModel)) {
      throw new Error(
          'Signature specified but a SignatureRunner was passed instead of a' +
          ' model');
    }
    outputs = await model.run(signature, inputs);
  } else {
    outputs = await model.run(inputs);
  }

  // Delete all the LiteRT inputs that we created to pass to the model.
  mapOnLiteRt(inputs, tensor => {
    tensor.delete();
  });

  // Convert LiteRT outputs to TFJS tensors.
  return mapOnLiteRt(outputs, (tensor, keyOrIndex) => {
    let tfjsTensor: tf.Tensor;
    try {
      tfjsTensor = litertToTfjs(tensor);
    } catch (e) {
      if (e instanceof TensorConversionError && keyOrIndex !== undefined) {
        e.setTensor(keyOrIndex);
      }
      throw e;
    } finally {
      // It is safe to delete the LiteRT tensor because TFJS will have already
      // enqueued the WebGPU buffer copy.
      tensor.delete();
    }
    return tfjsTensor;
  });
}
