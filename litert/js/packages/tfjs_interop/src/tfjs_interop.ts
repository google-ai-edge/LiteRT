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

import {Accelerator, CompiledModel, ConverterFactory, CpuTensorReference, DType, DTYPE_TO_ARRAY_TYPE, getGlobalLiteRt, LiteRt, LiteRtNotLoadedError, SignatureRunner, SUPPORTED_DTYPES, Tensor} from '@litertjs/core';
import type {WebGPUBackend} from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';

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

function getConverterFactory(): ConverterFactory {
  try {
    return getGlobalLiteRt().getConverterFactory();
  } catch (e) {
    if (e instanceof LiteRtNotLoadedError) {
      console.error(
          `You must load a global LiteRT instance before calling ` +
          `conversion functions without a liteRt parameter.`);
      throw e;
    }
    throw e;
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

function getBhwcShapeForTfjs(tfjsTensor: tf.Tensor):
    [number, number, number, number] {
  const shape4d: [number, number, number, number] = [1, 1, 1, 1];
  switch (tfjsTensor.shape.length) {
    case 1:
      shape4d[3] = tfjsTensor.shape[0];
      break;
    case 2:
      shape4d[3] = tfjsTensor.shape[1];
      shape4d[2] = tfjsTensor.shape[0];
      break;
    case 3:
      shape4d[3] = tfjsTensor.shape[2];
      shape4d[2] = tfjsTensor.shape[1];
      shape4d[1] = tfjsTensor.shape[0];
      break;
    case 4:
      shape4d[3] = tfjsTensor.shape[3];
      shape4d[2] = tfjsTensor.shape[2];
      shape4d[1] = tfjsTensor.shape[1];
      shape4d[0] = tfjsTensor.shape[0];
      break;
    default:
      // TODO: Support higher rank tensors for WebGPU inference, once ML Drift
      // supports it.
      // ML Drift currently only supports 1D~4D tensors for the converted TFLite
      // model inference. LiteRT-Web won't be able to support higher rank
      // tensors for WebGPU accelerator until ML Drift supports it.
      throw new Error(
          'Only 1D~4D tensors are supported, but got shape: ' +
          tfjsTensor.shape.toString() + '.');
  }
  return shape4d;
}

/**
 * Convert a TFJS tensor to a LiteRT tensor on the given accelerator.
 *
 * When converting from TFJS WebGPU to LiteRT Wasm CPU, first call
 * `await tensor.data()` to make sure the data is on CPU.
 *
 * Converting from TFJS to LiteRT WebGPU is only supported when using the TFJS
 * WebGPU backend. If you need to create WebGPU tensors from another TFJS
 * backend, copy them to the `wasm` accelerator and then convert them with
 * `litertTensor.moveTo('webgpu');`.
 */
export function tfjsToLitert(
    tfjsTensor: tf.Tensor, accelerator: Accelerator = 'webgpu',
    liteRt?: LiteRt): Tensor {
  if (accelerator === 'wasm') {
    return tfjsToLitertCpu(
        tfjsTensor,
        liteRt ? liteRt.getConverterFactory() : getConverterFactory());
  } else if (accelerator === 'webgpu') {
    return tfjsToLitertWebGpu(
        tfjsTensor,
        liteRt ? liteRt.getConverterFactory() : getConverterFactory());
  } else {
    throw new Error('Unsupported accelerator: ' + accelerator);
  }
}

function tfjsToLitertCpu(
    tfjsTensor: tf.Tensor, converterFactory: ConverterFactory): Tensor {
  // TODO: Warn the user that, if they're using the WebGPU backend, they should
  // run `await tensor.data()` on the tensors that aren't already on CPU.
  // Otherwise, this will be expensive.
  // In most cases, the data is already on CPU, but if it comes from another
  // model or a complex preprocessing pipeline, it may be on GPU.

  const dtype = tfjsDtypeToLiteRt(tfjsTensor.dtype);
  const arrayType = DTYPE_TO_ARRAY_TYPE[dtype];

  if (tf.getBackend() === 'webgpu') {
    const backend = tf.backend() as WebGPUBackend;
    if (backend.tensorMap.has(tfjsTensor.dataId)) {
      const tensorData = backend.tensorMap.get(tfjsTensor.dataId)!;
      if (!tensorData.values) {
        throw new TensorConversionError(
            'TFJS tensor data is on WebGPU but not on CPU. You must first ' +
            'call `await tensor.data();` to cache the tensor on CPU. Then, ' +
            'you can use the tensor with the LiteRT.js Wasm backend.');
      }
    }
  }

  const tfjsData = tfjsTensor.dataSync();

  const cpuTensor = new converterFactory.wasm.CpuTensor(
      tfjsData.length * arrayType.BYTES_PER_ELEMENT);
  const cpuTensorUint8Array = cpuTensor.data();

  const cpuTensorArray = new arrayType(
      cpuTensorUint8Array.buffer as ArrayBuffer,
      cpuTensorUint8Array.byteOffset,
      tfjsData.length);
  cpuTensorArray.set(tfjsData);

  return new Tensor({
    type: {
      dtype,
      layout: {
        dimensions: tfjsTensor.shape,
      }
    },
    accelerator: 'wasm',
    reference: cpuTensor,
  });
}

function tfjsToLitertWebGpu(
    tfjsTensor: tf.Tensor, converterFactory: ConverterFactory): Tensor {
  if (tf.getBackend() !== 'webgpu') {
    throw new TensorConversionError(
        'Only the TFJS WebGPU backend is supported when converting to WebGPU LiteRT tensors.');
  }

  const backend = tf.backend() as WebGPUBackend;
  if (!converterFactory.isWebGpuDeviceCompatible(backend.device)) {
    throw new Error(
        'In order to convert from TFJS to LiteRT, both libraries must ' +
        'be initialized to use the same WebGPU device.');
  }

  // dataToGPU is currently bugged and doesn't work if data is on CPU.
  // TODO(msoulanille): Remove this when dataToGPU is fixed.
  if (!backend.tensorMap.get(tfjsTensor.dataId)?.resource) {
    // Then the data is on CPU. Upload it.
    backend.uploadToGPU(tfjsTensor.dataId);
  }

  const gpuData = tfjsTensor.dataToGPU();
  const buffer = gpuData.buffer;
  if (!buffer) {
    throw new TensorConversionError('TFJS tensor did not have a GPU buffer.');
  }

  if (!SUPPORTED_DTYPES.has(tfjsTensor.dtype)) {
    throw new TensorConversionError(
        'Unsupported type: ' + tfjsTensor.dtype + '.');
  }
  const shape4d = getBhwcShapeForTfjs(tfjsTensor);
  const converter = converterFactory.makeConverterFromTfjs(
      tfjsTensor.dtype, shape4d[0], shape4d[1], shape4d[2], shape4d[3]);
  const tensorReference = converter.convertFromTfjs(buffer);
  // Dispose the gpuData tensor to free the buffer it controls. This is okay
  // since `convertFromTfjs` has already submitted the command queue.
  gpuData.tensorRef.dispose();

  return new Tensor({
    type: {
      dtype: tfjsDtypeToLiteRt(tfjsTensor.dtype),
      layout: {
        dimensions: shape4d,
      }
    },
    accelerator: 'webgpu',
    reference: tensorReference,
  });
}

/**
 * Convert a LiteRT tensor to a TFJS tensor.
 *
 * The tensor is copied to the respective TFJS backend.
 * - Wasm CPU tensors will be copied to the TFJS tensor's CPU cache. This is
 *   supported on all TFJS backends.
 * - WebGPU tensors will be copied to the TFJS WebGPU backend. This is only
 *   supported on the TFJS WebGPU backend.
 */
export function litertToTfjs(tensor: Tensor, liteRt?: LiteRt): tf.Tensor {
  if (tensor.accelerator === 'webgpu') {
    return litertToTfjsWebGpu(
        tensor, liteRt ? liteRt.getConverterFactory() : getConverterFactory());
  } else if (tensor.accelerator === 'wasm') {
    return litertToTfjsCpu(
        tensor, liteRt ? liteRt.getConverterFactory() : getConverterFactory());
  } else {
    throw new Error('Unsupported accelerator: ' + tensor.accelerator);
  }
}

function litertToTfjsCpu(
    tensor: Tensor, converterFactory: ConverterFactory): tf.Tensor {
  const cpuTensor = tensor.reference as CpuTensorReference;
  if (!(cpuTensor instanceof converterFactory.wasm.CpuTensor)) {
    throw new TensorConversionError('Tensor reference is not a CpuTensor.');
  }

  const cpuTensorUint8Array = cpuTensor.data();
  const arrayType = DTYPE_TO_ARRAY_TYPE[tensor.type.dtype];

  const cpuTensorArray = new arrayType(
      cpuTensorUint8Array.buffer as ArrayBuffer, cpuTensorUint8Array.byteOffset,
      cpuTensorUint8Array.length / arrayType.BYTES_PER_ELEMENT);

  return tf.tensor(
      // We don't want to share the buffer since TFJS and LiteRT tensors can
      // have different lifetimes.
      cpuTensorArray.slice(),
      /* shape= */[...tensor.type.layout.dimensions],
      /* dtype= */ tensor.type.dtype);
}

function litertToTfjsWebGpu(
    tensor: Tensor, converterFactory: ConverterFactory): tf.Tensor {
  if (tf.getBackend() !== 'webgpu') {
    throw new Error('Only WebGPU backend is supported.');
  }

  const backend = tf.backend() as WebGPUBackend;
  if (!converterFactory.isWebGpuDeviceCompatible(backend.device)) {
    throw new Error(
        'In order to convert from LiteRT to TFJS, both libraries must ' +
        'be initialized to use the same WebGPU device.');
  }

  const converter = converterFactory.makeConverterToTfjs(tensor.reference);
  const buffer = converter.convertToTfjs(tensor.reference);

  // Solution could be: (1) fix TFJS's data() API. The size of buffer (MLD may
  // add paddings for stride-4) could be different from the size of the tensor,
  // but TFJS's data() returns an array of the size of the buffer, rather than
  // the size of the tensor. (2) use TFJS ops to strip out the paddings, but it
  // performs an extra GPU operation and has an extra intermdiate buffer.
  const tfjsTensor = tf.tensor(
      {buffer},
      /* shape= */[...tensor.type.layout.dimensions],
      /* dtype= */ tensor.type.dtype);

  return tfjsTensor;
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
 * Map a Container of one value type to the same Container type of another value
 * type.
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
 * Map a Container of LiteRT tensors to the same Container type but with new
 * values from f.
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
 * If a signature name is not provided, the default signature will be used.
 *
 * When calling a Wasm CPU LiteRT model with WebGPU TFJS tensors, please first
 * call `await tensor.data()` on each tensor to more efficiently copy the
 * tensors to CPU.
 */
// Without signature name
export function runWithTfjsTensors(
    model: CompiledModel|SignatureRunner, input: tf.Tensor|tf.Tensor[],
    liteRt?: LiteRt): tf.Tensor[];
export function runWithTfjsTensors(
    model: CompiledModel|SignatureRunner, input: Record<string, tf.Tensor>,
    liteRt?: LiteRt): Record<string, tf.Tensor>;
// With signature name
export function runWithTfjsTensors(
    model: CompiledModel, signature: string, input: tf.Tensor|tf.Tensor[],
    liteRt?: LiteRt): tf.Tensor[];
export function runWithTfjsTensors(
    model: CompiledModel, signature: string, input: Record<string, tf.Tensor>,
    liteRt?: LiteRt): Record<string, tf.Tensor>;
// The following signatures are needed since TS won't automatically distribute
// a union type across the above function signatures.
// https://github.com/microsoft/TypeScript/issues/14107
export function runWithTfjsTensors(
    model: CompiledModel|SignatureRunner,
    input: tf.Tensor|tf.Tensor[]|Record<string, tf.Tensor>,
    liteRt?: LiteRt): tf.Tensor[]|Record<string, tf.Tensor>;
export function runWithTfjsTensors(
    model: CompiledModel, signature: string,
    input: tf.Tensor|tf.Tensor[]|Record<string, tf.Tensor>,
    liteRt?: LiteRt): tf.Tensor[]|Record<string, tf.Tensor>;
export function runWithTfjsTensors(
    model: CompiledModel|SignatureRunner,
    inputOrSignatureName: string|tf.Tensor|tf.Tensor[]|
    Record<string, tf.Tensor>,
    maybeInputOrLiteRt?: tf.Tensor|tf.Tensor[]|Record<string, tf.Tensor>|LiteRt,
    maybeLiteRt?: LiteRt): tf.Tensor[]|Record<string, tf.Tensor> {
  let signature: string|undefined;
  let tfjsInputs: Record<string, tf.Tensor>|tf.Tensor[]|tf.Tensor;
  let liteRt: LiteRt|undefined;
  if (typeof inputOrSignatureName === 'string') {
    signature = inputOrSignatureName;
    tfjsInputs = maybeInputOrLiteRt as Container<tf.Tensor>;
    liteRt = maybeLiteRt;
  } else {
    tfjsInputs = inputOrSignatureName;
    liteRt = maybeInputOrLiteRt as LiteRt | undefined;
  }

  // Convert TFJS inputs to LiteRT tensors.
  const inputs = mapOnTfjs(tfjsInputs, (tfjsTensor, keyOrIndex) => {
    try {
      return tfjsToLitert(tfjsTensor, model.accelerator, liteRt);
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
    if (model instanceof SignatureRunner) {
      throw new Error(
          'Signature specified but a SignatureRunner was passed instead of a' +
          ' model');
    }
    outputs = model.run(signature, inputs);
  } else {
    outputs = model.run(inputs);
  }

  // Delete all the LiteRT inputs that we created to pass to the model.
  mapOnLiteRt(inputs, tensor => {
    tensor.delete();
  });

  // Convert LiteRT outputs to TFJS tensors.
  return mapOnLiteRt(outputs, (tensor, keyOrIndex) => {
    let tfjsTensor: tf.Tensor;
    try {
      tfjsTensor = litertToTfjs(tensor, liteRt);
    } catch (e) {
      if (e instanceof TensorConversionError && keyOrIndex !== undefined) {
        e.setTensor(keyOrIndex);
      }
      throw e;
    } finally {
      // Two outputs will never share the same buffer since we always make
      // a copy.
      tensor.delete();
    }
    return tfjsTensor;
  });
}
