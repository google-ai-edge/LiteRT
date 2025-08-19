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

import {Accelerator, Dimensions, DType, DTYPE_TO_ARRAY_TYPE, TypedArray, TypedArrayConstructor, typedArrayToDtype} from './constants';
import {getGlobalLiteRt} from './global_litert';
import {CpuTensorReference, OpaqueTensorReference} from './wasm_binding_types';

/**
 * Metadata about a Tensor including its type and layout.
 *
 * This type only includes information about the tensor itself, not the
 * accelerator it is on.
 */
export interface TensorType {
  dtype: DType;
  layout: {dimensions: Dimensions};
}

// Functions for copying tensors between accelerators.
interface CopyFunctionSet {
  moveTo?: (tensor: Tensor) => Tensor | Promise<Tensor>;
  copyTo?: (tensor: Tensor) => Tensor | Promise<Tensor>;
}

type TensorCopyFunctions = Partial<Record<
    Accelerator /* from */,
    Partial<Record<Accelerator /* to */, CopyFunctionSet>>>>;



/**
 * Data for constructing a Tensor directly from an OpaqueTensorReference.
 *
 * Most users should not need to construct a Tensor this way. They should
 * instead construct their tensors from TypedArrays or other data sources.
 */
export interface TensorReferenceData {
  type: TensorType;
  accelerator: Accelerator;
  reference: OpaqueTensorReference;
}

function isTensorReferenceData(data: unknown): data is TensorReferenceData {
  const maybeData = data as TensorReferenceData;
  return (
      maybeData !== undefined && typeof maybeData === 'object' &&
      typeof maybeData.type === 'object' &&
      typeof maybeData.accelerator === 'string' &&
      typeof maybeData.reference === 'object');
}

/**
 * A tensor that is passed to or from a model.
 *
 * Tensors may be on the CPU (`wasm` accelerator), on the GPU (`webgpu`
 * accelerator), or on another LiteRt-specific accelerator.
 * They can be converted between accelerators with `copyTo` and `moveTo`.
 * Tensors on the `wasm` accelerator can be converted to the TypedArray
 * corresponding to their data type.
 */
export class Tensor {
  // This contains properties of TensorWrapper but organized in a more
  // JS-friendly way. Some properties may be missing, such as when the user
  // creates their own Tensor.
  //
  // Additionally, instances of this interface are not associated with a
  // specific TfLite Interpreter.

  static copyFunctions: TensorCopyFunctions = {};
  private readonly tensorReferenceData: TensorReferenceData;
  private deletedInternal = false;

  constructor(data: TensorReferenceData);
  constructor(data: TypedArray, shape?: Dimensions);
  constructor(
      dataOrTypedArray: TensorReferenceData|TypedArray, shape?: Dimensions) {
    if (isTensorReferenceData(dataOrTypedArray)) {
      this.tensorReferenceData = dataOrTypedArray;
    } else {
      this.tensorReferenceData = {
        type: {
          dtype: typedArrayToDtype(dataOrTypedArray),
          layout: {
            dimensions: shape ?? [dataOrTypedArray.length],
          }
        },
        accelerator: 'wasm',
        reference: typedArrayToCpuTensorReference(dataOrTypedArray),
      };
    }
  }

  /**
   * Returns the datatype of the tensor.
   */
  get type(): TensorType {
    return this.tensorReferenceData.type;
  }

  /**
   * Returns the accelerator the tensor is stored on.
   */
  get accelerator(): Accelerator {
    return this.tensorReferenceData.accelerator;
  }

  /**
   * Returns the internal reference to the tensor data.
   *
   * Users should not rely on this call, and should use `toTypedArray` instead
   * if they are trying to view Tensor data.
   */
  get reference(): OpaqueTensorReference {
    return this.tensorReferenceData.reference;
  }

  static fromTypedArray(data: TypedArray, shape?: Dimensions): Tensor {
    return new Tensor(data, shape);
  }

  /**
   * Returns the data of the tensor as a TypedArray.
   *
   * The returned TypedArray is a copy of the data, and this method does not
   * delete the original tensor.
   * @throws An error if the tensor is not on Wasm.
   */
  toTypedArray(): TypedArray {
    if (this.accelerator !== 'wasm') {
      throw new Error(
          'Tensor must be on Wasm to be converted to a TypedArray.');
    }

    const typedArrayConstructor = DTYPE_TO_ARRAY_TYPE[this.type.dtype];
    const cpuTensorReference = this.reference as CpuTensorReference;
    const data = cpuTensorReference.data();
    return new typedArrayConstructor(
               // Cast is needed to avoid 'SharedArrayBuffer' in the type.
               data.buffer as ArrayBuffer, data.byteOffset,
               data.length / typedArrayConstructor.BYTES_PER_ELEMENT)
        .slice();
  }

  /**
   * Copies the tensor to the given accelerator.
   *
   * @param accelerator The accelerator to copy to.
   * @return A promise that resolves to the copied tensor.
   */
  async copyTo(accelerator: Accelerator): Promise<Tensor> {
    const copyFunctions = Tensor.copyFunctions[this.accelerator];
    if (!copyFunctions) {
      throw new Error(
          `Accelerator ${this.accelerator} does not support copying`);
    }
    const copyFunctionSet = copyFunctions[accelerator];
    if (!copyFunctionSet || !copyFunctionSet.copyTo) {
      const supportedCopyDestinations =
          Object.entries(copyFunctions)
              .filter(([key, value]) => value.copyTo)
              .map(([key, value]) => key);
      throw new Error(`Accelerator ${
          this.accelerator} does not support copying to ${
          accelerator}. It supports copying to the following accelerators: [${
          supportedCopyDestinations.join(', ')}].`);
    }
    return copyFunctionSet.copyTo(this);
  }

  /**
   * Moves the tensor to the given accelerator, deleting the original.
   *
   * @param accelerator The accelerator to move to.
   * @return A promise that resolves to the moved tensor.
   */
  async moveTo(accelerator: Accelerator): Promise<Tensor> {
    const copyFunctions = Tensor.copyFunctions[this.accelerator];
    if (!copyFunctions) {
      throw new Error(
          `Accelerator ${this.accelerator} does not support moving`);
    }
    const copyFunctionSet = copyFunctions[accelerator];
    if (!copyFunctionSet || !copyFunctionSet.moveTo) {
      const supportedMoveDestinations =
          Object.entries(copyFunctions)
              .filter(([key, value]) => value.moveTo)
              .map(([key, value]) => key);
      throw new Error(`Accelerator ${
          this.accelerator} does not support moving to ${
          accelerator}. It supports moving to the following accelerators: [${
          supportedMoveDestinations.join(', ')}].`);
    }
    return copyFunctionSet.moveTo(this);
  }

  get deleted(): boolean {
    return this.deletedInternal;
  }

  delete() {
    this.tensorReferenceData.reference.delete?.();
    this.deletedInternal = true;
  }
}

/**
 * An error thrown when a tensor of the wrong type is passed to a model.
 */
export class TensorTypeError extends Error {
  constructor(
      name: string,
      index: number,
      expected: DType,
      actual: DType,
  ) {
    super(`Input tensor for ${name} at position ${index} has type ${
        actual}, but signature expects ${expected}.`);
  }
}

/**
 * An error thrown when a tensor of the wrong shape is passed to a model.
 */
export class TensorShapeError extends Error {
  constructor(
      name: string,
      expected: {join(joinWith: string): string;},
      actual: {join(joinWith: string): string;},
  ) {
    const expectedShapeString = `[${expected.join(', ')}]`;
    const actualShapeString = `[${actual.join(', ')}]`;
    super(
        `Input tensor for ${name} has shape ${actualShapeString}, but ` +
        `signature expects ${expectedShapeString}.`);
  }
}

function typedArrayToCpuTensorReference(data: TypedArray):
    OpaqueTensorReference {
  const globalLiteRt = getGlobalLiteRt();

  const arrayType = data.constructor as TypedArrayConstructor;
  const cpuTensor = new globalLiteRt.liteRtWasm.CpuTensor(
      data.length * arrayType.BYTES_PER_ELEMENT);
  const cpuTensorUint8Array = cpuTensor.data();

  const cpuTensorArray = new arrayType(
      // Cast is needed to avoid 'SharedArrayBuffer' in the type.
      cpuTensorUint8Array.buffer as ArrayBuffer, cpuTensorUint8Array.byteOffset,
      data.length);
  cpuTensorArray.set(data);

  return cpuTensor;
}
