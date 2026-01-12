/**
 * g3-format-prettier
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

import {
  Accelerator,
  AcceleratorDefaultTensorBufferType,
  TensorBufferTypeToAccelerator,
} from './accelerator_types';
import {
  DType,
  getDataType,
  TypedArray,
  TypedArrayConstructor,
} from './datatypes';
import {Environment, WithEnvironment} from './environment';
import {getGlobalLiteRt} from './global_litert';
import {
  Deletable,
  ElementTypeName,
  LiteRtTensorBuffer,
  TensorBufferType,
  TensorBufferTypeName,
} from './wasm_binding_types';
import {emscriptenVectorToArray, fillEmscriptenVector} from './wasm_utils';

/**
 * Options for copying a Tensor to another accelerator or buffer type.
 */
export interface CopyOptions {
  environment?: Environment;
}

/**
 * A function that copies a Tensor to another accelerator or buffer type.
 *
 * @param tensor The tensor to copy.
 * @param options Options for the copy operation.
 * @return A promise that resolves to the copied tensor.
 */
export type TensorCopyFn = (
  tensor: Tensor,
  options?: CopyOptions,
) => Tensor | Promise<Tensor>;

/**
 * A set of copy functions for copying tensors between accelerators.
 */
interface CopyFunctionSet {
  moveTo?: TensorCopyFn;
  copyTo?: TensorCopyFn;
}

/**
 * A map of copy functions for copying tensors between accelerators.
 *
 * The first key is the source buffer type. The second key is the destination
 * buffer type.
 */
type TensorCopyFunctions = Map<
  TensorBufferType /* from */,
  Map<TensorBufferType /* to */, CopyFunctionSet>
>;

/**
 * The dimensions or shape of a Tensor.
 */
export type Dimensions = Int32Array | number[];

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

interface TensorConstructorArgs {
  typedArray?: TypedArray;
  gpuBuffer?: GPUBuffer;
  liteRtTensorBuffer?: LiteRtTensorBuffer;
  shape?: Dimensions;
  dataType?: DType;
  environment?: Environment;
  onDelete?: () => void;
}

type Arg = TensorConstructorArgs[keyof TensorConstructorArgs];

function parseData(remainingArgs: Arg[]): {
  typedArray?: TypedArray;
  gpuBuffer?: GPUBuffer;
  liteRtTensorBuffer?: LiteRtTensorBuffer;
} {
  const data = remainingArgs.shift();
  const liteRtWasm = getGlobalLiteRt().liteRtWasm;
  if (data instanceof liteRtWasm.LiteRtTensorBuffer) {
    return {liteRtTensorBuffer: data};
  } else if (ArrayBuffer.isView(data)) {
    return {typedArray: data};
  } else if (data instanceof GPUBuffer) {
    return {gpuBuffer: data};
  } else {
    throw new Error(
      `Unknown type (${
        data?.constructor.name ?? data
      }) provided to create a Tensor`,
    );
  }
}

function parseShape(remainingArgs: Arg[]): {shape?: Dimensions} {
  if (
    Array.isArray(remainingArgs[0]) ||
    remainingArgs[0] instanceof Int32Array
  ) {
    return {shape: remainingArgs.shift() as Dimensions};
  } else {
    return {}; // Shape will just be flat.
  }
}

function shiftUntilDefined(remainingArgs: Arg[]) {
  while (remainingArgs.length > 0 && remainingArgs[0] === undefined) {
    remainingArgs.shift();
  }
}

function parseDataType(remainingArgs: Arg[]): {dataType?: DType} {
  shiftUntilDefined(remainingArgs);
  if (typeof remainingArgs[0] === 'string') {
    // Perhaps this should also support passing in the enum value instead of
    // just the string?
    const dtype = remainingArgs.shift() as DType;
    // Call getDataType to ensure it's actually a valid DType string.
    return {dataType: getDataType(dtype).dtype};
  } else {
    return {};
  }
}

function parseEnvironment(remainingArgs: Arg[]): {environment?: Environment} {
  shiftUntilDefined(remainingArgs);
  if (remainingArgs[0] instanceof Environment) {
    return {environment: remainingArgs.shift() as Environment};
  } else {
    return {};
  }
}

function parseOnDelete(remainingArgs: Arg[]): {onDelete?: () => void} {
  shiftUntilDefined(remainingArgs);
  if (remainingArgs[0] instanceof Function) {
    return {onDelete: remainingArgs.shift() as () => void};
  } else {
    return {};
  }
}

function parseArgs(args: Arg[]): TensorConstructorArgs {
  return {
    ...parseData(args),
    ...parseShape(args),
    ...parseDataType(args),
    ...parseEnvironment(args),
    ...parseOnDelete(args),
  };
}

/**
 * A tensor that is passed to or from a model.
 */
export class Tensor implements Deletable, WithEnvironment {
  readonly liteRtTensorBuffer!: LiteRtTensorBuffer;
  readonly type: TensorType;
  readonly environment: Environment;
  private deletedInternal = false;
  private onDelete: (() => void) | undefined;

  static copyFunctions: TensorCopyFunctions = new Map();

  constructor(
    data: TypedArray,
    shape?: Dimensions,
    environment?: Environment,
    onDelete?: () => void,
  );
  constructor(
    liteRtTensorBuffer: LiteRtTensorBuffer,
    environment?: Environment,
    onDelete?: () => void,
  );
  constructor(
    gpuBuffer: GPUBuffer,
    shape: Dimensions,
    dataType: DType,
    environment?: Environment,
    onDelete?: () => void,
  );
  constructor(
    a: TypedArray | LiteRtTensorBuffer | GPUBuffer,
    b?: Dimensions | Environment,
    c?: DType | Environment | (() => void),
    d?: Environment | (() => void),
    e?: () => void,
  ) {
    const {
      typedArray,
      gpuBuffer,
      liteRtTensorBuffer,
      shape,
      dataType,
      environment,
      onDelete,
    } = parseArgs([a, b, c, d, e]);

    this.onDelete = onDelete; // Typically used for cleaning up WebGPU buffers.

    // Note: We can't easily verify that the GPUBuffer and the environment
    // share the same device. There's no `.device` property on the GPUBuffer.
    this.environment = environment ?? getGlobalLiteRt().getDefaultEnvironment();

    if (liteRtTensorBuffer) {
      if (shape) {
        throw new Error(
          'A LiteRtTensorBuffer cannot be provided with a shape.',
        );
      }
      if (dataType) {
        throw new Error(
          'A LiteRtTensorBuffer cannot be provided with a data type.',
        );
      }
      this.liteRtTensorBuffer = liteRtTensorBuffer;
    } else if (gpuBuffer) {
      if (!shape) {
        throw new Error('A GPUBuffer must be provided with a shape.');
      }
      if (!dataType) {
        throw new Error('A GPUBuffer must be provided with a data type.');
      }
      const [liteRtTensorBuffer, webGpuBufferPtr] =
        webGpuBufferToLiteRtTensorBuffer(
          gpuBuffer,
          shape,
          dataType,
          this.environment,
        );
      this.liteRtTensorBuffer = liteRtTensorBuffer;

      const onDelete = this.onDelete;
      this.onDelete = () => {
        const liteRtWasm = getGlobalLiteRt().liteRtWasm;
        liteRtWasm.wgpuBufferRelease(webGpuBufferPtr);
        onDelete?.();
      };
    } else if (typedArray) {
      this.liteRtTensorBuffer = typedArrayToLiteRtTensorBuffer(
        typedArray,
        shape,
        environment,
      );
    } else {
      throw new Error('No data provided to create a Tensor.');
    }

    this.type = liteRtTensorBufferToTensorType(this.liteRtTensorBuffer);
  }

  static fromTypedArray(
    data: TypedArray,
    shape?: Dimensions,
    environment?: Environment,
  ): Tensor {
    return new Tensor(data, shape, environment);
  }

  private ensureNotDeleted() {
    if (this.deleted) {
      throw new Error('Tensor is deleted and cannot be used.');
    }
  }

  async data(): Promise<TypedArray> {
    this.ensureNotDeleted();
    if (
      this.liteRtTensorBuffer.bufferType().value ===
      TensorBufferType.HOST_MEMORY
    ) {
      return this.toTypedArray();
    }
    const copy = await this.copyTo('wasm');
    const data = await copy.data();
    copy.delete();
    return data;
  }

  toTypedArray(): TypedArray {
    this.ensureNotDeleted();
    const liteRtWasm = getGlobalLiteRt().liteRtWasm;
    if (this.liteRtTensorBuffer.isWebGpuMemory()) {
      throw new Error(
        'Cannot convert a Tensor with WebGPU memory to a TypedArray.',
      );
    }
    if (
      this.liteRtTensorBuffer.bufferType().value !==
      liteRtWasm.LiteRtTensorBufferType.HOST_MEMORY.value
    ) {
      throw new Error(
        'Cannot convert a Tensor with non-host memory to a TypedArray.',
      );
    }
    if (
      this.liteRtTensorBuffer.size() !== this.liteRtTensorBuffer.packedSize() ||
      this.liteRtTensorBuffer.offset() !== 0
    ) {
      throw new Error('Tensors with strides or padding are not yet supported.');
    }

    const rankedTensorType = this.liteRtTensorBuffer.tensorType();
    const elementType = rankedTensorType.elementType();
    const byteWidth = liteRtWasm.liteRtGetByteWidth(elementType);
    rankedTensorType.delete();

    const typedArrayConstructor = getDataType(
      elementType.value,
    ).typedArrayConstructor;
    if (typedArrayConstructor.BYTES_PER_ELEMENT !== byteWidth) {
      throw new Error(
        `Byte width ${byteWidth} of the tensor's element type ${
          ElementTypeName[elementType.value]
        } ` +
          `does not match the expected byte width ${typedArrayConstructor.BYTES_PER_ELEMENT} of the ${typedArrayConstructor.name}.`,
      );
    }

    const dataPtr = this.liteRtTensorBuffer.lock(
      getGlobalLiteRt().liteRtWasm.LiteRtTensorBufferLockMode.READ,
    );
    try {
      const uint8Array = liteRtWasm.HEAPU8.slice(
        dataPtr,
        dataPtr + this.liteRtTensorBuffer.packedSize(),
      );

      const typedArray = new typedArrayConstructor(
        uint8Array.buffer,
        uint8Array.byteOffset,
        uint8Array.byteLength / byteWidth,
      );

      return typedArray;
    } finally {
      this.liteRtTensorBuffer.unlock();
    }
  }

  getBufferType(): TensorBufferType {
    this.ensureNotDeleted();
    return this.liteRtTensorBuffer.bufferType().value;
  }

  /**
   * Returns the underlying GPUBuffer of the Tensor.
   *
   * Note that the lifetime of the returned GPUBuffer is dependant upon how the
   * Tensor was created. If the Tensor was constructed from a GPUBuffer, then
   * the GPUBuffer will NOT be released when the Tensor is deleted. If the
   * Tensor was copied/moved to GPU from host memory, then the GPU buffer will
   * be released when the Tensor is deleted.
   *
   * The GPU buffer may be larger than the actual data in the tensor.
   *
   * @return The GPUBuffer containing the Tensor's data.
   */
  toGpuBuffer(): GPUBuffer {
    this.ensureNotDeleted();
    const liteRtWasm = getGlobalLiteRt().liteRtWasm;
    if (!this.liteRtTensorBuffer.isWebGpuMemory()) {
      throw new Error(
        'Cannot convert a Tensor with non-WebGPU memory to a GPUBuffer.',
      );
    }
    const bufferTypeValue = this.liteRtTensorBuffer.bufferType().value;
    if (
      bufferTypeValue !==
        liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER.value &&
      bufferTypeValue !==
        liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER_FP16.value &&
      bufferTypeValue !==
        liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER_PACKED.value
    ) {
      throw new Error(
        'Cannot convert a Tensor with host memory to a GPUBuffer.',
      );
    }
    // TODO: markoristic - Support tensors with strides or padding.
    if (
      this.liteRtTensorBuffer.size() !== this.liteRtTensorBuffer.packedSize() ||
      this.liteRtTensorBuffer.offset() !== 0
    ) {
      throw new Error('Tensors with strides or padding are not yet supported.');
    }

    const gpuBufferId = this.liteRtTensorBuffer.getWebGpuBuffer();
    return liteRtWasm.WebGPU.getJsObject(gpuBufferId);
  }

  private getCopyFunctionSet(
    destination: Accelerator | TensorBufferType,
  ): [CopyFunctionSet, TensorBufferType] {
    this.ensureNotDeleted();
    const sourceBufferType = this.getBufferType();
    const copyFunctions = Tensor.copyFunctions.get(sourceBufferType);
    if (!copyFunctions) {
      throw new Error(
        `TensorBufferType ${
          TensorBufferTypeName[sourceBufferType] ?? sourceBufferType
        } does not support copying or moving`,
      );
    }

    const destinationBufferType =
      typeof destination === 'string'
        ? AcceleratorDefaultTensorBufferType[destination]
        : destination;

    if (destinationBufferType == null) {
      throw new Error(
        `Unknown destination '${destination}' for copying or moving.`,
      );
    }

    const copyFunctionSet = copyFunctions.get(destinationBufferType);
    if (!copyFunctionSet) {
      const supportedDestinations = [...copyFunctions].map(
        ([key]) => TensorBufferTypeName[key] ?? key,
      );
      throw new Error(
        `TensorBufferType ${
          TensorBufferTypeName[sourceBufferType]
        } does not support copying or moving to ${
          TensorBufferTypeName[destinationBufferType]
        }. It supports the following TensorBufferTypes: [${supportedDestinations.join(
          ', ',
        )}].`,
      );
    }
    return [copyFunctionSet, destinationBufferType];
  }

  /**
   * Copies the tensor to the given accelerator.
   *
   * @param destination The accelerator or buffer type to copy to.
   * @return A promise that resolves to the copied tensor.
   */
  async copyTo(
    destination: Accelerator | TensorBufferType,
    options?: CopyOptions,
  ): Promise<Tensor> {
    const [copyFunctionSet, destinationBufferType] =
      this.getCopyFunctionSet(destination);

    if (!copyFunctionSet.copyTo) {
      throw new Error(
        `Copying to ${TensorBufferTypeName[destinationBufferType]} is not supported by this tensor.`,
      );
    }
    return copyFunctionSet.copyTo(this, options);
  }

  /**
   * Moves the tensor to the given accelerator.
   *
   * @param destination The accelerator or buffer type to move to.
   * @return A promise that resolves to the moved tensor.
   */
  async moveTo(
    destination: Accelerator | TensorBufferType,
    options?: CopyOptions,
  ): Promise<Tensor> {
    const [copyFunctionSet, destinationBufferType] =
      this.getCopyFunctionSet(destination);

    if (!copyFunctionSet.moveTo) {
      throw new Error(
        `Moving to ${TensorBufferTypeName[destinationBufferType]} is not supported by this tensor.`,
      );
    }
    return copyFunctionSet.moveTo(this, options);
  }

  get bufferType(): TensorBufferType {
    return this.liteRtTensorBuffer.bufferType().value;
  }

  get accelerator(): Accelerator {
    const accelerator = TensorBufferTypeToAccelerator[this.bufferType];
    if (accelerator === undefined) {
      throw new Error(
        `TensorBufferType ${
          TensorBufferTypeName[this.bufferType]
        } has an unknown accelerator type.`,
      );
    }
    return accelerator;
  }

  get deleted(): boolean {
    return this.deletedInternal;
  }

  delete() {
    if (this.deletedInternal) {
      return;
    }
    this.deletedInternal = true;
    this.liteRtTensorBuffer.delete();
    this.onDelete?.();
  }
}

/**
 * Get the TensorType of a LiteRtTensorBuffer.
 */
function liteRtTensorBufferToTensorType(
  liteRtTensorBuffer: LiteRtTensorBuffer,
): TensorType {
  const liteRtRankedTensorType = liteRtTensorBuffer.tensorType();
  const elementType = liteRtRankedTensorType.elementType();
  const liteRtLayout = liteRtRankedTensorType.layout();
  const dimensions = liteRtLayout.dimensions();

  // Delete temporary emscripten objects.
  liteRtLayout.delete();
  liteRtRankedTensorType.delete();
  // `elementType` does not need to be deleted because it is an enum value.
  // `dimensions` is deleted by `emscriptenVectorToArray`.

  return {
    dtype: getDataType(elementType.value).dtype,
    layout: {dimensions: emscriptenVectorToArray(dimensions)},
  };
}

/**
 * Creates a LiteRtTensorBuffer from a GPUBuffer.
 *
 * Returns the LiteRtTensorBuffer and the WebGPU buffer Wasm heap pointer. In
 * the emscripten implementation, the pointer actually refers to the index of
 * the buffer in the WebGPU module's Internals.jsObjects array, and must be
 * released with `wgpuBufferRelease`.
 */
function webGpuBufferToLiteRtTensorBuffer(
  gpuBuffer: GPUBuffer,
  shape: Dimensions,
  dtype: DType,
  environment: Environment,
): [LiteRtTensorBuffer, number] {
  const globalLiteRt = getGlobalLiteRt();
  const liteRtWasm = globalLiteRt.liteRtWasm;

  // Create a LiteRtLayout from the shape.
  const dimensionsVector = new liteRtWasm.VectorInt32();
  fillEmscriptenVector(shape, dimensionsVector);
  const layout = liteRtWasm.LiteRtLayout.create(dimensionsVector);
  dimensionsVector.delete();

  const rankedTensorType = liteRtWasm.LiteRtRankedTensorType.create(
    {value: getDataType(dtype).elementType},
    layout,
  );
  layout.delete();

  const importedGpuBufferPtr = liteRtWasm.WebGPU.importJsBuffer(gpuBuffer);

  const liteRtTensorBuffer =
    liteRtWasm.LiteRtTensorBuffer.createFromWebGpuBuffer(
      environment.liteRtEnvironment,
      rankedTensorType,
      liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER_PACKED,
      importedGpuBufferPtr,
      gpuBuffer.size,
    );
  rankedTensorType.delete();

  return [liteRtTensorBuffer, importedGpuBufferPtr];
}

/**
 * Creates a LiteRtTensorBuffer from a TypedArray and optional shape.
 */
function typedArrayToLiteRtTensorBuffer(
  data: TypedArray,
  shape?: Dimensions,
  environment?: Environment,
): LiteRtTensorBuffer {
  const globalLiteRt = getGlobalLiteRt();
  const liteRtWasm = globalLiteRt.liteRtWasm;
  environment = environment ?? globalLiteRt.getDefaultEnvironment();

  const elementType = getDataType(data).elementType;

  // Create a LiteRtLayout from the shape.
  const dimensionsVector = new liteRtWasm.VectorInt32();
  fillEmscriptenVector(shape ?? [data.length], dimensionsVector);
  const layout = liteRtWasm.LiteRtLayout.create(dimensionsVector);
  dimensionsVector.delete();

  // Check that the number of elements in the layout matches the number of
  // elements in the TypedArray.
  const expectedNumElements = layout.numElements();
  if (data.length !== expectedNumElements) {
    layout.delete();
    throw new Error(
      `Number of elements ${data.length} of the provided TypedArray ` +
        `does not match the expected number of elements ${expectedNumElements}.`,
    );
  }

  // Create a LiteRtRankedTensorType from the element type and layout.
  const rankedTensorType = liteRtWasm.LiteRtRankedTensorType.create(
    {value: elementType},
    layout,
  );
  layout.delete(); // Delete our copy of the layout.

  // Check that the byte length of the TypedArray matches the expected byte
  // length of the LiteRtRankedTensorType.
  const arrayType = data.constructor as TypedArrayConstructor;
  const bufferSize = arrayType.BYTES_PER_ELEMENT * data.length;
  const expectedBufferSize = rankedTensorType.bytes();
  if (bufferSize !== expectedBufferSize) {
    rankedTensorType.delete();
    throw new Error(
      `Byte length ${bufferSize} of the provided TypedArray ` +
        `does not match the expected buffer size ${expectedBufferSize}.`,
    );
  }

  // Create the LiteRtTensorBuffer.
  const liteRtTensorBuffer = liteRtWasm.LiteRtTensorBuffer.createManaged(
    environment.liteRtEnvironment,
    liteRtWasm.LiteRtTensorBufferType.HOST_MEMORY,
    rankedTensorType,
    bufferSize,
  );
  rankedTensorType.delete(); // Delete our copy.

  // Write the data to the LiteRtTensorBuffer.
  const dataPtr = liteRtTensorBuffer.lock(
    liteRtWasm.LiteRtTensorBufferLockMode.WRITE,
  );
  try {
    const uint8Data = new Uint8Array(
      data.buffer,
      data.byteOffset,
      data.byteLength,
    );
    liteRtWasm.HEAPU8.set(uint8Data, dataPtr);
  } finally {
    liteRtTensorBuffer.unlock();
  }

  return liteRtTensorBuffer;
}
