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
  LiteRtTensorHandle,
  LiteRtWasm,
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
  liteRtTensorHandle?: LiteRtTensorHandle;
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
  liteRtTensorHandle?: LiteRtTensorHandle;
} {
  const data = remainingArgs.shift();
  const liteRtWasm = getGlobalLiteRt().liteRtWasm;
  if (data instanceof liteRtWasm.LiteRtTensorHandle) {
    return {liteRtTensorHandle: data};
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
  readonly liteRtTensorHandle!: LiteRtTensorHandle;
  /** @deprecated Use liteRtTensorHandle instead. */
  get liteRtTensorBuffer(): LiteRtTensorHandle {
    console.warn(
      'liteRtTensorBuffer is deprecated. Use liteRtTensorHandle instead.',
    );
    return this.liteRtTensorHandle;
  }
  readonly type: TensorType;
  get shape(): Dimensions {
    return this.type.layout.dimensions;
  }
  get dtype(): DType {
    return this.type.dtype;
  }
  readonly environment: Environment;
  private deletedInternal = false;
  private onDelete: (() => void) | undefined;

  static copyFunctions: TensorCopyFunctions = new Map();

  constructor(
    data: TypedArray,
    shape?: Dimensions,
    dataType?: DType,
    environment?: Environment,
    onDelete?: () => void,
  );
  constructor(
    data: TypedArray,
    shape?: Dimensions,
    environment?: Environment,
    onDelete?: () => void,
  );
  constructor(
    liteRtTensorHandle: LiteRtTensorHandle,
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
    a: TypedArray | LiteRtTensorHandle | LiteRtTensorBuffer | GPUBuffer,
    b?: Dimensions | Environment,
    c?: DType | Environment | (() => void),
    d?: Environment | (() => void),
    e?: () => void,
  ) {
    const {
      typedArray,
      gpuBuffer,
      liteRtTensorHandle,
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

    const handle = liteRtTensorHandle ?? liteRtTensorBuffer;
    if (handle) {
      if (shape) {
        throw new Error(
          'A LiteRtTensorHandle cannot be provided with a shape.',
        );
      }
      if (dataType) {
        throw new Error(
          'A LiteRtTensorHandle cannot be provided with a data type.',
        );
      }
      this.liteRtTensorHandle = handle;
    } else if (gpuBuffer) {
      if (!shape) {
        throw new Error('A GPUBuffer must be provided with a shape.');
      }
      if (!dataType) {
        throw new Error('A GPUBuffer must be provided with a data type.');
      }
      const [liteRtTensorHandle, webGpuBufferPtr] =
        webGpuBufferToLiteRtTensorHandle(
          gpuBuffer,
          shape,
          dataType,
          this.environment,
        );
      this.liteRtTensorHandle = liteRtTensorHandle;

      const onDelete = this.onDelete;
      this.onDelete = () => {
        const liteRtWasm = getGlobalLiteRt().liteRtWasm;
        liteRtWasm.wgpuBufferRelease(webGpuBufferPtr);
        onDelete?.();
      };
    } else if (typedArray) {
      this.liteRtTensorHandle = typedArrayToLiteRtTensorHandle(
        typedArray,
        shape,
        dataType,
        environment,
      );
    } else {
      throw new Error('No data provided to create a Tensor.');
    }

    this.type = liteRtTensorHandleToTensorType(this.liteRtTensorHandle);
  }

  static fromTypedArray(
    data: TypedArray,
    shape?: Dimensions,
    environment?: Environment,
  ): Tensor {
    return new Tensor(data, shape, environment);
  }

  static placeholderCounter = 0;

  /**
   * Creates a symbolic placeholder Tensor for JIT graph compilation.
   */
  static createPlaceholder(options: {
    shape?: Dimensions;
    dataType?: DType;
    environment?: Environment;
    name?: string;
  } = {}): Tensor {
    const globalLiteRt = getGlobalLiteRt();
    const liteRtWasm = globalLiteRt.liteRtWasm;
    const shape = options.shape ?? [1];
    const dtype = options.dataType ?? 'float32';
    const name = options.name || `placeholder_${Tensor.placeholderCounter++}`;

    const dimensionsVector = new liteRtWasm.VectorInt32();
    fillEmscriptenVector(shape, dimensionsVector);
    const layout = liteRtWasm.LiteRtLayout.create(dimensionsVector);
    dimensionsVector.delete();

    const rankedTensorType = liteRtWasm.LiteRtRankedTensorType.create(
      {value: getDataType(dtype).elementType},
      layout,
    );
    layout.delete();

    const handle = liteRtWasm.LiteRtTensorHandle.createPlaceholder(
      rankedTensorType,
      name,
    );
    rankedTensorType.delete();

    return new Tensor(handle, options.environment);
  }

  ensureNotDeleted() {
    if (this.deleted) {
      throw new Error('Tensor is deleted and cannot be used.');
    }
  }

  add(other: Tensor): Tensor {
    return add(this, other);
  }

  mul(other: Tensor): Tensor {
    return mul(this, other);
  }

  sub(other: Tensor): Tensor {
    return sub(this, other);
  }

  div(other: Tensor): Tensor {
    return div(this, other);
  }

  pow(other: Tensor): Tensor {
    return pow(this, other);
  }

  minimum(other: Tensor): Tensor {
    return minimum(this, other);
  }

  maximum(other: Tensor): Tensor {
    return maximum(this, other);
  }

  floorDiv(other: Tensor): Tensor {
    return floorDiv(this, other);
  }

  floorMod(other: Tensor): Tensor {
    return floorMod(this, other);
  }

  equal(other: Tensor): Tensor {
    return equal(this, other);
  }

  notEqual(other: Tensor): Tensor {
    return notEqual(this, other);
  }

  less(other: Tensor): Tensor {
    return less(this, other);
  }

  greater(other: Tensor): Tensor {
    return greater(this, other);
  }

  lessEqual(other: Tensor): Tensor {
    return lessEqual(this, other);
  }

  greaterEqual(other: Tensor): Tensor {
    return greaterEqual(this, other);
  }

  logicalAnd(other: Tensor): Tensor {
    return logicalAnd(this, other);
  }

  logicalOr(other: Tensor): Tensor {
    return logicalOr(this, other);
  }

  logicalNot(): Tensor {
    return logicalNot(this);
  }

  select(trueVal: Tensor, falseVal: Tensor): Tensor {
    return select(this, trueVal, falseVal);
  }

  relu(): Tensor {
    return relu(this);
  }

  toString(): string {
    this.ensureNotDeleted();
    return `${this.type.dtype}[${Array.from(this.type.layout.dimensions).join(
      ', ',
    )}]`;
  }

  batchMatMul(other: Tensor, adjX?: boolean, adjY?: boolean): Tensor {
    return batchMatMul(this, other, adjX, adjY);
  }

  fullyConnected(weights: Tensor, bias?: Tensor): Tensor {
    return fullyConnected(this, weights, bias);
  }

  softmax(beta?: number): Tensor {
    return softmax(this, beta);
  }

  logistic(): Tensor {
    return logistic(this);
  }

  tanh(): Tensor {
    return tanh(this);
  }

  gelu(approximate?: boolean): Tensor {
    return gelu(this, approximate);
  }

  relu6(): Tensor {
    return relu6(this);
  }

  leakyRelu(alpha?: number): Tensor {
    return leakyRelu(this, alpha);
  }

  elu(): Tensor {
    return elu(this);
  }

  hardSwish(): Tensor {
    return hardSwish(this);
  }

  logSoftmax(): Tensor {
    return logSoftmax(this);
  }

  conv2d(filter: Tensor, options?: Conv2dOptions): Tensor {
    return conv2d(this, filter, options);
  }

  depthwiseConv2d(filter: Tensor, options?: DepthwiseConv2dOptions): Tensor {
    return depthwiseConv2d(this, filter, options);
  }

  reshape(shape: number[]): Tensor {
    return reshape(this, shape);
  }

  transpose(perm: number[]): Tensor {
    return transpose(this, perm);
  }

  concat(others: Tensor | Tensor[], axis: number = 0): Tensor {
    const list = Array.isArray(others) ? [this, ...others] : [this, others];
    return concat(list, axis);
  }

  slice(begin: number[], size: number[]): Tensor {
    return slice(this, begin, size);
  }

  pad(paddings: Tensor | number[][]): Tensor {
    return pad(this, paddings);
  }

  expandDims(axis: number): Tensor {
    return expandDims(this, axis);
  }

  squeeze(squeezeDims?: number[]): Tensor {
    return squeeze(this, squeezeDims);
  }

  tile(multiples: number[]): Tensor {
    return tile(this, multiples);
  }

  unpack(num: number, axis: number = 0): Tensor[] {
    return unpack(this, num, axis);
  }

  split(numSplits: number, axis: number = 0): Tensor[] {
    return split(this, numSplits, axis);
  }

  mean(axes?: number | number[], keepDims: boolean = false): Tensor {
    return mean(this, axes, keepDims);
  }

  sum(axes?: number | number[], keepDims: boolean = false): Tensor {
    return sum(this, axes, keepDims);
  }

  reduceMax(axes?: number | number[], keepDims: boolean = false): Tensor {
    return reduceMax(this, axes, keepDims);
  }

  argMax(axis: number = 0, outputType: 'int32' | 'int64' = 'int32'): Tensor {
    return argMax(this, axis, outputType);
  }

  resizeBilinear(
    size: [number, number] | number[],
    alignCornersOrOptions?: boolean | ResizeOptions,
    halfPixelCenters?: boolean,
  ): Tensor {
    return resizeBilinear(this, size, alignCornersOrOptions, halfPixelCenters);
  }

  resizeNearestNeighbor(
    size: [number, number] | number[],
    alignCornersOrOptions?: boolean | ResizeOptions,
    halfPixelCenters?: boolean,
  ): Tensor {
    return resizeNearestNeighbor(
      this,
      size,
      alignCornersOrOptions,
      halfPixelCenters,
    );
  }

  maxPool2d(
    filterSizeOrOptions?: number | [number, number] | Pool2dOptions,
    strides?: number | [number, number],
    padding?: Padding,
  ): Tensor {
    return maxPool2d(this, filterSizeOrOptions, strides, padding);
  }

  avgPool2d(
    filterSizeOrOptions?: number | [number, number] | Pool2dOptions,
    strides?: number | [number, number],
    padding?: Padding,
  ): Tensor {
    return avgPool2d(this, filterSizeOrOptions, strides, padding);
  }

  transposeConv2d(
    filter: Tensor,
    outputShape: number[],
    options: TransposeConv2dOptions = {},
  ): Tensor {
    return transposeConv2d(this, filter, outputShape, options);
  }

  abs(): Tensor {
    return abs(this);
  }

  neg(): Tensor {
    return neg(this);
  }

  sqrt(): Tensor {
    return sqrt(this);
  }

  rsqrt(): Tensor {
    return rsqrt(this);
  }

  exp(): Tensor {
    return exp(this);
  }

  log(): Tensor {
    return log(this);
  }

  sin(): Tensor {
    return sin(this);
  }

  cos(): Tensor {
    return cos(this);
  }

  ceil(): Tensor {
    return ceil(this);
  }

  floor(): Tensor {
    return floor(this);
  }

  round(): Tensor {
    return round(this);
  }

  async data(): Promise<TypedArray> {
    this.ensureNotDeleted();
    if (
      this.liteRtTensorHandle.bufferType().value ===
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
    if (this.liteRtTensorHandle.isWebGpuMemory()) {
      throw new Error(
        'Cannot convert a Tensor with WebGPU memory to a TypedArray.',
      );
    }
    if (
      this.liteRtTensorHandle.bufferType().value !==
      liteRtWasm.LiteRtTensorBufferType.HOST_MEMORY.value
    ) {
      throw new Error(
        'Cannot convert a Tensor with non-host memory to a TypedArray.',
      );
    }
    if (
      this.liteRtTensorHandle.size() !== this.liteRtTensorHandle.packedSize() ||
      this.liteRtTensorHandle.offset() !== 0
    ) {
      throw new Error('Tensors with strides or padding are not yet supported.');
    }

    const rankedTensorType = this.liteRtTensorHandle.tensorType();
    const elementType = rankedTensorType.elementType();
    const byteWidth = liteRtWasm.liteRtGetByteWidth(elementType);
    rankedTensorType.delete();

    const typedArrayConstructor = getDataType(
      elementType.value,
    ).typedArrayConstructor;
    if (typedArrayConstructor === undefined) {
      throw new Error(
        `DType ${
          ElementTypeName[elementType.value]
        } is not supported in this environment (missing TypedArray constructor).`,
      );
    }
    if (typedArrayConstructor.BYTES_PER_ELEMENT !== byteWidth) {
      throw new Error(
        `Byte width ${byteWidth} of the tensor's element type ${
          ElementTypeName[elementType.value]
        } ` +
          `does not match the expected byte width ${typedArrayConstructor.BYTES_PER_ELEMENT} of the ${typedArrayConstructor.name}.`,
      );
    }

    const dataPtr = this.liteRtTensorHandle.lock(
      getGlobalLiteRt().liteRtWasm.LiteRtTensorBufferLockMode.READ,
    );
    try {
      const uint8Array = liteRtWasm.HEAPU8.slice(
        dataPtr,
        dataPtr + this.liteRtTensorHandle.packedSize(),
      );

      const typedArray = new typedArrayConstructor(
        uint8Array.buffer,
        uint8Array.byteOffset,
        uint8Array.byteLength / byteWidth,
      );

      return typedArray;
    } finally {
      this.liteRtTensorHandle.unlock();
    }
  }

  getBufferType(): TensorBufferType {
    this.ensureNotDeleted();
    return this.liteRtTensorHandle.bufferType().value;
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
    if (!this.liteRtTensorHandle.isWebGpuMemory()) {
      throw new Error(
        'Cannot convert a Tensor with non-WebGPU memory to a GPUBuffer.',
      );
    }
    const bufferTypeValue = this.liteRtTensorHandle.bufferType().value;
    if (
      bufferTypeValue !==
        liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER.value &&
      bufferTypeValue !==
        liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER_PACKED.value
    ) {
      throw new Error(
        'Cannot convert a Tensor with host memory to a GPUBuffer.',
      );
    }
    // TODO: markoristic - Support tensors with strides or padding.
    if (
      this.liteRtTensorHandle.size() !== this.liteRtTensorHandle.packedSize() ||
      this.liteRtTensorHandle.offset() !== 0
    ) {
      throw new Error('Tensors with strides or padding are not yet supported.');
    }

    const gpuBufferId = this.liteRtTensorHandle.getWebGpuBuffer();
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
    return this.liteRtTensorHandle.bufferType().value;
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
    this.liteRtTensorHandle.delete();
    this.onDelete?.();
  }
}

/**
 * Get the TensorType of a LiteRtTensorHandle.
 */
function liteRtTensorHandleToTensorType(
  liteRtTensorHandle: LiteRtTensorHandle,
): TensorType {
  const liteRtRankedTensorType = liteRtTensorHandle.tensorType();
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
 * Creates a LiteRtTensorHandle from a GPUBuffer.
 *
 * Returns the LiteRtTensorHandle and the WebGPU buffer Wasm heap pointer. In
 * the emscripten implementation, the pointer actually refers to the index of
 * the buffer in the WebGPU module's Internals.jsObjects array, and must be
 * released with `wgpuBufferRelease`.
 */
function webGpuBufferToLiteRtTensorHandle(
  gpuBuffer: GPUBuffer,
  shape: Dimensions,
  dtype: DType,
  environment: Environment,
): [LiteRtTensorHandle, number] {
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

  const liteRtTensorHandle =
    liteRtWasm.LiteRtTensorHandle.createFromWebGpuBuffer(
      environment.liteRtEnvironment,
      rankedTensorType,
      liteRtWasm.LiteRtTensorBufferType.WEB_GPU_BUFFER_PACKED,
      importedGpuBufferPtr,
      gpuBuffer.size,
    );
  rankedTensorType.delete();

  return [liteRtTensorHandle, importedGpuBufferPtr];
}

/**
 * Creates a LiteRtTensorHandle from a TypedArray and optional shape.
 */
function typedArrayToLiteRtTensorHandle(
  data: TypedArray,
  shape?: Dimensions,
  dataType?: DType,
  environment?: Environment,
): LiteRtTensorHandle {
  const globalLiteRt = getGlobalLiteRt();
  const liteRtWasm = globalLiteRt.liteRtWasm;
  environment = environment ?? globalLiteRt.getDefaultEnvironment();

  const elementType = dataType
    ? getDataType(dataType).elementType
    : getDataType(data).elementType;
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

  // Create the LiteRtTensorHandle.
  const liteRtTensorHandle = liteRtWasm.LiteRtTensorHandle.createManaged(
    environment.liteRtEnvironment,
    liteRtWasm.LiteRtTensorBufferType.HOST_MEMORY,
    rankedTensorType,
    bufferSize,
  );
  rankedTensorType.delete(); // Delete our copy.

  // Write the data to the LiteRtTensorHandle.
  const dataPtr = liteRtTensorHandle.lock(
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
    liteRtTensorHandle.unlock();
  }

  return liteRtTensorHandle;
}

function makeBinOp(
  op: (
    wasm: LiteRtWasm,
    a: LiteRtTensorHandle,
    b: LiteRtTensorHandle,
  ) => LiteRtTensorHandle,
): (a: Tensor, b: Tensor) => Tensor {
  return (a: Tensor, b: Tensor) => {
    if (a.environment !== b.environment) {
      throw new Error(
        'Cannot perform arithmetic operations on tensors from different environments.',
      );
    }
    a.ensureNotDeleted();
    b.ensureNotDeleted();
    const resultHandle = op(
      getGlobalLiteRt().liteRtWasm,
      a.liteRtTensorHandle,
      b.liteRtTensorHandle,
    );
    return new Tensor(resultHandle, a.environment);
  };
}

function makeUnaryOp(
  op: (wasm: LiteRtWasm, a: LiteRtTensorHandle) => LiteRtTensorHandle,
): (a: Tensor) => Tensor {
  return (a: Tensor) => {
    a.ensureNotDeleted();
    const resultHandle = op(getGlobalLiteRt().liteRtWasm, a.liteRtTensorHandle);
    return new Tensor(resultHandle, a.environment);
  };
}

/**
 * Adds two tensors element-wise.
 */
export const add: (a: Tensor, b: Tensor) => Tensor = makeBinOp((wasm, a, b) =>
  wasm.add(a, b),
);

/**
 * Multiplies two tensors element-wise.
 */
export const mul: (a: Tensor, b: Tensor) => Tensor = makeBinOp((wasm, a, b) =>
  wasm.mul(a, b),
);

/**
 * Subtracts tensor b from tensor a element-wise.
 */
export const sub: (a: Tensor, b: Tensor) => Tensor = makeBinOp((wasm, a, b) =>
  wasm.sub(a, b),
);

/**
 * Divides tensor a by tensor b element-wise.
 */
export const div: (a: Tensor, b: Tensor) => Tensor = makeBinOp((wasm, a, b) =>
  wasm.div(a, b),
);

/**
 * Computes the power of one tensor to another (a^b) element-wise.
 */
export const pow: (a: Tensor, b: Tensor) => Tensor = makeBinOp((wasm, a, b) =>
  wasm.pow(a, b),
);

/**
 * Computes the element-wise minimum of two tensors.
 */
export const minimum: (a: Tensor, b: Tensor) => Tensor = makeBinOp(
  (wasm, a, b) => wasm.minimum(a, b),
);

/**
 * Computes the element-wise maximum of two tensors.
 */
export const maximum: (a: Tensor, b: Tensor) => Tensor = makeBinOp(
  (wasm, a, b) => wasm.maximum(a, b),
);

/**
 * Computes the element-wise floor division (floor(a / b)).
 */
export const floorDiv: (a: Tensor, b: Tensor) => Tensor = makeBinOp(
  (wasm, a, b) => wasm.floorDiv(a, b),
);

/**
 * Computes the element-wise floor division remainder (floorMod(a, b)).
 */
export const floorMod: (a: Tensor, b: Tensor) => Tensor = makeBinOp(
  (wasm, a, b) => wasm.floorMod(a, b),
);

/**
 * Returns the truth value of (a == b) element-wise.
 */
export const equal: (a: Tensor, b: Tensor) => Tensor = makeBinOp(
  (wasm, a, b) => wasm.equal(a, b),
);

/**
 * Returns the truth value of (a != b) element-wise.
 */
export const notEqual: (a: Tensor, b: Tensor) => Tensor = makeBinOp(
  (wasm, a, b) => wasm.notEqual(a, b),
);

/**
 * Returns the truth value of (a < b) element-wise.
 */
export const less: (a: Tensor, b: Tensor) => Tensor = makeBinOp((wasm, a, b) =>
  wasm.less(a, b),
);

/**
 * Returns the truth value of (a > b) element-wise.
 */
export const greater: (a: Tensor, b: Tensor) => Tensor = makeBinOp(
  (wasm, a, b) => wasm.greater(a, b),
);

/**
 * Returns the truth value of (a >= b) element-wise.
 */
export const greaterEqual: (a: Tensor, b: Tensor) => Tensor = makeBinOp(
  (wasm, a, b) => wasm.greaterEqual(a, b),
);

/**
 * Returns the truth value of (a <= b) element-wise.
 */
export const lessEqual: (a: Tensor, b: Tensor) => Tensor = (a, b) =>
  greaterEqual(b, a);

/**
 * Computes logical AND element-wise.
 */
export const logicalAnd: (a: Tensor, b: Tensor) => Tensor = makeBinOp(
  (wasm, a, b) => wasm.logicalAnd(a, b),
);

/**
 * Computes logical OR element-wise.
 */
export const logicalOr: (a: Tensor, b: Tensor) => Tensor = makeBinOp(
  (wasm, a, b) => wasm.logicalOr(a, b),
);

/**
 * Computes logical NOT element-wise.
 */
export const logicalNot: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.logicalNot(a),
);

/**
 * Selects elements from trueVal or falseVal depending on condition.
 */
export function select(
  condition: Tensor,
  trueVal: Tensor,
  falseVal: Tensor,
): Tensor {
  condition.ensureNotDeleted();
  trueVal.ensureNotDeleted();
  falseVal.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.select(
    condition.liteRtTensorHandle,
    trueVal.liteRtTensorHandle,
    falseVal.liteRtTensorHandle,
  );
  return new Tensor(resultHandle, condition.environment);
}

/**
 * Computes rectified linear unit element-wise.
 */
export const relu: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.relu(a),
);

/**
 * Multiplies slices of two tensors in batches.
 *
 * @param x The first input tensor with rank >= 2.
 * @param y The second input tensor with rank >= 2.
 * @param adjX Whether to transpose the last two dimensions of x.
 * @param adjY Whether to transpose the last two dimensions of y.
 * @returns The matrix product tensor.
 */
export function batchMatMul(
    x: Tensor,
    y: Tensor,
    adjX: boolean = false,
    adjY: boolean = false,
    ): Tensor {
  x.ensureNotDeleted();
  y.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.batchMatMul(
      x.liteRtTensorHandle,
      y.liteRtTensorHandle,
      adjX,
      adjY,
  );
  return new Tensor(resultHandle, x.environment);
}

/**
 * Computes a matrix multiplication with optional bias.
 *
 * @param input Input tensor of shape [batch, in_features] or higher rank.
 * @param weights Weights matrix of shape [out_features, in_features].
 * @param bias Optional bias vector of shape [out_features].
 * @returns Output tensor of shape [batch, out_features].
 */
export function fullyConnected(
    input: Tensor,
    weights: Tensor,
    bias?: Tensor,
    ): Tensor {
  input.ensureNotDeleted();
  weights.ensureNotDeleted();
  if (bias) bias.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.fullyConnected(
      input.liteRtTensorHandle,
      weights.liteRtTensorHandle,
      bias ? bias.liteRtTensorHandle : (undefined as any),
  );
  return new Tensor(resultHandle, input.environment);
}

/**
 * Computes the softmax activation of a tensor.
 *
 * @param a The input tensor.
 * @param beta Optional scaling factor (default: 1.0).
 * @returns The softmax output tensor.
 */
export function softmax(a: Tensor, beta: number = 1.0): Tensor {
  a.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.softmax(a.liteRtTensorHandle, beta);
  return new Tensor(resultHandle, a.environment);
}

/**
 * Computes the logistic (sigmoid) activation element-wise: 1 / (1 + exp(-x)).
 */
export const logistic: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.logistic(a),
);

/**
 * Computes the hyperbolic tangent activation element-wise.
 */
export const tanh: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.tanh(a),
);

/**
 * Computes the Gaussian Error Linear Unit (GELU) activation.
 *
 * @param input The input tensor.
 * @param approximate Whether to use the faster tanh approximation.
 * @returns The GELU output tensor.
 */
export function gelu(input: Tensor, approximate: boolean = false): Tensor {
  input.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.gelu(input.liteRtTensorHandle, approximate);
  return new Tensor(resultHandle, input.environment);
}

/**
 * Computes Rectified Linear 6 activation: min(max(0, x), 6).
 */
export const relu6: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.relu6(a),
);

/**
 * Computes Leaky ReLU activation: max(alpha * x, x).
 */
export function leakyRelu(a: Tensor, alpha: number = 0.2): Tensor {
  a.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.leakyRelu(a.liteRtTensorHandle, alpha);
  return new Tensor(resultHandle, a.environment);
}

/**
 * Computes Exponential Linear Unit (ELU) activation: x < 0 ? exp(x) - 1 : x.
 */
export const elu: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.elu(a),
);

/**
 * Computes Hard Swish activation: x * relu6(x + 3) / 6.
 */
export const hardSwish: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.hardSwish(a),
);

/**
 * Computes Log-Softmax activation element-wise.
 */
export const logSoftmax: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.logSoftmax(a),
);

/**
 * Padding mode for 2D spatial operations.
 */
export type Padding = 'SAME' | 'VALID' | 0 | 1;

/**
 * Options for 2D convolution operations.
 */
export interface Conv2dOptions {
  strideH?: number;
  strideW?: number;
  padding?: Padding;
  dilationH?: number;
  dilationW?: number;
  bias?: Tensor;
}

/**
 * Options for depthwise 2D convolution operations.
 */
export interface DepthwiseConv2dOptions extends Conv2dOptions {
  depthMultiplier?: number;
}

/**
 * Options for image resizing operations.
 */
export interface ResizeOptions {
  alignCorners?: boolean;
  halfPixelCenters?: boolean;
}

/**
 * Options for 2D pooling operations (max pooling and average pooling).
 */
export interface Pool2dOptions {
  filterSize?: number | [number, number];
  filterHeight?: number;
  filterWidth?: number;
  strides?: number | [number, number];
  strideH?: number;
  strideW?: number;
  padding?: Padding;
}

/**
 * Options for 2D transposed convolution operations.
 */
export interface TransposeConv2dOptions {
  strides?: number | [number, number];
  strideH?: number;
  strideW?: number;
  padding?: Padding;
  bias?: Tensor;
}

function parsePadding(padding: Padding = 'SAME'): number {
  if (typeof padding === 'number') {
    return padding;
  }
  return padding.toUpperCase() === 'VALID' ? 1 : 0;
}

/**
 * Performs a standard 2D convolution on an input tensor of shape [B, H, W, C_in].
 *
 * @param input Input tensor of shape [batch, height, width, in_channels].
 * @param filter Filter weights tensor of shape [out_channels, filter_h, filter_w, in_channels].
 * @param options Optional strides, padding, dilation, and bias.
 * @returns Convolved output tensor.
 */
export function conv2d(
  input: Tensor,
  filter: Tensor,
  options: Conv2dOptions = {},
): Tensor {
  input.ensureNotDeleted();
  filter.ensureNotDeleted();
  if (options.bias) options.bias.ensureNotDeleted();
  const strideH = options.strideH ?? 1;
  const strideW = options.strideW ?? 1;
  const padding = parsePadding(options.padding);
  const dilationH = options.dilationH ?? 1;
  const dilationW = options.dilationW ?? 1;
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.conv2d(
    input.liteRtTensorHandle,
    filter.liteRtTensorHandle,
    options.bias ? options.bias.liteRtTensorHandle : (undefined as any),
    strideH,
    strideW,
    padding,
    dilationH,
    dilationW,
  );
  return new Tensor(resultHandle, input.environment);
}

/**
 * Performs a depthwise 2D convolution on an input tensor of shape [B, H, W, C_in].
 *
 * @param input Input tensor of shape [batch, height, width, in_channels].
 * @param filter Filter weights tensor of shape [1, filter_h, filter_w, in_channels * depth_multiplier].
 * @param options Optional strides, padding, depth multiplier, dilation, and bias.
 * @returns Convolved output tensor.
 */
export function depthwiseConv2d(
  input: Tensor,
  filter: Tensor,
  options: DepthwiseConv2dOptions = {},
): Tensor {
  input.ensureNotDeleted();
  filter.ensureNotDeleted();
  if (options.bias) options.bias.ensureNotDeleted();
  const strideH = options.strideH ?? 1;
  const strideW = options.strideW ?? 1;
  const padding = parsePadding(options.padding);
  const depthMultiplier = options.depthMultiplier ?? 1;
  const dilationH = options.dilationH ?? 1;
  const dilationW = options.dilationW ?? 1;
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.depthwiseConv2d(
    input.liteRtTensorHandle,
    filter.liteRtTensorHandle,
    options.bias ? options.bias.liteRtTensorHandle : (undefined as any),
    strideH,
    strideW,
    padding,
    depthMultiplier,
    dilationH,
    dilationW,
  );
  return new Tensor(resultHandle, input.environment);
}

/**
 * Reshapes a tensor to a new shape.
 *
 * @param input The input tensor.
 * @param shape The target dimensions.
 * @returns The reshaped tensor.
 */
export function reshape(input: Tensor, shape: number[]): Tensor {
  input.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.reshape(input.liteRtTensorHandle, shape);
  return new Tensor(resultHandle, input.environment);
}

/**
 * Permutes the dimensions of a tensor according to a given permutation.
 *
 * @param input The input tensor.
 * @param perm The permutation array specifying the dimension ordering.
 * @returns The transposed tensor.
 */
export function transpose(input: Tensor, perm: number[]): Tensor {
  input.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.transpose(input.liteRtTensorHandle, perm);
  return new Tensor(resultHandle, input.environment);
}

/**
 * Concatenates a list of tensors along a specified axis.
 *
 * @param inputs Array of tensors with matching shapes except along axis.
 * @param axis The dimension along which to concatenate (default: 0).
 * @returns The concatenated tensor.
 */
export function concat(inputs: Tensor[], axis: number = 0): Tensor {
  if (inputs.length === 0) {
    throw new Error('concat requires at least one input tensor.');
  }
  for (const t of inputs) {
    t.ensureNotDeleted();
  }
  const rank = inputs[0].type.layout.dimensions.length;
  const normalizedAxis = axis < 0 ? axis + rank : axis;
  if (normalizedAxis < 0 || normalizedAxis >= rank) {
    throw new Error(
      `concat axis ${axis} is out of bounds for tensor of rank ${rank}.`,
    );
  }
  const wasm = getGlobalLiteRt().liteRtWasm;
  const handles = inputs.map((t) => t.liteRtTensorHandle);
  const resultHandle = wasm.concatenation(handles, normalizedAxis);
  return new Tensor(resultHandle, inputs[0].environment);
}

export const concatenation = concat;

/**
 * Extracts a sub-tensor slice from an input tensor.
 *
 * @param input The input tensor.
 * @param begin 0-based starting offsets for each dimension.
 * @param size Number of elements to extract along each dimension.
 * @returns The sliced tensor.
 */
export function slice(
  input: Tensor,
  begin: number[],
  size: number[],
): Tensor {
  input.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.slice(input.liteRtTensorHandle, begin, size);
  return new Tensor(resultHandle, input.environment);
}

/**
 * Pads a tensor according to the specified padding dimensions.
 *
 * @param a The input tensor.
 * @param paddings A 2D int32 tensor or array of shape [rank, 2] indicating before/after padding per axis.
 * @returns The padded tensor.
 */
export function pad(a: Tensor, paddings: Tensor | number[][]): Tensor {
  a.ensureNotDeleted();
  const rank = a.type.layout.dimensions.length;
  let paddingsTensor: Tensor;
  let ownsPaddingsTensor = false;
  if (paddings instanceof Tensor) {
    paddings.ensureNotDeleted();
    if (paddings.type.dtype !== 'int32') {
      throw new Error(
        `pad requires paddings tensor to have dtype 'int32', but got '${paddings.type.dtype}'.`,
      );
    }
    const padDims = paddings.type.layout.dimensions;
    if (padDims.length !== 2 || padDims[0] !== rank || padDims[1] !== 2) {
      throw new Error(
        `pad requires paddings shape to be [${rank}, 2], but got [${padDims.join(', ')}].`,
      );
    }
    paddingsTensor = paddings;
  } else {
    if (paddings.length !== rank || paddings.some((p) => p.length !== 2)) {
      throw new Error(`pad requires paddings shape to be [${rank}, 2].`);
    }
    const flatPaddings = new Int32Array(rank * 2);
    for (let i = 0; i < rank; ++i) {
      if (paddings[i][0] < 0 || paddings[i][1] < 0) {
        throw new Error('pad values must be non-negative.');
      }
      flatPaddings[i * 2] = paddings[i][0];
      flatPaddings[i * 2 + 1] = paddings[i][1];
    }
    paddingsTensor = Tensor.fromTypedArray(
      flatPaddings,
      [rank, 2],
      a.environment,
    );
    ownsPaddingsTensor = true;
  }
  let resultHandle: LiteRtTensorHandle;
  try {
    const wasm = getGlobalLiteRt().liteRtWasm;
    resultHandle = wasm.pad(
      a.liteRtTensorHandle,
      paddingsTensor.liteRtTensorHandle,
    );
  } finally {
    if (ownsPaddingsTensor) {
      paddingsTensor.delete();
    }
  }
  return new Tensor(resultHandle, a.environment);
}

/**
 * Inserts a dimension of 1 into a tensor's shape.
 *
 * @param input The input tensor.
 * @param axis The dimension index at which to expand the shape.
 * @returns The expanded tensor.
 */
export function expandDims(input: Tensor, axis: number): Tensor {
  input.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.expandDims(input.liteRtTensorHandle, axis);
  return new Tensor(resultHandle, input.environment);
}

/**
 * Removes dimensions of size 1 from the shape of a tensor.
 *
 * @param input The input tensor.
 * @param squeezeDims Optional list of dimensions to squeeze. If empty, squeezes all dimensions of size 1.
 * @returns The squeezed tensor.
 */
export function squeeze(input: Tensor, squeezeDims?: number[]): Tensor {
  input.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.squeeze(input.liteRtTensorHandle, squeezeDims);
  return new Tensor(resultHandle, input.environment);
}

/**
 * Constructs a tensor by tiling a given tensor multiples times.
 *
 * @param input The input tensor.
 * @param multiples 1D array of integers specifying the number of repeats per dimension.
 * @returns The tiled tensor.
 */
export function tile(input: Tensor, multiples: number[]): Tensor {
  input.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.tile(input.liteRtTensorHandle, multiples);
  return new Tensor(resultHandle, input.environment);
}

/**
 * Packs a list of rank-R tensors into one rank-(R+1) tensor.
 *
 * @param inputs Array of tensors with identical shapes.
 * @param axis The axis along which to pack (default: 0).
 * @returns The packed tensor.
 */
export function pack(inputs: Tensor[], axis: number = 0): Tensor {
  if (inputs.length === 0) {
    throw new Error('pack requires at least one input tensor.');
  }
  for (const t of inputs) {
    t.ensureNotDeleted();
  }
  const rank = inputs[0].type.layout.dimensions.length;
  const normalizedAxis = axis < 0 ? axis + rank + 1 : axis;
  if (normalizedAxis < 0 || normalizedAxis > rank) {
    throw new Error(
      `pack axis ${axis} is out of bounds for input tensor of rank ${rank}.`,
    );
  }
  const wasm = getGlobalLiteRt().liteRtWasm;
  const handles = inputs.map((t) => t.liteRtTensorHandle);
  const resultHandle = wasm.pack(handles, normalizedAxis);
  return new Tensor(resultHandle, inputs[0].environment);
}

/**
 * Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
 *
 * @param input The input tensor.
 * @param num The number of tensors to unpack along axis.
 * @param axis The axis along which to unpack (default: 0).
 * @returns An array of unpacked tensors.
 */
export function unpack(input: Tensor, num: number, axis: number = 0): Tensor[] {
  input.ensureNotDeleted();
  const rank = input.type.layout.dimensions.length;
  const normalizedAxis = axis < 0 ? axis + rank : axis;
  if (normalizedAxis < 0 || normalizedAxis >= rank) {
    throw new Error(
      `unpack axis ${axis} is out of bounds for tensor of rank ${rank}.`,
    );
  }
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandles = wasm.unpack(
    input.liteRtTensorHandle,
    num,
    normalizedAxis,
  );
  return resultHandles.map((h) => new Tensor(h, input.environment));
}

/**
 * Splits a tensor into sub-tensors along a specified axis.
 *
 * @param input The input tensor.
 * @param numSplits The number of pieces to split the tensor into.
 * @param axis The dimension along which to split (default: 0).
 * @returns An array of split tensors.
 */
export function split(
  input: Tensor,
  numSplits: number,
  axis: number = 0,
): Tensor[] {
  input.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandles = wasm.split(input.liteRtTensorHandle, axis, numSplits);
  return resultHandles.map((h) => new Tensor(h, input.environment));
}

/**
 * Computes the mean of elements across specified dimensions of a tensor.
 *
 * @param a The input tensor.
 * @param axes The dimensions to reduce. If undefined, reduces all dimensions.
 * @param keepDims Whether to retain reduced dimensions with length 1.
 * @returns The reduced tensor.
 */
export function mean(
  a: Tensor,
  axes?: number | number[],
  keepDims: boolean = false,
): Tensor {
  a.ensureNotDeleted();
  let axesArray: number[];
  if (axes === undefined) {
    const rank = a.type.layout.dimensions.length;
    axesArray = Array.from({length: rank}, (_, i) => i);
  } else if (typeof axes === 'number') {
    axesArray = [axes];
  } else {
    axesArray = axes;
  }
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.mean(a.liteRtTensorHandle, axesArray, keepDims);
  return new Tensor(resultHandle, a.environment);
}

/**
 * Computes the sum of elements across specified dimensions of a tensor.
 *
 * @param a The input tensor.
 * @param axes The dimensions to reduce. If undefined, reduces all dimensions.
 * @param keepDims Whether to retain reduced dimensions with length 1.
 * @returns The reduced tensor.
 */
export function sum(
  a: Tensor,
  axes?: number | number[],
  keepDims: boolean = false,
): Tensor {
  a.ensureNotDeleted();
  let axesArray: number[];
  if (axes === undefined) {
    const rank = a.type.layout.dimensions.length;
    axesArray = Array.from({length: rank}, (_, i) => i);
  } else if (typeof axes === 'number') {
    axesArray = [axes];
  } else {
    axesArray = axes;
  }
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.sum(a.liteRtTensorHandle, axesArray, keepDims);
  return new Tensor(resultHandle, a.environment);
}

/**
 * Computes the maximum of elements across specified dimensions of a tensor.
 *
 * @param a The input tensor.
 * @param axes The dimensions to reduce. If undefined, reduces all dimensions.
 * @param keepDims Whether to retain reduced dimensions with length 1.
 * @returns The reduced tensor.
 */
export function reduceMax(
  a: Tensor,
  axes?: number | number[],
  keepDims: boolean = false,
): Tensor {
  a.ensureNotDeleted();
  let axesArray: number[];
  if (axes === undefined) {
    const rank = a.type.layout.dimensions.length;
    axesArray = Array.from({length: rank}, (_, i) => i);
  } else if (typeof axes === 'number') {
    axesArray = [axes];
  } else {
    axesArray = axes;
  }
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.reduceMax(a.liteRtTensorHandle, axesArray, keepDims);
  return new Tensor(resultHandle, a.environment);
}

/**
 * Returns the index with the largest value across axes of a tensor.
 *
 * @param a The input tensor.
 * @param axis The dimension to reduce across (default: 0).
 * @param outputType The output integer data type ('int32' or 'int64', default: 'int32').
 * @returns The tensor containing the argmax indices.
 */
export function argMax(
  a: Tensor,
  axis: number = 0,
  outputType: 'int32' | 'int64' = 'int32',
): Tensor {
  a.ensureNotDeleted();
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.argMax(a.liteRtTensorHandle, axis, outputType);
  return new Tensor(resultHandle, a.environment);
}

/**
 * Resizes 4D images using bilinear interpolation.
 *
 * @param input 4D tensor of shape [batch, height, width, channels].
 * @param size Target [targetHeight, targetWidth].
 * @param alignCornersOrOptions If boolean, alignCorners flag; if ResizeOptions, options object.
 * @param halfPixelCenters Whether half pixel centers are used.
 * @returns The resized tensor.
 */
export function resizeBilinear(
  input: Tensor,
  size: [number, number] | number[],
  alignCornersOrOptions?: boolean | ResizeOptions,
  halfPixelCenters?: boolean,
): Tensor {
  input.ensureNotDeleted();
  let alignCorners = false;
  let halfPixel = false;
  if (typeof alignCornersOrOptions === 'boolean') {
    alignCorners = alignCornersOrOptions;
    halfPixel = halfPixelCenters ?? false;
  } else if (alignCornersOrOptions) {
    alignCorners = alignCornersOrOptions.alignCorners ?? false;
    halfPixel = alignCornersOrOptions.halfPixelCenters ?? false;
  }
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.resizeBilinear(
    input.liteRtTensorHandle,
    Array.from(size),
    alignCorners,
    halfPixel,
  );
  return new Tensor(resultHandle, input.environment);
}

/**
 * Resizes 4D images using nearest neighbor interpolation.
 *
 * @param input 4D tensor of shape [batch, height, width, channels].
 * @param size Target [targetHeight, targetWidth].
 * @param alignCornersOrOptions If boolean, alignCorners flag; if ResizeOptions, options object.
 * @param halfPixelCenters Whether half pixel centers are used.
 * @returns The resized tensor.
 */
export function resizeNearestNeighbor(
  input: Tensor,
  size: [number, number] | number[],
  alignCornersOrOptions?: boolean | ResizeOptions,
  halfPixelCenters?: boolean,
): Tensor {
  input.ensureNotDeleted();
  let alignCorners = false;
  let halfPixel = false;
  if (typeof alignCornersOrOptions === 'boolean') {
    alignCorners = alignCornersOrOptions;
    halfPixel = halfPixelCenters ?? false;
  } else if (alignCornersOrOptions) {
    alignCorners = alignCornersOrOptions.alignCorners ?? false;
    halfPixel = alignCornersOrOptions.halfPixelCenters ?? false;
  }
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.resizeNearestNeighbor(
    input.liteRtTensorHandle,
    Array.from(size),
    alignCorners,
    halfPixel,
  );
  return new Tensor(resultHandle, input.environment);
}

function parsePoolParams(
  filterSizeOrOptions?: number | [number, number] | Pool2dOptions,
  strides?: number | [number, number],
  padding?: Padding,
): {
  filterHeight: number;
  filterWidth: number;
  strideH: number;
  strideW: number;
  paddingVal: number;
} {
  let filterHeight = 2;
  let filterWidth = 2;
  let strideH = 2;
  let strideW = 2;
  let paddingVal = 0;

  if (
    typeof filterSizeOrOptions === 'object' &&
    !Array.isArray(filterSizeOrOptions)
  ) {
    const opts = filterSizeOrOptions;
    if (opts.filterSize !== undefined) {
      if (Array.isArray(opts.filterSize)) {
        filterHeight = opts.filterSize[0];
        filterWidth = opts.filterSize[1];
      } else {
        filterHeight = opts.filterSize;
        filterWidth = opts.filterSize;
      }
    }
    if (opts.filterHeight !== undefined) filterHeight = opts.filterHeight;
    if (opts.filterWidth !== undefined) filterWidth = opts.filterWidth;

    if (opts.strides !== undefined) {
      if (Array.isArray(opts.strides)) {
        strideH = opts.strides[0];
        strideW = opts.strides[1];
      } else {
        strideH = opts.strides;
        strideW = opts.strides;
      }
    } else {
      strideH = filterHeight;
      strideW = filterWidth;
    }
    if (opts.strideH !== undefined) strideH = opts.strideH;
    if (opts.strideW !== undefined) strideW = opts.strideW;
    paddingVal = parsePadding(opts.padding);
  } else {
    if (filterSizeOrOptions !== undefined) {
      if (Array.isArray(filterSizeOrOptions)) {
        filterHeight = filterSizeOrOptions[0];
        filterWidth = filterSizeOrOptions[1];
      } else {
        filterHeight = filterSizeOrOptions;
        filterWidth = filterSizeOrOptions;
      }
    }
    if (strides !== undefined) {
      if (Array.isArray(strides)) {
        strideH = strides[0];
        strideW = strides[1];
      } else {
        strideH = strides;
        strideW = strides;
      }
    } else {
      strideH = filterHeight;
      strideW = filterWidth;
    }
    paddingVal = parsePadding(padding);
  }
  return {filterHeight, filterWidth, strideH, strideW, paddingVal};
}

/**
 * Performs 2D max pooling on an input tensor of shape [B, H, W, C].
 *
 * @param input 4D tensor.
 * @param filterSizeOrOptions Filter dimensions or pool options object.
 * @param strides Stride dimensions along height and width.
 * @param padding Padding mode ('SAME' or 'VALID').
 * @returns Max pooled tensor.
 */
export function maxPool2d(
  input: Tensor,
  filterSizeOrOptions?: number | [number, number] | Pool2dOptions,
  strides?: number | [number, number],
  padding?: Padding,
): Tensor {
  input.ensureNotDeleted();
  const {filterHeight, filterWidth, strideH, strideW, paddingVal} =
    parsePoolParams(filterSizeOrOptions, strides, padding);
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.maxPool2d(
    input.liteRtTensorHandle,
    filterHeight,
    filterWidth,
    strideH,
    strideW,
    paddingVal,
  );
  return new Tensor(resultHandle, input.environment);
}

/**
 * Performs 2D average pooling on an input tensor of shape [B, H, W, C].
 *
 * @param input 4D tensor.
 * @param filterSizeOrOptions Filter dimensions or pool options object.
 * @param strides Stride dimensions along height and width.
 * @param padding Padding mode ('SAME' or 'VALID').
 * @returns Average pooled tensor.
 */
export function avgPool2d(
  input: Tensor,
  filterSizeOrOptions?: number | [number, number] | Pool2dOptions,
  strides?: number | [number, number],
  padding?: Padding,
): Tensor {
  input.ensureNotDeleted();
  const {filterHeight, filterWidth, strideH, strideW, paddingVal} =
    parsePoolParams(filterSizeOrOptions, strides, padding);
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.avgPool2d(
    input.liteRtTensorHandle,
    filterHeight,
    filterWidth,
    strideH,
    strideW,
    paddingVal,
  );
  return new Tensor(resultHandle, input.environment);
}

/**
 * Performs a 2D transposed convolution (deconvolution) on an input tensor.
 *
 * @param input Input tensor of shape [batch, height, width, in_channels].
 * @param filter Filter weights tensor of shape [out_channels, filter_h, filter_w, in_channels].
 * @param outputShape 4D target output shape [batch, out_h, out_w, out_channels].
 * @param options Strides, padding, and optional bias.
 * @returns Deconvolved output tensor.
 */
export function transposeConv2d(
  input: Tensor,
  filter: Tensor,
  outputShape: number[],
  options: TransposeConv2dOptions = {},
): Tensor {
  input.ensureNotDeleted();
  filter.ensureNotDeleted();
  if (options.bias) options.bias.ensureNotDeleted();
  let strideH = 1;
  let strideW = 1;
  if (options.strides !== undefined) {
    if (Array.isArray(options.strides)) {
      strideH = options.strides[0];
      strideW = options.strides[1];
    } else {
      strideH = options.strides;
      strideW = options.strides;
    }
  }
  if (options.strideH !== undefined) strideH = options.strideH;
  if (options.strideW !== undefined) strideW = options.strideW;
  const padding = parsePadding(options.padding);
  const wasm = getGlobalLiteRt().liteRtWasm;
  const resultHandle = wasm.transposeConv2d(
    input.liteRtTensorHandle,
    filter.liteRtTensorHandle,
    Array.from(outputShape),
    options.bias ? options.bias.liteRtTensorHandle : (undefined as any),
    strideH,
    strideW,
    padding,
  );
  return new Tensor(resultHandle, input.environment);
}

/**
 * Computes the absolute value element-wise.
 */
export const abs: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.abs(a),
);

/**
 * Computes numerical negative (-x) element-wise.
 */
export const neg: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.neg(a),
);

/**
 * Computes the square root element-wise.
 */
export const sqrt: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.sqrt(a),
);

/**
 * Computes reciprocal of square root (1 / sqrt(x)) element-wise.
 */
export const rsqrt: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.rsqrt(a),
);

/**
 * Computes exponential (e^x) element-wise.
 */
export const exp: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.exp(a),
);

/**
 * Computes natural logarithm (ln(x)) element-wise.
 */
export const log: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.log(a),
);

/**
 * Computes sine (sin(x)) element-wise.
 */
export const sin: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.sin(a),
);

/**
 * Computes cosine (cos(x)) element-wise.
 */
export const cos: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.cos(a),
);

/**
 * Computes ceiling (smallest integer >= x) element-wise.
 */
export const ceil: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.ceil(a),
);

/**
 * Computes floor (largest integer <= x) element-wise.
 */
export const floor: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.floor(a),
);

/**
 * Rounds elements to the nearest integer element-wise.
 */
export const round: (a: Tensor) => Tensor = makeUnaryOp((wasm, a) =>
  wasm.round(a),
);
