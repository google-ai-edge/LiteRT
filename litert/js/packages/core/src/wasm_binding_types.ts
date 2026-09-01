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

import {WasmModule} from '@litertjs/wasm-utils';

// These external interfaces that are implemented by the C++ code must use the
// `declare` keyword to prevent the JS Compiler from renaming them.

// Several interfaces in this file represent bound C++ objects. These interfaces
// do not follow TypeScript's structural typing rules (a user should not be able
// to assign an object they create themselves to one of these interfaces).
//
// We use unique symbol keys to enforce nominal typing. This is similar to the
// approach in https://www.typescriptlang.org/play/?#example/nominal-typing, but
// unique symbol keys are safer than string keys since they cannot be assigned
// by the user.
// https://www.typescriptlang.org/docs/handbook/type-compatibility.html
// https://www.typescriptlang.org/docs/handbook/symbols.html#unique-symbol

/**
 * An object that can be deleted.
 */
export declare interface Deletable {
  delete(): void;
}

/**
 * A C++ vector of elements.
 */
export declare interface EmscriptenVector<T> extends Deletable {
  size(): number;
  get(index: number): T&Deletable;
  push_back(item: T): void;
}

declare const emscriptenVectorInt32Brand: unique symbol;

/**
 * A C++ vector of int32_t.
 */
export declare interface EmscriptenVectorInt32 extends
    EmscriptenVector<number> {
  [emscriptenVectorInt32Brand]: void;
}

/**
 * A C++ vector of int32_t.
 */
export declare interface EmscriptenVectorInt32Constructor {
  new(): EmscriptenVectorInt32;
}

// Define a unique symbol to act as a brand for LiteRtEnvironment.
declare const liteRtEnvironmentBrand: unique symbol;

/**
 * The constructor for the C++ litert::Environment class in Wasm.
 */
export declare interface LiteRtEnvironmentConstructor {
  // Do not use this constructor. It is only here to allow checking
  // `instanceof LiteRtEnvironment` in TS. Use `create` instead.
  new(...args: never[]): LiteRtEnvironment;
  create(webGpuDevice: GPUDevice|null): LiteRtEnvironment;
}

/**
 * A C++ litert::Environment in Wasm.
 */
export declare interface LiteRtEnvironment extends Deletable {
  // Enforce nominal typing.
  [liteRtEnvironmentBrand]: void;
}

/**
 * Options for configuring the WebGPU delegate.
 */
export declare interface LiteRtGpuOptions {
  precision?: 'fp16'|'fp32';
}

/**
 * Options for configuring the WebNN delegate.
 */
export declare interface LiteRtWebNNOptions {
  devicePreference?: 'cpu'|'gpu'|'npu';
  powerPreference?: 'default'|'high-performance'|'low-power';
  precision?: 'fp32'|'fp16';
}

/**
 * Options for loading and compiling a LiteRt model.
 */
export declare interface LiteRtCompileOptions {
  accelerator?: 'wasm'|'webgpu'|'webnn';
  gpuOptions?: LiteRtGpuOptions;
  webNNOptions?: LiteRtWebNNOptions;
}

declare const liteRtModelBrand: unique symbol;

/**
 * The constructor for the C++ litert::Layout class in Wasm.
 */
export declare interface LiteRtLayoutConstructor {
  new(...args: never[]): LiteRtLayout;
  create(dimensions: EmscriptenVector<number>): LiteRtLayout;
}

/**
 * A C++ litert::Layout in Wasm.
 */
export declare interface LiteRtLayout extends Deletable {
  rank(): number;
  dimensions(): EmscriptenVector<number>;
  hasStrides(): boolean;
  strides(): EmscriptenVector<number>;
  numElements(): number;
}

/**
 * The constructor for the C++ litert::RankedTensorType class in Wasm.
 */
export declare interface LiteRtRankedTensorTypeConstructor {
  new(...args: never[]): LiteRtRankedTensorType;
  create(elementType: EmscriptenEnumElement<ElementType>, layout: LiteRtLayout):
      LiteRtRankedTensorType;
}

/**
 * A C++ litert::RankedTensorType in Wasm.
 */
export declare interface LiteRtRankedTensorType extends Deletable {
  elementType(): EmscriptenEnumElement<ElementType>;
  layout(): LiteRtLayout;
  bytes(): number;
}

/**
 * A C++ litert::Model in Wasm.
 */
export declare interface LiteRtModel extends Deletable {
  [liteRtModelBrand]: void;
  getNumSignatures(): number;
  getSignature(signatureIndex: number): LiteRtSimpleSignature;
  getInputTensorType(signatureIndex: number, inputIndex: number):
      LiteRtRankedTensorType;
  getOutputTensorType(signatureIndex: number, outputIndex: number):
      LiteRtRankedTensorType;
}

/**
 * A C++ litert::SimpleSignature in Wasm.
 */
export declare interface LiteRtSimpleSignature extends Deletable {
  key(): string;
  inputNames(): EmscriptenVector<string>;
  outputNames(): EmscriptenVector<string>;
}

/**
 * A C++ litert::TensorBufferRequirements in Wasm.
 */
export declare interface LiteRtTensorBufferRequirements extends Deletable {
  supportedTypes(): EmscriptenVector<LiteRtTensorBufferType>;
}

declare const liteRtCompiledModelBrand: unique symbol;

/**
 * A C++ litert::CompiledModel in Wasm.
 */
export declare interface LiteRtCompiledModel extends Deletable {
  [liteRtCompiledModelBrand]: void;
  getInputBufferRequirements(signatureIndex: number, inputIndex: number):
      LiteRtTensorBufferRequirements;
  getOutputBufferRequirements(signatureIndex: number, outputIndex: number):
      LiteRtTensorBufferRequirements;
  run(signatureIndex: number, inputTensors: LiteRtTensorHandle[]):
      LiteRtTensorHandle[]|Promise<LiteRtTensorHandle[]>;
  isFullyAccelerated(): boolean;
}

/**
 * ElementType enum representing the types of elements in tensors.
 * Values must match litert::ElementType in
 * litert_model_types.h
 *
 * It is reproduced separately here so it can be used in JS before the Wasm
 * module loads and for typechecking.
 */
export const ElementType = {
  NONE: 0,
  FLOAT32: 1,
  INT32: 2,
  UINT8: 3,
  INT64: 4,
  STRING: 5,
  BOOL: 6,
  INT16: 7,
  COMPLEX64: 8,
  INT8: 9,
  FLOAT16: 10,
  FLOAT64: 11,
  COMPLEX128: 12,
  UINT64: 13,
  RESOURCE: 14,
  VARIANT: 15,
  UINT32: 16,
  UINT16: 17,
  INT4: 18,
  BFLOAT16: 19,
} as const;

/**
 * The type for possible values of a C++ litert::ElementType.
 */
export type ElementType = (typeof ElementType)[keyof typeof ElementType];

/**
 * The keys of the ElementType enum.
 *
 * Used for error messages.
 */
export const ElementTypeName = {
  [ElementType.NONE]: 'NONE',
  [ElementType.FLOAT32]: 'FLOAT32',
  [ElementType.INT32]: 'INT32',
  [ElementType.UINT8]: 'UINT8',
  [ElementType.INT64]: 'INT64',
  [ElementType.STRING]: 'STRING',
  [ElementType.BOOL]: 'BOOL',
  [ElementType.INT16]: 'INT16',
  [ElementType.COMPLEX64]: 'COMPLEX64',
  [ElementType.INT8]: 'INT8',
  [ElementType.FLOAT16]: 'FLOAT16',
  [ElementType.FLOAT64]: 'FLOAT64',
  [ElementType.COMPLEX128]: 'COMPLEX128',
  [ElementType.UINT64]: 'UINT64',
  [ElementType.RESOURCE]: 'RESOURCE',
  [ElementType.VARIANT]: 'VARIANT',
  [ElementType.UINT32]: 'UINT32',
  [ElementType.UINT16]: 'UINT16',
  [ElementType.INT4]: 'INT4',
  [ElementType.BFLOAT16]: 'BFLOAT16',
} as const;

/**
 * The union type of the keys of the ElementType enum.
 */
export type ElementTypeName =
    (typeof ElementTypeName)[keyof typeof ElementTypeName];

/**
 * A C++ enum value in Wasm.
 */
export declare interface EmscriptenEnumElement<T> {
  value: T;
}

type EmscriptenEnum<T extends object> = {
  [K in keyof T]: EmscriptenEnumElement<T[K]>;
};

/**
 * A C++ litert::TensorBufferType enum value.
 *
 * This is reproduced separately here so it can be used in JS before the Wasm
 * module loads and for typechecking (i.e., in tensor.copyTo).
 */
export const TensorBufferType = {
  HOST_MEMORY: 1,
  WEB_GPU_BUFFER: 20,
  WEB_GPU_BUFFER_FP16: 21,
  WEB_GPU_BUFFER_PACKED: 26,
} as const;

/**
 * The type for possible values of a C++ litert::TensorBufferType.
 */
export type TensorBufferType =
    (typeof TensorBufferType)[keyof typeof TensorBufferType];

/**
 * The keys of the TensorBufferType enum.
 *
 * Used for error messages.
 */
export const TensorBufferTypeName = {
  [TensorBufferType.HOST_MEMORY]: 'HOST_MEMORY',
  [TensorBufferType.WEB_GPU_BUFFER]: 'WEB_GPU_BUFFER',
  [TensorBufferType.WEB_GPU_BUFFER_FP16]: 'WEB_GPU_BUFFER_FP16',
  [TensorBufferType.WEB_GPU_BUFFER_PACKED]: 'WEB_GPU_BUFFER_PACKED',
} as const;

/**
 * The union type of the keys of the TensorBufferType enum.
 */
export type TensorBufferTypeName =
    (typeof TensorBufferTypeName)[keyof typeof TensorBufferTypeName];

/**
 * The C++ litert::TensorBufferType enum, containing its values.
 */
type LiteRtTensorBufferTypeEnum = EmscriptenEnum<typeof TensorBufferType>;

/**
 * The type for possible values of a C++ litert::TensorBufferType.
 */
export type LiteRtTensorBufferType =
    LiteRtTensorBufferTypeEnum[keyof LiteRtTensorBufferTypeEnum];

/**
 * A C++ litert::TensorBufferLockMode enum value.
 */
declare interface LiteRtTensorBufferLockModeEnum {
  READ: EmscriptenEnumElement<0>;
  WRITE: EmscriptenEnumElement<1>;
  READ_WRITE: EmscriptenEnumElement<2>;
}

/**
 * The type for possible values of a C++ litert::TensorBufferLockMode.
 */
export type LiteRtTensorBufferLockMode =
    LiteRtTensorBufferLockModeEnum[keyof LiteRtTensorBufferLockModeEnum];

/**
 * The constructor for the C++ litert::tensor::TensorHandle class in Wasm.
 */
export declare interface LiteRtTensorHandleConstructor {
  /**
   * Do not use this constructor. It is only here to allow checking
   * `instanceof LiteRtTensorHandle` in TS. Use `createManaged` instead.
   */
  new(...args: never[]): LiteRtTensorHandle;

  createPlaceholder(
      tensorType: LiteRtRankedTensorType,
      name: string,
      ): LiteRtTensorHandle;

  createManaged(
      environment: LiteRtEnvironment,
      bufferType: LiteRtTensorBufferType,
      tensorType: LiteRtRankedTensorType,
      bufferSize: number,
      ): LiteRtTensorHandle;

  createFromWebGpuBuffer(
      environment: LiteRtEnvironment,
      rankedTensorType: LiteRtRankedTensorType,
      tensorBufferType: LiteRtTensorBufferType,
      webGpuBufferPtr: number, /* use wasm.WebGPU.importJsBuffer() */
      size: number,
      ): LiteRtTensorHandle;
}

/**
 * An instance of the C++ litert::tensor::TensorHandle class in Wasm.
 */
export declare interface LiteRtTensorHandle extends Deletable {
  lock(mode: LiteRtTensorBufferLockMode): number;  // Returns the pointer to the locked buffer.
  unlock(): void;
  bufferType(): LiteRtTensorBufferType;
  tensorType(): LiteRtRankedTensorType;
  isWebGpuMemory(): boolean;
  getWebGpuBuffer(): number;  // Use wasm.WebGPU.getJsObject() to get GPUBuffer.
  size(): number;
  packedSize(): number;
  offset(): number;
}

/** @deprecated Use LiteRtTensorHandle instead. */
export type LiteRtTensorBuffer = LiteRtTensorHandle;
/** @deprecated Use LiteRtTensorHandleConstructor instead. */
export type LiteRtTensorBufferConstructor = LiteRtTensorHandleConstructor;

/**
 * Specification for a signature graph in a multi-signature model.
 */
export interface LiteRtSignatureGraphSpec {
  name: string;
  inputHandles?: LiteRtTensorHandle[];
  inputs?: LiteRtTensorHandle[];
  outputHandles?: LiteRtTensorHandle[];
  outputs?: LiteRtTensorHandle[];
}

/**
 * Interface for the C++ LiteRt bindings.
 */
export declare interface LiteRtWasm extends WasmModule {
  setupLogging(): void;
  LiteRtEnvironment: LiteRtEnvironmentConstructor;
  loadModel(
      environment: LiteRtEnvironment,
      modelDataPtr: number,
      modelSize: number,
      ): LiteRtModel;
  createModelDataFromTensorGraph(
      signatures: LiteRtSignatureGraphSpec[],
      ): {modelDataPtr: number; modelSize: number};
  createModelDataFromTensorGraph(
      inputHandles: LiteRtTensorHandle[],
      outputHandles: LiteRtTensorHandle[],
      ): {modelDataPtr: number; modelSize: number};
  compileModel(
      environment: LiteRtEnvironment,
      model: LiteRtModel,
      options?: LiteRtCompileOptions,
      ): LiteRtCompiledModel|Promise<LiteRtCompiledModel>;
  wgpuBufferRelease(bufferPtr: number): void;
  LiteRtTensorHandle: LiteRtTensorHandleConstructor;
  LiteRtTensorBufferType: LiteRtTensorBufferTypeEnum;
  LiteRtTensorBufferLockMode: LiteRtTensorBufferLockModeEnum;
  LiteRtLayout: LiteRtLayoutConstructor;
  LiteRtRankedTensorType: LiteRtRankedTensorTypeConstructor;
  VectorInt32: EmscriptenVectorInt32Constructor;
  liteRtGetByteWidth(elementType: EmscriptenEnumElement<ElementType>): number;
  WebGPU: WasmWebGpuObjectInterface;
  checkTensorBufferCompatible(
      tensorHandle: LiteRtTensorHandle,
      expectedRankedTensorType: LiteRtRankedTensorType,
      requirements: LiteRtTensorBufferRequirements,
      ): void;
  registerStreamWeightsCallback(callback: Function|undefined): void;
  getStreamWeightsCallback(): Function|undefined;
  getThreadCount(): number;
  add(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  mul(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  sub(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  div(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  relu(a: LiteRtTensorHandle): LiteRtTensorHandle;
  batchMatMul(
    x: LiteRtTensorHandle,
    y: LiteRtTensorHandle,
    adjX?: boolean,
    adjY?: boolean,
  ): LiteRtTensorHandle;
  fullyConnected(
    input: LiteRtTensorHandle,
    weights: LiteRtTensorHandle,
    bias?: LiteRtTensorHandle,
  ): LiteRtTensorHandle;
  softmax(a: LiteRtTensorHandle, beta?: number): LiteRtTensorHandle;
  logistic(a: LiteRtTensorHandle): LiteRtTensorHandle;
  tanh(a: LiteRtTensorHandle): LiteRtTensorHandle;
  gelu(input: LiteRtTensorHandle, approximate?: boolean): LiteRtTensorHandle;
  conv2d(
    input: LiteRtTensorHandle,
    filter: LiteRtTensorHandle,
    bias?: LiteRtTensorHandle,
    strideH?: number,
    strideW?: number,
    padding?: number,
    dilationH?: number,
    dilationW?: number,
  ): LiteRtTensorHandle;
  depthwiseConv2d(
    input: LiteRtTensorHandle,
    filter: LiteRtTensorHandle,
    bias?: LiteRtTensorHandle,
    strideH?: number,
    strideW?: number,
    padding?: number,
    depthMultiplier?: number,
    dilationH?: number,
    dilationW?: number,
  ): LiteRtTensorHandle;
  reshape(input: LiteRtTensorHandle, shape: number[]): LiteRtTensorHandle;
  transpose(input: LiteRtTensorHandle, perm: number[]): LiteRtTensorHandle;
  concatenation(
    inputs: LiteRtTensorHandle[],
    axis: number,
  ): LiteRtTensorHandle;
  slice(
    input: LiteRtTensorHandle,
    begin: number[],
    size: number[],
  ): LiteRtTensorHandle;
  pad(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  mean(
    a: LiteRtTensorHandle,
    axes: number[],
    keepDims: boolean,
  ): LiteRtTensorHandle;
  sum(
    a: LiteRtTensorHandle,
    axes: number[],
    keepDims: boolean,
  ): LiteRtTensorHandle;
  abs(a: LiteRtTensorHandle): LiteRtTensorHandle;
  neg(a: LiteRtTensorHandle): LiteRtTensorHandle;
  sqrt(a: LiteRtTensorHandle): LiteRtTensorHandle;
  rsqrt(a: LiteRtTensorHandle): LiteRtTensorHandle;
  exp(a: LiteRtTensorHandle): LiteRtTensorHandle;
  log(a: LiteRtTensorHandle): LiteRtTensorHandle;
  sin(a: LiteRtTensorHandle): LiteRtTensorHandle;
  cos(a: LiteRtTensorHandle): LiteRtTensorHandle;
  ceil(a: LiteRtTensorHandle): LiteRtTensorHandle;
  floor(a: LiteRtTensorHandle): LiteRtTensorHandle;
  round(a: LiteRtTensorHandle): LiteRtTensorHandle;
  pow(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  minimum(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  maximum(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  floorDiv(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  floorMod(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  equal(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  notEqual(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  less(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  greater(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  greaterEqual(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  logicalAnd(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  logicalOr(a: LiteRtTensorHandle, b: LiteRtTensorHandle): LiteRtTensorHandle;
  logicalNot(a: LiteRtTensorHandle): LiteRtTensorHandle;
  select(
    condition: LiteRtTensorHandle,
    trueVal: LiteRtTensorHandle,
    falseVal: LiteRtTensorHandle,
  ): LiteRtTensorHandle;
  relu6(a: LiteRtTensorHandle): LiteRtTensorHandle;
  leakyRelu(a: LiteRtTensorHandle, alpha?: number): LiteRtTensorHandle;
  elu(a: LiteRtTensorHandle): LiteRtTensorHandle;
  hardSwish(a: LiteRtTensorHandle): LiteRtTensorHandle;
  logSoftmax(a: LiteRtTensorHandle): LiteRtTensorHandle;
  expandDims(input: LiteRtTensorHandle, axis: number): LiteRtTensorHandle;
  squeeze(input: LiteRtTensorHandle, squeezeDims?: number[]): LiteRtTensorHandle;
  tile(input: LiteRtTensorHandle, multiples: number[]): LiteRtTensorHandle;
  pack(tensors: LiteRtTensorHandle[], axis: number): LiteRtTensorHandle;
  unpack(input: LiteRtTensorHandle, num: number, axis: number): LiteRtTensorHandle[];
  split(
    input: LiteRtTensorHandle,
    axis: number,
    numSplits: number,
  ): LiteRtTensorHandle[];
  reduceMax(
    a: LiteRtTensorHandle,
    axes: number[],
    keepDims: boolean,
  ): LiteRtTensorHandle;
  argMax(
    a: LiteRtTensorHandle,
    axis: number,
    outputType?: string,
  ): LiteRtTensorHandle;
  resizeBilinear(
    input: LiteRtTensorHandle,
    size: number[],
    alignCorners: boolean,
    halfPixelCenters: boolean,
  ): LiteRtTensorHandle;
  resizeNearestNeighbor(
    input: LiteRtTensorHandle,
    size: number[],
    alignCorners: boolean,
    halfPixelCenters: boolean,
  ): LiteRtTensorHandle;
  maxPool2d(
    input: LiteRtTensorHandle,
    filterHeight: number,
    filterWidth: number,
    strideH: number,
    strideW: number,
    padding: number,
  ): LiteRtTensorHandle;
  avgPool2d(
    input: LiteRtTensorHandle,
    filterHeight: number,
    filterWidth: number,
    strideH: number,
    strideW: number,
    padding: number,
  ): LiteRtTensorHandle;
  transposeConv2d(
    input: LiteRtTensorHandle,
    filter: LiteRtTensorHandle,
    outputShape: number[],
    bias: LiteRtTensorHandle | undefined,
    strideH: number,
    strideW: number,
    padding: number,
  ): LiteRtTensorHandle;
  gather(
    input: LiteRtTensorHandle,
    indices: LiteRtTensorHandle,
    axis: number,
    batchDims: number,
  ): LiteRtTensorHandle;
  gatherNd(
    input: LiteRtTensorHandle,
    indices: LiteRtTensorHandle,
  ): LiteRtTensorHandle;
  oneHot(
    indices: LiteRtTensorHandle,
    depth: number,
    onValue: number,
    offValue: number,
    axis: number,
  ): LiteRtTensorHandle;
  embeddingLookup(
    weights: LiteRtTensorHandle,
    ids: LiteRtTensorHandle,
  ): LiteRtTensorHandle;
  topK(
    input: LiteRtTensorHandle,
    k: number,
  ): [LiteRtTensorHandle, LiteRtTensorHandle];
  nonMaxSuppressionV5(
    boxes: LiteRtTensorHandle,
    scores: LiteRtTensorHandle,
    maxOutputSize: number,
    iouThreshold: number,
    scoreThreshold: number,
    softNmsSigma: number,
  ): [LiteRtTensorHandle, LiteRtTensorHandle, LiteRtTensorHandle];
}

/**
 * Interface for sharing WebGPU objects between WASM and the JS side.
 */
export declare interface WasmWebGpuObjectInterface {
  getJsObject(id: number): GPUBuffer;
  importJsBuffer(buffer: GPUBuffer): number;
}
