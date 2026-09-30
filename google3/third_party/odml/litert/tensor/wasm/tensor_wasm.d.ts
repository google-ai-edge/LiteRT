/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * Supported LiteRT Tensor Types mapping identically to flatbuffer conventions.
 */
export const enum TensorType {
  UNKNOWN = 0,
  BOOL = 1,
  I8 = 4,
  I32 = 6,
  FP32 = 14,
}

/**
 * Core Tensor interface exposing zero-copy buffer access, attributes, and
 * operations.
 */
export interface Tensor {
  // Attributes and Accessors
  getName(): string;
  getType(): TensorType;
  getShape(): {size(): number; get(index: number): number};
  getData(): Promise<Float32Array|Int32Array|Int8Array|Uint8Array|null>;
  getMutableData(): Promise<Float32Array|Int32Array|Int8Array|Uint8Array|null>;
  getDataPointer(): number;
  getWebGpuBuffer(): number;
  getQuantization():
      {scales: number[]; zeroPoints: number[]; quantizedDimension: number};

  setName(name: string): void;
  setType(type: TensorType): void;
  setShape(shape: number[]): void;
  setQuantization(params: {
    scales: number[]; zeroPoints: number[]; quantizedDimension: number
  }): void;
  setData(data: Float32Array|Int32Array|Int8Array|Uint8Array): void;

  // Unary & Activation Operations
  abs(): Tensor;
  relu(): Tensor;
  relu6(): Tensor;
  elu(): Tensor;
  hardSwish(): Tensor;
  logSoftmax(): Tensor;
  logistic(): Tensor;
  neg(): Tensor;
  sqrt(): Tensor;
  cos(): Tensor;
  sin(): Tensor;
  exp(): Tensor;
  log(): Tensor;
  ceil(): Tensor;
  floor(): Tensor;
  sign(): Tensor;
  round(): Tensor;
  logicalNot(): Tensor;

  // Binary & Logic Operations
  add(other: Tensor): Tensor;
  sub(other: Tensor): Tensor;
  mul(other: Tensor): Tensor;
  div(other: Tensor): Tensor;
  pow(other: Tensor): Tensor;
  minimum(other: Tensor): Tensor;
  maximum(other: Tensor): Tensor;
  less(other: Tensor): Tensor;
  greater(other: Tensor): Tensor;
  lessEqual(other: Tensor): Tensor;
  greaterEqual(other: Tensor): Tensor;
  equal(other: Tensor): Tensor;
  notEqual(other: Tensor): Tensor;
  logicalAnd(other: Tensor): Tensor;
  logicalOr(other: Tensor): Tensor;
  floorDiv(other: Tensor): Tensor;
  floorMod(other: Tensor): Tensor;
  batchMatMul(other: Tensor, adj_x?: boolean, adj_y?: boolean): Tensor;

  // Reduction & Shaping Operations
  sum(axes: number[], keepDims: boolean): Tensor;
  reduceMax(axes: number[], keepDims: boolean): Tensor;
  mean(axes: number[], keepDims: boolean): Tensor;
  cumsum(axis: number, exclusive: boolean, reverse: boolean): Tensor;

  expandDims(axis: number): Tensor;
  squeeze(dims: number[]): Tensor;
  reshape(shape: number[]): Tensor;
  pad(padTensor: Tensor): Tensor;

  // Spatial, Decoding & Extractor Operations
  averagePool2d(
      filterHeight: number, filterWidth: number, strideH: number,
      strideW: number, padding: number): Tensor;
  maxPool2d(
      filterHeight: number, filterWidth: number, strideH: number,
      strideW: number, padding: number): Tensor;
  fullyConnected(weights: Tensor): Tensor;
  select(condition: Tensor, other: Tensor): Tensor;
  selectV2(condition: Tensor, other: Tensor): Tensor;
  concatenation(others: Tensor[], axis: number): Tensor;
  pack(others: Tensor[], axis: number): Tensor;
  slice(begin: number[], size: number[]): Tensor;
  tile(multiples: number[]): Tensor;
  transpose(perm: number[]): Tensor;

  spaceToDepth(blockSize: number): Tensor;
  depthToSpace(blockSize: number): Tensor;
  reverse(axes: Tensor): Tensor;
  resizeBilinear(
      size: number[], alignCorners: boolean, halfPixelCenters: boolean): Tensor;
  resizeNearestNeighbor(
      size: number[], alignCorners: boolean, halfPixelCenters: boolean): Tensor;

  unpack(num: number, axis: number): Tensor[];
  split(axis: Tensor, numSplits: number): Tensor[];
  gather(indices: Tensor, axis: number): Tensor;
  gatherNd(indices: Tensor): Tensor;
  oneHot(depth: Tensor, onValue: Tensor, offValue: Tensor, axis: number):
      Tensor;
  argMax(axis: number): Tensor;
  topK(k: number): Tensor[];

  gelu(): Tensor;
  embeddingLookup(ids: Tensor, outputType: TensorType): Tensor;
  dynamicUpdateSlice(update: Tensor, startIndices: Tensor): Tensor;
  nonMaxSuppressionV5(
      scores: Tensor, maxOutputSize: Tensor, iouThreshold: Tensor,
      scoreThreshold: Tensor, softNmsSigma: Tensor): Tensor[];

  // Deallocate resource
  delete(): void;
}

/**
 * Execution Runner interfaces mapping dynamic framework buffers.
 */
export interface CompiledModelRunner {
  run(): Promise<boolean>;
  delete(): void;
}

export interface LambdaModelRunner {
  run(): Promise<boolean>;
  setInput(name: string, tensor: Tensor): boolean;
  setInputBinary(name: string, array: Uint8Array): boolean;
  getInput(name: string): Tensor;
  getOutput(name: string): Tensor;
  isNull(): boolean;
  delete(): void;
}

export interface LitertDynamicRunner {
  run(): Promise<boolean>;
  getInput(name: string): Tensor;
  getOutput(name: string): Tensor;
  getInputByIndex(index: number): Tensor;
  getOutputByIndex(index: number): Tensor;
  setInput(name: string, tensor: Tensor): boolean;
  setInputBinary(name: string, array: Uint8Array): boolean;
  isNull(): boolean;
  delete(): void;
}

export interface LiteRTRunner {
  run(): Promise<boolean>;
  getInput(nameOrIndex: string|number): Tensor;
  getOutput(nameOrIndex: string|number): Tensor;
  setInput(name: string, tensor: Tensor): boolean;
  setInputBinary(name: string, array: Uint8Array): boolean;
  delete(): void;
}

/**
 * Zero-copy WebGPU Staging buffer facade.
 */
export interface WebGpuBuffer {
  setGPUBuffer(deviceBuffer: any): boolean;
  getGPUBuffer(): any;
  delete(): void;
}

/**
 * LiteRT WebAssembly module instance namespace interface.
 */
export interface LiteRtWasmModule {
  TensorType: typeof TensorType;
  Tensor: {new(): Tensor;};
  CompiledModelRunner: {new(): CompiledModelRunner;};
  LambdaModelRunner: {new(): LambdaModelRunner;};
  LitertDynamicRunner: {new(): LitertDynamicRunner;};
  WebGpuBuffer: {new(): WebGpuBuffer;};

  createStaticLambdaRunner(inputs: any, outputs: any): LambdaModelRunner;
  createDynamicRunnerFromBuffer(buffer: Uint8Array, accelerators: number):
      LitertDynamicRunner;

  // Sugar methods from wrapper
  createTensor(
      options: {name?: string, type?: string|TensorType, shape?: number[]}):
      Tensor;
  createGraphRunner(inputs: any, outputs: any, accelerators?: number|{
    value: number
  }): LiteRTRunner;
  createModelRunner(buffer: Uint8Array, accelerators?: number|{value: number}):
      LiteRTRunner;

  HwAccelerators: {CPU: {value: number}; GPU: {value: number};};
}

/**
 * Emscripten default factory loader returning the compiled sandbox instance.
 */
export default function createTensorModule(options?: any):
    Promise<LiteRtWasmModule>;

export function createLiteRT(options?: any): Promise<LiteRtWasmModule>;
export function wrapLiteRTModule(module: any): LiteRtWasmModule;
