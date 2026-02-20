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
 * Options for loading and compiling a LiteRt model.
 */
export declare interface LiteRtCompileOptions {
  accelerator?: 'wasm'|'webgpu'|'webnn';
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
  run(signatureIndex: number,
      inputTensors: LiteRtTensorBuffer[]): LiteRtTensorBuffer[];
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
 * The constructor for the C++ litert::TensorBuffer class in Wasm.
 */
export declare interface LiteRtTensorBufferConstructor {
  /**
   * Do not use this constructor. It is only here to allow checking
   * `instanceof LiteRtTensorBuffer` in TS. Use `createManaged` instead.
   */
  new(...args: never[]): LiteRtTensorBuffer;

  createManaged(
      environment: LiteRtEnvironment,
      bufferType: LiteRtTensorBufferType,
      tensorType: LiteRtRankedTensorType,
      bufferSize: number,
      ): LiteRtTensorBuffer;

  createFromWebGpuBuffer(
      environment: LiteRtEnvironment,
      rankedTensorType: LiteRtRankedTensorType,
      tensorBufferType: LiteRtTensorBufferType,
      webGpuBufferPtr: number, /* use wasm.WebGPU.importJsBuffer() */
      size: number,
      ): LiteRtTensorBuffer;
}

/**
 * An instance of the C++ litert::TensorBuffer class in Wasm.
 */
export declare interface LiteRtTensorBuffer extends Deletable {
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
  compileModel(
      environment: LiteRtEnvironment,
      model: LiteRtModel,
      options?: LiteRtCompileOptions,
      ): LiteRtCompiledModel|Promise<LiteRtCompiledModel>;
  wgpuBufferRelease(bufferPtr: number): void;
  LiteRtTensorBuffer: LiteRtTensorBufferConstructor;
  LiteRtTensorBufferType: LiteRtTensorBufferTypeEnum;
  LiteRtTensorBufferLockMode: LiteRtTensorBufferLockModeEnum;
  LiteRtLayout: LiteRtLayoutConstructor;
  LiteRtRankedTensorType: LiteRtRankedTensorTypeConstructor;
  VectorInt32: EmscriptenVectorInt32Constructor;
  liteRtGetByteWidth(elementType: EmscriptenEnumElement<ElementType>): number;
  WebGPU: WasmWebGpuObjectInterface;
  checkTensorBufferCompatible(
      tensorBuffer: LiteRtTensorBuffer,
      expectedRankedTensorType: LiteRtRankedTensorType,
      requirements: LiteRtTensorBufferRequirements,
      ): void;
  getThreadCount(): number;
}

/**
 * Interface for sharing WebGPU objects between WASM and the JS side.
 */
export declare interface WasmWebGpuObjectInterface {
  getJsObject(id: number): GPUBuffer;
  importJsBuffer(buffer: GPUBuffer): number;
}
