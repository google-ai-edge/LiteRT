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

import type {Accelerator, DType} from './constants';
import type {GPUDeviceWithAdapterInfo} from './gpu_utils';

// These external interfaces that are implemented by the C++ code must use the
// `declare` keyword to prevent the JS Compiler from renaming them.

/**
 * An interface for objects that can be deleted.
 */
export declare interface Deletable {
  delete(): void;
}

/**
 * A C++ vector of elements.
 */
export declare interface EmscriptenVector<T> {
  size(): number;
  get(index: number): T&Deletable;
  delete(): void;
}

/**
 * A C++ vector of elements that can be written to.
 */
export declare interface WritableEmscriptenVector<T> extends
    EmscriptenVector<T> {
  push_back(elem: T): void;
  delete(): void;
}

/**
 * A wrapper around the C++ LiteRt Tensor owned by a specific interpreter.
 */
export declare interface TensorWrapper {
  // In C++, this class has a pointer to a TfLiteTensor associated with a
  // particular TfLite Interpreter. Additionally, it may have a pointer to the
  // MLDrift SpatialTensor if it is a GPU tensor. The memory behind these
  // pointers may change whenever the interpreter is run (as part of the setup
  // or execution of the interpreter).
  //
  // Consequently, this should never be exposed to the user. Instead, JS classes
  // should copy Tensors to or from TensorWrappers as needed.
  type(): DType;
  name(): string;
  shape(): Int32Array;
  accelerator(): Accelerator;
  delete(): void;
}
/**
 * An opaque reference to MLDrift GPU memory or WASM CPU memory that backs a
 * tensor. It is not associated with a specific interpreter.
 */
export declare interface OpaqueTensorReference {
  // This interface differs from TensorWrapper in that it is not associated
  // with a specific TfLite Interpreter / Tensor. We assume that any
  // OpaqueTensorReference that is exposed to JavaScript is free to be used with
  // any LiteRt interpreter and will not be overwritten or deleted when an
  // interpreter is run.
  //
  // This should always be associated with a Tensor, which has metadata that the
  // TfLite Tensor we no longer have access to would otherwise hold.
  //
  // We don't expose which accelerator this is stored on here because instances
  // of this can be backed by any type in C++. Store the accelerator on the
  // Tensor instead.
  //
  // Using 'declare' in the declaration to avoid the bundler renaming
  // properties.

  // This helps TypeScript prevent accidentally assigning a Tensor (or some
  // other type) to an OpaqueTensorReference, however, it will not actually be
  // present in real instances of this interface.
  _do_not_use_me_tag: Symbol;
  delete?(): void;  // Some refs must be deleted, others are GCd.
}

type DriftTensorCacheKey =
    [number, number, number, number, number, number, number, number];

/**
 * A reference to MLDrift GPU memory that backs a tensor. It is not associated
 * with a specific interpreter.
 */
export declare interface DriftTensor extends OpaqueTensorReference {
  getCacheKey(): DriftTensorCacheKey;
}

/**
 * A reference to CPU memory that backs a tensor. It is not associated with a
 * specific interpreter.
 */
export declare interface CpuTensorReference extends OpaqueTensorReference {
  data(): Uint8Array;
  size(): number;
}

/**
 * A wrapper around the C++ LiteRt SignatureRunner owned by a specific
 * interpreter.
 */
export declare interface SignatureRunnerWrapper {
  inputs(): EmscriptenVector<TensorWrapper>;
  outputs(): EmscriptenVector<TensorWrapper>;
  invoke(): void;
  copyInputs(srcs: EmscriptenVector<OpaqueTensorReference>): void;
  copyOutputs(): EmscriptenVector<OpaqueTensorReference>;
  makeTensorVector(): WritableEmscriptenVector<OpaqueTensorReference>;
  // Do not `delete()` this. It's passed by reference and freed on the C++ side.
}

/**
 * Interface for the C++ LiteRt interpreter.
 */
export declare interface LiteRtInterpreter extends SignatureRunnerWrapper {
  listSignatures(): EmscriptenVector<string>;
  getSignatureRunner(signatureKey: string): SignatureRunnerWrapper;
  delete(): void;
}

/**
 * A custom error reporting function for receiving errors from LiteRt.
 * @param message The error message from LiteRt.
 * @returns The error object that will be thrown. For correct control flow,
 *     LiteRt will always throw an error, but this function can be used to
 *     change the error message or type.
 */
export type ErrorReporter = (message: string) => Error;

/**
 * Interface for the C++ LiteRt bindings.
 */
export declare interface LiteRtWasm extends WasmModule, WebgpuConversionWasm {
  loadAndCompileWebGpu(modelDataPtr: number, modelSize: number):
      LiteRtInterpreter;
  loadAndCompileCpu(modelDataPtr: number, modelSize: number): LiteRtInterpreter;
  setupLogging(): void;
  setErrorReporter(errorReporter: ErrorReporter): void;
  preinitializedWebGPUDevice: GPUDeviceWithAdapterInfo;
}

/**
 * Interface for the C++ WebGPU conversion bindings.
 */
export declare interface WebgpuConversionWasm {
  makeConverterFromTfjs(
      type: string, b: number, h: number, w: number,
      c: number): NativeInputConverter;

  // TODO(msoulanille): I think we make a new one of these every time. We
  // should cache them. Do we then need to send an initial example tensor to
  // initialize the converter?
  makeConverterToTfjs(tensorReference: OpaqueTensorReference):
      NativeOutputConverter;

  WebGPU: WasmWebGpuObjectInterface;
  preinitializedWebGPUDevice: GPUDevice;
  CpuTensor: {new(size: number): CpuTensorReference;};
}

/**
 * Converts a LiteRT DriftTensor to a WebGPU buffer in TF.js tensor format.
 */
export declare interface NativeOutputConverter {
  convertToTfjs(tensorRef: OpaqueTensorReference): number;
  delete(): void;
}

/**
 * Converts a WebGPU buffer in TF.js tensor format to a LiteRT DriftTensor.
 */
export declare interface NativeInputConverter {
  convertFromTfjs(tfjsBufferPtr: number): OpaqueTensorReference;
  delete(): void;
}

/**
 * Interface for sharing WebGPU objects between WASM and the JS side.
 */
export declare interface WasmWebGpuObjectInterface {
  getJsObject(id: number): GPUBuffer;
  importJsBuffer(buffer: GPUBuffer): number;
}
