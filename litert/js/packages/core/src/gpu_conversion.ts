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

import {Box, GpuErrorReporter, popErrorScopes, pushErrorScopes} from './gpu_utils';
import {memoize} from './memoize';
import type {DriftTensor, NativeInputConverter, NativeOutputConverter, OpaqueTensorReference, WebgpuConversionWasm} from './wasm_binding_types';

/**
 * Converter from WebGPU buffers in TF.js tensor format to LiteRT Tensors.
 * Can be made using the ConverterFactory class.
 */
export class InputConverter {
  constructor(
      private readonly converter: NativeInputConverter,
      private readonly wasm: WebgpuConversionWasm,
      private readonly gpuErrorReporter: Box<GpuErrorReporter>) {}

  convertFromTfjs(buffer: GPUBuffer): OpaqueTensorReference {
    pushErrorScopes(this.wasm.preinitializedWebGPUDevice);
    const bufferPtr = this.wasm.WebGPU.importJsBuffer(buffer);
    const tensor = this.converter.convertFromTfjs(bufferPtr);
    popErrorScopes(
        this.wasm.preinitializedWebGPUDevice, 'convertFromTfjs',
        this.gpuErrorReporter.val);
    return tensor;
  }

  delete() {
    this.converter.delete();
  }
}

/**
 * Converter from LiteRT Tensors to WebGPU buffers in TF.js tensor format.
 * Can be made using the ConverterFactory class.
 */
export class OutputConverter {
  constructor(
      private readonly converter: NativeOutputConverter,
      private readonly wasm: WebgpuConversionWasm,
      private readonly gpuErrorReporter: Box<GpuErrorReporter>) {}

  convertToTfjs(tensor: OpaqueTensorReference): GPUBuffer {
    pushErrorScopes(this.wasm.preinitializedWebGPUDevice);
    const bufferPtr = this.converter.convertToTfjs(tensor);
    const buffer = this.wasm.WebGPU.getJsObject(bufferPtr);
    popErrorScopes(
        this.wasm.preinitializedWebGPUDevice, 'convertToTfjs',
        this.gpuErrorReporter.val);
    return buffer;
  }

  delete() {
    this.converter.delete();
  }
}

/**
 * Factory for LiteRT WebGPU-based tensor converters.
 */
export class ConverterFactory {
  constructor(
      readonly wasm: WebgpuConversionWasm,
      private readonly gpuErrorReporter: Box<GpuErrorReporter>) {}

  /**
   * Returns true if this ConverterFactory uses the same WebGPU device as the
   * one passed in.
   */
  isWebGpuDeviceCompatible(device: GPUDevice) {
    return (device === this.wasm.preinitializedWebGPUDevice);
  }

  /**
   * Returns an InputConverter for quickly converting WebGPU buffers in TF.js
   * tensor format into the corresponding LiteRT Tensors. Each InputConverter is
   * created for a given type and [B,H,W,C] shape, so the converter can be
   * reused, but only for tensors of the same type and shape.
   */
  makeConverterFromTfjs =
      memoize(this.makeConverterFromTfjsInternal.bind(this));

  private makeConverterFromTfjsInternal(
      type: string, b: number, h: number, w: number,
      c: number): InputConverter {
    pushErrorScopes(this.wasm.preinitializedWebGPUDevice);
    const nativeConverter = this.wasm.makeConverterFromTfjs(type, b, h, w, c);
    popErrorScopes(
        this.wasm.preinitializedWebGPUDevice, 'makeConverterFromTfjs',
        this.gpuErrorReporter.val);
    return new InputConverter(
        nativeConverter, this.wasm, this.gpuErrorReporter);
  }

  /**
   * Returns an OutputConverter for quickly converting LiteRT Tensors into the
   * the corresponding WebGPU buffer in TF.js tensor format. Each
   * OutputConverter is created to match the specifications of the given Tensor
   * (type and [B,H,W,C] shape), so the converter can be reused, but only for
   * Tensors of the same type and shape.
   */
  makeConverterToTfjs = memoize(
      this.makeConverterToTfjsInternal.bind(this), ([opaqueReference]) => {
        const driftTensor = opaqueReference as DriftTensor;
        return driftTensor.getCacheKey();
      });

  private makeConverterToTfjsInternal(tensorReference: OpaqueTensorReference):
      OutputConverter {
    pushErrorScopes(this.wasm.preinitializedWebGPUDevice);
    const nativeConverter = this.wasm.makeConverterToTfjs(tensorReference);
    popErrorScopes(
        this.wasm.preinitializedWebGPUDevice, 'makeConverterToTfjs',
        this.gpuErrorReporter.val);
    return new OutputConverter(
        nativeConverter, this.wasm, this.gpuErrorReporter);
  }
}
