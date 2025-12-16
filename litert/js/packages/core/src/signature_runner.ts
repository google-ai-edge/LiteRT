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

import {DType, getDataType} from './datatypes';
import {getGlobalLiteRt} from './global_litert';
import {CompileOptions, TensorInputs} from './model_types';
import {Tensor} from './tensor';
import {LiteRtCompiledModel, LiteRtModel, LiteRtRankedTensorType, LiteRtSimpleSignature, LiteRtTensorBufferRequirements, TensorBufferType} from './wasm_binding_types';
import {emscriptenVectorToArray} from './wasm_utils';

/**
 * Description of a tensor in a LiteRT model.
 */
export interface TensorDetails {
  readonly name: string;
  readonly index: number;
  readonly dtype: DType;
  readonly shape: Int32Array;
  readonly supportedBufferTypes: ReadonlySet<TensorBufferType>;
}

/**
 * A signature of a LiteRT model that can be run like the CompiledModel itself.
 * Every model has at least one signature, but some models have multiple
 * signatures.
 */
export interface SignatureRunner {
  readonly key: string;
  readonly options: Required<CompileOptions>;
  getInputDetails(): readonly TensorDetails[];
  getOutputDetails(): readonly TensorDetails[];

  run(input: Tensor|Tensor[]): Promise<Tensor[]>;
  run(input: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  run(input: Tensor|Tensor[]|
      Record<string, Tensor>): Promise<Tensor[]|Record<string, Tensor>>;
}

/**
 * A signature of a compiled LiteRT model.
 */
export class CompiledModelSignatureRunner implements SignatureRunner {
  private readonly inputDetails: readonly TensorDetails[];
  private readonly outputDetails: readonly TensorDetails[];
  private readonly liteRtSimpleSignature: LiteRtSimpleSignature;
  private deletedInternal = false;

  constructor(
      private readonly signatureIndex: number,
      private readonly liteRtModel: LiteRtModel,
      private readonly liteRtCompiledModel: LiteRtCompiledModel,
      readonly options: Required<CompileOptions>) {
    this.liteRtSimpleSignature = liteRtModel.getSignature(signatureIndex);

    // Input details
    const inputNames =
        emscriptenVectorToArray(this.liteRtSimpleSignature.inputNames());
    const inputDetails: TensorDetails[] = [];
    for (let i = 0; i < inputNames.length; i++) {
      const name = inputNames[i];
      const tensorType = liteRtModel.getInputTensorType(signatureIndex, i);
      const requirements =
          liteRtCompiledModel.getInputBufferRequirements(signatureIndex, i);
      inputDetails.push(makeTensorDetails(name, i, tensorType, requirements));
    }
    this.inputDetails = Object.freeze(inputDetails);

    // Output details
    const outputNames =
        emscriptenVectorToArray(this.liteRtSimpleSignature.outputNames());
    const outputDetails: TensorDetails[] = [];
    for (let i = 0; i < outputNames.length; i++) {
      const name = outputNames[i];
      const tensorType = liteRtModel.getOutputTensorType(signatureIndex, i);
      const requirements =
          liteRtCompiledModel.getOutputBufferRequirements(signatureIndex, i);
      outputDetails.push(makeTensorDetails(name, i, tensorType, requirements));
    }
    this.outputDetails = Object.freeze(outputDetails);
  }

  /**
   * The string key corresponding to this signature in the model.
   */
  get key(): string {
    this.ensureNotDeleted();
    return this.liteRtSimpleSignature.key();
  }

  /**
   * Get details about each input tensor.
   */
  getInputDetails(): readonly TensorDetails[] {
    this.ensureNotDeleted();
    return this.inputDetails;
  }

  /**
   * Get details about each output tensor.
   */
  getOutputDetails(): readonly TensorDetails[] {
    this.ensureNotDeleted();
    return this.outputDetails;
  }

  /**
   * Runs the signature with positional input tensors.
   * @param input The input tensors.
   * @return The output tensors.
   */
  run(input: Tensor|Tensor[]): Promise<Tensor[]>;
  /**
   * Runs the signature with named input tensors.
   * @param input A record of named input tensors.
   * @return A record of named output tensors.
   */
  run(input: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  // Typescript won't automatically distribute a union over the above
  // signatures, so we need to explicitly declare it below.
  // https://github.com/microsoft/TypeScript/issues/14107
  run(input: Tensor|Tensor[]|
      Record<string, Tensor>): Promise<Tensor[]|Record<string, Tensor>>;
  async run(input: TensorInputs) {
    this.ensureNotDeleted();
    const inputArray = this.inputsToArray(input);
    const {inputsOnAccelerator, cleanup} =
        await this.ensureInputsOnAccelerator(inputArray);

    let outputArray: Tensor[]|undefined;
    try {
      outputArray = this.runWithArray(inputsOnAccelerator);
    } finally {
      cleanup();
    }

    if (Array.isArray(input) || input instanceof Tensor) {
      return outputArray;
    } else {
      return this.outputsToRecord(outputArray);
    }
  }

  private inputsToArray(input: TensorInputs): Tensor[] {
    if (Array.isArray(input)) {
      if (input.length !== this.inputDetails.length) {
        throw new Error(
            `run() called with ${input.length} ` +
            `inputs, but signature expects ${this.inputDetails.length} inputs`);
      }
      return input;
    }
    if (input instanceof Tensor) {
      if (this.inputDetails.length !== 1) {
        throw new Error(
            `run() called with a single tensor, but signature expects ${
                this.inputDetails.length} inputs`);
      }
      return [input];
    }
    // Must insert in the same order as the inputDetails.
    const inputArray: Tensor[] = [];
    for (const inputDetails of this.inputDetails) {
      if (!(inputDetails.name in input)) {
        throw new Error(
            `run() called with input record that is missing ` +
            `input ${inputDetails.name} with index ${inputDetails.index}`);
      }
      inputArray.push(input[inputDetails.name]);
    }
    return inputArray;
  }

  private outputsToRecord(output: Tensor[]): Record<string, Tensor> {
    const outputRecord: Record<string, Tensor> = {};
    for (let i = 0; i < this.outputDetails.length; i++) {
      outputRecord[this.outputDetails[i].name] = output[i];
    }
    return outputRecord;
  }

  /**
   * Ensures that all input tensors are on the correct accelerator. Copies any
   * tensors that are not on the correct accelerator.
   *
   * @param inputs The input tensors to be passed to the signature. They must
   *     be in the same order and quantity as the input details.
   * @return A promise that resolves to a list of input tensors that are on the
   *     correct accelerator, and a cleanup function that deletes any tensors
   *     that were copied.
   */
  private async ensureInputsOnAccelerator(inputs: Tensor[]): Promise<{
    inputsOnAccelerator: Tensor[];
    cleanup: () => void;  // Deletes any copies of tensors that were made.
  }> {
    const toDelete: Tensor[] = [];
    const inputsOnAccelerator: Tensor[] = [];
    const inputDetails = this.getInputDetails();

    if (inputs.length !== inputDetails.length) {
      throw new Error(`ensureInputsOnAccelerator() called with ${
          inputs.length} inputs, but signature expects ${
          inputDetails.length} inputs`);
    }

    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      const bufferType = input.getBufferType();
      const supportedBufferTypes = inputDetails[i].supportedBufferTypes;
      if (supportedBufferTypes.size === 0) {
        throw new Error(`Tensor ${inputDetails[i].name} with index ${
            inputDetails[i].index} has no supported buffer types.`);
      }
      if (supportedBufferTypes.has(bufferType)) {
        inputsOnAccelerator.push(input);
      } else {
        const newBufferType = supportedBufferTypes.values().next().value;
        const copy = await input.copyTo(newBufferType);
        toDelete.push(copy);
        inputsOnAccelerator.push(copy);
      }
    }
    return {
      inputsOnAccelerator,
      cleanup: () => {
        for (const tensor of toDelete) {
          tensor.delete();
        }
      },
    };
  }

  private runWithArray(input: Tensor[]): Tensor[] {
    // b/458345985: When on WebGPU, will need to check stride & alignment
    for (let i = 0; i < input.length; i++) {
      const inputTensor = input[i];
      const expectedRankedTensorType =
          this.liteRtModel.getInputTensorType(this.signatureIndex, i);
      const inputRequirements =
          this.liteRtCompiledModel.getInputBufferRequirements(
              this.signatureIndex, i);
      getGlobalLiteRt().liteRtWasm.checkTensorBufferCompatible(
          inputTensor.liteRtTensorBuffer, expectedRankedTensorType,
          inputRequirements);

      expectedRankedTensorType.delete();
      inputRequirements.delete();
    }

    const outputTensorBuffers = this.liteRtCompiledModel.run(
        this.signatureIndex, input.map((tensor) => tensor.liteRtTensorBuffer));
    return outputTensorBuffers.map(
        (tensorBuffer) => new Tensor(tensorBuffer, this.options.environment));
  }

  get deleted(): boolean {
    return this.deletedInternal;
  }

  private ensureNotDeleted() {
    if (this.deleted) {
      throw new Error(
          'CompiledModelSignatureRunner is deleted and cannot be used.');
    }
  }

  delete() {
    if (this.deletedInternal) {
      return;
    }
    this.deletedInternal = true;
    this.liteRtSimpleSignature.delete();
  }
}

/**
 * Creates a TensorDetails object from the given information.
 *
 * Deletes the input LiteRtRankedTensorType and LiteRtTensorBufferRequirements
 * objects.
 */
function makeTensorDetails(
    name: string, index: number, tensorType: LiteRtRankedTensorType,
    requirements: LiteRtTensorBufferRequirements): TensorDetails {
  const layout = tensorType.layout();
  const dimensions = emscriptenVectorToArray(layout.dimensions());
  layout.delete();
  const supportedBufferTypes =
      new Set(emscriptenVectorToArray(requirements.supportedTypes())
                  .map(({value}) => value));

  const details: TensorDetails = {
    name,
    index,
    dtype: getDataType(tensorType.elementType().value).dtype,
    shape: new Int32Array(dimensions),
    supportedBufferTypes,
  };
  tensorType.delete();
  requirements.delete();
  return details;
}