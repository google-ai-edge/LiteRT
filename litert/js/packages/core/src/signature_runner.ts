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

import type {Accelerator} from './constants';
import {Tensor, TensorTypeError} from './tensor';
import type {EmscriptenVector, SignatureRunnerWrapper, TensorWrapper} from './wasm_binding_types';

/**
 * A signature of a LiteRT model that can be run like the CompiledModel itself.
 * Every model has at least one signature, but some models have multiple
 * signatures.
 */
export abstract class SignatureRunner {
  protected inputTensors = new Map<string, TensorWrapper>();
  protected inputTensorsVector: EmscriptenVector<TensorWrapper>;
  protected outputTensors = new Map<string, TensorWrapper>();
  protected outputTensorsVector: EmscriptenVector<TensorWrapper>;
  abstract readonly accelerator: Accelerator;
  deleted = false;

  constructor(
      protected readonly signatureRunnerWrapper: SignatureRunnerWrapper) {
    this.inputTensorsVector = this.signatureRunnerWrapper.inputs();
    for (let i = 0; i < this.inputTensorsVector.size(); ++i) {
      const tensor = this.inputTensorsVector.get(i);
      this.inputTensors.set(tensor.name(), tensor);
    }

    this.outputTensorsVector = this.signatureRunnerWrapper.outputs();
    for (let i = 0; i < this.outputTensorsVector.size(); ++i) {
      const tensor = this.outputTensorsVector.get(i);
      this.outputTensors.set(tensor.name(), tensor);
    }
  }

  private checkTypes(inputs: Tensor[]): void {
    const inputTensorsList = [...this.inputTensors.values()];
    for (let i = 0; i < inputTensorsList.length; ++i) {
      const tensorWrapper = inputTensorsList[i];
      const tensor = inputs[i];
      const expectedDType = tensorWrapper.type();
      if (expectedDType !== tensor.type.dtype) {
        throw new TensorTypeError(
            tensorWrapper.name(), i, expectedDType, tensor.type.dtype);
      }
    }
  }

  /**
   * Runs the signature with the given input tensors and returns the outputs.
   */
  run(input: Tensor|Tensor[]): Tensor[];
  run(input: Record<string, Tensor>): Record<string, Tensor>;
  // Typescript won't automatically distribute a union over the above
  // signatures, so we need to explicitly declare it below.
  // https://github.com/microsoft/TypeScript/issues/14107
  run(input: Tensor|Tensor[]|
      Record<string, Tensor>): Tensor[]|Record<string, Tensor>;
  run(input: Tensor|Tensor[]|
      Record<string, Tensor>): Tensor[]|Record<string, Tensor> {
    if (this.deleted) {
      throw new Error('Signature has been deleted. Please reload the model.');
    }
    let inputArray: Tensor[];
    let shouldReturnArray = true;
    if (Array.isArray(input)) {
      if (input.length !== this.inputTensors.size) {
        throw new Error(
            `run() called with ${input.length} ` +
            `inputs, but signature expects ${this.inputTensors.size} inputs`);
      }
      inputArray = input;
    } else if (input instanceof Tensor) {
      if (this.inputTensors.size !== 1) {
        throw new Error(
            `run() called with a single tensor, but signature expects ${
                this.inputTensors.size} inputs`);
      }
      inputArray = [input];
    } else {
      shouldReturnArray = false;
      // Must insert in the same order as the inputTensors map.
      inputArray = [];
      for (const name of this.inputTensors.keys()) {
        const tensor = input[name];
        if (!tensor) {
          throw new Error(`Expected input tensor with name '${
              name}', but none was provided.`);
        }
        inputArray.push(tensor);
      }
    }

    this.checkTypes(inputArray);
    const outputArray = this.runWithArray(inputArray);

    // In most cases, we return an array of tensors.
    if (shouldReturnArray) {
      return outputArray;
    }

    // If the input was a record, we need to return a record of outputs.
    const output: Record<string, Tensor> = {};

    // This order of outputTensors.keys() is the same as in the
    // outputTensorsVector (see the constructor) because `Map.prototype.keys`
    // iterates in insertion order.
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map
    const names = [...this.outputTensors.keys()];
    for (let i = 0; i < names.length; i++) {
      output[names[i]] = outputArray[i];
    }
    return output;
  }

  protected pushErrorScopes() {
    // Only implemented by subclasses that use the GPU.
  }

  protected popErrorScopes(callsite: string) {
    // Only implemented by subclasses that use the GPU.
  }

  /**
   * Runs the default signature of the model with the given input tensors and
   * returns the outputs.
   */
  protected runWithArray(input: Tensor[]): Tensor[] {
    const tensors = this.signatureRunnerWrapper.makeTensorVector();
    for (const tensor of input) {
      // TODO: Assert this is a DriftTensor.
      tensors.push_back(tensor.reference);
    }

    // Perform input copy.
    this.pushErrorScopes();
    this.signatureRunnerWrapper.copyInputs(tensors);
    this.popErrorScopes('copyInputs');

    tensors.delete();

    // Actually run the signature.
    this.pushErrorScopes();
    this.signatureRunnerWrapper.invoke();
    this.popErrorScopes('invoke');

    // Copy tensors from the interpreter.
    this.pushErrorScopes();
    const outputTensorReferences = this.signatureRunnerWrapper.copyOutputs();
    this.popErrorScopes('copyOutputs');

    const output: Tensor[] = [];
    for (let i = 0; i < this.outputTensorsVector.size(); ++i) {
      const tensorWrapper = this.outputTensorsVector.get(i);
      const tensorReference = outputTensorReferences.get(i);
      output.push(new Tensor({
        type: {
          dtype: tensorWrapper.type(),
          layout: {dimensions: tensorWrapper.shape()}
        },
        accelerator: tensorWrapper.accelerator(),
        reference: tensorReference,
      }));
      tensorWrapper.delete();  // Free this copy from the `.get` call.
      // Do not free the tensorReference here since it has been passed to the
      // output Tensor. It will be freed when the Tensor is freed.
    }
    outputTensorReferences.delete();

    return output;
  }

  /**
   * Get details about each input tensor.
   */
  getInputDetails() {
    return getTensorMapDetails(this.inputTensors);
  }

  /**
   * Get details about each output tensor.
   */
  getOutputDetails() {
    return getTensorMapDetails(this.outputTensors);
  }

  delete() {
    if (this.deleted) {
      return;
    }
    // Delete all the copies
    for (const tensor of this.inputTensors.values()) {
      tensor.delete();
    }
    this.inputTensors.clear();
    this.inputTensorsVector.delete();

    for (const tensor of this.outputTensors.values()) {
      tensor.delete();
    }
    this.outputTensors.clear();
    this.outputTensorsVector.delete();

    this.deleted = true;
    // Note that we don't delete the signatureRunnerWrapper here since it's a
    // reference owned by the interpreter.
  }
}

function getTensorMapDetails(tensors: Map<string, TensorWrapper>) {
  return [...tensors.entries()].map(
      ([name, tensor], index) =>
          ({name, index, shape: tensor.shape(), dtype: tensor.type()}));
}
