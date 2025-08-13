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

import {Accelerator} from './constants';
import {SignatureRunner} from './signature_runner';
import {Tensor} from './tensor';
import type {SignatureRunnerWrapper, TensorWrapper} from './wasm_binding_types';
import {LiteRtInterpreter} from './wasm_binding_types';

/**
 * Represents a loaded LiteRt model.
 */
export class CompiledModel {
  protected inputTensors = new Map<string, TensorWrapper>();
  protected outputTensors = new Map<string, TensorWrapper>();
  readonly signatures: Record<string, SignatureRunner> = {};
  protected readonly primarySignature: SignatureRunner;
  readonly accelerator: Accelerator;
  deleted = false;

  constructor(
      protected readonly liteRtInterpreter: LiteRtInterpreter,
      makeSignatureRunner:
          (signatureRunnerWrapper: SignatureRunnerWrapper) => SignatureRunner,
      private readonly onDelete: () => void) {
    const tfliteInputs = this.liteRtInterpreter.inputs();
    for (let i = 0; i < tfliteInputs.size(); ++i) {
      const tensor = tfliteInputs.get(i);
      this.inputTensors.set(tensor.name(), tensor);
    }

    const signaturesVector = this.liteRtInterpreter.listSignatures();
    for (let i = 0; i < signaturesVector.size(); ++i) {
      const signatureName = signaturesVector.get(i);
      this.signatures[signatureName] = makeSignatureRunner(
          this.liteRtInterpreter.getSignatureRunner(signatureName));
    }
    this.primarySignature = makeSignatureRunner(liteRtInterpreter);
    this.accelerator = this.primarySignature.accelerator;
  }

  private checkDeleted() {
    if (this.deleted) {
      throw new Error('Model has been deleted. Please reload the model.');
    }
  }

  /**
   * Runs the model with the given input tensors and returns the outputs.
   *
   * If the first argument is a string, it is interpreted as a signature name
   * and the second argument is interpreted as the input tensors for that
   * signature. Otherwise, the first argument is interpreted as the input
   * tensors for the primary signature.
   */
  run(input: Tensor|Tensor[]): Tensor[];
  run(input: Record<string, Tensor>): Record<string, Tensor>;
  run(signatureName: string, input: Tensor|Tensor[]): Tensor[];
  run(signatureName: string,
      input: Record<string, Tensor>): Record<string, Tensor>;
  // Typescript won't automatically distribute a union over the above
  // signatures, so we need to explicitly declare them below.
  // https://github.com/microsoft/TypeScript/issues/14107
  run(input: Tensor|Tensor[]|
      Record<string, Tensor>): Tensor[]|Record<string, Tensor>;
  run(signatureName: string, input: Tensor|Tensor[]|Record<string, Tensor>):
      Tensor[]|Record<string, Tensor>;
  run(inputOrSignatureName: string|Tensor|Tensor[]|Record<string, Tensor>,
      maybeInput?: Tensor|Tensor[]|Record<string, Tensor>) {
    this.checkDeleted();
    if (typeof inputOrSignatureName === 'string') {
      const signatureName = inputOrSignatureName;
      const input = maybeInput;
      const signature = this.signatures[signatureName];
      if (!signature) {
        const signatures = Object.keys(this.signatures).join(', ');
        throw new Error(`Signature '${
            signatureName}' not found in the model. Available signatures: ${
            signatures}`);
      }
      if (!input) {
        throw new Error(`No input provided for signature '${signatureName}'.`);
      }

      return signature.run(input);
    } else {
      return this.primarySignature.run(inputOrSignatureName);
    }
  }

  /**
   * Returns the input details for the primary signature.
   */
  getInputDetails() {
    this.checkDeleted();
    return this.primarySignature.getInputDetails();
  }

  /**
   * Returns the output details for the primary signature.
   */
  getOutputDetails() {
    this.checkDeleted();
    return this.primarySignature.getOutputDetails();
  }

  delete() {
    if (this.deleted) {
      return;
    }
    // Set this now to prevent reentry, although that shouldn't happen.
    this.deleted = true;

    for (const signature of Object.values(this.signatures)) {
      signature.delete();
    }

    // This is separate from the other signatures
    this.primarySignature.delete();

    for (const input of this.inputTensors.values()) {
      input.delete();
    }
    for (const output of this.outputTensors.values()) {
      output.delete();
    }

    this.liteRtInterpreter.delete();
    this.onDelete();
  }
}
