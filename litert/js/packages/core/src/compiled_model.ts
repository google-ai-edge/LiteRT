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

import {Model} from './model';
import {CompileOptions, TensorInputs} from './model_types';
import {CompiledModelSignatureRunner, SignatureRunner, TensorDetails} from './signature_runner';
import {Tensor} from './tensor';
import {Deletable, LiteRtCompiledModel} from './wasm_binding_types';

/**
 * Represents a compiled LiteRt model.
 */
export class CompiledModel implements Deletable, SignatureRunner {
  private readonly defaultSignature: SignatureRunner;
  private readonly compiledModelSignatureRunners:
      Record<string, CompiledModelSignatureRunner>;
  readonly key: string;
  private deletedInternal = false;

  constructor(
      private readonly model: Model,
      private readonly liteRtCompiledModel: LiteRtCompiledModel,
      readonly options: Required<CompileOptions>,
      private readonly onDelete: () => void,
  ) {
    const numSignatures = model.liteRtModel.getNumSignatures();
    const compiledModelSignatureRunners:
        Record<string, CompiledModelSignatureRunner> = {};
    for (let i = 0; i < numSignatures; i++) {
      const compiledModelSignatureRunner = new CompiledModelSignatureRunner(
          i, model.liteRtModel, liteRtCompiledModel, options);
      compiledModelSignatureRunners[compiledModelSignatureRunner.key] =
          compiledModelSignatureRunner;
    }
    this.compiledModelSignatureRunners =
        Object.freeze(compiledModelSignatureRunners);

    this.defaultSignature = Object.values(this.signatures)[0]!;
    this.key = this.defaultSignature.key;
  }

  get signatures(): Record<string, SignatureRunner> {
    this.ensureNotDeleted();
    return this.compiledModelSignatureRunners;
  }

  getInputDetails(): readonly TensorDetails[] {
    this.ensureNotDeleted();
    return this.defaultSignature.getInputDetails();
  }

  getOutputDetails(): readonly TensorDetails[] {
    this.ensureNotDeleted();
    return this.defaultSignature.getOutputDetails();
  }

  /**
   * Runs the default signature with positional input tensors.
   * @param input The input tensors.
   * @return A promise that resolves to the output tensors.
   */
  run(input: Tensor|Tensor[]): Promise<Tensor[]>;
  /**
   * Runs the default signature with named input tensors.
   * @param input A record of named input tensors.
   * @return A promise that resolves to a record of named output tensors.
   */
  run(input: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  /**
   * Runs a specific signature by name with positional input tensors.
   * @param signatureName The name of the signature to run.
   * @param input The input tensors.
   * @return A promise that resolves to the output tensors.
   */
  run(signatureName: string, input: Tensor|Tensor[]): Promise<Tensor[]>;
  /**
   * Runs a specific signature by name with named input tensors.
   * @param signatureName The name of the signature to run.
   * @param input A record of named input tensors.
   * @return A promise that resolves to a record of named output tensors.
   */
  run(signatureName: string,
      input: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  // Typescript won't automatically distribute a union over the above
  // signatures, so we need to explicitly declare them below.
  // https://github.com/microsoft/TypeScript/issues/14107
  run(input: Tensor|Tensor[]|
      Record<string, Tensor>): Promise<Tensor[]|Record<string, Tensor>>;
  run(signatureName: string, input: Tensor|Tensor[]|Record<string, Tensor>):
      Promise<Tensor[]|Record<string, Tensor>>;
  async run(
      inputOrSignatureName: string|Tensor|Tensor[]|Record<string, Tensor>,
      maybeInput?: Tensor|Tensor[]|Record<string, Tensor>) {
    this.ensureNotDeleted();
    const [signature, input] =
        this.parseRunInputs(inputOrSignatureName, maybeInput);
    return await signature.run(input);
  }

  private parseRunInputs(
      inputOrSignatureName: string|TensorInputs,
      maybeInput?: TensorInputs): [SignatureRunner, TensorInputs] {
    let signature: SignatureRunner;
    let input: TensorInputs;
    if (typeof inputOrSignatureName === 'string') {
      signature = this.signatures[inputOrSignatureName];
      if (!signature) {
        throw new Error(
            `No signature named ${inputOrSignatureName} found in model.`);
      }
      if (!maybeInput) {
        throw new Error(
            `No input provided for signature ${inputOrSignatureName}`);
      }
      input = maybeInput;
    } else {
      signature = this.defaultSignature;
      input = inputOrSignatureName;
    }
    return [signature, input];
  }

  get deleted(): boolean {
    return this.deletedInternal;
  }

  private ensureNotDeleted() {
    if (this.deleted) {
      throw new Error('CompiledModel is deleted and cannot be used.');
    }
  }

  delete() {
    // For now, we enforce a 1:1 correspondence between models and compiled
    // models. If there's a need in the future, we can revise this.
    if (this.deletedInternal) {
      return;
    }
    this.deletedInternal = true;
    this.liteRtCompiledModel.delete();
    this.model.delete();
    for (const signatureRunner of Object.values(
             this.compiledModelSignatureRunners)) {
      signatureRunner.delete();
    }
    this.onDelete();
  }
}
