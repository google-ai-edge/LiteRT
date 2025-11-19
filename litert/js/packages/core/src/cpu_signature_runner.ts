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
import {Tensor, TensorShapeError} from './tensor';
import type {SignatureRunnerWrapper} from './wasm_binding_types';

/**
 * A signature runner that runs on the CPU.
 */
export class CpuSignatureRunner extends SignatureRunner {
  accelerator: Accelerator = 'wasm';

  constructor(signatureRunnerWrapper: SignatureRunnerWrapper) {
    super(signatureRunnerWrapper);
  }

  /**
   * Throws an error if the input tensors have different shapes than the
   * signature.
   *
   * Note that this may be overrestrictive since it doesn't account for
   * automatically expanding / contracting dimensions (e.g. [1, 1, 224, 224] vs
   * [224, 224]).
   */
  protected checkShapes(input: Tensor[]): void {
    // TODO: b/393137695 - Write a version that can be applied to the GPU side
    // as well so error messages can be consistent?

    let i = 0;
    for (const tensorWrapper of this.inputTensors.values()) {
      const tensor = input[i++];
      const shape = tensor.type.layout.dimensions;
      const expectedShape = tensorWrapper.shape();
      if (expectedShape.length !== shape.length) {
        throw new TensorShapeError(tensorWrapper.name(), expectedShape, shape);
      }
      for (let j = 0; j < shape.length; ++j) {
        if (shape[j] !== expectedShape[j]) {
          throw new TensorShapeError(
              tensorWrapper.name(), expectedShape, shape);
        }
      }
    }
  }

  protected override runWithArray(input: Tensor[]): Tensor[] {
    // Since we are not using MLDrift, we need to manually check shapes.
    this.checkShapes(input);
    return super.runWithArray(input);
  }
}
