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
import {Box, GpuErrorReporter, popErrorScopes, pushErrorScopes} from './gpu_utils';
import {SignatureRunner} from './signature_runner';
import type {SignatureRunnerWrapper} from './wasm_binding_types';

/**
 * A signature runner that runs on the GPU.
 */
export class GpuSignatureRunner extends SignatureRunner {
  accelerator: Accelerator = 'webgpu';

  constructor(
      signatureRunnerWrapper: SignatureRunnerWrapper,
      // TODO: Remove Interpreter from the constructor. It's only here so we can
      // access utilities for GPU input/output copying and making vectors.
      private readonly device: GPUDevice,
      private readonly gpuErrorReporter: Box<GpuErrorReporter>) {
    super(signatureRunnerWrapper);
  }

  protected override pushErrorScopes() {
    pushErrorScopes(this.device);
  }

  protected override popErrorScopes(callsite: string) {
    popErrorScopes(this.device, callsite, this.gpuErrorReporter.val);
  }
}
