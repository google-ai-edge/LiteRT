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

import {Environment} from './environment';
import {Tensor} from './tensor';
import {LiteRtCompileOptions} from './wasm_binding_types';

/**
 * Options for loading and compiling a LiteRt model.
 */
export interface CompileOptions extends LiteRtCompileOptions {
  environment?: Environment;
}

/**
 * The possible tensor inputs to the run() method.
 */
export type TensorInputs = Tensor|Tensor[]|Record<string, Tensor>;
