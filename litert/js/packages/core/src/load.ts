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

import {createWasmLib} from '@litertjs/wasm-utils';

import {LiteRt} from './litert_web';
import {supportsRelaxedSimd} from './wasm_feature_detect';

const WASM_JS_FILE_NAME = 'litert_wasm_internal.js';
const WASM_JS_COMPAT_FILE_NAME = 'litert_wasm_internal_compat.js';

/**
 * Load the LiteRt library with WASM from the given URL. Does not set the
 * global LiteRT instance.
 */
export async function load(path: string): Promise<LiteRt> {
  const relaxedSimd = await supportsRelaxedSimd();
  const fileName = relaxedSimd ? WASM_JS_FILE_NAME : WASM_JS_COMPAT_FILE_NAME;

  if (path.endsWith('.wasm')) {
    console.warn(
        'Please load the `.js` file corresponding to the `.wasm` file, or ' +
        'load the directory containing it.');
    path = `${path.slice(0, -5)}.js`;
  } else if (path.endsWith('/')) {
    path = `${path}${fileName}`;
  } else if (!path.endsWith('.js')) {
    path = `${path}/${fileName}`;
  }

  return createWasmLib(LiteRt, path);
}