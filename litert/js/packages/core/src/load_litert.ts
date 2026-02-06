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

// Placeholder for internal dependency on trusted resource url type

import {Environment} from './environment';
import {getGlobalLiteRt, getGlobalLiteRtPromise, hasGlobalLiteRt, hasGlobalLiteRtPromise, setGlobalLiteRt, setGlobalLiteRtPromise} from './global_litert';
import {LiteRt} from './litert_web';
import {load, LoadOptions} from './load';

type UrlString = string;

/**
 * Options for loading LiteRT.
 *
 * @property threads Whether to load the threaded version of the Wasm module.
 *     Defaults to false. Unused when specifying a .js file directly instead of
 *     a directory containing the Wasm files.
 * @property jspi Whether to load the JSPI version of the Wasm module. Defaults
 *     to false. Unused when specifying a .js file directly instead of a
 *     directory containing the Wasm files.
 **/
export interface LoadLiteRtOptions extends LoadOptions {}

/**
 * Load LiteRT.js Wasm files from the given URL. This needs to be called before
 * calling any other LiteRT functions.
 *
 * The URL can be:
 *
 * - A directory containing the LiteRT Wasm files (e.g. `.../wasm/`), or
 * - The LiteRT Wasm's js file (e.g. `.../litert_wasm_internal.js`)
 *
 * If the URL is to a directory, LiteRT.js will detect what WASM features are
 * available in the browser and load the compatible WASM file. If the URL is
 * to a file, it will be loaded as is.
 *
 * @param path The path to the directory containing the LiteRT Wasm files, or
 *     the full URL of the LiteRT Wasm .js file.
 */
export function loadLiteRt(
    path: UrlString, options?: LoadLiteRtOptions): Promise<LiteRt> {
  if (hasGlobalLiteRtPromise()) {
    throw new Error('LiteRT is already loading / loaded.');
  }
  setGlobalLiteRtPromise(load(path, options)
                             .then(async liteRt => {
                               setGlobalLiteRt(liteRt);
                               liteRt.setDefaultEnvironment(
                                   await Environment.create());
                               return liteRt;
                             })
                             .catch(error => {
                               setGlobalLiteRtPromise(undefined);
                               throw error;
                             }));
  return getGlobalLiteRtPromise()!;
}

/**
 * Unload the LiteRt WASM module.
 *
 * This deletes the global LiteRT instance and invalidate any models,
 * signatures, and tensors associated with it. You will need to call
 * loadLiteRt() again to reload the module.
 */
export function unloadLiteRt(): void {
  if (hasGlobalLiteRtPromise() && !hasGlobalLiteRt()) {
    throw new Error(
        'LiteRT is loading and can not be unloaded or canceled ' +
        'until it is finished loading.');
  }

  if (hasGlobalLiteRt()) {
    getGlobalLiteRt().delete();
    setGlobalLiteRt(undefined);
  }
  setGlobalLiteRtPromise(undefined);
}
