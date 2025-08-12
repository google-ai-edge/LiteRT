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

import type {LiteRt} from './litert_web';

/**
 * An error thrown when LiteRT is not loaded.
 */
export class LiteRtNotLoadedError extends Error {
  constructor() {
    super(
        'LiteRT is not initialized yet. Please call loadLiteRt() and wait for its ' +
        'promise to resolve to load the LiteRT WASM module.');
  }
}

let globalLiteRt: LiteRt|undefined = undefined;
let globalLiteRtPromise: Promise<LiteRt>|undefined = undefined;

/**
 * Get the global LiteRT instance.
 *
 * In most cases, you can call the functions exported by this module that wrap
 * the global LiteRT instance instead.
 */
export function getGlobalLiteRt(): LiteRt {
  if (!globalLiteRt) {
    throw new LiteRtNotLoadedError();
  }
  return globalLiteRt;
}

/**
 * Check if the global LiteRT instance is defined.
 *
 * Only exposed internally.
 */
export function hasGlobalLiteRt(): boolean {
  return Boolean(globalLiteRt);
}

/**
 * Set the global LiteRT instance.
 *
 * Only exposed internally.
 */
export function setGlobalLiteRt(liteRt: LiteRt|undefined) {
  globalLiteRt = liteRt;
}

/**
 * Resolves when the currently loading / loaded LiteRT instance is loaded.
 *
 * If `loadLiteRt()` has been called, this function returns a promise that
 * resolves when the LiteRT instance is loaded. Otherwise, it returns undefined.
 *
 * If LiteRT has failed to load before this function is called or has been
 * manually unloaded with `unloadLiteRt()`, this function also returns
 * undefined.
 */
export function getGlobalLiteRtPromise(): Promise<LiteRt>|undefined {
  return globalLiteRtPromise;
}

/**
 * Check if the global LiteRT instance promise is defined.
 *
 * Only exposed internally.
 */
export function hasGlobalLiteRtPromise(): boolean {
  return Boolean(globalLiteRtPromise);
}

/**
 * Set the global LiteRT instance promise.
 *
 * Only exposed internally.
 */
export function setGlobalLiteRtPromise(promise: Promise<LiteRt>|undefined) {
  globalLiteRtPromise = promise;
}
