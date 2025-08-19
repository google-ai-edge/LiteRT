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

/**
 * Fetches each URL in urls, executes them one-by-one in the order they are
 * passed, and then returns (or throws if something went amiss).
 */
declare function importScripts(...urls: Array<string|URL>): void;

/**
 * Load a script from the given URL.
 */
export async function runScript(scriptUrl: string) {
  if (typeof importScripts === 'function') {
    importScripts(scriptUrl.toString());
  } else {
    const script = document.createElement('script');
    (script as {src:string}).src = scriptUrl.toString();
    script.crossOrigin = 'anonymous';
    return new Promise<void>((resolve, revoke) => {
      script.addEventListener('load', () => {
        resolve();
      }, false);
      script.addEventListener('error', e => {
        revoke(e);
      }, false);
      // TODO: b/424626721 - Remove scripts from the DOM after they have
      // loaded.
      document.body.appendChild(script);
    });
  }
}
