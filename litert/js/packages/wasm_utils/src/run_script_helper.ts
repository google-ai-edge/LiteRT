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
 * Fetches the script at `scriptUrl` and runs it as a classic script in the
 * global scope, so that its top-level declarations (e.g. the `ModuleFactory`
 * of an Emscripten loader script) become globals, as with `importScripts`.
 */
async function fetchAndRunScript(scriptUrl: string) {
  const response = await fetch(scriptUrl);
  if (!response.ok) {
    throw new Error(
        `Failed to fetch ${scriptUrl}: ${response.status} ${
            response.statusText}`);
  }
  const source = await response.text();
  // An indirect eval runs the code as a sloppy-mode global script, unlike a
  // dynamic import(), which would run it as a strict-mode module whose
  // declarations stay private to it. The script is the same one importScripts
  // would have run; pages with a Content Security Policy must allow
  // 'unsafe-eval' for this fallback.
  (0, eval)(`${source}\n//# sourceURL=${scriptUrl}`);
}

/**
 * Load a script from the given URL.
 */
export async function runScript(scriptUrl: string) {
  if (typeof importScripts === 'function') {
    try {
      importScripts(scriptUrl.toString());
    } catch (e) {
      // Module workers (`new Worker(url, {type: 'module'})`, the default for
      // bundlers such as Vite) define importScripts, but calling it throws a
      // TypeError, and they have no document to add a script element to.
      if (!(e instanceof TypeError)) {
        throw e;
      }
      await fetchAndRunScript(scriptUrl.toString());
    }
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
