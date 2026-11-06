/**
 * Copyright 2026 Google LLC
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

// Run with `npm test` (Node 22.18+ / 24, which run TypeScript natively).

import assert from 'node:assert/strict';
import {afterEach, beforeEach, test} from 'node:test';

import {runScript} from '../src/run_script_helper.ts';

const SCRIPT_URL = 'https://example.com/litertlm_wasm_internal.js';
// Mimics the top of an Emscripten MODULARIZE loader script, which declares the
// factory with a sloppy-mode top-level `var`.
const LOADER_SCRIPT = 'var ModuleFactory = function() { return "module"; };';

const scope = globalThis as unknown as Record<string, unknown>;
let originalFetch: unknown;

/**
 * Makes the global scope look like a module worker
 * (`new Worker(url, {type: 'module'})`): importScripts is defined, but calling
 * it throws, and there is no document.
 */
function installModuleWorkerImportScripts() {
  scope['importScripts'] = () => {
    throw new TypeError(
        'Failed to execute \'importScripts\' on \'WorkerGlobalScope\': ' +
        'Module scripts don\'t support importScripts().');
  };
}

beforeEach(() => {
  originalFetch = scope['fetch'];
});

afterEach(() => {
  scope['fetch'] = originalFetch;
  delete scope['importScripts'];
  delete scope['ModuleFactory'];
});

// Regression test for https://github.com/google-ai-edge/LiteRT-LM/issues/4080:
// loading a Wasm module failed in module workers.
test('runScript loads a classic loader script in a module worker', async () => {
  installModuleWorkerImportScripts();
  const fetchedUrls: string[] = [];
  scope['fetch'] = async (url: string) => {
    fetchedUrls.push(url);
    return new Response(LOADER_SCRIPT);
  };

  await runScript(SCRIPT_URL);

  assert.deepEqual(fetchedUrls, [SCRIPT_URL]);
  assert.equal(typeof scope['ModuleFactory'], 'function');
  assert.equal((scope['ModuleFactory'] as () => string)(), 'module');
});

test('runScript reports a failed fetch in a module worker', async () => {
  installModuleWorkerImportScripts();
  scope['fetch'] = async () =>
      new Response('not found', {status: 404, statusText: 'Not Found'});

  await assert.rejects(runScript(SCRIPT_URL), /Failed to fetch .* 404/);
});

test('runScript uses importScripts in a classic worker', async () => {
  const importedUrls: string[] = [];
  scope['importScripts'] = (url: string) => {
    importedUrls.push(url);
  };
  scope['fetch'] = async () => {
    throw new Error('fetch must not be called');
  };

  await runScript(SCRIPT_URL);

  assert.deepEqual(importedUrls, [SCRIPT_URL]);
});

test('runScript rethrows importScripts errors other than TypeError',
     async () => {
       scope['importScripts'] = () => {
         throw new Error('NetworkError: failed to load script');
       };

       await assert.rejects(runScript(SCRIPT_URL), /NetworkError/);
     });
