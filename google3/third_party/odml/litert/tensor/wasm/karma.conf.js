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

/**
 * @fileoverview Karma configuration for tensor_wasm unit tests.
 */

module.exports = function(config) {
  // Proxy requests for Emscripten output files directly to the bazel output directory
  config.proxies['/wasm/'] = '/base/third_party/odml/litert/tensor/wasm/';

  // Ignore SSL validation errors for local testing
  config.proxyValidateSSL = false;

  // Set appropriate cross-origin headers to avoid SharedArrayBuffer restrictions
  config.hostname = 'localhost';
  config.set({
    customHeaders: (config.customHeaders || []).concat([
      {
        match: '.*',
        name: 'Cross-Origin-Embedder-Policy',
        value: 'require-corp'
      },
      {
        match: '.*',
        name: 'Cross-Origin-Opener-Policy',
        value: 'same-origin'
      }
    ])
  });
};
