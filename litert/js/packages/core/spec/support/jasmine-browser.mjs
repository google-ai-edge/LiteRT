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

import * as express from 'express';

// See https://github.com/jasmine/jasmine-browser-runner/blob/main/lib/types.js
// for options

/**
 * A handler for express.static that sets CORS headers. This is required so that
 * tests that use wasm threads can run.
 * @param {import('http').ServerResponse} res
 */
function setCorsHeaders(res) {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
}

export default {
  srcDir: "src",
  // srcFiles should usually be left empty when using ES modules, because you'll
  // explicitly import sources from your specs.
  srcFiles: [],
  specDir: ".",
  // To match the internal build, test files are included in `src/`. We build
  // a bundle of them to run with jasmine and output it to `spec_dist/` so that
  // `dist/` can contain just the files we publish to npm.
  //
  // This is done in the `test` script in the package.json.
  //
  // If another test file is added, we should create a `index_test.ts`
  // entrypoint.
  specFiles: [
    "spec_dist/**/*_test.js",
  ],
  helpers: [],
  esmFilenameExtension: ".js",
  // Set to true if you need to load module src files instead of loading via the spec files.
  modulesWithSideEffectsInSrcFiles: false,
  // Allows the use of top-level await in src/spec/helper files. This is off by
  // default because it makes files load more slowly.
  enableTopLevelAwait: false,
  env: {
    stopSpecOnExpectationFailure: false,
    stopOnSpecFailure: false,
    random: true,
    // Fail if a suite contains multiple suites or specs with the same name.
    forbidDuplicateNames: true
  },
  middleware: {
    // Serve all files with CORS headers to enable wasm threads.
    '/': (req, res, next) => {
      setCorsHeaders(res);
      next();
    },
    '/wasm': express.static('./wasm/'),
    '/testdata': express.static('./testdata/'),
  },

  // For security, listen only to localhost. You can also specify a different
  // hostname or IP address, or remove the property or set it to "*" to listen
  // to all network interfaces.
  listenAddress: "localhost",

  // The hostname that the browser will use to connect to the server.
  hostname: "localhost",

  browser: {
    name: "headlessChrome"
  }
};