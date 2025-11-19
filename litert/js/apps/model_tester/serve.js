#!/usr/bin/env node
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

import open from 'open';
import express from 'express';
import { ArgumentParser } from 'argparse';
import path from 'path';
import {fileURLToPath} from 'url';

// Get the path to the `dist` directory of the package.
const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const staticFilesPath = path.join(dirname, 'dist');

const parser = new ArgumentParser({
  description: 'LiteRT.js Model Tester',
});

parser.add_argument('--public', {
  action: 'store_true',
  help: 'Host the model tester publicly. By default, only connections from'
    + ' localhost are allowed.',
});

parser.add_argument('--port', {
  type: Number,
  default: 8123,
});

parser.add_argument('--open', {
  action: 'store_true',
  help: 'Whether to open Chrome. Defaults to true',
  default: true,
});

const args = parser.parse_args();
const host = args['public'] ? '0.0.0.0' : '127.0.0.1';

console.log(staticFilesPath);

const app = express();
app.use((_req, res, next) => {
  // Set headers to enable Wasm pthread emulation.
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  next();
});
app.use('/', express.static(staticFilesPath));
app.listen(args.port, host);
console.log(
    `Serving the LiteRT.js model tester ${
        args['public'] ? 'publicly' : 'locally'
    } on port ${args.port}`,
);
const localUrl = `http://127.0.0.1:${args.port}`;
console.log(`Local: ${localUrl}`);

if (args.open) {
  try {
    await open(localUrl);
  } catch (e) {
    console.warn(e.message);
  }
}
