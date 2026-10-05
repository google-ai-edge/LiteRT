// Copyright 2026 Google LLC.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// node benchmark_node.mjs interpreter.js compiled_model.js model.tflite
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import {gzipSync} from 'node:zlib';
import {benchmark} from './byte_model_benchmark.mjs';

const paths = process.argv.slice(2).map(path => resolve(path));
if (paths.length !== 3) {
  throw new Error('Usage: node benchmark_node.mjs interpreter.js compiled_model.js model.tflite');
}
const require = createRequire(import.meta.url);
const binaries = paths.slice(0, 2).map(path => readFileSync(path.replace(/\.js$/, '.wasm')));
const model = readFileSync(paths[2]);
const results = await benchmark(require(paths[0]), binaries[0],
    require(paths[1]), binaries[1], model);
const sizes = bytes => ({raw: bytes.length, gzip: gzipSync(bytes, {level: 9}).length});
console.log(JSON.stringify({node: process.version,
  modelSha256: createHash('sha256').update(model).digest('hex'),
  sizes: {interpreter: sizes(binaries[0]), compiledModel: sizes(binaries[1]),
    interpreterJs: sizes(readFileSync(paths[0])),
    compiledModelJs: sizes(readFileSync(paths[1]))}, ...results}, null, 2));
