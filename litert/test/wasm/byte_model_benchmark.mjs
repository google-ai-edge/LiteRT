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

// Works in Node and in a browser. Timings exclude network and model download.
export const prompts = [
  'create an image of a cat', 'draw a watercolor landscape',
  'generate a video of a running dog', 'make a short video about space',
  'create a song about summer', 'compose some relaxing piano music',
  'what is the capital of france', 'explain how photosynthesis works',
  'help me plan my weekend', 'write a python function to sort a list',
  'craete an image', 'craete an video', 'make me a picture of a robot',
  'produce a jazz track', 'tell me a joke about cats',
  '  CREATE\t an\n IMAGE   of a mountain  ',
  'create an image of café ☕', '你好，帮我画一张猫的图片', '',
  'make an image of '.repeat(20),
];

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function encode(text, length) {
  const normalized = text.trim().toLowerCase().replace(/\s+/gu, ' ');
  const bytes = new TextEncoder().encode(normalized).subarray(0, length);
  const input = new Int32Array(length);
  bytes.forEach((value, index) => { input[index] = value + 1; });
  return input;
}

function predict(runner, text) {
  const {module: m, pointer, length} = runner;
  m.HEAP32.set(encode(text, length), pointer / 4);
  const start = performance.now();
  check(m._predict(pointer, length) === 0, 'predict failed');
  const milliseconds = performance.now() - start;
  const outputs = Array.from({length: m._get_output_count()}, (_, index) => {
    const start = m._get_output(index) / 4;
    const count = m._get_output_length(index);
    return Array.from(m.HEAPF32.subarray(start, start + count));
  }).sort((a, b) => a.length - b.length);
  return {milliseconds, outputs};
}

async function createRunner(factory, wasmBinary, modelBytes) {
  const start = performance.now();
  const module = await factory({wasmBinary});
  const moduleLoadMs = performance.now() - start;
  check(!(typeof SharedArrayBuffer !== 'undefined' &&
          module.HEAPU8.buffer instanceof SharedArrayBuffer),
        'single-threaded runner unexpectedly uses shared memory');
  const modelPointer = module._malloc(modelBytes.length);
  check(modelPointer !== 0, 'model allocation failed');
  module.HEAPU8.set(modelBytes, modelPointer);
  const loadStart = performance.now();
  const status = module._load_model(modelPointer, modelBytes.length);
  const modelLoadMs = performance.now() - loadStart;
  module._free(modelPointer);
  check(status === 0, `model load failed: ${status}`);
  const length = module._get_input_length();
  check(length === 128, `unexpected input length: ${length}`);
  const pointer = module._malloc(length * 4);
  check(pointer !== 0, 'input allocation failed');
  check(module._predict(pointer, length - 1) !== 0,
        'incorrect input length was accepted');
  return {module, moduleLoadMs, modelLoadMs, pointer, length};
}

function topLabel(probabilities) {
  return probabilities.indexOf(Math.max(...probabilities));
}

function distribution(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  return {
    count: sorted.length,
    medianMs: sorted[Math.floor(sorted.length / 2)],
    p95Ms: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * .95))],
    meanMs: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
  };
}

export async function benchmark(interpreterFactory, interpreterWasm,
                                compiledFactory, compiledWasm, modelBytes,
                                iterations = 100) {
  const runners = [
    await createRunner(interpreterFactory, interpreterWasm, modelBytes),
    await createRunner(compiledFactory, compiledWasm, modelBytes),
  ];
  let maxProbabilityDiff = 0;
  let maxEmbeddingDiff = 0;
  let labelMismatches = 0;
  const firstPredictMs = [];
  const classifications = [];
  for (const text of prompts) {
    const predictions = runners.map(runner => predict(runner, text));
    if (!firstPredictMs.length) {
      firstPredictMs.push(...predictions.map(p => p.milliseconds));
    }
    for (const prediction of predictions) {
      check(prediction.outputs.length === 2 &&
            prediction.outputs[0].length === 4 &&
            prediction.outputs[1].length === 128, 'unexpected output shapes');
      check(prediction.outputs.flat().every(Number.isFinite), 'nonfinite output');
    }
    for (let output = 0; output < 2; ++output) {
      const diffs = predictions[0].outputs[output].map((v, index) =>
          Math.abs(v - predictions[1].outputs[output][index]));
      const maximum = Math.max(...diffs);
      if (output === 0) maxProbabilityDiff = Math.max(maxProbabilityDiff, maximum);
      else maxEmbeddingDiff = Math.max(maxEmbeddingDiff, maximum);
    }
    const probabilities = predictions.map(p => p.outputs[0]);
    if (topLabel(probabilities[0]) !== topLabel(probabilities[1])) ++labelMismatches;
    classifications.push({text, probabilities: probabilities[1]});
  }
  check(labelMismatches === 0 && maxProbabilityDiff < 1e-6 &&
        maxEmbeddingDiff < 1e-6, 'runtime parity failed');
  const samples = [[], []];
  // Alternate order to reduce bias from runtime warmup and host activity.
  for (let iteration = 0; iteration < iterations; ++iteration) {
    for (const index of iteration % 2 ? [0, 1] : [1, 0]) {
      samples[index].push(predict(runners[index],
          prompts[iteration % prompts.length]).milliseconds);
    }
  }
  const timings = runners.map((runner, index) => ({
    moduleLoadMs: runner.moduleLoadMs,
    modelLoadMs: runner.modelLoadMs,
    firstPredictMs: firstPredictMs[index],
    warm: distribution(samples[index]),
  }));
  // A corrupt download must fail cleanly, and a subsequent valid load must
  // restore the runner without leaving buffers or model state stale.
  for (const runner of runners) {
    const m = runner.module;
    const modelPointer = m._malloc(modelBytes.length);
    check(modelPointer !== 0, 'reload allocation failed');
    m.HEAPU8.fill(0, modelPointer, modelPointer + 8);
    check(m._load_model(modelPointer, 8) !== 0, 'corrupt model was accepted');
    check(m._predict(runner.pointer, runner.length) !== 0,
          'predict succeeded after a failed load');
    m.HEAPU8.set(modelBytes, modelPointer);
    check(m._load_model(modelPointer, modelBytes.length) === 0, 'reload failed');
    m._free(modelPointer);
    const reloaded = predict(runner, prompts[0]).outputs[0];
    check(reloaded.every((value, index) =>
        Math.abs(value - classifications[0].probabilities[index]) < 1e-6),
        'prediction changed after reload');
    m._free(runner.pointer);
  }
  return {prompts: prompts.length, iterations, maxProbabilityDiff,
    maxEmbeddingDiff, labelMismatches, corruptModelRejected: true,
    reloadPassed: true, interpreter: timings[0],
    compiledModel: timings[1], classifications};
}
