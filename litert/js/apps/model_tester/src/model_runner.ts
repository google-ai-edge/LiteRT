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

import type {ConsoleMessage} from './console_renderer';

/**
 * Well-known name for the LiteRT WASM CPU accelerator.
 * This should be unique. It is stored in JSON and displayed in the UI.
 */
export const LITERT_WASM_CPU = 'LiteRT WASM CPU';

/**
 * Well-known name for the LiteRT WebGPU accelerator.
 * This should be unique. It is stored in JSON and displayed in the UI.
 */
export const LITERT_WASM_GPU = 'LiteRT WebGPU';

/**
 * A class that runs a model for testing.
 */
export interface ModelRunner {
  getSignatures(): Array<{name: string}>;
  run(signature?: string, benchmarkRunCount?: number): Promise<RunResult>;
}

/**
 * A TypedArray
 */
export type TypedArray =
    |Uint8Array|Int8Array|Uint16Array|Int16Array|Float32Array|Int32Array|
    Uint32Array|Float64Array|BigUint64Array|BigInt64Array;

/**
 * A tensor that can be serialized to json.
 */
export interface SerializableTensor {
  readonly shape: readonly number[];
  readonly dtype: string;
  readonly data?: TypedArray;  // Usually omitted when serializing to json.
}

/**
 * A benchmark sample.
 */
export interface BenchmarkSample {
  latency: number;
}

/**
 * The result of a model run.
 */
export interface ModelResult {
  benchmark?: {samples: BenchmarkSample[];};
  tensors: {
    record: Record<string, SerializableTensor>;
    array?: SerializableTensor[];
  };
  meanSquaredError?: Record<string, number>;  // vs CPU backend
}

/**
 * An error that indicates that the model is not supported.
 */
export class UnsupportedError extends Error {}

/**
 * A type that is either a value or an error.
 */
export type Maybe<T> =|{
  value: T;
  error?: undefined;  // These are here to make optional chaining work.
}
|{
  value?: undefined;
  error: string;  // Easier to serialize.
};

/**
 * The result of a model run.
 */
export interface RunResult {
  results:
      Record<string, Maybe<ModelResult>>;  // Record for JSON serialization.
  consoleMessages?: ConsoleMessage[];
}

/**
 * Compares the results of two model runs.
 */
export function compareResults(
    firstResult: ModelResult, secondResult: ModelResult) {
  const firstKeys = new Set(Object.keys(firstResult.tensors.record!));
  const secondKeys = new Set(Object.keys(firstResult.tensors.record!));

  const difference = symmetricDifference(firstKeys, secondKeys);
  if (difference.size) {
    throw new Error(`Outputs do not have the same keys: ${difference}`);
  }

  const mses: Record<string, number> = {};
  for (const key of firstKeys) {
    const firstTensor = firstResult.tensors.record![key];
    const secondTensor = secondResult.tensors.record![key];
    const mse = meanSquaredError(firstTensor, secondTensor);

    mses[key] = mse;
  }

  return mses;
}

/**
 * A polyfill for Set.prototype.symmetricDifference, which is not supported in
 * all browsers.
 */
function symmetricDifference<T>(setA: Set<T>, setB: Set<T>): Set<T> {
  const difference = new Set<T>();
  for (const elem of setA) {
    if (!setB.has(elem)) {
      difference.add(elem);
    }
  }
  for (const elem of setB) {
    if (!setA.has(elem)) {
      difference.add(elem);
    }
  }
  return difference;
}

/**
 * Calculates the mean squared error between two tensors.
 */
export function meanSquaredError(a: SerializableTensor, b: SerializableTensor) {
  if (!a.data || !b.data) {
    throw new Error('Missing tensor data');
  }
  if (a.data.length !== b.data.length) {
    throw new Error(`Tensor length ${
        a.data.length} does not equal other tensor's length ${b.data.length}`);
  }

  const squaredDifference = a.data.map((aVal, index) => {
    if (typeof aVal === 'string') {
      throw new Error(`Tensor value was a string`);
    }

    const bVal = b.data![index];
    if (typeof bVal === 'string') {
      throw new Error(`Tensor value was a string`);
    }

    if (typeof aVal === 'bigint' && typeof bVal === 'bigint') {
      // Ideally, we wouldn't tell TypeScript that this is both a number and
      // a bigint, but the map function is mapping on a TypedArray that contains
      // either numbers or bigints.
      return ((aVal - bVal) ** 2n) as unknown as number & bigint;
    } else if (typeof aVal !== 'bigint' && typeof bVal !== 'bigint') {
      return ((aVal - bVal) ** 2) as unknown as number & bigint;
    } else {
      throw new Error('One result was a bigint while the other was a number');
    }
  });

  let sum = 0;
  for (const element of squaredDifference) {
    sum += Number(element);
  }

  return sum / a.data.length;
}

/**
 * Returns a Maybe containing the given value.
 */
export function just<T>(t: T): Maybe<T> {
  return {value: t};
}

/**
 * Turns a callable function into a Maybe containing its result or error.
 * Works for async and sync functions.
 */
export function toMaybe<T>(f: () => Promise<T>): Promise<Maybe<T>>;
export function toMaybe<T>(f: () => T): Maybe<T>;
export function toMaybe<T>(f: () => T | Promise<T>): Maybe<T>|
    Promise<Maybe<T>> {
  try {
    const value = f();
    if (value instanceof Promise) {
      return value.then(
          value => {
            return {value};
          },
          e => {
            console.error(e);
            const error = e instanceof Error ? e : new Error(String(e));
            return {error: error.stack ?? error.message};
          });
    } else {
      return {value};
    }
  } catch (e) {
    console.error(e);
    const error = e instanceof Error ? e : new Error(String(e));
    return {error: error.stack ?? error.message};
  }
}
