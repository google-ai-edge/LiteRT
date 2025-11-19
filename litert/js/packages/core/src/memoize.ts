/**
 * @fileoverview A memoization utility for JavaScript.
 *
 * This utility provides a function `memoize` that can be used to memoize
 * functions. A memoized function will only be called once for each unique set
 * of arguments, and the result will be cached and returned on subsequent calls.
 *
 * Example usage:
 *
 * ```typescript
 * const memoizedAdd = memoize((a, b) => a + b);
 * console.log(memoizedAdd(1, 2)); // Output: 3
 * console.log(memoizedAdd(1, 2)); // Output: 3
 * ```
 *
 * In this example, the `memoizedAdd` function will only be called once, even
 * though it is called twice. The result of the first call will be cached and
 * returned on the second call.
 *
 * @license
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

class TreeMap<K, V> extends Map<K, TreeMap<K, V>> {
  val?: V;

  constructor() {
    super();
  }

  private getMap(key: K[]): TreeMap<K, V>|undefined;
  private getMap(key: K[], createIfMissing: false): TreeMap<K, V>|undefined;
  private getMap(key: K[], createIfMissing: true): TreeMap<K, V>;
  private getMap(key: K[], createIfMissing = false): TreeMap<K, V>|undefined {
    let map: TreeMap<K, V> = this;
    for (let i = 0; i < key.length; i++) {
      if (!map.has(key[i])) {
        if (!createIfMissing) {
          return undefined;
        }
        map.set(key[i], new TreeMap<K, V>());
      }
      map = map.get(key[i])!;
    }
    return map;
  }

  getPath(key: K[]): V|undefined {
    return this.getMap(key)?.val;
  }

  hasPath(key: K[]): boolean {
    return this.getMap(key) !== undefined;
  }

  setPath(key: K[], val: V) {
    const map = this.getMap(key, /* createIfMissing= */ true);
    map.val = val;
  }
}

/**
 * Memoize a function.
 * @param f The function to memoize.
 * @param getKey A function that takes the arguments to the function and returns
 *     a key that uniquely identifies the function call.
 * @returns A memoized function that will only be called once for each unique
 *     key.
 */
export function memoize<T, Args extends unknown[]>(
    f: (...args: Args) => T,
    getKey: (args: Args) => unknown[] = x => x): (...args: Args) => T {
  const cache = new TreeMap<unknown, T>();
  return (...args: Args): T => {
    const key = getKey(args);
    if (!cache.hasPath(key)) {
      cache.setPath(key, f(...args));
    }
    return cache.getPath(key)!;
  };
}
