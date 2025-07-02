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
 * Checks for relaxed SIMD support.
 *
 * (module
 *  (func (result v128)
 *    i32.const 1
 *    i8x16.splat
 *    i32.const 2
 *    i8x16.splat
 *    i8x16.relaxed_swizzle
 *  )
 * )
 */
const WASM_RELAXED_SIMD_CHECK = new Uint8Array([
  0, 97, 115, 109, 1,  0, 0,  0, 1,   5,  1,  96, 0,   1,  123, 3,   2, 1,
  0, 10, 15,  1,   13, 0, 65, 1, 253, 15, 65, 2,  253, 15, 253, 128, 2, 11
]);

const WASM_FEATURE_VALUES: {relaxedSimd: Promise<boolean>|undefined} = {
  relaxedSimd: undefined,
};

async function tryWasm(wasm: Uint8Array): Promise<boolean> {
  try {
    await WebAssembly.instantiate(wasm);
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Returns true if the browser supports relaxed SIMD.
 */
export async function supportsRelaxedSimd(): Promise<boolean> {
  if (WASM_FEATURE_VALUES.relaxedSimd === undefined) {
    WASM_FEATURE_VALUES.relaxedSimd = tryWasm(WASM_RELAXED_SIMD_CHECK);
  }
  return WASM_FEATURE_VALUES.relaxedSimd!;
}
