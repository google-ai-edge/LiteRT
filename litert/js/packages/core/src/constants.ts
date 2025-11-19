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
 * The accelerators that LiteRt Web supports.
 */
export const ACCELERATORS = ['webgpu', 'wasm'] as const;
/**
 * The type for accelerators that LiteRt Web supports.
 */
export type Accelerator = (typeof ACCELERATORS)[number];

/**
 * A map from data types to the corresponding JavaScript array type.
 */
export const DTYPE_TO_ARRAY_TYPE = Object.freeze({
  // noType is not supported.
  'float32': Float32Array,
  'int32': Int32Array,

  // The following types are disabled until we support them in C++.
  /*
  'uint8': Uint8Array,
  // TODO(msoulanille): int64 is not supported yet because BigInt64Array makes
  // TFJS integration more complicated.
  // 'int64': BigInt64Array,
  // String is not supported.
  // TODO(msoulanille): bool will require special handling in C++.
  // TFJS WebGPU stores bool in a 32 bit integer.
  // However, tf.data() returns a Uint8Array.
  // Unclear if we should follow TFJS or whatever LiteRt xnnpack does.
  'bool': Uint8Array,
  'int16': Int16Array,
  // Complex64 is not supported.
  'int8': Int8Array,
  // JS does not have a Float16Array.
  // TODO(msoulanille): This will require special handling in C++.
  'float16': Float32Array,
  'float64': Float64Array,
  // Complex128 is not supported.
  // TODO(msoulanille): uint64 is not supported yet because BigInt64Array makes
  // TFJS integration more complicated.
  // 'uint64': BigInt64Array,
  // Resource and Variant are not supported.
  'uint32': Uint32Array,
  'uint16': Uint16Array,
  // TODO(msoulanille): This will require special handling in C++.
  'int4': Uint8Array,
  // TODO(msoulanille): This will require special handling in C++.
  'bfloat16': Float32Array,
  */
} as const);

/**
 * The data type of a Tensor.
 */
export type DType = keyof typeof DTYPE_TO_ARRAY_TYPE;

// This is intentionally not typed as `ReadonlySet<DType>` because we want to
// make it easy to call 'has' on this set with strings.
/**
 * The set of supported data types in LiteRt Web. These are represented by
 * strings.
 */
export const SUPPORTED_DTYPES: ReadonlySet<string> =
    new Set(Object.keys(DTYPE_TO_ARRAY_TYPE));

/**
 * The dimensions, or shape, of a Tensor.
 */
export type Dimensions = Int32Array|number[];

/**
 * The constructor for a TypedArray.
 */
export type TypedArrayConstructor = (typeof DTYPE_TO_ARRAY_TYPE)[DType];

/**
 * A TypedArray.
 */
export type TypedArray = InstanceType<TypedArrayConstructor>;

declare global {
  // TypeScript is missing some signatures for these constructor types.
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Int32Array/Int32Array
  interface Int32ArrayConstructor {
    new(buffer: ArrayBuffer, byteOffset?: number, length?: number): Int32Array;
  }

  interface Float32ArrayConstructor {
    new(buffer: ArrayBuffer, byteOffset?: number,
        length?: number): Float32Array;
  }
}

/**
 * Converts a TypedArray to its corresponding DType.
 */
export function typedArrayToDtype(data: TypedArray): DType {
  if (data instanceof Float32Array) {
    return 'float32';
  } else if (data instanceof Int32Array) {
    return 'int32';
  }
  throw new Error(
      `Unsupported typed array type ${(data as Uint8Array).constructor.name}.`);
}
