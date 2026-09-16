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

import {ElementType, ElementTypeName} from './wasm_binding_types';

enum DTypeInternal {
  FLOAT32 = 'float32',
  INT32 = 'int32',
  UINT8 = 'uint8',
  INT8 = 'int8',
  BOOL = 'bool',
  UINT16 = 'uint16',
  INT16 = 'int16',
  FLOAT16 = 'float16',
  INT64 = 'int64',
  UINT64 = 'uint64',
  FLOAT64 = 'float64',
}

// Only used in this file to ensure that DATATYPES follows a consistent pattern.
// Do not export this type because it's not specific enough.
interface DataTypeMappingInternal {
  dtype: DTypeInternal;
  typedArrayConstructor: Function | undefined;
  elementType: number;
}

// Check for the existence of potentially missing TypedArray constructors.
const Float16ArrayCtor =
    typeof Float16Array !== 'undefined' ? Float16Array : undefined;
const BigInt64ArrayCtor =
    typeof BigInt64Array !== 'undefined' ? BigInt64Array : undefined;
const BigUint64ArrayCtor =
    typeof BigUint64Array !== 'undefined' ? BigUint64Array : undefined;

/**
 * An array of objects for matching datatype strings, TypedArray constructors,
 * and LiteRT element types.
 *
 * The TypedArray constructor is not guaranteed to be unique across these values
 * since JavaScript does not have as many TypedArray types as LiteRT has
 * element types.
 */
const DATATYPES = Object.freeze([
  {
    dtype: DTypeInternal.FLOAT32,
    typedArrayConstructor: Float32Array,
    elementType: ElementType.FLOAT32
  } as const,
  {
    dtype: DTypeInternal.INT32,
    typedArrayConstructor: Int32Array,
    elementType: ElementType.INT32
  } as const,
  {
    dtype: DTypeInternal.UINT8,
    typedArrayConstructor: Uint8Array,
    elementType: ElementType.UINT8
  } as const,
  {
    dtype: DTypeInternal.INT8,
    typedArrayConstructor: Int8Array,
    elementType: ElementType.INT8
  } as const,
  {
    dtype: DTypeInternal.BOOL,
    typedArrayConstructor: Uint8Array,
    elementType: ElementType.BOOL
  } as const,
  {
    dtype: DTypeInternal.UINT16,
    typedArrayConstructor: Uint16Array,
    elementType: ElementType.UINT16
  } as const,
  {
    dtype: DTypeInternal.INT16,
    typedArrayConstructor: Int16Array,
    elementType: ElementType.INT16
  } as const,
  {
    dtype: DTypeInternal.FLOAT16,
    typedArrayConstructor: Float16ArrayCtor,
    elementType: ElementType.FLOAT16
  } as const,
  {
    dtype: DTypeInternal.INT64,
    typedArrayConstructor: BigInt64ArrayCtor,
    elementType: ElementType.INT64
  } as const,
  {
    dtype: DTypeInternal.UINT64,
    typedArrayConstructor: BigUint64ArrayCtor,
    elementType: ElementType.UINT64
  } as const,
  {
    dtype: DTypeInternal.FLOAT64,
    typedArrayConstructor: Float64Array,
    elementType: ElementType.FLOAT64
  } as const,
] as const satisfies DataTypeMappingInternal[]);

/**
 * Defines how a given datatype is mapped to a string, TypedArray constructor,
 * and LiteRT element type.
 */
export type DataTypeMapping = (typeof DATATYPES)[number];

/** The supported tensor data types. */
export type DType = `${DTypeInternal}`;

/**
 * The constructor for a TypedArray.
 */
export type TypedArrayConstructor =
    Float32ArrayConstructor | Int32ArrayConstructor | Uint8ArrayConstructor |
    Int8ArrayConstructor | Uint16ArrayConstructor | Int16ArrayConstructor |
    Float16ArrayConstructor | BigInt64ArrayConstructor | BigUint64ArrayConstructor |
    Float64ArrayConstructor;

/**
 * A TypedArray with number elements.
 */
export type NumberTypedArray =
    Float32Array | Int32Array | Uint8Array | Int8Array | Uint16Array |
    Int16Array | Float16Array | Float64Array;

/**
 * A TypedArray with bigint elements.
 */
export type BigIntTypedArray = BigInt64Array | BigUint64Array;

/**
 * A TypedArray.
 */
export type TypedArray = NumberTypedArray | BigIntTypedArray;

declare global {
  // TypeScript is missing some signatures for these constructor types.
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Int32Array/Int32Array
  interface Float32ArrayConstructor {
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): Float32Array;
  }

  interface Int32ArrayConstructor {
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): Int32Array;
  }

  interface Uint8ArrayConstructor {
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): Uint8Array;
  }

  interface Int8ArrayConstructor {
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): Int8Array;
  }

  interface Uint16ArrayConstructor {
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): Uint16Array;
  }

  interface Int16ArrayConstructor {
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): Int16Array;
  }

  interface Float16ArrayConstructor {
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): Float16Array;
  }

  interface BigInt64ArrayConstructor {
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): BigInt64Array;
  }

  interface BigUint64ArrayConstructor {
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): BigUint64Array;
  }

  interface Float64ArrayConstructor {
    new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): Float64Array;
  }
}

type DataTypeMappingLookupKey =
    DType |
    DataTypeMapping['typedArrayConstructor'] |
    TypedArray |
    DataTypeMapping['elementType'] |
    ElementType;

/**
 * Look up a DataTypeMapping from a datatype string, TypedArray constructor,
 * or LiteRT element type.
 */
export function getDataType(val: DataTypeMappingLookupKey): DataTypeMapping {
  // Note that:
  //   - DataTypeMapping['dtype']
  //   - DataTypeMapping['typedArrayConstructor']
  //   - DataTypeMapping['elementType']
  // store disjoint types, so `val` will always match exactly one (or zero)
  // DataTypeMapping.
  for (const dataTypeMapping of DATATYPES) {
    if (dataTypeMapping.dtype === val ||
        dataTypeMapping.typedArrayConstructor === val ||
        (dataTypeMapping.typedArrayConstructor &&
         val instanceof dataTypeMapping.typedArrayConstructor) ||
        dataTypeMapping.elementType === val) {
      return dataTypeMapping;
    }
  }

  // Error handling
  if (typeof val === 'string') {
    throw new Error(`DType ${val} is not supported.`);
  } else if (val instanceof Object) {
    throw new Error(`Typed array ${
                    'name' in val ? val.name :
                                    val.constructor.name} is not supported.`);
  } else {
    throw new Error(`Element type ${
        ElementTypeName[val as keyof typeof ElementTypeName] ??
        val} is not supported.`);
  }
}

/**
 * Check if a DType is supported in the current environment.
 */
export function isDTypeSupported(dtype: DType): boolean {
  const mapping = DATATYPES.find(d => d.dtype === dtype);
  if (!mapping) return false;
  return mapping.typedArrayConstructor !== undefined;
}