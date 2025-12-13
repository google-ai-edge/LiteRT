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

// Only used in this file to ensure that DATATYPES follows a consistent pattern.
// Do not export this type because it's not specific enough.
interface DataTypeMappingInternal {
  dtype: string;
  typedArrayConstructor: {new(...args: unknown[]): unknown};
  elementType: number;
}

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
    dtype: 'float32',
    typedArrayConstructor: Float32Array,
    elementType: ElementType.FLOAT32
  } as const,
  {
    dtype: 'int32',
    typedArrayConstructor: Int32Array,
    elementType: ElementType.INT32
  } as const,
  {
    dtype: 'uint8',
    typedArrayConstructor: Uint8Array,
    elementType: ElementType.UINT8
  } as const,
] as const satisfies DataTypeMappingInternal[]);

/**
 * Defines how a given datatype is mapped to a string, TypedArray constructor,
 * and LiteRT element type.
 */
export type DataTypeMapping = (typeof DATATYPES)[number];

/**
 * The data type of a Tensor.
 */
export type DType = DataTypeMapping['dtype'];

/**
 * The constructor for a TypedArray.
 */
export type TypedArrayConstructor = DataTypeMapping['typedArrayConstructor'];

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

type DataTypeMappingLookupKey =
    DataTypeMapping['dtype']|DataTypeMapping['typedArrayConstructor']|
    InstanceType<DataTypeMapping['typedArrayConstructor']>|
    DataTypeMapping['elementType']|ElementType;

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
        val instanceof dataTypeMapping.typedArrayConstructor ||
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
    throw new Error(
        `Element type ${ElementTypeName[val] ?? val} is not supported.`);
  }
}