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
 * Converts a URL to a Uint8Array.
 */
export async function urlToUint8Array(url: string|URL): Promise<Uint8Array> {
  const response = await fetch(url);
  return new Uint8Array(await response.arrayBuffer());
}

/**
 * Converts a ReadableStreamDefaultReader to a Uint8Array.
 */
export async function readableStreamDefaultReaderToUint8Array(
    reader: ReadableStreamDefaultReader<Uint8Array>): Promise<Uint8Array> {
  let byteOffset = 0;
  let array = new Uint8Array(1024 /* arbitrary starting size */);
  const MAX_ARRAY_SIZE = 2e9;  // Chrome gets flaky with sizes > 2GB.

  // Collecting all the chunks and then copying them would be easier, but this
  // is more memory efficient.
  while (true) {
    const {done, value} = await reader.read();
    if (value) {
      if (array.byteLength < byteOffset + value.byteLength) {
        if (byteOffset + value.byteLength > MAX_ARRAY_SIZE) {
          throw new Error(`Model is too large (> ${MAX_ARRAY_SIZE} bytes).`);
        }

        // Allocate more space, but double the size to avoid reallocating too
        // often.
        // Note: This will not work with huge models since we store everything
        // in one ArrayBuffer, but more things will need to be refactored for
        // those anyway.
        const newArray = new Uint8Array(Math.min(
            MAX_ARRAY_SIZE, Math.max(array.byteLength, value.byteLength) * 2));
        newArray.set(array);
        array = newArray;
      }
      array.set(value, byteOffset);
      byteOffset += value.byteLength;
    }
    if (done) {
      break;
    }
  }

  // Resize to the exact byte length. Could use `.subarray`, but we'd like to
  // avoid keeping the extra bytes allocated.
  return array.slice(0, byteOffset);
}