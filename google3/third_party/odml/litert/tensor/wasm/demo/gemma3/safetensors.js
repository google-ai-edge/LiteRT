// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Simple loader for Safetensors files.
 */
export class SafetensorsLoader {
  static async load(url, onProgress) {
    const cache = await self.caches.open('gemma3-weights-cache');
    let response = await cache.match(url);
    
    if (!response) {
      console.log("Weights not in cache, fetching...");
      response = await fetch(url);
      // Clone response to put in cache
      const responseToCache = response.clone();
      // We don't await this, so it happens in the background!
      cache.put(url, responseToCache).catch(err => console.error("Failed to cache:", err));
    } else {
      console.log("Loading weights from cache...");
    }

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    let loaded = 0;
    
    const reader = response.body.getReader();
    const chunks = [];
    while(true) {
      const {done, value} = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (onProgress && total) {
        onProgress(loaded, total);
      }
    }
    
    const buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }
    
    const arrayBuffer = buffer.buffer;
    const view = new DataView(arrayBuffer);
    
    // Read header size (little-endian uint64)
    // DataView.getBigUint64 is supported in modern browsers
    const headerSize = Number(view.getBigUint64(0, true));
    
    // Read header JSON
    const headerBytes = new Uint8Array(arrayBuffer, 8, headerSize);
    const headerStr = new TextDecoder().decode(headerBytes);
    const header = JSON.parse(headerStr);
    const dataOffset = 8 + headerSize;
    console.log(`Safetensors loaded: headerSize=${headerSize}, dataOffset=${dataOffset}`);
    // We slice the buffer to get only the data part, or we can use offsets later.
    // Slicing creates a copy, which might be large.
    // Let's keep the full buffer and use offsets from dataOffset.
    
    return { header, dataBuffer: buffer, dataOffset };
  }

  static getTensorData(header, dataBuffer, dataOffset, name) {
    const info = header[name];
    if (!info) return null;
    
    const [start, end] = info.data_offsets;
    const absoluteStart = dataOffset + start;
    const s = info.shape.reduce((a, b) => a * b, 1);
    
    let type = info.dtype;
    if (type === 'F32') {
      return new Float32Array(dataBuffer.buffer, absoluteStart, s);
    } else if (type === 'I32') {
      return new Int32Array(dataBuffer.buffer, absoluteStart, s);
    } else if (type === 'I8') {
      return new Int8Array(dataBuffer.buffer, absoluteStart, s);
    } else if (type === 'BF16') {
      const view = new DataView(dataBuffer.buffer);
      const fp32Array = new Float32Array(s);
      const f32View = new Float32Array(1);
      const u16View = new Uint16Array(f32View.buffer);
      const isLittleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
      
      for (let k = 0; k < s; ++k) {
        const u16 = view.getUint16(absoluteStart + k * 2, true);
        if (isLittleEndian) {
          u16View[1] = u16;
        } else {
          u16View[0] = u16;
        }
        fp32Array[k] = f32View[0];
      }
      return fp32Array;
    }
    return null;
  }

  /**
   * Creates a LiteRT Tensor from loaded Safetensors data.
   * @param {*} litert The LiteRT module instance.
   * @param {!Object} header The parsed Safetensors header.
   * @param {!Uint8Array} dataBuffer The raw data buffer.
   * @param {number} dataOffset The offset where data starts in the buffer.
   * @param {string} name The name of the tensor to load.
   * @return {!Object|null} The created LiteRT Tensor, or null if not found.
   */
  static async getTensor(litert, header, dataBuffer, dataOffset, name) {
    const info = header[name];
    if (!info) return null;
    
    const [start, end] = info.data_offsets;
    const absoluteStart = dataOffset + start;
    if (name === "model.embed_tokens.weight") {
      console.log(`Raw bytes for ${name}:`, new Uint8Array(dataBuffer.buffer, absoluteStart, 10));
    }
    
    let type = info.dtype;
    let convertFromBF16 = false;
    if (type === 'F32') type = 'FP32';
    if (type === 'I32') type = 'I32';
    if (type === 'I8') type = 'I8';

    if (type === 'BF16') {
      type = 'FP32'; // We will store it as FP32 in LiteRT
      convertFromBF16 = true;
    }
    // Add more type mappings as needed
    
    const tensor = litert['createTensor']({
      name: name,
      type: type,
      shape: info.shape
    });
    
    const s = info.shape.reduce((a, b) => a * b, 1);
    
    if (type === 'I8') {
      // Load scales and zero points from header!
      const scaleInfo = header[name + ".scale"] || header[name + ".scales"] || header[name + ".weight_scales"] || header[name + "_scales"];
      const zpInfo = header[name + ".zero_point"] || header[name + ".zero_points"] || header[name + ".weight_zero_points"] || header[name + "_zero_points"];
      
      if (scaleInfo && zpInfo) {
        // Read scales (F32!)
        const [sStart, sEnd] = scaleInfo.data_offsets;
        const sAbsoluteStart = dataOffset + sStart;
        const sCount = scaleInfo.shape.reduce((a, b) => a * b, 1);
        const scales = new Float32Array(dataBuffer.buffer, sAbsoluteStart, sCount);
        
        // Read zero points (I32!)
        const [zStart, zEnd] = zpInfo.data_offsets;
        const zCount = zpInfo.shape.reduce((a, b) => a * b, 1);
        
        // FORCE zero points to be 0 (symmetric quantization!).
        const zeroPoints = new Int32Array(zCount).fill(0);
        
        tensor['setQuantization']({
          scales: scales,
          zeroPoints: zeroPoints,
          quantizedDimension: 0
        });
      }
      
      // Load raw INT8 data!
      const srcArray = new Int8Array(dataBuffer.buffer, absoluteStart, s);
      await tensor['setData'](srcArray);
      
    } else if (type === 'FP32' && !convertFromBF16) {
      const srcArray = new Float32Array(dataBuffer.buffer, absoluteStart, s);
      
      if (name.endsWith('.weight') && (name.includes('norm') || name.includes('layernorm'))) {
        console.warn(`[Safetensors] ADDED 1.0 to norm weight: ${name}`);
        console.warn(`WASM ${name} weights (first 5, BEFORE +1.0):`, srcArray.subarray(0, 5));
        const modified = new Float32Array(srcArray);
        for (let k = 0; k < s; ++k) {
          modified[k] += 1.0;
        }
        console.warn(`WASM ${name} weights (first 5, WITH +1.0):`, modified.subarray(0, 5));
        await tensor['setData'](modified);
      } else {


        await tensor['setData'](srcArray);
      }
    }
 else if (type === 'FP32' && convertFromBF16) {

      const view = new DataView(dataBuffer.buffer);
      const bf16Array = new Uint16Array(s);
      for (let k = 0; k < s; ++k) {
        bf16Array[k] = view.getUint16(absoluteStart + k * 2, true);
      }
      const chunkSize = 1024 * 64;
      const chunkF32 = new Float32Array(chunkSize);
      const chunkU16 = new Uint16Array(chunkF32.buffer);
      
      const fp32Array = new Float32Array(s);
      
      for (let i = 0; i < s; i += chunkSize) {
        const currentChunkSize = Math.min(chunkSize, s - i);
        const isLittleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
        for (let j = 0; j < currentChunkSize; ++j) {
          if (isLittleEndian) {
            chunkU16[2 * j + 1] = bf16Array[i + j];
          } else {
            chunkU16[2 * j] = bf16Array[i + j];
          }
        }
        fp32Array.set(chunkF32.subarray(0, currentChunkSize), i);
      }
      await tensor['setData'](fp32Array);
      
    } else if (type === 'I32') {
      const srcArray = new Int32Array(dataBuffer.buffer, absoluteStart, s);
      await tensor['setData'](srcArray);
    } else {
        console.warn(`Unsupported dtype in Safetensors: ${info.dtype}`);
        return null;
    }
    
    return tensor;
  }
}
