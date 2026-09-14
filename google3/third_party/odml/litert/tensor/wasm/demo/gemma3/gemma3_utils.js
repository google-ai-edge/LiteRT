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

import { SafetensorsLoader } from './safetensors.js?v=1084';

/**
 * Performs on-the-fly symmetrical INT4 per-row de-quantization embedding vector extraction on JS CPU space.
 * Extracts and maps two 4-bit signed integers from every single byte dynamically back to FP32.
 * 
 * @param {!Array<number>|!Int32Array|!Object} tokens - List or dynamic sequence of input token identifiers.
 * @param {!Uint8Array} packedUint8 - The dynamic packed I4 embeddings weight table loaded from Safetensors.
 * @param {!Float32Array} scalesFloat32 - Row-wise Float32 de-quantization scales multipliers arrays.
 * @param {number} vocabSize - Token index boundaries ceiling (vocab_size).
 * @param {number} embDim - Embedding channels dimensions variables (emb_dim).
 * @returns {!Float32Array} Flat Float32Array activations vector of shape [seqLen * embDim].
 */
export function embeddingLookupCpuINT4(tokens, packedUint8, scalesFloat32, vocabSize, embDim) {
  const seqLen = tokens.size ? tokens.size() : tokens.length;
  const embeddings = new Float32Array(seqLen * embDim);
  
  for (let i = 0; i < seqLen; ++i) {
    let tokenId = tokens.get ? tokens.get(i) : tokens[i];
    if (tokenId < 0 || tokenId >= vocabSize) {
      tokenId = 0;
    }
    
    const rowScale = scalesFloat32[tokenId];
    const rowOffsetBytes = (tokenId * embDim) / 2;
    const outOffset = i * embDim;
    
    // Extract and symmetrically de-quantize two 4-bit integers per byte on the fly!
    for (let d = 0; d < embDim; d += 2) {
      const byteIdx = rowOffsetBytes + (d / 2);
      const unsignedByte = packedUint8[byteIdx];
      
      const unsignedA = unsignedByte & 0x0F;
      const unsignedB = (unsignedByte >> 4) & 0x0F;
      
      // Two's complement signed integers mapping [-8 to 7] math bounds
      const quantA = (unsignedA >= 8) ? (unsignedA - 16) : unsignedA;
      const quantB = (unsignedB >= 8) ? (unsignedB - 16) : unsignedB;
      
      embeddings[outOffset + d] = quantA * rowScale;
      embeddings[outOffset + d + 1] = quantB * rowScale;
    }
  }
  return embeddings;
}

/**
 * Performs unquantized, highly stable Float32 (FP32) input embedding lookup on JS CPU space.
 * Bypasses all quantization precision rounding noise anomalies at Layer 0 entry points.
 * 
 * @param {!Array<number>|!Int32Array|!Object} tokens - List or sequence array containing token index elements.
 * @param {!Float32Array} embeddingTable - The flat unquantized Float32 Embeddings table loaded from Safetensors.
 * @param {number} vocabSize - Dynamic vocabulary boundaries limit.
 * @param {number} embDim - Embedding channels channels limits (emb_dim).
 * @returns {!Float32Array} The high-fidelity FP32 embeddings activations vector of shape [seqLen * embDim].
 */
export function embeddingLookupCpuFP32(tokens, embeddingTable, vocabSize, embDim) {
  const seqLen = tokens.size ? tokens.size() : tokens.length;
  const embeddings = new Float32Array(seqLen * embDim);
  
  for (let i = 0; i < seqLen; ++i) {
    let tokenId = tokens.get ? tokens.get(i) : tokens[i];
    if (tokenId < 0 || tokenId >= vocabSize) {
      tokenId = 0;
    }
    const srcOffset = tokenId * embDim;
    for (let j = 0; j < embDim; ++j) {
      embeddings[i * embDim + j] = embeddingTable[srcOffset + j];
    }
  }
  return embeddings;
}

/**
 * Pre-computes dynamic un-shifted RoPE rotary positional embedding cos and sin values tables for a given step position index.
 * Creates standard shape vectors containing pre-computed parameters for dynamic WebGPU shaders input uploads.
 * 
 * @param {number} position - Symmetrical current autoregressive generation step sequence position index (cacheLen).
 * @param {number} headDim - Query head variables dim context boundaries limit (head_dim).
 * @param {number} ropeBase - Positional base temperature scaling coefficient bounds limit.
 * @returns {!object} An object containing cosValues and sinValues arrays of head dimensions limits.
 */
export function computeRopeCosSinForPosition(position, headDim, ropeBase) {
  const cosValues = new Float32Array(headDim);
  const sinValues = new Float32Array(headDim);
  const halfDim = headDim / 2;
  for (let i = 0; i < halfDim; ++i) {
    const freq = 1.0 / Math.pow(ropeBase, 2.0 * i / headDim);
    const angle = position * freq;
    const cosVal = Math.cos(angle);
    const sinVal = Math.sin(angle);
    cosValues[i] = cosVal;
    cosValues[halfDim + i] = cosVal;
    sinValues[i] = sinVal;
    sinValues[halfDim + i] = sinVal;
  }
  return { cosValues, sinValues };
}

/**
 * Asynchronously loads and instantiates the dynamic SentencePiece Tokenizer WASM module under sandboxed boundaries.
 * Writes the `.model` flat buffer directly into sandboxed WASM filesystem workspace dynamically.
 * 
 * @param {!Function} createTokenizerModuleFn - Webpack/loader function to initialize tokenizer Emscripten modules.
 * @param {string} modelUrl - Symmetrical tokenizer model path URL.
 * @returns {!Promise<!object>} A promise resolving to a loaded, active native SentencePiece Tokenizer object context.
 * @throws {Error} If tokenizer initialization exceptions throw in runtime.
 */
export async function loadTokenizer(createTokenizerModuleFn, modelUrl) {
  const tokenizerModule = await createTokenizerModuleFn();
  const tokenizerResponse = await fetch(modelUrl);
  const tokenizerBytes = await tokenizerResponse.arrayBuffer();
  tokenizerModule.FS.writeFile('/tokenizer.model', new Uint8Array(tokenizerBytes));
  const tokenizer = tokenizerModule.loadTokenizer('/tokenizer.model');
  
  if (tokenizer.isNull()) {
    throw new Error("Failed to load tokenizer.");
  }
  return tokenizer;
}

/**
 * Asynchronously loads, parsing Safetensors filesets URL headers, allocating and calibrating unquantized weights.
 * Correctly applies the standard layernorm variance offset factor calibrations (+1.0 fixes) natively inside graph inputs.
 * 
 * @param {!object} litert - LiteRT module context bindings containing creating tensor handles API.
 * @param {!GPUDevice} gpuDevice - Active hardware WebGPU device.
 * @param {string} url - Safetensors model parameters checkpoint paths URL.
 * @param {!Function} onProgress - Optional progress callback triggered during downloads pipelines.
 * @param {!Function} onWeightLoad - Optional callback triggered cross single parameter tables mappings steps.
 * @returns {!Promise<!object>} A promise resolving to a dictionary containing high-precision dynamic Float32/tensor weights tables mapping.
 */
export async function loadWeights(litert, gpuDevice, url, onProgress, onWeightLoad) {
  const { header, dataBuffer, dataOffset } = await SafetensorsLoader.load(url, onProgress);
  
  const rawWeights = {};
  for (const name in header) {
    rawWeights[name] = SafetensorsLoader.getTensorData(header, dataBuffer, dataOffset, name);
  }

  const weights = {};
  for (const name in rawWeights) {
    if (onWeightLoad) {
      onWeightLoad(name);
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    if (!name.startsWith("model.") && !name.startsWith("lm_head.")) {
      continue;
    }

    const data = rawWeights[name];
    if (!data) continue;

    if (name === "model.embed_tokens.weight") {
      weights[name] = data;
      continue;
    }

    // Regular tensor (not scales, and not quantized weight)
    const tensor = await SafetensorsLoader.getTensor(litert, header, dataBuffer, dataOffset, name);
    if (!tensor) continue;

    // Apply scale+1 fix for norm weights if needed (as in original code)
    if (name.endsWith('.input_layernorm.weight') ||
        name.endsWith('.post_attention_layernorm.weight') ||
        name.endsWith('.pre_feedforward_layernorm.weight') ||
        name.endsWith('.post_feedforward_layernorm.weight') ||
        name.endsWith('.self_attn.q_norm.weight') ||
        name.endsWith('.self_attn.k_norm.weight') ||
        name === 'model.norm.weight') {
      const newData = new Float32Array(data.length);
      for (let i = 0; i < data.length; ++i) {
        newData[i] = data[i] + 1.0;
      }
      await tensor.setData(newData);
    }

    weights[name] = tensor;
  }
  return weights;
}
