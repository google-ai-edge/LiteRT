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

import { Gemma3GraphBuilder } from './gemma3_graph.js?v=1084';
import { embeddingLookupCpuFP32, computeRopeCosSinForPosition } from './gemma3_utils.js?v=1084';

/**
 * Maps an Emscripten C++ buffer handle to a standard JavaScript WebGPU GPUBuffer instance.
 * Handles different possible Embind integration namespaces dynamically.
 * 
 * @param {number} wgpuBufferId - The registered C++ WebGPU buffer identifier handle.
 * @param {!object} module - Emscripten module reference containing WebGPU bindings.
 * @returns {!GPUBuffer} The mapped standard GPUBuffer instance.
 * @throws {Error} If Emscripten WebGPU bindings cannot be located in runtime heap.
 */
function getGpuBuffer(wgpuBufferId, module) {
  if (typeof WebGPU !== 'undefined' && WebGPU.getJsObject) {
    return WebGPU.getJsObject(wgpuBufferId);
  }
  if (module && module.WebGPU && module.WebGPU.getJsObject) {
    return module.WebGPU.getJsObject(wgpuBufferId);
  }
  if (typeof Module !== 'undefined' && Module.WebGPU && Module.WebGPU.getJsObject) {
    return Module.WebGPU.getJsObject(wgpuBufferId);
  }
  throw new Error("Emscripten WebGPU integration not found! Cannot map C++ WGPUBuffer to JS GPUBuffer.");
}

/**
 * Safely uploads a Float32 activation data buffer directly to a VRAM input buffer in a single queue write pass.
 * Gracefully handles pruned or optimized tensors with null/undefined handles.
 * 
 * @param {!GPUDevice} device - The active hardware GPU device.
 * @param {number} wgpuBufferId - The target input buffer handle.
 * @param {!Float32Array|!Int32Array} data - Typed array containing data values to upload.
 * @param {!object} module - Emscripten module context containing WebGPU helper bindings.
 */
function writeGpuInput(device, wgpuBufferId, data, module) {
  if (!wgpuBufferId) return;
  const gpuBuffer = getGpuBuffer(wgpuBufferId, module);
  if (!gpuBuffer) return;
  
  device.queue.writeBuffer(gpuBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
}

/**
 * Asynchronously maps and reads back dynamic output tensors data from VRAM as Float32Array values.
 * Employs a temporary GPU staging read buffer copy command to safely bypass delegate mapping blocks.
 * 
 * @param {!GPUDevice} device - The active hardware WebGPU device interface.
 * @param {number} wgpuBufferId - The target output VRAM buffer handle to map and read.
 * @param {number} sizeBytes - Slices constraints size limits to copy from the output buffer.
 * @param {!object} module - Emscripten module reference containing bindings lookups.
 * @returns {!Promise<!Float32Array>} A promise resolving to a Float32 staging staging typed array data.
 */
async function readGpuOutput(device, wgpuBufferId, sizeBytes, module) {
  if (!wgpuBufferId) return new ArrayBuffer(0);
  const gpuBuffer = getGpuBuffer(wgpuBufferId, module);
  if (!gpuBuffer) return new ArrayBuffer(0);

  const actualSize = Math.min(sizeBytes, gpuBuffer.size);
  const stagingBuffer = device.createBuffer({
    size: actualSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(gpuBuffer, 0, stagingBuffer, 0, actualSize);

  device.queue.submit([commandEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = stagingBuffer.getMappedRange();
  const data = new Float32Array(arrayBuffer.slice(0));
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return data;
}

/**
 * Asynchronously reads dynamic output indices / labels tensors maps from VRAM as Int32Array.
 * Uses temporary WebGPU command encoder copy pipelines to staging buffers safely.
 * 
 * @param {!GPUDevice} device - The active hardware GPU device interface.
 * @param {number} wgpuBufferId - The target C++ output buffer handle to staging read.
 * @param {number} sizeBytes - Total bytes count limit to extract from the output buffer.
 * @param {!object} module - Emscripten module reference maps.
 * @returns {!Promise<!Int32Array>} A promise resolving to an Int32 mapping typed array data.
 */
async function readGpuInt32Output(device, wgpuBufferId, sizeBytes, module) {
  const gpuBuffer = getGpuBuffer(wgpuBufferId, module);

  const actualSize = Math.min(sizeBytes, gpuBuffer.size);
  const stagingBuffer = device.createBuffer({
    size: actualSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(gpuBuffer, 0, stagingBuffer, 0, actualSize);

  device.queue.submit([commandEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = stagingBuffer.getMappedRange();
  const data = new Int32Array(arrayBuffer.slice(0));
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return data;
}

/**
 * Samples dynamic Candidate vocabulary index elements using Softmax Temperature Random Multinomial selection.
 * Aggressively applies standard subtraction limits max log values parameters to prevent exponential arithmetic overflows.
 * 
 * @param {!Float32Array} scores - Dynamic Candidate Float32 Logits Log scores returned from in-graph Top-K Shaders.
 * @param {!Int32Array} indices - Mapped Int32 Candidate token vocabulary index labels from Top-K Shaders.
 * @param {number=} [temperature] - Inverse scale scaling Softmax temperature (Default 0.7).
 * @returns {number} The selected next Predicted token index vocabulary label.
 */
function multinomialSample(scores, indices, temperature = 0.7) {
  const numCandidates = scores.length;
  const scaledScores = new Float32Array(numCandidates);
  
  // 1. Apply temperature scaling
  for (let i = 0; i < numCandidates; ++i) {
    scaledScores[i] = scores[i] / temperature;
  }
  
  // 2. Prevent exponential float arithmetic overflows by subtracting max score
  let maxScore = scaledScores[0];
  for (let i = 1; i < numCandidates; ++i) {
    if (scaledScores[i] > maxScore) maxScore = scaledScores[i];
  }
  
  // 3. Softmax exps calculations
  const exps = new Float32Array(numCandidates);
  let sumExps = 0.0;
  for (let i = 0; i < numCandidates; ++i) {
    exps[i] = Math.exp(scaledScores[i] - maxScore);
    sumExps += exps[i];
  }
  
  // 4. Cumulative probability random selection multinomial loop
  const r = Math.random();
  let cumulativeProb = 0.0;
  for (let i = 0; i < numCandidates; ++i) {
    cumulativeProb += exps[i] / sumExps;
    if (r <= cumulativeProb) {
      return indices[i];
    }
  }
  return indices[numCandidates - 1];
}

export class Gemma3Runner {
  constructor(litert, device, tokenizer, weights, config, useGpu = false) {
    this.litert = litert;
    this.device = device;
    this.tokenizer = tokenizer;
    this.weights = weights;
    this.config = config;
    this.useGpu = useGpu;

    
    this.builder = new Gemma3GraphBuilder(litert, config);
    this.runner = null;
    
    this.maxSeqLen = 512; // Restored to standard context limits safely with INT4 memory efficiency!
    
    // Tensors that need to be kept alive or reused
    this.dummyInput = null;
    this.dummyCos = null;
    this.dummySin = null;
    this.dummyMask = null;
    this.dummyStartIndices = null;
    
    this.currentKeyCaches = [];
    this.currentValueCaches = [];
    this.inputKeyCaches = [];
    this.inputValueCaches = [];
    
    this.cacheLen = 0;
  }
  
  
  async setup() {
    const litert = this.litert;
    const config = this.config;
    const weights = this.weights;
    
    // Create dummy tensors for prefill graph
    this.dummyInputPrefill = litert.createTensor({ type: 'FP32', shape: [1, this.maxSeqLen, config.emb_dim] });
    this.dummyInputPrefill.setName("embedded_input");
    
    this.dummyCosPrefill = litert.createTensor({ type: 'FP32', shape: [1, 1, this.maxSeqLen, config.head_dim] });
    this.dummyCosPrefill.setName("rope_cos");
    
    this.dummySinPrefill = litert.createTensor({ type: 'FP32', shape: [1, 1, this.maxSeqLen, config.head_dim] });
    this.dummySinPrefill.setName("rope_sin");
    
    this.dummyCosLocalPrefill = litert.createTensor({ type: 'FP32', shape: [1, 1, this.maxSeqLen, config.head_dim] });
    this.dummyCosLocalPrefill.setName("rope_local_cos");
    
    this.dummySinLocalPrefill = litert.createTensor({ type: 'FP32', shape: [1, 1, this.maxSeqLen, config.head_dim] });
    this.dummySinLocalPrefill.setName("rope_local_sin");
    
    this.dummyMaskPrefill = litert.createTensor({ type: 'FP32', shape: [1, 1, this.maxSeqLen, this.maxSeqLen] });
    this.dummyMaskPrefill.setName("attention_mask");
    
    this.dummyPrefillSliceIndex = litert.createTensor({ type: 'I32', shape: [1] });
    this.dummyPrefillSliceIndex.setName("prefill_slice_index");

    // Create dummy tensors for decode graph
    this.dummyInputDecode = litert.createTensor({ type: 'FP32', shape: [1, 1, config.emb_dim] });
    this.dummyInputDecode.setName("embedded_input");
    
    this.dummyCosDecode = litert.createTensor({ type: 'FP32', shape: [1, 1, 1, config.head_dim] });
    this.dummyCosDecode.setName("rope_cos");
    
    this.dummySinDecode = litert.createTensor({ type: 'FP32', shape: [1, 1, 1, config.head_dim] });
    this.dummySinDecode.setName("rope_sin");
    
    this.dummyCosLocalDecode = litert.createTensor({ type: 'FP32', shape: [1, 1, 1, config.head_dim] });
    this.dummyCosLocalDecode.setName("rope_local_cos");
    
    this.dummySinLocalDecode = litert.createTensor({ type: 'FP32', shape: [1, 1, 1, config.head_dim] });
    this.dummySinLocalDecode.setName("rope_local_sin");

    
    this.dummyMaskDecode = litert.createTensor({ type: 'FP32', shape: [1, 1, 1, this.maxSeqLen] });
    this.dummyMaskDecode.setName("attention_mask");

    // Splitting input placeholders parameters list completely pruned to unblock setups CPU Heap JIT spikes!
    this.dummyLmHeadWeightsPart = [];

    // Shared dummy tensors start indices to align with graphs
    this.dummyStartIndices = litert.createTensor({ type: 'I32', shape: [4] });
    this.dummyStartIndices.setName("start_indices");

    this.currentKeyCaches = [];
    this.currentValueCaches = [];
    this.inputKeyCaches = [];
    this.inputValueCaches = [];
    for (let i = 0; i < config.n_layers; ++i) {
      const k = litert.createTensor({ type: 'FP32', shape: [1, 1, this.maxSeqLen, config.head_dim] });
      k.setName(`key_cache_${i}`);
      const v = litert.createTensor({ type: 'FP32', shape: [1, 1, this.maxSeqLen, config.head_dim] });
      v.setName(`value_cache_${i}`);
      
      this.currentKeyCaches.push(k);
      this.currentValueCaches.push(v);
    }

    // Dynamic JavaScript-packed INT4 Symmetrical Per-Row Quantization setups!
    console.log("[Setup] Packaging Gemma 3 Embedding weights lookup to INT4 (I4) Constant Graph matrices...");
    const embedWeightTensor = weights["model.embed_tokens.weight"];
    const embedDataView = (embedWeightTensor.getData ? await embedWeightTensor.getData() : embedWeightTensor); // Direct float memory pointers view support!
    
    const vocabSize = config.vocab_size;
    const embDim = config.emb_dim;
    
    const packedBytesCount = (vocabSize * embDim) / 2; // exactly 83,886,080 bytes (80 MB)!
    const packedUint8 = new Uint8Array(packedBytesCount);
    const scalesFloat32 = new Float32Array(vocabSize);
    
    // Symmetrical INT4 Per-Row min-max scaling limits loops!
    for (let v = 0; v < vocabSize; ++v) {
      let maxAbs = 0.0;
      const rowOffset = v * embDim;
      for (let d = 0; d < embDim; ++d) {
        const val = Math.abs(embedDataView[rowOffset + d]);
        if (val > maxAbs) maxAbs = val;
      }
      
      // scale to signed 4-bit elements [-8 to 7] limits math
      const scale = maxAbs > 0 ? (maxAbs / 8.0) : 1.0;
      scalesFloat32[v] = scale;
      
      const invScale = 1.0 / scale;
      for (let d = 0; d < embDim; d += 2) {
        // Symmetrically pack two signed 4-bit integers per one byte!
        const valA = embedDataView[rowOffset + d];
        const valB = embedDataView[rowOffset + d + 1];
        
        const quantA = Math.max(-8, Math.min(7, Math.round(valA * invScale)));
        const quantB = Math.max(-8, Math.min(7, Math.round(valB * invScale)));
        
        const unsignedA = quantA & 0x0F;
        const unsignedB = quantB & 0x0F;
        
        const byteIdx = (rowOffset + d) / 2;
        packedUint8[byteIdx] = (unsignedB << 4) | unsignedA;
      }
    }
    
    console.log("[Setup] Serializing standard INT4 Constant weight table directly inside graph Flatbuffer...");
    const lmHeadWeightsConstant = litert.createTensor({
      name: "lm_head.weight",
      type: "I4", // Fully unblocked for standard setData copies support natively in WASM!
      shape: [vocabSize, embDim]
    });
    await lmHeadWeightsConstant.setData(packedUint8);
    lmHeadWeightsConstant.setQuantization({
      scales: scalesFloat32,
      zeroPoints: new Int32Array(vocabSize).fill(0),
      quantizedDimension: 0
    });

    // Build prefill graph using our static constant weight matrix handle, skipping logits to avoid memory OOM!
    const prefillGraph = await this.builder.buildGemma3FromEmbeddings(
      this.dummyInputPrefill,
      this.dummyCosPrefill, this.dummySinPrefill,
      this.dummyCosLocalPrefill, this.dummySinLocalPrefill,
      this.dummyMaskPrefill,
      this.dummyMaskPrefill,
      this.dummyStartIndices,
      this.dummyPrefillSliceIndex,
      this.currentKeyCaches,
      this.currentValueCaches,
      weights,
      lmHeadWeightsConstant,
      false
    );
    
    // Build decode graph using our static constant weight matrix handle, retaining FC logits for inferences!
    const decodeGraph = await this.builder.buildGemma3FromEmbeddings(
      this.dummyInputDecode,
      this.dummyCosDecode, this.dummySinDecode,
      this.dummyCosLocalDecode, this.dummySinLocalDecode,
      this.dummyMaskDecode,
      this.dummyMaskDecode,
      this.dummyStartIndices,
      this.dummyPrefillSliceIndex,
      this.currentKeyCaches,
      this.currentValueCaches,
      weights,
      lmHeadWeightsConstant,
      true
    );

    this.builder.promises = [];

    // Create multi-signature runner
    const signatures = {
      "prefill": {
        outputs: [prefillGraph.topk_values, prefillGraph.topk_indices, ...prefillGraph.updatedKeyCaches, ...prefillGraph.updatedValueCaches]
      },
      "decode": {
        outputs: [decodeGraph.topk_values, decodeGraph.topk_indices, ...decodeGraph.updatedKeyCaches, ...decodeGraph.updatedValueCaches]
      }
    };

    const accelerator = this.useGpu ? litert.HwAccelerators.GPU_WEBGPU : litert.HwAccelerators.CPU;
    console.log(`[Setup] Creating MultiSignatureRunner (useGpu=${this.useGpu})...`);

    this.runner = await litert.createMultiSignatureRunner(signatures, accelerator);
    
    // Get input tensors from runner to use their WebGPU buffers for decode
    for (let i = 0; i < config.n_layers; ++i) {
      this.inputKeyCaches.push(this.runner.getInput("decode", `key_cache_${i}`));
      this.inputValueCaches.push(this.runner.getInput("decode", `value_cache_${i}`));
    }
    
    // Multi-Signature Runner compiles easily! JIT models serialization setup spikes reduced safely!
    console.log("[Setup] MultiSignature Runner created successfully! Static INT4 constants embedded directly inside Flatbuffer structures.");
    
    // Store dynamic packed tables references locally for CPU lookups!
    this.packedEmbeddingWeights = packedUint8;
    this.embeddingScales = scalesFloat32;
    this.embedTokensWeightsFP32 = embedDataView; // Cache dynamic FP32 embed activations weights on JS CPU space victoriously!
    
    // WIPE 100% of remaining transformer layers Float32 weights buffers from CPU memory entirely to save memory!
    this.weights = {};
    console.log("[Setup] Freed remaining dynamic layers Float32 weights CPU Javascript heap memory safely.");
  }

  async resetKVCache() {
    console.log("[Runner] Resetting persistent runner graph cache states and offsets to 0...");
    this.cacheLen = 0;
    console.log("[Runner] Persistent model graph state successfully reset for the next turn.");
  }

  async prefill(tokens) {
    this.cacheLen = 0;
    
    // Parallel prefill: process all prompt tokens at once!
    const tokenVec = [];
    const seqLen = tokens.size ? tokens.size() : tokens.length;
    for (let i = 0; i < seqLen; ++i) {
      tokenVec.push(tokens.get ? tokens.get(i) : tokens[i]);
    }
    console.log(`[Prefill] Processing ${seqLen} dynamic prompt tokens...`);
    
    const tokenEmbedding = embeddingLookupCpuFP32(tokenVec, this.embedTokensWeightsFP32, this.config.vocab_size, this.config.emb_dim);
    const embScale = Math.sqrt(this.config.emb_dim);
    const scaledEmbedding = tokenEmbedding.map(x => x * embScale);
    
    let effectiveSeqLen = seqLen;
    if (effectiveSeqLen > this.maxSeqLen) {
      console.warn(`Prompt too long (${seqLen} tokens), truncating to ${this.maxSeqLen}`);
      effectiveSeqLen = this.maxSeqLen;
    }
    
    const paddedInput = new Float32Array(this.maxSeqLen * this.config.emb_dim);
    paddedInput.set(scaledEmbedding.subarray(0, effectiveSeqLen * this.config.emb_dim));

    const globalCosData = new Float32Array(this.maxSeqLen * this.config.head_dim);
    const globalSinData = new Float32Array(this.maxSeqLen * this.config.head_dim);
    const localCosData = new Float32Array(this.maxSeqLen * this.config.head_dim);
    const localSinData = new Float32Array(this.maxSeqLen * this.config.head_dim);
    
    const ropeLocalBase = this.config.rope_local_base || 10000.0;
    
    for (let i = 0; i < this.maxSeqLen; ++i) {
      const globalRoPE = computeRopeCosSinForPosition(i, this.config.head_dim, this.config.rope_global_base);
      globalCosData.set(globalRoPE.cosValues, i * this.config.head_dim);
      globalSinData.set(globalRoPE.sinValues, i * this.config.head_dim);
      
      const localRoPE = computeRopeCosSinForPosition(i, this.config.head_dim, ropeLocalBase);
      localCosData.set(localRoPE.cosValues, i * this.config.head_dim);
      localSinData.set(localRoPE.sinValues, i * this.config.head_dim);
    }
    
    const maskData = new Float32Array(this.maxSeqLen * this.maxSeqLen).fill(-1e9);
    for (let i = 0; i < this.maxSeqLen; ++i) {
      for (let j = 0; j <= i; ++j) {
        if (i < seqLen && j < seqLen) {
          maskData[i * this.maxSeqLen + j] = 0.0;
        }
      }
    }

    // Write inputs
    if (this.useGpu) {
      writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("prefill", "embedded_input"), paddedInput, this.litert);
      writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("prefill", "rope_cos"), globalCosData, this.litert);
      writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("prefill", "rope_sin"), globalSinData, this.litert);
      writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("prefill", "rope_local_cos"), localCosData, this.litert);
      writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("prefill", "rope_local_sin"), localSinData, this.litert);
      writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("prefill", "start_indices"), new Int32Array([0, 0, 0, 0]), this.litert);
      writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("prefill", "prefill_slice_index"), new Int32Array([seqLen - 1]), this.litert);
      writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("prefill", "attention_mask"), maskData, this.litert);

      this.device.pushErrorScope('validation');
      this.device.pushErrorScope('out-of-memory');
      this.device.pushErrorScope('internal');
    } else {
      this.runner.setInputBinary("prefill", "embedded_input", paddedInput);
      this.runner.setInputBinary("prefill", "rope_cos", globalCosData);
      this.runner.setInputBinary("prefill", "rope_sin", globalSinData);
      this.runner.setInputBinary("prefill", "rope_local_cos", localCosData);
      this.runner.setInputBinary("prefill", "rope_local_sin", localSinData);
      this.runner.setInputBinary("prefill", "start_indices", new Int32Array([0, 0, 0, 0]));
      this.runner.setInputBinary("prefill", "prefill_slice_index", new Int32Array([seqLen - 1]));
      this.runner.setInputBinary("prefill", "attention_mask", maskData);
    }
    
    const success = await this.runner.run("prefill");
    
    if (this.useGpu) {
      const internalError = await this.device.popErrorScope();
      const oomError = await this.device.popErrorScope();
      const validationError = await this.device.popErrorScope();
      
      if (validationError) console.error(`WebGPU Validation Error (prefill):`, validationError.message || validationError);
      if (oomError) console.error(`WebGPU OOM Error (prefill):`, oomError.message || oomError);
      if (internalError) console.error(`WebGPU Internal Error (prefill):`, internalError.message || internalError);
    }

    if (!success) {
      console.error("Runner.run('prefill') failed!");
      return;
    }
    
    // Persist Prefill KV Cache to Decode Inputs!
    if (this.useGpu) {
      const commandEncoder = this.device.createCommandEncoder();
      for (let i = 0; i < this.config.n_layers; ++i) {
        const outKeyId = this.runner.getOutputWebGpuBuffer("prefill", `output_key_cache_${i}`);
        const outValId = this.runner.getOutputWebGpuBuffer("prefill", `output_value_cache_${i}`);
        const inKeyId = this.runner.getInputWebGpuBuffer("decode", `key_cache_${i}`);
        const inValId = this.runner.getInputWebGpuBuffer("decode", `value_cache_${i}`);
        
        const outKeyGpu = getGpuBuffer(outKeyId, this.litert);
        const outValGpu = getGpuBuffer(outValId, this.litert);
        const inKeyGpu = getGpuBuffer(inKeyId, this.litert);
        const inValGpu = getGpuBuffer(inValId, this.litert);

        
        commandEncoder.copyBufferToBuffer(outKeyGpu, 0, inKeyGpu, 0, outKeyGpu.size);
        commandEncoder.copyBufferToBuffer(outValGpu, 0, inValGpu, 0, outValGpu.size);
      }
      this.device.queue.submit([commandEncoder.finish()]);
    } else {
      // Copy and Pad Prefill KV Cache to Decode Inputs (CPU Fallback)!
      for (let i = 0; i < this.config.n_layers; ++i) {
        const keyTensor = this.runner.getOutput("prefill", `output_key_cache_${i}`);
        const valTensor = this.runner.getOutput("prefill", `output_value_cache_${i}`);
        const keyData = await keyTensor.getData(); 
        const valData = await valTensor.getData();
        
        const paddedKey = new Float32Array(this.maxSeqLen * 256);
        const paddedVal = new Float32Array(this.maxSeqLen * 256);
        
        paddedKey.set(keyData);
        paddedVal.set(valData);
        
        this.runner.setInputBinary("decode", `key_cache_${i}`, new Uint8Array(paddedKey.buffer));
        this.runner.setInputBinary("decode", `value_cache_${i}`, new Uint8Array(paddedVal.buffer));
      }
    }


    // Read outputs Top-K Candidates from VRAM (Surgically isolated for absolute Greedy Top-1 stability)
    let nextToken;
    if (this.useGpu) {
      const outputSize = 1 * 10 * 4; // exactly 40 bytes for 10 elements!
      const topkIndices = await readGpuInt32Output(this.device, this.runner.getOutputWebGpuBuffer("prefill", "topk_indices"), outputSize, this.litert);
      nextToken = topkIndices[0]; // Sorted descending, index 0 is the Greedy Top-1 predictions!
    } else {
      const topkIndices = await this.runner.getOutput("prefill", "topk_indices").getData();
      nextToken = topkIndices[0];
    }

    console.log(`[Prefill] nextToken=${nextToken}`);
    
    this.cacheLen = seqLen;
    return nextToken;
  }
  
  async decode(currentToken, maxTokens, onToken) {
    
    for (let step = 0; step < maxTokens; ++step) {
      const tokenStr = this.tokenizer.DecodeToken(currentToken);
      onToken(tokenStr);
      
      if (currentToken === 1 || currentToken === 105 || currentToken === 106) break;
      
      const tokenVec = [currentToken];
      const tokenEmbedding = embeddingLookupCpuFP32(tokenVec, this.embedTokensWeightsFP32, this.config.vocab_size, this.config.emb_dim);
      const embScale = Math.sqrt(this.config.emb_dim);
      const scaledEmbedding = tokenEmbedding.map(x => x * embScale);
      const globalRoPE = computeRopeCosSinForPosition(this.cacheLen, this.config.head_dim, this.config.rope_global_base);
      const ropeLocalBase = this.config.rope_local_base || 10000.0;
      const localRoPE = computeRopeCosSinForPosition(this.cacheLen, this.config.head_dim, ropeLocalBase);
      
      const maskData = new Float32Array(this.maxSeqLen).fill(-1e9);
      for (let j = 0; j <= this.cacheLen; ++j) {
        maskData[j] = 0.0;
      }

      // Write inputs
      if (this.useGpu) {
        writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("decode", "embedded_input"), scaledEmbedding, this.litert);
        writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("decode", "rope_cos"), globalRoPE.cosValues, this.litert);
        writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("decode", "rope_sin"), globalRoPE.sinValues, this.litert);
        writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("decode", "rope_local_cos"), localRoPE.cosValues, this.litert);
        writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("decode", "rope_local_sin"), localRoPE.sinValues, this.litert);
        writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("decode", "start_indices"), new Int32Array([0, 0, this.cacheLen, 0]), this.litert);
        writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("decode", "prefill_slice_index"), new Int32Array([0]), this.litert);
        writeGpuInput(this.device, this.runner.getInputWebGpuBuffer("decode", "attention_mask"), maskData, this.litert);

        this.device.pushErrorScope('validation');
        this.device.pushErrorScope('out-of-memory');
        this.device.pushErrorScope('internal');
      } else {
        this.runner.setInputBinary("decode", "embedded_input", scaledEmbedding);
        this.runner.setInputBinary("decode", "rope_cos", globalRoPE.cosValues);
        this.runner.setInputBinary("decode", "rope_sin", globalRoPE.sinValues);
        this.runner.setInputBinary("decode", "rope_local_cos", localRoPE.cosValues);
        this.runner.setInputBinary("decode", "rope_local_sin", localRoPE.sinValues);
        this.runner.setInputBinary("decode", "start_indices", new Int32Array([0, 0, this.cacheLen, 0]));
        this.runner.setInputBinary("decode", "prefill_slice_index", new Int32Array([0]));
        this.runner.setInputBinary("decode", "attention_mask", maskData);
      }

      
      const success = await this.runner.run("decode");
      
      if (this.useGpu) {
        const internalError = await this.device.popErrorScope();
        const oomError = await this.device.popErrorScope();
        const validationError = await this.device.popErrorScope();
        
        if (validationError) console.error(`WebGPU Validation Error (decode step ${step}):`, validationError.message || validationError);
        if (oomError) console.error(`WebGPU OOM Error (decode step ${step}):`, oomError.message || oomError);
        if (internalError) console.error(`WebGPU Internal Error (decode step ${step}):`, internalError.message || internalError);
      }

      if (!success) {
        console.error("Runner.run('decode') failed at step", step);
        break;
      }
      
      // Read predicted token index directly from GPU-enforced dynamic ArgMax reduced output!
      // Read outputs Top-K Candidates from VRAM (Surgically isolated for absolute Greedy Top-1 stability)
      let nextToken;
      if (this.useGpu) {
        const outputSize = 1 * 10 * 4; // exactly 40 bytes for 10 elements!
        const topkIndices = await readGpuInt32Output(this.device, this.runner.getOutputWebGpuBuffer("decode", "topk_indices"), outputSize, this.litert);
        nextToken = topkIndices[0]; // Sorted descending, index 0 is the Greedy Top-1 predictions!
      } else {
        const topkIndices = await this.runner.getOutput("decode", "topk_indices").getData();
        nextToken = topkIndices[0];
      }
      
      currentToken = nextToken;

      if (this.useGpu) {
        const commandEncoder = this.device.createCommandEncoder();
        for (let i = 0; i < this.config.n_layers; ++i) {
          const outKeyId = this.runner.getOutputWebGpuBuffer("decode", `output_key_cache_${i}`);
          const outValId = this.runner.getOutputWebGpuBuffer("decode", `output_value_cache_${i}`);
          const inKeyId = this.runner.getInputWebGpuBuffer("decode", `key_cache_${i}`);
          const inValId = this.runner.getInputWebGpuBuffer("decode", `value_cache_${i}`);
          
          const outKeyGpu = getGpuBuffer(outKeyId, this.litert);
          const outValGpu = getGpuBuffer(outValId, this.litert);
          const inKeyGpu = getGpuBuffer(inKeyId, this.litert);
          const inValGpu = getGpuBuffer(inValId, this.litert);

          
          commandEncoder.copyBufferToBuffer(outKeyGpu, 0, inKeyGpu, 0, outKeyGpu.size);
          commandEncoder.copyBufferToBuffer(outValGpu, 0, inValGpu, 0, outValGpu.size);
        }
        this.device.queue.submit([commandEncoder.finish()]);
      } else {
        // Standard CPU Fallback Persistence
        for (let i = 0; i < this.config.n_layers; ++i) {
          const keyTensor = this.runner.getOutput("decode", `output_key_cache_${i}`);
          const valTensor = this.runner.getOutput("decode", `output_value_cache_${i}`);
          const keyData = await keyTensor.getData(); 
          const valData = await valTensor.getData();
          
          this.runner.setInputBinary("decode", `key_cache_${i}`, new Uint8Array(keyData.buffer));
          this.runner.setInputBinary("decode", `value_cache_${i}`, new Uint8Array(valData.buffer));
        }
      }
      this.cacheLen++;
    }
  }
  
  delete() {
    if (this.runner) this.runner.delete();
    if (this.dummyInputPrefill) this.dummyInputPrefill.delete();
    if (this.dummyInputDecode) this.dummyInputDecode.delete();
    if (this.dummyLmHeadWeightsPart) {
      for (let p = 0; p < 6; ++p) {
        if (this.dummyLmHeadWeightsPart[p]) this.dummyLmHeadWeightsPart[p].delete();
      }
    }

    if (this.dummyCosPrefill) this.dummyCosPrefill.delete();
    if (this.dummySinPrefill) this.dummySinPrefill.delete();
    if (this.dummyCosDecode) this.dummyCosDecode.delete();
    if (this.dummySinDecode) this.dummySinDecode.delete();
    
    if (this.dummyCosLocalPrefill) this.dummyCosLocalPrefill.delete();
    if (this.dummySinLocalPrefill) this.dummySinLocalPrefill.delete();
    if (this.dummyCosLocalDecode) this.dummyCosLocalDecode.delete();
    if (this.dummySinLocalDecode) this.dummySinLocalDecode.delete();
    
    if (this.dummyMaskPrefill) this.dummyMaskPrefill.delete();
    if (this.dummyMaskDecode) this.dummyMaskDecode.delete();
    
    if (this.dummyStartIndices) this.dummyStartIndices.delete();
    for (let i = 0; i < this.config.n_layers; ++i) {
      this.currentKeyCaches[i].delete();
      this.currentValueCaches[i].delete();
    }
  }
}
