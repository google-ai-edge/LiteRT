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
 * Builds the Gemma 3 graph in JavaScript using the LiteRT Tensor API.
 */
export class Gemma3GraphBuilder {
  constructor(litert, config) {
    this.litert = litert;
    this.config = config;
    this.promises = [];
  }

  async getWeight(weights, name, type, shape) {
    const w = weights[name];
    if (w) {
      // Quantization params are now set in safetensors.js
      
      
      return w;
    }
    const t = this.litert.createTensor({ name, type, shape });
    t.allocateBuffer();
    return t;
  }

  async rmsNorm(input, scale, eps = 1e-6) {
    const xSquared = input.mul(input);
    const shape = input.getShape();
    const lastAxis = shape.size() - 1;

    const meanSquared = xSquared.mean([lastAxis], true);
    
    const epsTensor = this.litert.createTensor({ type: 'FP32', shape: [1] });
    await epsTensor.setData(new Float32Array([eps]));
    
    const variancePlusEps = meanSquared.add(epsTensor);
    const invRms = variancePlusEps.rsqrt();
    const xNorm = input.mul(invRms);
    
    const output = xNorm.mul(scale);
    return { output, xSquared, meanSquared, variancePlusEps, invRms, xNorm };
  }

  geluTanh(input) {
    return input.gelu(true);
  }

  async makeFeedForwardLayer(input, name, weights) {
    const gateProj = await this.getWeight(weights, `${name}.gate_proj.weight`, 'I8', [this.config.hidden_dim, this.config.emb_dim]);
    const upProj = await this.getWeight(weights, `${name}.up_proj.weight`, 'I8', [this.config.hidden_dim, this.config.emb_dim]);
    const downProj = await this.getWeight(weights, `${name}.down_proj.weight`, 'I8', [this.config.emb_dim, this.config.hidden_dim]);

    const gate = this.geluTanh(input.fullyConnected(gateProj));
    const up = input.fullyConnected(upProj);
    return gate.mul(up).fullyConnected(downProj);
  }

  async applyRotaryEmbedding(x, cos, sin) {
    const shape = x.getShape();
    const halfDim = shape.get(3) / 2;
    
    let cos_tiled = cos;
    let sin_tiled = sin;
    const n_heads = shape.get(1);
    if (n_heads > 1) {
      cos_tiled = cos.tile([1, n_heads, 1, 1]);
      sin_tiled = sin.tile([1, n_heads, 1, 1]);
    }

    const x1 = x.slice([0, 0, 0, 0], [shape.get(0), shape.get(1), shape.get(2), halfDim]);
    const x2 = x.slice([0, 0, 0, halfDim], [shape.get(0), shape.get(1), shape.get(2), halfDim]);

    const negOne = this.litert.createTensor({ type: 'FP32', shape: [1] });
    await negOne.setData(new Float32Array([-1.0]));
    const neg_x2 = x2.mul(negOne);
    const rotated = neg_x2.concatenation([x1], 3);

    const x_cos = x.mul(cos_tiled);
    const rotated_sin = rotated.mul(sin_tiled);
    return x_cos.add(rotated_sin);
  }

  async makeSelfAttentionLayer(input, name, isSlidingAttention, attentionMask, cos, sin, keyCache, valueCache, startIndices, weights) {
    const qkvOutDim = this.config.n_heads * this.config.head_dim;
    const kvOutDim = this.config.n_kv_groups * this.config.head_dim;

    const qProj = await this.getWeight(weights, `${name}.q_proj.weight`, 'I8', [qkvOutDim, this.config.emb_dim]);
    const kProj = await this.getWeight(weights, `${name}.k_proj.weight`, 'I8', [kvOutDim, this.config.emb_dim]);
    const vProj = await this.getWeight(weights, `${name}.v_proj.weight`, 'I8', [kvOutDim, this.config.emb_dim]);
    const oProj = await this.getWeight(weights, `${name}.o_proj.weight`, 'I8', [this.config.emb_dim, qkvOutDim]);

    let q = input.fullyConnected(qProj);
    let k = input.fullyConnected(kProj);
    let v = input.fullyConnected(vProj);

    // Debug logging removed

    const qNorm = await this.getWeight(weights, `${name}.q_norm.weight`, 'FP32', [this.config.head_dim]);
    const kNorm = await this.getWeight(weights, `${name}.k_norm.weight`, 'FP32', [this.config.head_dim]);



    const inputShape = input.getShape();
    const batchSize = inputShape.get(0);
    const seqLen = inputShape.get(1);

    q = q.reshape([batchSize, seqLen, this.config.n_heads, this.config.head_dim]);
    q = q.transpose([0, 2, 1, 3]); // [batch, n_heads, seq, head_dim]

    k = k.reshape([batchSize, seqLen, this.config.n_kv_groups, this.config.head_dim]);
    k = k.transpose([0, 2, 1, 3]); // [batch, n_kv_groups, seq, head_dim]

    v = v.reshape([batchSize, seqLen, this.config.n_kv_groups, this.config.head_dim]);
    v = v.transpose([0, 2, 1, 3]); // [batch, n_kv_groups, seq, head_dim]

    q = (await this.rmsNorm(q, qNorm, this.config.rms_norm_eps)).output;
    k = (await this.rmsNorm(k, kNorm, this.config.rms_norm_eps)).output;

    q = await this.applyRotaryEmbedding(q, cos, sin);
    k = await this.applyRotaryEmbedding(k, cos, sin);

    let k_untiled = k;
    let v_untiled = v;

    if (keyCache && valueCache && startIndices) {
      k_untiled = keyCache.dynamicUpdateSlice(k, startIndices);
      v_untiled = valueCache.dynamicUpdateSlice(v, startIndices);
    } else if (keyCache && valueCache) {
      k_untiled = keyCache.concatenation([k], 2);
      v_untiled = valueCache.concatenation([v], 2);
    }

    let kForAttn = k_untiled;
    let vForAttn = v_untiled;

    const scale = 1.0 / Math.sqrt(this.config.query_pre_attn_scalar);
    const scaleTensor = this.litert.createTensor({ type: 'FP32', shape: [1] });
    await scaleTensor.setData(new Float32Array([scale]));

    // Tile K and V if needed for multi-query or grouped-query attention
    if (this.config.n_heads % this.config.n_kv_groups === 0) {
      const numGroups = this.config.n_heads / this.config.n_kv_groups;
      if (numGroups > 1) {
        kForAttn = kForAttn.tile([1, numGroups, 1, 1]);
        vForAttn = vForAttn.tile([1, numGroups, 1, 1]);
      }
    }

    // WORKAROUND for WebGPU BatchMatMul adj_y bug: Transpose K explicitly!
    const kForAttnTransposed = kForAttn.transpose([0, 1, 3, 2]);
    let unscaledScores = q.batchMatMul(kForAttnTransposed, false, false); // [1, 4, 512, 512]

    let scores = unscaledScores.mul(scaleTensor);
    
    if (attentionMask) {
      scores = scores.add(attentionMask);
    }
    
    const attnWeights = scores.softmax(1.0);

    let attnOutput = attnWeights.batchMatMul(vForAttn, false, false); // [1, 4, 512, 256]
    attnOutput = attnOutput.transpose([0, 2, 1, 3]); // [1, 512, 4, 256]
    attnOutput = attnOutput.reshape([batchSize, seqLen, qkvOutDim]); // [1, 512, 1024]



    const output = attnOutput.fullyConnected(oProj);

    return { output, keyCache: k_untiled, valueCache: v_untiled, q, k, v, scores, kForAttn, vForAttn, unscaledScores };
  }



  async buildGemma3FromEmbeddings(embeddedInput, ropeGlobalCos, ropeGlobalSin, ropeLocalCos, ropeLocalSin, slidingAttentionMask, globalAttentionMask, startIndices, prefillSliceIndex, keyCaches, valueCaches, weights, lmHeadWeights, isDecode = true) {
    let hiddenStates = embeddedInput;

    const updatedKeyCaches = [];
    const updatedValueCaches = [];
    const layerOutputs = [];

    for (let i = 0; i < this.config.n_layers; ++i) {
      const layerPrefix = `model.layers.${i}`;
      const isSliding = !((this.config.sliding_window_pattern > 0) && ((i + 1) % this.config.sliding_window_pattern === 0));

      const attentionMask = isSliding ? slidingAttentionMask : globalAttentionMask;
      const cos = isSliding ? ropeLocalCos : ropeGlobalCos;
      const sin = isSliding ? ropeLocalSin : ropeGlobalSin;

      const inputNormScale = await this.getWeight(weights, `${layerPrefix}.input_layernorm.weight`, 'FP32', [this.config.emb_dim]);
      const rmsNormResult = await this.rmsNorm(hiddenStates, inputNormScale, this.config.rms_norm_eps);
      const normedInput = rmsNormResult.output;
      if (i === 0) {
        normedInput.setName(`layer_0_normed_input`);
        layerOutputs.push(normedInput);
        
        rmsNormResult.xSquared.setName(`layer_0_xSquared`);
        layerOutputs.push(rmsNormResult.xSquared);
        rmsNormResult.meanSquared.setName(`layer_0_meanSquared`);
        layerOutputs.push(rmsNormResult.meanSquared);
        rmsNormResult.variancePlusEps.setName(`layer_0_variancePlusEps`);
        layerOutputs.push(rmsNormResult.variancePlusEps);
        rmsNormResult.invRms.setName(`layer_0_invRms`);
        layerOutputs.push(rmsNormResult.invRms);
        rmsNormResult.xNorm.setName(`layer_0_xNorm`);
        layerOutputs.push(rmsNormResult.xNorm);
      }

      const attnOutput = await this.makeSelfAttentionLayer(
        normedInput,
        `${layerPrefix}.self_attn`,
        isSliding,
        attentionMask,
        cos,
        sin,
        keyCaches ? keyCaches[i] : null,
        valueCaches ? valueCaches[i] : null,
        startIndices,
        weights
      );
      if (i === 0) {
        attnOutput.output.setName(`layer_0_attn_output`);
        layerOutputs.push(attnOutput.output);
        attnOutput.q.setName(`layer_0_q`);
        layerOutputs.push(attnOutput.q);
        attnOutput.k.setName(`layer_0_k`);
        layerOutputs.push(attnOutput.k);
        attnOutput.v.setName(`layer_0_v`);
        layerOutputs.push(attnOutput.v);
        if (attnOutput.kForAttn) {
          attnOutput.kForAttn.setName(`layer_0_kForAttn`);
          layerOutputs.push(attnOutput.kForAttn);
        }
        attnOutput.scores.setName(`layer_0_scores`);
        layerOutputs.push(attnOutput.scores);
        if (attnOutput.unscaledScores) {
          attnOutput.unscaledScores.setName(`layer_0_unscaled_scores`);
          layerOutputs.push(attnOutput.unscaledScores);
        }
      }

      attnOutput.keyCache.setName(`output_key_cache_${i}`);
      attnOutput.valueCache.setName(`output_value_cache_${i}`);
      updatedKeyCaches.push(attnOutput.keyCache);
      updatedValueCaches.push(attnOutput.valueCache);


      const postAttnNormScale = await this.getWeight(weights, `${layerPrefix}.post_attention_layernorm.weight`, 'FP32', [this.config.emb_dim]);
      const normedAttnOutput = (await this.rmsNorm(attnOutput.output, postAttnNormScale, this.config.rms_norm_eps)).output;

      hiddenStates = hiddenStates.add(normedAttnOutput);

      const preFfnNormScale = await this.getWeight(weights, `${layerPrefix}.pre_feedforward_layernorm.weight`, 'FP32', [this.config.emb_dim]);
      const normedForFfn = (await this.rmsNorm(hiddenStates, preFfnNormScale, this.config.rms_norm_eps)).output;

      const ffnOutput = await this.makeFeedForwardLayer(normedForFfn, `${layerPrefix}.mlp`, weights);

      const postFfnNormScale = await this.getWeight(weights, `${layerPrefix}.post_feedforward_layernorm.weight`, 'FP32', [this.config.emb_dim]);
      const normedFfnOutput = (await this.rmsNorm(ffnOutput, postFfnNormScale, this.config.rms_norm_eps)).output;

      hiddenStates = hiddenStates.add(normedFfnOutput);
      hiddenStates.setName(`layer_${i}_output`);
      layerOutputs.push(hiddenStates);
    }

    const finalNormScale = await this.getWeight(weights, "model.norm.weight", 'FP32', [this.config.emb_dim]);
    const finalOutput = (await this.rmsNorm(hiddenStates, finalNormScale, this.config.rms_norm_eps)).output;
 
    let logitsInput = finalOutput;
    if (!isDecode) {
      // Dynamic Projections Slicing: Gather activations at last prompt position dynamically in VRAM before projections!
      logitsInput = finalOutput.gather(prefillSliceIndex, 1);
    }

    // Unified dynamic Model Constant weights Projections Serialized directly inside graph Flatbuffer with NO signature inputs!
    const logitsOutput = this.buildLogitsGraph(logitsInput, lmHeadWeights);

    // GPU Graph Reduction: Compute dynamic Top-K Candidates entirely on backend accelerator shaders!
    const topKResult = logitsOutput.topK(10); // Extract top 10 candidates dynamically as a TensorVector!
    const topkValues = topKResult.get(0);
    const topkIndices = topKResult.get(1);
    topKResult.delete(); // Wipes out intermediate wrapper to prevent WASM memory leaks!
    
    topkValues.setName("topk_values");
    topkIndices.setName("topk_indices");
    
    return { topk_values: topkValues, topk_indices: topkIndices, updatedKeyCaches, updatedValueCaches, layerOutputs };

  }

  buildLogitsGraph(hiddenStates, weightsTensor) {
    const shape = hiddenStates.getShape();
    const seqLen = shape.get(1);
    const embDim = shape.get(2);
    
    // Reshape 3D sequence to standard 4D [1, 1, seqLen, embDim] to ensure GPU delegate maps to standard FC MatMul kernels!
    const input4d = hiddenStates.reshape([1, 1, seqLen, embDim]);
    const logits4d = input4d.fullyConnected(weightsTensor);
    
    // Reshape 4D logits output back to [1, seqLen, vocabSize] before graph reductions!
    const vocabSize = weightsTensor.getShape().get(0);
    return logits4d.reshape([1, seqLen, vocabSize]);
  }
}


