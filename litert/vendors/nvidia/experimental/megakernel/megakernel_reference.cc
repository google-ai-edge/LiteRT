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

// The decode step as today's per-op launches (LiteRtNvidiaMegakernelLaunch
// Reference), in the order and with the arguments of the megakernel plugin
// layer's interpreted step (megakernel_plugin.cc), each layer's
// buffers in its own scratch.

#include <cstdint>

#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/experimental/megakernel/megakernel.h"
#include "litert/vendors/nvidia/experimental/megakernel/megakernel_steps.h"
#include "litert/vendors/nvidia/experimental/megakernel/program.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"
#include "litert/vendors/nvidia/trtllm/qkv_norm_rope.h"

namespace {

// The mask fill of odml.sdpa_transposed's decomposition.
constexpr float kAttentionFill = -10000.0f;

}  // namespace

extern "C" cudaError_t LiteRtNvidiaMegakernelLaunchReference(
    const LiteRtNvidiaMegakernelProgram* program,
    const LiteRtNvidiaMegakernelStep* step, void* workspace,
    const LiteRtNvidiaMegakernelReferenceStreams* streams,
    cudaStream_t stream) {
  if (program == nullptr || step == nullptr || workspace == nullptr ||
      streams == nullptr || program->soft_cap <= 0.0f) {
    return cudaErrorInvalidValue;
  }
  const auto& p = *program;
  auto* bytes = static_cast<char*>(workspace);
  auto at = [&](uint64_t offset) { return bytes + offset; };
  const float eps = p.norm_eps;
  cudaError_t status = LiteRtNvidiaMegakernelToBf16(
      step->embeddings, at(p.scratch[0].x), p.model_dim, stream);
  for (int l = 0; l < p.layers && status == cudaSuccess; ++l) {
    const LiteRtNvidiaMegakernelLayer& s = p.layer[l];
    const LiteRtNvidiaMegakernelScratch& w = p.scratch[l];
    void* next = l + 1 < p.layers ? at(p.scratch[l + 1].x) : at(p.final_x);
    const int32_t depth = s.head_dim;
    const int32_t group = s.q_heads / s.kv_heads;
    // The input norm and the qkv projection.
    status = LiteRtNvidiaLaunchBf16Int4BlockGemvRowsNormed(
        at(w.x), s.qkv.weights, s.qkv.scales, s.input_norm, eps,
        /*norm_multiplier=*/1.0f, at(w.qkv), /*rows=*/1, s.qkv.cols, s.qkv.rows,
        /*gate=*/0, stream);
    // The head norms and RoPE; a global layer's values are its keys
    // normalized without a scale.
    const void* qkv = at(w.qkv);
    if (status == cudaSuccess && s.shared_kv != 0) {
      status = LiteRtNvidiaMegakernelRepeatKeys(
          qkv, at(w.qkv_values), s.q_heads * depth, s.kv_heads * depth, stream);
      qkv = at(w.qkv_values);
    }
    if (status == cudaSuccess) {
      const LiteRtNvidiaQkvNormRopeParams params = {
          /*tokens=*/1,         s.q_heads,
          s.kv_heads,           depth,
          s.rotary_dim,
          /*has_v_norm=*/1,     eps,
          s.rope_min_timescale, s.rope_max_timescale};
      status = LiteRtNvidiaLaunchQkvNormRope(&params, qkv, step->position,
                                             s.q_norm, s.k_norm, at(w.q),
                                             at(w.k), at(w.v), stream);
    }
    // The token into the caches, and the attention over them.
    if (status == cudaSuccess) {
      status = LiteRtNvidiaMegakernelWriteCacheRows(
          at(w.k), at(w.v), s.k_cache, s.v_cache, step->position, s.kv_heads,
          s.cache_len, depth, s.ring, stream);
    }
    if (status == cudaSuccess) {
      status = LiteRtNvidiaLaunchDecodeAttention(
          at(w.q), /*q_bf16=*/true, s.k_cache, s.v_cache, step->mask[s.mask],
          /*mask_rows=*/1, s.kv_heads, group, s.cache_len, depth,
          kAttentionFill, at(w.attention_out), at(w.attention_acc), stream);
    }
    // The o projection, its norm and the residual update.
    if (status == cudaSuccess) {
      status = LiteRtNvidiaLaunchBf16Int4BlockGemvRows(
          at(w.attention_out), s.o.weights, s.o.scales, at(w.o), /*rows=*/1,
          s.o.cols, s.o.rows, /*gate=*/0, stream);
    }
    if (status == cudaSuccess) {
      status = LiteRtNvidiaMegakernelNormResidual(at(w.o), at(w.x),
                                                  s.post_attention_norm, eps,
                                                  at(w.h), p.model_dim, stream);
    }
    // The feed-forward block, as the MoE plugin launches it.
    if (status == cudaSuccess) {
      const LiteRtNvidiaMoeShape shape = {
          /*rows=*/1,  s.top_k,         s.experts,
          p.model_dim, s.expert_hidden, s.expert_gate};
      const LiteRtNvidiaMoeBlockParams block = {
          s.dense_hidden, s.dense_gate,         s.residual_multiplier,
          s.dense_norm,   s.dense_post,         s.moe_post,
          s.sum_post,     streams->side_stream, streams->fork,
          streams->join};
      status = LiteRtNvidiaLaunchMoeLayerBlock(
          &shape, &s.arena, at(w.h), s.per_expert_scale, s.router_norm,
          s.router_multiplier, s.expert_norm, eps, /*renormalize=*/1, &block,
          at(w.moe), next, stream);
    }
  }
  // The final norm, the head and its soft cap.
  if (status == cudaSuccess) {
    status = LiteRtNvidiaLaunchBf16Int4BlockGemvRowsNormed(
        at(p.final_x), p.head.weights, p.head.scales, p.final_norm, eps,
        /*norm_multiplier=*/1.0f, at(p.head_raw), /*rows=*/1, p.head.cols,
        p.head.rows, /*gate=*/0, stream);
  }
  if (status == cudaSuccess) {
    status =
        LiteRtNvidiaMegakernelSoftCap(at(p.head_raw), p.soft_cap_in, p.soft_cap,
                                      step->logits, p.vocab, stream);
  }
  if (status == cudaSuccess) {
    status = LiteRtNvidiaMegakernelNormToFloat(at(p.final_x), p.final_norm, eps,
                                               step->activations, p.model_dim,
                                               stream);
  }
  return status;
}
