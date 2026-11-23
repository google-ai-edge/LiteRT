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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_PROGRAM_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_PROGRAM_H_

#include <cstddef>
#include <cstdint>

#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"

// The decode step of a Gemma 4 mixture-of-experts model (26B-A4B: 30 layers,
// sliding and global attention, a dense feed-forward beside 128 routed
// experts, the head and its soft cap) as a program for one persistent kernel.
//
// The caller describes the model (LiteRtNvidiaMegakernelModel: the weights,
// norms and caches of every layer, as the per-op launches take them today);
// LiteRtNvidiaMegakernelBuildProgram (megakernel.h) turns the description
// into a LiteRtNvidiaMegakernelProgram, a plain struct the caller copies to
// device memory whenever its pointers change. A description without
// pointers (all null) builds the program's shapes alone: its
// workspace_bytes and shared_bytes are right, so a caller can size the
// workspace before any pointer exists. A step's inputs and outputs
// (LiteRtNvidiaMegakernelStep) are passed at every launch. Everything here
// is plain data, read by the host and the device alike: pointers are device
// pointers, sizes are elements unless they say bytes.
//
// The program is a list of instructions in an order that respects their
// data dependencies. Every CTA walks the whole list; an instruction's items
// (tiles of a projection's rows, heads, chunks of keys) are split over the
// CTAs in a fixed way, so no work queue is needed. Before its first item a
// CTA waits until every CTA has finished the instructions named in `waits`
// in this step: each instruction has a counter that every CTA increments
// once it is done with its share, and the counters carry the step's epoch,
// so nothing is reset between steps.

// A blockwise INT4 projection as the model file holds it
// (int4_block_gemv.h): weights [rows, cols / 2] bytes, two signed values per
// byte (the even one in the low nibble), and FP16 scales [rows, cols / 32].
struct LiteRtNvidiaMegakernelProjection {
  const uint8_t* weights;
  const void* scales;
  int32_t rows;
  int32_t cols;
};

// One decoder layer. Attention: rms_norm -> qkv projection -> per-head
// norms and RoPE -> the token's key and value into the caches -> attention
// over the caches -> o projection; then h = x + rms_norm(o) *
// post_attention_norm. The feed-forward block from h is
// LiteRtNvidiaLaunchMoeLayerBlock's (int4_block_gemv.h): a dense gated
// feed-forward and the routed experts, each with its norms, and
//   x' = (h + n_s(n_dp(dense) + n_mp(experts))) * residual_multiplier.
struct LiteRtNvidiaMegakernelLayer {
  // A sliding layer (shared_kv == 0) projects q, k and v ([q_heads + 2 *
  // kv_heads] rows of head_dim in qkv) and treats them as
  // LiteRtNvidiaLaunchQkvNormRope does (has_v_norm); a global layer
  // (shared_kv != 0) projects q and k only ([q_heads + kv_heads] rows of
  // head_dim) and takes v as the k rows normalized without a scale.
  int32_t shared_kv;
  int32_t q_heads;
  int32_t kv_heads;
  int32_t head_dim;    // 256 or 512
  int32_t rotary_dim;  // dims rotated (proportional RoPE), even, <= head_dim
  float rope_min_timescale;
  float rope_max_timescale;
  const float* input_norm;  // [model_dim], fused into the qkv projection
  LiteRtNvidiaMegakernelProjection qkv;
  const float* q_norm;  // [head_dim]
  const float* k_norm;  // [head_dim]
  // FP16 caches [kv_heads, cache_len, head_dim], keys and values as rows.
  // The token goes to row position % cache_len (ring != 0) or position
  // clamped to the cache; the attention spans all cache_len rows under the
  // step's mask LiteRtNvidiaMegakernelStep::mask[mask].
  void* k_cache;
  void* v_cache;
  int32_t cache_len;
  int32_t ring;
  int32_t mask;
  LiteRtNvidiaMegakernelProjection o;  // [model_dim, q_heads * head_dim]
  const float* post_attention_norm;    // [model_dim]

  // The feed-forward block (LiteRtNvidiaLaunchMoeLayerBlock's arguments).
  // The arena holds the router's rows, the dense feed-forward's rows and the
  // experts' (the MoE plugin's layout: compact expert scales, the experts'
  // rows expert by expert or in the model's order).
  LiteRtNvidiaMoeArena arena;
  int32_t experts;
  int32_t top_k;
  int32_t expert_hidden;  // an expert's hidden size
  int32_t expert_gate;    // 1: tanh GELU, 2: erf GELU
  int32_t dense_hidden;
  int32_t dense_gate;
  const float* per_expert_scale;  // [experts]
  const float* router_norm;       // [model_dim]
  float router_multiplier;
  const float* expert_norm;  // [model_dim]
  const float* dense_norm;   // [model_dim]
  const float* dense_post;   // [model_dim]
  const float* moe_post;     // [model_dim]
  const float* sum_post;     // [model_dim]
  float residual_multiplier;
};

constexpr int kLiteRtNvidiaMegakernelMaxLayers = 64;

// The model: its layers in order, then the final norm, the head and its soft
// cap. `layer` is a host array of `layers` entries (the builder copies it).
struct LiteRtNvidiaMegakernelModel {
  int32_t layers;
  int32_t model_dim;  // a multiple of 32, at most 4096
  int32_t vocab;
  float norm_eps;  // every rms_norm's epsilon
  const LiteRtNvidiaMegakernelLayer* layer;
  const float* final_norm;                // [model_dim]
  LiteRtNvidiaMegakernelProjection head;  // [vocab, model_dim]
  // logits = tanh(z * soft_cap_in) * soft_cap as the graph computes it, a
  // BF16 tensor at every step (soft_cap_in is the graph's constant, BF16(1 /
  // soft_cap) for Gemma 4, not 1 / soft_cap); soft_cap 0: none.
  float soft_cap;
  float soft_cap_in;
};

// A step's inputs and outputs (device pointers), passed at every launch.
struct LiteRtNvidiaMegakernelStep {
  const float* embeddings;  // FP32 [model_dim]
  const int32_t* position;  // [1]: the token's position
  // The attention masks a layer selects with its `mask`: bool [cache_len]
  // shared by the rows of the layers that use it (the sliding layers' ring,
  // the global layers' whole cache).
  const bool* mask[2];
  float* activations;  // FP32 [model_dim]: the final norm's output
  float* logits;       // FP32 [vocab]
};

// What an instruction does. Items are what its CTAs split among them: tiles
// of a projection's rows go to the CTAs' warps, the other items to whole
// CTAs.
enum LiteRtNvidiaMegakernelOpcode : int32_t {
  // The layer input x (the step's embeddings as BF16, or the previous
  // layer's block output, computed by every CTA from its pieces), the input
  // norm and the qkv projection. Items: tiles.
  kLiteRtNvidiaMegakernelOpQkv = 1,
  // Per head (q heads, then k heads, then v heads): its norm and RoPE; the
  // k and v heads also go into the caches as FP16. Items: heads.
  kLiteRtNvidiaMegakernelOpNormRope = 2,
  // The chunks of keys of one split of one KV head for all its query rows:
  // the partial softmax and values (DecodeAttentionPartialBlock). Items:
  // splits * kv_heads.
  kLiteRtNvidiaMegakernelOpAttention = 3,
  // The splits of one query row of a KV head combined
  // (DecodeAttentionCombineBlock). Items: query rows * kv_heads.
  kLiteRtNvidiaMegakernelOpAttentionCombine = 4,
  // The o projection. Items: tiles.
  kLiteRtNvidiaMegakernelOpOutput = 5,
  // h = x + rms_norm(o) (every CTA computes it); the router's logits from
  // h. Items: tiles.
  kLiteRtNvidiaMegakernelOpRouter = 6,
  // The dense feed-forward's gated projection from h. Items: tiles.
  kLiteRtNvidiaMegakernelOpDenseGateUp = 7,
  // The routing (top_k of the router's softmax, every CTA computes it) and
  // the routed experts' gated projections. Items: top_k * tiles of an
  // expert's hidden rows.
  kLiteRtNvidiaMegakernelOpExpertHidden = 8,
  // The dense feed-forward's down projection. Items: tiles.
  kLiteRtNvidiaMegakernelOpDenseDown = 9,
  // The routed experts' down projections, each times its routing weight,
  // summed over the experts. Items: tiles of model_dim rows (all experts).
  kLiteRtNvidiaMegakernelOpExpertDown = 10,
  // The last layer's block output, the activations (the final norm's
  // output), the head over the normed output and its soft cap into the
  // logits. Items: tiles.
  kLiteRtNvidiaMegakernelOpHead = 11,
};

constexpr int kLiteRtNvidiaMegakernelMaxWaits = 4;

struct LiteRtNvidiaMegakernelInstruction {
  int32_t opcode;  // LiteRtNvidiaMegakernelOpcode
  int32_t layer;   // the layer it belongs to; the tail's: layers
  int32_t items;
  int32_t tile_rows;  // rows of an item (projections): a unit's rows
  int32_t parts;      // units per item (projections): 1, 2 (gate, up), top_k
  // Earlier instructions (indices into the program) that every CTA must
  // have finished in this step before this one starts; -1: none.
  int32_t waits[kLiteRtNvidiaMegakernelMaxWaits];
};

// A layer's scratch in the workspace: byte offsets of its buffers, each its
// own so that no buffer is written twice in a step (the kernels read some
// of them through the non-coherent cache).
struct LiteRtNvidiaMegakernelScratch {
  uint64_t x;    // BF16 [model_dim]: the layer input
  uint64_t qkv;  // BF16 [qkv rows]
  // BF16 [(q_heads + 2 kv_heads) * head_dim]: the reference's qkv with the
  // keys repeated as the values of a global layer (shared_kv).
  uint64_t qkv_values;
  uint64_t q;              // BF16 [q_heads, head_dim]
  uint64_t k;              // BF16 [kv_heads, head_dim]: the token's key
  uint64_t v;              // BF16 [kv_heads, head_dim]: and value
  uint64_t attention_acc;  // FP32 [kv_heads, splits, group, head_dim]
  uint64_t attention_max;  // FP32 [kv_heads, splits, group]
  uint64_t attention_sum;  // FP32 [kv_heads, splits, group]
  uint64_t attention_out;  // BF16 [q_heads, head_dim]
  uint64_t o;              // BF16 [model_dim]
  uint64_t h;              // BF16 [model_dim]
  uint64_t router;         // BF16 [experts]: the router's logits
  uint64_t routing;        // int32 [top_k] experts, then FP32 [top_k] weights
  uint64_t expert_hidden;  // FP32 [top_k, expert_hidden]
  uint64_t experts_out;    // FP32 [model_dim]: the experts' weighted sum
  uint64_t dense_hidden;   // BF16 [dense_hidden]
  uint64_t dense_out;      // BF16 [model_dim]
  uint64_t moe;            // the reference's MoE layer block workspace
  int32_t splits;          // the attention's splits of the keys per KV head
  int32_t chunk;           // keys per chunk
};

constexpr int kLiteRtNvidiaMegakernelMaxInstructions = 512;
// Per CTA: eight consumer warps and the warp that streams the weights.
constexpr int kLiteRtNvidiaMegakernelThreads = 288;
constexpr int kLiteRtNvidiaMegakernelMaxCtas = 1024;
constexpr int kLiteRtNvidiaMegakernelVersion = 2;

// The program: the model, its instructions, the scratch layout and the grid
// it was built for. Copied to device memory as is.
struct LiteRtNvidiaMegakernelProgram {
  int32_t version;       // kLiteRtNvidiaMegakernelVersion
  int32_t ctas;          // the grid the program was built for
  int32_t threads;       // threads per CTA
  int32_t shared_bytes;  // dynamic shared memory per CTA
  int32_t instruction_count;
  int32_t layers;
  int32_t model_dim;
  int32_t vocab;
  float norm_eps;
  float soft_cap;
  float soft_cap_in;
  int32_t max_activation;  // the longest projection input (shared memory)
  int32_t work_bytes;      // shared memory of the instructions' own
  int32_t mask_len[2];     // each mask's keys: the longest cache reading it
  const float* final_norm;
  LiteRtNvidiaMegakernelProjection head;
  uint64_t workspace_bytes;  // the scratch, LiteRtNvidiaMegakernelLaunch's
  uint64_t final_x;          // BF16 [model_dim]: the last layer's output
  uint64_t head_raw;         // BF16 [vocab]: the head before the soft cap
  // int32 [2, kLiteRtNvidiaMegakernelMaxCtas]: per CTA, the last visible key
  // of its slice of each mask (-1: none), which bounds the keys the
  // attention reads.
  uint64_t mask_last;
  LiteRtNvidiaMegakernelLayer layer[kLiteRtNvidiaMegakernelMaxLayers];
  LiteRtNvidiaMegakernelScratch scratch[kLiteRtNvidiaMegakernelMaxLayers];
  LiteRtNvidiaMegakernelInstruction
      instruction[kLiteRtNvidiaMegakernelMaxInstructions];
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_PROGRAM_H_
