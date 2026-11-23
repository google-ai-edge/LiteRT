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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_PLUGIN_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_PLUGIN_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/nvidia/compiler/moe_plugin.h"
#include "NvInferRuntime.h"

namespace litert::nvidia {

// Experimental. The decode step of a Gemma 4 mixture-of-experts model (the
// whole body of its decode signature) as one TensorRT plugin layer that runs
// the step as a program (program.h). The compiler describes the step
// (MegakernelDescription: where every weight, norm and cache of every layer
// is; megakernel_lowering.inc); the plugin builds the program from the
// description and the device addresses of its inputs, and runs it, as
// LITERT_NVIDIA_MEGAKERNEL says when an execution context attaches:
//
//   1            the megakernel: the whole step in one cooperative launch
//                (megakernel.h);
//   interpreted  today's per-op launches in the order of the engine's graph,
//                the program interpreted on the host
//                (LiteRtNvidiaMegakernelLaunchReference: the projections,
//                the qkv norms and RoPE, the decode attention and the MoE
//                layer block of trtllm/, and megakernel_steps.h for the
//                steps the graph runs as TensorRT layers).
//
// Unset or 0, the compiler does not lower the body to this layer, and no
// engine holds it.
//
//   inputs   the step's signature inputs other than the caches, in the
//            signature's order (embeddings FP32, positions INT32, the masks
//            BOOL, the cache parameters INT32); the caches FP16 [1, H, S, D],
//            each layer's key cache and then its value cache (held as rows),
//            which the plugin updates in place; the norms FP32 [n], every
//            FP32 vector the program reads; the weight holders INT64, the
//            segments of the weight store that hold the projections.
//   outputs  the logits FP32 [1, 1, vocab] and the activations FP32
//            [1, 1, model_dim] (the final norm's output); the cache row each
//            layer's token went to, INT32 [layers]; and each cache's new row,
//            FP16 [1, H, 1, D], which native in-place cache writes put where
//            the plugin put it, so that the engine's cache outputs alias
//            their inputs.

// A projection's INT4 weights [rows, cols] and FP16 scales per 32 of its
// cols, at byte offsets in the segment of weight holder `holder`.
struct MegakernelProjectionSource {
  int32_t holder = 0;
  int64_t weights = 0;
  int64_t scales = 0;
  int32_t rows = 0;
  int32_t cols = 0;
};

// One decoder layer (program.h, LiteRtNvidiaMegakernelLayer). A norm is the
// offset of its FP32 vector in the norms input; a cache the index of its
// input among the caches.
struct MegakernelLayerSource {
  int32_t shared_kv = 0;
  int32_t q_heads = 0;
  int32_t kv_heads = 0;
  int32_t head_dim = 0;
  int32_t rotary_dim = 0;
  float rope_min_timescale = 1.0f;
  float rope_max_timescale = 10000.0f;
  int64_t input_norm = 0;
  MegakernelProjectionSource qkv;
  int64_t q_norm = 0;
  int64_t k_norm = 0;
  int32_t k_cache = 0;
  int32_t v_cache = 0;
  int32_t cache_len = 0;
  int32_t ring = 0;
  // The index of the layer's mask among MegakernelDescription::masks.
  int32_t mask = 0;
  MegakernelProjectionSource o;
  int64_t post_attention_norm = 0;
  // The feed-forward block: the MoE plugin's arena of the experts, the
  // router's rows and the dense feed-forward's (moe_plugin.h), and the
  // block's parameters.
  MoeExpertSource arena;
  int32_t experts = 0;
  int32_t top_k = 0;
  int32_t expert_hidden = 0;
  int32_t expert_gate = 0;
  int32_t dense_gate = 0;
  int64_t per_expert_scale = 0;
  int64_t router_norm = 0;
  float router_multiplier = 1.0f;
  int64_t expert_norm = 0;
  int64_t dense_norm = 0;
  int64_t dense_post = 0;
  int64_t moe_post = 0;
  int64_t sum_post = 0;
  float residual_multiplier = 1.0f;
};

struct MegakernelDescription {
  // The inputs in front of the caches: their types, and which of them are
  // the embeddings, the positions and the masks.
  std::vector<int32_t> input_types;  // nvinfer1::DataType
  int32_t embeddings = 0;
  int32_t positions = 0;
  std::vector<int32_t> masks;
  int32_t model_dim = 0;
  int32_t vocab = 0;
  float norm_eps = 1e-6f;
  // logits = tanh(z * soft_cap_in) * soft_cap; soft_cap 0: none.
  float soft_cap = 0.0f;
  float soft_cap_in = 0.0f;
  std::vector<MegakernelLayerSource> layers;
  int64_t final_norm = 0;
  MegakernelProjectionSource head;
  // The FP32 values of the norms input, the number of weight holders and the
  // weight store's granule (each holder is one granule larger than its
  // segment, which starts at the first granule boundary in it).
  int64_t norm_values = 0;
  int32_t holders = 0;
  int64_t holder_granule = 0;
};

// The description as the plugin's serialized field, and back; nothing if the
// bytes are not a description of this version.
std::vector<uint8_t> SerializeMegakernelDescription(
    const MegakernelDescription& description);
std::optional<MegakernelDescription> ParseMegakernelDescription(
    absl::Span<const uint8_t> bytes);

nvinfer1::IPluginV3* CreateMegakernelPlugin(
    const MegakernelDescription& description) noexcept;

// Referenced by the dispatch library so the creator's registration object is
// retained when linking the shared library used for engine deserialization.
void EnsureMegakernelPluginRegistered() noexcept;

}  // namespace litert::nvidia

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_EXPERIMENTAL_MEGAKERNEL_MEGAKERNEL_PLUGIN_H_
