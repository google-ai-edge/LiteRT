/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Describes the external KV and mask boundaries of the authored Gemma4 graphs.
#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_ACTIVE_GRAPH_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_ACTIVE_GRAPH_CONTEXT_H_

#include <vector>

#include "tensor/tensor.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Per-layer KV cache and query boundary handles populated while authoring a
// Gemma4 graph signature. All handles default to Invalid() so unused fields on
// KV-sharing layers do not allocate dummy graph nodes.
struct ActiveLayerGraph {
  // Index of the layer that owns this layer's KV cache (equals the layer index
  // for owner layers and points to an earlier owner for shared layers).
  int owner = -1;
  // Whether this layer uses global attention (true) or local sliding attention.
  bool global = false;

  // External INT8 KV history and alignment-padding inputs (owner layers only).
  TensorHandle past_key = TensorHandle::Invalid();
  TensorHandle past_value = TensorHandle::Invalid();
  TensorHandle pad_key = TensorHandle::Invalid();
  TensorHandle pad_value = TensorHandle::Invalid();

  // Concatenated active KV tensors consumed by attention (shared across layers
  // that point to the same owner).
  TensorHandle active_key = TensorHandle::Invalid();
  TensorHandle active_value = TensorHandle::Invalid();

  // Flattened query tensor used when attention masks are not broadcast.
  TensorHandle query = TensorHandle::Invalid();

  // Newly produced INT8 KV outputs for the current chunk/token (owner layers).
  TensorHandle new_key = TensorHandle::Invalid();
  TensorHandle new_value = TensorHandle::Invalid();
};

// Mutable authoring state shared across layers while building one Gemma4
// signature (prefill or decode). Callers configure `broadcast_mask` and
// `dynamic_prefill_rows`, then inspect `layers`, `local_mask`, and
// `global_mask` after `BuildGemma4Graph` to name outputs and declare dynamic
// shape signatures.
struct ActiveGraphContext {
  // Index of the layer currently being authored by `BuildGemma4Graph`.
  int layer = 0;
  // If true, keep heads on the batch axis so one mask broadcasts across heads.
  bool broadcast_mask = false;
  // If true, author dynamic leading-dimension slices/reshapes for prefill rows.
  bool dynamic_prefill_rows = false;
  // Shared causal mask input for local sliding-window attention layers.
  TensorHandle local_mask = TensorHandle::Invalid();
  // Shared causal mask input for global attention layers.
  TensorHandle global_mask = TensorHandle::Invalid();
  // Mapping from each layer index to its KV cache owner layer index.
  std::vector<int> owners;
  // Per-layer boundary handles indexed by layer index.
  std::vector<ActiveLayerGraph> layers;
};

}  // namespace litert::tensor::examples::experimental::gemma4::cpu

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_ACTIVE_GRAPH_CONTEXT_H_
