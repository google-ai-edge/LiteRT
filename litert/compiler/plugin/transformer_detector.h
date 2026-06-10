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

#ifndef ODML_LITERT_LITERT_COMPILER_PLUGIN_TRANSFORMER_DETECTOR_H_
#define ODML_LITERT_LITERT_COMPILER_PLUGIN_TRANSFORMER_DETECTOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"

// Transformer-block pattern detection for the LiteRT partitioner.
//
// This component sits between the IHV (vendor) compiler plugin's op-selection
// (`CompilerPlugin::Partition`) and the graph-slicing step that hands work to
// the delegate. Vendor plugins report which individual ops they can accelerate,
// but they have no view of the higher-level *structure* of the graph. Many NPU
// delegates compile and schedule far more efficiently when an entire transformer
// block (attention + feed-forward + the surrounding normalization) is presented
// as a single, self-contained partition rather than as a scattering of small
// per-op islands.
//
// `TransformerBlockDetector` scans the use-def graph of a subgraph, finds the
// subgraphs of ops that form a transformer block, and rewrites the per-op
// partition indices so that all ops belonging to the same block share one
// index. The existing grouping strategies (see algo.h) then naturally outline
// each block into its own subgraph for the delegate to compile.
//
// The detector is conservative: it only ever *merges* the partition indices of
// ops the vendor already selected. It never adds an unselected op to a
// partition, so it cannot ask a delegate to compile something it did not claim
// to support.

namespace litert::internal {

// A single detected transformer block: the ordered list of ops that make it up
// plus a coarse classification of the dominant sub-structure that triggered the
// match. `ops` is always a subset of the ops the vendor selected.
struct TransformerBlock {
  enum class Kind {
    // Self-attention core: Q*K^T -> scale -> softmax -> *V, expressed with
    // batch_matmul/fully_connected, an optional scale mul/div, and a softmax.
    kAttention,
    // Gated/MLP feed-forward: two projections with a non-linearity
    // (gelu/logistic) and an element-wise gate in between.
    kFeedForward,
    // Pre/post layer normalization or RMS-norm cluster
    // (mean/sub/square/rsqrt/mul/add).
    kNormalization,
    // A full decoder/encoder block fused from the above when they are
    // contiguous in execution order and connected.
    kFullBlock,
  };

  // The attention span of a decoder layer. Gemma-style models interleave
  // sliding-window (local) layers with full-context (global) layers; an IHV
  // delegate may map them to different hardware paths or buffer budgets.
  enum class AttentionScope {
    kUnknown,
    kLocal,   // sliding-window attention (uses the local mask / local RoPE)
    kGlobal,  // full-context attention (uses the global mask / global RoPE)
  };

  Kind kind;
  std::vector<LiteRtOp> ops;
  // The partition index assigned to every op in this block.
  LiteRtParamIndex partition_index;

  // Attention span of this layer, inferred from which mask / position-embedding
  // / KV-cache tensors the block consumes. kUnknown when the block is not an
  // attention-bearing layer or the signals are absent.
  AttentionScope attention_scope = AttentionScope::kUnknown;

  // Zero-based layer index in execution order among attention-bearing blocks
  // (the Nth decoder layer). -1 for non-attention blocks. Used as the identity
  // for KV-cache-sharing relationships.
  int layer_index = -1;

  // Names of the KV-cache tensors (subgraph inputs/outputs) this layer reads or
  // writes, e.g. "...kv_cache_k_3". Empty if none were identified.
  std::vector<std::string> kv_cache_tensors;

  // If this layer reuses the KV cache produced/consumed by an earlier layer
  // (cross-layer KV sharing, as in some grouped/global-cache designs), this is
  // that layer's `layer_index`. -1 when the layer owns its own KV cache.
  int shares_kv_cache_with = -1;
};

// Tunables controlling how aggressively blocks are detected and merged. The
// defaults are chosen for decoder-only LLMs (Gemma/Llama-style) but the knobs
// allow vendors to adapt to their op-coverage and scheduling constraints.
struct TransformerDetectorOptions {
  // Minimum number of vendor-selected ops a candidate block must contain to be
  // treated as a block. Filters out incidental matches in non-transformer
  // graphs.
  size_t min_block_ops = 4;

  // When true, adjacent attention / feed-forward / normalization clusters that
  // are connected in the use-def graph are fused into a single kFullBlock
  // partition. When false, each cluster becomes its own partition.
  bool fuse_adjacent_clusters = true;

  // When true, a softmax is required for a region to be classified as
  // attention. Disable for linear-attention variants.
  bool require_softmax_for_attention = true;

  // When true, a connected region that contains more than one self-attention
  // (i.e. several softmaxes — the whole decoder stack arrives as a single
  // component because the residual stream links every layer) is segmented into
  // one block per layer, using the self-attention softmax as the per-layer
  // anchor. This is what an NPU delegate wants: bounded, self-contained
  // per-layer partitions rather than one partition spanning the entire model.
  //
  // When false, the entire connected region is emitted as one block.
  bool segment_into_layers = true;

  // When true, each attention-bearing block is annotated with its attention
  // scope (local/global), layer index, KV-cache tensors, and any cross-layer
  // KV-cache-sharing relationship. Inference is name- and shape-based over the
  // subgraph I/O tensors the block touches; disable to skip the extra walk.
  bool annotate_attention_metadata = true;

  // Layer indices at which to start a new partition. A cut value `c` means
  // "layer `c` begins a new partition". For a 35-layer model, cuts {12, 24}
  // yield three partitions: layers [0,12), [12,24), [24,35). Out-of-range or
  // unsorted values are sanitized (sorted, deduplicated) before use; cut 0 is
  // ignored since every grouping already starts a partition at layer 0.
  //
  // How an EMPTY list is interpreted depends on `coalesce_layers`:
  //   * coalesce_layers == false (TransformerBlock): empty -> one partition per
  //     transformer layer.
  //   * coalesce_layers == true (TransformerLayerCut): empty -> ALL layers
  //     coalesced into a single partition ("cut nowhere"). Non-empty always
  //     coalesces consecutive layers between cuts regardless of this flag.
  std::vector<int> layer_cut_indices;

  // Selects how empty `layer_cut_indices` is treated (see above). Set true for
  // the TransformerLayerCut strategy so an unspecified graph becomes one
  // partition rather than one-per-layer.
  bool coalesce_layers = false;
};

// Detects transformer blocks among the vendor-selected ops of `subgraph`.
//
// `selected_ops` is the output of `CompilerPlugin::Partition` for `subgraph`:
// the ops the delegate claims it can compile, each tagged with the vendor's own
// partition index. The returned vector lists the blocks found, in execution
// order; ops not belonging to any block are not reported.
//
// This call is read-only: it does not mutate the model.
std::vector<TransformerBlock> DetectTransformerBlocks(
    const std::vector<LiteRtOpWithPartitionIndex>& selected_ops,
    LiteRtSubgraph subgraph, const TransformerDetectorOptions& options = {});

// Rewrites the partition indices of `selected_ops` in place so that every op
// belonging to a detected transformer block shares a single, unique partition
// index, while ops outside any block keep distinct indices. The result is a
// drop-in replacement for the vendor's `selected_ops` that, when passed to a
// grouping strategy (GroupPartitions/GroupPartitionsV2), yields one partition
// per transformer block.
//
// Returns the number of blocks that were re-grouped.
size_t RepartitionByTransformerBlocks(
    std::vector<LiteRtOpWithPartitionIndex>& selected_ops,
    LiteRtSubgraph subgraph, const TransformerDetectorOptions& options = {});

// Resolves a per-signature layer-cut spec to the cut indices for one signature.
//
// The spec is a ';'-separated list of groups, each "key=cuts" where `cuts` is a
// ','-separated list of layer indices. A group with an empty key (or a bare
// list with no '=') is the DEFAULT applied to any signature without an explicit
// group. An explicit group for `signature_key` overrides the default.
//
// Examples:
//   "16"                         -> all signatures cut at {16}
//   "prefill_128=16,32;decode=8" -> prefill_128 {16,32}, decode {8}, others {}
//   "=16;decode=8"               -> decode {8}, all others {16}
//
// Whitespace around keys and numbers is tolerated. Non-numeric cut tokens are
// skipped. Returns the resolved (unsorted, unsanitized) cut indices for
// `signature_key`; empty when neither an explicit group nor a default applies.
std::vector<int> ResolveLayerCuts(absl::string_view spec,
                                  absl::string_view signature_key);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_COMPILER_PLUGIN_TRANSFORMER_DETECTOR_H_
