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

#include "litert/compiler/plugin/transformer_detector.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/core/model/model.h"

namespace litert::internal {
namespace {

// True if `op` is a StableHLO composite normalization (odml.rms_norm /
// odml.l2_norm / odml.group_norm). These arrive as a single composite op when
// the vendor plugin supports the composite directly (so it is not inlined into
// its mean/rsqrt/mul decomposition). The transformer-block detector must treat
// such a composite as the layer's normalization signal, exactly as it treats
// the decomposed form.
bool IsNormCompositeOp(LiteRtOp op) {
  if (op->OpCode() != kLiteRtOpCodeShloComposite) {
    return false;
  }
  auto info = GetOptionsAs<CompositeOptions>(op);
  if (!info) {
    return false;
  }
  return info->name == CompositeOptions::kRmsNorm ||
         info->name == CompositeOptions::kL2Norm ||
         info->name == CompositeOptions::kGroupNorm;
}

// Op-code histogram for a connected region of selected ops. Only the codes that
// participate in transformer-block classification are tracked; everything else
// is rolled into `other`.
struct OpHistogram {
  size_t matmul = 0;       // batch_matmul / fully_connected (projections, QK^T)
  size_t softmax = 0;      // attention probability normalization
  size_t activation = 0;   // gelu / logistic / tanh (feed-forward non-linearity)
  size_t norm_core = 0;    // mean / square / rsqrt / sum (layer/RMS-norm core)
  size_t elementwise = 0;  // add / sub / mul / div (scale, residual, gate)
  size_t reshape = 0;      // reshape / transpose (head split/merge)
  size_t norm_composite = 0;  // odml.rms_norm / l2_norm / group_norm composite
  size_t total = 0;
};

bool IsMatmul(LiteRtOpCode code) {
  return code == kLiteRtOpCodeTflBatchMatmul ||
         code == kLiteRtOpCodeTflFullyConnected;
}

bool IsActivation(LiteRtOpCode code) {
  return code == kLiteRtOpCodeTflGelu || code == kLiteRtOpCodeTflLogistic ||
         code == kLiteRtOpCodeTflTanh;
}

bool IsNormCore(LiteRtOpCode code) {
  return code == kLiteRtOpCodeTflMean || code == kLiteRtOpCodeTflSquare ||
         code == kLiteRtOpCodeTflRsqrt || code == kLiteRtOpCodeTflSum ||
         code == kLiteRtOpCodeTflPow;
}

bool IsElementwise(LiteRtOpCode code) {
  return code == kLiteRtOpCodeTflAdd || code == kLiteRtOpCodeTflSub ||
         code == kLiteRtOpCodeTflMul || code == kLiteRtOpCodeTflDiv;
}

bool IsShapeOp(LiteRtOpCode code) {
  return code == kLiteRtOpCodeTflReshape || code == kLiteRtOpCodeTflTranspose ||
         code == kLiteRtOpCodeTflConcatenation ||
         code == kLiteRtOpCodeTflStridedSlice || code == kLiteRtOpCodeTflCast;
}

OpHistogram Tally(const std::vector<LiteRtOp>& ops) {
  OpHistogram h;
  for (auto* op : ops) {
    const auto code = op->OpCode();
    ++h.total;
    if (IsMatmul(code)) {
      ++h.matmul;
    } else if (code == kLiteRtOpCodeTflSoftmax) {
      ++h.softmax;
    } else if (IsActivation(code)) {
      ++h.activation;
    } else if (IsNormCore(code)) {
      ++h.norm_core;
    } else if (IsElementwise(code)) {
      ++h.elementwise;
    } else if (IsShapeOp(code)) {
      ++h.reshape;
    } else if (IsNormCompositeOp(op)) {
      // A normalization composite is a complete norm on its own, so it counts
      // as both a norm_core and the elementwise scale it folds in.
      ++h.norm_composite;
      ++h.norm_core;
      ++h.elementwise;
    }
  }
  return h;
}

// Classifies a connected region. Returns false (via the out-param staying
// unset) for regions that do not resemble any transformer sub-structure.
bool Classify(const OpHistogram& h,
              const TransformerDetectorOptions& options,
              TransformerBlock::Kind& kind_out) {
  // Attention: at least one matmul (QK^T / *V) plus a softmax. Some delegates
  // implement linear attention without softmax, hence the toggle.
  const bool has_attention =
      h.matmul >= 1 &&
      (h.softmax >= 1 || !options.require_softmax_for_attention);

  // Feed-forward: matmul projection(s) with a non-linearity. Gemma-style gated
  // MLPs additionally carry an element-wise gate, but the activation is the
  // discriminating signal.
  const bool has_feed_forward = h.matmul >= 1 && h.activation >= 1;

  // Normalization: either the decomposed mean/square/rsqrt core (with a
  // scale/bias elementwise) OR a single normalization composite (odml.rms_norm
  // etc.), which is self-contained.
  const bool has_norm =
      h.norm_composite >= 1 || (h.norm_core >= 1 && h.elementwise >= 1);

  if (has_attention) {
    kind_out = TransformerBlock::Kind::kAttention;
    return true;
  }
  if (has_feed_forward) {
    kind_out = TransformerBlock::Kind::kFeedForward;
    return true;
  }
  if (has_norm) {
    kind_out = TransformerBlock::Kind::kNormalization;
    return true;
  }
  return false;
}

// Finds connected components among `selected` using the use-def graph,
// restricted to ops in `selected`. This mirrors the connectivity walk in
// algo.cc's DisjointSets but operates over the raw selected-op set rather than
// per-vendor partition indices, since we want structural blocks regardless of
// how the vendor numbered them.
//
// Returns components in deterministic execution order (the order ops appear in
// the subgraph), so downstream partition indices are stable across runs.
std::vector<std::vector<LiteRtOp>> ConnectedComponents(
    const absl::flat_hash_set<LiteRtOp>& selected, LiteRtSubgraph subgraph) {
  // Union-find over the selected set.
  absl::flat_hash_map<LiteRtOp, LiteRtOp> parent;
  parent.reserve(selected.size());
  for (auto* op : selected) {
    parent[op] = op;
  }

  std::function<LiteRtOp(LiteRtOp)> find = [&](LiteRtOp op) -> LiteRtOp {
    auto& p = parent[op];
    if (p != op) {
      p = find(p);
    }
    return p;
  };
  auto unite = [&](LiteRtOp a, LiteRtOp b) {
    auto ra = find(a);
    auto rb = find(b);
    if (ra != rb) {
      parent[ra] = rb;
    }
  };

  // Connect an op to any selected op that consumes one of its outputs.
  for (auto* op : selected) {
    for (auto* output : op->Outputs()) {
      for (auto* user : output->Users()) {
        if (selected.contains(user)) {
          unite(op, user);
        }
      }
    }
  }

  // Bucket ops by representative, preserving subgraph execution order.
  absl::flat_hash_map<LiteRtOp, size_t> root_to_index;
  std::vector<std::vector<LiteRtOp>> components;
  for (auto* op : subgraph->Ops()) {
    if (!selected.contains(op)) {
      continue;
    }
    auto root = find(op);
    auto it = root_to_index.find(root);
    if (it == root_to_index.end()) {
      root_to_index[root] = components.size();
      components.push_back({op});
    } else {
      components[it->second].push_back(op);
    }
  }
  return components;
}

// True if `op` reads a tensor whose name contains `needle`.
bool ReadsNamed(LiteRtOp op, absl::string_view needle) {
  for (auto* in : op->Inputs()) {
    if (absl::StrContains(in->Name(), needle)) {
      return true;
    }
  }
  return false;
}

bool ReadsKvCache(LiteRtOp op) { return ReadsNamed(op, "kv_cache"); }

// True if `op` reads a rotary position embedding (RoPE). RoPE is applied to Q/K
// at the very start of a decoder layer's attention, before the KV-cache read
// and softmax, so it is the most reliable marker of a layer's beginning.
bool ReadsPosEmb(LiteRtOp op) { return ReadsNamed(op, "pos_emb"); }

// Segments a connected component that spans multiple transformer layers into
// one sub-component per layer. The whole decoder stack arrives as a single
// connected component because the residual stream threads through every layer;
// an NPU delegate, however, wants bounded per-layer partitions.
//
// The ops are in subgraph execution order (ConnectedComponents guarantees it).
// The reliable per-layer landmark is the RoPE position-embedding read: it is
// the FIRST attention op of a layer, ahead of the KV-cache read, mask and
// softmax. The KV cache is NOT a safe boundary marker because graphs differ in
// where the V read lands relative to the softmax (the `probs * V` matmul can
// execute *after* softmax, e.g. Gemma4-12B), so cutting on any kv read can slice
// a layer's own V read into the next segment.
//
// We therefore close the current segment when we encounter the next layer's
// RoPE read AFTER having already seen this segment's softmax. The boundary lands
// in the FFN gap (after softmax_N, before pos_emb_{N+1}), so every layer keeps
// its own pos_emb, KV tensors, mask and softmax. When a graph exposes no pos_emb
// names (e.g. the synthetic unit tests), we fall back to softmax-to-softmax
// cuts.
//
// The LAST segment produced this way also contains the trailing logits head
// (final norm + vocab projection), because no further RoPE/softmax follows to
// trigger a cut. SplitTrailingHead (below) peels that head off so it becomes a
// separate, non-transformer partition rather than being fused into the last
// decoder layer.
std::vector<std::vector<LiteRtOp>> SegmentByLayer(
    const std::vector<LiteRtOp>& component) {
  const bool has_pos_emb =
      std::any_of(component.begin(), component.end(), ReadsPosEmb);

  std::vector<std::vector<LiteRtOp>> layers;
  std::vector<LiteRtOp> current;
  bool seen_softmax = false;  // softmax seen in the current segment

  auto flush = [&]() {
    layers.push_back(std::move(current));
    current.clear();
    seen_softmax = false;
  };

  for (auto* op : component) {
    const bool is_softmax = op->OpCode() == kLiteRtOpCodeTflSoftmax;

    if (has_pos_emb) {
      // Cut just before the next layer's RoPE read.
      if (seen_softmax && ReadsPosEmb(op) && !current.empty()) {
        flush();
      }
    } else if (is_softmax && seen_softmax && !current.empty()) {
      // Fallback for graphs without pos_emb tensors.
      flush();
    }

    current.push_back(op);
    if (is_softmax) {
      seen_softmax = true;
    }
  }
  if (!current.empty()) {
    layers.push_back(std::move(current));
  }
  return layers;
}

// Within a single segment's ordered ops, returns the index of the first op that
// belongs to the trailing logits head, or -1 if the segment has no such tail.
//
// The logits head is the run of ops AFTER the segment's last self-attention
// (softmax) that carries no further attention and ends at the model's logits
// output. We anchor on the last softmax: everything strictly after the last
// softmax and after the attention's output projection is the post-attention
// FFN + final norm + LM head. We only peel when that tail actually drives a
// non-KV subgraph output (the logits), so intermediate decoder layers — whose
// tails feed the next layer, not an output — are never split.
int FindLogitsHeadStart(const std::vector<LiteRtOp>& segment,
                        const absl::flat_hash_set<LiteRtTensor>& sg_outputs) {
  // Index of the last softmax in the segment.
  int last_softmax = -1;
  for (int i = 0; i < static_cast<int>(segment.size()); ++i) {
    if (segment[i]->OpCode() == kLiteRtOpCodeTflSoftmax) {
      last_softmax = i;
    }
  }
  if (last_softmax < 0) {
    return -1;  // not an attention segment; nothing to peel
  }

  // Does any op after the last softmax produce a large, non-KV subgraph output
  // (the logits)? KV-cache updates are also outputs, so exclude tensors whose
  // name marks them as KV caches.
  auto is_logits_output = [&](LiteRtTensor t) {
    return sg_outputs.contains(t) &&
           !absl::StrContains(t->Name(), "kv_cache");
  };
  bool tail_drives_logits = false;
  for (int i = last_softmax + 1; i < static_cast<int>(segment.size()); ++i) {
    for (auto* out : segment[i]->Outputs()) {
      if (is_logits_output(out)) {
        tail_drives_logits = true;
        break;
      }
    }
    if (tail_drives_logits) break;
  }
  if (!tail_drives_logits) {
    return -1;
  }

  // Peel from the final normalization that precedes the logits projection: the
  // last norm op in the segment (RMS/L2/group composite, or the decomposed
  // mean/rsqrt core) marks the start of the head. Fall back to "just after the
  // attention output projection" — i.e. the first matmul after the last softmax
  // — if no explicit norm is found.
  int head_start = -1;
  for (int i = last_softmax + 1; i < static_cast<int>(segment.size()); ++i) {
    const auto code = segment[i]->OpCode();
    if (IsNormCompositeOp(segment[i]) || IsNormCore(code)) {
      head_start = i;  // keep advancing to the LAST norm
    }
  }
  if (head_start < 0) {
    for (int i = last_softmax + 1; i < static_cast<int>(segment.size()); ++i) {
      if (IsMatmul(segment[i]->OpCode())) {
        head_start = i;
        break;
      }
    }
  }
  return head_start;
}

// Collects the set of tensors a block touches that are also subgraph
// inputs/outputs (i.e. cross the partition boundary). KV caches, masks, and
// position embeddings all enter the per-layer block as such boundary tensors,
// so they are the signals we classify against.
absl::flat_hash_set<LiteRtTensor> BoundaryTensors(
    const std::vector<LiteRtOp>& block, LiteRtSubgraph subgraph) {
  absl::flat_hash_set<LiteRtTensor> sg_io;
  for (auto* t : subgraph->Inputs()) {
    sg_io.insert(t);
  }
  for (auto* t : subgraph->Outputs()) {
    sg_io.insert(t);
  }
  absl::flat_hash_set<LiteRtTensor> touched;
  for (auto* op : block) {
    for (auto* in : op->Inputs()) {
      if (sg_io.contains(in)) {
        touched.insert(in);
      }
    }
    for (auto* out : op->Outputs()) {
      if (sg_io.contains(out)) {
        touched.insert(out);
      }
    }
  }
  return touched;
}

// True if `name` looks like a KV-cache tensor.
bool IsKvCacheName(absl::string_view name) {
  return absl::StrContains(name, "kv_cache");
}

// Annotates a block in place with attention scope, KV-cache tensors, and layer
// index. Scope is inferred from the names of the boundary tensors the block
// consumes: Gemma exposes distinct `mask_local`/`mask_global` and
// `pos_emb_local_*`/`pos_emb_*` tensors per layer. Returns true if the block is
// attention-bearing (so the caller can assign a layer index).
bool AnnotateAttentionMetadata(TransformerBlock& block,
                               LiteRtSubgraph subgraph) {
  const bool is_attention = block.kind == TransformerBlock::Kind::kAttention ||
                            block.kind == TransformerBlock::Kind::kFullBlock;
  if (!is_attention) {
    return false;
  }

  auto boundary = BoundaryTensors(block.ops, subgraph);

  // The authoritative per-layer scope signal is the rotary position embedding:
  // sliding-window layers read `pos_emb_local_*` (a shorter table) while
  // full-context layers read the global `pos_emb_{cos,sin}`. Every layer reads
  // exactly one of these in its attention preamble. The attention mask is NOT
  // reliable: a graph may expose a single shared `mask_global`/`mask_local`
  // tensor that several layers touch (observed on Gemma4-E2B/12B), so it is used
  // only as a fallback tiebreak when no pos_emb signal is present.
  bool pos_local = false;
  bool pos_global = false;
  bool mask_local = false;
  bool mask_global = false;
  for (auto* t : boundary) {
    const absl::string_view name = t->Name();
    if (IsKvCacheName(name)) {
      block.kv_cache_tensors.emplace_back(name);
    }
    if (absl::StrContains(name, "pos_emb_local")) {
      pos_local = true;
    } else if (absl::StrContains(name, "pos_emb_cos") ||
               absl::StrContains(name, "pos_emb_sin")) {
      pos_global = true;
    } else if (absl::StrContains(name, "mask_local")) {
      mask_local = true;
    } else if (absl::StrContains(name, "mask_global")) {
      mask_global = true;
    }
  }

  // Prefer pos_emb; fall back to mask only when pos_emb is absent/ambiguous.
  bool is_local = pos_local;
  bool is_global = pos_global;
  if (pos_local == pos_global) {  // both or neither pos_emb signal
    is_local = mask_local;
    is_global = mask_global;
  }

  if (is_local && !is_global) {
    block.attention_scope = TransformerBlock::AttentionScope::kLocal;
  } else if (is_global && !is_local) {
    block.attention_scope = TransformerBlock::AttentionScope::kGlobal;
  } else {
    block.attention_scope = TransformerBlock::AttentionScope::kUnknown;
  }

  std::sort(block.kv_cache_tensors.begin(), block.kv_cache_tensors.end());
  return true;
}

// Populates `shares_kv_cache_with` by detecting layers whose KV-cache tensor
// sets overlap. The first layer to reference a given KV-cache tensor owns it;
// any later layer that references the same tensor is recorded as sharing with
// the owner. This captures cross-layer KV-cache reuse without assuming a
// particular naming convention beyond tensor identity.
void LinkSharedKvCaches(std::vector<TransformerBlock>& blocks) {
  // Map a KV-cache tensor name to the layer_index that first owned it.
  absl::flat_hash_map<std::string, int> owner_of;
  for (auto& block : blocks) {
    if (block.layer_index < 0 || block.kv_cache_tensors.empty()) {
      continue;
    }
    for (const auto& name : block.kv_cache_tensors) {
      auto it = owner_of.find(name);
      if (it == owner_of.end()) {
        owner_of[name] = block.layer_index;
      } else if (it->second != block.layer_index) {
        // This layer reuses a cache another layer already owns.
        block.shares_kv_cache_with = it->second;
      }
    }
  }
}

}  // namespace

std::vector<TransformerBlock> DetectTransformerBlocks(
    const std::vector<LiteRtOpWithPartitionIndex>& selected_ops,
    LiteRtSubgraph subgraph, const TransformerDetectorOptions& options) {
  std::vector<TransformerBlock> blocks;
  if (selected_ops.empty()) {
    return blocks;
  }

  absl::flat_hash_set<LiteRtOp> selected;
  selected.reserve(selected_ops.size());
  for (const auto& [op, _] : selected_ops) {
    selected.insert(op);
  }

  // Subgraph outputs, used to identify the trailing logits head (an op whose
  // output is a non-KV subgraph output).
  absl::flat_hash_set<LiteRtTensor> sg_outputs(subgraph->Outputs().begin(),
                                               subgraph->Outputs().end());

  auto components = ConnectedComponents(selected, subgraph);

  // Classify each connected component.
  LiteRtParamIndex next_index = 0;
  for (auto& component : components) {
    if (component.size() < options.min_block_ops) {
      continue;
    }
    const auto histogram = Tally(component);
    TransformerBlock::Kind kind;
    if (!Classify(histogram, options, kind)) {
      continue;
    }

    // A connected component carrying more than one self-attention (multiple
    // softmaxes) is the whole decoder stack arriving as one component because
    // the residual stream links every layer. Segment it into per-layer blocks
    // so the delegate gets bounded partitions instead of one model-sized one.
    if (options.segment_into_layers && histogram.softmax >= 2) {
      auto layers = SegmentByLayer(component);

      // Peel the trailing logits head (final norm + vocab projection) off the
      // last segment so it becomes a separate, non-transformer partition rather
      // than being fused into the last decoder layer. The head is emitted as a
      // kNormalization block: it is NOT counted as a transformer layer (see
      // AnnotateAttentionMetadata) and gets its own partition.
      std::vector<LiteRtOp> logits_head;
      if (!layers.empty()) {
        auto& last = layers.back();
        const int head_start = FindLogitsHeadStart(last, sg_outputs);
        if (head_start > 0) {
          logits_head.assign(last.begin() + head_start, last.end());
          last.erase(last.begin() + head_start, last.end());
        }
      }

      for (auto& layer : layers) {
        if (layer.size() < options.min_block_ops) {
          continue;
        }
        const auto layer_histogram = Tally(layer);
        TransformerBlock::Kind layer_kind;
        if (!Classify(layer_histogram, options, layer_kind)) {
          continue;
        }
        if (options.fuse_adjacent_clusters) {
          const int signals =
              (layer_histogram.matmul >= 1 && layer_histogram.softmax >= 1) +
              (layer_histogram.activation >= 1) +
              (layer_histogram.norm_core >= 1);
          if (signals >= 2) {
            layer_kind = TransformerBlock::Kind::kFullBlock;
          }
        }
        blocks.push_back(
            TransformerBlock{layer_kind, std::move(layer), next_index++});
      }

      // Emit the logits head as its own non-transformer block (always, so its
      // ops form one partition instead of fragmenting).
      if (!logits_head.empty()) {
        blocks.push_back(TransformerBlock{TransformerBlock::Kind::kNormalization,
                                          std::move(logits_head),
                                          next_index++});
      }
      continue;
    }

    // When clusters are fused, a connected component that contains more than one
    // structural signal (attention + feed-forward, or norm + either) is a full
    // block. Connectivity already did the fusing: a single component spanning
    // attention and MLP means they share tensors and should ship together.
    if (options.fuse_adjacent_clusters) {
      const int signals = (histogram.matmul >= 1 && histogram.softmax >= 1) +
                          (histogram.activation >= 1) +
                          (histogram.norm_core >= 1);
      if (signals >= 2) {
        kind = TransformerBlock::Kind::kFullBlock;
      }
    }
    blocks.push_back(TransformerBlock{kind, std::move(component), next_index++});
  }

  // Annotate attention-bearing blocks with scope / layer index / KV caches, in
  // execution order, then link cross-layer KV-cache sharing.
  if (options.annotate_attention_metadata) {
    int layer_index = 0;
    for (auto& block : blocks) {
      if (AnnotateAttentionMetadata(block, subgraph)) {
        block.layer_index = layer_index++;
      }
    }
    LinkSharedKvCaches(blocks);
  }

  return blocks;
}

size_t RepartitionByTransformerBlocks(
    std::vector<LiteRtOpWithPartitionIndex>& selected_ops,
    LiteRtSubgraph subgraph, const TransformerDetectorOptions& options) {
  auto blocks = DetectTransformerBlocks(selected_ops, subgraph, options);
  if (blocks.empty()) {
    LITERT_LOG(LITERT_INFO,
               "TransformerBlockDetector: no transformer blocks found, "
               "leaving %lu selected ops untouched.",
               selected_ops.size());
    return 0;
  }

  // Optionally coalesce consecutive transformer LAYERS into multi-layer
  // partitions, breaking only at the requested cut indices. We remap each
  // block's partition index to the index of the inter-cut "bucket" its layer
  // falls in. Non-attention blocks (layer_index < 0) keep their own block index
  // offset past the buckets so they are never merged into a layer group.
  //
  // Example: layers 0..34, cuts {12,24} -> buckets [0,12)->0, [12,24)->1,
  // [24,35)->2. Layers in the same bucket share a partition index, so the
  // union-find grouping (GroupPartitions) emits one partition per bucket.
  //
  // Coalescing runs when explicit cuts were given, OR when `coalesce_layers` is
  // set (the TransformerLayerCut strategy) even with no cuts: an empty cut list
  // then puts every layer in bucket 0 -> the whole stack becomes one partition
  // ("cut nowhere"), rather than the one-partition-per-layer fallback.
  absl::flat_hash_map<LiteRtParamIndex, LiteRtParamIndex> block_to_group;
  if (!options.layer_cut_indices.empty() || options.coalesce_layers) {
    // Sanitize cuts: drop <=0 and duplicates, sort ascending.
    std::vector<int> cuts;
    for (int c : options.layer_cut_indices) {
      if (c > 0) {
        cuts.push_back(c);
      }
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

    // Assign each attention layer to a bucket: bucket = number of cuts <= its
    // layer_index. Reuse one group index per (subgraph-)distinct bucket.
    LiteRtParamIndex next_group = 0;
    absl::flat_hash_map<int, LiteRtParamIndex> bucket_to_group;
    for (const auto& block : blocks) {
      if (block.layer_index < 0) {
        continue;  // handled below as its own group
      }
      int bucket = 0;
      for (int c : cuts) {
        if (block.layer_index >= c) {
          ++bucket;
        }
      }
      auto it = bucket_to_group.find(bucket);
      if (it == bucket_to_group.end()) {
        bucket_to_group[bucket] = next_group;
        block_to_group[block.partition_index] = next_group;
        ++next_group;
      } else {
        block_to_group[block.partition_index] = it->second;
      }
    }
    // Non-attention blocks each get a unique trailing group so they stay
    // separate partitions.
    for (const auto& block : blocks) {
      if (block.layer_index < 0) {
        block_to_group[block.partition_index] = next_group++;
      }
    }
  }

  auto group_of = [&](LiteRtParamIndex block_index) -> LiteRtParamIndex {
    if (block_to_group.empty()) {
      return block_index;  // one partition per layer (no cuts)
    }
    return block_to_group[block_index];
  };

  // Map each op that belongs to a block to its assigned (possibly grouped)
  // partition index.
  absl::flat_hash_map<LiteRtOp, LiteRtParamIndex> op_to_block;
  for (const auto& block : blocks) {
    const LiteRtParamIndex group = group_of(block.partition_index);
    for (auto* op : block.ops) {
      op_to_block[op] = group;
    }
  }

  LiteRtParamIndex max_group = 0;
  for (const auto& [op, group] : op_to_block) {
    max_group = std::max(max_group, group);
  }

  // Decide the partition index for selected ops that are NOT part of any
  // detected block (e.g. the input-embedding preamble before layer 0, or small
  // interstitial ops the layer segmentation did not absorb).
  //
  // When coalescing (the layer-cut strategy), such stray ops must NOT each
  // become their own partition — that is what split prefill_128 into 8 when it
  // should be 1. Instead, fold each stray op into the group of the nearest
  // PRECEDING block in execution order (so the preamble joins the first bucket,
  // and an op between layers N and N+1 joins layer N's group). Connectivity in
  // GroupPartitions then merges them into the adjacent island. Strays that
  // precede every block fall into the first group (group 0).
  //
  // Without coalescing (the per-layer TransformerBlock strategy, or no detector
  // grouping at all), keep the original behavior: each stray op gets its own
  // distinct index so it is never merged into a neighboring block.
  absl::flat_hash_map<LiteRtOp, LiteRtParamIndex> stray_to_group;
  if (options.coalesce_layers && !op_to_block.empty()) {
    LiteRtParamIndex current_group = 0;  // first bucket, for leading preamble
    for (auto* op : subgraph->Ops()) {
      auto it = op_to_block.find(op);
      if (it != op_to_block.end()) {
        current_group = it->second;  // entered a block; track its group
      } else {
        stray_to_group[op] = current_group;
      }
    }
  }

  LiteRtParamIndex next_free_index =
      op_to_block.empty() ? 0 : max_group + 1;
  for (auto& [op, index] : selected_ops) {
    if (auto it = op_to_block.find(op); it != op_to_block.end()) {
      index = it->second;
    } else if (auto sit = stray_to_group.find(op); sit != stray_to_group.end()) {
      index = sit->second;
    } else {
      index = next_free_index++;
    }
  }

  // Count distinct partition indices actually assigned across ALL selected ops
  // (blocks + folded strays + any singleton strays).
  absl::flat_hash_set<LiteRtParamIndex> distinct_groups;
  for (const auto& [op, index] : selected_ops) {
    distinct_groups.insert(index);
  }
  const size_t num_partitions = distinct_groups.size();
  LITERT_LOG(LITERT_INFO,
             "TransformerBlockDetector: grouped %lu selected ops from %lu "
             "transformer layer(s) into %lu partition(s).",
             selected_ops.size(), blocks.size(), num_partitions);
  return num_partitions;
}

std::vector<int> ResolveLayerCuts(absl::string_view spec,
                                  absl::string_view signature_key) {
  auto parse_cut_list = [](absl::string_view list) {
    std::vector<int> cuts;
    for (absl::string_view tok : absl::StrSplit(list, ',', absl::SkipEmpty())) {
      tok = absl::StripAsciiWhitespace(tok);
      int value = 0;
      if (absl::SimpleAtoi(tok, &value)) {
        cuts.push_back(value);
      }
    }
    return cuts;
  };

  std::vector<int> default_cuts;
  bool have_default = false;
  std::vector<int> matched_cuts;
  bool have_match = false;

  for (absl::string_view group : absl::StrSplit(spec, ';', absl::SkipEmpty())) {
    group = absl::StripAsciiWhitespace(group);
    if (group.empty()) {
      continue;
    }
    const size_t eq = group.find('=');
    if (eq == absl::string_view::npos) {
      // Bare list: the default applied to all signatures.
      default_cuts = parse_cut_list(group);
      have_default = true;
      continue;
    }
    absl::string_view key =
        absl::StripAsciiWhitespace(group.substr(0, eq));
    absl::string_view list = group.substr(eq + 1);
    if (key.empty()) {
      default_cuts = parse_cut_list(list);
      have_default = true;
    } else if (key == signature_key) {
      matched_cuts = parse_cut_list(list);
      have_match = true;
    }
  }

  if (have_match) {
    return matched_cuts;
  }
  if (have_default) {
    return default_cuts;
  }
  return {};
}

}  // namespace litert::internal
