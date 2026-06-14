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

#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/core/model/model.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

// Minimal in-memory graph builder for assembling synthetic transformer-shaped
// subgraphs. Each AddOp appends one op whose single output feeds the next op,
// forming a linear chain (sufficient for connectivity-based detection).
class ChainBuilder {
 public:
  explicit ChainBuilder(LiteRtSubgraphT& subgraph) : subgraph_(subgraph) {
    // Seed the chain with a subgraph input tensor.
    prev_ = &subgraph_.EmplaceTensor();
    subgraph_.Inputs().push_back(prev_);
  }

  // Appends an op of the given code, wiring the running tensor as its input and
  // a fresh tensor as its output. Returns the op for selection.
  LiteRtOp AddOp(LiteRtOpCode code) {
    auto& op = subgraph_.EmplaceOp();
    op.SetOpCode(code);
    AttachInput(prev_, op);
    auto& out = subgraph_.EmplaceTensor();
    AttachOutput(&out, op);
    prev_ = &out;
    return &op;
  }

  // Appends a StableHLO composite op with the given composite name (e.g.
  // "odml.rms_norm"), wired into the chain like AddOp.
  LiteRtOp AddCompositeOp(const std::string& composite_name) {
    auto& op = subgraph_.EmplaceOp();
    op.SetOpCode(kLiteRtOpCodeShloComposite);
    tflite::StableHLOCompositeOptionsT options;
    options.name = composite_name;
    options.decomposition_subgraph_index = 0;
    litert::internal::TflOptions2 tfl_options;
    tfl_options.type = ::tflite::BuiltinOptions2_StableHLOCompositeOptions;
    tfl_options.Set(std::move(options));
    litert::internal::SetTflOptions2(op, std::move(tfl_options));
    AttachInput(prev_, op);
    auto& out = subgraph_.EmplaceTensor();
    AttachOutput(&out, op);
    prev_ = &out;
    return &op;
  }

  // Appends an op that additionally reads a named subgraph-input tensor (e.g. a
  // KV cache, mask, or position embedding). The named tensor is registered as a
  // subgraph input so it crosses the partition boundary.
  LiteRtOp AddOpReading(LiteRtOpCode code, const std::string& tensor_name) {
    auto& named = subgraph_.EmplaceTensor();
    named.SetName(tensor_name);
    subgraph_.Inputs().push_back(&named);

    auto& op = subgraph_.EmplaceOp();
    op.SetOpCode(code);
    AttachInput(prev_, op);
    AttachInput(&named, op);
    auto& out = subgraph_.EmplaceTensor();
    AttachOutput(&out, op);
    prev_ = &out;
    return &op;
  }

  // Marks the current running tensor as a named subgraph output (e.g. logits).
  void MarkOutput(const std::string& name) {
    prev_->SetName(name);
    subgraph_.Outputs().push_back(prev_);
  }

 private:
  LiteRtSubgraphT& subgraph_;
  LiteRtTensor prev_ = nullptr;
};

// Builds one decoder layer following the real Gemma landmark order:
//   pos_emb (RoPE) -> KV-cache read -> mask -> softmax -> FFN.
// `scope_tag` is "local" or "global"; `idx` selects the per-layer KV cache.
// When `kv_idx` differs from `idx`, the layer reuses another layer's cache
// (cross-layer KV sharing).
std::vector<LiteRtOp> BuildDecoderLayer(ChainBuilder& b, const char* scope_tag,
                                        int idx, int kv_idx) {
  const std::string emb = std::string("pos_emb_") +
                          (std::string(scope_tag) == "local" ? "local_cos"
                                                             : "cos");
  const std::string mask = std::string("mask_") + scope_tag;
  return {
      b.AddOpReading(kLiteRtOpCodeTflMul, emb),  // RoPE on Q
      b.AddOpReading(kLiteRtOpCodeTflFullyConnected,
                     "kv_cache_k_" + std::to_string(kv_idx)),
      b.AddOpReading(kLiteRtOpCodeTflBatchMatmul,
                     "kv_cache_v_" + std::to_string(kv_idx)),
      b.AddOpReading(kLiteRtOpCodeTflAdd, mask),  // mask add
      b.AddOp(kLiteRtOpCodeTflSoftmax),
      b.AddOp(kLiteRtOpCodeTflBatchMatmul),     // probs * V
      b.AddOp(kLiteRtOpCodeTflFullyConnected),  // output projection
      b.AddOp(kLiteRtOpCodeTflFullyConnected),  // FFN up
      b.AddOp(kLiteRtOpCodeTflGelu),
      b.AddOp(kLiteRtOpCodeTflFullyConnected),  // FFN down
  };
}

std::vector<LiteRtOpWithPartitionIndex> SelectAll(
    const std::vector<LiteRtOp>& ops) {
  std::vector<LiteRtOpWithPartitionIndex> selected;
  selected.reserve(ops.size());
  for (auto* op : ops) {
    // Tag every op with the same vendor partition index; the detector is
    // expected to override these.
    selected.push_back({op, 0});
  }
  return selected;
}

// Builds a single attention block: reshape -> Q*K^T -> scale -> softmax -> *V
// -> output projection.
std::vector<LiteRtOp> BuildAttention(ChainBuilder& b) {
  return {
      b.AddOp(kLiteRtOpCodeTflReshape),
      b.AddOp(kLiteRtOpCodeTflBatchMatmul),  // Q * K^T
      b.AddOp(kLiteRtOpCodeTflMul),          // scale by 1/sqrt(d)
      b.AddOp(kLiteRtOpCodeTflSoftmax),      // attention probs
      b.AddOp(kLiteRtOpCodeTflBatchMatmul),  // probs * V
      b.AddOp(kLiteRtOpCodeTflFullyConnected),  // output projection
  };
}

// Builds a gated feed-forward block: up-projection -> gelu -> gate mul ->
// down-projection.
std::vector<LiteRtOp> BuildFeedForward(ChainBuilder& b) {
  return {
      b.AddOp(kLiteRtOpCodeTflFullyConnected),  // up projection
      b.AddOp(kLiteRtOpCodeTflGelu),            // activation
      b.AddOp(kLiteRtOpCodeTflMul),             // gate
      b.AddOp(kLiteRtOpCodeTflFullyConnected),  // down projection
  };
}

TEST(TransformerDetectorTest, DetectsAttentionBlock) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  auto attention = BuildAttention(b);

  auto selected = SelectAll(attention);
  auto blocks = DetectTransformerBlocks(selected, &subgraph);

  ASSERT_EQ(blocks.size(), 1);
  EXPECT_EQ(blocks.front().kind, TransformerBlock::Kind::kAttention);
  EXPECT_EQ(blocks.front().ops.size(), attention.size());
}

TEST(TransformerDetectorTest, DetectsFeedForwardBlock) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  auto ffn = BuildFeedForward(b);

  auto selected = SelectAll(ffn);
  auto blocks = DetectTransformerBlocks(selected, &subgraph);

  ASSERT_EQ(blocks.size(), 1);
  EXPECT_EQ(blocks.front().kind, TransformerBlock::Kind::kFeedForward);
}

TEST(TransformerDetectorTest, FusesContiguousAttentionAndFeedForward) {
  // A whole decoder block: attention immediately followed by feed-forward, all
  // connected in one chain. With fusion on (default), the single connected
  // component carries both attention and feed-forward signals, so it is
  // classified as a full block.
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  auto attention = BuildAttention(b);
  auto ffn = BuildFeedForward(b);

  std::vector<LiteRtOp> all = attention;
  all.insert(all.end(), ffn.begin(), ffn.end());

  auto selected = SelectAll(all);
  auto blocks = DetectTransformerBlocks(selected, &subgraph);

  ASSERT_EQ(blocks.size(), 1);
  EXPECT_EQ(blocks.front().kind, TransformerBlock::Kind::kFullBlock);
  EXPECT_EQ(blocks.front().ops.size(), all.size());
}

TEST(TransformerDetectorTest, RepartitionAssignsOneIndexPerBlock) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  auto attention = BuildAttention(b);

  auto selected = SelectAll(attention);
  const size_t num_blocks =
      RepartitionByTransformerBlocks(selected, &subgraph);
  ASSERT_EQ(num_blocks, 1);

  // Every op in the block must share the same (block 0) partition index.
  for (const auto& [op, index] : selected) {
    EXPECT_EQ(index, 0u);
  }
}

TEST(TransformerDetectorTest, IgnoresNonTransformerGraph) {
  // A plain elementwise chain has no attention/ffn/norm structure.
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  std::vector<LiteRtOp> ops = {
      b.AddOp(kLiteRtOpCodeTflAdd),
      b.AddOp(kLiteRtOpCodeTflMul),
      b.AddOp(kLiteRtOpCodeTflAdd),
      b.AddOp(kLiteRtOpCodeTflMul),
  };

  auto selected = SelectAll(ops);
  auto blocks = DetectTransformerBlocks(selected, &subgraph);
  EXPECT_TRUE(blocks.empty());

  // Repartition reports zero blocks, so downstream grouping behaves exactly as
  // before (the selected ops are left for the normal strategy to island).
  const size_t num_blocks =
      RepartitionByTransformerBlocks(selected, &subgraph);
  EXPECT_EQ(num_blocks, 0);
}

TEST(TransformerDetectorTest, RespectsMinBlockOps) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  auto attention = BuildAttention(b);  // 6 ops

  auto selected = SelectAll(attention);
  TransformerDetectorOptions options;
  options.min_block_ops = 100;  // Larger than the block.
  auto blocks = DetectTransformerBlocks(selected, &subgraph, options);
  EXPECT_TRUE(blocks.empty());
}

// Builds a 3-layer stack (local, global, local), each layer owning its own KV
// cache, and checks per-layer segmentation, scope classification, and layer
// indexing.
TEST(TransformerDetectorTest, SegmentsLayersAndClassifiesScope) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  std::vector<LiteRtOp> all;
  for (auto [scope, idx] :
       std::vector<std::pair<const char*, int>>{
           {"local", 0}, {"global", 1}, {"local", 2}}) {
    auto layer = BuildDecoderLayer(b, scope, idx, /*kv_idx=*/idx);
    all.insert(all.end(), layer.begin(), layer.end());
  }

  auto selected = SelectAll(all);
  auto blocks = DetectTransformerBlocks(selected, &subgraph);

  ASSERT_EQ(blocks.size(), 3);
  EXPECT_EQ(blocks[0].attention_scope,
            TransformerBlock::AttentionScope::kLocal);
  EXPECT_EQ(blocks[1].attention_scope,
            TransformerBlock::AttentionScope::kGlobal);
  EXPECT_EQ(blocks[2].attention_scope,
            TransformerBlock::AttentionScope::kLocal);

  EXPECT_EQ(blocks[0].layer_index, 0);
  EXPECT_EQ(blocks[1].layer_index, 1);
  EXPECT_EQ(blocks[2].layer_index, 2);

  // Each layer owns exactly its own k_N and v_N cache (no drift into the next
  // layer), and none share.
  EXPECT_EQ(blocks[0].kv_cache_tensors.size(), 2);
  EXPECT_EQ(blocks[1].kv_cache_tensors.size(), 2);
  EXPECT_EQ(blocks[0].shares_kv_cache_with, -1);
  EXPECT_EQ(blocks[1].shares_kv_cache_with, -1);
  EXPECT_EQ(blocks[2].shares_kv_cache_with, -1);
}

// Two layers that read the SAME KV cache (cross-layer KV sharing): the later
// layer must report sharing with the earlier one.
TEST(TransformerDetectorTest, DetectsSharedKvCache) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  std::vector<LiteRtOp> all;
  auto l0 = BuildDecoderLayer(b, "local", /*idx=*/0, /*kv_idx=*/0);
  auto l1 = BuildDecoderLayer(b, "local", /*idx=*/1, /*kv_idx=*/0);  // reuse 0
  all.insert(all.end(), l0.begin(), l0.end());
  all.insert(all.end(), l1.begin(), l1.end());

  auto selected = SelectAll(all);
  auto blocks = DetectTransformerBlocks(selected, &subgraph);

  ASSERT_EQ(blocks.size(), 2);
  EXPECT_EQ(blocks[0].layer_index, 0);
  EXPECT_EQ(blocks[1].layer_index, 1);
  EXPECT_EQ(blocks[0].shares_kv_cache_with, -1);
  // Layer 1 reuses layer 0's cache.
  EXPECT_EQ(blocks[1].shares_kv_cache_with, 0);
}

// Helper: builds an `n`-layer local-attention stack and returns the flat op
// list (each layer owns its own KV cache).
std::vector<LiteRtOp> BuildStack(ChainBuilder& b, int n) {
  std::vector<LiteRtOp> all;
  for (int i = 0; i < n; ++i) {
    auto layer = BuildDecoderLayer(b, "local", /*idx=*/i, /*kv_idx=*/i);
    all.insert(all.end(), layer.begin(), layer.end());
  }
  return all;
}

// Counts the distinct partition indices assigned across selected ops.
size_t DistinctPartitions(
    const std::vector<LiteRtOpWithPartitionIndex>& selected) {
  absl::flat_hash_set<LiteRtParamIndex> ids;
  for (const auto& [op, idx] : selected) {
    ids.insert(idx);
  }
  return ids.size();
}

// With layer cuts, consecutive layers are coalesced into one partition per
// inter-cut chunk.
TEST(TransformerDetectorTest, LayerCutsGroupConsecutiveLayers) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  auto all = BuildStack(b, /*n=*/6);  // layers 0..5

  // Cuts at {2, 4} -> buckets [0,2), [2,4), [4,6) => 3 partitions.
  auto selected = SelectAll(all);
  TransformerDetectorOptions options;
  options.layer_cut_indices = {2, 4};
  const size_t num_partitions =
      RepartitionByTransformerBlocks(selected, &subgraph, options);
  EXPECT_EQ(num_partitions, 3);
  EXPECT_EQ(DistinctPartitions(selected), 3);

  // Layers 0 and 1 must share a partition index; layer 2 must differ.
  // Find an op from each layer via its KV-cache input and compare indices.
  auto index_of_layer = [&](int kv_idx) -> LiteRtParamIndex {
    const std::string want = "kv_cache_k_" + std::to_string(kv_idx);
    for (const auto& [op, idx] : selected) {
      for (auto* in : op->Inputs()) {
        if (std::string(in->Name()).find(want) != std::string::npos) {
          return idx;
        }
      }
    }
    return std::numeric_limits<LiteRtParamIndex>::max();
  };
  EXPECT_EQ(index_of_layer(0), index_of_layer(1));   // same chunk
  EXPECT_NE(index_of_layer(1), index_of_layer(2));   // crosses cut at 2
  EXPECT_EQ(index_of_layer(2), index_of_layer(3));   // same chunk
  EXPECT_NE(index_of_layer(3), index_of_layer(4));   // crosses cut at 4
  EXPECT_EQ(index_of_layer(4), index_of_layer(5));   // same chunk
}

// Cuts are sanitized: out-of-order, duplicate, and out-of-range values are
// normalized; cut 0 and cuts beyond the last layer are no-ops.
TEST(TransformerDetectorTest, LayerCutsAreSanitized) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  auto all = BuildStack(b, /*n=*/4);  // layers 0..3

  auto selected = SelectAll(all);
  TransformerDetectorOptions options;
  // 0 ignored; duplicates collapse; 2 is the only effective cut; 99 is past end.
  options.layer_cut_indices = {2, 0, 2, 99};
  const size_t num_partitions =
      RepartitionByTransformerBlocks(selected, &subgraph, options);
  // Effective cut {2} -> buckets [0,2), [2,4) => 2 partitions (99 adds none
  // because no layer reaches it).
  EXPECT_EQ(num_partitions, 2);
}

// No cuts -> one partition per layer (unchanged baseline behavior).
TEST(TransformerDetectorTest, NoCutsIsOnePartitionPerLayer) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  auto all = BuildStack(b, /*n=*/5);

  auto selected = SelectAll(all);
  const size_t num_partitions =
      RepartitionByTransformerBlocks(selected, &subgraph);  // default options
  EXPECT_EQ(num_partitions, 5);
}

// coalesce_layers + empty cuts (the TransformerLayerCut "cut nowhere" case):
// the whole stack collapses into a single partition, NOT one per layer.
TEST(TransformerDetectorTest, CoalesceWithNoCutsIsSinglePartition) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  auto all = BuildStack(b, /*n=*/5);

  auto selected = SelectAll(all);
  TransformerDetectorOptions options;
  options.coalesce_layers = true;  // layer_cut_indices stays empty
  const size_t num_partitions =
      RepartitionByTransformerBlocks(selected, &subgraph, options);
  EXPECT_EQ(num_partitions, 1);
  EXPECT_EQ(DistinctPartitions(selected), 1);
}

// coalesce_layers with explicit cuts behaves the same as cuts alone.
TEST(TransformerDetectorTest, CoalesceWithCutsRespectsCuts) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  auto all = BuildStack(b, /*n=*/6);

  auto selected = SelectAll(all);
  TransformerDetectorOptions options;
  options.coalesce_layers = true;
  options.layer_cut_indices = {3};  // [0,3) and [3,6) => 2 partitions
  const size_t num_partitions =
      RepartitionByTransformerBlocks(selected, &subgraph, options);
  EXPECT_EQ(num_partitions, 2);
}

// The trailing logits head (final norm + vocab projection producing a non-KV
// subgraph output) is peeled into its own NON-transformer block: it does not
// count as a layer, and a cut at index N over L layers yields N-split + head.
TEST(TransformerDetectorTest, LogitsHeadIsSeparateNonTransformerPartition) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  // 4 decoder layers...
  auto all = BuildStack(b, /*n=*/4);
  // ...followed by a logits head: final RMS-norm composite + vocab projection,
  // whose output is a (non-KV) subgraph output.
  std::vector<LiteRtOp> head = {
      b.AddCompositeOp(std::string(CompositeOptions::kRmsNorm)),
      b.AddOp(kLiteRtOpCodeTflFullyConnected),  // vocab projection
  };
  b.MarkOutput("logits");
  all.insert(all.end(), head.begin(), head.end());

  auto selected = SelectAll(all);
  auto blocks = DetectTransformerBlocks(selected, &subgraph);

  // 4 attention layers (layer_index 0..3) + 1 non-transformer head block.
  size_t layers = 0, non_layers = 0;
  for (const auto& blk : blocks) {
    if (blk.layer_index >= 0) {
      ++layers;
    } else {
      ++non_layers;
    }
  }
  EXPECT_EQ(layers, 4);
  EXPECT_EQ(non_layers, 1);

  // Cut at layer 2 -> [0,2) [2,4) + head = 3 partitions.
  auto sel2 = SelectAll(all);
  TransformerDetectorOptions options;
  options.coalesce_layers = true;
  options.layer_cut_indices = {2};
  EXPECT_EQ(RepartitionByTransformerBlocks(sel2, &subgraph, options), 3);
}

// ResolveLayerCuts: per-signature spec parsing.
TEST(TransformerDetectorTest, ResolveLayerCutsBareDefault) {
  // A bare list applies to every signature.
  EXPECT_EQ(ResolveLayerCuts("16", "decode"), (std::vector<int>{16}));
  EXPECT_EQ(ResolveLayerCuts("16", "prefill_128"), (std::vector<int>{16}));
  EXPECT_EQ(ResolveLayerCuts("8,16,24", "anything"),
            (std::vector<int>{8, 16, 24}));
}

TEST(TransformerDetectorTest, ResolveLayerCutsPerSignature) {
  const char* spec = "prefill_128=16,32;decode=8";
  EXPECT_EQ(ResolveLayerCuts(spec, "prefill_128"),
            (std::vector<int>{16, 32}));
  EXPECT_EQ(ResolveLayerCuts(spec, "decode"), (std::vector<int>{8}));
  // A signature not named in the spec, with no default, gets no cuts.
  EXPECT_TRUE(ResolveLayerCuts(spec, "other").empty());
}

TEST(TransformerDetectorTest, ResolveLayerCutsDefaultPlusOverride) {
  const char* spec = "=16;decode=8";
  EXPECT_EQ(ResolveLayerCuts(spec, "decode"), (std::vector<int>{8}));
  // Empty-key group is the default for everything else.
  EXPECT_EQ(ResolveLayerCuts(spec, "prefill_128"), (std::vector<int>{16}));
}

TEST(TransformerDetectorTest, ResolveLayerCutsToleratesWhitespaceAndJunk) {
  EXPECT_EQ(ResolveLayerCuts(" prefill_128 = 16 , 32 ; decode = 8 ",
                             "prefill_128"),
            (std::vector<int>{16, 32}));
  // Non-numeric tokens are skipped.
  EXPECT_EQ(ResolveLayerCuts("decode=8,abc,16", "decode"),
            (std::vector<int>{8, 16}));
  EXPECT_TRUE(ResolveLayerCuts("", "decode").empty());
}

// A normalization expressed as a single odml.rms_norm composite op must be
// recognized as a normalization signal, the same as the decomposed form.
TEST(TransformerDetectorTest, RecognizesNormCompositeAsNormalization) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  // rms_norm composite -> FC -> gelu -> FC: a feed-forward layer whose only
  // norm signal is the composite. min_block_ops default is 4.
  std::vector<LiteRtOp> ops = {
      b.AddCompositeOp(std::string(CompositeOptions::kRmsNorm)),
      b.AddOp(kLiteRtOpCodeTflFullyConnected),
      b.AddOp(kLiteRtOpCodeTflGelu),
      b.AddOp(kLiteRtOpCodeTflFullyConnected),
  };

  auto selected = SelectAll(ops);
  auto blocks = DetectTransformerBlocks(selected, &subgraph);

  // The region is detected (matmul + activation -> feed_forward, fused with the
  // norm composite into a full block).
  ASSERT_EQ(blocks.size(), 1);
  EXPECT_EQ(blocks.front().kind, TransformerBlock::Kind::kFullBlock);
  EXPECT_EQ(blocks.front().ops.size(), ops.size());
}

// A standalone normalization composite with no matmul/activation is still
// classified as a normalization block.
TEST(TransformerDetectorTest, NormCompositeOnlyIsNormalization) {
  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  ChainBuilder b(subgraph);
  std::vector<LiteRtOp> ops = {
      b.AddCompositeOp(std::string(CompositeOptions::kRmsNorm)),
      b.AddCompositeOp(std::string(CompositeOptions::kL2Norm)),
      b.AddCompositeOp(std::string(CompositeOptions::kGroupNorm)),
      b.AddOp(kLiteRtOpCodeTflAdd),
  };

  auto selected = SelectAll(ops);
  auto blocks = DetectTransformerBlocks(selected, &subgraph);

  ASSERT_EQ(blocks.size(), 1);
  EXPECT_EQ(blocks.front().kind, TransformerBlock::Kind::kNormalization);
}

}  // namespace
}  // namespace litert::internal
