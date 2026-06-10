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

// Standalone harness that exercises the transformer-block detector against a
// real .tflite model. It loads the model, and for each subgraph simulates an
// IHV delegate that selects every op (the maximal-coverage case), then runs
// `DetectTransformerBlocks` and reports the blocks found. This mirrors what the
// partitioner does between vendor op-selection and graph slicing, without
// requiring a vendor plugin .so.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/compiler/plugin/transformer_detector.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"

ABSL_FLAG(std::string, model_path, "", "Path to the .tflite model to analyze.");
ABSL_FLAG(size_t, min_block_ops, 4,
          "Minimum ops for a region to count as a transformer block.");
ABSL_FLAG(bool, fuse_adjacent_clusters, true,
          "Fuse connected attention/ffn/norm clusters into full blocks.");
ABSL_FLAG(bool, require_softmax_for_attention, true,
          "Require a softmax for a region to classify as attention.");
ABSL_FLAG(bool, per_block, false,
          "Print one line per detected block (verbose).");
ABSL_FLAG(std::string, layer_cuts, "",
          "Per-signature layer-cut spec (e.g. --layer_cuts=16 or "
          "--layer_cuts=\"prefill_128=16,32;decode=8\"). When set, consecutive "
          "layers between cuts are coalesced into one partition. Cuts are "
          "resolved per subgraph signature.");
ABSL_FLAG(int, histogram_for_subgraph, -1,
          "If >=0, print the op-code histogram for that subgraph and exit.");
ABSL_FLAG(int, dump_io_for_subgraph, -1,
          "If >=0, print the input/output tensor names+shapes for that "
          "subgraph and exit.");
ABSL_FLAG(int, trace_ops_for_subgraph, -1,
          "If >=0, print the op sequence (marking softmax and kv-cache reads) "
          "for that subgraph and exit.");
ABSL_FLAG(int, trace_ops_limit, 60, "Max ops to print in --trace_ops mode.");
ABSL_FLAG(int, trace_ops_from, 0,
          "Start index for --trace_ops mode; when >0, prints every op in "
          "[from, limit).");

namespace litert::internal {
namespace {

std::string ShapeStr(const LiteRtTensorT& t) {
  auto ranked = t.Ranked();
  if (!ranked) {
    return "<unranked>";
  }
  std::string s = "[";
  for (int i = 0; i < ranked->layout.rank; ++i) {
    if (i) s += ",";
    s += std::to_string(ranked->layout.dimensions[i]);
  }
  s += "]";
  return s;
}

const char* ScopeName(TransformerBlock::AttentionScope scope) {
  switch (scope) {
    case TransformerBlock::AttentionScope::kLocal:
      return "local";
    case TransformerBlock::AttentionScope::kGlobal:
      return "global";
    case TransformerBlock::AttentionScope::kUnknown:
      return "-";
  }
  return "-";
}

const char* KindName(TransformerBlock::Kind kind) {
  switch (kind) {
    case TransformerBlock::Kind::kAttention:
      return "attention";
    case TransformerBlock::Kind::kFeedForward:
      return "feed_forward";
    case TransformerBlock::Kind::kNormalization:
      return "normalization";
    case TransformerBlock::Kind::kFullBlock:
      return "full_block";
  }
  return "unknown";
}

// Simulates a permissive IHV delegate: select every op in the subgraph, all
// tagged with partition index 0 (as a real plugin's Partition() might before
// any structural grouping).
std::vector<LiteRtOpWithPartitionIndex> SelectAllOps(LiteRtSubgraph subgraph) {
  std::vector<LiteRtOpWithPartitionIndex> selected;
  selected.reserve(subgraph->Ops().size());
  for (auto* op : subgraph->Ops()) {
    selected.push_back({op, 0});
  }
  return selected;
}

int Run(const std::string& model_path,
        const TransformerDetectorOptions& options, bool per_block,
        int histogram_for_subgraph, const std::string& layer_cut_spec) {
  auto model = LoadModelFromFile(model_path);
  if (!model) {
    ABSL_LOG(ERROR) << "Failed to load model '" << model_path
                    << "': " << model.Error().Message();
    return 1;
  }

  auto& m = **model;
  absl::PrintF("Model: %s\n", model_path);
  absl::PrintF("Subgraphs: %lu\n\n", m.NumSubgraphs());

  // Map each subgraph to its signature key for per-signature cut resolution.
  absl::flat_hash_map<LiteRtSubgraph, std::string> subgraph_to_signature;
  for (auto* sig : m.Signatures()) {
    subgraph_to_signature[&sig->GetSubgraph()] = std::string(sig->Key());
  }

  const int dump_io = absl::GetFlag(FLAGS_dump_io_for_subgraph);
  if (dump_io >= 0) {
    if (static_cast<size_t>(dump_io) >= m.NumSubgraphs()) {
      ABSL_LOG(ERROR) << "subgraph index out of range";
      return 1;
    }
    auto* sg = m.Subgraphs()[dump_io];
    absl::PrintF("Subgraph[%d] inputs (%lu):\n", dump_io, sg->NumInputs());
    for (size_t i = 0; i < sg->NumInputs(); ++i) {
      const auto& t = sg->Input(i);
      absl::PrintF("  in[%lu] %-40s %s\n", i, std::string(t.Name()),
                   ShapeStr(t));
    }
    absl::PrintF("Subgraph[%d] outputs (%lu):\n", dump_io, sg->NumOutputs());
    for (size_t i = 0; i < sg->NumOutputs(); ++i) {
      const auto& t = sg->Output(i);
      absl::PrintF("  out[%lu] %-40s %s\n", i, std::string(t.Name()),
                   ShapeStr(t));
    }
    return 0;
  }

  const int trace_ops = absl::GetFlag(FLAGS_trace_ops_for_subgraph);
  if (trace_ops >= 0) {
    if (static_cast<size_t>(trace_ops) >= m.NumSubgraphs()) {
      ABSL_LOG(ERROR) << "subgraph index out of range";
      return 1;
    }
    auto* sg = m.Subgraphs()[trace_ops];
    const int limit = absl::GetFlag(FLAGS_trace_ops_limit);
    const int trace_from = absl::GetFlag(FLAGS_trace_ops_from);
    // Tensors that are subgraph outputs (for spotting the logits head).
    absl::flat_hash_set<LiteRtTensor> sg_outputs(sg->Outputs().begin(),
                                                 sg->Outputs().end());
    int n = 0;
    for (auto* op : sg->Ops()) {
      const int idx = n++;
      if (idx < trace_from) continue;
      if (idx >= limit) break;
      std::string reads;
      for (auto* in : op->Inputs()) {
        const auto nm = in->Name();
        if (nm.find("kv_cache") != absl::string_view::npos ||
            nm.find("mask") != absl::string_view::npos ||
            nm.find("pos_emb") != absl::string_view::npos) {
          reads += " READS:" + std::string(nm);
        }
      }
      bool produces_output = false;
      for (auto* out : op->Outputs()) {
        if (sg_outputs.contains(out)) produces_output = true;
      }
      const bool is_sm = op->OpCode() == kLiteRtOpCodeTflSoftmax;
      // In --trace_ops_from mode print every op; otherwise only landmarks.
      if (trace_from > 0 || !reads.empty() || is_sm || produces_output) {
        absl::PrintF("  [%4d] opcode %-4d %s%s%s\n", idx,
                     static_cast<int>(op->OpCode()), is_sm ? "<SOFTMAX>" : "",
                     produces_output ? " <SG_OUTPUT>" : "", reads);
      }
    }
    return 0;
  }

  if (histogram_for_subgraph >= 0) {
    if (static_cast<size_t>(histogram_for_subgraph) >= m.NumSubgraphs()) {
      ABSL_LOG(ERROR) << "subgraph index out of range";
      return 1;
    }
    auto* sg = m.Subgraphs()[histogram_for_subgraph];
    std::map<int, size_t> by_code;
    for (auto* op : sg->Ops()) {
      by_code[static_cast<int>(op->OpCode())]++;
    }
    absl::PrintF("Subgraph[%d] op-code histogram (%lu ops):\n",
                 histogram_for_subgraph, sg->Ops().size());
    for (const auto& [code, count] : by_code) {
      absl::PrintF("  opcode %-4d  x%lu\n", code, count);
    }
    return 0;
  }

  // Mirror PartitionModel: composite decomposition subgraphs are not passed to
  // the partitioner (their bodies are inlined or compiled via the composite op),
  // so they must not be analyzed as standalone graphs here.
  absl::flat_hash_set<int> decomp_subgraphs;
  for (size_t i = 0; i < m.NumSubgraphs(); ++i) {
    for (auto* op : m.Subgraphs()[i]->Ops()) {
      auto info = GetOptionsAs<CompositeOptions>(op);
      if (info) {
        decomp_subgraphs.insert(info->subgraph);
      }
    }
  }
  absl::PrintF("Composite decomposition subgraphs (skipped): %lu\n\n",
               decomp_subgraphs.size());

  size_t total_blocks = 0;
  size_t total_ops_in_blocks = 0;
  size_t total_ops = 0;
  size_t local_layers = 0;
  size_t global_layers = 0;
  size_t shared_kv_layers = 0;
  std::map<std::string, size_t> kind_counts;

  for (size_t i = 0; i < m.NumSubgraphs(); ++i) {
    if (decomp_subgraphs.contains(static_cast<int>(i))) {
      continue;
    }
    auto* subgraph = m.Subgraphs()[i];
    const size_t num_ops = subgraph->Ops().size();
    total_ops += num_ops;

    // Resolve this subgraph's layer cuts from the per-signature spec.
    TransformerDetectorOptions sg_options = options;
    std::string signature_key;
    if (auto it = subgraph_to_signature.find(subgraph);
        it != subgraph_to_signature.end()) {
      signature_key = it->second;
    }
    if (!layer_cut_spec.empty()) {
      // Mirror the TransformerLayerCut strategy: graphs without a cut coalesce
      // into a single partition rather than one-per-layer.
      sg_options.coalesce_layers = true;
      sg_options.layer_cut_indices =
          ResolveLayerCuts(layer_cut_spec, signature_key);
    }

    auto selected = SelectAllOps(subgraph);
    auto blocks = DetectTransformerBlocks(selected, subgraph, sg_options);

    size_t ops_in_blocks = 0;
    size_t transformer_layers = 0;  // blocks that are real transformer layers
    size_t non_transformer_blocks = 0;
    for (const auto& block : blocks) {
      ops_in_blocks += block.ops.size();
      kind_counts[KindName(block.kind)]++;
      if (block.layer_index >= 0) {
        transformer_layers++;
      } else {
        non_transformer_blocks++;
      }
      if (block.attention_scope == TransformerBlock::AttentionScope::kLocal) {
        local_layers++;
      } else if (block.attention_scope ==
                 TransformerBlock::AttentionScope::kGlobal) {
        global_layers++;
      }
      if (block.shares_kv_cache_with >= 0) {
        shared_kv_layers++;
      }
      if (per_block && block.layer_index >= 0) {
        std::string shares =
            block.shares_kv_cache_with >= 0
                ? absl::StrFormat(", shares KV with layer %d",
                                  block.shares_kv_cache_with)
                : "";
        std::string kv;
        for (const auto& n : block.kv_cache_tensors) {
          kv += " " + n;
        }
        absl::PrintF("  sg[%lu] layer %2d  %-11s  %-6s  KV{%s }%s\n", i,
                     block.layer_index, KindName(block.kind),
                     ScopeName(block.attention_scope), kv, shares);
      }
    }

    total_blocks += blocks.size();
    total_ops_in_blocks += ops_in_blocks;

    // Run the actual repartition path (which applies any layer cuts) to report
    // the number of partitions the partitioner would hand to the delegate.
    size_t num_partitions = blocks.size();
    if (sg_options.coalesce_layers || !sg_options.layer_cut_indices.empty()) {
      auto selected_copy = SelectAllOps(subgraph);
      num_partitions =
          RepartitionByTransformerBlocks(selected_copy, subgraph, sg_options);
    }

    if (!blocks.empty()) {
      absl::PrintF(
          "  subgraph[%lu] (sig='%s'): %lu ops -> %lu transformer layer(s) "
          "(+%lu non-transformer block(s)) -> %lu partition(s), %lu ops "
          "covered (%.1f%%)\n",
          i, signature_key, num_ops, transformer_layers, non_transformer_blocks,
          num_partitions,
          ops_in_blocks, num_ops ? 100.0 * ops_in_blocks / num_ops : 0.0);
    }
  }

  absl::PrintF("\n==== Summary ====\n");
  absl::PrintF("Total ops:               %lu\n", total_ops);
  absl::PrintF("Total blocks detected:   %lu\n", total_blocks);
  absl::PrintF("Ops covered by blocks:   %lu (%.1f%%)\n", total_ops_in_blocks,
               total_ops ? 100.0 * total_ops_in_blocks / total_ops : 0.0);
  absl::PrintF("Local attention layers:  %lu\n", local_layers);
  absl::PrintF("Global attention layers: %lu\n", global_layers);
  absl::PrintF("Layers sharing KV cache: %lu\n", shared_kv_layers);
  absl::PrintF("Blocks by kind:\n");
  for (const auto& [kind, count] : kind_counts) {
    absl::PrintF("  %-14s %lu\n", kind, count);
  }
  return 0;
}

}  // namespace
}  // namespace litert::internal

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  const auto model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    ABSL_LOG(ERROR) << "--model_path is required.";
    return 1;
  }

  litert::internal::TransformerDetectorOptions options;
  options.min_block_ops = absl::GetFlag(FLAGS_min_block_ops);
  options.fuse_adjacent_clusters = absl::GetFlag(FLAGS_fuse_adjacent_clusters);
  options.require_softmax_for_attention =
      absl::GetFlag(FLAGS_require_softmax_for_attention);

  return litert::internal::Run(model_path, options,
                               absl::GetFlag(FLAGS_per_block),
                               absl::GetFlag(FLAGS_histogram_for_subgraph),
                               absl::GetFlag(FLAGS_layer_cuts));
}
