// Copyright 2024 Google LLC.
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

#include "litert/compiler/plugin/algo.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/insert_order_map.h"
#include "litert/core/model/model.h"

namespace litert::internal {
namespace {

//
// flatlist to partition(s)
//===----------------------------------------------------------------------===//

class DisjointSets {
 public:
  static std::vector<std::vector<LiteRtOp>> GetPartitionsFromFlatList(
      const std::vector<LiteRtOpWithPartitionIndex>& flat_op_list);

 private:
  void Insert(LiteRtOp op, LiteRtOp parent);
  std::vector<std::vector<LiteRtOp>> GetBuckets();
  LiteRtOp GetBucket(LiteRtOp op);
  InsertOrderMap<LiteRtOp, LiteRtOp> map_;
};

//===----------------------------------------------------------------------===//
// LiteRt Core union-find algorithm.
//
// This algorithm is used to group partitions into sub DAGs.
// The input to the algorithm is a list of ops with the their partition index.
//
//        [ (op_0, 0),
//          (op_1, 0),
//          (op_2, 0),
//            ...
//          (op_7, 1),
//          (op_8, 1), ...]
//
// Union-find algorithm is run on each partition (list of ops with same
// partition index).
//
// For each partition, the input to the union find algorithm is a list of
// ops with the same partition index. For example,
//
//        [ op_0, op_1, op_2, op3, op_4, op_5 ...]
//
// The output of the union find algorithm is a list of list of ops, where each
// list is a disjoint set(a sub DAG within the original Subgraph). For
// example,
//
//        [ [op_0, op_1, op_6],
//          [op_2, op_3],
//          [op_4, op_5] ... ]
//
//  Similarly, algorithm on the next parition would return something like
//
//        [ [op_7, op_8, op_9],
//          [op_10, op_11],
//          [op_12, op_13] ... ]
//
// We aggregate all disjoint sets into the result buckets. For example,
//
//        [ [op_0, op_1, op_6]
//          [op_2, op_3] ,
//          [op_4, op_5],
//          [op_7, op_8, op_9],
//          [op_10, op_11],
//          [op_12, op_13] ... ]
//===----------------------------------------------------------------------===//
std::vector<std::vector<LiteRtOp>> DisjointSets::GetPartitionsFromFlatList(
    const std::vector<LiteRtOpWithPartitionIndex>& flat_op_list) {
  // Find all unique partition indices. Use unique partition index as key and
  // store the ops for each partition index as value of the map.
  absl::flat_hash_map<LiteRtParamIndex, std::vector<LiteRtOp>> partition_map;
  for (int i = 0; i < flat_op_list.size(); ++i) {
    partition_map[flat_op_list[i].second].push_back(flat_op_list[i].first);
  }

  // A vector of disjoint sets, where each partition contains op with the same
  // partition index.
  std::vector<DisjointSets> partitions;

  // A vector of all unique partition indices for iterative access. We kept this
  // vector so vendor plugin returned partition indices does not have to be
  // zero-based.
  std::vector<LiteRtParamIndex> flat_partition_indices;
  for (auto& partition_index : partition_map) {
    flat_partition_indices.push_back(partition_index.first);
  }

  // Resize the partitions vector to the number of unique partition indices.
  partitions.resize(flat_partition_indices.size());

  // Resulting buckets of the union find algorithm.
  std::vector<std::vector<LiteRtOp>> all_buckets;

  // Run union-find algorithm on each partition.
  for (int i = 0; i < flat_partition_indices.size(); ++i) {
    // For each partition, initialize the disjoint sets.
    for (auto* op : partition_map[flat_partition_indices[i]]) {
      partitions[i].map_.InsertOrAssign(op, op);
    }
    // For each partition, find all disjoint sets.
    for (auto* op : partition_map[flat_partition_indices[i]]) {
      for (auto* output : op->Outputs()) {
        for (auto* user : output->Users()) {
          if (!partitions[i].map_.Contains(user)) {
            continue;
          }
          partitions[i].Insert(op, user);
        }
      }
    }
    // Aggregate all disjoint sets into the result buckets.
    for (auto& bucket : partitions[i].GetBuckets()) {
      all_buckets.push_back(std::move(bucket));
    }
  }
  return all_buckets;
}

void DisjointSets::Insert(LiteRtOp op, LiteRtOp parent) {
  auto* parent_bucket = GetBucket(parent);
  auto* op_bucket = GetBucket(op);
  if (op_bucket == parent_bucket) {
    return;
  }
  map_.InsertOrAssign(op_bucket, parent_bucket);
}

// Get all disjoint sets.
std::vector<std::vector<LiteRtOp>> DisjointSets::GetBuckets() {
  // NOLINTBEGIN
  std::unordered_map<LiteRtOp, std::vector<LiteRtOp>> invert_map;
  // NOLINTEND
  for (auto it = map_.Begin(); it != map_.End(); ++it) {
    auto* bucket = GetBucket(it->first);

    if (invert_map.find(bucket) == invert_map.end()) {
      invert_map.insert_or_assign(bucket, std::vector<LiteRtOp>{});
    }

    invert_map[bucket].push_back(it->first);
  }

  std::vector<std::vector<LiteRtOp>> res;
  res.reserve(invert_map.size());

  for (auto& entry : invert_map) {
    res.push_back(std::move(entry.second));
  }

  return res;
}

// Gets the pointer which serves as the key for given ops bucket. Collapses
// paths to amortize.
LiteRtOp DisjointSets::GetBucket(LiteRtOp op) {
  auto it = map_.Find(op);
  auto* parent = it->get().second;
  if (op != parent) {
    parent = GetBucket(parent);
    map_.InsertOrAssign(op, parent);
  }
  return parent;
}

// Performs a topological sort on the given subgraph using Kahn's algorithm.
bool TopologicalSort(LiteRtSubgraphT& subgraph) {
  absl::flat_hash_map<const LiteRtOpT*, int> in_degree;
  absl::flat_hash_set<const LiteRtOpT*> subgraph_ops;
  for (auto* op : subgraph.Ops()) {
    subgraph_ops.insert(op);
  }

  for (auto* op : subgraph.Ops()) {
    int deg = 0;
    for (auto* input : op->Inputs()) {
      if (input && input->DefiningOp()) {
        if (subgraph_ops.contains(input->DefiningOp())) {
          deg++;
        }
      }
    }
    in_degree[op] = deg;
  }

  std::vector<LiteRtOp> ready;
  for (auto* op : subgraph.Ops()) {
    if (in_degree[op] == 0) {
      ready.push_back(op);
    }
  }

  std::vector<LiteRtOp> sorted;
  sorted.reserve(subgraph.Ops().size());

  size_t head = 0;
  while (head < ready.size()) {
    auto* op = ready[head++];
    sorted.push_back(op);

    for (auto* output : op->Outputs()) {
      for (auto* user : output->Users()) {
        if (subgraph_ops.contains(user)) {
          in_degree[user]--;
          if (in_degree[user] == 0) {
            ready.push_back(user);
          }
        }
      }
    }
  }

  if (sorted.size() != subgraph.Ops().size()) {
    return false;
  }

  subgraph.ReorderOps(sorted);
  return true;
}

//
// slice partitions out of a subgraph (into new subgraphs)
//===----------------------------------------------------------------------===//

class GraphSlicer {
 public:
  // Slices "partitions" from "root" into the empty subgraph "slice". Assumes
  // the partition is a valid sub-DAG, and replaces it with a single
  // tfl.custom_op in "root". A reference to that op is returned.
  static litert::Expected<LiteRtOp> SlicePartitionFromGraph(
      LiteRtSubgraphT& root, LiteRtSubgraph slice,
      std::vector<LiteRtOp>& partition);

 private:
  explicit GraphSlicer(LiteRtSubgraph slice) : slice_(slice) {}

  litert::Expected<void> CloneInto(const LiteRtOpT& op);

  void RerouteTensorsThroughCustomOp(const LiteRtSubgraphT& root);

  LiteRtSubgraph slice_;
  // Maps tensor in old subgraph to tensor in new subgraph.
  InsertOrderMap<LiteRtTensor, LiteRtTensor> tensor_map_;
  LiteRtOp dispatch_op_ = nullptr;
};

//===----------------------------------------------------------------------===//
// GraphSlicer Implementation
//===----------------------------------------------------------------------===//

litert::Expected<LiteRtOp> GraphSlicer::SlicePartitionFromGraph(
    LiteRtSubgraphT& root, LiteRtSubgraph slice,
    std::vector<LiteRtOp>& partition) {
  GraphSlicer slicer(slice);

  // Find all inputs used by partition.
  absl::flat_hash_set<const LiteRtTensorT*> used_tensors;
  for (auto* op : partition) {
    used_tensors.insert(op->Inputs().cbegin(), op->Inputs().cend());
  }
  for (auto* old_input : root.Inputs()) {
    if (used_tensors.contains(old_input)) {
      LITERT_ASSIGN_OR_RETURN(auto* new_input,
                              MakeClone(*slicer.slice_, *old_input));
      slicer.slice_->Inputs().push_back(new_input);
      slicer.tensor_map_.InsertOrAssign(old_input, new_input);
    }
  }

  for (auto* op : partition) {
    LITERT_RETURN_IF_ERROR(slicer.CloneInto(*op));
  }

  for (auto* op : partition) {
    Drop(*op);
  }

  // Reuse the storage from the last op in partition.
  // The actual topological order will be fixed at the end.
  slicer.dispatch_op_ = partition.back();

  ABSL_DCHECK(slicer.dispatch_op_->Inputs().empty());
  ABSL_DCHECK(slicer.dispatch_op_->Outputs().empty());
  MakeDispatchOp(*slicer.dispatch_op_);
  slicer.RerouteTensorsThroughCustomOp(root);

  DCE(root);

  if (!TopologicalSort(root)) {
    LITERT_LOG(LITERT_WARNING,
               "Failed to topologically sort subgraph after slicing!");
  }

  return slicer.dispatch_op_;
}

void GraphSlicer::RerouteTensorsThroughCustomOp(const LiteRtSubgraphT& root) {
  for (auto it = tensor_map_.Begin(); it != tensor_map_.End(); ++it) {
    auto* old_tensor = it->first;
    auto* new_tensor = it->second;

    // Reroute tensors which need to be passed into the scope of the new
    // subgraph to inputs of the custom op.
    if (new_tensor->DefiningOp() == nullptr && !IsConstant(*new_tensor)) {
      AttachInput(old_tensor, *dispatch_op_);
      continue;
    }

    // Reroute custom op as the definer of tensors within the removed partition
    // and referenced later in the root graph.
    if ((!old_tensor->Users().empty() && !IsConstant(*old_tensor)) ||
        FindOutput(root, *old_tensor)) {
      AttachOutput(old_tensor, *dispatch_op_);
      slice_->Outputs().push_back(new_tensor);
    }
  }
}

litert::Expected<void> GraphSlicer::CloneInto(const LiteRtOpT& old_op) {
  auto& new_op = MakeClone(*slice_, old_op);

  for (auto i = 0; i < old_op.NumInputs(); ++i) {
    auto* old_input = old_op.Inputs().at(i);
    LiteRtTensor new_input;
    if (tensor_map_.Contains(old_input)) {
      // If old_input is already in the map then map[input] is its cloned
      // counterpart in the new graph.
      auto it = tensor_map_.Find(old_input);
      new_input = it->get().second;
    } else {
      // Otherwise, it must be a new subgraph input (or constant).
      LITERT_ASSIGN_OR_RETURN(new_input, MakeClone(*slice_, *old_input));
      if (!IsConstant(*new_input)) {
        slice_->Inputs().push_back(new_input);
      }

      tensor_map_.InsertOrAssign(old_input, new_input);
    }

    AttachInput(new_input, new_op);
  }

  for (int i = 0; i < old_op.NumOutputs(); ++i) {
    auto* old_output = old_op.Outputs().at(i);
    LITERT_ASSIGN_OR_RETURN(auto* new_output, MakeClone(*slice_, *old_output));
    AttachOutput(new_output, new_op);

    // Update the values defined in scope of the new subgraph.
    tensor_map_.InsertOrAssign(old_output, new_output);
  }

  return {};
}

// Maximum quotient-DAG node count for Phase 2 bitset contraction.
constexpr int kMaxContractionNodes = 8192;
constexpr size_t kMaxCandidatePairs = 500000;

// Compact bitset for transitive reachability in the quotient DAG.
class BitSet {
 public:
  explicit BitSet(int num_bits = 0)
      : num_bits_(num_bits),
        words_(static_cast<size_t>(std::max(0, num_bits) + 63) / 64,
               uint64_t{0}) {}

  void Set(int bit) {
    ABSL_DCHECK_GE(bit, 0);
    ABSL_DCHECK_LT(bit, num_bits_);
    const size_t idx = static_cast<size_t>(bit);
    words_[idx >> 6] |= (uint64_t{1} << (idx & 63));
  }

  void Clear(int bit) {
    ABSL_DCHECK_GE(bit, 0);
    ABSL_DCHECK_LT(bit, num_bits_);
    const size_t idx = static_cast<size_t>(bit);
    words_[idx >> 6] &= ~(uint64_t{1} << (idx & 63));
  }

  void OrWith(const BitSet& other) {
    ABSL_DCHECK_EQ(words_.size(), other.words_.size());
    for (size_t i = 0; i < words_.size(); ++i) {
      words_[i] |= other.words_[i];
    }
  }

  bool Intersects(const BitSet& other) const {
    ABSL_DCHECK_EQ(words_.size(), other.words_.size());
    for (size_t i = 0; i < words_.size(); ++i) {
      if ((words_[i] & other.words_[i]) != 0) {
        return true;
      }
    }
    return false;
  }

  template <typename Fn>
  void ForEachSetBit(Fn&& fn) const {
    for (size_t word_idx = 0; word_idx < words_.size(); ++word_idx) {
      uint64_t word = words_[word_idx];
      while (word != 0) {
        const int bit = absl::countr_zero(word);
        fn(static_cast<int>(word_idx * 64) + bit);
        word &= (word - 1);
      }
    }
  }

 private:
  int num_bits_ = 0;
  std::vector<uint64_t> words_;
};

struct SubgraphDag {
  int num_ops = 0;
  std::vector<LiteRtOp> id_to_op;
  std::vector<std::vector<int>> succs;
  std::vector<std::vector<int>> preds;
  std::vector<int> topo_order;
  std::vector<int> topo_pos;
  std::vector<bool> is_selected;
  std::vector<LiteRtParamIndex> partition_index;
  int num_unique_indices = 0;
};

struct EpochBlock {
  int epoch = 0;
  LiteRtParamIndex partition_index = 0;
  int first_topo_pos = 0;
  std::vector<int> op_ids;
};

// Builds a deduplicated graph and Kahn topological order for `subgraph`.
// Any ops lying on or downstream of a cycle in malformed IR retain
// `topo_pos[i] == -1` and are excluded from `topo_order`.
void BuildSubgraphDag(const std::vector<LiteRtOpWithPartitionIndex>& ops,
                      LiteRtSubgraph subgraph, SubgraphDag& dag) {
  const auto& subgraph_ops = subgraph->Ops();
  dag.id_to_op.reserve(subgraph_ops.size());
  absl::flat_hash_map<const LiteRtOpT*, int> op_to_id;
  op_to_id.reserve(subgraph_ops.size());
  for (LiteRtOp op : subgraph_ops) {
    if (op == nullptr) {
      continue;
    }
    auto [it, inserted] =
        op_to_id.emplace(op, static_cast<int>(dag.id_to_op.size()));
    if (inserted) {
      dag.id_to_op.push_back(op);
    }
  }
  dag.num_ops = static_cast<int>(dag.id_to_op.size());
  if (dag.num_ops == 0) {
    return;
  }

  dag.succs.resize(dag.num_ops);
  dag.preds.resize(dag.num_ops);
  dag.is_selected.assign(dag.num_ops, false);
  dag.partition_index.assign(dag.num_ops, 0);

  for (const auto& [op, part_idx] : ops) {
    auto it = op_to_id.find(op);
    if (it == op_to_id.end()) {
      continue;
    }
    dag.is_selected[it->second] = true;
    dag.partition_index[it->second] = part_idx;
  }

  absl::flat_hash_set<LiteRtParamIndex> unique_indices;
  for (int i = 0; i < dag.num_ops; ++i) {
    if (dag.is_selected[i]) {
      unique_indices.insert(dag.partition_index[i]);
    }
  }
  dag.num_unique_indices = static_cast<int>(unique_indices.size());
  if (dag.num_unique_indices == 0) {
    return;
  }

  for (int u = 0; u < dag.num_ops; ++u) {
    for (const auto* output : dag.id_to_op[u]->Outputs()) {
      if (output == nullptr) {
        continue;
      }
      for (const auto* user : output->Users()) {
        auto it = op_to_id.find(user);
        if (it != op_to_id.end()) {
          dag.succs[u].push_back(it->second);
        }
      }
    }
  }
  for (int v = 0; v < dag.num_ops; ++v) {
    for (const auto* input : dag.id_to_op[v]->Inputs()) {
      if (input == nullptr || input->DefiningOp() == nullptr) {
        continue;
      }
      auto it = op_to_id.find(input->DefiningOp());
      if (it != op_to_id.end()) {
        dag.succs[it->second].push_back(v);
      }
    }
  }

  std::vector<int> last_seen(dag.num_ops, -1);
  for (int u = 0; u < dag.num_ops; ++u) {
    size_t write = 0;
    for (int v : dag.succs[u]) {
      if (last_seen[v] == u) {
        continue;
      }
      last_seen[v] = u;
      dag.succs[u][write++] = v;
      dag.preds[v].push_back(u);
    }
    dag.succs[u].resize(write);
  }

  std::vector<int> in_degree(dag.num_ops, 0);
  std::vector<int> ready;
  ready.reserve(dag.num_ops);
  for (int i = 0; i < dag.num_ops; ++i) {
    in_degree[i] = static_cast<int>(dag.preds[i].size());
    if (in_degree[i] == 0) {
      ready.push_back(i);
    }
  }

  dag.topo_order.reserve(dag.num_ops);
  dag.topo_pos.assign(dag.num_ops, -1);
  size_t head = 0;
  while (head < ready.size()) {
    const int u = ready[head++];
    dag.topo_pos[u] = static_cast<int>(dag.topo_order.size());
    dag.topo_order.push_back(u);
    for (int v : dag.succs[u]) {
      if (--in_degree[v] == 0) {
        ready.push_back(v);
      }
    }
  }
}

// Computes alternating-label epoch depth along `dag.topo_order`.
// Unselected ops without selected ancestors receive -1 so they can execute
// before epoch 0 without introducing a partition boundary.
std::vector<int> ComputeEpochs(const SubgraphDag& dag) {
  std::vector<int> epoch(dag.num_ops, 0);
  for (int node : dag.topo_order) {
    if (!dag.is_selected[node]) {
      int max_epoch = -1;
      for (int pred : dag.preds[node]) {
        max_epoch = std::max(max_epoch, epoch[pred]);
      }
      epoch[node] = max_epoch;
      continue;
    }
    int max_epoch = 0;
    const LiteRtParamIndex node_index = dag.partition_index[node];
    for (int pred : dag.preds[node]) {
      if (dag.is_selected[pred] && dag.partition_index[pred] == node_index) {
        max_epoch = std::max(max_epoch, epoch[pred]);
      } else if (epoch[pred] >= 0) {
        max_epoch = std::max(max_epoch, epoch[pred] + 1);
      }
    }
    epoch[node] = max_epoch;
  }
  return epoch;
}

// Groups selected ops by (epoch, partition_index) in topological order.
std::vector<EpochBlock> BuildEpochBlocks(const SubgraphDag& dag,
                                         const std::vector<int>& epoch) {
  absl::flat_hash_map<std::pair<int, LiteRtParamIndex>, int> block_map;
  std::vector<EpochBlock> blocks;
  for (int node : dag.topo_order) {
    if (!dag.is_selected[node]) {
      continue;
    }
    const std::pair<int, LiteRtParamIndex> key{epoch[node],
                                               dag.partition_index[node]};
    auto [it, inserted] = block_map.emplace(key, blocks.size());
    if (inserted) {
      EpochBlock block;
      block.epoch = key.first;
      block.partition_index = key.second;
      block.first_topo_pos = dag.topo_pos[node];
      blocks.push_back(std::move(block));
    }
    blocks[it->second].op_ids.push_back(node);
  }

  std::sort(blocks.begin(), blocks.end(),
            [](const EpochBlock& a, const EpochBlock& b) {
              if (a.epoch != b.epoch) {
                return a.epoch < b.epoch;
              }
              if (a.first_topo_pos != b.first_topo_pos) {
                return a.first_topo_pos < b.first_topo_pos;
              }
              return a.partition_index < b.partition_index;
            });
  return blocks;
}

// Iteratively merges same-index epoch blocks in the quotient DAG when doing so
// preserves acyclicity (no intermediate path between the merged blocks).
void ContractMultiIndexQuotientDag(const SubgraphDag& dag,
                                   const std::vector<int>& epoch,
                                   std::vector<EpochBlock>& blocks) {
  const int num_blocks = static_cast<int>(blocks.size());
  if (num_blocks <= 1) {
    return;
  }

  // Only unselected ops that lie between selected ops (have both a selected
  // ancestor and a selected descendant) can witness a cycle.
  std::vector<bool> has_selected_desc(dag.num_ops, false);
  for (int i = static_cast<int>(dag.topo_order.size()) - 1; i >= 0; --i) {
    const int node = dag.topo_order[i];
    if (dag.is_selected[node]) {
      has_selected_desc[node] = true;
    }
    if (!has_selected_desc[node]) {
      continue;
    }
    for (int pred : dag.preds[node]) {
      has_selected_desc[pred] = true;
    }
  }

  std::vector<int> op_to_q(dag.num_ops, -1);
  for (int b = 0; b < num_blocks; ++b) {
    for (int op_id : blocks[b].op_ids) {
      op_to_q[op_id] = b;
    }
  }
  std::vector<int> single_op_nodes;
  for (int node : dag.topo_order) {
    if (!dag.is_selected[node] && epoch[node] >= 0 && has_selected_desc[node]) {
      op_to_q[node] = num_blocks + static_cast<int>(single_op_nodes.size());
      single_op_nodes.push_back(node);
    }
  }
  const int num_q = num_blocks + static_cast<int>(single_op_nodes.size());
  if (num_q > kMaxContractionNodes) {
    LITERT_LOG(LITERT_WARNING,
               "Quotient DAG has %d nodes (limit %d); skipping Phase 2 "
               "multi-index contraction.",
               num_q, kMaxContractionNodes);
    return;
  }

  std::vector<std::vector<int>> q_succs(num_q);
  std::vector<int> q_in_degree(num_q, 0);
  std::vector<int> last_seen(num_q, -1);
  // Process each quotient node's constituent ops contiguously so `last_seen`
  // accurately deduplicates edges even when ops are interleaved in `subgraph`.
  auto add_q_edges_from_op = [&](int qu, int u) {
    for (int v : dag.succs[u]) {
      const int qv = op_to_q[v];
      if (qv < 0 || qu == qv || last_seen[qv] == qu) {
        continue;
      }
      last_seen[qv] = qu;
      q_succs[qu].push_back(qv);
      q_in_degree[qv]++;
    }
  };
  for (int b = 0; b < num_blocks; ++b) {
    for (int u : blocks[b].op_ids) {
      add_q_edges_from_op(b, u);
    }
  }
  for (size_t i = 0; i < single_op_nodes.size(); ++i) {
    add_q_edges_from_op(num_blocks + static_cast<int>(i), single_op_nodes[i]);
  }

  std::vector<int> q_topo;
  q_topo.reserve(num_q);
  std::vector<int> ready;
  ready.reserve(num_q);
  for (int q = 0; q < num_q; ++q) {
    if (q_in_degree[q] == 0) {
      ready.push_back(q);
    }
  }
  size_t head = 0;
  while (head < ready.size()) {
    const int q = ready[head++];
    q_topo.push_back(q);
    for (int v : q_succs[q]) {
      if (--q_in_degree[v] == 0) {
        ready.push_back(v);
      }
    }
  }
  ABSL_DCHECK_EQ(static_cast<int>(q_topo.size()), num_q);
  if (static_cast<int>(q_topo.size()) != num_q) {
    return;
  }

  std::vector<BitSet> reach(num_q, BitSet(num_q));
  std::vector<BitSet> anc(num_q, BitSet(num_q));
  for (int i = num_q - 1; i >= 0; --i) {
    const int q = q_topo[i];
    for (int v : q_succs[q]) {
      reach[q].OrWith(reach[v]);
      reach[q].Set(v);
    }
  }
  for (int q : q_topo) {
    for (int v : q_succs[q]) {
      anc[v].OrWith(anc[q]);
      anc[v].Set(q);
    }
  }

  absl::flat_hash_map<LiteRtParamIndex, std::vector<int>> blocks_by_index;
  for (int b = 0; b < num_blocks; ++b) {
    blocks_by_index[blocks[b].partition_index].push_back(b);
  }
  std::vector<LiteRtParamIndex> sorted_indices;
  sorted_indices.reserve(blocks_by_index.size());
  for (const auto& [part_idx, list] : blocks_by_index) {
    sorted_indices.push_back(part_idx);
  }
  std::sort(sorted_indices.begin(), sorted_indices.end());

  std::vector<bool> alive(num_q, true);
  std::vector<int> rep(num_blocks);
  for (int b = 0; b < num_blocks; ++b) {
    rep[b] = b;
  }
  auto find_rep = [&](int x) {
    while (rep[x] != x) {
      rep[x] = rep[rep[x]];
      x = rep[x];
    }
    return x;
  };

  auto try_merge = [&](int raw_a, int raw_b) {
    int a = find_rep(raw_a);
    int b = find_rep(raw_b);
    if (a == b) {
      return;
    }
    if (blocks[b].first_topo_pos < blocks[a].first_topo_pos) {
      std::swap(a, b);
    }
    if (reach[a].Intersects(anc[b]) || reach[b].Intersects(anc[a])) {
      return;
    }

    BitSet merged_reach = reach[a];
    merged_reach.OrWith(reach[b]);
    merged_reach.Clear(a);
    merged_reach.Clear(b);

    BitSet merged_anc = anc[a];
    merged_anc.OrWith(anc[b]);
    merged_anc.Clear(a);
    merged_anc.Clear(b);

    reach[a] = merged_reach;
    anc[a] = merged_anc;

    merged_anc.ForEachSetBit([&](int pred_node) {
      if (!alive[pred_node]) {
        return;
      }
      reach[pred_node].OrWith(merged_reach);
      reach[pred_node].Set(a);
      reach[pred_node].Clear(b);
    });
    merged_reach.ForEachSetBit([&](int succ_node) {
      if (!alive[succ_node]) {
        return;
      }
      anc[succ_node].OrWith(merged_anc);
      anc[succ_node].Set(a);
      anc[succ_node].Clear(b);
    });

    alive[b] = false;
    rep[b] = a;
    blocks[a].op_ids.insert(blocks[a].op_ids.end(), blocks[b].op_ids.begin(),
                            blocks[b].op_ids.end());
    blocks[b].op_ids.clear();
  };

  size_t total_pairs = 0;
  for (LiteRtParamIndex part_idx : sorted_indices) {
    const std::vector<int>& list = blocks_by_index.at(part_idx);
    if (list.size() > 1) {
      total_pairs += (list.size() * (list.size() - 1)) / 2;
    }
  }

  if (total_pairs <= kMaxCandidatePairs) {
    struct CandidatePair {
      int epoch_gap;
      int first_topo_a;
      int first_topo_b;
      int a;
      int b;
    };
    std::vector<CandidatePair> candidates;
    candidates.reserve(total_pairs);
    for (LiteRtParamIndex part_idx : sorted_indices) {
      const std::vector<int>& list = blocks_by_index.at(part_idx);
      for (size_t i = 0; i < list.size(); ++i) {
        const int a = list[i];
        for (size_t j = i + 1; j < list.size(); ++j) {
          const int b = list[j];
          candidates.push_back({blocks[b].epoch - blocks[a].epoch,
                                blocks[a].first_topo_pos,
                                blocks[b].first_topo_pos, a, b});
        }
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const CandidatePair& lhs, const CandidatePair& rhs) {
                if (lhs.epoch_gap != rhs.epoch_gap) {
                  return lhs.epoch_gap < rhs.epoch_gap;
                }
                if (lhs.first_topo_a != rhs.first_topo_a) {
                  return lhs.first_topo_a < rhs.first_topo_a;
                }
                return lhs.first_topo_b < rhs.first_topo_b;
              });
    for (const CandidatePair& pair : candidates) {
      try_merge(pair.a, pair.b);
    }
  } else {
    for (LiteRtParamIndex part_idx : sorted_indices) {
      const std::vector<int>& list = blocks_by_index.at(part_idx);
      for (size_t dist = 1; dist < list.size(); ++dist) {
        for (size_t i = 0; i + dist < list.size(); ++i) {
          try_merge(list[i], list[i + dist]);
        }
      }
    }
  }

  std::vector<int> alive_blocks;
  alive_blocks.reserve(num_blocks);
  for (int b = 0; b < num_blocks; ++b) {
    if (!alive[b] || blocks[b].op_ids.empty()) {
      continue;
    }
    std::sort(blocks[b].op_ids.begin(), blocks[b].op_ids.end(),
              [&](int lhs, int rhs) {
                return dag.topo_pos[lhs] < dag.topo_pos[rhs];
              });
    blocks[b].first_topo_pos = dag.topo_pos[blocks[b].op_ids.front()];
    alive_blocks.push_back(b);
  }

  // Emit contracted blocks in quotient-DAG topological order, breaking ties
  // deterministically by `first_topo_pos`.
  std::vector<int> pred_block_count(num_blocks, 0);
  for (int b : alive_blocks) {
    anc[b].ForEachSetBit([&](int pred_q) {
      if (pred_q < num_blocks && alive[pred_q]) {
        pred_block_count[b]++;
      }
    });
  }

  using ReadyBlock = std::pair<int, int>;  // (first_topo_pos, block_id)
  std::priority_queue<ReadyBlock, std::vector<ReadyBlock>,
                      std::greater<ReadyBlock>>
      ready_blocks;
  for (int b : alive_blocks) {
    if (pred_block_count[b] == 0) {
      ready_blocks.push({blocks[b].first_topo_pos, b});
    }
  }

  std::vector<EpochBlock> contracted;
  contracted.reserve(alive_blocks.size());
  while (!ready_blocks.empty()) {
    const int b = ready_blocks.top().second;
    ready_blocks.pop();
    reach[b].ForEachSetBit([&](int succ_q) {
      if (succ_q < num_blocks && alive[succ_q]) {
        if (--pred_block_count[succ_q] == 0) {
          ready_blocks.push({blocks[succ_q].first_topo_pos, succ_q});
        }
      }
    });
    contracted.push_back(std::move(blocks[b]));
  }
  ABSL_DCHECK_EQ(contracted.size(), alive_blocks.size());
  blocks = std::move(contracted);
}

}  // namespace

//===----------------------------------------------------------------------===//
// LiteRt Core partitioning algorithm.
//
// This algorithm is used to partition a flat list of ops into disjoint sets.
// Given a flat list of pairs of (op, partition_index), group ops by partition
// index to create a list of partitions.
//
// For example, if the input is:
//   (op1, 0), (op3, 1), (op2, 0), (op4, 2)
// The output will be:
//   [[op1, op2], [op3], [op4]]
//===----------------------------------------------------------------------===//

std::vector<std::vector<LiteRtOp>> GetPartitionsFromFlatListV2(
    const std::vector<LiteRtOpWithPartitionIndex>& ops,
    LiteRtSubgraph subgraph) {
  if (subgraph->Ops().empty()) {
    LITERT_LOG(LITERT_INFO, "Found empty subgraph, skipping partitioning");
    return {};
  }
  // Map of op to partition index.
  absl::flat_hash_map<LiteRtOp, LiteRtParamIndex> op_to_partition_index;
  for (LiteRtOpWithPartitionIndex op : ops) {
    op_to_partition_index[op.first] = op.second;
  }

  // Retain the execution order of the ops.
  //
  // Note: We do not use op index to maintain execution order, because inlined
  // ops from decomposition subgraph do not have respective index in the main
  // subgraph. Instead, we use the op order in the subgraph to maintain the
  // execution order.
  std::vector<std::pair<LiteRtOp, std::optional<LiteRtParamIndex>>> sorted_ops;
  sorted_ops.reserve(subgraph->Ops().size());
  for (LiteRtOp& op : subgraph->Ops()) {
    auto it = op_to_partition_index.find(op);
    if (it == op_to_partition_index.end()) {
      sorted_ops.push_back({op, std::nullopt});
    } else {
      sorted_ops.push_back({op, it->second});
    }
  }

  // Partition based on the execution order and partition index.
  std::vector<std::vector<LiteRtOp>> partitions;
  std::vector<LiteRtOp> current_partition;
  ABSL_DCHECK(!sorted_ops.empty());
  std::optional<LiteRtParamIndex> current_partition_index =
      sorted_ops.front().second;

  for (const auto& [op, index] : sorted_ops) {
    if (index != current_partition_index) {
      if (!current_partition.empty()) {
        partitions.push_back(std::move(current_partition));
        current_partition.clear();
      }
      current_partition_index = index;
    }
    if (current_partition_index.has_value()) {
      current_partition.push_back(op);
    }
  }
  if (!current_partition.empty()) {
    partitions.push_back(std::move(current_partition));
  }
  return partitions;
}

std::vector<std::vector<LiteRtOp>> GroupPartitions(
    const std::vector<LiteRtOpWithPartitionIndex>& ops,
    LiteRtSubgraph subgraph) {
  return DisjointSets::GetPartitionsFromFlatList(ops);
}

std::vector<std::vector<LiteRtOp>> GroupPartitionsV2(
    const std::vector<LiteRtOpWithPartitionIndex>& ops,
    LiteRtSubgraph subgraph) {
  return GetPartitionsFromFlatListV2(ops, subgraph);
}

std::vector<std::vector<LiteRtOp>> GroupPartitionsAcyclic(
    const std::vector<LiteRtOpWithPartitionIndex>& ops,
    LiteRtSubgraph subgraph) {
  if (subgraph == nullptr || subgraph->Ops().empty() || ops.empty()) {
    return {};
  }

  SubgraphDag dag;
  BuildSubgraphDag(ops, subgraph, dag);
  if (dag.num_unique_indices == 0) {
    return {};
  }

  if (static_cast<int>(dag.topo_order.size()) < dag.num_ops) {
    LITERT_LOG(LITERT_WARNING,
               "Input subgraph contains %d ops in or downstream of a cycle; "
               "emitting cyclic remainder ops as singleton partitions.",
               dag.num_ops - static_cast<int>(dag.topo_order.size()));
  }

  std::vector<int> epoch = ComputeEpochs(dag);
  std::vector<EpochBlock> blocks = BuildEpochBlocks(dag, epoch);
  if (dag.num_unique_indices > 1) {
    ContractMultiIndexQuotientDag(dag, epoch, blocks);
  }

  std::vector<std::vector<LiteRtOp>> partitions;
  partitions.reserve(blocks.size());
  for (const EpochBlock& block : blocks) {
    if (block.op_ids.empty()) {
      continue;
    }
    std::vector<LiteRtOp> group;
    group.reserve(block.op_ids.size());
    for (int op_id : block.op_ids) {
      group.push_back(dag.id_to_op[op_id]);
    }
    partitions.push_back(std::move(group));
  }

  if (static_cast<int>(dag.topo_order.size()) < dag.num_ops) {
    for (int i = 0; i < dag.num_ops; ++i) {
      if (dag.is_selected[i] && dag.topo_pos[i] < 0) {
        partitions.push_back({dag.id_to_op[i]});
      }
    }
  }
  return partitions;
}

Expected<LiteRtOp> OutlinePartition(LiteRtSubgraphT& root, LiteRtSubgraph slice,
                                    std::vector<LiteRtOp>& partition) {
  return GraphSlicer::SlicePartitionFromGraph(root, slice, partition);
}

Expected<void> InlineSubgraph(LiteRtModelT& model, LiteRtOpT& destination_op,
                              LiteRtSubgraph source_subgraph) {
  // Check if the input and output tensors of the destination op are the same as
  // the input and output tensors of the source subgraph.
  if (destination_op.Inputs().size() != source_subgraph->Inputs().size() ||
      destination_op.Outputs().size() != source_subgraph->Outputs().size()) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Input/output tensors of destination op and source subgraph "
        "are not the same.");
    ;
  }
  auto verify_tensor_type = [&](const LiteRtTensorT& dest_tensor,
                                const LiteRtTensorT& source_tensor) {
    auto& destination_op_tensor_type = dest_tensor.Type();
    auto& source_subgraph_tensor_type = source_tensor.Type();
    if (destination_op_tensor_type.first != source_subgraph_tensor_type.first) {
      LITERT_LOG(LITERT_ERROR,
                 "Tensors of destination op and source subgraph "
                 "are not the same type.");
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (destination_op_tensor_type.first == kLiteRtUnrankedTensorType) {
      if (!LiteRtIsSameUnrankedTensorType(
              &destination_op_tensor_type.second.unranked_tensor_type,
              &source_subgraph_tensor_type.second.unranked_tensor_type)) {
        LITERT_LOG(LITERT_ERROR,
                   "tensors of destination op and source subgraph "
                   "does not have the same unranked tensor type.");
        return kLiteRtStatusErrorInvalidArgument;
      }
    }
    if (destination_op_tensor_type.first == kLiteRtRankedTensorType) {
      // check element type is the same.
      if (destination_op_tensor_type.second.ranked_tensor_type.element_type !=
          source_subgraph_tensor_type.second.ranked_tensor_type.element_type) {
        LITERT_LOG(LITERT_ERROR,
                   "Tensors of destination op and source subgraph "
                   "does not have the same element type.");
        return kLiteRtStatusErrorInvalidArgument;
      }
      bool is_same_layout = false;
      LiteRtIsSameLayout(
          &destination_op_tensor_type.second.ranked_tensor_type.layout,
          &source_subgraph_tensor_type.second.ranked_tensor_type.layout,
          &is_same_layout);
      if (!is_same_layout) {
        LITERT_LOG(LITERT_ERROR,
                   "Tensors of destination op and source subgraph "
                   "does not have the same ranked tensor layout.");
        return kLiteRtStatusErrorInvalidArgument;
      }
    }
    return kLiteRtStatusOk;
  };

  for (int i = 0; i < destination_op.Inputs().size(); ++i) {
    LITERT_RETURN_IF_ERROR(verify_tensor_type(
        *destination_op.Inputs().at(i), *source_subgraph->Inputs().at(i)));
  }
  for (int i = 0; i < destination_op.Outputs().size(); ++i) {
    LITERT_RETURN_IF_ERROR(verify_tensor_type(
        *destination_op.Outputs().at(i), *source_subgraph->Outputs().at(i)));
  }

  // Find the main subgraph containing the destination op.
  LiteRtSubgraphT* main_subgraph = nullptr;
  for (auto& subgraph : model.Subgraphs()) {
    for (auto& op : subgraph->Ops()) {
      if (op == &destination_op) {
        main_subgraph = subgraph;
        break;
      }
    }
    if (main_subgraph != nullptr) {
      break;
    }
  }
  if (main_subgraph == nullptr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Failed to find the subgraph containing the op during "
                      "inlining.");
  }

  // Maps of cloned tensors and ops in main subgraph.
  InsertOrderMap</*original*/ LiteRtTensor, /*cloned*/ LiteRtTensor>
      tensor_map_;
  InsertOrderMap</*original*/ LiteRtOp, /*cloned*/ LiteRtOp> op_map_;

  // Copy all ops from decomp subgraph to main subgraph. The new ops are
  // inserted after the composite op to maintain topological order.
  auto op_index_ret = FindInd(main_subgraph->Ops().cbegin(),
                              main_subgraph->Ops().cend(), &destination_op);
  if (!op_index_ret) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Failed to find op in given subgraph during inlining.");
  }
  int op_index = op_index_ret.value();

  for (auto& decomp_op : source_subgraph->Ops()) {
    auto& new_op = main_subgraph->EmplaceOpAt(op_index++);
    CloneTo(*decomp_op, new_op);
    op_map_.InsertOrAssign(decomp_op, &new_op);
  }

  // Copy all tensors from decomp subgraph to main subgraph.
  for (auto& decomp_tensor : source_subgraph->Tensors()) {
    auto& new_tensor = main_subgraph->EmplaceTensor();
    CloneTo(*decomp_tensor, new_tensor);
    tensor_map_.InsertOrAssign(decomp_tensor, &new_tensor);
  }

  // Restore the topology of the decomp subgraph to the main subgraph.
  for (auto& decomp_op : source_subgraph->Ops()) {
    auto cloned_op = op_map_.Find(decomp_op)->get().second;
    for (auto& decomp_input : decomp_op->Inputs()) {
      auto cloned_input = tensor_map_.Find(decomp_input)->get().second;
      cloned_op->Inputs().push_back(cloned_input);
      // Restore the user list of the cloned input tensor. Note: this only
      // needs to be done once when swiping through the input.
      cloned_input->Users().push_back(cloned_op);
      cloned_input->UserArgInds().push_back(cloned_op->Inputs().size() - 1);
    }

    for (auto& decomp_output : decomp_op->Outputs()) {
      auto cloned_output = tensor_map_.Find(decomp_output)->get().second;
      // `CloneTo` does not copy the defining op, so attach it explicitly.
      AttachOutput(cloned_output, *cloned_op);
    }
  }

  // Connect the input tensor of the composite op to users of the input
  // tensors of the decomp subgraph.
  for (int input_ind = 0; input_ind < destination_op.Inputs().size();
       ++input_ind) {
    auto& decomp_input_tensor = source_subgraph->Input(input_ind);
    auto users = decomp_input_tensor.Users();
    auto user_args = decomp_input_tensor.UserArgInds();
    // The composite op is removed after inlining, so drop every use it holds
    // on this input, even if the decomposition never reads it. Iterate
    // backwards because `RemoveUse` shifts later uses.
    auto& composite_input = destination_op.Input(input_ind);
    for (size_t use_ind = composite_input.NumUses(); use_ind-- > 0;) {
      if (composite_input.Users().at(use_ind) == &destination_op) {
        composite_input.RemoveUse(use_ind);
      }
    }
    for (int user_ind = 0; user_ind < users.size(); ++user_ind) {
      auto& cloned_op = op_map_.Find(users.at(user_ind))->get().second;
      cloned_op->Inputs().at(user_args.at(user_ind)) = &composite_input;
      // Update the user list of the original composite op input tensors.
      composite_input.Users().push_back(cloned_op);
      composite_input.UserArgInds().push_back(user_args.at(user_ind));
    }
  }

  // We need to clean up the cloned GraphInput tensors of the decomp subgraph,
  // since all input tensors were reused from the main subgraph.
  for (auto& decomp_input : source_subgraph->Inputs()) {
    auto& cloned_input = tensor_map_.Find(decomp_input)->get().second;
    cloned_input->Users().clear();
    cloned_input->UserArgInds().clear();
    cloned_input->ClearDefiningOp();
  }

  // Reroute the cloned tensor to the original output of the composite op.
  for (int output_ind = 0; output_ind < destination_op.Outputs().size();
       ++output_ind) {
    auto& decomp_output_tensor = source_subgraph->Output(output_ind);
    auto& cloned_tensor = tensor_map_.Find(&decomp_output_tensor)->get().second;
    auto& composite_output_tensor = destination_op.Output(output_ind);
    auto users = composite_output_tensor.Users();
    auto user_args = composite_output_tensor.UserArgInds();

    for (int user_ind = 0; user_ind < users.size(); ++user_ind) {
      users.at(user_ind)->Inputs().at(user_args.at(user_ind)) = cloned_tensor;
      cloned_tensor->Users().push_back(users.at(user_ind));
      cloned_tensor->UserArgInds().push_back(user_args.at(user_ind));
    }

    // In case of the output of the original composite op is in GraphOutputs,
    // we need to replace it with the cloned tensor.
    for (auto& graph_output : main_subgraph->Outputs()) {
      if (graph_output == &composite_output_tensor) {
        graph_output = cloned_tensor;
      }
    }
    // Signatures own separate tensor references. Updating only GraphOutputs
    // leaves their positional and named aliases dangling after DCE below.
    for (auto* signature : model.Signatures()) {
      if (&signature->GetSubgraph() == main_subgraph) {
        signature->RemapOutputTensor(&composite_output_tensor, cloned_tensor);
      }
    }
  }

  // Clear the output of original composite op, they are not used anymore.
  // Those tensor will be removed by DCE.
  // TODO(yunandrew): Refactor the code with using model_graph::Drop().
  for (auto original_output : destination_op.Outputs()) {
    original_output->Users().clear();
    original_output->UserArgInds().clear();
    original_output->ClearDefiningOp();
  }

  // Clear the input/ouput tensor of the original composite op, so the
  // composite op will be removed by DCE.
  destination_op.Inputs().clear();
  destination_op.Outputs().clear();

  // Clean up the subgraph.
  DCE(*main_subgraph);
  return Expected<void>();
}

}  // namespace litert::internal
