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
#include <optional>
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

// True if outlining `partition` from `root` produces at least one dispatch-op
// input or output edge.
bool HasBoundaryEdges(const LiteRtSubgraphT& root,
                      const std::vector<LiteRtOp>& partition) {
  if (partition.empty()) {
    return false;
  }
  absl::flat_hash_set<const LiteRtOpT*> partition_ops(partition.begin(),
                                                      partition.end());
  for (const LiteRtOp op : partition) {
    for (const LiteRtTensorT* input : op->Inputs()) {
      if (input != nullptr && !IsConstant(*input) &&
          (input->DefiningOp() == nullptr ||
           !partition_ops.contains(input->DefiningOp()))) {
        return true;
      }
    }
    for (const LiteRtTensorT* output : op->Outputs()) {
      if (FindOutput(root, *output)) {
        return true;
      }
      if (!IsConstant(*output)) {
        for (const LiteRtOpT* user : output->Users()) {
          if (!partition_ops.contains(user)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// GraphSlicer Implementation
//===----------------------------------------------------------------------===//

litert::Expected<LiteRtOp> GraphSlicer::SlicePartitionFromGraph(
    LiteRtSubgraphT& root, LiteRtSubgraph slice,
    std::vector<LiteRtOp>& partition) {
  // Reject partitions that would produce an edgeless dispatch op before
  // mutating `root` or `slice`: DCE would remove that dispatch op and leave
  // the caller with a dangling pointer.
  if (!HasBoundaryEdges(root, partition)) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Partition has no inputs or outputs");
  }

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
    MarkDead(*op);
  }

  // Reuse the storage from the last op in partition.
  // The actual topological order will be fixed at the end.
  slicer.dispatch_op_ = partition.back();

  ABSL_DCHECK(slicer.dispatch_op_->Inputs().empty());
  ABSL_DCHECK(slicer.dispatch_op_->Outputs().empty());
  MakeDispatchOp(*slicer.dispatch_op_);
  slicer.RerouteTensorsThroughCustomOp(root);

  ABSL_DCHECK(!slicer.dispatch_op_->Inputs().empty() ||
              !slicer.dispatch_op_->Outputs().empty());

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

// Max quotient-DAG vertices (epoch blocks + intervening CPU ops) for bitmatrix
// contraction; larger multi-index graphs keep epoch-layered blocks as-is.
constexpr int kMaxContractionNodes = 1024;
// 21-bit index cap (< 2M ops) allowing (22-bit, 21-bit, 21-bit) tuples to pack
// into uint64_t keys so all sorts share one std::sort<uint64_t*> instantiation:
// - `keys`: [63:42] epoch | [41:21] part_id | [20:0] topo_pos
// - `block_spans`: [63:42] epoch | [41:21] first_pos | [20:0] key_idx
// - `candidates`: [63:42] d_ep | [41:21] pos_a | [20:0] pos_b
constexpr int kMaxPackedDagOps = (1 << 21) - 1;
constexpr uint64_t kMask21 = (uint64_t{1} << 21) - 1;
constexpr uint64_t kHigh22Mask = ~((uint64_t{1} << 42) - 1);

#if defined(__clang__)
#define LITERT_SMALL_LOOP \
  _Pragma("clang loop vectorize(disable) unroll(disable)")
#else
#define LITERT_SMALL_LOOP
#endif

struct IntSlice {
  const int* b;
  const int* e;
  const int* begin() const { return b; }
  const int* end() const { return e; }
};

struct CsrView {
  const int* off = nullptr;
  const int* adj = nullptr;
  IntSlice operator[](int u) const { return {adj + off[u], adj + off[u + 1]}; }
};

void SortAndUnique(std::vector<uint64_t>& v) {
  std::sort(v.begin(), v.end());
  v.resize(static_cast<size_t>(std::unique(v.begin(), v.end()) - v.begin()));
}

void BuildSortedCsr(int num_nodes, const std::vector<uint64_t>& sorted_edges,
                    int* off, int* adj) {
  LITERT_SMALL_LOOP
  for (int i = 0; i <= num_nodes; ++i) {
    off[i] = 0;
  }
  LITERT_SMALL_LOOP
  for (size_t idx = 0; idx < sorted_edges.size(); ++idx) {
    const uint64_t e = sorted_edges[idx];
    ++off[static_cast<int>(e >> 32) + 1];
    adj[idx] = static_cast<int>(e & 0xffffffffU);
  }
  LITERT_SMALL_LOOP
  for (int i = 0; i < num_nodes; ++i) {
    off[i + 1] += off[i];
  }
}

// Flat pointer views into a single contiguous int[] arena for the subgraph DAG
// and epoch-block partitioning state.
struct DagWorkspace {
  int num_ops = 0;
  int num_unique_indices = 0;
  int topo_size = 0;
  int num_blocks = 0;
  int num_out_blocks = 0;

  CsrView succs;
  CsrView preds;
  int* topo_order = nullptr;
  int* topo_pos = nullptr;
  // Dense rank in [0, num_unique_indices) for selected ops; -1 for unselected.
  int* part_id = nullptr;
  int* epoch = nullptr;
  int* block_epoch = nullptr;
  int* block_part = nullptr;
  int* block_first_tpos = nullptr;
  int* block_to_out = nullptr;
  int* tpos_to_block = nullptr;
};

struct QuotientContraction {
  size_t num_words;
  int num_blocks;
  uint64_t* reach;
  uint64_t* anc;
  int* alive;
  int* rep;
  int* pred_block_count;

  static void SetBit(uint64_t* mat, size_t words, int row, int bit) {
    const size_t idx = static_cast<size_t>(bit);
    mat[static_cast<size_t>(row) * words + (idx >> 6)] |=
        (uint64_t{1} << (idx & 63));
  }

  static void ClearBit(uint64_t* mat, size_t words, int row, int bit) {
    const size_t idx = static_cast<size_t>(bit);
    mat[static_cast<size_t>(row) * words + (idx >> 6)] &=
        ~(uint64_t{1} << (idx & 63));
  }

  void OrRow(uint64_t* mat, int dst, int src) const {
    uint64_t* d = mat + static_cast<size_t>(dst) * num_words;
    const uint64_t* s = mat + static_cast<size_t>(src) * num_words;
    LITERT_SMALL_LOOP
    for (size_t w = 0; w < num_words; ++w) {
      d[w] |= s[w];
    }
  }

  bool RowsIntersect(int r_row, int a_row) const {
    const uint64_t* r = reach + static_cast<size_t>(r_row) * num_words;
    const uint64_t* a = anc + static_cast<size_t>(a_row) * num_words;
    LITERT_SMALL_LOOP
    for (size_t w = 0; w < num_words; ++w) {
      if ((r[w] & a[w]) != 0) {
        return true;
      }
    }
    return false;
  }

  int FindRep(int x) const {
    LITERT_SMALL_LOOP
    while (rep[x] != x) {
      rep[x] = rep[rep[x]];
      x = rep[x];
    }
    return x;
  }

  void UpdateRelatives(const uint64_t* iter_mat, uint64_t* upd_mat, int keep,
                       int drop) const {
    const uint64_t* r = iter_mat + static_cast<size_t>(keep) * num_words;
    LITERT_SMALL_LOOP
    for (size_t w = 0; w < num_words; ++w) {
      uint64_t word = r[w];
      while (word != 0) {
        const int node = static_cast<int>(w * 64) + absl::countr_zero(word);
        word &= (word - 1);
        if (alive[node]) {
          OrRow(upd_mat, node, keep);
          SetBit(upd_mat, num_words, node, keep);
          ClearBit(upd_mat, num_words, node, drop);
        }
      }
    }
  }

  void AddReachDeg(int src, int delta) const {
    const uint64_t* r = reach + static_cast<size_t>(src) * num_words;
    LITERT_SMALL_LOOP
    for (size_t w = 0; w < num_words; ++w) {
      uint64_t word = r[w];
      while (word != 0) {
        const int succ_q = static_cast<int>(w * 64) + absl::countr_zero(word);
        word &= (word - 1);
        if (succ_q < num_blocks && alive[succ_q]) {
          pred_block_count[succ_q] += delta;
        }
      }
    }
  }
};

// Assigns each op its alternating-label epoch: unselected ops with no selected
// ancestors get -1, while each transition into a selected op from a different
// partition label (or unselected op with a selected ancestor) increments epoch.
void ComputeEpochs(DagWorkspace& ws) {
  for (int t = 0; t < ws.topo_size; ++t) {
    const int node = ws.topo_order[t];
    const int node_part = ws.part_id[node];
    int max_epoch = (node_part < 0) ? -1 : 0;
    LITERT_SMALL_LOOP
    for (int pred : ws.preds[node]) {
      const int pe = ws.epoch[pred];
      if (node_part < 0 || ws.part_id[pred] == node_part) {
        max_epoch = std::max(max_epoch, pe);
      } else if (pe >= 0) {
        max_epoch = std::max(max_epoch, pe + 1);
      }
    }
    ws.epoch[node] = max_epoch;
  }
}

// Groups selected ops by `(epoch, part_id)` in topological order. For a single
// partition index, these epoch blocks already achieve the minimum acyclic
// partition count.
void BuildEpochBlocks(DagWorkspace& ws, std::vector<uint64_t>& keys,
                      std::vector<uint64_t>& block_spans) {
  keys.clear();
  LITERT_SMALL_LOOP
  for (int t = 0; t < ws.topo_size; ++t) {
    const int node = ws.topo_order[t];
    if (ws.part_id[node] >= 0) {
      keys.push_back((static_cast<uint64_t>(ws.epoch[node]) << 42) |
                     (static_cast<uint64_t>(ws.part_id[node]) << 21) |
                     static_cast<uint64_t>(t));
    }
  }
  std::sort(keys.begin(), keys.end());

  block_spans.clear();
  for (size_t i = 0; i < keys.size();) {
    const uint64_t prefix = keys[i] >> 21;
    const uint64_t first_tpos = keys[i] & kMask21;
    block_spans.push_back((keys[i] & kHigh22Mask) | (first_tpos << 21) |
                          static_cast<uint64_t>(i));
    LITERT_SMALL_LOOP
    while (i < keys.size() && (keys[i] >> 21) == prefix) {
      ++i;
    }
  }
  if (ws.num_unique_indices > 1) {
    std::sort(block_spans.begin(), block_spans.end());
  }

  const int num_blocks = static_cast<int>(block_spans.size());
  ws.num_blocks = num_blocks;
  ws.num_out_blocks = num_blocks;
  LITERT_SMALL_LOOP
  for (int t = 0; t < ws.topo_size; ++t) {
    ws.tpos_to_block[t] = -1;
  }

  LITERT_SMALL_LOOP
  for (int b = 0; b < num_blocks; ++b) {
    size_t idx = static_cast<size_t>(block_spans[b] & kMask21);
    const uint64_t prefix = keys[idx] >> 21;
    const int first_tpos = static_cast<int>(keys[idx] & kMask21);
    const int first_node = ws.topo_order[first_tpos];
    ws.block_epoch[b] = ws.epoch[first_node];
    ws.block_part[b] = ws.part_id[first_node];
    ws.block_first_tpos[b] = first_tpos;
    ws.block_to_out[b] = b;
    LITERT_SMALL_LOOP
    while (idx < keys.size() && (keys[idx] >> 21) == prefix) {
      ws.tpos_to_block[static_cast<size_t>(keys[idx] & kMask21)] = b;
      ++idx;
    }
  }
}

// Contracts same-`part_id` epoch blocks across different epochs when no
// intermediate quotient-DAG node lies on a directed path between them,
// maintaining irreflexive transitive reachability/ancestry bitmatrices.
void ContractMultiIndexQuotientDag(DagWorkspace& ws, int* op_scratch,
                                   std::vector<int>& q_arena,
                                   std::vector<uint64_t>& u64_a,
                                   std::vector<uint64_t>& bitmat) {
  const int num_blocks = ws.num_blocks;
  if (num_blocks <= 1) {
    return;
  }

  int* has_selected_desc = op_scratch;
  int* op_to_q = has_selected_desc + ws.num_ops;
  LITERT_SMALL_LOOP
  for (int i = 0; i < ws.num_ops; ++i) {
    has_selected_desc[i] = 0;
    op_to_q[i] = -1;
  }
  for (int t = ws.topo_size - 1; t >= 0; --t) {
    const int node = ws.topo_order[t];
    if (ws.part_id[node] >= 0) {
      has_selected_desc[node] = 1;
    }
    if (!has_selected_desc[node]) {
      continue;
    }
    LITERT_SMALL_LOOP
    for (int pred : ws.preds[node]) {
      has_selected_desc[pred] = 1;
    }
  }

  int num_q = num_blocks;
  LITERT_SMALL_LOOP
  for (int t = 0; t < ws.topo_size; ++t) {
    const int node = ws.topo_order[t];
    int q = ws.tpos_to_block[t];
    if (q < 0 && ws.epoch[node] >= 0 && has_selected_desc[node]) {
      q = num_q++;
    }
    op_to_q[node] = q;
  }
  if (num_q > kMaxContractionNodes) {
    return;
  }

  u64_a.clear();
  for (int t = 0; t < ws.topo_size; ++t) {
    const int u = ws.topo_order[t];
    const int qu = op_to_q[u];
    if (qu < 0) {
      continue;
    }
    LITERT_SMALL_LOOP
    for (int v : ws.succs[u]) {
      const int qv = op_to_q[v];
      if (qv >= 0 && qu != qv) {
        u64_a.push_back((static_cast<uint64_t>(qu) << 32) |
                        static_cast<uint32_t>(qv));
      }
    }
  }
  SortAndUnique(u64_a);

  ws.succs = {};
  ws.preds = {};
  const size_t qstride = static_cast<size_t>(num_q + 1);
  q_arena.assign(7 * qstride + u64_a.size(), 0);
  int* q_succs_off = q_arena.data();
  int* q_in_degree = q_succs_off + qstride;
  int* q_topo = q_in_degree + qstride;
  int* alive = q_topo + qstride;
  int* rep = alive + qstride;
  int* pred_block_count = rep + qstride;
  int* root_to_out = pred_block_count + qstride;
  int* q_succs_adj = root_to_out + qstride;

  LITERT_SMALL_LOOP
  for (uint64_t qe : u64_a) {
    ++q_in_degree[static_cast<int>(qe & 0xffffffffU)];
  }
  BuildSortedCsr(num_q, u64_a, q_succs_off, q_succs_adj);
  const CsrView q_succs{q_succs_off, q_succs_adj};

  int q_topo_size = 0;
  LITERT_SMALL_LOOP
  for (int q = 0; q < num_q; ++q) {
    alive[q] = 1;
    if (q_in_degree[q] == 0) {
      q_topo[q_topo_size++] = q;
    }
  }
  for (int head = 0; head < q_topo_size; ++head) {
    const int q = q_topo[head];
    LITERT_SMALL_LOOP
    for (int v : q_succs[q]) {
      if (--q_in_degree[v] == 0) {
        q_topo[q_topo_size++] = v;
      }
    }
  }
  if (q_topo_size != num_q) {
    return;
  }

  const size_t num_words = static_cast<size_t>(num_q + 63) / 64;
  const size_t mat_words = static_cast<size_t>(num_q) * num_words;
  bitmat.assign(2 * mat_words, 0);
  QuotientContraction qc{
      num_words, num_blocks, bitmat.data(),   bitmat.data() + mat_words,
      alive,     rep,        pred_block_count};

  for (int i = num_q - 1; i >= 0; --i) {
    const int q = q_topo[i];
    LITERT_SMALL_LOOP
    for (int v : q_succs[q]) {
      qc.OrRow(qc.reach, q, v);
      QuotientContraction::SetBit(qc.reach, num_words, q, v);
    }
  }
  for (int i = 0; i < num_q; ++i) {
    const int q = q_topo[i];
    LITERT_SMALL_LOOP
    for (int v : q_succs[q]) {
      qc.OrRow(qc.anc, v, q);
      QuotientContraction::SetBit(qc.anc, num_words, v, q);
    }
  }

  LITERT_SMALL_LOOP
  for (int b = 0; b < num_blocks; ++b) {
    rep[b] = b;
  }

  // Blocks are ordered by `(epoch, first_topo_pos)` with unique `(epoch,
  // part_id)`, so `a < b` with equal `part_id` implies `block_epoch[a] <
  // block_epoch[b]`.
  u64_a.clear();
  for (int a = 0; a < num_blocks; ++a) {
    LITERT_SMALL_LOOP
    for (int b = a + 1; b < num_blocks; ++b) {
      if (ws.block_part[a] == ws.block_part[b]) {
        u64_a.push_back(
            (static_cast<uint64_t>(ws.block_epoch[b] - ws.block_epoch[a])
             << 42) |
            (static_cast<uint64_t>(ws.block_first_tpos[a]) << 21) |
            static_cast<uint64_t>(ws.block_first_tpos[b]));
      }
    }
  }
  std::sort(u64_a.begin(), u64_a.end());

  int num_alive = num_blocks;
  for (uint64_t key : u64_a) {
    int a = qc.FindRep(
        ws.tpos_to_block[static_cast<size_t>((key >> 21) & kMask21)]);
    int b = qc.FindRep(ws.tpos_to_block[static_cast<size_t>(key & kMask21)]);
    if (a == b) {
      continue;
    }
    if (ws.block_first_tpos[b] < ws.block_first_tpos[a]) {
      std::swap(a, b);
    }
    if (qc.RowsIntersect(a, b) || qc.RowsIntersect(b, a)) {
      continue;
    }

    qc.OrRow(qc.reach, a, b);
    QuotientContraction::ClearBit(qc.reach, num_words, a, a);
    QuotientContraction::ClearBit(qc.reach, num_words, a, b);

    qc.OrRow(qc.anc, a, b);
    QuotientContraction::ClearBit(qc.anc, num_words, a, a);
    QuotientContraction::ClearBit(qc.anc, num_words, a, b);

    qc.UpdateRelatives(qc.anc, qc.reach, a, b);
    qc.UpdateRelatives(qc.reach, qc.anc, a, b);

    alive[b] = 0;
    rep[b] = a;
    --num_alive;
  }

  for (int b = 0; b < num_blocks; ++b) {
    if (alive[b]) {
      qc.AddReachDeg(b, 1);
    }
  }

  for (int step = 0; step < num_alive; ++step) {
    int chosen = -1;
    LITERT_SMALL_LOOP
    for (int b = 0; b < num_blocks; ++b) {
      if (alive[b] && pred_block_count[b] == 0 &&
          (chosen < 0 ||
           ws.block_first_tpos[b] < ws.block_first_tpos[chosen])) {
        chosen = b;
      }
    }
    if (chosen < 0) {
      ABSL_DCHECK_GE(chosen, 0);
      return;
    }
    alive[chosen] = 0;
    root_to_out[chosen] = step;
    qc.AddReachDeg(chosen, -1);
  }

  ws.num_out_blocks = num_alive;
  LITERT_SMALL_LOOP
  for (int b = 0; b < num_blocks; ++b) {
    ws.block_to_out[b] = root_to_out[qc.FindRep(b)];
  }
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
  if (subgraph == nullptr || subgraph->Ops().empty()) {
    LITERT_LOG(LITERT_INFO, "Found empty subgraph, skipping partitioning");
    return {};
  }
  absl::flat_hash_map<LiteRtOp, LiteRtParamIndex> op_to_partition_index;
  op_to_partition_index.reserve(ops.size());
  for (const auto& [op, index] : ops) {
    if (op != nullptr) {
      op_to_partition_index[op] = index;
    }
  }

  std::vector<std::vector<LiteRtOp>> partitions;
  std::vector<LiteRtOp> current_partition;
  std::optional<LiteRtParamIndex> current_index;
  for (LiteRtOp op : subgraph->Ops()) {
    auto it = op_to_partition_index.find(op);
    const std::optional<LiteRtParamIndex> index =
        (it == op_to_partition_index.end())
            ? std::nullopt
            : std::optional<LiteRtParamIndex>(it->second);
    if (index != current_index) {
      if (!current_partition.empty()) {
        partitions.push_back(std::move(current_partition));
        current_partition.clear();
      }
      current_index = index;
    }
    if (current_index.has_value()) {
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

  const auto& subgraph_ops = subgraph->Ops();
  std::vector<LiteRtOp> id_to_op;
  id_to_op.reserve(subgraph_ops.size());
  absl::flat_hash_map<LiteRtOp, LiteRtParamIndex> op_to_id;
  op_to_id.reserve(subgraph_ops.size());
  for (LiteRtOp op : subgraph_ops) {
    if (op != nullptr && op_to_id.emplace(op, id_to_op.size()).second) {
      id_to_op.push_back(op);
      if (id_to_op.size() > static_cast<size_t>(kMaxPackedDagOps)) {
        return GetPartitionsFromFlatListV2(ops, subgraph);
      }
    }
  }
  const int num_ops = static_cast<int>(id_to_op.size());
  if (num_ops == 0) {
    return {};
  }

  const size_t stride = static_cast<size_t>(num_ops + 1);
  std::vector<int> arena(13 * stride, 0);
  int* succs_off = arena.data();
  int* preds_off = succs_off + stride;
  int* topo_order = preds_off + stride;
  int* topo_pos = topo_order + stride;
  int* part_id = topo_pos + stride;
  int* in_degree = part_id + stride;
  int* ready = in_degree + stride;

  DagWorkspace ws;
  ws.num_ops = num_ops;
  ws.topo_order = topo_order;
  ws.topo_pos = topo_pos;
  ws.part_id = part_id;
  ws.epoch = ready + stride;
  ws.block_epoch = ws.epoch + stride;
  ws.block_part = ws.block_epoch + stride;
  ws.block_first_tpos = ws.block_part + stride;
  ws.block_to_out = ws.block_first_tpos + stride;
  ws.tpos_to_block = ws.block_to_out + stride;

  LITERT_SMALL_LOOP
  for (int i = 0; i < num_ops; ++i) {
    topo_pos[i] = -1;
    part_id[i] = -1;
  }

  std::vector<uint64_t> u64_a;
  std::vector<uint64_t> u64_b(static_cast<size_t>(num_ops), 0);
  for (const auto& [op, part_idx] : ops) {
    auto it = op_to_id.find(op);
    if (it != op_to_id.end()) {
      const size_t id = it->second;
      part_id[id] = 0;
      u64_b[id] = static_cast<uint64_t>(part_idx);
    }
  }
  LITERT_SMALL_LOOP
  for (int i = 0; i < num_ops; ++i) {
    if (part_id[i] >= 0) {
      u64_a.push_back(u64_b[i]);
    }
  }
  if (u64_a.empty()) {
    return {};
  }
  SortAndUnique(u64_a);
  ws.num_unique_indices = static_cast<int>(u64_a.size());
  LITERT_SMALL_LOOP
  for (int i = 0; i < num_ops; ++i) {
    if (part_id[i] >= 0) {
      part_id[i] = static_cast<int>(
          std::lower_bound(u64_a.begin(), u64_a.end(), u64_b[i]) -
          u64_a.begin());
    }
  }

  u64_a.clear();
  for (int v = 0; v < num_ops; ++v) {
    for (const auto* input : id_to_op[v]->Inputs()) {
      if (input == nullptr || input->DefiningOp() == nullptr) {
        continue;
      }
      auto it = op_to_id.find(input->DefiningOp());
      if (it != op_to_id.end()) {
        u64_a.push_back((static_cast<uint64_t>(it->second) << 32) |
                        static_cast<uint32_t>(v));
      }
    }
  }
  SortAndUnique(u64_a);

  std::vector<int> adj_buf(2 * u64_a.size(), 0);
  int* succs_adj = adj_buf.data();
  int* preds_adj = succs_adj + u64_a.size();
  BuildSortedCsr(num_ops, u64_a, succs_off, succs_adj);
  LITERT_SMALL_LOOP
  for (uint64_t& e : u64_a) {
    e = (e << 32) | (e >> 32);
  }
  std::sort(u64_a.begin(), u64_a.end());
  BuildSortedCsr(num_ops, u64_a, preds_off, preds_adj);
  ws.succs = {succs_off, succs_adj};
  ws.preds = {preds_off, preds_adj};

  int ready_size = 0;
  LITERT_SMALL_LOOP
  for (int i = 0; i < num_ops; ++i) {
    in_degree[i] = preds_off[i + 1] - preds_off[i];
    if (in_degree[i] == 0) {
      ready[ready_size++] = i;
    }
  }
  for (int head = 0; head < ready_size; ++head) {
    const int u = ready[head];
    topo_pos[u] = ws.topo_size;
    topo_order[ws.topo_size++] = u;
    LITERT_SMALL_LOOP
    for (int v : ws.succs[u]) {
      if (--in_degree[v] == 0) {
        ready[ready_size++] = v;
      }
    }
  }

  ComputeEpochs(ws);
  BuildEpochBlocks(ws, u64_a, u64_b);
  if (ws.num_unique_indices > 1) {
    // `in_degree` + `ready` (`2 * stride` ints) are dead after Kahn's sort,
    // and `adj_buf` is dead once `q_edges` is extracted from `ws.succs`.
    ContractMultiIndexQuotientDag(ws, in_degree, adj_buf, u64_a, u64_b);
  }

  std::vector<std::vector<LiteRtOp>> partitions(ws.num_out_blocks);
  for (int t = 0; t < ws.topo_size; ++t) {
    const int b = ws.tpos_to_block[t];
    if (b >= 0) {
      partitions[ws.block_to_out[b]].push_back(id_to_op[topo_order[t]]);
    }
  }

  if (ws.topo_size < num_ops) {
    for (int i = 0; i < num_ops; ++i) {
      if (part_id[i] >= 0 && topo_pos[i] < 0) {
        partitions.emplace_back(1, id_to_op[i]);
      }
    }
  }
  return partitions;
}

#undef LITERT_SMALL_LOOP

Expected<LiteRtOp> OutlinePartition(LiteRtSubgraphT& root, LiteRtSubgraph slice,
                                    std::vector<LiteRtOp>& partition) {
  return GraphSlicer::SlicePartitionFromGraph(root, slice, partition);
}

Expected<void> CheckInlinable(const LiteRtOpT& destination_op,
                              const LiteRtSubgraphT& source_subgraph) {
  // Check if the input and output tensors of the destination op are the same as
  // the input and output tensors of the source subgraph.
  if (destination_op.Inputs().size() != source_subgraph.Inputs().size() ||
      destination_op.Outputs().size() != source_subgraph.Outputs().size()) {
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
    LITERT_RETURN_IF_ERROR(verify_tensor_type(*destination_op.Inputs().at(i),
                                              *source_subgraph.Inputs().at(i)));
  }
  for (int i = 0; i < destination_op.Outputs().size(); ++i) {
    LITERT_RETURN_IF_ERROR(verify_tensor_type(
        *destination_op.Outputs().at(i), *source_subgraph.Outputs().at(i)));
  }

  return {};
}

Expected<void> InlineSubgraph(LiteRtModelT& model, LiteRtOpT& destination_op,
                              LiteRtSubgraph source_subgraph) {
  if (source_subgraph == nullptr) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Source subgraph is null.");
  }
  LITERT_RETURN_IF_ERROR(CheckInlinable(destination_op, *source_subgraph));

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

  // Detach the composite op and clear its decomposition reference so DCE
  // removes it.
  destination_op.Inputs().clear();
  destination_op.Outputs().clear();
  ClearTflOptions(destination_op);
  destination_op.ClearCustomOptions();

  // Clean up the subgraph.
  DCE(*main_subgraph);
  return Expected<void>();
}

}  // namespace litert::internal
