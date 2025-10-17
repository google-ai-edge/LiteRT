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

#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
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

//
// slice partitions out of a subgraph (into new subgraphs)
//===----------------------------------------------------------------------===//

class GraphSlicer {
 public:
  // Slices "partitions" from "root" into the empty subgraph "slice". Assumes
  // the partition is a valid sub-DAG, and replaces it witha single
  // tfl.custom_op in "root". A reference to that op is returned.
  static LiteRtOp SlicePartitionFromGraph(LiteRtSubgraphT& root,
                                          LiteRtSubgraph slice,
                                          std::vector<LiteRtOp>& partition);

 private:
  explicit GraphSlicer(LiteRtSubgraph slice) : slice_(slice) {}

  void CloneInto(const LiteRtOpT& op);

  void RerouteTensorsThroughCustomOp(const LiteRtSubgraphT& root);

  LiteRtSubgraph slice_;
  // Maps tensor in old subgraph to tensor in new subgraph.
  InsertOrderMap<LiteRtTensor, LiteRtTensor> tensor_map_;
  LiteRtOp dispatch_op_ = nullptr;
};

LiteRtOp GraphSlicer::SlicePartitionFromGraph(
    LiteRtSubgraphT& root, LiteRtSubgraph slice,
    std::vector<LiteRtOp>& partition) {
  GraphSlicer slicer(slice);

  // Register input tensors of the sliced partition WRT to their original order
  // in the root subgraph. This ensures the order of input tensors of the
  // later outlined custom op is the same as the order of input tensors of the
  // GraphInputs.
  absl::flat_hash_set<LiteRtTensor> used_tensors;

  // Get all tensors used in the partition.
  for (auto* op : partition) {
    used_tensors.insert(op->Inputs().cbegin(), op->Inputs().cend());
  }
  for (auto* old_input : root.Inputs()) {
    if (used_tensors.contains(old_input)) {
      auto* new_input = &MakeClone(*slicer.slice_, *old_input);
      slicer.slice_->Inputs().push_back(new_input);
      slicer.tensor_map_.InsertOrAssign(old_input, new_input);
    }
  }

  for (auto* op : partition) {
    slicer.CloneInto(*op);
  }

  for (auto* op : partition) {
    Drop(*op);
  }

  // Reuse the storage from the last op in partition to maintain
  // topological order.
  slicer.dispatch_op_ = partition.back();

  ABSL_DCHECK(slicer.dispatch_op_->Inputs().empty());
  ABSL_DCHECK(slicer.dispatch_op_->Outputs().empty());
  MakeDispatchOp(*slicer.dispatch_op_);
  slicer.RerouteTensorsThroughCustomOp(root);

  DCE(root);

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

void GraphSlicer::CloneInto(const LiteRtOpT& old_op) {
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
      new_input = &MakeClone(*slice_, *old_input);
      if (!IsConstant(*new_input)) {
        slice_->Inputs().push_back(new_input);
      }

      tensor_map_.InsertOrAssign(old_input, new_input);
    }

    AttachInput(new_input, new_op);
  }

  for (int i = 0; i < old_op.NumOutputs(); ++i) {
    auto* old_output = old_op.Outputs().at(i);
    auto* new_output = &MakeClone(*slice_, *old_output);
    AttachOutput(new_output, new_op);

    // Update the values defined in scope of the new subgraph.
    tensor_map_.InsertOrAssign(old_output, new_output);
  }
}

}  // namespace

std::vector<std::vector<LiteRtOp>> GroupPartitions(
    const std::vector<LiteRtOpWithPartitionIndex>& ops) {
  return DisjointSets::GetPartitionsFromFlatList(ops);
}

LiteRtOp OutlinePartition(LiteRtSubgraphT& root, LiteRtSubgraph slice,
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
  LiteRtSubgraphT* main_subgraph;
  for (auto& subgraph : model.Subgraphs()) {
    for (auto& op : subgraph->Ops()) {
      if (op == &destination_op) {
        main_subgraph = subgraph;
        break;
      }
    }
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
      cloned_op->Outputs().push_back(cloned_output);
    }
  }

  // Connect the input tensor of the composite op to users of the input
  // tensors of the decomp subgraph.
  for (int input_ind = 0; input_ind < destination_op.Inputs().size();
       ++input_ind) {
    auto& decomp_input_tensor = source_subgraph->Input(input_ind);
    auto users = decomp_input_tensor.Users();
    auto user_args = decomp_input_tensor.UserArgInds();
    for (int user_ind = 0; user_ind < users.size(); ++user_ind) {
      auto& cloned_op = op_map_.Find(users.at(user_ind))->get().second;
      cloned_op->Inputs().at(user_args.at(user_ind)) =
          &destination_op.Input(input_ind);
      // For each input tensor of the original composite op, remove
      // the use of the original composite op input, since the composite op
      // will be removed.
      for (int input_user_ind = 0;
           input_user_ind < destination_op.Input(input_ind).NumUses();
           ++input_user_ind) {
        if (destination_op.Input(input_ind).Users().at(input_user_ind) ==
            &destination_op) {
          destination_op.Input(input_ind).RemoveUse(input_user_ind);
        }
      }
      // Update the user list of the original composite op input tensors.
      destination_op.Input(input_ind).Users().push_back(cloned_op);
      destination_op.Input(input_ind).UserArgInds().push_back(
          user_args.at(user_ind));
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
