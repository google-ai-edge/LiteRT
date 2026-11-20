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
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_model_predicates.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/graph_validation.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_serialize.h"
#include "litert/test/load_test_model.h"

namespace litert::internal {
namespace {

bool IsTopologicallySorted(const LiteRtSubgraphT& subgraph) {
  absl::flat_hash_set<const LiteRtTensorT*> ready_tensors;
  for (const auto* in : subgraph.Inputs()) {
    ready_tensors.insert(in);
  }
  for (const auto* tensor : subgraph.Tensors()) {
    if (IsConstant(*tensor)) {
      ready_tensors.insert(tensor);
    }
  }

  for (const auto& op : subgraph.Ops()) {
    for (const auto* input : op->Inputs()) {
      if (input == nullptr) continue;
      if (!ready_tensors.contains(input)) {
        return false;
      }
    }
    for (const auto* output : op->Outputs()) {
      ready_tensors.insert(output);
    }
  }
  return true;
}

using PartitionFunc = decltype(&GroupPartitions);
class PartitionTest : public ::testing::TestWithParam<PartitionFunc> {};

INSTANTIATE_TEST_SUITE_P(PartitionStrategies, PartitionTest,
                         ::testing::Values(&GroupPartitions, &GroupPartitionsV2,
                                           &GroupPartitionsAcyclic));

TEST_P(PartitionTest, SimpleMultiOp) {
  auto model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  {
    std::vector<LiteRtOpWithPartitionIndex> selected_ops;
    selected_ops.push_back({ops.at(1).Get(), 0});
    selected_ops.push_back({ops.at(2).Get(), 0});

    auto partitions = GetParam()(selected_ops, subgraph->Get());
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions.front().size(), 2);

    EXPECT_EQ(partitions.front().at(0), selected_ops.at(0).first);
    EXPECT_EQ(partitions.front().at(1), selected_ops.at(1).first);
  }

  {
    std::vector<LiteRtOpWithPartitionIndex> selected_ops;
    selected_ops.push_back({ops.at(1).Get(), 0});
    selected_ops.push_back({ops.at(3).Get(), 0});

    auto partitions = GetParam()(selected_ops, subgraph->Get());
    ASSERT_EQ(partitions.size(), 2);
    ASSERT_EQ(partitions.front().size(), 1);
    ASSERT_EQ(partitions.back().size(), 1);

    auto p1_op_code = partitions.front().front()->OpCode();
    auto p2_op_code = partitions.back().front()->OpCode();

    ASSERT_TRUE((p1_op_code == kLiteRtOpCodeTflMul &&
                 p2_op_code == kLiteRtOpCodeTflAdd) ||
                (p1_op_code == kLiteRtOpCodeTflAdd &&
                 p2_op_code == kLiteRtOpCodeTflMul));
  }

  {
    std::vector<LiteRtOpWithPartitionIndex> selected_ops;

    auto partitions = GetParam()(selected_ops, subgraph->Get());
    ASSERT_EQ(partitions.size(), 0);
  }

  {
    std::vector<LiteRtOpWithPartitionIndex> selected_ops;
    selected_ops.push_back({ops.at(0).Get(), 0});
    selected_ops.push_back({ops.at(1).Get(), 0});
    selected_ops.push_back({ops.at(2).Get(), 0});
    selected_ops.push_back({ops.at(3).Get(), 0});

    auto partitions = GetParam()(selected_ops, subgraph->Get());
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions.front().size(), 4);

    EXPECT_EQ(partitions.front().at(0), selected_ops.at(0).first);
    EXPECT_EQ(partitions.front().at(1), selected_ops.at(1).first);
    EXPECT_EQ(partitions.front().at(2), selected_ops.at(2).first);
    EXPECT_EQ(partitions.front().at(3), selected_ops.at(3).first);
  }
}

TEST(TestSliceSubgraphSimpleMultiOp, OnePartition) {
  auto model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  std::vector<LiteRtOp> partition;
  partition.push_back(ops.at(1).Get());
  partition.push_back(ops.at(2).Get());

  auto sliced_graph = litert::Subgraph(&model.Get()->EmplaceSubgraph());
  LITERT_ASSIGN_OR_ABORT(
      auto* dispatch_op,
      OutlinePartition(*subgraph->Get(), sliced_graph.Get(), partition));

  const auto& internal_sliced = *sliced_graph.Get();
  ASSERT_TRUE(ValidateSubgraphIO(internal_sliced));
  ASSERT_TRUE(ValidateLocalTopology(internal_sliced.Ops().cbegin(),
                                    internal_sliced.Ops().cend()));

  auto edited_subgraph_ops = subgraph->Ops();

  ASSERT_EQ(edited_subgraph_ops.size(), 3);
  ASSERT_EQ(edited_subgraph_ops.at(0).Code(), kLiteRtOpCodeTflAdd);
  ASSERT_EQ(edited_subgraph_ops.at(1).Code(), kLiteRtOpCodeTflCustom);
  ASSERT_EQ(edited_subgraph_ops.at(2).Code(), kLiteRtOpCodeTflAdd);

  auto sliced_subgraph_ops = sliced_graph.Ops();

  ASSERT_EQ(sliced_subgraph_ops.size(), 2);
  ASSERT_EQ(sliced_subgraph_ops[0].Code(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(sliced_subgraph_ops[1].Code(), kLiteRtOpCodeTflMul);

  ASSERT_EQ(dispatch_op, edited_subgraph_ops.at(1).Get());
  const Op hal_call(dispatch_op);

  {
    const auto dispatch_op_ins = hal_call.Inputs();

    ASSERT_EQ(dispatch_op_ins.size(), 1);

    auto hal_input_defining_op = dispatch_op_ins.front().DefiningOp();
    ASSERT_EQ(hal_input_defining_op->op, edited_subgraph_ops.at(0).Get());
    ASSERT_EQ(hal_input_defining_op->op_output_index, 0);

    const auto sliced_subgraph_inputs = sliced_graph.Inputs();

    ASSERT_EQ(sliced_subgraph_inputs.size(), 1);

    ASSERT_TRUE(MatchUses(sliced_subgraph_inputs.front(),
                          {UseInfo{sliced_subgraph_ops.front().Code(), 0},
                           UseInfo{sliced_subgraph_ops.front().Code(), 0}}));
    ASSERT_TRUE(sliced_subgraph_inputs.front().IsSubgraphInput());
  }

  {
    const auto hal_call_outs = hal_call.Outputs();
    ASSERT_EQ(hal_call_outs.size(), 1);
    const auto& hal_call_out = hal_call_outs.front();

    ASSERT_TRUE(MatchUses(hal_call_out,
                          {UseInfo{edited_subgraph_ops.back().Code(), 0},
                           UseInfo{edited_subgraph_ops.back().Code(), 1}}));

    auto sliced_subgraph_outputs = sliced_graph.Outputs();

    ASSERT_EQ(sliced_subgraph_outputs.size(), 1);

    const auto defining_op = sliced_subgraph_outputs.front().DefiningOp();
    ASSERT_EQ(defining_op->op, sliced_subgraph_ops.back().Get());
    ASSERT_EQ(defining_op->op_output_index, 0);

    ASSERT_TRUE(sliced_subgraph_outputs.front().Uses().empty());
  }
}

TEST(TestSliceSubgraphSimpleMultiOp, TwoPartitions) {
  auto model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  std::vector<LiteRtOp> partition_1;
  partition_1.push_back(ops.at(0).Get());

  auto sliced_graph_1 = litert::Subgraph(&model.Get()->EmplaceSubgraph());
  LITERT_ABORT_IF_ERROR(
      OutlinePartition(*(subgraph->Get()), sliced_graph_1.Get(), partition_1));

  const auto& internal_slice_1 = *sliced_graph_1.Get();
  ASSERT_TRUE(ValidateSubgraphIO(internal_slice_1));
  ASSERT_TRUE(ValidateLocalTopology(internal_slice_1.Ops().cbegin(),
                                    internal_slice_1.Ops().cend()));

  std::vector<LiteRtOp> partition_2;
  partition_2.push_back(ops.at(2).Get());
  partition_2.push_back(ops.at(3).Get());

  auto sliced_graph_2 = litert::Subgraph(&model.Get()->EmplaceSubgraph());
  LITERT_ABORT_IF_ERROR(
      OutlinePartition(*(subgraph->Get()), sliced_graph_2.Get(), partition_2));

  const auto& internal_slice_2 = *sliced_graph_2.Get();
  ASSERT_TRUE(ValidateSubgraphIO(internal_slice_2));
  ASSERT_TRUE(ValidateLocalTopology(internal_slice_2.Ops().cbegin(),
                                    internal_slice_2.Ops().cend()));

  auto edited_subgraph_ops = subgraph->Ops();

  ASSERT_EQ(edited_subgraph_ops.size(), 3);
  ASSERT_EQ(edited_subgraph_ops.at(0).Code(), kLiteRtOpCodeTflCustom);
  ASSERT_EQ(edited_subgraph_ops.at(1).Code(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(edited_subgraph_ops.at(2).Code(), kLiteRtOpCodeTflCustom);

  {
    auto sliced_ops = sliced_graph_1.Ops();

    ASSERT_EQ(sliced_ops.size(), 1);
    ASSERT_EQ(sliced_ops.at(0).Code(), kLiteRtOpCodeTflAdd);
  }

  {
    auto sliced_ops = sliced_graph_2.Ops();

    ASSERT_EQ(sliced_ops.size(), 2);
    ASSERT_EQ(sliced_ops.at(0).Code(), kLiteRtOpCodeTflMul);
    ASSERT_EQ(sliced_ops.at(1).Code(), kLiteRtOpCodeTflAdd);
  }
}

TEST(TestSliceSubgraphTopoSortFork, ForkPartition) {
  auto model = litert::testing::LoadTestFileModel("topo_sort_fork.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  // We expect ops to be:
  // 0: Add (A)
  // 1: Mul (B)
  // 2: Mul (C)
  // 3: Add (D)
  // 4: Add (E)
  ASSERT_EQ(ops.size(), 5);
  EXPECT_EQ(ops.at(0).Code(), kLiteRtOpCodeTflAdd);  // A
  EXPECT_EQ(ops.at(1).Code(), kLiteRtOpCodeTflMul);  // B
  EXPECT_EQ(ops.at(2).Code(), kLiteRtOpCodeTflMul);  // C
  EXPECT_EQ(ops.at(3).Code(), kLiteRtOpCodeTflAdd);  // D
  EXPECT_EQ(ops.at(4).Code(), kLiteRtOpCodeTflAdd);  // E

  auto* op_d = ops.at(3).Get();
  auto* op_e = ops.at(4).Get();

  std::vector<LiteRtOp> partition;
  partition.push_back(ops.at(0).Get());  // A
  partition.push_back(ops.at(1).Get());  // B
  partition.push_back(ops.at(2).Get());  // C

  auto sliced_graph = litert::Subgraph(&model.Get()->EmplaceSubgraph());
  LITERT_ASSIGN_OR_ABORT(
      auto* dispatch_op,
      OutlinePartition(*subgraph->Get(), sliced_graph.Get(), partition));

  const auto& internal_sliced = *sliced_graph.Get();
  // ASSERT_TRUE(ValidateSubgraphIO(internal_sliced));
  ASSERT_TRUE(ValidateLocalTopology(internal_sliced.Ops().cbegin(),
                                    internal_sliced.Ops().cend()));

  auto edited_subgraph_ops = subgraph->Ops();
  ASSERT_EQ(edited_subgraph_ops.size(), 3);

  const Op hal_call(dispatch_op);
  const auto hal_call_outs = hal_call.Outputs();

  // We expect 2 outputs because %1 (from B) is used by E (outside)
  // and %2 (from C) is used by D (outside).
  ASSERT_EQ(hal_call_outs.size(), 2);

  bool connected_to_e = false;
  bool connected_to_d = false;

  for (const auto& out : hal_call_outs) {
    for (const auto& use : out.Uses()) {
      if (use.user.Get() == op_e) {
        connected_to_e = true;
      }
      if (use.user.Get() == op_d) {
        connected_to_d = true;
      }
    }
  }

  EXPECT_TRUE(connected_to_e)
      << "Output of composite op is not connected to E!";
  EXPECT_TRUE(connected_to_d)
      << "Output of composite op is not connected to D!";
  EXPECT_TRUE(IsTopologicallySorted(*subgraph->Get()));
}

TEST(TestSliceSubgraphTopoSortForkSplit, ForkSplitPartition) {
  auto model =
      litert::testing::LoadTestFileModel("topo_sort_fork_split.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  LiteRtOp op_a = nullptr;
  LiteRtOp op_b = nullptr;
  LiteRtOp op_c = nullptr;
  LiteRtOp op_d = nullptr;
  LiteRtOp op_e = nullptr;

  for (const auto& op_ref : ops) {
    if (op_ref.Code() == kLiteRtOpCodeTflAdd) {
      auto* op = op_ref.Get();
      if (op->Inputs()[0]->DefiningOp() == nullptr &&
          !IsConstant(*op->Inputs()[0])) {
        op_a = op;
      }
    } else if (op_ref.Code() == kLiteRtOpCodeTflSplit) {
      op_b = op_ref.Get();
    } else if (op_ref.Code() == kLiteRtOpCodeTflMul) {
      op_c = op_ref.Get();
    }
  }

  ASSERT_TRUE(op_a);
  ASSERT_TRUE(op_b);
  ASSERT_TRUE(op_c);

  auto* c_out = op_c->Outputs()[0];
  auto* b_out_1 = op_b->Outputs()[1];

  for (const auto& op_ref : ops) {
    if (op_ref.Code() == kLiteRtOpCodeTflAdd) {
      auto* op = op_ref.Get();
      if (op == op_a) continue;
      if (op->Inputs()[0] == c_out || op->Inputs()[1] == c_out) {
        op_d = op;
      } else if (op->Inputs()[0] == b_out_1 || op->Inputs()[1] == b_out_1) {
        op_e = op;
      }
    }
  }

  ASSERT_TRUE(op_d);
  ASSERT_TRUE(op_e);

  std::vector<LiteRtOp> partition = {op_a, op_b, op_c};

  auto sliced_graph = litert::Subgraph(&model.Get()->EmplaceSubgraph());
  LITERT_ASSIGN_OR_ABORT(
      auto* dispatch_op,
      OutlinePartition(*subgraph->Get(), sliced_graph.Get(), partition));

  const auto& internal_sliced = *sliced_graph.Get();
  ASSERT_TRUE(ValidateSubgraphIO(internal_sliced));

  const Op hal_call(dispatch_op);
  const auto hal_call_outs = hal_call.Outputs();
  ASSERT_EQ(hal_call_outs.size(), 2);

  bool connected_to_e = false;
  bool connected_to_d = false;

  for (const auto& out : hal_call_outs) {
    for (const auto& use : out.Uses()) {
      if (use.user.Get() == op_e) {
        connected_to_e = true;
      }
      if (use.user.Get() == op_d) {
        connected_to_d = true;
      }
    }
  }

  EXPECT_TRUE(connected_to_e)
      << "Output of composite op is not connected to E!";
  EXPECT_TRUE(connected_to_d)
      << "Output of composite op is not connected to D!";
  EXPECT_TRUE(IsTopologicallySorted(*subgraph->Get()));
}

TEST(TestSliceSubgraphTopoSortForkViolation, ForkPartitionViolation) {
  auto model =
      litert::testing::LoadTestFileModel("topo_sort_fork_violation.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  ASSERT_EQ(ops.size(), 5);
  EXPECT_EQ(ops.at(0).Code(), kLiteRtOpCodeTflAdd);  // A
  EXPECT_EQ(ops.at(1).Code(), kLiteRtOpCodeTflMul);  // B
  EXPECT_EQ(ops.at(2).Code(), kLiteRtOpCodeTflAdd);  // E
  EXPECT_EQ(ops.at(3).Code(), kLiteRtOpCodeTflMul);  // C
  EXPECT_EQ(ops.at(4).Code(), kLiteRtOpCodeTflAdd);  // D

  auto* op_e = ops.at(2).Get();
  auto* op_d = ops.at(4).Get();

  std::vector<LiteRtOp> partition;
  partition.push_back(ops.at(0).Get());  // A
  partition.push_back(ops.at(1).Get());  // B
  partition.push_back(ops.at(3).Get());  // C

  auto sliced_graph = litert::Subgraph(&model.Get()->EmplaceSubgraph());
  LITERT_ASSIGN_OR_ABORT(
      auto* dispatch_op,
      OutlinePartition(*subgraph->Get(), sliced_graph.Get(), partition));

  const auto& internal_sliced = *sliced_graph.Get();
  // ASSERT_TRUE(ValidateSubgraphIO(internal_sliced));
  ASSERT_TRUE(ValidateLocalTopology(internal_sliced.Ops().cbegin(),
                                    internal_sliced.Ops().cend()));

  auto edited_subgraph_ops = subgraph->Ops();
  ASSERT_EQ(edited_subgraph_ops.size(), 3);

  const Op hal_call(dispatch_op);
  const auto hal_call_outs = hal_call.Outputs();

  ASSERT_EQ(hal_call_outs.size(), 2);

  bool connected_to_e = false;
  bool connected_to_d = false;

  for (const auto& out : hal_call_outs) {
    for (const auto& use : out.Uses()) {
      if (use.user.Get() == op_e) {
        connected_to_e = true;
      }
      if (use.user.Get() == op_d) {
        connected_to_d = true;
      }
    }
  }

  EXPECT_TRUE(connected_to_e)
      << "Output of composite op is not connected to E!";
  EXPECT_TRUE(connected_to_d)
      << "Output of composite op is not connected to D!";

  EXPECT_TRUE(IsTopologicallySorted(*subgraph->Get()))
      << "Main subgraph is not topologically sorted!";
}

TEST(TestSliceSubgraphTopoSortInterleaved, InterleavedPartition) {
  auto model =
      litert::testing::LoadTestFileModel("topo_sort_interleaved.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  ASSERT_EQ(ops.size(), 6);
  EXPECT_EQ(ops.at(0).Code(), kLiteRtOpCodeTflAdd);  // A1
  EXPECT_EQ(ops.at(1).Code(), kLiteRtOpCodeTflMul);  // B1
  EXPECT_EQ(ops.at(2).Code(), kLiteRtOpCodeTflAdd);  // C1
  EXPECT_EQ(ops.at(3).Code(), kLiteRtOpCodeTflAdd);  // A2
  EXPECT_EQ(ops.at(4).Code(), kLiteRtOpCodeTflMul);  // B2
  EXPECT_EQ(ops.at(5).Code(), kLiteRtOpCodeTflAdd);  // C2

  std::vector<LiteRtOp> partition;
  partition.push_back(ops.at(1).Get());  // B1
  partition.push_back(ops.at(4).Get());  // B2

  auto sliced_graph = litert::Subgraph(&model.Get()->EmplaceSubgraph());
  LITERT_ABORT_IF_ERROR(
      OutlinePartition(*subgraph->Get(), sliced_graph.Get(), partition));

  const auto& internal_sliced = *sliced_graph.Get();
  ASSERT_TRUE(ValidateSubgraphIO(internal_sliced));

  EXPECT_TRUE(IsTopologicallySorted(*subgraph->Get()))
      << "Main subgraph is not topologically sorted!";
}

TEST(OutlinePartitionTest, SlicesBlockWiseQuantizedOp) {
  LiteRtModelT model;
  auto& main_sg = model.EmplaceSubgraph();

  auto& input = main_sg.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 16}));
  main_sg.Inputs().push_back(&input);

  auto& output = main_sg.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 32}));
  main_sg.Outputs().push_back(&output);

  auto& scales = main_sg.EmplaceTensor();
  scales.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {32, 1}));
  OwningBufferRef<uint8_t> scales_data(sizeof(float) * 32);
  SetWeightsFromOwnedBuffer(scales.Weights(), std::move(scales_data));

  auto& weight = main_sg.EmplaceTensor();
  weight.SetType(MakeRankedTensorType(kLiteRtElementTypeInt4, {32, 16}));
  OwningBufferRef<uint8_t> weight_data(32 * 16 / 2);
  SetWeightsFromOwnedBuffer(weight.Weights(), std::move(weight_data));
  weight.SetQarams(
      MakeBlockWiseQuantization(&scales, nullptr, /*block_size=*/16));

  auto& fc_op = main_sg.EmplaceOp();
  fc_op.SetOpCode(kLiteRtOpCodeTflFullyConnected);
  AttachInput(&input, fc_op);
  AttachInput(&weight, fc_op);
  AttachOutput(&output, fc_op);

  std::vector<LiteRtOp> partition = {&fc_op};

  auto& slice_sg = model.EmplaceSubgraph();
  LITERT_ABORT_IF_ERROR(OutlinePartition(main_sg, &slice_sg, partition));

  EXPECT_EQ(slice_sg.Ops().size(), 1);
  EXPECT_EQ(slice_sg.Ops().front()->OpCode(), kLiteRtOpCodeTflFullyConnected);

  const auto* slice_weight = slice_sg.Ops().front()->Inputs().at(1);
  ASSERT_EQ(slice_weight->Qparams().first, kLiteRtQuantizationBlockWise);
  const auto& bw = slice_weight->Qparams().second.block_wise;
  ASSERT_NE(bw.scales, nullptr);
  EXPECT_NE(bw.scales, &scales);
  EXPECT_EQ(bw.block_size, 16);

  auto serialized = SerializeModel(std::move(model));
  EXPECT_TRUE(serialized.HasValue());
}

TEST_P(PartitionTest, PartitionWithIndex) {
  auto model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  auto subgraph = model.MainSubgraph();
  EXPECT_TRUE(subgraph);

  auto ops = subgraph->Ops();

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  {
    std::vector<LiteRtOpWithPartitionIndex> selected_ops;
    selected_ops.push_back({ops.at(1).Get(), 0});
    selected_ops.push_back({ops.at(2).Get(), 1});

    auto partitions = GetParam()(selected_ops, subgraph->Get());
    ASSERT_EQ(partitions.size(), 2);
    ASSERT_EQ(partitions.front().size(), 1);
    ASSERT_EQ(partitions.back().size(), 1);

    absl::flat_hash_set<LiteRtOp> ops_in_partition;
    for (int i = 0; i < partitions.size(); ++i) {
      for (const auto& op : partitions.at(i)) {
        ops_in_partition.insert(op);
      }
    }
    for (int i = 0; i < partitions.size(); ++i) {
      EXPECT_TRUE(ops_in_partition.contains(selected_ops.at(i).first));
    }
  }

  {
    std::vector<LiteRtOpWithPartitionIndex> selected_ops;
    selected_ops.push_back({ops.at(0).Get(), 1});
    selected_ops.push_back({ops.at(1).Get(), 2});
    selected_ops.push_back({ops.at(2).Get(), 3});
    selected_ops.push_back({ops.at(3).Get(), 4});

    auto partitions = GetParam()(selected_ops, subgraph->Get());
    ASSERT_EQ(partitions.size(), 4);

    absl::flat_hash_set<LiteRtOp> ops_in_partition;
    for (int i = 0; i < partitions.size(); ++i) {
      for (const auto& op : partitions.at(i)) {
        ops_in_partition.insert(op);
      }
    }
    for (int i = 0; i < partitions.size(); ++i) {
      EXPECT_TRUE(ops_in_partition.contains(selected_ops.at(i).first));
    }
  }

  // Guard against `max_partition_index + 1` sentinel wraparound when an op uses
  // `std::numeric_limits<LiteRtParamIndex>::max()` alongside index `0` and
  // unselected ops (`ops.at(0)` and `ops.at(3)`).
  {
    const LiteRtParamIndex max_idx =
        std::numeric_limits<LiteRtParamIndex>::max();
    std::vector<LiteRtOpWithPartitionIndex> selected_ops = {
        {ops.at(1).Get(), 0},
        {ops.at(2).Get(), max_idx},
    };

    auto partitions = GetParam()(selected_ops, subgraph->Get());
    ASSERT_EQ(partitions.size(), 2);
    absl::flat_hash_set<LiteRtOp> ops_in_partition;
    for (const auto& part : partitions) {
      EXPECT_EQ(part.size(), 1);
      ops_in_partition.insert(part.front());
    }
    EXPECT_TRUE(ops_in_partition.contains(ops.at(1).Get()));
    EXPECT_TRUE(ops_in_partition.contains(ops.at(2).Get()));
    EXPECT_FALSE(ops_in_partition.contains(ops.at(0).Get()));
    EXPECT_FALSE(ops_in_partition.contains(ops.at(3).Get()));
  }
}

TEST(TestCompositeInlining, RemapsSignatureOutputAliases) {
  auto model_wrap = testing::LoadTestFileModel("rms_norm_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto& graph = model.Subgraph(0);
  ASSERT_EQ(graph.Outputs().size(), 1);
  ASSERT_FALSE(model.Signatures().empty());
  auto* original = model.Signatures().front();
  auto& alias = model.EmplaceSignature(
      &graph, original->InputNames(),
      std::vector<LiteRtTensor>(graph.Inputs().begin(), graph.Inputs().end()),
      std::vector<std::string>{"output_alias"},
      std::vector<LiteRtTensor>{graph.Outputs().front()}, "alias");
  auto* input = original->GetInputTensor(0);

  ASSERT_TRUE(
      InlineSubgraph(model, graph.Op(0), &model.Subgraph(1)).HasValue());
  EXPECT_EQ(original->GetOutputTensor(0), graph.Outputs().front());
  EXPECT_EQ(alias.GetOutputTensor(0), graph.Outputs().front());
  auto named_output = alias.FindOutputTensor("output_alias");
  ASSERT_TRUE(named_output.HasValue());
  EXPECT_EQ(*named_output, graph.Outputs().front());
  EXPECT_EQ(original->GetInputTensor(0), input);
  EXPECT_EQ(alias.GetInputTensor(0), input);
}

TEST(TestCompositeInlining, FailsIfDestinationOpIsNotInModel) {
  LiteRtModelT model;
  model.EmplaceSubgraph();
  auto& decomp_subgraph = model.EmplaceSubgraph();

  LiteRtModelT other_model;
  auto& orphan_op = other_model.EmplaceSubgraph().EmplaceOp();

  auto result = InlineSubgraph(model, orphan_op, &decomp_subgraph);
  ASSERT_FALSE(result.HasValue());
  EXPECT_THAT(
      result.Error().Message(),
      ::testing::HasSubstr("Failed to find the subgraph containing the op"));
}

TEST(TestCompositeInlining, inlineSimpleComposite) {
  auto model_wrap = testing::LoadTestFileModel("rms_norm_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  EXPECT_EQ(model.MainSubgraph()->Ops().size(), 1);

  auto& main_subgraph = model.Subgraph(0);
  auto& decomp_subgraph = model.Subgraph(1);
  auto& decomp_op = main_subgraph.Op(0);
  EXPECT_EQ(main_subgraph.Ops().size(), 1);
  EXPECT_EQ(decomp_subgraph.Ops().size(), 7);

  EXPECT_TRUE(InlineSubgraph(model, decomp_op, &decomp_subgraph).HasValue());
  EXPECT_EQ(model.MainSubgraph()->Ops().size(), 7);
  EXPECT_EQ(model.NumSubgraphs(), 2);

  // Check the topology of the main subgraph and decomp subgraph. Since there is
  // only one composite op in the main subgraph, after inlining, the main
  // subgraph should be the same as the decomp subgraph.
  for (int op_index = 0; op_index < model.MainSubgraph()->Ops().size();
       ++op_index) {
    auto& inlined_op = model.MainSubgraph()->Op(op_index);
    auto& original_op = decomp_subgraph.Op(op_index);
    EXPECT_EQ(inlined_op.OpCode(), original_op.OpCode());
    EXPECT_EQ(inlined_op.Inputs().size(), original_op.Inputs().size());
    for (int input_index = 0; input_index < inlined_op.Inputs().size();
         ++input_index) {
      EXPECT_EQ(inlined_op.Input(input_index).TensorIndex(),
                original_op.Input(input_index).TensorIndex());
    }
    EXPECT_EQ(inlined_op.Outputs().size(), original_op.Outputs().size());
    for (int output_index = 0; output_index < inlined_op.Outputs().size();
         ++output_index) {
      EXPECT_EQ(inlined_op.Output(output_index).TensorIndex(),
                original_op.Output(output_index).TensorIndex());
    }
  }
  for (int tensor_index = 0;
       tensor_index < model.MainSubgraph()->Tensors().size(); ++tensor_index) {
    auto& inlined_tensor = model.MainSubgraph()->Tensor(tensor_index);
    auto& original_tensor = decomp_subgraph.Tensor(tensor_index);
    EXPECT_EQ(inlined_tensor.TensorIndex(), original_tensor.TensorIndex());
    EXPECT_EQ(inlined_tensor.Users().size(), original_tensor.Users().size());
    EXPECT_EQ(inlined_tensor.UserArgInds().size(),
              original_tensor.UserArgInds().size());
    for (int user_arg_index = 0;
         user_arg_index < inlined_tensor.UserArgInds().size();
         ++user_arg_index) {
      EXPECT_EQ(inlined_tensor.UserArgInds().at(user_arg_index),
                original_tensor.UserArgInds().at(user_arg_index));
    }
  }

  // Test serialization.
  auto serialized = SerializeModel(std::move(model));
  EXPECT_TRUE(serialized);
}

// Test inlining a non trivial case, where the input and output tensors of the
// composite op are produced/consumed by other ops in the main subgraph.
TEST(TestCompositeInlining, InlineSimpleComposite2) {
  auto model_wrap = testing::LoadTestFileModel("rms_norm_composite_2.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  auto& main_subgraph = model.Subgraph(0);
  auto& decomp_subgraph = model.Subgraph(1);
  auto& decomp_op = main_subgraph.Op(1);
  EXPECT_EQ(main_subgraph.Ops().size(), 3);
  EXPECT_EQ(decomp_subgraph.Ops().size(), 7);

  EXPECT_TRUE(InlineSubgraph(model, decomp_op, &decomp_subgraph).HasValue());
  EXPECT_EQ(model.MainSubgraph()->Ops().size(), 9);
  EXPECT_EQ(model.MainSubgraph()->Tensors().size(), 14);
  EXPECT_EQ(model.NumSubgraphs(), 2);
  auto serialized = SerializeModel(std::move(model));
  EXPECT_TRUE(serialized);
}

TEST(TestCompositeInlining, inlineSimpleComposite3) {
  auto model_wrap = testing::LoadTestFileModel("rms_norm_composite_3.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  auto& main_subgraph = model.Subgraph(0);
  auto& decomp_subgraph = model.Subgraph(1);
  auto& decomp_op = main_subgraph.Op(1);
  EXPECT_EQ(main_subgraph.Ops().size(), 2);
  EXPECT_EQ(decomp_subgraph.Ops().size(), 7);
  EXPECT_EQ(decomp_subgraph.Tensors().size(), 12);

  EXPECT_TRUE(InlineSubgraph(model, decomp_op, &decomp_subgraph).HasValue());
  EXPECT_EQ(model.MainSubgraph()->Ops().size(), 8);
  EXPECT_EQ(model.NumSubgraphs(), 2);
  auto serialized = SerializeModel(std::move(model));
  EXPECT_TRUE(serialized);
}

TEST(TestCompositeInlining, NonMatchingTensorShapeInlining) {
  auto model_wrap = testing::LoadTestFileModel("multi_subgraph_mul.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  EXPECT_EQ(model.MainSubgraph()->Ops().size(), 1);

  auto& subgraph_1 = model.Subgraph(0);
  auto& subgraph_2 = model.Subgraph(1);
  auto& dest_op = subgraph_1.Op(0);
  ASSERT_EQ(InlineSubgraph(model, dest_op, &subgraph_2).Error().Status(),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(TestCompositeInlining, NonMatchingTensorTypeInlining) {
  auto model_wrap = testing::LoadTestFileModel("multi_subgraph_mul_2.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  EXPECT_EQ(model.MainSubgraph()->Ops().size(), 1);

  auto& subgraph_1 = model.Subgraph(0);
  auto& subgraph_2 = model.Subgraph(1);
  auto& dest_op = subgraph_1.Op(0);
  ASSERT_EQ(InlineSubgraph(model, dest_op, &subgraph_2).Error().Status(),
            kLiteRtStatusErrorInvalidArgument);

  auto& subgraph_3 = model.Subgraph(2);
  auto& subgraph_4 = model.Subgraph(3);
  auto& dest_op_2 = subgraph_3.Op(0);
  ASSERT_EQ(InlineSubgraph(model, dest_op_2, &subgraph_4).Error().Status(),
            kLiteRtStatusErrorInvalidArgument);
}

// Builds a model whose main subgraph has a single composite op
// `comp(main_in0, main_in0, main_in1) -> main_out` decomposed into
// `Add(d_in0, d_in1)`. The composite reads `main_in0` twice and passes
// `main_in1` to a decomposition input (`d_in2`) that is never used.
struct DuplicateAndUnusedInputComposite {
  LiteRtModelT model;
  LiteRtSubgraphT* main_sg;
  LiteRtSubgraphT* decomp_sg;
  LiteRtOpT* comp_op;
  LiteRtTensorT* main_in0;
  LiteRtTensorT* main_in1;

  DuplicateAndUnusedInputComposite() {
    main_sg = &model.EmplaceSubgraph();
    decomp_sg = &model.EmplaceSubgraph();
    const auto type = MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 4});

    main_in0 = &main_sg->EmplaceTensor();
    main_in0->SetType(type);
    main_in1 = &main_sg->EmplaceTensor();
    main_in1->SetType(type);
    auto& main_out = main_sg->EmplaceTensor();
    main_out.SetType(type);
    main_sg->Inputs().assign({main_in0, main_in1});
    main_sg->Outputs().assign({&main_out});

    comp_op = &main_sg->EmplaceOp();
    comp_op->SetOpCode(kLiteRtOpCodeShloComposite);
    AttachInput(main_in0, *comp_op);
    AttachInput(main_in0, *comp_op);
    AttachInput(main_in1, *comp_op);
    AttachOutput(&main_out, *comp_op);

    auto& d_in0 = decomp_sg->EmplaceTensor();
    d_in0.SetType(type);
    auto& d_in1 = decomp_sg->EmplaceTensor();
    d_in1.SetType(type);
    auto& d_in2 = decomp_sg->EmplaceTensor();
    d_in2.SetType(type);
    auto& d_out = decomp_sg->EmplaceTensor();
    d_out.SetType(type);
    decomp_sg->Inputs().assign({&d_in0, &d_in1, &d_in2});
    decomp_sg->Outputs().assign({&d_out});

    auto& add_op = decomp_sg->EmplaceOp();
    add_op.SetOpCode(kLiteRtOpCodeTflAdd);
    AttachInput(&d_in0, add_op);
    AttachInput(&d_in1, add_op);
    AttachOutput(&d_out, add_op);
  }
};

TEST(TestCompositeInlining, SetsDefiningOpOnClonedOutputs) {
  DuplicateAndUnusedInputComposite g;
  ASSERT_TRUE(InlineSubgraph(g.model, *g.comp_op, g.decomp_sg));

  ASSERT_EQ(g.main_sg->Ops().size(), 1);
  auto* inlined_add = g.main_sg->Ops().front();
  ASSERT_EQ(inlined_add->OpCode(), kLiteRtOpCodeTflAdd);
  ASSERT_EQ(inlined_add->Outputs().size(), 1);
  auto* cloned_out = inlined_add->Outputs().front();
  EXPECT_EQ(cloned_out->DefiningOp(), inlined_add);
  EXPECT_EQ(cloned_out->DefiningOpOutInd(), 0);
  EXPECT_EQ(g.main_sg->Outputs().front(), cloned_out);
}

TEST(TestCompositeInlining, RemovesAllCompositeUsesOfDuplicatedInput) {
  DuplicateAndUnusedInputComposite g;
  ASSERT_TRUE(InlineSubgraph(g.model, *g.comp_op, g.decomp_sg));

  ASSERT_EQ(g.main_sg->Ops().size(), 1);
  auto* inlined_add = g.main_sg->Ops().front();
  // Exactly the two uses by the inlined op remain; none point at the composite.
  // Guards that each input slot's cleanup is self-contained rather than relying
  // on a later slot to remove a leftover composite use.
  ASSERT_EQ(g.main_in0->NumUses(), 2);
  EXPECT_EQ(g.main_in0->GetUse(0).first, inlined_add);
  EXPECT_EQ(g.main_in0->GetUse(0).second, 0);
  EXPECT_EQ(g.main_in0->GetUse(1).first, inlined_add);
  EXPECT_EQ(g.main_in0->GetUse(1).second, 1);
}

TEST(TestCompositeInlining, RemovesCompositeUseOfInputUnusedByDecomposition) {
  DuplicateAndUnusedInputComposite g;
  ASSERT_TRUE(InlineSubgraph(g.model, *g.comp_op, g.decomp_sg));

  EXPECT_EQ(g.main_in1->NumUses(), 0);
}

// Helper for building synthetic single-subgraph DAGs in unit tests.
struct SyntheticDag {
  LiteRtModelT model;
  LiteRtSubgraphT* subgraph = nullptr;

  SyntheticDag() { subgraph = &model.EmplaceSubgraph(); }

  LiteRtTensorT* AddInput() {
    auto& tensor = subgraph->EmplaceTensor();
    tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 4}));
    subgraph->Inputs().push_back(&tensor);
    return &tensor;
  }

  LiteRtOpT* AddOp(LiteRtOpCode code, std::vector<LiteRtTensorT*> inputs,
                   LiteRtTensorT** out_tensor = nullptr) {
    auto& op = subgraph->EmplaceOp();
    op.SetOpCode(code);
    for (auto* in : inputs) {
      AttachInput(in, op);
    }
    auto& out = subgraph->EmplaceTensor();
    out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 4}));
    AttachOutput(&out, op);
    if (out_tensor != nullptr) {
      *out_tensor = &out;
    }
    return &op;
  }

  void MarkOutput(LiteRtTensorT* tensor) {
    subgraph->Outputs().push_back(tensor);
  }
};

TEST(GroupPartitionsAcyclicTest, SplitsMinimalSkipConnectionCycleTrap) {
  // 3-op DAG: A(NPU) -> B(CPU) -> C(NPU), with skip edge A -> C.
  // GroupPartitions merges {A, C}, creating cycle {A, C} -> B -> {A, C}.
  SyntheticDag dag;
  auto* in = dag.AddInput();
  LiteRtTensorT* t_a = nullptr;
  LiteRtTensorT* t_b = nullptr;
  LiteRtTensorT* t_c = nullptr;
  auto* op_a = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_a);
  dag.AddOp(kLiteRtOpCodeTflAdd, {t_a, t_a}, &t_b);
  auto* op_c = dag.AddOp(kLiteRtOpCodeTflMul, {t_a, t_b}, &t_c);
  dag.MarkOutput(t_c);

  std::vector<LiteRtOpWithPartitionIndex> selected = {{op_a, 0}, {op_c, 0}};

  // Weakly connected merges both into one cyclic partition.
  auto wc = GroupPartitions(selected, dag.subgraph);
  EXPECT_EQ(wc.size(), 1);

  auto acyclic = GroupPartitionsAcyclic(selected, dag.subgraph);
  ASSERT_EQ(acyclic.size(), 2);
  EXPECT_EQ(acyclic[0], std::vector<LiteRtOp>({op_a}));
  EXPECT_EQ(acyclic[1], std::vector<LiteRtOp>({op_c}));

  for (auto& island : acyclic) {
    auto& slice = dag.model.EmplaceSubgraph();
    ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue());
    EXPECT_TRUE(ValidateSubgraphIO(slice));
    EXPECT_TRUE(
        ValidateLocalTopology(slice.Ops().cbegin(), slice.Ops().cend()));
    EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
  }
}

TEST(GroupPartitionsAcyclicTest, SplitsDiamondCycleTrap) {
  // 4-op DAG: A(NPU) -> B(CPU) -> C(NPU) -> D(NPU), plus skip edge A -> D.
  SyntheticDag dag;
  auto* in = dag.AddInput();
  LiteRtTensorT* t_a = nullptr;
  LiteRtTensorT* t_b = nullptr;
  LiteRtTensorT* t_c = nullptr;
  LiteRtTensorT* t_d = nullptr;
  auto* op_a = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_a);
  dag.AddOp(kLiteRtOpCodeTflAdd, {t_a, t_a}, &t_b);
  auto* op_c = dag.AddOp(kLiteRtOpCodeTflMul, {t_b, t_b}, &t_c);
  auto* op_d = dag.AddOp(kLiteRtOpCodeTflMul, {t_a, t_c}, &t_d);
  dag.MarkOutput(t_d);

  std::vector<LiteRtOpWithPartitionIndex> selected = {
      {op_a, 0}, {op_c, 0}, {op_d, 0}};
  auto acyclic = GroupPartitionsAcyclic(selected, dag.subgraph);
  ASSERT_EQ(acyclic.size(), 2);
  EXPECT_EQ(acyclic[0], std::vector<LiteRtOp>({op_a}));
  EXPECT_EQ(acyclic[1], std::vector<LiteRtOp>({op_c, op_d}));

  for (auto& island : acyclic) {
    auto& slice = dag.model.EmplaceSubgraph();
    ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue());
    EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
  }
}

TEST(GroupPartitionsAcyclicTest, AvoidsDirectEdgeCrossLayerTrap) {
  // 6-op DAG: A(NPU) -> B(NPU), A -> c1(CPU) -> u(NPU), v(NPU) -> c2(CPU) -> B.
  // Merging A and B greedily traps {v, u}, yielding 3 NPU partitions.
  // Optimal acyclic epochs merge {A, v} and {B, u} into 2 partitions.
  SyntheticDag dag;
  auto* in = dag.AddInput();
  LiteRtTensorT* t_a = nullptr;
  LiteRtTensorT* t_v = nullptr;
  LiteRtTensorT* t_c1 = nullptr;
  LiteRtTensorT* t_c2 = nullptr;
  LiteRtTensorT* t_b = nullptr;
  LiteRtTensorT* t_u = nullptr;
  auto* op_a = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_a);
  auto* op_v = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_v);
  dag.AddOp(kLiteRtOpCodeTflAdd, {t_a, t_a}, &t_c1);
  dag.AddOp(kLiteRtOpCodeTflAdd, {t_v, t_v}, &t_c2);
  auto* op_b = dag.AddOp(kLiteRtOpCodeTflMul, {t_a, t_c2}, &t_b);
  auto* op_u = dag.AddOp(kLiteRtOpCodeTflMul, {t_c1, t_c1}, &t_u);
  dag.MarkOutput(t_b);
  dag.MarkOutput(t_u);

  std::vector<LiteRtOpWithPartitionIndex> selected = {
      {op_a, 0}, {op_v, 0}, {op_b, 0}, {op_u, 0}};
  auto acyclic = GroupPartitionsAcyclic(selected, dag.subgraph);
  ASSERT_EQ(acyclic.size(), 2);
  EXPECT_THAT(acyclic[0], ::testing::UnorderedElementsAre(op_a, op_v));
  EXPECT_THAT(acyclic[1], ::testing::UnorderedElementsAre(op_b, op_u));

  for (auto& island : acyclic) {
    auto& slice = dag.model.EmplaceSubgraph();
    ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue());
    EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
  }
}

TEST(GroupPartitionsAcyclicTest, ReordersIndependentCpuOpsToEliminateCuts) {
  // Interleaved execution order:
  //   0: NPU1(in)
  //   1: CPU0(in)    (independent of NPU1!)
  //   2: NPU2(NPU1)
  //   3: CPU1(CPU0)  (independent of NPU1, NPU2!)
  //   4: NPU3(NPU2, CPU1)
  // GroupPartitionsV2 cuts into 3 NPU partitions; Acyclic merges all 3 into 1!
  SyntheticDag dag;
  auto* in = dag.AddInput();
  LiteRtTensorT* t_npu1 = nullptr;
  LiteRtTensorT* t_cpu0 = nullptr;
  LiteRtTensorT* t_npu2 = nullptr;
  LiteRtTensorT* t_cpu1 = nullptr;
  LiteRtTensorT* t_npu3 = nullptr;
  auto* op_npu1 = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_npu1);
  dag.AddOp(kLiteRtOpCodeTflAdd, {in, in}, &t_cpu0);
  auto* op_npu2 = dag.AddOp(kLiteRtOpCodeTflMul, {t_npu1, t_npu1}, &t_npu2);
  dag.AddOp(kLiteRtOpCodeTflAdd, {t_cpu0, t_cpu0}, &t_cpu1);
  auto* op_npu3 = dag.AddOp(kLiteRtOpCodeTflMul, {t_npu2, t_cpu1}, &t_npu3);
  dag.MarkOutput(t_npu3);

  // Also test out-of-order selected_ops input vector.
  std::vector<LiteRtOpWithPartitionIndex> selected = {
      {op_npu3, 0}, {op_npu1, 0}, {op_npu2, 0}};

  auto v2 = GroupPartitionsV2(selected, dag.subgraph);
  EXPECT_EQ(v2.size(), 3);

  auto acyclic = GroupPartitionsAcyclic(selected, dag.subgraph);
  ASSERT_EQ(acyclic.size(), 1);
  EXPECT_EQ(acyclic[0], std::vector<LiteRtOp>({op_npu1, op_npu2, op_npu3}));

  auto& slice = dag.model.EmplaceSubgraph();
  ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, acyclic[0]).HasValue());
  EXPECT_TRUE(ValidateSubgraphIO(slice));
  EXPECT_TRUE(ValidateLocalTopology(slice.Ops().cbegin(), slice.Ops().cend()));
  EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
}

TEST(GroupPartitionsAcyclicTest, MultiIndexContractionStaysAcyclicAndMerges) {
  // Multi-index DAG:
  //   a1(0) -> a2(0) -> b1(1)
  //   b2(1) -> a3(0)
  // {a1, a2, a3} can merge into one partition of index 0, leaving b1 and b2
  // separate (since b2 -> {a1, a2, a3} -> b1 blocks merging b1 and b2).
  SyntheticDag dag;
  auto* in = dag.AddInput();
  LiteRtTensorT* t_a1 = nullptr;
  LiteRtTensorT* t_a2 = nullptr;
  LiteRtTensorT* t_b1 = nullptr;
  LiteRtTensorT* t_b2 = nullptr;
  LiteRtTensorT* t_a3 = nullptr;
  auto* op_a1 = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_a1);
  auto* op_a2 = dag.AddOp(kLiteRtOpCodeTflMul, {t_a1, t_a1}, &t_a2);
  auto* op_b1 = dag.AddOp(kLiteRtOpCodeTflSub, {t_a2, t_a2}, &t_b1);
  auto* op_b2 = dag.AddOp(kLiteRtOpCodeTflSub, {in, in}, &t_b2);
  auto* op_a3 = dag.AddOp(kLiteRtOpCodeTflMul, {t_b2, t_b2}, &t_a3);
  dag.MarkOutput(t_b1);
  dag.MarkOutput(t_a3);

  std::vector<LiteRtOpWithPartitionIndex> selected = {
      {op_a1, 0}, {op_a2, 0}, {op_b1, 1}, {op_b2, 1}, {op_a3, 0}};
  auto acyclic = GroupPartitionsAcyclic(selected, dag.subgraph);
  ASSERT_EQ(acyclic.size(), 3);

  for (auto& island : acyclic) {
    auto& slice = dag.model.EmplaceSubgraph();
    ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue());
    EXPECT_TRUE(ValidateSubgraphIO(slice));
    EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
  }
}

TEST(GroupPartitionsAcyclicTest, MultiIndexCrossCyclePreventsMutualMerge) {
  // Cross-chain DAG where each same-index pair is individually convex in G,
  // yet merging both creates a cycle:
  //   a1(0) -> b1(1),  b2(1) -> a2(0)
  // Only one of {a1, a2} or {b1, b2} may merge (yielding 3 partitions total).
  SyntheticDag dag;
  auto* in = dag.AddInput();
  LiteRtTensorT* t_a1 = nullptr;
  LiteRtTensorT* t_b1 = nullptr;
  LiteRtTensorT* t_b2 = nullptr;
  LiteRtTensorT* t_a2 = nullptr;
  auto* op_a1 = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_a1);
  auto* op_b1 = dag.AddOp(kLiteRtOpCodeTflSub, {t_a1, t_a1}, &t_b1);
  auto* op_b2 = dag.AddOp(kLiteRtOpCodeTflSub, {in, in}, &t_b2);
  auto* op_a2 = dag.AddOp(kLiteRtOpCodeTflMul, {t_b2, t_b2}, &t_a2);
  dag.MarkOutput(t_b1);
  dag.MarkOutput(t_a2);

  std::vector<LiteRtOpWithPartitionIndex> selected = {
      {op_a1, 0}, {op_b1, 1}, {op_b2, 1}, {op_a2, 0}};
  auto acyclic = GroupPartitionsAcyclic(selected, dag.subgraph);
  ASSERT_EQ(acyclic.size(), 3);

  for (auto& island : acyclic) {
    auto& slice = dag.model.EmplaceSubgraph();
    ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue());
    EXPECT_TRUE(ValidateSubgraphIO(slice));
    EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
  }
}

TEST(GroupPartitionsAcyclicTest, HandlesPreEpochAndTrailingUnselectedOps) {
  // Leading unselected ops (epoch -1) and trailing unselected ops after the
  // last selected op must not introduce extra partitions or cuts.
  SyntheticDag dag;
  auto* in = dag.AddInput();
  LiteRtTensorT* t_pre1 = nullptr;
  LiteRtTensorT* t_pre2 = nullptr;
  LiteRtTensorT* t_npu1 = nullptr;
  LiteRtTensorT* t_npu2 = nullptr;
  LiteRtTensorT* t_post1 = nullptr;
  LiteRtTensorT* t_post2 = nullptr;
  dag.AddOp(kLiteRtOpCodeTflAdd, {in, in}, &t_pre1);
  dag.AddOp(kLiteRtOpCodeTflAdd, {t_pre1, t_pre1}, &t_pre2);
  auto* op_npu1 = dag.AddOp(kLiteRtOpCodeTflMul, {t_pre2, t_pre2}, &t_npu1);
  auto* op_npu2 = dag.AddOp(kLiteRtOpCodeTflMul, {t_npu1, t_npu1}, &t_npu2);
  dag.AddOp(kLiteRtOpCodeTflAdd, {t_npu2, t_npu2}, &t_post1);
  dag.AddOp(kLiteRtOpCodeTflAdd, {t_post1, t_post1}, &t_post2);
  dag.MarkOutput(t_post2);

  std::vector<LiteRtOpWithPartitionIndex> selected = {{op_npu1, 0},
                                                      {op_npu2, 0}};
  auto acyclic = GroupPartitionsAcyclic(selected, dag.subgraph);
  ASSERT_EQ(acyclic.size(), 1);
  EXPECT_EQ(acyclic[0], std::vector<LiteRtOp>({op_npu1, op_npu2}));

  auto& slice = dag.model.EmplaceSubgraph();
  ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, acyclic[0]).HasValue());
  EXPECT_TRUE(ValidateSubgraphIO(slice));
  EXPECT_TRUE(ValidateLocalTopology(slice.Ops().cbegin(), slice.Ops().cend()));
  EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
}

TEST(GroupPartitionsAcyclicTest, MultiIndexDirectEdgeMergePreservesAcyclicity) {
  // Direct edge a(0) -> b(0) separated across Phase 1 epochs solely by x(1) ->
  // b(0). No intermediate node lies strictly between a and b, so Phase 2 merges
  // {a, b} along the direct edge while keeping x separate.
  SyntheticDag dag;
  auto* in = dag.AddInput();
  LiteRtTensorT* t_a = nullptr;
  LiteRtTensorT* t_x = nullptr;
  LiteRtTensorT* t_b = nullptr;
  auto* op_a = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_a);
  auto* op_x = dag.AddOp(kLiteRtOpCodeTflSub, {in, in}, &t_x);
  auto* op_b = dag.AddOp(kLiteRtOpCodeTflMul, {t_a, t_x}, &t_b);
  dag.MarkOutput(t_b);

  std::vector<LiteRtOpWithPartitionIndex> selected = {
      {op_a, 0}, {op_x, 1}, {op_b, 0}};
  auto acyclic = GroupPartitionsAcyclic(selected, dag.subgraph);
  ASSERT_EQ(acyclic.size(), 2);
  EXPECT_THAT(acyclic, ::testing::UnorderedElementsAre(
                           std::vector<LiteRtOp>({op_x}),
                           std::vector<LiteRtOp>({op_a, op_b})));

  for (auto& island : acyclic) {
    auto& slice = dag.model.EmplaceSubgraph();
    ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue());
    EXPECT_TRUE(ValidateSubgraphIO(slice));
    EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
  }
}

TEST(GroupPartitionsAcyclicTest,
     MultiIndexInterleavedOpsDeduplicatesQuotientEdges) {
  // Interleave ops in subgraph storage order that share quotient block qu=0
  // (op_a1 and op_a2 at epoch 0, index 0) with op_b (epoch 0, index 1) sitting
  // in between, all feeding op_t (epoch 1, index 2), plus op_a3 (epoch 1,
  // index 0) fed by op_b. Phase 2 must deduplicate q_succs[0] -> 2 cleanly
  // and contract {op_a1, op_a2, op_a3}.
  SyntheticDag dag;
  auto* in = dag.AddInput();
  LiteRtTensorT* t_a1 = nullptr;
  LiteRtTensorT* t_b = nullptr;
  LiteRtTensorT* t_a2 = nullptr;
  LiteRtTensorT* t_t = nullptr;
  LiteRtTensorT* t_a3 = nullptr;
  auto* op_a1 = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_a1);
  auto* op_b = dag.AddOp(kLiteRtOpCodeTflSub, {in, in}, &t_b);
  auto* op_a2 = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_a2);
  auto* op_t =
      dag.AddOp(kLiteRtOpCodeTflConcatenation, {t_a1, t_b, t_a2}, &t_t);
  auto* op_a3 = dag.AddOp(kLiteRtOpCodeTflMul, {t_b, t_b}, &t_a3);
  dag.MarkOutput(t_t);
  dag.MarkOutput(t_a3);

  std::vector<LiteRtOpWithPartitionIndex> selected = {
      {op_a1, 0}, {op_b, 1}, {op_a2, 0}, {op_t, 2}, {op_a3, 0}};
  auto acyclic = GroupPartitionsAcyclic(selected, dag.subgraph);
  ASSERT_EQ(acyclic.size(), 3);
  EXPECT_THAT(acyclic,
              ::testing::UnorderedElementsAre(
                  std::vector<LiteRtOp>({op_b}), std::vector<LiteRtOp>({op_t}),
                  std::vector<LiteRtOp>({op_a1, op_a2, op_a3})));

  for (auto& island : acyclic) {
    auto& slice = dag.model.EmplaceSubgraph();
    ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue());
    EXPECT_TRUE(ValidateSubgraphIO(slice));
    EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
  }
}

// Verifies every structural, topological, and host-graph acyclicity invariant
// of `GroupPartitionsAcyclic` on an acyclic `subgraph`. Returns an empty string
// on success or a human-readable failure description.
std::string CheckAcyclicPartitioningInvariants(
    const LiteRtSubgraphT& subgraph,
    const std::vector<LiteRtOpWithPartitionIndex>& selected_ops,
    const std::vector<std::vector<LiteRtOp>>& partitions) {
  absl::flat_hash_set<LiteRtOp> subgraph_op_set;
  for (LiteRtOp op : subgraph.Ops()) {
    if (op != nullptr) {
      subgraph_op_set.insert(op);
    }
  }

  absl::flat_hash_map<LiteRtOp, LiteRtParamIndex> expected_selected;
  for (const auto& [op, part_idx] : selected_ops) {
    if (subgraph_op_set.contains(op)) {
      expected_selected[op] = part_idx;
    }
  }

  absl::flat_hash_map<LiteRtOp, int> op_to_part;
  absl::flat_hash_map<LiteRtOp, size_t> op_pos_in_part;
  for (size_t p = 0; p < partitions.size(); ++p) {
    const std::vector<LiteRtOp>& group = partitions[p];
    if (group.empty()) {
      return absl::StrCat("Partition ", p, " is empty.");
    }
    auto first_it = expected_selected.find(group.front());
    if (first_it == expected_selected.end()) {
      return absl::StrCat("Unselected op in partition ", p, ".");
    }
    const LiteRtParamIndex expected_idx = first_it->second;
    for (size_t i = 0; i < group.size(); ++i) {
      LiteRtOp op = group[i];
      auto sel_it = expected_selected.find(op);
      if (sel_it == expected_selected.end()) {
        return absl::StrCat("Unselected op at index ", i, " in partition ", p,
                            ".");
      }
      if (sel_it->second != expected_idx) {
        return absl::StrCat("Mixed partition_index in partition ", p, ".");
      }
      if (!op_to_part.emplace(op, static_cast<int>(p)).second) {
        return absl::StrCat("Duplicate op emitted across/within partitions.");
      }
      op_pos_in_part[op] = i;
    }
  }

  if (op_to_part.size() != expected_selected.size()) {
    return absl::StrCat("Partitioned op count (", op_to_part.size(),
                        ") != expected selected op count (",
                        expected_selected.size(), ").");
  }

  // Assign supernode IDs: [0, P) for partitions, [P, P + U) for unselected ops.
  const int num_parts = static_cast<int>(partitions.size());
  absl::flat_hash_map<LiteRtOp, int> op_to_supernode = op_to_part;
  int num_supernodes = num_parts;
  for (LiteRtOp op : subgraph.Ops()) {
    if (op != nullptr && !op_to_supernode.contains(op)) {
      op_to_supernode[op] = num_supernodes++;
    }
  }

  // Collect all intra-partition order checks and contracted host-graph edges.
  std::vector<absl::flat_hash_set<int>> host_adj_sets(num_supernodes);
  auto check_edge = [&](LiteRtOp u, LiteRtOp v) -> std::string {
    if (u == nullptr || v == nullptr || !subgraph_op_set.contains(u) ||
        !subgraph_op_set.contains(v)) {
      return "";
    }
    const int su = op_to_supernode.at(u);
    const int sv = op_to_supernode.at(v);
    if (su == sv) {
      if (u != v && su < num_parts &&
          op_pos_in_part.at(u) >= op_pos_in_part.at(v)) {
        return absl::StrCat(
            "Intra-partition topological violation in "
            "partition ",
            su, ".");
      }
      return "";
    }
    host_adj_sets[su].insert(sv);
    return "";
  };

  for (LiteRtOp u : subgraph.Ops()) {
    if (u == nullptr) {
      continue;
    }
    for (const LiteRtTensorT* out : u->Outputs()) {
      if (out == nullptr) {
        continue;
      }
      for (LiteRtOp v : out->Users()) {
        std::string err = check_edge(u, v);
        if (!err.empty()) {
          return err;
        }
      }
    }
    for (const LiteRtTensorT* in : u->Inputs()) {
      if (in != nullptr && in->DefiningOp() != nullptr) {
        std::string err = check_edge(in->DefiningOp(), u);
        if (!err.empty()) {
          return err;
        }
      }
    }
  }

  std::vector<std::vector<int>> host_adj(num_supernodes);
  std::vector<int> in_degree(num_supernodes, 0);
  for (int u = 0; u < num_supernodes; ++u) {
    host_adj[u].assign(host_adj_sets[u].begin(), host_adj_sets[u].end());
    std::sort(host_adj[u].begin(), host_adj[u].end());
    for (int v : host_adj[u]) {
      in_degree[v]++;
    }
  }

  // 3-color DFS cycle check on the contracted host graph.
  std::vector<int> color(num_supernodes, 0);
  auto has_cycle_dfs = [&](auto& self, int u) -> bool {
    color[u] = 1;
    for (int v : host_adj[u]) {
      if (color[v] == 1) {
        return true;
      }
      if (color[v] == 0 && self(self, v)) {
        return true;
      }
    }
    color[u] = 2;
    return false;
  };
  for (int i = 0; i < num_supernodes; ++i) {
    if (color[i] == 0 && has_cycle_dfs(has_cycle_dfs, i)) {
      return "Cycle detected in contracted host graph by 3-color DFS.";
    }
  }

  // Kahn topological sort check + inter-partition reachability order check.
  std::vector<int> ready;
  for (int i = 0; i < num_supernodes; ++i) {
    if (in_degree[i] == 0) {
      ready.push_back(i);
    }
  }
  std::vector<int> topo;
  topo.reserve(num_supernodes);
  for (size_t head = 0; head < ready.size(); ++head) {
    const int u = ready[head];
    topo.push_back(u);
    for (int v : host_adj[u]) {
      if (--in_degree[v] == 0) {
        ready.push_back(v);
      }
    }
  }
  if (static_cast<int>(topo.size()) != num_supernodes) {
    return "Cycle detected in contracted host graph by Kahn topological sort.";
  }

  // Verify returned partitions themselves follow topological order: if
  // partition `i` can reach partition `j` in the contracted host graph, `i <
  // j`.
  std::vector<std::vector<bool>> reach(
      num_supernodes, std::vector<bool>(num_supernodes, false));
  for (int idx = num_supernodes - 1; idx >= 0; --idx) {
    const int u = topo[idx];
    for (int v : host_adj[u]) {
      reach[u][v] = true;
      for (int w = 0; w < num_supernodes; ++w) {
        if (reach[v][w]) {
          reach[u][w] = true;
        }
      }
    }
  }
  for (int i = 0; i < num_parts; ++i) {
    for (int j = 0; j < i; ++j) {
      if (reach[i][j]) {
        return absl::StrCat("Partition ", i,
                            " reaches earlier-emitted partition ", j, ".");
      }
    }
  }

  return "";
}

TEST(GroupPartitionsAcyclicTest, HandlesEmptyNullAndTrivialInputs) {
  SyntheticDag empty_dag;
  EXPECT_TRUE(GroupPartitionsAcyclic({}, nullptr).empty());
  EXPECT_TRUE(GroupPartitionsAcyclic({}, empty_dag.subgraph).empty());

  SyntheticDag single_dag;
  LiteRtTensorT* in = single_dag.AddInput();
  LiteRtTensorT* out = nullptr;
  LiteRtOp op = single_dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &out);
  single_dag.MarkOutput(out);

  // Zero selected ops.
  EXPECT_TRUE(GroupPartitionsAcyclic({}, single_dag.subgraph).empty());

  // Foreign op pointer from a different subgraph is ignored safely.
  SyntheticDag other_dag;
  LiteRtTensorT* other_in = other_dag.AddInput();
  LiteRtOp foreign_op =
      other_dag.AddOp(kLiteRtOpCodeTflMul, {other_in, other_in});
  EXPECT_TRUE(
      GroupPartitionsAcyclic({{foreign_op, 0}}, single_dag.subgraph).empty());

  // Single selected op with max LiteRtParamIndex value (no +1 overflow).
  const LiteRtParamIndex max_idx = std::numeric_limits<LiteRtParamIndex>::max();
  std::vector<LiteRtOpWithPartitionIndex> max_sel = {{op, max_idx}};
  auto single_part = GroupPartitionsAcyclic(max_sel, single_dag.subgraph);
  ASSERT_EQ(single_part.size(), 1);
  EXPECT_EQ(single_part[0], std::vector<LiteRtOp>({op}));
  EXPECT_EQ(CheckAcyclicPartitioningInvariants(*single_dag.subgraph, max_sel,
                                               single_part),
            "");

  // Duplicate entries in `selected_ops` never duplicate ops in output.
  SyntheticDag two_op_dag;
  LiteRtTensorT* in2 = two_op_dag.AddInput();
  LiteRtTensorT* t1 = nullptr;
  LiteRtTensorT* t2 = nullptr;
  LiteRtOp op1 = two_op_dag.AddOp(kLiteRtOpCodeTflMul, {in2, in2}, &t1);
  LiteRtOp op2 = two_op_dag.AddOp(kLiteRtOpCodeTflMul, {t1, t1}, &t2);
  two_op_dag.MarkOutput(t2);

  std::vector<LiteRtOpWithPartitionIndex> dup_same = {
      {op1, 0}, {op1, 0}, {op2, 0}};
  auto res_same = GroupPartitionsAcyclic(dup_same, two_op_dag.subgraph);
  ASSERT_EQ(res_same.size(), 1);
  EXPECT_EQ(res_same[0], std::vector<LiteRtOp>({op1, op2}));
  EXPECT_EQ(CheckAcyclicPartitioningInvariants(*two_op_dag.subgraph, dup_same,
                                               res_same),
            "");

  std::vector<LiteRtOpWithPartitionIndex> dup_reassigned = {
      {op1, 0}, {op2, 0}, {op1, 1}};
  auto res_reassigned =
      GroupPartitionsAcyclic(dup_reassigned, two_op_dag.subgraph);
  ASSERT_EQ(res_reassigned.size(), 2);
  EXPECT_EQ(CheckAcyclicPartitioningInvariants(*two_op_dag.subgraph,
                                               dup_reassigned, res_reassigned),
            "");
}

TEST(GroupPartitionsAcyclicTest, HandlesDuplicateInputsAndMultiOutputEdges) {
  SyntheticDag dag;
  LiteRtTensorT* in = dag.AddInput();

  // Multi-output producer `op_split` with 3 outputs: 2 feed `op_consumer` and
  // 1 skips across `op_cpu` into `op_sink` alongside `op_consumer` read 3x.
  LiteRtOpT& split_op = dag.subgraph->EmplaceOp();
  split_op.SetOpCode(kLiteRtOpCodeTflSplit);
  AttachInput(in, split_op);
  LiteRtTensorT& s_out0 = dag.subgraph->EmplaceTensor();
  LiteRtTensorT& s_out1 = dag.subgraph->EmplaceTensor();
  LiteRtTensorT& s_out2 = dag.subgraph->EmplaceTensor();
  const auto tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 4});
  s_out0.SetType(tensor_type);
  s_out1.SetType(tensor_type);
  s_out2.SetType(tensor_type);
  AttachOutput(&s_out0, split_op);
  AttachOutput(&s_out1, split_op);
  AttachOutput(&s_out2, split_op);

  // `op_consumer` consumes `s_out0` twice and `s_out1` once (3 parallel edges
  // from `split_op`).
  LiteRtTensorT* t_cons = nullptr;
  LiteRtOp op_cons = dag.AddOp(kLiteRtOpCodeTflConcatenation,
                               {&s_out0, &s_out0, &s_out1}, &t_cons);

  // Unselected CPU op consumes `t_cons` 3 times.
  LiteRtTensorT* t_cpu = nullptr;
  dag.AddOp(kLiteRtOpCodeTflConcatenation, {t_cons, t_cons, t_cons}, &t_cpu);

  // `op_sink` consumes skip output `s_out2` twice and `t_cpu` twice.
  LiteRtTensorT* t_sink = nullptr;
  LiteRtOp op_sink = dag.AddOp(kLiteRtOpCodeTflConcatenation,
                               {&s_out2, t_cpu, &s_out2, t_cpu}, &t_sink);
  dag.MarkOutput(t_sink);

  std::vector<LiteRtOpWithPartitionIndex> selected = {
      {&split_op, 0}, {op_cons, 0}, {op_sink, 0}};
  auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
  ASSERT_EQ(partitions.size(), 2);
  EXPECT_EQ(partitions[0], std::vector<LiteRtOp>({&split_op, op_cons}));
  EXPECT_EQ(partitions[1], std::vector<LiteRtOp>({op_sink}));
  EXPECT_EQ(
      CheckAcyclicPartitioningInvariants(*dag.subgraph, selected, partitions),
      "");

  for (auto& island : partitions) {
    auto& slice = dag.model.EmplaceSubgraph();
    ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue());
    EXPECT_TRUE(ValidateSubgraphIO(slice));
    EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
  }
}

TEST(GroupPartitionsAcyclicTest, HandlesDisconnectedComponentsAndIsolatedOps) {
  SyntheticDag dag;
  LiteRtTensorT* in0 = dag.AddInput();
  LiteRtTensorT* in1 = dag.AddInput();

  // Component 1: a1(0) -> cpu1 -> a2(0)
  LiteRtTensorT* t_a1 = nullptr;
  LiteRtTensorT* t_cpu1 = nullptr;
  LiteRtTensorT* t_a2 = nullptr;
  LiteRtOp op_a1 = dag.AddOp(kLiteRtOpCodeTflMul, {in0, in0}, &t_a1);
  dag.AddOp(kLiteRtOpCodeTflAdd, {t_a1, t_a1}, &t_cpu1);
  LiteRtOp op_a2 = dag.AddOp(kLiteRtOpCodeTflMul, {t_cpu1, t_cpu1}, &t_a2);

  // Component 2 (disconnected): b1(1) -> b2(1)
  LiteRtTensorT* t_b1 = nullptr;
  LiteRtTensorT* t_b2 = nullptr;
  LiteRtOp op_b1 = dag.AddOp(kLiteRtOpCodeTflSub, {in1, in1}, &t_b1);
  LiteRtOp op_b2 = dag.AddOp(kLiteRtOpCodeTflSub, {t_b1, t_b1}, &t_b2);

  // Isolated selected op reading external input, isolated unselected op, and
  // synthetic 0-input 0-output selected op.
  LiteRtTensorT* t_iso_sel = nullptr;
  LiteRtTensorT* t_iso_cpu = nullptr;
  LiteRtOp op_iso_sel = dag.AddOp(kLiteRtOpCodeTflMul, {in0, in1}, &t_iso_sel);
  dag.AddOp(kLiteRtOpCodeTflAdd, {in0, in1}, &t_iso_cpu);
  LiteRtOpT& op_zero_io = dag.subgraph->EmplaceOp();
  op_zero_io.SetOpCode(kLiteRtOpCodeTflMul);

  dag.MarkOutput(t_a2);
  dag.MarkOutput(t_b2);
  dag.MarkOutput(t_iso_sel);
  dag.MarkOutput(t_iso_cpu);

  // Multi-index: index 1 + isolated ops merge cleanly while {op_a1} and {op_a2}
  // remain separated by `cpu1`.
  std::vector<LiteRtOpWithPartitionIndex> multi_sel = {
      {op_a1, 0}, {op_a2, 0},      {op_b1, 1},
      {op_b2, 1}, {op_iso_sel, 1}, {&op_zero_io, 0}};
  auto multi_parts = GroupPartitionsAcyclic(multi_sel, dag.subgraph);
  // Index 1 merges {op_b1, op_b2, op_iso_sel} into 1 partition; index 0 merges
  // {op_zero_io} into {op_a1} (or {op_a2}), leaving 2 partitions for index 0 ->
  // 3 partitions total.
  EXPECT_EQ(multi_parts.size(), 3);
  EXPECT_EQ(
      CheckAcyclicPartitioningInvariants(*dag.subgraph, multi_sel, multi_parts),
      "");
}

TEST(GroupPartitionsAcyclicTest,
     PropagatesReachabilityAcrossDistantAncestorsOnSequentialMerges) {
  // Part A: 2-merge ancestor propagation across multi-hop CPU chains:
  //   x(1) -> p1(CPU) -> p2(CPU) -> u(0)
  //   v(0) -> q1(CPU) -> q2(CPU) -> y(1)
  // `x` is a 3-hop ancestor of `u` (not a direct parent), and `y` is a 3-hop
  // descendant of `v`. Merging `(u, v)` first MUST propagate `reach[{u, v}]`
  // all the way back to `x` (and `anc[{u, v}]` forward to `y`) so the second
  // candidate `(x, y)` is rejected rather than forming cycle
  // `{x, y} -> p1 -> p2 -> {u, v} -> q1 -> q2 -> {x, y}`.
  {
    SyntheticDag dag;
    LiteRtTensorT* in = dag.AddInput();
    LiteRtTensorT* t_x = nullptr;
    LiteRtTensorT* t_p1 = nullptr;
    LiteRtTensorT* t_p2 = nullptr;
    LiteRtTensorT* t_u = nullptr;
    LiteRtTensorT* t_v = nullptr;
    LiteRtTensorT* t_q1 = nullptr;
    LiteRtTensorT* t_q2 = nullptr;
    LiteRtTensorT* t_y = nullptr;

    // Place `v` first so `(v, u)` has smaller `first_topo_pos` and merges
    // before `(x, y)`.
    LiteRtOp op_v = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_v);
    dag.AddOp(kLiteRtOpCodeTflAdd, {t_v, t_v}, &t_q1);
    dag.AddOp(kLiteRtOpCodeTflAdd, {t_q1, t_q1}, &t_q2);
    LiteRtOp op_y = dag.AddOp(kLiteRtOpCodeTflSub, {t_q2, t_q2}, &t_y);

    LiteRtOp op_x = dag.AddOp(kLiteRtOpCodeTflSub, {in, in}, &t_x);
    dag.AddOp(kLiteRtOpCodeTflAdd, {t_x, t_x}, &t_p1);
    dag.AddOp(kLiteRtOpCodeTflAdd, {t_p1, t_p1}, &t_p2);
    LiteRtOp op_u = dag.AddOp(kLiteRtOpCodeTflMul, {t_p2, t_p2}, &t_u);

    dag.MarkOutput(t_y);
    dag.MarkOutput(t_u);

    std::vector<LiteRtOpWithPartitionIndex> selected = {
        {op_v, 0}, {op_u, 0}, {op_x, 1}, {op_y, 1}};
    auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
    ASSERT_EQ(partitions.size(), 3);
    EXPECT_EQ(
        CheckAcyclicPartitioningInvariants(*dag.subgraph, selected, partitions),
        "");

    for (auto& island : partitions) {
      auto& slice = dag.model.EmplaceSubgraph();
      ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue());
      EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
    }
  }

  // Part B: 3-merge cascading transitive chain across 4 backend indices:
  //   d1(3) -> c1(2) -> cpu1 -> a1(0)
  //   a2(0) -> cpu2 -> b1(1)
  //   b2(1) -> cpu3 -> c2(2) -> d2(3)
  // Merging (a1, a2), then (b1, b2), then (c1, c2) creates a 3-supernode
  // transitive path d1 -> {c1,c2} -> cpu1 -> {a1,a2} -> cpu2 -> {b1,b2} ->
  // cpu3 -> {c1,c2} -> d2? Wait: even better, make (a1, a2), (b1, b2), (c1, c2)
  // ALL individually & jointly acyclic:
  //   d1(3) -> cpu0 -> a1(0)
  //   a2(0) -> cpu1 -> b1(1)
  //   b2(1) -> cpu2 -> c1(2)
  //   c2(2) -> cpu3 -> d2(3)
  // Here (a1, a2), (b1, b2), and (c1, c2) ALL legitimately merge, building a
  // 3-merge transitive bridge `d1 -> ... -> {a1,a2} -> ... -> {b1,b2} -> ... ->
  // {c1,c2} -> ... -> d2` that MUST block the 4th merge `(d1, d2)`!
  {
    SyntheticDag dag;
    LiteRtTensorT* in = dag.AddInput();
    LiteRtTensorT* t_a2 = nullptr;
    LiteRtTensorT* t_cpu1 = nullptr;
    LiteRtTensorT* t_b1 = nullptr;
    LiteRtTensorT* t_b2 = nullptr;
    LiteRtTensorT* t_cpu2 = nullptr;
    LiteRtTensorT* t_c1 = nullptr;
    LiteRtTensorT* t_c2 = nullptr;
    LiteRtTensorT* t_cpu3 = nullptr;
    LiteRtTensorT* t_d2 = nullptr;
    LiteRtTensorT* t_d1 = nullptr;
    LiteRtTensorT* t_cpu0 = nullptr;
    LiteRtTensorT* t_a1 = nullptr;

    LiteRtOp op_a2 = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_a2);
    dag.AddOp(kLiteRtOpCodeTflAdd, {t_a2, t_a2}, &t_cpu1);
    LiteRtOp op_b1 = dag.AddOp(kLiteRtOpCodeTflSub, {t_cpu1, t_cpu1}, &t_b1);

    LiteRtOp op_b2 = dag.AddOp(kLiteRtOpCodeTflSub, {in, in}, &t_b2);
    dag.AddOp(kLiteRtOpCodeTflAdd, {t_b2, t_b2}, &t_cpu2);
    LiteRtOp op_c1 = dag.AddOp(kLiteRtOpCodeTflMul, {t_cpu2, t_cpu2}, &t_c1);

    LiteRtOp op_c2 = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_c2);
    dag.AddOp(kLiteRtOpCodeTflAdd, {t_c2, t_c2}, &t_cpu3);
    LiteRtOp op_d2 = dag.AddOp(kLiteRtOpCodeTflSub, {t_cpu3, t_cpu3}, &t_d2);

    LiteRtOp op_d1 = dag.AddOp(kLiteRtOpCodeTflSub, {in, in}, &t_d1);
    dag.AddOp(kLiteRtOpCodeTflAdd, {t_d1, t_d1}, &t_cpu0);
    LiteRtOp op_a1 = dag.AddOp(kLiteRtOpCodeTflMul, {t_cpu0, t_cpu0}, &t_a1);

    dag.MarkOutput(t_b1);
    dag.MarkOutput(t_c1);
    dag.MarkOutput(t_d2);
    dag.MarkOutput(t_a1);

    std::vector<LiteRtOpWithPartitionIndex> selected = {
        {op_a1, 0}, {op_a2, 0}, {op_b1, 1}, {op_b2, 1},
        {op_c1, 2}, {op_c2, 2}, {op_d1, 3}, {op_d2, 3}};
    auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
    // {a1,a2}, {b1,b2}, {c1,c2} merge (3 partitions), while d1 and d2 MUST
    // remain separate (2 partitions) -> 5 partitions total.
    ASSERT_EQ(partitions.size(), 5);
    EXPECT_EQ(
        CheckAcyclicPartitioningInvariants(*dag.subgraph, selected, partitions),
        "");

    for (auto& island : partitions) {
      auto& slice = dag.model.EmplaceSubgraph();
      ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue());
      EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph));
    }
  }

  // Part C: Merged supernode with earlier `first_topo_pos` than a producer
  // partition is still emitted AFTER all its producer partitions.
  //   Branch 1: b(0) at topo_pos 0 (epoch 0)
  //   Branch 2: y(1) at topo_pos 1 (epoch 0) -> a(0) at topo_pos 2 (epoch 1)
  // `{b, a}` has `first_topo_pos = 0 < first_topo_pos(y) = 1`, yet `y -> {b,
  // a}`!
  {
    SyntheticDag dag;
    LiteRtTensorT* in = dag.AddInput();
    LiteRtTensorT* t_b = nullptr;
    LiteRtTensorT* t_y = nullptr;
    LiteRtTensorT* t_a = nullptr;
    LiteRtOp op_b = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_b);
    LiteRtOp op_y = dag.AddOp(kLiteRtOpCodeTflSub, {in, in}, &t_y);
    LiteRtOp op_a = dag.AddOp(kLiteRtOpCodeTflMul, {t_y, t_y}, &t_a);
    dag.MarkOutput(t_b);
    dag.MarkOutput(t_a);

    std::vector<LiteRtOpWithPartitionIndex> selected = {
        {op_b, 0}, {op_y, 1}, {op_a, 0}};
    auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
    ASSERT_EQ(partitions.size(), 2);
    EXPECT_EQ(partitions[0], std::vector<LiteRtOp>({op_y}));
    EXPECT_EQ(partitions[1], std::vector<LiteRtOp>({op_b, op_a}));
    EXPECT_EQ(
        CheckAcyclicPartitioningInvariants(*dag.subgraph, selected, partitions),
        "");
  }
}

TEST(GroupPartitionsAcyclicTest,
     ExercisesBitSetWordBoundariesAndMultiWordContraction) {
  // Test exact 64-bit word boundaries (63, 64, 65, 127, 128, 129, 200) in both
  // single-index epoch chains and multi-index quotient contraction.
  for (int num_epochs : {63, 64, 65, 127, 128, 129, 200}) {
    // 1. Single-index skip-connected ladder forcing `num_epochs` distinct
    // epochs.
    {
      SyntheticDag dag;
      LiteRtTensorT* prev = dag.AddInput();
      std::vector<LiteRtOpWithPartitionIndex> selected;
      selected.reserve(num_epochs);
      for (int e = 0; e < num_epochs; ++e) {
        LiteRtTensorT* t_sel = nullptr;
        LiteRtOp op_sel = dag.AddOp(kLiteRtOpCodeTflMul, {prev, prev}, &t_sel);
        selected.push_back({op_sel, 0});
        if (e + 1 < num_epochs) {
          LiteRtTensorT* t_cpu = nullptr;
          dag.AddOp(kLiteRtOpCodeTflAdd, {t_sel, prev}, &t_cpu);
          prev = t_cpu;
        } else {
          prev = t_sel;
        }
      }
      dag.MarkOutput(prev);

      auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
      ASSERT_EQ(static_cast<int>(partitions.size()), num_epochs)
          << "num_epochs=" << num_epochs;
      EXPECT_EQ(CheckAcyclicPartitioningInvariants(*dag.subgraph, selected,
                                                   partitions),
                "")
          << "num_epochs=" << num_epochs;
    }

    // 2. Multi-index ladder with `num_epochs` alternating blocks across indices
    // {0, 1} PLUS an independent side branch of index 0 and index 1 blocks that
    // CAN contract into the main ladder across multi-word BitSets.
    {
      SyntheticDag dag;
      LiteRtTensorT* in = dag.AddInput();
      LiteRtTensorT* prev = in;
      std::vector<LiteRtOpWithPartitionIndex> selected;
      selected.reserve(num_epochs + 4);
      for (int e = 0; e < num_epochs; ++e) {
        LiteRtTensorT* t_sel = nullptr;
        const LiteRtParamIndex part_idx = static_cast<LiteRtParamIndex>(e & 1);
        LiteRtOp op_sel = dag.AddOp(kLiteRtOpCodeTflMul, {prev, prev}, &t_sel);
        selected.push_back({op_sel, part_idx});
        prev = t_sel;
      }
      // Side branch: s0(0) -> cpu_side -> s1(1) with no path back to the main
      // chain, exercising multi-word `Set`, `Clear`, `OrWith`, `Intersects`,
      // and `ForEachSetBit` (`countr_zero`) across words 0..3.
      LiteRtTensorT* t_s0 = nullptr;
      LiteRtTensorT* t_cpu_side = nullptr;
      LiteRtTensorT* t_s1 = nullptr;
      LiteRtOp op_s0 = dag.AddOp(kLiteRtOpCodeTflMul, {in, in}, &t_s0);
      dag.AddOp(kLiteRtOpCodeTflAdd, {t_s0, t_s0}, &t_cpu_side);
      LiteRtOp op_s1 =
          dag.AddOp(kLiteRtOpCodeTflSub, {t_cpu_side, t_cpu_side}, &t_s1);
      selected.push_back({op_s0, 0});
      selected.push_back({op_s1, 1});

      dag.MarkOutput(prev);
      dag.MarkOutput(t_s1);

      auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
      // Both `op_s0` and `op_s1` merge into existing ladder blocks without
      // adding extra partitions beyond `num_epochs`.
      EXPECT_EQ(static_cast<int>(partitions.size()), num_epochs)
          << "num_epochs=" << num_epochs;
      EXPECT_EQ(CheckAcyclicPartitioningInvariants(*dag.subgraph, selected,
                                                   partitions),
                "")
          << "num_epochs=" << num_epochs;
    }
  }
}

TEST(GroupPartitionsAcyclicTest,
     HandlesMalformedCyclicIrWithoutDroppingOpsOrCreatingMultiOpCycles) {
  // Case 1: 2-op cycle of selected ops `s1(0) <-> s2(0)`.
  // Must terminate, keep both selected ops, and emit 2 singleton partitions
  // rather than merging them into a cyclic 2-op partition.
  {
    SyntheticDag dag;
    LiteRtTensorT* in = dag.AddInput();
    LiteRtOpT& s1 = dag.subgraph->EmplaceOp();
    LiteRtOpT& s2 = dag.subgraph->EmplaceOp();
    s1.SetOpCode(kLiteRtOpCodeTflMul);
    s2.SetOpCode(kLiteRtOpCodeTflMul);
    LiteRtTensorT& t1 = dag.subgraph->EmplaceTensor();
    LiteRtTensorT& t2 = dag.subgraph->EmplaceTensor();
    const auto type = MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 4});
    t1.SetType(type);
    t2.SetType(type);
    AttachOutput(&t1, s1);
    AttachOutput(&t2, s2);
    AttachInput(in, s1);
    AttachInput(&t2, s1);
    AttachInput(&t1, s2);
    AttachInput(in, s2);
    dag.MarkOutput(&t2);

    std::vector<LiteRtOpWithPartitionIndex> selected = {{&s1, 0}, {&s2, 0}};
    auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
    ASSERT_EQ(partitions.size(), 2);
    EXPECT_EQ(partitions[0], std::vector<LiteRtOp>({&s1}));
    EXPECT_EQ(partitions[1], std::vector<LiteRtOp>({&s2}));
  }

  // Case 2: Self-loop on `s1(0) -> s1(0)` feeding `s2(0)`.
  {
    SyntheticDag dag;
    LiteRtTensorT* in = dag.AddInput();
    LiteRtOpT& s1 = dag.subgraph->EmplaceOp();
    LiteRtOpT& s2 = dag.subgraph->EmplaceOp();
    s1.SetOpCode(kLiteRtOpCodeTflMul);
    s2.SetOpCode(kLiteRtOpCodeTflMul);
    LiteRtTensorT& t1 = dag.subgraph->EmplaceTensor();
    LiteRtTensorT& t2 = dag.subgraph->EmplaceTensor();
    const auto type = MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 4});
    t1.SetType(type);
    t2.SetType(type);
    AttachOutput(&t1, s1);
    AttachInput(in, s1);
    AttachInput(&t1, s1);  // Self-loop on s1.
    AttachOutput(&t2, s2);
    AttachInput(&t1, s2);
    AttachInput(&t1, s2);
    dag.MarkOutput(&t2);

    std::vector<LiteRtOpWithPartitionIndex> selected = {{&s1, 0}, {&s2, 0}};
    auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
    ASSERT_EQ(partitions.size(), 2);
    EXPECT_EQ(partitions[0], std::vector<LiteRtOp>({&s1}));
    EXPECT_EQ(partitions[1], std::vector<LiteRtOp>({&s2}));
  }

  // Case 3: Acyclic skip-edge trap prefix `a(0) -> b(CPU) -> c(0)` with skip
  // edge `a -> c`, stored out-of-order as `[a, c, b]` in `subgraph->Ops()`,
  // followed by a downstream cycle `x(0) <-> y(0)` fed by `c`.
  // `GroupPartitionsAcyclic` must still split `{a}` and `{c}` in the acyclic
  // Kahn prefix (not merge `{a, c}` across `b`!) and emit `{x}`, `{y}` as
  // singletons without dropping any selected op.
  {
    SyntheticDag dag;
    LiteRtTensorT* in = dag.AddInput();
    LiteRtOpT& op_a = dag.subgraph->EmplaceOp();
    LiteRtOpT& op_c = dag.subgraph->EmplaceOp();
    LiteRtOpT& op_b = dag.subgraph->EmplaceOp();
    LiteRtOpT& op_x = dag.subgraph->EmplaceOp();
    LiteRtOpT& op_y = dag.subgraph->EmplaceOp();
    op_a.SetOpCode(kLiteRtOpCodeTflMul);
    op_b.SetOpCode(kLiteRtOpCodeTflAdd);
    op_c.SetOpCode(kLiteRtOpCodeTflMul);
    op_x.SetOpCode(kLiteRtOpCodeTflMul);
    op_y.SetOpCode(kLiteRtOpCodeTflMul);

    const auto type = MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 4});
    LiteRtTensorT& t_a = dag.subgraph->EmplaceTensor();
    LiteRtTensorT& t_b = dag.subgraph->EmplaceTensor();
    LiteRtTensorT& t_c = dag.subgraph->EmplaceTensor();
    LiteRtTensorT& t_x = dag.subgraph->EmplaceTensor();
    LiteRtTensorT& t_y = dag.subgraph->EmplaceTensor();
    t_a.SetType(type);
    t_b.SetType(type);
    t_c.SetType(type);
    t_x.SetType(type);
    t_y.SetType(type);

    AttachInput(in, op_a);
    AttachInput(in, op_a);
    AttachOutput(&t_a, op_a);

    AttachInput(&t_a, op_b);
    AttachInput(&t_a, op_b);
    AttachOutput(&t_b, op_b);

    AttachInput(&t_a, op_c);
    AttachInput(&t_b, op_c);
    AttachOutput(&t_c, op_c);

    AttachInput(&t_c, op_x);
    AttachInput(&t_y, op_x);
    AttachOutput(&t_x, op_x);

    AttachInput(&t_x, op_y);
    AttachInput(&t_c, op_y);
    AttachOutput(&t_y, op_y);
    dag.MarkOutput(&t_y);

    std::vector<LiteRtOpWithPartitionIndex> selected = {
        {&op_a, 0}, {&op_c, 0}, {&op_x, 0}, {&op_y, 0}};
    auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
    ASSERT_EQ(partitions.size(), 4);
    EXPECT_EQ(partitions[0], std::vector<LiteRtOp>({&op_a}));
    EXPECT_EQ(partitions[1], std::vector<LiteRtOp>({&op_c}));
    EXPECT_EQ(partitions[2], std::vector<LiteRtOp>({&op_x}));
    EXPECT_EQ(partitions[3], std::vector<LiteRtOp>({&op_y}));
  }
}

// Builds a random DAG using `rng` with `num_ops` ops, optional duplicate/skip
// edges, multi-output ops, non-canonical op storage order, and up to
// `num_backend_indices` partition indices.
void PopulateRandomDag(std::mt19937& rng, int num_ops, int num_backend_indices,
                       double select_prob, bool deep_ladder, SyntheticDag& dag,
                       std::vector<LiteRtOpWithPartitionIndex>& selected_ops) {
  const auto tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 4});
  std::uniform_int_distribution<int> num_inputs_dist(1, 3);
  const int num_graph_inputs = num_inputs_dist(rng);
  std::vector<LiteRtTensorT*> available_tensors;
  for (int i = 0; i < num_graph_inputs; ++i) {
    available_tensors.push_back(dag.AddInput());
  }

  std::vector<LiteRtOp> topo_ops(num_ops, nullptr);
  for (int i = 0; i < num_ops; ++i) {
    LiteRtOpT& op = dag.subgraph->EmplaceOp();
    topo_ops[i] = &op;
  }

  std::bernoulli_distribution select_dist(select_prob);
  std::uniform_int_distribution<int> index_dist(0, num_backend_indices - 1);
  std::bernoulli_distribution multi_out_dist(0.15);
  std::bernoulli_distribution dup_input_dist(0.25);

  for (int i = 0; i < num_ops; ++i) {
    LiteRtOpT& op = *topo_ops[i];
    const bool selected = select_dist(rng);
    op.SetOpCode(selected ? kLiteRtOpCodeTflMul : kLiteRtOpCodeTflAdd);
    if (selected) {
      selected_ops.push_back(
          {&op, static_cast<LiteRtParamIndex>(index_dist(rng))});
    }

    if (deep_ladder && i > 0) {
      // Connect from immediate predecessor and an earlier skip ancestor to
      // force high epoch depth along the ladder.
      const int prev_idx = static_cast<int>(available_tensors.size()) - 1;
      const int skip_idx = std::max(0, prev_idx - 2);
      AttachInput(available_tensors[prev_idx], op);
      AttachInput(available_tensors[skip_idx], op);
    } else {
      std::uniform_int_distribution<int> fanin_dist(1, 3);
      const int fanin = fanin_dist(rng);
      std::uniform_int_distribution<size_t> tensor_pick(
          0, available_tensors.size() - 1);
      LiteRtTensorT* first_in = available_tensors[tensor_pick(rng)];
      AttachInput(first_in, op);
      for (int k = 1; k < fanin; ++k) {
        if (dup_input_dist(rng)) {
          AttachInput(first_in, op);
        } else {
          AttachInput(available_tensors[tensor_pick(rng)], op);
        }
      }
    }

    const int num_outs = multi_out_dist(rng) ? 2 : 1;
    for (int o = 0; o < num_outs; ++o) {
      LiteRtTensorT& out = dag.subgraph->EmplaceTensor();
      out.SetType(tensor_type);
      AttachOutput(&out, op);
      available_tensors.push_back(&out);
      if (i == num_ops - 1) {
        dag.MarkOutput(&out);
      }
    }
  }

  // Randomly permute both `subgraph->Ops()` storage order and `selected_ops`
  // order to ensure `GroupPartitionsAcyclic` never relies on caller/storage
  // topological presorting.
  std::vector<LiteRtOp> shuffled_storage(dag.subgraph->Ops().begin(),
                                         dag.subgraph->Ops().end());
  std::shuffle(shuffled_storage.begin(), shuffled_storage.end(), rng);
  dag.subgraph->ReorderOps(shuffled_storage);
  std::shuffle(selected_ops.begin(), selected_ops.end(), rng);
}

TEST(GroupPartitionsAcyclicTest,
     RandomSingleIndexDagsAreAcyclicCompleteAndDeterministic) {
  for (uint32_t seed = 1000; seed < 1120; ++seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> size_dist(12, 80);
    std::uniform_real_distribution<double> prob_dist(0.15, 0.95);
    const int num_ops = size_dist(rng);
    const double select_prob = prob_dist(rng);

    SyntheticDag dag;
    std::vector<LiteRtOpWithPartitionIndex> selected;
    PopulateRandomDag(rng, num_ops, /*num_backend_indices=*/1, select_prob,
                      /*deep_ladder=*/(seed % 4 == 0), dag, selected);

    auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
    ASSERT_EQ(
        CheckAcyclicPartitioningInvariants(*dag.subgraph, selected, partitions),
        "")
        << "seed=" << seed;

    // Determinism across repeated calls and reversed `selected` order.
    std::vector<LiteRtOpWithPartitionIndex> reversed_selected(selected.rbegin(),
                                                              selected.rend());
    EXPECT_EQ(GroupPartitionsAcyclic(reversed_selected, dag.subgraph),
              partitions)
        << "seed=" << seed;

    // Exercise full end-to-end `OutlinePartition` + `TopologicalSort` on every
    // 6th random graph (keeping `ReorderOps` topological first as required by
    // `GraphSlicer` preconditions on `root`).
    if (seed % 6 == 0 && !partitions.empty()) {
      for (auto& island : partitions) {
        auto& slice = dag.model.EmplaceSubgraph();
        ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue())
            << "seed=" << seed;
        EXPECT_TRUE(
            ValidateLocalTopology(slice.Ops().cbegin(), slice.Ops().cend()))
            << "seed=" << seed;
      }
      EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph)) << "seed=" << seed;
    }
  }
}

TEST(GroupPartitionsAcyclicTest,
     RandomMultiIndexDagsAreAcyclicCompleteAndDeterministic) {
  for (uint32_t seed = 2000; seed < 2150; ++seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> size_dist(15, 90);
    std::uniform_int_distribution<int> indices_dist(2, 4);
    std::uniform_real_distribution<double> prob_dist(0.25, 0.95);
    const int num_ops = size_dist(rng);
    const int num_indices = indices_dist(rng);
    const double select_prob = prob_dist(rng);

    SyntheticDag dag;
    std::vector<LiteRtOpWithPartitionIndex> selected;
    PopulateRandomDag(rng, num_ops, num_indices, select_prob,
                      /*deep_ladder=*/(seed % 5 == 0), dag, selected);

    auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
    ASSERT_EQ(
        CheckAcyclicPartitioningInvariants(*dag.subgraph, selected, partitions),
        "")
        << "seed=" << seed << " num_indices=" << num_indices;

    std::vector<LiteRtOpWithPartitionIndex> reversed_selected(selected.rbegin(),
                                                              selected.rend());
    EXPECT_EQ(GroupPartitionsAcyclic(reversed_selected, dag.subgraph),
              partitions)
        << "seed=" << seed;

    if (seed % 8 == 0 && !partitions.empty()) {
      for (auto& island : partitions) {
        auto& slice = dag.model.EmplaceSubgraph();
        ASSERT_TRUE(OutlinePartition(*dag.subgraph, &slice, island).HasValue())
            << "seed=" << seed;
        EXPECT_TRUE(
            ValidateLocalTopology(slice.Ops().cbegin(), slice.Ops().cend()))
            << "seed=" << seed;
      }
      EXPECT_TRUE(IsTopologicallySorted(*dag.subgraph)) << "seed=" << seed;
    }
  }
}

TEST(GroupPartitionsAcyclicTest, RandomDeepOverHundredEpochDagsAreAcyclic) {
  size_t max_observed_partitions = 0;
  for (uint32_t seed = 3000; seed < 3030; ++seed) {
    std::mt19937 rng(seed);
    const int num_ops = 260;
    const int num_indices = (seed % 2 == 0) ? 1 : 3;

    SyntheticDag dag;
    std::vector<LiteRtOpWithPartitionIndex> selected;
    PopulateRandomDag(rng, num_ops, num_indices, /*select_prob=*/0.6,
                      /*deep_ladder=*/true, dag, selected);

    auto partitions = GroupPartitionsAcyclic(selected, dag.subgraph);
    max_observed_partitions =
        std::max(max_observed_partitions, partitions.size());
    ASSERT_EQ(
        CheckAcyclicPartitioningInvariants(*dag.subgraph, selected, partitions),
        "")
        << "seed=" << seed;
  }
  EXPECT_GT(max_observed_partitions, 100U);
}

}  // namespace
}  // namespace litert::internal
