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

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
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
                         ::testing::Values(&GroupPartitions,
                                           &GroupPartitionsV2));

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

  for (auto op_ref : ops) {
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

  for (auto op_ref : ops) {
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

}  // namespace
}  // namespace litert::internal
