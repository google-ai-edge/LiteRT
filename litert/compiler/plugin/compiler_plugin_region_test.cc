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

// Tests for partitioning models with region-bearing ops (ops referencing other
// subgraphs, e.g. STABLEHLO_REDUCE_WINDOW, STABLEHLO_SORT, WHILE, IF).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/model/subgraph_refs.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Not;
using ::testing::UnorderedElementsAreArray;

constexpr absl::string_view kTestPluginSearchPath = "vendors/examples";

std::unique_ptr<LiteRtModelT> LoadModel(absl::string_view filename) {
  auto model = LoadModelFromFile(litert::testing::GetTestFilePath(filename));
  ABSL_CHECK(model) << model.Error().Message();
  return std::move(*model);
}

// Records which subgraphs were passed to the partition function and selects
// ops according to `select`.
class FakePartitioner {
 public:
  using Selector = std::function<bool(const LiteRtOpT&)>;

  explicit FakePartitioner(Selector select) : select_(std::move(select)) {}

  // The returned callable references this object and is meant to be passed
  // directly to PartitionModel.
  auto Fn() {
    return [this](LiteRtSubgraph subgraph)
               -> Expected<std::vector<LiteRtOpWithPartitionIndex>> {
      visited_.push_back(std::string(subgraph->Name()));
      std::vector<LiteRtOpWithPartitionIndex> selected;
      for (LiteRtOp op : subgraph->Ops()) {
        if (select_(*op)) {
          selected.push_back({op, 0});
        }
      }
      return selected;
    };
  }

  // Names of the subgraphs passed to the partition function.
  const std::vector<std::string>& Visited() const { return visited_; }

 private:
  Selector select_;
  std::vector<std::string> visited_;
};

FakePartitioner::Selector SelectAll() {
  return [](const LiteRtOpT&) { return true; };
}

FakePartitioner::Selector SelectOpCodes(
    absl::flat_hash_set<LiteRtOpCode> codes) {
  return [codes = std::move(codes)](const LiteRtOpT& op) {
    return codes.contains(op.OpCode());
  };
}

PartitionModelOptions WithRegionSupport() {
  PartitionModelOptions options;
  options.supports_region_subgraphs = true;
  return options;
}

PartitionModelOptions WithoutRegionSupport() { return PartitionModelOptions(); }

size_t CountOps(const LiteRtSubgraphT& subgraph, LiteRtOpCode code) {
  size_t count = 0;
  for (LiteRtOp op : subgraph.Ops()) {
    count += op->OpCode() == code;
  }
  return count;
}

size_t CountRegionOps(const LiteRtSubgraphT& subgraph) {
  size_t count = 0;
  for (LiteRtOp op : subgraph.Ops()) {
    count += HasRegionRefs(*op);
  }
  return count;
}

LiteRtOp FindOp(const LiteRtSubgraphT& subgraph, LiteRtOpCode code) {
  for (LiteRtOp op : subgraph.Ops()) {
    if (op->OpCode() == code) {
      return op;
    }
  }
  return nullptr;
}

std::vector<int32_t> RefIndices(const LiteRtOpT& op) {
  std::vector<int32_t> indices;
  for (const auto& ref : GetSubgraphRefs(op)) {
    indices.push_back(ref.index);
  }
  return indices;
}

void ExpectValid(const LiteRtModelT& model, const LiteRtModelT& partitions) {
  LITERT_EXPECT_OK(ValidateSubgraphRefs(model));
  LITERT_EXPECT_OK(ValidateSubgraphRefs(partitions));
  // Region subgraphs of the partitions model are uniquely named so that
  // backends importing it through MLIR get consistent symbols.
  absl::flat_hash_set<std::string> names;
  for (size_t i = partitions.NumEntrySubgraphs(); i < partitions.NumSubgraphs();
       ++i) {
    const std::string name(partitions.Subgraph(i).Name());
    EXPECT_FALSE(name.empty());
    EXPECT_TRUE(names.insert(name).second) << name;
  }
  // No private region subgraph contains a dispatch op.
  for (size_t i = partitions.NumEntrySubgraphs(); i < partitions.NumSubgraphs();
       ++i) {
    EXPECT_EQ(CountOps(partitions.Subgraph(i), kLiteRtOpCodeTflCustom), 0);
  }
}

//
// STABLEHLO_REDUCE_WINDOW (the motivating example).
//

TEST(RegionPartitionTest, ComputationBodyIsNeverPartitionedOnItsOwn) {
  // A plugin rejecting REDUCE_WINDOW but accepting the scalar MAXIMUM of its
  // body must not offload the body: the CPU REDUCE_WINDOW kernel requires a
  // single builtin op in its body.
  auto model = LoadModel("shlo_reduce_window_region.tflite");
  FakePartitioner partitioner(
      SelectOpCodes({kLiteRtOpCodeTflMaximum, kLiteRtOpCodeTflMul}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithoutRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  EXPECT_THAT(partitioner.Visited(), ElementsAre("main"));
  ASSERT_EQ(dispatch_ops.size(), 1);
  ASSERT_EQ(model->NumSubgraphs(), 2);
  EXPECT_EQ(CountOps(model->Subgraph(0), kLiteRtOpCodeShloWindow), 1);
  EXPECT_EQ(CountOps(model->Subgraph(1), kLiteRtOpCodeTflMaximum), 1);
  EXPECT_EQ(partitions.NumSubgraphs(), 1);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 1);
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, RegionOpSelectionIgnoredWithoutCapability) {
  auto model = LoadModel("shlo_reduce_window_region.tflite");
  FakePartitioner partitioner(SelectAll());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithoutRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  ASSERT_EQ(dispatch_ops.size(), 1);
  EXPECT_EQ(CountOps(model->Subgraph(0), kLiteRtOpCodeShloWindow), 1);
  ASSERT_EQ(partitions.NumSubgraphs(), 1);
  EXPECT_EQ(CountRegionOps(partitions.Subgraph(0)), 0);
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, RegionOpOffloadedWithItsBody) {
  auto model = LoadModel("shlo_reduce_window_region.tflite");
  FakePartitioner partitioner(SelectAll());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  // The body is not partitioned on its own...
  EXPECT_THAT(partitioner.Visited(), ElementsAre("main"));
  // ...and moves into the partitions model with its parent op, leaving no
  // orphaned subgraph (and no second dispatch op) behind.
  ASSERT_EQ(dispatch_ops.size(), 1);
  ASSERT_EQ(model->NumSubgraphs(), 1);
  EXPECT_EQ(model->Subgraph(0).Ops().size(), 1);

  ASSERT_EQ(partitions.NumSubgraphs(), 2);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 1);
  const auto* rw = FindOp(partitions.Subgraph(0), kLiteRtOpCodeShloWindow);
  ASSERT_NE(rw, nullptr);
  EXPECT_THAT(RefIndices(*rw), ElementsAre(1));
  EXPECT_EQ(partitions.Subgraph(1).Name(), "STABLEHLO_REDUCE_WINDOW1");
  EXPECT_EQ(CountOps(partitions.Subgraph(1), kLiteRtOpCodeTflMaximum), 1);
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, BodyIndexRemappedWithMultipleIslands) {
  // mul -> add -> reduce_window -> mul with ADD left on CPU yields two
  // islands; the body must follow the island holding the REDUCE_WINDOW.
  auto model = LoadModel("multi_island_reduce_window.tflite");
  FakePartitioner partitioner(SelectOpCodes(
      {kLiteRtOpCodeTflMul, kLiteRtOpCodeShloWindow, kLiteRtOpCodeTflMaximum}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  ASSERT_EQ(dispatch_ops.size(), 2);
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 2);
  ASSERT_EQ(partitions.NumSubgraphs(), 3);
  EXPECT_EQ(FindOp(partitions.Subgraph(0), kLiteRtOpCodeShloWindow), nullptr);
  const auto* rw = FindOp(partitions.Subgraph(1), kLiteRtOpCodeShloWindow);
  ASSERT_NE(rw, nullptr);
  EXPECT_THAT(RefIndices(*rw), ElementsAre(2));
  EXPECT_EQ(model->NumSubgraphs(), 1);
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, SharedComputationBodyIsCloned) {
  // Two REDUCE_WINDOW ops share one body; only the first one is offloaded.
  auto model = LoadModel("shlo_reduce_window_region.tflite");
  auto& main = model->Subgraph(0);
  LiteRtOp original = FindOp(main, kLiteRtOpCodeShloWindow);
  ASSERT_NE(original, nullptr);
  auto& second = MakeClone(main, *original);
  for (LiteRtTensor input : original->Inputs()) {
    AttachInput(input, second);
  }
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtTensor second_out,
                              MakeClone(main, original->Output(0)));
  AttachOutput(second_out, second);
  main.Outputs().push_back(second_out);

  FakePartitioner partitioner(
      [original](const LiteRtOpT& op) { return &op == original; });
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  ASSERT_EQ(dispatch_ops.size(), 1);
  ASSERT_EQ(partitions.NumSubgraphs(), 2);
  EXPECT_EQ(CountOps(partitions.Subgraph(1), kLiteRtOpCodeTflMaximum), 1);
  // The CPU REDUCE_WINDOW keeps the original body.
  ASSERT_EQ(model->NumSubgraphs(), 2);
  EXPECT_THAT(RefIndices(second), ElementsAre(1));
  ExpectValid(*model, partitions);
}

//
// Control flow.
//

TEST(RegionPartitionTest, ControlFlowBodiesPartitionedWhileParentOnCpu) {
  // Legacy behavior: a CPU WHILE keeps calling its bodies, which may contain
  // their own NPU partitions.
  auto model = LoadModel("tfl_while_region.tflite");
  FakePartitioner partitioner(
      SelectOpCodes({kLiteRtOpCodeTflMul, kLiteRtOpCodeTflSub}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithoutRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  EXPECT_THAT(
      partitioner.Visited(),
      UnorderedElementsAreArray({"main", "tfl.while_cond", "tfl.while_body"}));
  // One partition in main, one in the body, none in the cond.
  ASSERT_EQ(dispatch_ops.size(), 2);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 2);
  EXPECT_EQ(partitions.NumSubgraphs(), 2);
  ASSERT_EQ(model->NumSubgraphs(), 3);
  const auto* while_op = FindOp(model->Subgraph(0), kLiteRtOpCodeTflWhile);
  ASSERT_NE(while_op, nullptr);
  EXPECT_THAT(RefIndices(*while_op), ElementsAre(1, 2));
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, ControlFlowOpOffloadedWithItsBodies) {
  auto model = LoadModel("tfl_while_region.tflite");
  FakePartitioner partitioner(SelectAll());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  EXPECT_THAT(partitioner.Visited(), ElementsAre("main"));
  ASSERT_EQ(dispatch_ops.size(), 1);
  EXPECT_EQ(model->NumSubgraphs(), 1);
  ASSERT_EQ(partitions.NumSubgraphs(), 3);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 1);
  const auto* while_op = FindOp(partitions.Subgraph(0), kLiteRtOpCodeTflWhile);
  ASSERT_NE(while_op, nullptr);
  EXPECT_THAT(RefIndices(*while_op), ElementsAre(1, 2));
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, ControlFlowBodySharedWithCpuIsClonedBeforePartition) {
  // Two WHILE ops share cond/body. The first is offloaded, the second stays on
  // CPU and its body is partitioned. The offloaded WHILE must get a pristine
  // copy of the bodies (without dispatch ops).
  auto model = LoadModel("tfl_while_shared_body.tflite");
  LiteRtOp first_while = nullptr;
  for (LiteRtOp op : model->Subgraph(0).Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflWhile) {
      first_while = op;
      break;
    }
  }
  ASSERT_NE(first_while, nullptr);
  FakePartitioner partitioner([first_while](const LiteRtOpT& op) {
    return &op == first_while || op.OpCode() == kLiteRtOpCodeTflMul;
  });

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  // Main: one partition holding the first WHILE and the MUL. Body: one
  // partition holding its MUL.
  ASSERT_EQ(dispatch_ops.size(), 2);
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 2);
  ASSERT_EQ(partitions.NumSubgraphs(), 4);
  const auto* offloaded_while =
      FindOp(partitions.Subgraph(0), kLiteRtOpCodeTflWhile);
  ASSERT_NE(offloaded_while, nullptr);
  EXPECT_THAT(RefIndices(*offloaded_while), ElementsAre(2, 3));
  EXPECT_EQ(CountOps(partitions.Subgraph(3), kLiteRtOpCodeTflMul), 1);

  ASSERT_EQ(model->NumSubgraphs(), 3);
  const auto* cpu_while = FindOp(model->Subgraph(0), kLiteRtOpCodeTflWhile);
  ASSERT_NE(cpu_while, nullptr);
  EXPECT_THAT(RefIndices(*cpu_while), ElementsAre(1, 2));
  EXPECT_EQ(CountOps(model->Subgraph(2), kLiteRtOpCodeTflCustom), 1);
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, NestedRegionsWithParentOnCpu) {
  auto model = LoadModel("nested_region.tflite");
  FakePartitioner partitioner(
      SelectOpCodes({kLiteRtOpCodeTflMul, kLiteRtOpCodeTflSub}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithoutRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  // The reduction body nested in the while body is never visited.
  EXPECT_THAT(partitioner.Visited(),
              UnorderedElementsAreArray({"main", "while_cond", "while_body"}));
  EXPECT_EQ(model->NumSubgraphs(), 4);
  EXPECT_EQ(CountOps(model->Subgraph(3), kLiteRtOpCodeTflMul), 1);
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, NestedRegionsOffloadedTogether) {
  auto model = LoadModel("nested_region.tflite");
  FakePartitioner partitioner(SelectAll());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  EXPECT_THAT(partitioner.Visited(), ElementsAre("main"));
  ASSERT_EQ(dispatch_ops.size(), 1);
  EXPECT_EQ(model->NumSubgraphs(), 1);
  ASSERT_EQ(partitions.NumSubgraphs(), 4);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 1);
  LITERT_ASSERT_OK_AND_ASSIGN(auto graph, SubgraphCallGraph::Build(partitions));
  LITERT_ASSERT_OK_AND_ASSIGN(auto order, graph.TopologicalOrder());
  EXPECT_EQ(order.front(), 0);
  for (uint32_t sg = 1; sg < partitions.NumSubgraphs(); ++sg) {
    EXPECT_TRUE(graph.IsReferenced(sg)) << sg;
  }
  ExpectValid(*model, partitions);
}

//
// Regions inside composites.
//

TEST(RegionPartitionTest, RegionInInlinedDecomposition) {
  auto model = LoadModel("region_in_composite.tflite");
  FakePartitioner partitioner(SelectOpCodes(
      {kLiteRtOpCodeShloWindow, kLiteRtOpCodeTflMaximum, kLiteRtOpCodeTflMul}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  // The composite is inlined, after which REDUCE_WINDOW and MUL form one
  // island that takes the body along. The decomposition is removed.
  ASSERT_EQ(dispatch_ops.size(), 1);
  EXPECT_EQ(model->NumSubgraphs(), 1);
  ASSERT_EQ(partitions.NumSubgraphs(), 2);
  const auto* rw = FindOp(partitions.Subgraph(0), kLiteRtOpCodeShloWindow);
  ASSERT_NE(rw, nullptr);
  EXPECT_THAT(RefIndices(*rw), ElementsAre(1));
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, RegionInInlinedDecompositionStaysOnCpu) {
  auto model = LoadModel("region_in_composite.tflite");
  // The composite itself is not selected, so it is inlined. The plugin cannot
  // compile regions, so the inlined REDUCE_WINDOW stays on CPU with its body.
  FakePartitioner partitioner(SelectOpCodes(
      {kLiteRtOpCodeShloWindow, kLiteRtOpCodeTflMaximum, kLiteRtOpCodeTflMul}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithoutRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  ASSERT_EQ(model->NumSubgraphs(), 2);
  const auto* rw = FindOp(model->Subgraph(0), kLiteRtOpCodeShloWindow);
  ASSERT_NE(rw, nullptr);
  EXPECT_THAT(RefIndices(*rw), ElementsAre(1));
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, BodyOfOffloadedCompositeIsRemoved) {
  // The plugin compiles the composite directly; the decomposition and the
  // region body only it referenced are dropped.
  auto model = LoadModel("region_in_composite.tflite");
  FakePartitioner partitioner(
      SelectOpCodes({kLiteRtOpCodeShloComposite, kLiteRtOpCodeTflMul}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  ASSERT_EQ(dispatch_ops.size(), 1);
  EXPECT_EQ(model->NumSubgraphs(), 1);
  EXPECT_EQ(partitions.NumSubgraphs(), 1);
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, NpuCallWithRegionRequiresCapability) {
  auto model = LoadModel("npu_call_region.tflite");
  FakePartitioner partitioner(SelectOpCodes(
      {kLiteRtOpCodeShloWindow, kLiteRtOpCodeTflMaximum, kLiteRtOpCodeTflMul}));
  auto result =
      PartitionModel(partitioner.Fn(), WithoutRegionSupport(), *model);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorUnsupported);
  // The model is rejected before it is modified, so the next plugin (or the
  // CPU) still sees the original graph.
  ASSERT_EQ(model->NumSubgraphs(), 3);
  ASSERT_EQ(model->Subgraph(0).Ops().size(), 1);
  EXPECT_EQ(model->Subgraph(0).Op(0).OpCode(), kLiteRtOpCodeShloComposite);
  LITERT_EXPECT_OK(ValidateSubgraphRefs(*model));
}

TEST(RegionPartitionTest, NpuCallWithRegion) {
  auto model = LoadModel("npu_call_region.tflite");
  FakePartitioner partitioner(SelectOpCodes(
      {kLiteRtOpCodeShloWindow, kLiteRtOpCodeTflMaximum, kLiteRtOpCodeTflMul}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  ASSERT_EQ(dispatch_ops.size(), 1);
  EXPECT_EQ(model->NumSubgraphs(), 1);
  ASSERT_EQ(partitions.NumSubgraphs(), 2);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 1);
  EXPECT_EQ(partitions.Subgraph(0).Name(), "npu_decomp");
  const auto* rw = FindOp(partitions.Subgraph(0), kLiteRtOpCodeShloWindow);
  ASSERT_NE(rw, nullptr);
  EXPECT_THAT(RefIndices(*rw), ElementsAre(1));
  ExpectValid(*model, partitions);
}

//
// Sharing, naming, cleanup and invalid input.
//

// Appends a copy of `op` to `subgraph`, reading the same inputs and writing new
// outputs that are exposed as subgraph outputs (so that the copy is not
// dead-code eliminated).
LiteRtOp DuplicateOp(LiteRtSubgraphT& subgraph, LiteRtOpT& op) {
  auto& copy = MakeClone(subgraph, op);
  for (LiteRtTensor input : op.Inputs()) {
    AttachInput(input, copy);
  }
  for (LiteRtTensor output : op.Outputs()) {
    auto cloned = MakeClone(subgraph, *output);
    ABSL_CHECK(cloned);
    AttachOutput(*cloned, copy);
    subgraph.Outputs().push_back(*cloned);
  }
  return &copy;
}

TEST(RegionPartitionTest, SharedBodyWithNestedRegionIsClonedTransitively) {
  // Two WHILE ops share cond/body, and the body holds a REDUCE whose body is
  // shared as well. Only the first WHILE is offloaded.
  auto model = LoadModel("nested_region.tflite");
  auto& main = model->Subgraph(0);
  LiteRtOp first_while = FindOp(main, kLiteRtOpCodeTflWhile);
  ASSERT_NE(first_while, nullptr);
  DuplicateOp(main, *first_while);

  FakePartitioner partitioner(
      [first_while](const LiteRtOpT& op) { return &op == first_while; });
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  ASSERT_EQ(dispatch_ops.size(), 1);
  // The partition carries its own copies of cond, body and reduction body.
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 1);
  ASSERT_EQ(partitions.NumSubgraphs(), 4);
  LiteRtOp offloaded_while =
      FindOp(partitions.Subgraph(0), kLiteRtOpCodeTflWhile);
  ASSERT_NE(offloaded_while, nullptr);
  EXPECT_THAT(RefIndices(*offloaded_while), ElementsAre(1, 2));
  LiteRtOp offloaded_reduce =
      FindOp(partitions.Subgraph(2), kLiteRtOpCodeShloReduce);
  ASSERT_NE(offloaded_reduce, nullptr);
  EXPECT_THAT(RefIndices(*offloaded_reduce), ElementsAre(3));
  EXPECT_EQ(CountOps(partitions.Subgraph(3), kLiteRtOpCodeTflMul), 1);

  // The CPU WHILE keeps the originals.
  ASSERT_EQ(model->NumSubgraphs(), 4);
  LiteRtOp cpu_while = FindOp(model->Subgraph(0), kLiteRtOpCodeTflWhile);
  ASSERT_NE(cpu_while, nullptr);
  EXPECT_THAT(RefIndices(*cpu_while), ElementsAre(1, 2));
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, SignatureVisibleBodyIsPartitionedAndCloned) {
  // The WHILE body is also exposed through a signature, so it stays callable
  // from CPU even though its only caller is offloaded.
  auto model = LoadModel("tfl_while_region.tflite");
  auto& body = model->Subgraph(2);
  model->EmplaceSignature(&body, std::vector<std::string>{},
                          std::vector<LiteRtTensor>{},
                          std::vector<std::string>{},
                          std::vector<LiteRtTensor>{}, "body_signature");
  FakePartitioner partitioner(SelectAll());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  EXPECT_THAT(partitioner.Visited(),
              UnorderedElementsAreArray({"main", "tfl.while_body"}));
  // Main partition (WHILE + MUL) and body partition (SUB + MUL), followed by
  // the cond and a pristine copy of the body for the offloaded WHILE.
  ASSERT_EQ(dispatch_ops.size(), 2);
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 2);
  ASSERT_EQ(partitions.NumSubgraphs(), 4);
  // The signature subgraph stays in the model, now holding a dispatch op.
  ASSERT_EQ(model->NumSubgraphs(), 2);
  EXPECT_EQ(model->Subgraphs()[1], &body);
  EXPECT_EQ(CountOps(body, kLiteRtOpCodeTflCustom), 1);
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, NamesEmptyAndDuplicateRegionSubgraphs) {
  {
    auto model = LoadModel("tfl_while_region.tflite");
    model->Subgraph(1).SetName("");
    model->Subgraph(2).SetName("");
    FakePartitioner partitioner(SelectAll());
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto result,
        PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
    const auto& partitions = result.second;
    ASSERT_EQ(partitions.NumSubgraphs(), 3);
    EXPECT_EQ(partitions.Subgraph(1).Name(), "litert_region_1");
    EXPECT_EQ(partitions.Subgraph(2).Name(), "litert_region_2");
    ExpectValid(*model, partitions);
  }
  {
    auto model = LoadModel("tfl_while_region.tflite");
    model->Subgraph(1).SetName("body");
    model->Subgraph(2).SetName("body");
    FakePartitioner partitioner(SelectAll());
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto result,
        PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
    const auto& partitions = result.second;
    ASSERT_EQ(partitions.NumSubgraphs(), 3);
    EXPECT_EQ(partitions.Subgraph(1).Name(), "body");
    EXPECT_EQ(partitions.Subgraph(2).Name(), "litert_region_2");
    ExpectValid(*model, partitions);
  }
}

TEST(RegionPartitionTest, OrphanCleanupIsTransitive) {
  // main: composite(decomposition = 1)
  // 1:    WHILE(cond = 2, body = 3)
  // 3:    REDUCE_WINDOW(body = 4)
  // Offloading the composite drops its decomposition, which orphans 2 and 3,
  // and then 4.
  LiteRtModelT model;
  auto& main = model.EmplaceSubgraph();
  auto& decomposition = model.EmplaceSubgraph();
  model.EmplaceSubgraph();
  auto& while_body = model.EmplaceSubgraph();
  model.EmplaceSubgraph();

  auto& input = main.EmplaceTensor();
  auto& output = main.EmplaceTensor();
  main.Inputs().push_back(&input);
  main.Outputs().push_back(&output);
  auto& composite = main.EmplaceOp();
  composite.SetOpCode(kLiteRtOpCodeShloComposite);
  {
    ::tflite::StableHLOCompositeOptionsT opts;
    opts.name = "odml.some_composite";
    opts.decomposition_subgraph_index = 1;
    TflOptions2 options;
    options.Set(std::move(opts));
    SetTflOptions2(composite, std::move(options));
  }
  AttachInput(&input, composite);
  AttachOutput(&output, composite);

  auto& while_op = decomposition.EmplaceOp();
  while_op.SetOpCode(kLiteRtOpCodeTflWhile);
  {
    ::tflite::WhileOptionsT opts;
    opts.cond_subgraph_index = 2;
    opts.body_subgraph_index = 3;
    TflOptions options;
    options.Set(std::move(opts));
    SetTflOptions(while_op, std::move(options));
  }
  auto& reduce_window = while_body.EmplaceOp();
  reduce_window.SetOpCode(kLiteRtOpCodeShloWindow);
  {
    ::tflite::StablehloReduceWindowOptionsT opts;
    opts.body_subgraph_index = 4;
    TflOptions2 options;
    options.Set(std::move(opts));
    SetTflOptions2(reduce_window, std::move(options));
  }

  FakePartitioner partitioner(SelectOpCodes({kLiteRtOpCodeShloComposite}));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), model));
  auto& [dispatch_ops, partitions] = result;

  ASSERT_EQ(dispatch_ops.size(), 1);
  EXPECT_EQ(partitions.NumSubgraphs(), 1);
  EXPECT_EQ(model.NumSubgraphs(), 1);
  ExpectValid(model, partitions);
}

TEST(RegionPartitionTest, RejectsInvalidSubgraphRefWithoutMutation) {
  auto model = LoadModel("shlo_reduce_window_region.tflite");
  LiteRtOp rw = FindOp(model->Subgraph(0), kLiteRtOpCodeShloWindow);
  ASSERT_NE(rw, nullptr);
  UpdateSubgraphRefs(*rw, [](int32_t& index, SubgraphRefKind) { index = 7; });
  const size_t num_ops = model->Subgraph(0).Ops().size();
  FakePartitioner partitioner(SelectAll());

  auto result = PartitionModel(partitioner.Fn(), WithRegionSupport(), *model);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(partitioner.Visited(), ::testing::IsEmpty());
  EXPECT_EQ(model->NumSubgraphs(), 2);
  EXPECT_EQ(model->Subgraph(0).Ops().size(), num_ops);
}

TEST(RegionPartitionTest, RejectsCyclicSubgraphRefs) {
  // The WHILE body references the main subgraph.
  auto model = LoadModel("tfl_while_region.tflite");
  auto& recursive_while = model->Subgraph(2).EmplaceOp();
  recursive_while.SetOpCode(kLiteRtOpCodeTflWhile);
  ::tflite::WhileOptionsT opts;
  opts.cond_subgraph_index = 0;
  opts.body_subgraph_index = 0;
  TflOptions options;
  options.Set(std::move(opts));
  SetTflOptions(recursive_while, std::move(options));
  FakePartitioner partitioner(SelectAll());

  auto result = PartitionModel(partitioner.Fn(), WithRegionSupport(), *model);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(partitioner.Visited(), ::testing::IsEmpty());
  EXPECT_EQ(model->NumSubgraphs(), 3);
}

TEST(RegionPartitionTest, SubgraphsToPartitionSelectsControlFlowBody) {
  auto model = LoadModel("tfl_while_region.tflite");
  FakePartitioner partitioner(
      SelectOpCodes({kLiteRtOpCodeTflMul, kLiteRtOpCodeTflSub}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result, PartitionModel(partitioner.Fn(), WithoutRegionSupport(),
                                  *model, /*subgraphs_to_partition=*/{2}));
  auto& [dispatch_ops, partitions] = result;

  EXPECT_THAT(partitioner.Visited(), ElementsAre("tfl.while_body"));
  ASSERT_EQ(dispatch_ops.size(), 1);
  EXPECT_EQ(CountOps(model->Subgraph(0), kLiteRtOpCodeTflCustom), 0);
  EXPECT_EQ(CountOps(model->Subgraph(2), kLiteRtOpCodeTflCustom), 1);
  ExpectValid(*model, partitions);
}

TEST(RegionPartitionTest, MaxPartitionsAppliesWithRegions) {
  auto model = LoadModel("multi_island_reduce_window.tflite");
  FakePartitioner partitioner(SelectOpCodes(
      {kLiteRtOpCodeTflMul, kLiteRtOpCodeShloWindow, kLiteRtOpCodeTflMaximum}));
  PartitionModelOptions options = WithRegionSupport();
  options.max_partitions = 1;

  auto result = PartitionModel(partitioner.Fn(), options, *model);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorRuntimeFailure);
}

//
// Every region kind.
//

class RegionModelPartitionTest : public ::testing::TestWithParam<std::string> {
};

TEST_P(RegionModelPartitionTest, RegionsStayOnCpuWithoutRegionSupport) {
  auto model = LoadModel(GetParam());
  const size_t num_subgraphs = model->NumSubgraphs();
  // Names of computation bodies, which must never be visited.
  std::vector<std::string> computation_bodies;
  {
    LITERT_ASSERT_OK_AND_ASSIGN(auto graph, SubgraphCallGraph::Build(*model));
    for (uint32_t sg = 0; sg < num_subgraphs; ++sg) {
      if (graph.IsReferencedAs(sg, SubgraphRefKind::kComputation)) {
        computation_bodies.push_back(std::string(model->Subgraph(sg).Name()));
      }
    }
  }
  FakePartitioner partitioner(SelectAll());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithoutRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  for (const auto& body : computation_bodies) {
    EXPECT_THAT(partitioner.Visited(), Not(Contains(body)));
  }
  // Region-bearing ops stay on CPU, together with all their regions.
  EXPECT_GT(CountRegionOps(model->Subgraph(0)), 0);
  EXPECT_EQ(model->NumSubgraphs(), num_subgraphs);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), partitions.NumSubgraphs());
  EXPECT_EQ(partitions.NumEntrySubgraphs(), dispatch_ops.size());
  ExpectValid(*model, partitions);

  // The partitioned model still serializes and reloads with valid indices.
  LITERT_ASSERT_OK_AND_ASSIGN(auto serialized,
                              SerializeModel(std::move(*model)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto reloaded,
                              LoadModelFromBuffer(std::move(serialized)));
  LITERT_EXPECT_OK(ValidateSubgraphRefs(*reloaded));
}

TEST_P(RegionModelPartitionTest, RegionsOffloadedWithRegionSupport) {
  auto model = LoadModel(GetParam());
  const size_t num_subgraphs = model->NumSubgraphs();
  FakePartitioner partitioner(SelectAll());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result,
      PartitionModel(partitioner.Fn(), WithRegionSupport(), *model));
  auto& [dispatch_ops, partitions] = result;

  // Everything is offloaded as a single partition carrying all regions.
  EXPECT_THAT(partitioner.Visited(), ElementsAre("main"));
  ASSERT_EQ(dispatch_ops.size(), 1);
  EXPECT_EQ(model->NumSubgraphs(), 1);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 1);
  EXPECT_EQ(partitions.NumSubgraphs(), num_subgraphs);
  ExpectValid(*model, partitions);

  LITERT_ASSERT_OK_AND_ASSIGN(auto serialized,
                              SerializeModel(std::move(partitions)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto reloaded,
                              LoadModelFromBuffer(std::move(serialized)));
  LITERT_EXPECT_OK(ValidateSubgraphRefs(*reloaded));
}

// shlo_case_region is not covered: the model loader does not support
// STABLEHLO_CASE (op code above kLiteRtOpCodeShloComposite).
// tfl_call_once_region is not covered: DCE treats the input- and output-less
// CALL_ONCE op as dead, so outlining any op of its subgraph drops it.
// TODO: Keep side-effecting ops without outputs in DCE.
INSTANTIATE_TEST_SUITE_P(
    RegionModels, RegionModelPartitionTest,
    ::testing::Values("shlo_reduce_window_region.tflite",
                      "shlo_reduce_region.tflite", "shlo_sort_region.tflite",
                      "shlo_scatter_region.tflite", "shlo_while_region.tflite",
                      "tfl_while_region.tflite", "tfl_if_region.tflite",
                      "tfl_while_shared_body.tflite", "nested_region.tflite"),
    [](const ::testing::TestParamInfo<std::string>& info) {
      return info.param.substr(0, info.param.find('.'));
    });

//
// End to end with the example plugin, which reports region support but only
// selects MUL / SUB.
//

TEST(RegionApplyPluginTest, ComputationBodyStaysOnCpu) {
  // The scatter update computation is a single MUL; it must not be offloaded.
  auto model = LoadModel("shlo_scatter_region.tflite");
  auto plugins = CompilerPlugin::LoadPlugins(
      {litert::testing::GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_TRUE(plugins);
  ASSERT_TRUE(plugins->front().SupportsRegionSubgraphs());

  ApplyPluginsResult apply_result;
  LITERT_ASSERT_OK(ApplyPlugin(plugins->front(), *model, "ExampleSocModel", {},
                               apply_result));

  ASSERT_EQ(model->NumSubgraphs(), 2);
  EXPECT_EQ(CountOps(model->Subgraph(0), kLiteRtOpCodeShloScatter), 1);
  EXPECT_EQ(CountOps(model->Subgraph(0), kLiteRtOpCodeTflCustom), 1);
  EXPECT_EQ(CountOps(model->Subgraph(1), kLiteRtOpCodeTflCustom), 0);
  EXPECT_EQ(CountOps(model->Subgraph(1), kLiteRtOpCodeTflMul), 1);
  LITERT_EXPECT_OK(ValidateSubgraphRefs(*model));

  LITERT_ASSERT_OK_AND_ASSIGN(auto serialized,
                              SerializeModel(std::move(*model)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto reloaded,
                              LoadModelFromBuffer(std::move(serialized)));
  LITERT_EXPECT_OK(ValidateSubgraphRefs(*reloaded));
}

TEST(RegionApplyPluginTest, ControlFlowBodyPartitions) {
  auto model = LoadModel("tfl_while_region.tflite");
  auto plugins = CompilerPlugin::LoadPlugins(
      {litert::testing::GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_TRUE(plugins);

  ApplyPluginsResult apply_result;
  LITERT_ASSERT_OK(ApplyPlugin(plugins->front(), *model, "ExampleSocModel", {},
                               apply_result));

  ASSERT_EQ(model->NumSubgraphs(), 3);
  EXPECT_EQ(CountOps(model->Subgraph(0), kLiteRtOpCodeTflWhile), 1);
  EXPECT_EQ(CountOps(model->Subgraph(0), kLiteRtOpCodeTflCustom), 1);
  EXPECT_EQ(CountOps(model->Subgraph(2), kLiteRtOpCodeTflCustom), 1);
  for (LiteRtSubgraph sg : model->Subgraphs()) {
    for (LiteRtOp op : sg->Ops()) {
      if (op->OpCode() == kLiteRtOpCodeTflCustom) {
        EXPECT_TRUE(model->FindOpAsset(op));
      }
    }
  }
  LITERT_EXPECT_OK(ValidateSubgraphRefs(*model));
}

TEST(RegionApplyPluginTest, PrivateRegionSubgraphsAreNotCompiledAsCalls) {
  // A partitions model with one entry and one private region subgraph. The
  // region holds an op the example plugin cannot compile, so compilation only
  // succeeds if the plugin restricts itself to the entry subgraphs.
  auto model = LoadModel("mul_simple.tflite");
  auto plugins = CompilerPlugin::LoadPlugins(
      {litert::testing::GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_TRUE(plugins);
  LITERT_ASSERT_OK_AND_ASSIGN(auto partition_result,
                              PartitionModel(plugins->front(), *model));
  ASSERT_EQ(partition_result.first.size(), 1);
  auto& region = partition_result.second.EmplaceSubgraph();
  region.SetName("region");
  region.EmplaceOp().SetOpCode(kLiteRtOpCodeTflMaximum);
  partition_result.second.SetNumEntrySubgraphs(1);
  LiteRtOp dispatch_op = partition_result.first.front();

  ApplyPluginsResult apply_result;
  LITERT_ASSERT_OK(ApplyPluginWithPartition(plugins->front(), *model,
                                            std::move(partition_result),
                                            "ExampleSocModel", apply_result));
  EXPECT_TRUE(model->FindOpAsset(dispatch_op));
}

TEST(RegionApplyPluginTest, RejectsEntryCountMismatch) {
  auto model = LoadModel("mul_simple.tflite");
  auto plugins = CompilerPlugin::LoadPlugins(
      {litert::testing::GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_TRUE(plugins);
  LITERT_ASSERT_OK_AND_ASSIGN(auto partition_result,
                              PartitionModel(plugins->front(), *model));
  ASSERT_EQ(partition_result.first.size(), 1);
  partition_result.second.EmplaceSubgraph();
  partition_result.second.SetNumEntrySubgraphs(2);

  ApplyPluginsResult apply_result;
  auto status = ApplyPluginWithPartition(plugins->front(), *model,
                                         std::move(partition_result),
                                         "ExampleSocModel", apply_result);
  ASSERT_FALSE(status);
  EXPECT_EQ(status.Error().Status(), kLiteRtStatusErrorRuntimeFailure);
}

}  // namespace
}  // namespace litert::internal
