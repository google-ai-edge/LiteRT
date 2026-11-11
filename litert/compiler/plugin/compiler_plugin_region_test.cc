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

// Tests partitioning of models whose ops reference other subgraphs (region
// bodies, control-flow branches, composite decompositions), using an
// in-process fake compiler plugin.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/model/model.h"
#include "litert/core/model/subgraph_call_graph.h"
#include "litert/core/model/subgraph_refs.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "tflite/converter/schema/schema_generated.h"

// State of the fake plugin. The handle type is opaque to the runtime.
struct LiteRtCompilerPluginT {
  // Decides which ops `Partition` selects.
  std::function<bool(const LiteRtOpT&)> select = [](const LiteRtOpT&) {
    return false;
  };
  bool supports_region_subgraphs = false;
  // Emits a call for every subgraph instead of every entry subgraph.
  bool emit_call_per_subgraph = false;
  // Makes `Compile` fail.
  bool fail_compile = false;

  // Recorded by `Partition`.
  std::vector<const LiteRtSubgraphT*> partitioned_subgraphs;

  // Recorded by `Compile`.
  size_t compiled_num_subgraphs = 0;
  size_t compiled_num_entry_subgraphs = 0;
  std::vector<std::string> compiled_subgraph_names;
  bool compiled_refs_valid = false;
};

struct LiteRtCompiledResultT {
  std::string byte_code = "fake_byte_code";
  std::vector<std::string> calls;
};

namespace litert::internal {

// Gives tests access to the private parts of `CompilerPlugin`.
class CompilerPluginFriend {
 public:
  static CompilerPlugin Wrap(const LiteRtCompilerPluginInterface_V1* api,
                             LiteRtCompilerPlugin handle) {
    CompilerPlugin plugin;
    plugin.plugin_api_v1_ = api;
    plugin.plugin_handle_ = handle;
    return plugin;
  }
};

namespace {

using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::litert::IsError;

LiteRtStatus FakePartition(LiteRtCompilerPlugin plugin, const char* soc_model,
                           LiteRtSubgraph subgraph, LiteRtOpList selected_ops) {
  plugin->partitioned_subgraphs.push_back(subgraph);
  for (LiteRtOp op : subgraph->Ops()) {
    if (plugin->select(*op)) {
      selected_ops->Push(op, 0);
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus FakeCompile(LiteRtCompilerPlugin plugin, const char* soc_model,
                         LiteRtModel partitions,
                         LiteRtCompiledResult* compiled_result) {
  if (plugin->fail_compile) {
    return kLiteRtStatusErrorCompilation;
  }
  plugin->compiled_num_subgraphs = partitions->NumSubgraphs();
  plugin->compiled_num_entry_subgraphs = partitions->NumEntrySubgraphs();
  plugin->compiled_subgraph_names.clear();
  for (const LiteRtSubgraphT* subgraph : partitions->Subgraphs()) {
    plugin->compiled_subgraph_names.push_back(std::string(subgraph->Name()));
  }
  plugin->compiled_refs_valid = BuildSubgraphCallGraph(*partitions).HasValue();
  auto* result = new LiteRtCompiledResultT();
  const size_t num_calls = plugin->emit_call_per_subgraph
                               ? partitions->NumSubgraphs()
                               : partitions->NumEntrySubgraphs();
  for (size_t i = 0; i < num_calls; ++i) {
    result->calls.push_back(absl::StrCat("partition_", i));
  }
  *compiled_result = result;
  return kLiteRtStatusOk;
}

void FakeDestroyCompiledResult(LiteRtCompiledResult result) { delete result; }

LiteRtStatus FakeByteCode(LiteRtCompiledResult result, LiteRtParamIndex index,
                          const void** byte_code, size_t* byte_code_size) {
  *byte_code = result->byte_code.data();
  *byte_code_size = result->byte_code.size();
  return kLiteRtStatusOk;
}

LiteRtStatus FakeNumByteCode(LiteRtCompiledResult result,
                             LiteRtParamIndex* num_byte_code) {
  *num_byte_code = 1;
  return kLiteRtStatusOk;
}

LiteRtStatus FakeCallInfo(LiteRtCompiledResult result, LiteRtParamIndex index,
                          const void** call_info, size_t* call_info_size,
                          LiteRtParamIndex* byte_code_idx) {
  if (index >= result->calls.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *call_info = result->calls[index].data();
  *call_info_size = result->calls[index].size();
  *byte_code_idx = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus FakeNumCalls(LiteRtCompiledResult result,
                          LiteRtParamIndex* num_calls) {
  *num_calls = result->calls.size();
  return kLiteRtStatusOk;
}

LiteRtStatus FakeCapabilities(LiteRtCompilerPlugin plugin,
                              uint64_t* capabilities) {
  *capabilities = plugin->supports_region_subgraphs
                      ? kLiteRtCompilerPluginCapabilityRegionSubgraphs
                      : kLiteRtCompilerPluginCapabilityNone;
  return kLiteRtStatusOk;
}

const char* FakeSocManufacturer() { return "FakeSocManufacturer"; }

LiteRtStatus FakeSupportedHardware(LiteRtCompilerPlugin plugin,
                                   LiteRtHwAccelerators* supported_hardware) {
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

constexpr LiteRtCompilerPluginInterface_V1 MakeFakeInterface(
    uint16_t struct_size) {
  LiteRtCompilerPluginInterface_V1 api = {};
  api.abi_header.struct_size = struct_size;
  api.abi_header.major_version = 1;
  api.abi_header.minor_version = 1;
  api.get_compiler_plugin_soc_manufacturer = FakeSocManufacturer;
  api.get_compiler_plugin_supported_hardware = FakeSupportedHardware;
  api.compiler_plugin_partition = FakePartition;
  api.compiler_plugin_compile = FakeCompile;
  api.destroy_compiled_result = FakeDestroyCompiledResult;
  api.get_compiled_result_byte_code = FakeByteCode;
  api.get_compiled_result_num_byte_code = FakeNumByteCode;
  api.get_compiled_result_call_info = FakeCallInfo;
  api.get_num_compiled_result_calls = FakeNumCalls;
  api.get_compiler_plugin_capabilities = FakeCapabilities;
  return api;
}

constexpr LiteRtCompilerPluginInterface_V1 kFakeInterface =
    MakeFakeInterface(sizeof(LiteRtCompilerPluginInterface_V1));

// A 1.0 interface table: it ends before `get_compiler_plugin_capabilities`.
constexpr LiteRtCompilerPluginInterface_V1 kFakeInterfaceV1_0 =
    MakeFakeInterface(offsetof(LiteRtCompilerPluginInterface_V1,
                               get_compiler_plugin_capabilities));

bool IsDispatchOp(const LiteRtOpT& op) {
  return op.OpCode() == kLiteRtOpCodeTflCustom &&
         GetTflOpCodeInd(op) == kDispatchOpCodeTflInd;
}

size_t NumDispatchOps(const LiteRtSubgraphT& subgraph) {
  return absl::c_count_if(subgraph.Ops(),
                          [](const LiteRtOp op) { return IsDispatchOp(*op); });
}

// The ops of `subgraph` that reference other subgraphs.
std::vector<LiteRtOp> RegionOps(const LiteRtSubgraphT& subgraph) {
  std::vector<LiteRtOp> ops;
  for (LiteRtOp op : subgraph.Ops()) {
    if (IsRegionOp(*op)) {
      ops.push_back(op);
    }
  }
  return ops;
}

auto SelectCodes(absl::flat_hash_set<LiteRtOpCode> codes) {
  return [codes = std::move(codes)](const LiteRtOpT& op) {
    return codes.contains(op.OpCode());
  };
}

bool SelectAll(const LiteRtOpT& op) { return true; }

void SetCompositeOptions(LiteRtOpT& op, absl::string_view name,
                         int32_t decomposition) {
  tflite::StableHLOCompositeOptionsT options;
  options.name = std::string(name);
  options.decomposition_subgraph_index = decomposition;
  TflOptions2 tfl_options;
  tfl_options.type = tflite::BuiltinOptions2_StableHLOCompositeOptions;
  tfl_options.Set(std::move(options));
  SetTflOptions2(op, std::move(tfl_options));
}

// Adds an odml.cpu_call to `subgraph` that can be inlined: it reads `input`
// and its decomposition (a new subgraph) applies ABS to it. Returns the
// cpu_call op.
LiteRtOpT& AddInlinableCpuCall(LiteRtModelT& model, LiteRtSubgraphT& subgraph,
                               LiteRtTensorT& input) {
  LiteRtSubgraphT& decomposition = model.EmplaceSubgraph();
  LiteRtTensorT& decomposition_input = decomposition.EmplaceTensor();
  decomposition_input.SetType(input.Type());
  LiteRtTensorT& decomposition_output = decomposition.EmplaceTensor();
  decomposition_output.SetType(input.Type());
  decomposition.Inputs().push_back(&decomposition_input);
  decomposition.Outputs().push_back(&decomposition_output);
  LiteRtOpT& abs = decomposition.EmplaceOp();
  abs.SetOpCode(kLiteRtOpCodeTflAbs);
  AttachInput(&decomposition_input, abs);
  AttachOutput(&decomposition_output, abs);

  LiteRtTensorT& output = subgraph.EmplaceTensor();
  output.SetType(input.Type());
  LiteRtOpT& cpu_call = subgraph.EmplaceOp();
  cpu_call.SetOpCode(kLiteRtOpCodeShloComposite);
  SetCompositeOptions(cpu_call, CompositeOptions::kCpuCall,
                      static_cast<int32_t>(model.NumSubgraphs() - 1));
  AttachInput(&input, cpu_call);
  AttachOutput(&output, cpu_call);
  return cpu_call;
}

bool HasOpCode(const LiteRtSubgraphT& subgraph, LiteRtOpCode code) {
  return absl::c_any_of(subgraph.Ops(), [code](const LiteRtOp op) {
    return op->OpCode() == code;
  });
}

// Makes `op` a STABLEHLO_REDUCE_WINDOW over `body`.
void SetReduceWindowOptions(LiteRtOpT& op, int32_t body) {
  op.SetOpCode(kLiteRtOpCodeShloWindow);
  tflite::StablehloReduceWindowOptionsT options;
  options.body_subgraph_index = body;
  TflOptions2 tfl_options;
  tfl_options.type = tflite::BuiltinOptions2_StablehloReduceWindowOptions;
  tfl_options.Set(std::move(options));
  SetTflOptions2(op, std::move(tfl_options));
}

// Gives `subgraph` a signature with its own inputs and outputs.
void AddSignature(LiteRtModelT& model, LiteRtSubgraphT& subgraph,
                  absl::string_view key) {
  std::vector<std::string> input_names;
  std::vector<LiteRtTensor> inputs;
  for (LiteRtTensor input : subgraph.Inputs()) {
    input_names.push_back(absl::StrCat("in", inputs.size()));
    inputs.push_back(input);
  }
  std::vector<std::string> output_names;
  std::vector<LiteRtTensor> outputs;
  for (LiteRtTensor output : subgraph.Outputs()) {
    output_names.push_back(absl::StrCat("out", outputs.size()));
    outputs.push_back(output);
  }
  model.EmplaceSignature(&subgraph, input_names, inputs, output_names, outputs,
                         std::string(key));
}

// Adds an odml.cpu_call to `subgraph` that cannot be inlined: it has no
// inputs, while its decomposition (a new subgraph) has one.
void AddUninlinableCpuCall(LiteRtModelT& model, LiteRtSubgraphT& subgraph) {
  LiteRtSubgraphT& decomposition = model.EmplaceSubgraph();
  LiteRtTensorT& input = decomposition.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));
  decomposition.Inputs().push_back(&input);
  LiteRtOpT& cpu_call = subgraph.EmplaceOp();
  cpu_call.SetOpCode(kLiteRtOpCodeShloComposite);
  SetCompositeOptions(cpu_call, CompositeOptions::kCpuCall,
                      static_cast<int32_t>(model.NumSubgraphs() - 1));
}

// The first op in `subgraph` with `code`, or null.
LiteRtOp FindOp(const LiteRtSubgraphT& subgraph, LiteRtOpCode code) {
  for (LiteRtOp op : subgraph.Ops()) {
    if (op->OpCode() == code) {
      return op;
    }
  }
  return nullptr;
}

// Makes `op` a TFL_WHILE over `cond` and `body`.
void SetWhileOptions(LiteRtOpT& op, int32_t cond, int32_t body) {
  op.SetOpCode(kLiteRtOpCodeTflWhile);
  tflite::WhileOptionsT while_options;
  while_options.cond_subgraph_index = cond;
  while_options.body_subgraph_index = body;
  TflOptions tfl_options;
  tfl_options.type = tflite::BuiltinOptions_WhileOptions;
  tfl_options.Set(std::move(while_options));
  SetTflOptions(op, std::move(tfl_options));
}

// Turns `op` into a composite `name` whose decomposition is a new subgraph
// with `op`'s inputs and outputs and a copy of `op`. Returns the index of the
// decomposition.
int32_t WrapInComposite(LiteRtModelT& model, LiteRtOpT& op,
                        absl::string_view name) {
  LiteRtSubgraphT& decomposition = model.EmplaceSubgraph();
  LiteRtOpT& copy = decomposition.EmplaceOp();
  CloneTo(op, copy);
  for (LiteRtTensor input : op.Inputs()) {
    LiteRtTensorT& tensor = decomposition.EmplaceTensor();
    tensor.SetType(input->Type());
    decomposition.Inputs().push_back(&tensor);
    AttachInput(&tensor, copy);
  }
  for (LiteRtTensor output : op.Outputs()) {
    LiteRtTensorT& tensor = decomposition.EmplaceTensor();
    tensor.SetType(output->Type());
    decomposition.Outputs().push_back(&tensor);
    AttachOutput(&tensor, copy);
  }
  const int32_t index = static_cast<int32_t>(model.NumSubgraphs() - 1);
  ClearTflOptions(op);
  op.SetOpCode(kLiteRtOpCodeShloComposite);
  SetCompositeOptions(op, name, index);
  return index;
}

bool HasComposite(const LiteRtSubgraphT& subgraph) {
  return HasOpCode(subgraph, kLiteRtOpCodeShloComposite);
}

class RegionPartitionTest : public ::testing::Test {
 protected:
  CompilerPlugin MakePlugin(
      const LiteRtCompilerPluginInterface_V1* api = &kFakeInterface) {
    return CompilerPluginFriend::Wrap(api, &state_);
  }

  // Selects ops with one of `codes`, and records in `saw_abs_` whether
  // `Partition` was ever shown an ABS op (the body of the cpu_calls that
  // `AddInlinableCpuCall` adds).
  void SelectCodesAndWatchForAbs(absl::flat_hash_set<LiteRtOpCode> codes) {
    state_.select = [this, codes = std::move(codes)](const LiteRtOpT& op) {
      saw_abs_ |= op.OpCode() == kLiteRtOpCodeTflAbs;
      return codes.contains(op.OpCode());
    };
  }

  LiteRtCompilerPluginT state_;
  bool saw_abs_ = false;
};

TEST_F(RegionPartitionTest, OldInterfaceDoesNotReportCapability) {
  state_.supports_region_subgraphs = true;
  EXPECT_FALSE(MakePlugin(&kFakeInterfaceV1_0).SupportsRegionSubgraphs());
  EXPECT_TRUE(MakePlugin().SupportsRegionSubgraphs());
}

// Bodies of ops that interpret them internally are never partitioned. The
// plugin lacks the region capability here, so the region op stays on CPU.
class AtomicRegionTest
    : public RegionPartitionTest,
      public ::testing::WithParamInterface<absl::string_view> {};

TEST_P(AtomicRegionTest, BodyIsNeverPartitioned) {
  auto model_wrap = testing::LoadTestFileModel(GetParam());
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 2);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph body = model.Subgraphs()[1];
  const size_t num_body_ops = body->Ops().size();
  state_.select = SelectAll;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(state_.partitioned_subgraphs, Each(main));
  EXPECT_THAT(model.Subgraphs(), ElementsAre(main, body));
  EXPECT_EQ(body->Ops().size(), num_body_ops);
  EXPECT_EQ(NumDispatchOps(*body), 0);
  // The region op stays on CPU and still references the body.
  const auto region_ops = RegionOps(*main);
  ASSERT_THAT(region_ops, SizeIs(1));
  EXPECT_THAT(SubgraphRefs(*region_ops[0]), ElementsAre(1));
  // Other ops are offloaded, and no helper subgraphs are shipped.
  EXPECT_THAT(result.first, Not(IsEmpty()));
  EXPECT_EQ(result.second.NumEntrySubgraphs(), result.second.NumSubgraphs());
}

INSTANTIATE_TEST_SUITE_P(ComputationRegions, AtomicRegionTest,
                         ::testing::Values("shlo_reduce_region.tflite",
                                           "shlo_reduce_window_region.tflite",
                                           "shlo_scatter_region.tflite",
                                           "shlo_sort_region.tflite"));

TEST_F(RegionPartitionTest, CpuWhileBodyIsPartitioned) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  // main, cond and body were each partitioned once.
  EXPECT_THAT(state_.partitioned_subgraphs, SizeIs(3));
  ASSERT_EQ(model.NumSubgraphs(), 3);
  const auto while_ops = RegionOps(*main);
  ASSERT_THAT(while_ops, SizeIs(1));
  EXPECT_THAT(SubgraphRefs(*while_ops[0]), ElementsAre(1, 2));
  EXPECT_EQ(NumDispatchOps(*main), 1);
  EXPECT_EQ(NumDispatchOps(model.Subgraph(2)), 1);
  EXPECT_THAT(result.first, SizeIs(2));
  EXPECT_EQ(result.second.NumSubgraphs(), 2);
}

TEST_F(RegionPartitionTest, WhileSelectionIsDroppedWithoutCapability) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  state_.select = SelectCodes({kLiteRtOpCodeTflWhile, kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(RegionOps(*main), SizeIs(1));
  EXPECT_EQ(NumDispatchOps(model.Subgraph(2)), 1);
  EXPECT_EQ(result.second.NumEntrySubgraphs(), result.second.NumSubgraphs());
}

TEST_F(RegionPartitionTest, OffloadedWhileTakesItsBodies) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph cond = model.Subgraphs()[1];
  LiteRtSubgraph body = model.Subgraphs()[2];
  state_.supports_region_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeTflWhile, kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  // The bodies of the offloaded WHILE are not partitioned on their own.
  EXPECT_THAT(state_.partitioned_subgraphs, Each(main));
  EXPECT_THAT(model.Subgraphs(), ElementsAre(main));
  ASSERT_THAT(result.first, SizeIs(1));
  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumSubgraphs(), 3);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 1);
  EXPECT_THAT(partitions.Subgraphs(), ElementsAre(::testing::_, cond, body));
  EXPECT_EQ(NumDispatchOps(*body), 0);
  // Nothing is added to the signatures of the partitions model.
  EXPECT_THAT(partitions.Signatures(), IsEmpty());
  const auto while_ops = RegionOps(partitions.Subgraph(0));
  ASSERT_THAT(while_ops, SizeIs(1));
  EXPECT_THAT(SubgraphRefs(*while_ops[0]), ElementsAre(1, 2));
  EXPECT_TRUE(BuildSubgraphCallGraph(partitions).HasValue());
}

TEST_F(RegionPartitionTest, BodySharedByCpuAndNpuWhileIsCloned) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_shared_body.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph cond = model.Subgraphs()[1];
  LiteRtSubgraph body = model.Subgraphs()[2];
  const size_t num_body_ops = body->Ops().size();
  // Offload only the first WHILE in main, and MUL ops elsewhere.
  bool selected_while = false;
  state_.supports_region_subgraphs = true;
  state_.select = [&](const LiteRtOpT& op) {
    if (op.OpCode() == kLiteRtOpCodeTflWhile && !selected_while) {
      selected_while = true;
      return true;
    }
    return op.OpCode() == kLiteRtOpCodeTflMul &&
           !absl::c_linear_search(main->Ops(), &op);
  };

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  // The CPU WHILE keeps the originals, which are partitioned.
  EXPECT_THAT(model.Subgraphs(), ElementsAre(main, cond, body));
  const auto cpu_while_ops = RegionOps(*main);
  ASSERT_THAT(cpu_while_ops, SizeIs(1));
  EXPECT_THAT(SubgraphRefs(*cpu_while_ops[0]), ElementsAre(1, 2));
  EXPECT_EQ(NumDispatchOps(*body), 1);

  // The offloaded WHILE gets unpartitioned copies.
  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 2);
  ASSERT_EQ(partitions.NumSubgraphs(), 4);
  const auto& body_clone = partitions.Subgraph(3);
  EXPECT_EQ(body_clone.Ops().size(), num_body_ops);
  EXPECT_EQ(NumDispatchOps(body_clone), 0);
  EXPECT_TRUE(BuildSubgraphCallGraph(partitions).HasValue());
}

TEST_F(RegionPartitionTest, NestedAtomicBodyInCpuWhileIsNotPartitioned) {
  auto model_wrap = testing::LoadTestFileModel("nested_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 4);
  LiteRtOp reduce = FindOp(model.Subgraph(2), kLiteRtOpCodeShloReduce);
  ASSERT_NE(reduce, nullptr);
  ASSERT_THAT(SubgraphRefs(*reduce), ElementsAre(3));
  LiteRtSubgraph combiner = model.Subgraphs()[3];
  const size_t num_combiner_ops = combiner->Ops().size();
  state_.select = SelectAll;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));

  EXPECT_THAT(state_.partitioned_subgraphs, Not(Contains(combiner)));
  EXPECT_THAT(state_.partitioned_subgraphs, SizeIs(3));
  EXPECT_EQ(combiner->Ops().size(), num_combiner_ops);
  EXPECT_EQ(NumDispatchOps(*combiner), 0);
  EXPECT_TRUE(BuildSubgraphCallGraph(model).HasValue());
}

TEST_F(RegionPartitionTest, NpuCallWithRegionNeedsCapability) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  const size_t num_subgraphs = model.NumSubgraphs();

  auto plugin = MakePlugin();
  auto result = PartitionModel(plugin, model);

  EXPECT_THAT(result, IsError(kLiteRtStatusErrorUnsupported));
  EXPECT_EQ(model.NumSubgraphs(), num_subgraphs);
  EXPECT_THAT(state_.partitioned_subgraphs, IsEmpty());
}

TEST_F(RegionPartitionTest, NpuCallWithRegionShipsBodyAsHelper) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  state_.supports_region_subgraphs = true;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(model.Subgraphs(), ElementsAre(main));
  ASSERT_THAT(result.first, SizeIs(1));
  EXPECT_TRUE(IsDispatchOp(*result.first[0]));
  auto& partitions = result.second;
  EXPECT_EQ(partitions.NumSubgraphs(), 2);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 1);
  const auto region_ops = RegionOps(partitions.Subgraph(0));
  ASSERT_THAT(region_ops, SizeIs(1));
  EXPECT_THAT(SubgraphRefs(*region_ops[0]), ElementsAre(1));
}

// main: npu_call(decomp3) -> regular composite(decomp2) -> npu_call(decomp1).
// The plugin selects every composite; npu_call selections are ignored and the
// regular composite is outlined. Dispatch op i must compile entry i.
TEST_F(RegionPartitionTest, DispatchOpsMatchEntriesWithNpuCallsAndSelections) {
  auto model_wrap = testing::LoadTestFileModel("multi_composite.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 4);
  LiteRtSubgraphT& main = model.Subgraph(0);
  ASSERT_THAT(main.Ops(), SizeIs(3));
  LiteRtOp first_npu_call = main.Ops()[0];
  LiteRtOp second_npu_call = main.Ops()[2];
  LiteRtSubgraph decomp1 = model.Subgraphs()[1];
  LiteRtSubgraph decomp3 = model.Subgraphs()[3];
  // Swap the decompositions of the two npu_calls, so that the op order differs
  // from the decomposition index order.
  SetCompositeOptions(*first_npu_call, CompositeOptions::kNpuCall, 3);
  SetCompositeOptions(*second_npu_call, CompositeOptions::kNpuCall, 1);
  state_.select = SelectCodes({kLiteRtOpCodeShloComposite});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  const auto& dispatch_ops = result.first;
  const auto& partitions = result.second;
  ASSERT_THAT(dispatch_ops, SizeIs(3));
  ASSERT_EQ(partitions.NumSubgraphs(), 3);
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 3);
  for (LiteRtOp op : dispatch_ops) {
    EXPECT_TRUE(IsDispatchOp(*op));
  }

  // Entries are [outlined..., npu_call...], with the npu_calls in op order.
  EXPECT_TRUE(HasComposite(partitions.Subgraph(0)));
  EXPECT_THAT(partitions.Subgraphs(),
              ElementsAre(::testing::_, decomp3, decomp1));
  EXPECT_THAT(dispatch_ops,
              ElementsAre(::testing::_, first_npu_call, second_npu_call));

  // Dispatch ops are in the same order: the outlined composite consumes the
  // first npu_call's output and feeds the second npu_call.
  auto feeds = [](LiteRtOp producer, LiteRtOp consumer) {
    return producer->Outputs().size() == 1 &&
           absl::c_linear_search(producer->Outputs()[0]->Users(), consumer);
  };
  EXPECT_TRUE(feeds(dispatch_ops[1], dispatch_ops[0]));
  EXPECT_TRUE(feeds(dispatch_ops[0], dispatch_ops[2]));
}

TEST_F(RegionPartitionTest, SubgraphsToPartitionWalksBodiesOfSelectedRoots) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph body = model.Subgraphs()[2];
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  // Only main is selected; the WHILE that stays on CPU in main still has its
  // branches partitioned.
  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              PartitionModel(plugin, model, "", {0}));

  EXPECT_THAT(state_.partitioned_subgraphs, SizeIs(3));
  EXPECT_THAT(state_.partitioned_subgraphs, Contains(body));
  EXPECT_EQ(NumDispatchOps(*main), 1);
  EXPECT_EQ(NumDispatchOps(*body), 1);
  EXPECT_THAT(result.first, SizeIs(2));
}

TEST_F(RegionPartitionTest, SubgraphsToPartitionLeavesOtherRootsUnchanged) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  state_.supports_region_subgraphs = true;
  state_.select = SelectAll;

  auto plugin = MakePlugin();
  // Subgraph 1 is the npu_call body, not a CPU root, so nothing is
  // walked and the npu_call in main is left alone.
  LITERT_ASSERT_OK_AND_ASSIGN(auto result,
                              PartitionModel(plugin, model, "", {1}));

  EXPECT_THAT(state_.partitioned_subgraphs, IsEmpty());
  EXPECT_THAT(result.first, IsEmpty());
  EXPECT_EQ(result.second.NumSubgraphs(), 0);
  EXPECT_EQ(model.NumSubgraphs(), 3);
  ASSERT_THAT(main->Ops(), SizeIs(1));
  EXPECT_EQ(main->Ops().front()->OpCode(), kLiteRtOpCodeShloComposite);
}

TEST_F(RegionPartitionTest, NpuCallsSharingADecompositionGetOneEntryEach) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraphT& main = model.Subgraph(0);
  ASSERT_THAT(main.Ops(), SizeIs(1));
  // A second npu_call with the same decomposition and inputs.
  LiteRtOpT& npu_call = *main.Ops().front();
  LiteRtOpT& second_npu_call = MakeClone(main, npu_call);
  for (LiteRtTensor input : npu_call.Inputs()) {
    AttachInput(input, second_npu_call);
  }
  LiteRtTensorT& second_output = main.EmplaceTensor();
  AttachOutput(&second_output, second_npu_call);
  main.Outputs().push_back(&second_output);
  state_.supports_region_subgraphs = true;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(result.first, SizeIs(2));
  auto& partitions = result.second;
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 2);
  // Both entries reference the same reduction body, which ships once.
  ASSERT_EQ(partitions.NumSubgraphs(), 3);
  for (size_t entry = 0; entry < 2; ++entry) {
    const auto region_ops = RegionOps(partitions.Subgraph(entry));
    ASSERT_THAT(region_ops, SizeIs(1));
    EXPECT_THAT(SubgraphRefs(*region_ops[0]), ElementsAre(2));
  }
  EXPECT_THAT(model.Subgraphs(), ElementsAre(&main));
}

// main: npu_call(W) -> mul, where W contains a standard composite whose
// decomposition has a REDUCE_WINDOW. A plugin without region support
// compiles that composite as an opaque op, so the npu_call is accepted.
TEST_F(RegionPartitionTest,
       NpuCallWithRegionInNestedCompositeIsAcceptedWithoutCapability) {
  auto model_wrap = testing::LoadTestFileModel("region_in_composite.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraphT& main = model.Subgraph(0);
  LiteRtOp composite = FindOp(main, kLiteRtOpCodeShloComposite);
  ASSERT_NE(composite, nullptr);
  WrapInComposite(model, *composite, CompositeOptions::kNpuCall);

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(result.first, SizeIs(1));
  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumSubgraphs(), 1);
  for (LiteRtOp op : partitions.Subgraph(0).Ops()) {
    EXPECT_THAT(SubgraphRefs(*op), IsEmpty());
  }
  // The composite's decomposition and the reduction body are unused and pruned.
  EXPECT_THAT(model.Subgraphs(), ElementsAre(&main));
}

// Subgraph 3 is a CPU root excluded by `subgraphs_to_partition`, with
// a REDUCE_WINDOW that interprets the WHILE body. The body is partitioned for
// main's WHILE, so the REDUCE_WINDOW must get an unchanged copy.
TEST_F(RegionPartitionTest, AtomicBodyUsedByUnwalkedRootIsNotPartitioned) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph body = model.Subgraphs()[2];
  const size_t num_body_ops = body->Ops().size();
  LiteRtSubgraphT& other_root = model.EmplaceSubgraph();
  LiteRtOpT& reduce_window = other_root.EmplaceOp();
  SetReduceWindowOptions(reduce_window, /*body=*/2);
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model, "", {0}));

  EXPECT_THAT(state_.partitioned_subgraphs, Not(Contains(&other_root)));
  EXPECT_EQ(NumDispatchOps(*body), 1);
  const auto refs = SubgraphRefs(reduce_window);
  ASSERT_THAT(refs, SizeIs(1));
  LiteRtSubgraph reduce_window_body = model.Subgraphs()[refs[0]];
  EXPECT_NE(reduce_window_body, body);
  EXPECT_EQ(NumDispatchOps(*reduce_window_body), 0);
  EXPECT_EQ(reduce_window_body->Ops().size(), num_body_ops);
  EXPECT_TRUE(BuildSubgraphCallGraph(model).HasValue());
}

// The parameter is the composite name: odml.npu_call or odml.cpu_call.
class CompositeWithoutDecompositionTest
    : public RegionPartitionTest,
      public ::testing::WithParamInterface<absl::string_view> {};

TEST_P(CompositeWithoutDecompositionTest, IsRejectedUpFront) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  ASSERT_THAT(model.Subgraph(0).Ops(), SizeIs(1));
  LiteRtOpT& composite = *model.Subgraph(0).Ops().front();
  SetCompositeOptions(composite, GetParam(), -1);
  state_.supports_region_subgraphs = true;
  state_.select = SelectAll;

  auto plugin = MakePlugin();
  auto result = PartitionModel(plugin, model);

  EXPECT_THAT(result, IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(state_.partitioned_subgraphs, IsEmpty());
  EXPECT_EQ(model.NumSubgraphs(), 3);
  EXPECT_EQ(composite.OpCode(), kLiteRtOpCodeShloComposite);
}

INSTANTIATE_TEST_SUITE_P(
    Composites, CompositeWithoutDecompositionTest,
    ::testing::Values(CompositeOptions::kNpuCall, CompositeOptions::kCpuCall),
    [](const ::testing::TestParamInfo<absl::string_view>& info) -> std::string {
      return info.param == CompositeOptions::kNpuCall ? "NpuCall" : "CpuCall";
    });

TEST_F(RegionPartitionTest,
       NpuCallBodyWithUninlinableCpuCallIsRejectedUpFront) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  AddUninlinableCpuCall(model, model.Subgraph(1));
  const size_t num_subgraphs = model.NumSubgraphs();
  state_.supports_region_subgraphs = true;

  auto plugin = MakePlugin();
  auto result = PartitionModel(plugin, model);

  EXPECT_THAT(result, IsError(kLiteRtStatusErrorUnsupported));
  EXPECT_THAT(state_.partitioned_subgraphs, IsEmpty());
  EXPECT_EQ(model.NumSubgraphs(), num_subgraphs);
}

// The plugin selects main's WHILE, but the WHILE body contains a cpu_call that
// cannot be inlined, so the body cannot ship and the WHILE stays on CPU.
TEST_F(RegionPartitionTest, RegionOpWithUninlinableCpuCallInBodyStaysOnCpu) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  AddUninlinableCpuCall(model, model.Subgraph(2));
  state_.supports_region_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeTflWhile, kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  const auto while_ops = RegionOps(*main);
  ASSERT_THAT(while_ops, SizeIs(1));
  EXPECT_EQ(while_ops[0]->OpCode(), kLiteRtOpCodeTflWhile);
  auto& partitions = result.second;
  for (LiteRtSubgraph subgraph : partitions.Subgraphs()) {
    EXPECT_THAT(RegionOps(*subgraph), IsEmpty());
    EXPECT_FALSE(HasComposite(*subgraph));
  }
}

TEST_F(RegionPartitionTest, ApplyPluginsInvalidatesModelWhenCompileFails) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});
  state_.fail_compile = true;
  std::vector<CompilerPlugin> plugins;
  plugins.push_back(MakePlugin());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result, ApplyPlugins(&model, kLiteRtHwAcceleratorNpu, plugins));

  EXPECT_TRUE(result.model_invalidated);
  EXPECT_EQ(result.num_applied_plugins, 0);
  EXPECT_THAT(result.error_message, Not(IsEmpty()));
}

TEST_F(RegionPartitionTest, ApplyPluginsKeepsModelValidOnSuccess) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});
  std::vector<CompilerPlugin> plugins;
  plugins.push_back(MakePlugin());

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result, ApplyPlugins(&model, kLiteRtHwAcceleratorNpu, plugins));

  EXPECT_FALSE(result.model_invalidated);
  EXPECT_EQ(result.num_applied_plugins, 1);
}

// The npu_call body D contains a cpu_call and is also the body of a CPU
// REDUCE_WINDOW. The npu_call gets its own copy of D, and only that copy has
// the cpu_call inlined; the REDUCE_WINDOW's D is unchanged.
TEST_F(RegionPartitionTest, CpuCallInliningDoesNotTouchSharedNpuCallBody) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraphT& body = model.Subgraph(1);
  LiteRtOp mul = FindOp(body, kLiteRtOpCodeTflMul);
  ASSERT_NE(mul, nullptr);
  AddInlinableCpuCall(model, body, *mul->Outputs().front());
  LiteRtOpT& reduce_window = model.Subgraph(0).EmplaceOp();
  SetReduceWindowOptions(reduce_window, /*body=*/1);
  state_.supports_region_subgraphs = true;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  const auto refs = SubgraphRefs(reduce_window);
  ASSERT_THAT(refs, SizeIs(1));
  const LiteRtSubgraphT& cpu_body = model.Subgraph(refs[0]);
  EXPECT_TRUE(HasComposite(cpu_body));
  EXPECT_FALSE(HasOpCode(cpu_body, kLiteRtOpCodeTflAbs));
  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 1);
  EXPECT_FALSE(HasComposite(partitions.Subgraph(0)));
  EXPECT_TRUE(HasOpCode(partitions.Subgraph(0), kLiteRtOpCodeTflAbs));
  for (LiteRtSubgraph subgraph : partitions.Subgraphs()) {
    EXPECT_FALSE(HasComposite(*subgraph));
  }
}

// A cpu_call in the body of a REDUCE_WINDOW that stays on CPU is left alone:
// the body is pinned and the CPU runs the composite.
TEST_F(RegionPartitionTest, CpuCallInPinnedCpuBodyIsNotInlined) {
  auto model_wrap =
      testing::LoadTestFileModel("shlo_reduce_window_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 2);
  LiteRtSubgraphT& body = model.Subgraph(1);
  AddInlinableCpuCall(model, body, *body.Inputs().front());
  const size_t num_body_ops = body.Ops().size();
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));

  EXPECT_EQ(body.Ops().size(), num_body_ops);
  EXPECT_TRUE(HasComposite(body));
}

// A cpu_call on CPU that cannot be inlined stays a composite (with a
// warning); partitioning still succeeds.
TEST_F(RegionPartitionTest, UninlinableCpuCallOnCpuIsKept) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  LiteRtSubgraph main = model.Subgraphs()[0];
  AddUninlinableCpuCall(model, *main);
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));

  EXPECT_TRUE(HasComposite(*main));
  EXPECT_EQ(NumDispatchOps(*main), 1);
}

// The offloaded WHILE's body contains a cpu_call. It is inlined into the
// shipped body, and the cpu_call's decomposition is not shipped.
TEST_F(RegionPartitionTest, CpuCallInShippedHelperIsInlined) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraphT& body = model.Subgraph(2);
  LiteRtOp mul = FindOp(body, kLiteRtOpCodeTflMul);
  ASSERT_NE(mul, nullptr);
  AddInlinableCpuCall(model, body, *mul->Outputs().front());
  state_.supports_region_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeTflWhile, kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  auto& partitions = result.second;
  EXPECT_EQ(partitions.NumEntrySubgraphs(), 1);
  // The WHILE partition, its cond and its body; not the cpu_call body.
  EXPECT_EQ(partitions.NumSubgraphs(), 3);
  bool shipped_abs = false;
  for (LiteRtSubgraph subgraph : partitions.Subgraphs()) {
    EXPECT_FALSE(HasComposite(*subgraph));
    shipped_abs |= HasOpCode(*subgraph, kLiteRtOpCodeTflAbs);
  }
  EXPECT_TRUE(shipped_abs);
  EXPECT_TRUE(BuildSubgraphCallGraph(partitions).HasValue());
}

// Subgraph 3 is a CPU root excluded by `subgraphs_to_partition` with
// its own WHILE over main's cond and body. Main's WHILE has its body
// partitioned; subgraph 3's WHILE must keep unchanged branches.
TEST_F(RegionPartitionTest, BodySharedWithUnwalkedRootIsCloned) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph body = model.Subgraphs()[2];
  const size_t num_body_ops = body->Ops().size();
  LiteRtSubgraphT& other_root = model.EmplaceSubgraph();
  LiteRtOpT& other_while = other_root.EmplaceOp();
  SetWhileOptions(other_while, /*cond=*/1, /*body=*/2);
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model, "", {0}));

  EXPECT_THAT(state_.partitioned_subgraphs, Not(Contains(&other_root)));
  const auto main_while = RegionOps(*main);
  ASSERT_THAT(main_while, SizeIs(1));
  const auto main_refs = SubgraphRefs(*main_while[0]);
  ASSERT_THAT(main_refs, SizeIs(2));
  EXPECT_EQ(NumDispatchOps(model.Subgraph(main_refs[1])), 1);
  const auto other_refs = SubgraphRefs(other_while);
  ASSERT_THAT(other_refs, SizeIs(2));
  for (const int32_t ref : other_refs) {
    EXPECT_EQ(NumDispatchOps(model.Subgraph(ref)), 0);
  }
  EXPECT_NE(model.Subgraphs()[other_refs[1]], model.Subgraphs()[main_refs[1]]);
  EXPECT_EQ(model.Subgraph(other_refs[1]).Ops().size(), num_body_ops);
}

// The WHILE body also has a signature, and only main is selected. The body is
// still partitioned for main's WHILE, but in a copy: the signature keeps the
// unchanged body.
TEST_F(RegionPartitionTest, UnwalkedSignatureBodyIsPartitionedInACopy) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph body = model.Subgraphs()[2];
  const size_t num_body_ops = body->Ops().size();
  AddSignature(model, *body, "body");
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model, "", {0}));

  EXPECT_THAT(state_.partitioned_subgraphs, Not(Contains(body)));
  EXPECT_EQ(NumDispatchOps(*body), 0);
  EXPECT_EQ(body->Ops().size(), num_body_ops);
  LITERT_ASSERT_OK_AND_ASSIGN(auto signature, model.FindSignature("body"));
  EXPECT_EQ(&signature.get().GetSubgraph(), body);
  const auto main_while = RegionOps(*main);
  ASSERT_THAT(main_while, SizeIs(1));
  const auto main_refs = SubgraphRefs(*main_while[0]);
  ASSERT_THAT(main_refs, SizeIs(2));
  EXPECT_NE(model.Subgraphs()[main_refs[1]], body);
  EXPECT_EQ(NumDispatchOps(model.Subgraph(main_refs[1])), 1);
}

// Subgraph 1 is the body of main's REDUCE_WINDOW and also has a signature, so
// it is walked as a CPU root with no CPU control-flow callers. It is
// partitioned for its signature; the REDUCE_WINDOW gets an unchanged copy.
TEST_F(RegionPartitionTest, WalkedSignatureRootUsedAsAtomicBodyIsCloned) {
  auto model_wrap =
      testing::LoadTestFileModel("shlo_reduce_window_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 2);
  LiteRtSubgraph body = model.Subgraphs()[1];
  const size_t num_body_ops = body->Ops().size();
  AddSignature(model, *body, "body");
  LiteRtOp reduce_window = FindOp(model.Subgraph(0), kLiteRtOpCodeShloWindow);
  ASSERT_NE(reduce_window, nullptr);
  state_.select = SelectAll;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));

  EXPECT_THAT(state_.partitioned_subgraphs, Contains(body));
  EXPECT_GE(NumDispatchOps(*body), 1);
  const auto refs = SubgraphRefs(*reduce_window);
  ASSERT_THAT(refs, SizeIs(1));
  EXPECT_NE(model.Subgraphs()[refs[0]], body);
  EXPECT_EQ(NumDispatchOps(model.Subgraph(refs[0])), 0);
  EXPECT_EQ(model.Subgraph(refs[0]).Ops().size(), num_body_ops);
}

// Without region support, the npu_call body's standard composite is compiled
// as an opaque op, so a cpu_call that cannot be inlined in that composite's
// decomposition does not matter.
TEST_F(RegionPartitionTest,
       UninlinableCpuCallInUnshippedDecompositionIsIgnoredWithoutCapability) {
  auto model_wrap = testing::LoadTestFileModel("nested_composite.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  // main: npu_call(decomp1); decomp1: regular_composite(decomp2).
  AddUninlinableCpuCall(model, model.Subgraph(2));

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumSubgraphs(), 1);
  for (LiteRtOp op : partitions.Subgraph(0).Ops()) {
    EXPECT_THAT(SubgraphRefs(*op), IsEmpty());
  }
}

// Main's only op is a cpu_call whose body holds the npu_call. After inlining,
// that body is dead, so the npu_call's decomposition is not shared and is
// compiled as is.
TEST_F(RegionPartitionTest, NpuCallInlinedFromCpuCallBodyIsNotCloned) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraphT& main = model.Subgraph(0);
  ASSERT_THAT(main.Ops(), SizeIs(1));
  LiteRtSubgraph decomposition = model.Subgraphs()[1];
  LITERT_ASSERT_OK(CloneSubgraph(model, main, "cpu_call_body"));
  SetCompositeOptions(*main.Ops().front(), CompositeOptions::kCpuCall,
                      /*decomposition=*/3);
  state_.supports_region_subgraphs = true;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_EQ(NumDispatchOps(main), 1);
  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 1);
  EXPECT_EQ(&partitions.Subgraph(0), decomposition);
  // The entry and its REDUCE_WINDOW body; no copy of the entry.
  EXPECT_EQ(partitions.NumSubgraphs(), 2);
}

// Without region support, composites in entries are compiled as opaque ops
// and their decompositions are cleared. A cpu_call directly in an npu_call
// body is inlined before that, so its ops are compiled instead.
TEST_F(RegionPartitionTest, CpuCallInNpuCallBodyIsInlinedWithoutCapability) {
  auto model_wrap = testing::LoadTestFileModel("nested_composite.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  // main: npu_call(decomp1); decomp1: regular_composite(decomp2), cpu_call.
  LiteRtSubgraphT& decomposition = model.Subgraph(1);
  AddInlinableCpuCall(model, decomposition, *decomposition.Inputs().front());

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumSubgraphs(), 1);
  const LiteRtSubgraphT& entry = partitions.Subgraph(0);
  EXPECT_TRUE(HasOpCode(entry, kLiteRtOpCodeTflAbs));
  // Only the regular composite is left.
  EXPECT_EQ(absl::c_count_if(entry.Ops(),
                             [](const LiteRtOp op) {
                               return op->OpCode() ==
                                      kLiteRtOpCodeShloComposite;
                             }),
            1);
  for (LiteRtOp op : entry.Ops()) {
    EXPECT_THAT(SubgraphRefs(*op), IsEmpty());
  }
}

// The WHILE body is also the body of a REDUCE_WINDOW in main. The WHILE keeps
// the original, which is partitioned; the REDUCE_WINDOW gets an unchanged copy.
TEST_F(RegionPartitionTest, BodySharedByCpuWhileAndCpuReduceWindowIsCloned) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph body = model.Subgraphs()[2];
  const size_t num_body_ops = body->Ops().size();
  LiteRtOpT& reduce_window = main->EmplaceOp();
  SetReduceWindowOptions(reduce_window, /*body=*/2);
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));

  LiteRtOp while_op = FindOp(*main, kLiteRtOpCodeTflWhile);
  ASSERT_NE(while_op, nullptr);
  ASSERT_THAT(SubgraphRefs(*while_op), SizeIs(2));
  EXPECT_EQ(model.Subgraphs()[SubgraphRefs(*while_op)[1]], body);
  EXPECT_EQ(NumDispatchOps(*body), 1);
  const auto refs = SubgraphRefs(reduce_window);
  ASSERT_THAT(refs, SizeIs(1));
  const LiteRtSubgraphT& copy = model.Subgraph(refs[0]);
  EXPECT_NE(&copy, body);
  EXPECT_EQ(NumDispatchOps(copy), 0);
  EXPECT_EQ(copy.Ops().size(), num_body_ops);
}

// Main offloads the first WHILE and keeps the second on CPU; both share the
// body B, and B has a WHILE of its own over C. B and C are each partitioned
// once, for their CPU callers, and the offloaded WHILE ships unpartitioned
// copies of both.
TEST_F(RegionPartitionTest, NestedSharedBodiesArePartitionedOnceAndCloned) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_shared_body.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph body = model.Subgraphs()[2];
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtSubgraph inner,
                              CloneSubgraph(model, *body, "inner_body"));
  LiteRtOpT& inner_while = body->EmplaceOp();
  SetWhileOptions(inner_while, /*cond=*/1, /*body=*/3);
  bool selected_while = false;
  state_.supports_region_subgraphs = true;
  state_.select = [&](const LiteRtOpT& op) {
    if (op.OpCode() == kLiteRtOpCodeTflWhile && !selected_while &&
        absl::c_linear_search(main->Ops(), &op)) {
      selected_while = true;
      return true;
    }
    return op.OpCode() == kLiteRtOpCodeTflMul &&
           !absl::c_linear_search(main->Ops(), &op);
  };

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  for (const LiteRtSubgraphT* subgraph : state_.partitioned_subgraphs) {
    EXPECT_EQ(absl::c_count(state_.partitioned_subgraphs, subgraph), 1);
  }
  EXPECT_EQ(NumDispatchOps(*body), 1);
  ASSERT_THAT(SubgraphRefs(inner_while), SizeIs(2));
  EXPECT_EQ(model.Subgraphs()[SubgraphRefs(inner_while)[1]], inner);
  EXPECT_EQ(NumDispatchOps(*inner), 1);
  // Entries: the offloaded WHILE, and the MULs outlined from B and C. Helpers:
  // copies of cond, B and C.
  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 3);
  ASSERT_EQ(partitions.NumSubgraphs(), 6);
  for (size_t i = 3; i < partitions.NumSubgraphs(); ++i) {
    EXPECT_EQ(NumDispatchOps(partitions.Subgraph(i)), 0);
  }
  EXPECT_TRUE(BuildSubgraphCallGraph(partitions).HasValue());
  EXPECT_TRUE(BuildSubgraphCallGraph(model).HasValue());
}

// The offloaded WHILE's body h1 uses h2 (a REDUCE body); a CPU REDUCE_WINDOW in
// main also uses h1. Both are cloned: partitions are [entry, cond, h1', h2']
// and the CPU keeps h1 -> h2.
TEST_F(RegionPartitionTest, PackagingClonesSharedHelpersCallersFirst) {
  auto model_wrap = testing::LoadTestFileModel("nested_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 4);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph cond = model.Subgraphs()[1];
  LiteRtSubgraph body = model.Subgraphs()[2];
  LiteRtSubgraph combiner = model.Subgraphs()[3];
  LiteRtOp reduce = FindOp(*body, kLiteRtOpCodeShloReduce);
  ASSERT_NE(reduce, nullptr);
  ASSERT_THAT(SubgraphRefs(*reduce), ElementsAre(3));
  LiteRtOpT& reduce_window = main->EmplaceOp();
  SetReduceWindowOptions(reduce_window, /*body=*/2);
  state_.supports_region_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeTflWhile});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 1);
  ASSERT_EQ(partitions.NumSubgraphs(), 4);
  EXPECT_EQ(&partitions.Subgraph(1), cond);
  LiteRtOp shipped_reduce =
      FindOp(partitions.Subgraph(2), kLiteRtOpCodeShloReduce);
  ASSERT_NE(shipped_reduce, nullptr);
  EXPECT_THAT(SubgraphRefs(*shipped_reduce), ElementsAre(3));
  EXPECT_TRUE(BuildSubgraphCallGraph(partitions).HasValue());
  // The CPU side keeps the originals.
  EXPECT_THAT(model.Subgraphs(), ElementsAre(main, body, combiner));
  EXPECT_THAT(SubgraphRefs(reduce_window), ElementsAre(1));
  EXPECT_THAT(SubgraphRefs(*reduce), ElementsAre(2));
}

// A standard composite whose decomposition holds the WHILE is not selected,
// so it is inlined into main. The WHILE then stays on CPU and its body is
// partitioned in place: standard decompositions are not pinned.
TEST_F(RegionPartitionTest, WhileInlinedFromCompositeHasItsBodyPartitioned) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph body = model.Subgraphs()[2];
  LiteRtOp while_op = FindOp(*main, kLiteRtOpCodeTflWhile);
  ASSERT_NE(while_op, nullptr);
  WrapInComposite(model, *while_op, "odml.regular_composite");
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));

  // The decomposition was pruned; no clone was made.
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtOp inlined_while = FindOp(*main, kLiteRtOpCodeTflWhile);
  ASSERT_NE(inlined_while, nullptr);
  EXPECT_THAT(SubgraphRefs(*inlined_while), ElementsAre(1, 2));
  EXPECT_EQ(model.Subgraphs()[2], body);
  EXPECT_EQ(NumDispatchOps(*body), 1);
  EXPECT_FALSE(HasComposite(*main));
}

// The npu_call body's REDUCE_WINDOW and a CPU WHILE in main share subgraph 2.
// The WHILE keeps the original, which is partitioned; the shipped
// REDUCE_WINDOW gets an unchanged copy.
TEST_F(RegionPartitionTest, NpuCallBodySharedWithCpuWhileIsCloned) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph combiner = model.Subgraphs()[2];
  const size_t num_combiner_ops = combiner->Ops().size();
  LiteRtOpT& cpu_while = main->EmplaceOp();
  SetWhileOptions(cpu_while, /*cond=*/2, /*body=*/2);
  state_.supports_region_subgraphs = true;
  state_.select = [combiner](const LiteRtOpT& op) {
    return absl::c_linear_search(combiner->Ops(), &op);
  };

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(state_.partitioned_subgraphs, Contains(combiner));
  for (const int32_t ref : SubgraphRefs(cpu_while)) {
    EXPECT_EQ(model.Subgraphs()[ref], combiner);
  }
  EXPECT_EQ(NumDispatchOps(*combiner), 1);
  // Entries: the partition outlined from subgraph 2, then the npu_call body.
  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 2);
  LiteRtOp shipped = FindOp(partitions.Subgraph(1), kLiteRtOpCodeShloWindow);
  ASSERT_NE(shipped, nullptr);
  ASSERT_THAT(SubgraphRefs(*shipped), SizeIs(1));
  const LiteRtSubgraphT& copy = partitions.Subgraph(SubgraphRefs(*shipped)[0]);
  EXPECT_EQ(NumDispatchOps(copy), 0);
  EXPECT_EQ(copy.Ops().size(), num_combiner_ops);
}

// The npu_call's decomposition D is also a signature. D is partitioned on
// CPU for its signature, and the npu_call compiles an unchanged copy.
TEST_F(RegionPartitionTest, NpuCallDecompositionWithSignatureIsCloned) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph decomposition = model.Subgraphs()[1];
  AddSignature(model, *decomposition, "decomposition");
  state_.supports_region_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  LITERT_ASSERT_OK_AND_ASSIGN(auto signature,
                              model.FindSignature("decomposition"));
  EXPECT_EQ(&signature.get().GetSubgraph(), decomposition);
  EXPECT_EQ(NumDispatchOps(*decomposition), 1);
  EXPECT_EQ(NumDispatchOps(*main), 1);
  // Entries: the MUL partition outlined from D, then the npu_call's copy.
  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 2);
  const LiteRtSubgraphT& npu_call_entry = partitions.Subgraph(1);
  EXPECT_EQ(NumDispatchOps(npu_call_entry), 0);
  EXPECT_NE(FindOp(npu_call_entry, kLiteRtOpCodeTflMul), nullptr);
}

// A cpu_call in a CPU WHILE body: the body is partitioned without showing
// the cpu_call's body to the plugin, then the cpu_call is inlined.
TEST_F(RegionPartitionTest, PartitionNeverSeesCpuCallBodyInCpuWhileBody) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph body = model.Subgraphs()[2];
  LiteRtOp mul = FindOp(*body, kLiteRtOpCodeTflMul);
  ASSERT_NE(mul, nullptr);
  AddInlinableCpuCall(model, *body, *mul->Outputs().front());
  SelectCodesAndWatchForAbs({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));

  EXPECT_FALSE(saw_abs_);
  EXPECT_THAT(state_.partitioned_subgraphs, Contains(body));
  EXPECT_TRUE(HasOpCode(*body, kLiteRtOpCodeTflAbs));
  EXPECT_FALSE(HasComposite(*body));
}

// A cpu_call in a standard composite's decomposition: inlining the composite
// brings the cpu_call (not its body) into main for the second Partition().
TEST_F(RegionPartitionTest, PartitionNeverSeesCpuCallBodyInInlinedComposite) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtOp while_op = FindOp(*main, kLiteRtOpCodeTflWhile);
  ASSERT_NE(while_op, nullptr);
  LiteRtSubgraphT& decomposition = model.Subgraph(
      WrapInComposite(model, *while_op, "odml.regular_composite"));
  AddInlinableCpuCall(model, decomposition, *decomposition.Inputs().back());
  SelectCodesAndWatchForAbs({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));

  EXPECT_FALSE(saw_abs_);
  EXPECT_TRUE(HasOpCode(*main, kLiteRtOpCodeTflAbs));
  EXPECT_FALSE(HasComposite(*main));
}

// A cpu_call in a signature subgraph that is also the body of an offloaded
// WHILE: the CPU original and the shipped copy both get it inlined, after
// every Partition() call.
TEST_F(RegionPartitionTest,
       PartitionNeverSeesCpuCallBodyInPinnedSignatureRoot) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph body = model.Subgraphs()[2];
  LiteRtOp mul = FindOp(*body, kLiteRtOpCodeTflMul);
  ASSERT_NE(mul, nullptr);
  AddInlinableCpuCall(model, *body, *mul->Outputs().front());
  AddSignature(model, *body, "body");
  state_.supports_region_subgraphs = true;
  SelectCodesAndWatchForAbs({kLiteRtOpCodeTflWhile, kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_FALSE(saw_abs_);
  EXPECT_THAT(state_.partitioned_subgraphs, Contains(body));
  EXPECT_TRUE(HasOpCode(*body, kLiteRtOpCodeTflAbs));
  EXPECT_FALSE(HasComposite(*body));
  bool shipped_abs = false;
  for (LiteRtSubgraph subgraph : result.second.Subgraphs()) {
    EXPECT_FALSE(HasComposite(*subgraph));
    shipped_abs |= HasOpCode(*subgraph, kLiteRtOpCodeTflAbs);
  }
  EXPECT_TRUE(shipped_abs);
}

// A cpu_call in an npu_call decomposition: the decomposition is never
// partitioned, and the entry gets the cpu_call inlined.
TEST_F(RegionPartitionTest, PartitionNeverSeesCpuCallBodyInNpuCallBody) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph decomposition = model.Subgraphs()[1];
  LiteRtOp mul = FindOp(*decomposition, kLiteRtOpCodeTflMul);
  ASSERT_NE(mul, nullptr);
  AddInlinableCpuCall(model, *decomposition, *mul->Outputs().front());
  state_.supports_region_subgraphs = true;
  SelectCodesAndWatchForAbs({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_FALSE(saw_abs_);
  EXPECT_THAT(state_.partitioned_subgraphs, Not(Contains(decomposition)));
  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 1);
  EXPECT_TRUE(HasOpCode(partitions.Subgraph(0), kLiteRtOpCodeTflAbs));
  EXPECT_FALSE(HasComposite(partitions.Subgraph(0)));
}

// The WHILE is wrapped in a cpu_call. The cpu_call is inlined into main after
// the walk, so the WHILE ends up on CPU, but its cond and body (pinned as part
// of the cpu_call's body) are never partitioned.
TEST_F(RegionPartitionTest, WhileInCpuCallBodyKeepsItsBodiesUnpartitioned) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph cond = model.Subgraphs()[1];
  LiteRtSubgraph body = model.Subgraphs()[2];
  LiteRtOp while_op = FindOp(*main, kLiteRtOpCodeTflWhile);
  ASSERT_NE(while_op, nullptr);
  WrapInComposite(model, *while_op, CompositeOptions::kCpuCall);
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));

  EXPECT_THAT(state_.partitioned_subgraphs, Not(Contains(cond)));
  EXPECT_THAT(state_.partitioned_subgraphs, Not(Contains(body)));
  EXPECT_EQ(NumDispatchOps(*body), 0);
  LiteRtOp inlined_while = FindOp(*main, kLiteRtOpCodeTflWhile);
  ASSERT_NE(inlined_while, nullptr);
  EXPECT_THAT(SubgraphRefs(*inlined_while), ElementsAre(1, 2));
  EXPECT_FALSE(HasComposite(*main));
}

// Without region support, an npu_call whose body is a WHILE is rejected
// before anything is changed.
TEST_F(RegionPartitionTest, NpuCallWrappingWhileNeedsCapability) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtOp while_op = FindOp(*main, kLiteRtOpCodeTflWhile);
  ASSERT_NE(while_op, nullptr);
  WrapInComposite(model, *while_op, CompositeOptions::kNpuCall);
  const size_t num_subgraphs = model.NumSubgraphs();
  const size_t num_main_ops = main->Ops().size();
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  auto result = PartitionModel(plugin, model);

  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorUnsupported);
  EXPECT_TRUE(
      absl::StrContains(result.Error().Message(),
                        "kLiteRtCompilerPluginCapabilityRegionSubgraphs"));
  EXPECT_EQ(model.NumSubgraphs(), num_subgraphs);
  EXPECT_EQ(main->Ops().size(), num_main_ops);
  EXPECT_THAT(state_.partitioned_subgraphs, IsEmpty());
}

TEST_F(RegionPartitionTest, RegionInInlinedCompositeStaysAtomic) {
  auto model_wrap = testing::LoadTestFileModel("region_in_composite.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  // Everything but the composite, so that it gets inlined.
  state_.select = [](const LiteRtOpT& op) {
    return op.OpCode() != kLiteRtOpCodeShloComposite;
  };

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));

  // The decomposition was inlined and pruned; its REDUCE_WINDOW stays on CPU
  // with its body.
  EXPECT_THAT(state_.partitioned_subgraphs, Each(main));
  ASSERT_EQ(model.NumSubgraphs(), 2);
  const auto region_ops = RegionOps(*main);
  ASSERT_THAT(region_ops, SizeIs(1));
  EXPECT_EQ(region_ops[0]->OpCode(), kLiteRtOpCodeShloWindow);
  EXPECT_THAT(SubgraphRefs(*region_ops[0]), ElementsAre(1));
  EXPECT_EQ(NumDispatchOps(model.Subgraph(1)), 0);
}

TEST_F(RegionPartitionTest, CallOnceSurvivesOutlining) {
  auto model_wrap = testing::LoadTestFileModel("tfl_call_once_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 2);
  LiteRtSubgraph main = model.Subgraphs()[0];
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(result.first, SizeIs(1));
  const auto region_ops = RegionOps(*main);
  ASSERT_THAT(region_ops, SizeIs(1));
  EXPECT_EQ(region_ops[0]->OpCode(), kLiteRtOpCodeTflCallOnce);
  EXPECT_THAT(SubgraphRefs(*region_ops[0]), ElementsAre(1));
}

// A CALL_ONCE has no tensors, so on its own it cannot become a dispatch op.
TEST_F(RegionPartitionTest, SelectedCallOnceStaysOnCpu) {
  auto model_wrap = testing::LoadTestFileModel("tfl_call_once_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 2);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph init = model.Subgraphs()[1];
  state_.select = SelectCodes({kLiteRtOpCodeTflCallOnce});
  state_.supports_region_subgraphs = true;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(result.first, IsEmpty());
  EXPECT_EQ(NumDispatchOps(*main), 0);
  EXPECT_THAT(model.Subgraphs(), ElementsAre(main, init));
  const auto region_ops = RegionOps(*main);
  ASSERT_THAT(region_ops, SizeIs(1));
  EXPECT_EQ(region_ops[0]->OpCode(), kLiteRtOpCodeTflCallOnce);
  EXPECT_THAT(SubgraphRefs(*region_ops[0]), ElementsAre(1));
}

TEST_F(RegionPartitionTest, CpuIfBranchesArePartitioned) {
  auto model_wrap = testing::LoadTestFileModel("tfl_if_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  state_.select = SelectCodes({kLiteRtOpCodeTflAdd, kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(result.first, SizeIs(3));
  EXPECT_EQ(NumDispatchOps(model.Subgraph(1)), 1);
  EXPECT_EQ(NumDispatchOps(model.Subgraph(2)), 1);
}

TEST_F(RegionPartitionTest, CpuStablehloWhileBodyIsPartitioned) {
  auto model_wrap = testing::LoadTestFileModel("shlo_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(result.first, SizeIs(2));
  EXPECT_THAT(RegionOps(model.Subgraph(0)), SizeIs(1));
}

TEST_F(RegionPartitionTest, AtomicRegionOpSplitsIslandWithoutCapability) {
  auto model_wrap =
      testing::LoadTestFileModel("multi_island_reduce_window.tflite");
  auto& model = *model_wrap.Get();
  state_.select = SelectAll;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(result.first, SizeIs(2));
  EXPECT_EQ(result.second.NumSubgraphs(), 2);
}

TEST_F(RegionPartitionTest, AtomicRegionOpJoinsIslandWithCapability) {
  auto model_wrap =
      testing::LoadTestFileModel("multi_island_reduce_window.tflite");
  auto& model = *model_wrap.Get();
  LiteRtSubgraph main = model.Subgraphs()[0];
  state_.select = SelectAll;
  state_.supports_region_subgraphs = true;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  EXPECT_THAT(result.first, SizeIs(1));
  EXPECT_THAT(model.Subgraphs(), ElementsAre(main));
  EXPECT_EQ(result.second.NumSubgraphs(), 2);
  EXPECT_EQ(result.second.NumEntrySubgraphs(), 1);
}

TEST_F(RegionPartitionTest, PinnedSignatureSubgraphIsPartitionedAndCloned) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  LiteRtSubgraph main = model.Subgraphs()[0];
  LiteRtSubgraph body = model.Subgraphs()[2];
  const size_t num_body_ops = body->Ops().size();
  AddSignature(model, *body, "body");
  state_.supports_region_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeTflWhile, kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  // The signature keeps the original body, which is partitioned on CPU.
  EXPECT_THAT(state_.partitioned_subgraphs, Contains(body));
  EXPECT_THAT(model.Subgraphs(), ElementsAre(main, body));
  LITERT_ASSERT_OK_AND_ASSIGN(auto signature, model.FindSignature("body"));
  EXPECT_EQ(&signature.get().GetSubgraph(), body);
  EXPECT_EQ(NumDispatchOps(*body), 1);
  // The offloaded WHILE references an unpartitioned copy.
  auto& partitions = result.second;
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 2);
  ASSERT_EQ(partitions.NumSubgraphs(), 4);
  LiteRtOp while_op = FindOp(partitions.Subgraph(0), kLiteRtOpCodeTflWhile);
  ASSERT_NE(while_op, nullptr);
  ASSERT_THAT(SubgraphRefs(*while_op), SizeIs(2));
  const LiteRtSubgraphT& copy = partitions.Subgraph(SubgraphRefs(*while_op)[1]);
  EXPECT_EQ(copy.Ops().size(), num_body_ops);
  EXPECT_EQ(NumDispatchOps(copy), 0);
}

TEST_F(RegionPartitionTest, HelperSubgraphsHaveUniqueNonEmptyNames) {
  auto model_wrap = testing::LoadTestFileModel("tfl_while_region.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.NumSubgraphs(), 3);
  // Give both loop subgraphs the same name.
  model.Subgraph(1).SetName("loop");
  model.Subgraph(2).SetName("loop");
  state_.supports_region_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeTflWhile, kLiteRtOpCodeTflMul});

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto result, PartitionModel(plugin, model));

  const auto& partitions = result.second;
  ASSERT_EQ(partitions.NumSubgraphs(), 3);
  ASSERT_EQ(partitions.NumEntrySubgraphs(), 1);
  const std::string cond_name(partitions.Subgraph(1).Name());
  const std::string body_name(partitions.Subgraph(2).Name());
  EXPECT_THAT(cond_name, Not(IsEmpty()));
  EXPECT_THAT(body_name, Not(IsEmpty()));
  EXPECT_NE(cond_name, body_name);
  EXPECT_NE(partitions.Subgraph(0).Name(), cond_name);
  EXPECT_NE(partitions.Subgraph(0).Name(), body_name);
}

TEST_F(RegionPartitionTest, ApplyPluginCompilesOneCallPerEntry) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  state_.supports_region_subgraphs = true;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto partitions, PartitionModel(plugin, model));
  ApplyPluginsResult apply_result;
  LITERT_ASSERT_OK(ApplyPluginWithPartition(
      plugin, model, std::move(partitions), "", apply_result));

  EXPECT_EQ(state_.compiled_num_subgraphs, 2);
  EXPECT_EQ(state_.compiled_num_entry_subgraphs, 1);
  EXPECT_TRUE(state_.compiled_refs_valid);
  ASSERT_THAT(state_.compiled_subgraph_names, SizeIs(2));
  EXPECT_THAT(state_.compiled_subgraph_names[1], Not(IsEmpty()));
  ASSERT_THAT(model.Subgraph(0).Ops(), SizeIs(1));
  EXPECT_TRUE(model.FindOpAsset(model.Subgraph(0).Ops().front()));
}

TEST_F(RegionPartitionTest, ApplyPluginRejectsCallsForHelperSubgraphs) {
  auto model_wrap = testing::LoadTestFileModel("npu_call_region.tflite");
  auto& model = *model_wrap.Get();
  state_.supports_region_subgraphs = true;
  state_.emit_call_per_subgraph = true;

  auto plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto partitions, PartitionModel(plugin, model));
  ApplyPluginsResult apply_result;
  auto status = ApplyPluginWithPartition(plugin, model, std::move(partitions),
                                         "", apply_result);

  EXPECT_THAT(status, IsError(kLiteRtStatusErrorRuntimeFailure));
}

}  // namespace
}  // namespace litert::internal
