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
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
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

struct LiteRtCompilerPluginT {
  std::function<bool(const LiteRtOpT&)> select = [](const LiteRtOpT&) {
    return false;
  };
  std::function<LiteRtStatus(LiteRtModel, LiteRtParamIndex, LiteRtOpList)>
      custom_partition_v2;
  bool accepts_helper_subgraphs = false;
  bool emit_call_per_subgraph = false;
  bool fail_compile = false;

  bool called_partition_v1 = false;
  bool called_partition_v2 = false;
  std::vector<const LiteRtSubgraphT*> partitioned_subgraphs;
  // False if partition_v2 ever got a model with unresolvable references.
  bool partition_model_valid = true;

  bool called_compile_v1 = false;
  bool called_compile_v2 = false;
  size_t compiled_num_subgraphs = 0;
  size_t compiled_num_entry_subgraphs = 0;
  bool compiled_refs_valid = false;
  std::vector<std::vector<int32_t>> compiled_entry0_op_refs;
};

struct LiteRtCompiledResultT {
  std::string byte_code = "fake_byte_code";
  std::vector<std::string> calls;
};

namespace litert::internal {

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

LiteRtStatus SelectOpsInSubgraph(LiteRtCompilerPlugin plugin,
                                 LiteRtSubgraph subgraph,
                                 LiteRtOpList selected_ops) {
  plugin->partitioned_subgraphs.push_back(subgraph);
  for (LiteRtOp op : subgraph->Ops()) {
    if (plugin->select(*op)) {
      selected_ops->Push(op, 0);
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus FakePartition(LiteRtCompilerPlugin plugin, const char*,
                           LiteRtSubgraph subgraph, LiteRtOpList selected_ops) {
  plugin->called_partition_v1 = true;
  return SelectOpsInSubgraph(plugin, subgraph, selected_ops);
}

LiteRtStatus FakePartitionV2(LiteRtCompilerPlugin plugin, const char*,
                             LiteRtModel model, LiteRtParamIndex subgraph_index,
                             LiteRtOpList selected_ops) {
  plugin->called_partition_v2 = true;
  plugin->partition_model_valid &= BuildSubgraphCallGraph(*model).HasValue();
  LiteRtSubgraph subgraph = model->Subgraphs()[subgraph_index];
  if (plugin->custom_partition_v2) {
    plugin->partitioned_subgraphs.push_back(subgraph);
    return plugin->custom_partition_v2(model, subgraph_index, selected_ops);
  }
  return SelectOpsInSubgraph(plugin, subgraph, selected_ops);
}

LiteRtStatus RecordAndEmitCompile(LiteRtCompilerPlugin plugin,
                                  LiteRtModel partitions,
                                  LiteRtParamIndex num_entry_subgraphs,
                                  LiteRtCompiledResult* compiled_result) {
  if (plugin->fail_compile) {
    return kLiteRtStatusErrorCompilation;
  }
  plugin->compiled_num_subgraphs = partitions->NumSubgraphs();
  plugin->compiled_num_entry_subgraphs = num_entry_subgraphs;
  plugin->compiled_refs_valid = BuildSubgraphCallGraph(*partitions).HasValue();
  plugin->compiled_entry0_op_refs.clear();
  if (partitions->NumSubgraphs() > 0) {
    for (const LiteRtOpT* op : partitions->Subgraph(0).Ops()) {
      const auto refs = SubgraphRefs(*op);
      plugin->compiled_entry0_op_refs.emplace_back(refs.begin(), refs.end());
    }
  }
  auto* result = new LiteRtCompiledResultT();
  const size_t num_calls = plugin->emit_call_per_subgraph
                               ? partitions->NumSubgraphs()
                               : num_entry_subgraphs;
  for (size_t i = 0; i < num_calls; ++i) {
    result->calls.push_back(absl::StrCat("partition_", i));
  }
  *compiled_result = result;
  return kLiteRtStatusOk;
}

LiteRtStatus FakeCompile(LiteRtCompilerPlugin plugin, const char*,
                         LiteRtModel partitions, LiteRtCompiledResult* out) {
  plugin->called_compile_v1 = true;
  return RecordAndEmitCompile(plugin, partitions, partitions->NumSubgraphs(),
                              out);
}

LiteRtStatus FakeCompileV2(LiteRtCompilerPlugin plugin, const char*,
                           LiteRtModel partitions, LiteRtParamIndex num_entries,
                           LiteRtCompiledResult* out) {
  plugin->called_compile_v2 = true;
  return RecordAndEmitCompile(plugin, partitions, num_entries, out);
}

void FakeDestroyCompiledResult(LiteRtCompiledResult r) { delete r; }
const char* FakeSocManufacturer() { return "FakeSocManufacturer"; }
LiteRtStatus FakeSupportedHardware(LiteRtCompilerPlugin,
                                   LiteRtHwAccelerators* hw) {
  *hw = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}
LiteRtStatus FakeByteCode(LiteRtCompiledResult r, LiteRtParamIndex,
                          const void** data, size_t* size) {
  *data = r->byte_code.data();
  *size = r->byte_code.size();
  return kLiteRtStatusOk;
}
LiteRtStatus FakeNumByteCode(LiteRtCompiledResult, LiteRtParamIndex* n) {
  *n = 1;
  return kLiteRtStatusOk;
}
LiteRtStatus FakeNumCalls(LiteRtCompiledResult r, LiteRtParamIndex* n) {
  *n = r->calls.size();
  return kLiteRtStatusOk;
}
LiteRtStatus FakeCallInfo(LiteRtCompiledResult r, LiteRtParamIndex idx,
                          const void** data, size_t* size,
                          LiteRtParamIndex* bc_idx) {
  if (idx >= r->calls.size()) return kLiteRtStatusErrorIndexOOB;
  *data = r->calls[idx].data();
  *size = r->calls[idx].size();
  *bc_idx = 0;
  return kLiteRtStatusOk;
}

constexpr LiteRtCompilerPluginInterface_V1 MakeFakeInterface(
    uint16_t struct_size, uint16_t minor_version,
    LiteRtCompilerPluginPartitionV2T partition_v2,
    LiteRtCompilerPluginCompileV2T compile_v2) {
  LiteRtCompilerPluginInterface_V1 api = {};
  api.abi_header = {struct_size, 1, minor_version, 0};
  api.get_compiler_plugin_soc_manufacturer = FakeSocManufacturer;
  api.get_compiler_plugin_supported_hardware = FakeSupportedHardware;
  api.compiler_plugin_partition = FakePartition;
  api.compiler_plugin_compile = FakeCompile;
  api.destroy_compiled_result = FakeDestroyCompiledResult;
  api.get_compiled_result_byte_code = FakeByteCode;
  api.get_compiled_result_num_byte_code = FakeNumByteCode;
  api.get_compiled_result_call_info = FakeCallInfo;
  api.get_num_compiled_result_calls = FakeNumCalls;
  api.compiler_plugin_partition_v2 = partition_v2;
  api.compiler_plugin_compile_v2 = compile_v2;
  return api;
}

constexpr LiteRtCompilerPluginInterface_V1 kFakeInterfaceV1_0 =
    MakeFakeInterface(offsetof(LiteRtCompilerPluginInterface_V1,
                               compiler_plugin_partition_v2),
                      0, FakePartitionV2, FakeCompileV2);
constexpr LiteRtCompilerPluginInterface_V1 kFakeInterfaceV1 = MakeFakeInterface(
    sizeof(LiteRtCompilerPluginInterface_V1), 1, nullptr, nullptr);
constexpr LiteRtCompilerPluginInterface_V1 kFakeInterfacePartitionV2Only =
    MakeFakeInterface(sizeof(LiteRtCompilerPluginInterface_V1), 1,
                      FakePartitionV2, nullptr);
constexpr LiteRtCompilerPluginInterface_V1 kFakeInterfaceCompileV2Only =
    MakeFakeInterface(sizeof(LiteRtCompilerPluginInterface_V1), 1, nullptr,
                      FakeCompileV2);
constexpr LiteRtCompilerPluginInterface_V1 kFakeInterfaceV2 =
    MakeFakeInterface(sizeof(LiteRtCompilerPluginInterface_V1), 1,
                      FakePartitionV2, FakeCompileV2);

// Sets `op` to `code` with builtin options `opts`, held in a `U` (TflOptions
// or TflOptions2).
template <typename U, typename T, typename OptionsT>
void SetOpts(LiteRtOpT& op, LiteRtOpCode code, T type, OptionsT opts) {
  ClearTflOptions(op);
  op.SetOpCode(code);
  U u;
  u.type = type;
  u.Set(std::move(opts));
  if constexpr (std::is_same_v<U, TflOptions>) {
    SetTflOptions(op, std::move(u));
  } else {
    SetTflOptions2(op, std::move(u));
  }
}

void SetCompositeOptions(LiteRtOpT& op, absl::string_view name, int32_t sg) {
  tflite::StableHLOCompositeOptionsT o;
  o.name = std::string(name);
  o.decomposition_subgraph_index = sg;
  SetOpts<TflOptions2>(op, kLiteRtOpCodeShloComposite,
                       tflite::BuiltinOptions2_StableHLOCompositeOptions,
                       std::move(o));
}

void SetWhileOptions(LiteRtOpT& op, int32_t cond, int32_t body) {
  tflite::WhileOptionsT o;
  o.cond_subgraph_index = cond;
  o.body_subgraph_index = body;
  SetOpts<TflOptions>(op, kLiteRtOpCodeTflWhile,
                      tflite::BuiltinOptions_WhileOptions, std::move(o));
}

void SetStablehloWhileOptions(LiteRtOpT& op, int32_t cond, int32_t body) {
  tflite::StablehloWhileOptionsT o;
  o.cond_subgraph_index = cond;
  o.body_subgraph_index = body;
  SetOpts<TflOptions2>(op, kLiteRtOpCodeShloWhile,
                       tflite::BuiltinOptions2_StablehloWhileOptions,
                       std::move(o));
}

void SetIfOptions(LiteRtOpT& op, int32_t then_sg, int32_t else_sg) {
  tflite::IfOptionsT o;
  o.then_subgraph_index = then_sg;
  o.else_subgraph_index = else_sg;
  SetOpts<TflOptions>(op, kLiteRtOpCodeTflIf, tflite::BuiltinOptions_IfOptions,
                      std::move(o));
}

void SetCallOnceOptions(LiteRtOpT& op, int32_t init_sg) {
  tflite::CallOnceOptionsT o;
  o.init_subgraph_index = init_sg;
  SetOpts<TflOptions>(op, kLiteRtOpCodeTflCallOnce,
                      tflite::BuiltinOptions_CallOnceOptions, std::move(o));
}

void SetReduceWindowOptions(LiteRtOpT& op, int32_t body) {
  tflite::StablehloReduceWindowOptionsT o;
  o.body_subgraph_index = body;
  SetOpts<TflOptions2>(op, kLiteRtOpCodeShloWindow,
                       tflite::BuiltinOptions2_StablehloReduceWindowOptions,
                       std::move(o));
}

void SetCaseOptions(LiteRtOpT& op, int32_t branch0, int32_t branch1) {
  tflite::StablehloCaseOptionsT o;
  o.branch_subgraph_indices = {branch0, branch1};
  SetOpts<TflOptions2>(op, kLiteRtOpCodeShloCase,
                       tflite::BuiltinOptions2_StablehloCaseOptions,
                       std::move(o));
}

size_t NumDispatchOps(const LiteRtSubgraphT& sg) {
  return absl::c_count_if(sg.Ops(), [](const LiteRtOp op) {
    return op->OpCode() == kLiteRtOpCodeTflCustom &&
           GetTflOpCodeInd(*op) == kDispatchOpCodeTflInd;
  });
}

LiteRtOp FindOp(const LiteRtSubgraphT& sg, LiteRtOpCode code) {
  for (LiteRtOp op : sg.Ops()) {
    if (op->OpCode() == code) return op;
  }
  return nullptr;
}

bool HasOpCode(const LiteRtSubgraphT& sg, LiteRtOpCode code) {
  return FindOp(sg, code) != nullptr;
}

auto SelectCodes(absl::flat_hash_set<LiteRtOpCode> codes) {
  return [codes = std::move(codes)](const LiteRtOpT& op) {
    return codes.contains(op.OpCode());
  };
}

void AddSignature(LiteRtModelT& model, LiteRtSubgraphT& sg,
                  absl::string_view key) {
  std::vector<std::string> in_names, out_names;
  std::vector<LiteRtTensor> ins(sg.Inputs().begin(), sg.Inputs().end());
  std::vector<LiteRtTensor> outs(sg.Outputs().begin(), sg.Outputs().end());
  for (size_t i = 0; i < ins.size(); ++i)
    in_names.push_back(absl::StrCat("in", i));
  for (size_t i = 0; i < outs.size(); ++i)
    out_names.push_back(absl::StrCat("out", i));
  model.EmplaceSignature(&sg, in_names, ins, out_names, outs, std::string(key));
}

void AddInlinableCpuCall(LiteRtModelT& model, LiteRtSubgraphT& sg,
                         LiteRtTensorT& input) {
  LiteRtSubgraphT& decomp = model.EmplaceSubgraph();
  LiteRtTensorT& in = decomp.EmplaceTensor();
  LiteRtTensorT& out = decomp.EmplaceTensor();
  in.SetType(input.Type());
  out.SetType(input.Type());
  decomp.Inputs().push_back(&in);
  decomp.Outputs().push_back(&out);
  LiteRtOpT& abs = decomp.EmplaceOp();
  abs.SetOpCode(kLiteRtOpCodeTflAbs);
  AttachInput(&in, abs);
  AttachOutput(&out, abs);

  LiteRtTensorT& res = sg.EmplaceTensor();
  res.SetType(input.Type());
  LiteRtOpT& cpu_call = sg.EmplaceOp();
  SetCompositeOptions(cpu_call, CompositeOptions::kCpuCall,
                      static_cast<int32_t>(model.NumSubgraphs() - 1));
  AttachInput(&input, cpu_call);
  AttachOutput(&res, cpu_call);
}

void AddUninlinableCpuCall(LiteRtModelT& model, LiteRtSubgraphT& sg) {
  LiteRtSubgraphT& decomp = model.EmplaceSubgraph();
  LiteRtTensorT& in = decomp.EmplaceTensor();
  in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));
  decomp.Inputs().push_back(&in);
  LiteRtOpT& cpu_call = sg.EmplaceOp();
  SetCompositeOptions(cpu_call, CompositeOptions::kCpuCall,
                      static_cast<int32_t>(model.NumSubgraphs() - 1));
}

void WrapInComposite(LiteRtModelT& model, LiteRtOpT& op,
                     absl::string_view name) {
  LiteRtSubgraphT& decomp = model.EmplaceSubgraph();
  LiteRtOpT& copy = decomp.EmplaceOp();
  CloneTo(op, copy);
  for (LiteRtTensor input : op.Inputs()) {
    LiteRtTensorT& t = decomp.EmplaceTensor();
    t.SetType(input->Type());
    decomp.Inputs().push_back(&t);
    AttachInput(&t, copy);
  }
  for (LiteRtTensor output : op.Outputs()) {
    LiteRtTensorT& t = decomp.EmplaceTensor();
    t.SetType(output->Type());
    decomp.Outputs().push_back(&t);
    AttachOutput(&t, copy);
  }
  SetCompositeOptions(op, name, static_cast<int32_t>(model.NumSubgraphs() - 1));
}

class RegionPartitionTest : public ::testing::Test {
 protected:
  CompilerPlugin MakePlugin(
      const LiteRtCompilerPluginInterface_V1* api = nullptr) {
    if (api == nullptr) {
      api = state_.accepts_helper_subgraphs ? &kFakeInterfaceV2
                                            : &kFakeInterfaceV1;
    }
    return CompilerPluginFriend::Wrap(api, &state_);
  }

  LiteRtModelT& Load(absl::string_view file) {
    models_.push_back(testing::LoadTestFileModel(file));
    EXPECT_TRUE(models_.back());
    return *models_.back().Get();
  }

  void SelectCodesAndWatchForAbs(absl::flat_hash_set<LiteRtOpCode> codes) {
    state_.select = [this, codes = std::move(codes)](const LiteRtOpT& op) {
      saw_abs_ |= op.OpCode() == kLiteRtOpCodeTflAbs;
      return codes.contains(op.OpCode());
    };
  }

  LiteRtCompilerPluginT state_;
  bool saw_abs_ = false;
  std::vector<decltype(testing::LoadTestFileModel(""))> models_;
};

// --- Walk rules: atomic pinning & control-flow walking ---

TEST_F(RegionPartitionTest, AtomicRegionBodiesAreNeverPartitioned) {
  auto& model = Load("shlo_reduce_window_region.tflite");
  LiteRtOp region_op = FindOp(model.Subgraph(0), kLiteRtOpCodeShloWindow);
  const LiteRtSubgraphT* main = &model.Subgraph(0);
  const LiteRtSubgraphT* body = &model.Subgraph(SubgraphRefs(*region_op)[0]);
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});
  CompilerPlugin plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto res, PartitionModel(plugin, model));
  EXPECT_THAT(state_.partitioned_subgraphs, ElementsAre(main));
  EXPECT_EQ(NumDispatchOps(*body), 0);
  EXPECT_EQ(res.first.size(), 1);
}

TEST_F(RegionPartitionTest, CpuControlFlowBodiesArePartitioned) {
  using Setter = void (*)(LiteRtOpT&, int32_t, int32_t);
  for (Setter setter : {SetWhileOptions, SetStablehloWhileOptions, SetIfOptions,
                        SetCaseOptions}) {
    state_ = {};
    auto& model = Load("tfl_while_region.tflite");
    LiteRtOp cf = FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile);
    const auto refs = SubgraphRefs(*cf);
    setter(*cf, refs[0], refs[1]);

    state_.select = SelectCodes({kLiteRtOpCodeTflMul});
    CompilerPlugin plugin = MakePlugin();
    LITERT_ASSERT_OK(PartitionModel(plugin, model));
    EXPECT_THAT(state_.partitioned_subgraphs, SizeIs(3));
    EXPECT_EQ(NumDispatchOps(model.Subgraph(0)), 1);
    EXPECT_EQ(NumDispatchOps(model.Subgraph(refs[1])), 1);
  }
}

TEST_F(RegionPartitionTest, CallOnceStaysOnCpuAndInitSubgraphIsPartitioned) {
  auto& model = Load("tfl_while_region.tflite");
  const int32_t init_idx =
      SubgraphRefs(*FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile))[1];
  SetCallOnceOptions(model.Subgraph(0).EmplaceOp(), init_idx);

  state_.accepts_helper_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeTflCallOnce, kLiteRtOpCodeTflMul});
  CompilerPlugin plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));
  EXPECT_NE(FindOp(model.Subgraph(0), kLiteRtOpCodeTflCallOnce), nullptr);
  EXPECT_EQ(NumDispatchOps(model.Subgraph(0)), 1);
  EXPECT_EQ(NumDispatchOps(model.Subgraph(init_idx)), 1);
}

// --- Shared-body splitting & callers-first cloning ---

TEST_F(RegionPartitionTest, BodySharedByCpuAndNpuWhileIsCloned) {
  auto& model = Load("tfl_while_region.tflite");
  LiteRtOp first = FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile);
  const auto refs = SubgraphRefs(*first);
  LiteRtOpT& second = model.Subgraph(0).EmplaceOp();
  SetStablehloWhileOptions(second, refs[0], refs[1]);
  for (LiteRtTensor in : first->Inputs()) AttachInput(in, second);
  for (LiteRtTensor out : first->Outputs()) {
    LiteRtTensorT& t = model.Subgraph(0).EmplaceTensor();
    t.SetType(out->Type());
    AttachOutput(&t, second);
  }

  state_.accepts_helper_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeShloWhile, kLiteRtOpCodeTflMul});
  CompilerPlugin plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto res, PartitionModel(plugin, model));

  LiteRtOp cpu_while = FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile);
  ASSERT_NE(cpu_while, nullptr);
  EXPECT_EQ(NumDispatchOps(model.Subgraph(SubgraphRefs(*cpu_while)[1])), 1);

  EXPECT_TRUE(state_.partition_model_valid);
  LITERT_ASSERT_OK(BuildSubgraphCallGraph(res.second));
  // Entry 0 is main's island: main is walked first.
  LiteRtOp npu_while = FindOp(res.second.Subgraph(0), kLiteRtOpCodeShloWhile);
  ASSERT_NE(npu_while, nullptr);
  const auto& npu_body = res.second.Subgraph(SubgraphRefs(*npu_while)[1]);
  EXPECT_EQ(NumDispatchOps(npu_body), 0);
  EXPECT_TRUE(HasOpCode(npu_body, kLiteRtOpCodeTflMul));
}

TEST_F(RegionPartitionTest, SignatureRootAlsoAtomicBodyIsSplit) {
  // The signature is partitioned in place; REDUCE_WINDOW gets an unchanged
  // clone.
  auto& model = Load("shlo_reduce_window_region.tflite");
  LiteRtOp rw = FindOp(model.Subgraph(0), kLiteRtOpCodeShloWindow);
  LiteRtSubgraphT& body = model.Subgraph(SubgraphRefs(*rw)[0]);
  AddSignature(model, body, "body_sig");

  state_.select = [](const LiteRtOpT&) { return true; };
  CompilerPlugin plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));
  EXPECT_EQ(NumDispatchOps(body), 1);
  const auto& rw_body = model.Subgraph(SubgraphRefs(*rw)[0]);
  EXPECT_NE(&rw_body, &body);
  EXPECT_EQ(NumDispatchOps(rw_body), 0);
}

TEST_F(RegionPartitionTest, PackagingClonesSharedHelpersCallersFirst) {
  auto& model = Load("tfl_while_region.tflite");
  const int32_t body_idx =
      SubgraphRefs(*FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile))[1];
  model.EmplaceSubgraph().SetName("leaf");
  const int32_t leaf_idx = static_cast<int32_t>(model.NumSubgraphs() - 1);
  SetReduceWindowOptions(model.Subgraph(body_idx).EmplaceOp(), leaf_idx);
  SetReduceWindowOptions(model.Subgraph(0).EmplaceOp(), body_idx);

  state_.accepts_helper_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeTflWhile});
  CompilerPlugin plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto res, PartitionModel(plugin, model));
  LITERT_ASSERT_OK(BuildSubgraphCallGraph(model));
  LITERT_ASSERT_OK(BuildSubgraphCallGraph(res.second));
  EXPECT_EQ(res.second.NumSubgraphs(), 4);  // entry + cond + body + leaf.
}

// --- subgraphs_to_partition ---

TEST_F(RegionPartitionTest, SubgraphsToPartitionExcludesRootsAndClonesShared) {
  // Excluded signature root is never walked even if reached via CPU WHILE.
  {
    auto& model = Load("tfl_while_region.tflite");
    const int32_t body_idx =
        SubgraphRefs(*FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile))[1];
    AddSignature(model, model.Subgraph(body_idx), "excluded_sig");
    const LiteRtSubgraphT* body = &model.Subgraph(body_idx);

    state_.select = SelectCodes({kLiteRtOpCodeTflMul});
    CompilerPlugin plugin = MakePlugin();
    LITERT_ASSERT_OK_AND_ASSIGN(auto res,
                                PartitionModel(plugin, model, "", {0}));
    EXPECT_THAT(state_.partitioned_subgraphs, Not(Contains(body)));
    EXPECT_EQ(NumDispatchOps(*body), 0);
    EXPECT_EQ(res.first.size(), 1);
  }
  // Helper shared by walked root and excluded root is cloned for excluded root.
  {
    state_ = {};
    auto& model = Load("tfl_while_region.tflite");
    LiteRtOp while_op = FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile);
    const auto refs = SubgraphRefs(*while_op);
    LiteRtSubgraphT& other_root = model.EmplaceSubgraph();
    LiteRtOpT& other_while = other_root.EmplaceOp();
    SetWhileOptions(other_while, refs[0], refs[1]);

    state_.select = SelectCodes({kLiteRtOpCodeTflMul});
    CompilerPlugin plugin = MakePlugin();
    LITERT_ASSERT_OK(PartitionModel(plugin, model, "", {0}));
    EXPECT_TRUE(state_.partition_model_valid);
    EXPECT_NE(SubgraphRefs(*while_op)[1], SubgraphRefs(other_while)[1]);
    EXPECT_EQ(NumDispatchOps(model.Subgraph(SubgraphRefs(*while_op)[1])), 1);
    EXPECT_EQ(NumDispatchOps(model.Subgraph(SubgraphRefs(other_while)[1])), 0);
  }
}

// --- npu_call & cpu_call ---

TEST_F(RegionPartitionTest, NpuCallWithRegionNeedsCompileV2) {
  auto& model = Load("npu_call_region.tflite");
  const size_t n = model.NumSubgraphs();
  CompilerPlugin v1 = MakePlugin(&kFakeInterfaceV1);
  EXPECT_THAT(PartitionModel(v1, model),
              IsError(kLiteRtStatusErrorUnsupported));
  EXPECT_EQ(model.NumSubgraphs(), n);
  EXPECT_EQ(NumDispatchOps(model.Subgraph(0)), 0);
}

TEST_F(RegionPartitionTest, NpuCallSharedOrSignedDecompositionIsCloned) {
  auto& model = Load("npu_call_region.tflite");
  LiteRtOp npu0 = FindOp(model.Subgraph(0), kLiteRtOpCodeShloComposite);
  const int32_t decomp_idx = *CompositeDecompositionIndex(*npu0);
  SetCompositeOptions(model.Subgraph(0).EmplaceOp(), CompositeOptions::kNpuCall,
                      decomp_idx);
  AddSignature(model, model.Subgraph(decomp_idx), "decomp_sig");

  state_.accepts_helper_subgraphs = true;
  CompilerPlugin plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto res, PartitionModel(plugin, model));
  EXPECT_EQ(res.first.size(), 2);
  EXPECT_EQ(NumDispatchOps(model.Subgraph(0)), 2);
  EXPECT_EQ(model.Signatures().size(), 2);
  LITERT_ASSERT_OK(BuildSubgraphCallGraph(model));
  LITERT_ASSERT_OK(BuildSubgraphCallGraph(res.second));
}

TEST_F(RegionPartitionTest, CompositeWithoutDecompositionRejectedUpFront) {
  for (absl::string_view name :
       {CompositeOptions::kNpuCall, CompositeOptions::kCpuCall}) {
    auto& model = Load("npu_call_region.tflite");
    SetCompositeOptions(*FindOp(model.Subgraph(0), kLiteRtOpCodeShloComposite),
                        name, -1);
    state_.accepts_helper_subgraphs = true;
    CompilerPlugin plugin = MakePlugin();
    EXPECT_THAT(PartitionModel(plugin, model),
                IsError(kLiteRtStatusErrorInvalidArgument));
  }
}

TEST_F(RegionPartitionTest, CpuCallInliningTwoStageAndGuards) {
  // Stage 1: cpu_call in CPU WHILE body hidden from Partition & inlined;
  // uninlinable cpu_call on CPU kept.
  {
    auto& model = Load("tfl_while_region.tflite");
    LiteRtSubgraphT& body = model.Subgraph(
        SubgraphRefs(*FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile))[1]);
    AddInlinableCpuCall(model, body, *body.Inputs()[1]);
    AddUninlinableCpuCall(model, model.Subgraph(0));

    SelectCodesAndWatchForAbs({kLiteRtOpCodeTflMul});
    CompilerPlugin plugin = MakePlugin();
    LITERT_ASSERT_OK(PartitionModel(plugin, model));
    EXPECT_FALSE(saw_abs_);
    EXPECT_TRUE(HasOpCode(body, kLiteRtOpCodeTflAbs));
    EXPECT_FALSE(HasOpCode(body, kLiteRtOpCodeShloComposite));
    EXPECT_TRUE(HasOpCode(model.Subgraph(0), kLiteRtOpCodeShloComposite));
  }
  // Pinned atomic CPU body keeps cpu_call un-inlined.
  {
    state_ = {};
    auto& model = Load("shlo_reduce_window_region.tflite");
    LiteRtSubgraphT& body = model.Subgraph(
        SubgraphRefs(*FindOp(model.Subgraph(0), kLiteRtOpCodeShloWindow))[0]);
    AddInlinableCpuCall(model, body, *body.Inputs()[0]);
    CompilerPlugin plugin = MakePlugin();
    LITERT_ASSERT_OK(PartitionModel(plugin, model));
    EXPECT_TRUE(HasOpCode(body, kLiteRtOpCodeShloComposite));
    EXPECT_FALSE(HasOpCode(body, kLiteRtOpCodeTflAbs));
  }
  // Stage 2 (v2): cpu_call in shipped NPU helper inlined & body pruned.
  {
    state_ = {};
    state_.accepts_helper_subgraphs = true;
    state_.select = SelectCodes({kLiteRtOpCodeTflWhile});
    auto& model = Load("tfl_while_region.tflite");
    LiteRtSubgraphT& body = model.Subgraph(
        SubgraphRefs(*FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile))[1]);
    AddInlinableCpuCall(model, body, *body.Inputs()[1]);
    CompilerPlugin plugin = MakePlugin();
    LITERT_ASSERT_OK_AND_ASSIGN(auto res, PartitionModel(plugin, model));
    EXPECT_EQ(res.second.NumSubgraphs(), 3);
    LiteRtOp moved_w = FindOp(res.second.Subgraph(0), kLiteRtOpCodeTflWhile);
    const auto& moved_body = res.second.Subgraph(SubgraphRefs(*moved_w)[1]);
    EXPECT_TRUE(HasOpCode(moved_body, kLiteRtOpCodeTflAbs));
    EXPECT_FALSE(HasOpCode(moved_body, kLiteRtOpCodeShloComposite));
  }
  // Stage 2 (v1): cpu_call inside npu_call body inlined & pruned (0 helpers).
  {
    state_ = {};
    auto& model = Load("nested_composite.tflite");
    LiteRtSubgraphT& npu_body = model.Subgraph(*CompositeDecompositionIndex(
        *FindOp(model.Subgraph(0), kLiteRtOpCodeShloComposite)));
    AddInlinableCpuCall(model, npu_body, *npu_body.Inputs()[0]);
    CompilerPlugin v1 = MakePlugin(&kFakeInterfaceV1);
    LITERT_ASSERT_OK_AND_ASSIGN(auto res, PartitionModel(v1, model));
    EXPECT_EQ(res.first.size(), 1);
    EXPECT_EQ(res.second.NumSubgraphs(), 1);
    EXPECT_TRUE(HasOpCode(res.second.Subgraph(0), kLiteRtOpCodeTflAbs));
  }
  // Uninlinable cpu_call in an npu_call body: rejected up front.
  {
    state_ = {};
    state_.accepts_helper_subgraphs = true;
    auto& model = Load("npu_call_region.tflite");
    LiteRtOp npu = FindOp(model.Subgraph(0), kLiteRtOpCodeShloComposite);
    AddUninlinableCpuCall(model,
                          model.Subgraph(*CompositeDecompositionIndex(*npu)));
    CompilerPlugin v2 = MakePlugin(&kFakeInterfaceV2);
    EXPECT_THAT(PartitionModel(v2, model),
                IsError(kLiteRtStatusErrorUnsupported));
  }
  // Uninlinable cpu_call in a selected WHILE's body: the WHILE stays on CPU.
  {
    state_ = {};
    state_.accepts_helper_subgraphs = true;
    state_.select = SelectCodes({kLiteRtOpCodeTflWhile});
    auto& model = Load("tfl_while_region.tflite");
    LiteRtOp w = FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile);
    AddUninlinableCpuCall(model, model.Subgraph(SubgraphRefs(*w)[1]));
    CompilerPlugin v2 = MakePlugin(&kFakeInterfaceV2);
    LITERT_ASSERT_OK_AND_ASSIGN(auto res, PartitionModel(v2, model));
    EXPECT_THAT(res.first, IsEmpty());
    EXPECT_NE(FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile), nullptr);
  }
}

TEST_F(RegionPartitionTest, InlinedCompositeControlFlowIsWalked) {
  auto& model = Load("tfl_while_region.tflite");
  LiteRtOp w = FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile);
  const int32_t body_idx = SubgraphRefs(*w)[1];
  WrapInComposite(model, *w, "test.wrapped_while");
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});
  CompilerPlugin plugin = MakePlugin();
  LITERT_ASSERT_OK(PartitionModel(plugin, model));
  EXPECT_EQ(NumDispatchOps(model.Subgraph(body_idx)), 1);
}

// --- Helper naming & partition_v2 contract ---

TEST_F(RegionPartitionTest, HelperSubgraphsHaveUniqueNonEmptyNames) {
  auto& model = Load("tfl_while_region.tflite");
  const auto refs =
      SubgraphRefs(*FindOp(model.Subgraph(0), kLiteRtOpCodeTflWhile));
  model.Subgraph(refs[0]).SetName("");
  model.Subgraph(refs[1]).SetName("dup");
  model.EmplaceSubgraph().SetName("dup");
  SetReduceWindowOptions(model.Subgraph(refs[1]).EmplaceOp(),
                         static_cast<int32_t>(model.NumSubgraphs() - 1));

  state_.accepts_helper_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeTflWhile});
  CompilerPlugin plugin = MakePlugin();
  LITERT_ASSERT_OK_AND_ASSIGN(auto res, PartitionModel(plugin, model));
  EXPECT_EQ(model.NumSubgraphs(), 1);  // The helpers were moved.
  ASSERT_EQ(res.second.NumSubgraphs(), 4);
  absl::flat_hash_set<std::string> names;
  for (size_t i = 1; i < res.second.NumSubgraphs(); ++i) {
    const absl::string_view name = res.second.Subgraph(i).Name();
    EXPECT_FALSE(name.empty());
    EXPECT_TRUE(names.insert(std::string(name)).second);
  }
}

TEST_F(RegionPartitionTest, RejectsSelectionOutsideSubgraphBeforeInlining) {
  // The unselected composite in main would be inlined: the selection of an op
  // in its decomposition must be rejected first, without mutating the model.
  state_.accepts_helper_subgraphs = true;
  state_.custom_partition_v2 = [](LiteRtModel m, LiteRtParamIndex,
                                  LiteRtOpList out) {
    out->Push(m->Subgraph(1).Ops().front(), 0);
    return kLiteRtStatusOk;
  };
  auto& model = Load("rms_norm_composite.tflite");
  CompilerPlugin plugin = MakePlugin();
  bool modified = false;
  EXPECT_THAT(PartitionModel(plugin, model, "", {}, &modified),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_FALSE(modified);
  EXPECT_NE(FindOp(model.Subgraph(0), kLiteRtOpCodeShloComposite), nullptr);
  std::vector<CompilerPlugin> plugins;
  plugins.push_back(MakePlugin());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result, ApplyPlugins(&model, kLiteRtHwAcceleratorNpu, plugins));
  EXPECT_FALSE(result.model_invalidated);
}

// --- v1 vs v2 composite packaging, compile_v2 & v1 adapter ---

// A v1 plugin gets a nested npu_call as an opaque op, not as a helper.
TEST_F(RegionPartitionTest, V1NestedNpuCallIsClearedAndShipsNoHelper) {
  auto& model = Load("nested_composite.tflite");
  LiteRtSubgraphT& npu_body = model.Subgraph(*CompositeDecompositionIndex(
      *FindOp(model.Subgraph(0), kLiteRtOpCodeShloComposite)));
  LiteRtOp inner = FindOp(npu_body, kLiteRtOpCodeShloComposite);
  SetCompositeOptions(*inner, CompositeOptions::kNpuCall,
                      *CompositeDecompositionIndex(*inner));
  CompilerPlugin v1 = MakePlugin(&kFakeInterfaceV1);
  LITERT_ASSERT_OK_AND_ASSIGN(auto res, PartitionModel(v1, model));
  ASSERT_EQ(res.first.size(), 1);
  EXPECT_EQ(res.second.NumSubgraphs(), res.first.size());
  EXPECT_EQ(CompositeDecompositionIndex(*inner), -1);
}

TEST_F(RegionPartitionTest, V2ShipsCompositeHelperAndClonesWhenShared) {
  auto& model = Load("rms_norm_composite.tflite");
  LiteRtOp comp = FindOp(model.Subgraph(0), kLiteRtOpCodeShloComposite);
  AddSignature(model, model.Subgraph(*CompositeDecompositionIndex(*comp)),
               "decomp_sig");

  state_.accepts_helper_subgraphs = true;
  state_.select = SelectCodes({kLiteRtOpCodeShloComposite});
  CompilerPlugin v2 = MakePlugin(&kFakeInterfaceV2);
  ApplyPluginsResult result;
  LITERT_ASSERT_OK(ApplyPlugin(v2, model, "", {}, result));
  EXPECT_EQ(state_.compiled_num_entry_subgraphs, 1);
  EXPECT_EQ(state_.compiled_num_subgraphs, 2);
  EXPECT_THAT(state_.compiled_entry0_op_refs, Contains(ElementsAre(1)));
  LITERT_ASSERT_OK(BuildSubgraphCallGraph(model));
}

TEST_F(RegionPartitionTest, ApplyPluginEnforcesOneCallPerEntry) {
  state_.accepts_helper_subgraphs = true;
  state_.emit_call_per_subgraph = true;
  auto& model = Load("npu_call_region.tflite");
  CompilerPlugin v2 = MakePlugin(&kFakeInterfaceV2);
  ApplyPluginsResult result;
  EXPECT_THAT(ApplyPlugin(v2, model, "", {}, result),
              IsError(kLiteRtStatusErrorRuntimeFailure));
}

TEST_F(RegionPartitionTest, V1AdapterClearsDiscardsAndValidatesBeforeMutating) {
  // Clears composites and discards helpers when valid.
  {
    auto& partitions = Load("rms_norm_composite.tflite");
    CompilerPlugin v1 = MakePlugin(&kFakeInterfaceV1);
    LITERT_ASSERT_OK(v1.Compile(&partitions, /*num_entry_subgraphs=*/1));
    EXPECT_TRUE(state_.called_compile_v1);
    EXPECT_EQ(state_.compiled_num_subgraphs, 1);
    EXPECT_THAT(state_.compiled_entry0_op_refs, Each(IsEmpty()));
  }
  // Fails without mutating composite if entry also references a helper via a
  // non-composite op.
  {
    state_ = {};
    auto& partitions = Load("shlo_reduce_window_region.tflite");
    partitions.EmplaceSubgraph();
    const int32_t comp_idx =
        static_cast<int32_t>(partitions.NumSubgraphs() - 1);
    LiteRtOpT& comp_op = partitions.Subgraph(0).EmplaceOp();
    SetCompositeOptions(comp_op, "test.comp", comp_idx);

    CompilerPlugin v1 = MakePlugin(&kFakeInterfaceV1);
    EXPECT_THAT(v1.Compile(&partitions, /*num_entry_subgraphs=*/1),
                IsError(kLiteRtStatusErrorInvalidGraphInvariant));
    EXPECT_FALSE(state_.called_compile_v1);
    EXPECT_EQ(CompositeDecompositionIndex(comp_op), comp_idx);
    EXPECT_EQ(partitions.NumSubgraphs(), 3);
  }
  // Rejects out-of-range num_entry_subgraphs.
  {
    auto& partitions = Load("one_mul.tflite");
    CompilerPlugin v1 = MakePlugin(&kFakeInterfaceV1);
    EXPECT_THAT(v1.Compile(&partitions, /*num_entry_subgraphs=*/2),
                IsError(kLiteRtStatusErrorInvalidArgument));
  }
}

// --- v1/v2 compatibility matrix ---

struct CompatibilityRow {
  std::string name;
  const LiteRtCompilerPluginInterface_V1* api;
  bool expect_partition_v2;
  bool expect_compile_v2;
};

class CompatibilityMatrixTest
    : public RegionPartitionTest,
      public ::testing::WithParamInterface<CompatibilityRow> {};

TEST_P(CompatibilityMatrixTest, BehavesPerCompatibilityMatrix) {
  const CompatibilityRow& row = GetParam();
  auto& model = Load("tfl_while_region.tflite");
  state_.select = SelectCodes({kLiteRtOpCodeTflWhile, kLiteRtOpCodeTflMul});
  CompilerPlugin plugin = MakePlugin(row.api);
  EXPECT_EQ(plugin.AcceptsHelperSubgraphs(), row.expect_compile_v2);

  ApplyPluginsResult result;
  LITERT_ASSERT_OK(ApplyPlugin(plugin, model, "", {}, result));
  EXPECT_EQ(state_.called_partition_v2, row.expect_partition_v2);
  EXPECT_EQ(state_.called_partition_v1, !row.expect_partition_v2);
  EXPECT_EQ(state_.called_compile_v2, row.expect_compile_v2);
  EXPECT_EQ(state_.called_compile_v1, !row.expect_compile_v2);
  EXPECT_EQ(state_.compiled_num_entry_subgraphs, row.expect_compile_v2 ? 1 : 2);
  EXPECT_EQ(state_.compiled_num_subgraphs, row.expect_compile_v2 ? 3 : 2);
  EXPECT_TRUE(state_.compiled_refs_valid);
}

INSTANTIATE_TEST_SUITE_P(
    V1V2Combinations, CompatibilityMatrixTest,
    ::testing::Values(
        CompatibilityRow{"V1Partition_V1Compile", &kFakeInterfaceV1, false,
                         false},
        CompatibilityRow{"V2Partition_V1Compile",
                         &kFakeInterfacePartitionV2Only, true, false},
        CompatibilityRow{"V1Partition_V2Compile", &kFakeInterfaceCompileV2Only,
                         false, true},
        CompatibilityRow{"V2Partition_V2Compile", &kFakeInterfaceV2, true,
                         true},
        // ABI 1.0 struct without the v2 members.
        CompatibilityRow{"Abi1_0", &kFakeInterfaceV1_0, false, false}),
    [](const ::testing::TestParamInfo<CompatibilityRow>& info) {
      return info.param.name;
    });

// --- ApplyPlugins pre-mutation vs post-mutation invalidation ---

TEST_F(RegionPartitionTest, PreMutationFailureDoesNotInvalidateModel) {
  auto& model = Load("npu_call_region.tflite");
  LiteRtCompilerPluginT v1_state, v2_state;
  v2_state.accepts_helper_subgraphs = true;
  std::vector<CompilerPlugin> plugins;
  plugins.push_back(CompilerPluginFriend::Wrap(&kFakeInterfaceV1, &v1_state));
  plugins.push_back(CompilerPluginFriend::Wrap(&kFakeInterfaceV2, &v2_state));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result, ApplyPlugins(&model, kLiteRtHwAcceleratorNpu, plugins));
  EXPECT_FALSE(result.model_invalidated);
  EXPECT_EQ(result.num_applied_plugins, 1);
  EXPECT_TRUE(v2_state.called_compile_v2);
}

TEST_F(RegionPartitionTest, PostMutationFailureInvalidatesModel) {
  auto& model = Load("tfl_while_region.tflite");
  state_.select = SelectCodes({kLiteRtOpCodeTflMul});
  state_.fail_compile = true;
  std::vector<CompilerPlugin> plugins;
  plugins.push_back(MakePlugin(&kFakeInterfaceV1));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto result, ApplyPlugins(&model, kLiteRtHwAcceleratorNpu, plugins));
  EXPECT_TRUE(result.model_invalidated);
  EXPECT_EQ(result.num_applied_plugins, 0);
}

}  // namespace
}  // namespace litert::internal
