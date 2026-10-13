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

#include "litert/compiler/plugin/compiler_plugin.h"

#include <array>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_builder.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/core/build_stamp.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"
#include "litert/tools/dump.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using testing::UniqueTestDirectory;

constexpr absl::string_view kTestPluginSearchPath = "vendors/examples";

constexpr absl::string_view kTestManufacturer = "ExampleSocManufacturer";
constexpr absl::string_view kTestModels = "ExampleSocModel";

using testing::GetLiteRtPath;

TEST(CompilerPluginTest, LoadTestPlugin) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);
  ASSERT_EQ(plugins->front().SocModels().size(), 1);
  EXPECT_EQ(plugins->front().SocModels().front(), kTestModels);
}

TEST(CompilerPluginTest, FindTestPluginOk) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin,
      CompilerPlugin::FindPlugin(kTestManufacturer,
                                 {GetLiteRtPath(kTestPluginSearchPath)}));
  EXPECT_EQ(plugin.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, FindTestPluginWithOptionsOk) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  auto litert_options = Options::Create();
  auto compiler_options = CompilerOptions::Create();
  compiler_options->SetPartitionStrategy(
      kLiteRtCompilerOptionsPartitionStrategyDefault);
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueCompilerOptionsData(
      compiler_options->Get(), &identifier, &payload, &payload_deleter));
  LiteRtOpaqueOptions opaque_opts = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(identifier, payload,
                                             payload_deleter, &opaque_opts));
  auto opaque_compiler_options =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
  litert_options->AddOpaqueOptions(std::move(opaque_compiler_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto litert_built_options,
                              litert::internal::LiteRtOptionsPtrBuilder::Build(
                                  *litert_options, env.GetHolder()));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin,
      CompilerPlugin::FindPlugin(kTestManufacturer,
                                 {GetLiteRtPath(kTestPluginSearchPath)},
                                 /*env=*/nullptr, litert_built_options.get()));
  EXPECT_EQ(plugin.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, GetOptionsFromTestPluginOk) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  auto litert_options = Options::Create();
  auto compiler_options = CompilerOptions::Create();
  compiler_options->SetPartitionStrategy(
      kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueCompilerOptionsData(
      compiler_options->Get(), &identifier, &payload, &payload_deleter));
  LiteRtOpaqueOptions opaque_opts = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(identifier, payload,
                                             payload_deleter, &opaque_opts));
  auto opaque_compiler_options =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
  litert_options->AddOpaqueOptions(std::move(opaque_compiler_options));
  LITERT_ASSERT_OK_AND_ASSIGN(auto litert_built_options,
                              litert::internal::LiteRtOptionsPtrBuilder::Build(
                                  *litert_options, env.GetHolder()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin,
      CompilerPlugin::FindPlugin(kTestManufacturer,
                                 {GetLiteRtPath(kTestPluginSearchPath)},
                                 /*env=*/nullptr, litert_built_options.get()));

  auto compiler_options_from_plugin = plugin.CompilerOptions();
  ASSERT_TRUE(compiler_options_from_plugin);
  EXPECT_EQ(compiler_options_from_plugin->partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
}

TEST(CompilerPluginTest, FindTestPluginNotFound) {
  auto plugin =
      CompilerPlugin::FindPlugin("not_a_soc", {kTestPluginSearchPath});
  EXPECT_FALSE(plugin);
}

TEST(CompilerPluginTest, LoadTestPluginWithMalformed) {
  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);
  Touch(Join({dir->Str(), "notLibLiteRt.so"}));

  auto plugins = CompilerPlugin::LoadPlugins({dir->Str()});

  ASSERT_EQ(plugins->size(), 0);
}

TEST(CompilerPluginTest, MultipleValidPlugins) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath),
                                   GetLiteRtPath(kTestPluginSearchPath)});

  ASSERT_EQ(plugins->size(), 2);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);
  EXPECT_EQ(plugins->back().SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MoveAssign) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  CompilerPlugin other = std::move(plugins->front());

  EXPECT_EQ(other.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MoveConstruct) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  CompilerPlugin other(std::move(plugins->front()));

  EXPECT_EQ(other.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, SocModels) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  EXPECT_THAT(plugins->front().SocModels(),
              ::testing::ElementsAreArray({kTestModels}));
}

TEST(CompilerPluginTest, SdkVersion) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  auto sdk_version = plugins->front().SdkVersion();
  ASSERT_TRUE(sdk_version);
  EXPECT_EQ(*sdk_version, "1.0.0-example");
}

TEST(CompilerPluginTest, Partition) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  auto model = testing::LoadTestFileModel("mul_simple.tflite");
  auto subgraph = model.MainSubgraph();
  auto ops = plugins->front().Partition(subgraph->Get());
  ASSERT_TRUE(ops);

  EXPECT_EQ(ops->size(), 2);
}

TEST(CompilerPluginTest, Compile) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  auto& model = *model_wrap.Get();

  auto result = plugins->front().Compile(&model, "JustInTime");
  ASSERT_TRUE(result);

  auto byte_code = result->ByteCode();
  ASSERT_TRUE(byte_code && byte_code->Size() > 0);

  auto num_calls = result->NumCalls();
  ASSERT_TRUE(num_calls);
  ASSERT_EQ(*num_calls, 1);

  auto call_info = result->CallInfo(0);
  ASSERT_TRUE(call_info);

  auto handle_or = result->GetHandle(0);
  ASSERT_TRUE(handle_or.HasValue());
  EXPECT_NE(handle_or.Value(), nullptr);
}

TEST(CompilerPluginTest, CompileNonPartitionedModel) {
  LiteRtModelT model;
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);
  auto result = plugins->front().Compile(&model, kTestModels);
  ASSERT_TRUE(result);
}

TEST(CompilerPluginTest, Dump) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);

  std::stringstream dump;
  Dump(plugins->front(), dump);

  ASSERT_EQ(dump.str(),
            "SocManufacturer: ExampleSocManufacturer\nSocModels: { "
            "ExampleSocModel }\n");
}

TEST(PartitionModelTest, Simple) {
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  auto& model = *model_wrap.Get();

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 1);

  const auto& [ops, new_model] = *partition_result;

  EXPECT_EQ(ops.size(), 1);
  EXPECT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);

  EXPECT_EQ(new_model.NumSubgraphs(), 1);
  EXPECT_EQ(new_model.Subgraphs().front()->Ops().size(), 2);
}

TEST(PartitionModelTest, PartitionDirect) {
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  auto& model = *model_wrap.Get();

  std::vector<LiteRtOpWithPartitionIndex> selected_ops = {
      {model.MainSubgraph()->Ops().front(), 0},
      {model.MainSubgraph()->Ops().back(), 0}};

  auto partition_result = PartitionModelDirect(std::move(selected_ops), model);
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 1);

  const auto& [ops, new_model] = *partition_result;

  EXPECT_EQ(ops.size(), 1);
  EXPECT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);

  EXPECT_EQ(new_model.NumSubgraphs(), 1);
  EXPECT_EQ(new_model.Subgraphs().front()->Ops().size(), 2);
}

TEST(PartitionModelTest, MultiSubgraph) {
  auto model_wrap = testing::LoadTestFileModel("multi_subgraph_mul.tflite");
  auto& model = *model_wrap.Get();

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 2);

  const auto& [ops, new_model] = *partition_result;

  EXPECT_EQ(ops.size(), 2);
  EXPECT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_EQ(ops.back()->OpCode(), kLiteRtOpCodeTflCustom);

  EXPECT_EQ(model.NumSubgraphs(), 2);
  EXPECT_EQ(new_model.Subgraphs().front()->Ops().size(), 1);
  EXPECT_EQ(new_model.Subgraphs().back()->Ops().size(), 1);
}

TEST(PartitionModelTest, MultiSubgraphWithSelectedSubgraphs) {
  auto model_wrap = testing::LoadTestFileModel("multi_subgraph_mul.tflite");
  auto& model = *model_wrap.Get();

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model, kTestModels, {1});
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 2);

  const auto& [ops, new_model] = *partition_result;

  EXPECT_EQ(ops.size(), 1);
  EXPECT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);

  EXPECT_EQ(new_model.NumSubgraphs(), 1);
  EXPECT_EQ(new_model.Subgraphs().front()->Ops().size(), 1);
}

TEST(PartitionModelTest, ProtectedCpuCall) {
  // Construct a model with a mul op inside a cpu_call.
  // The plugin should NOT take the cpu_call, and it should be inlined.
  LiteRtModelT model;
  auto& main_subgraph = model.EmplaceSubgraph();
  auto& decomp_subgraph = model.EmplaceSubgraph();

  // Create tensors for mul in decomp.
  auto& t1 = decomp_subgraph.EmplaceTensor();
  t1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));
  auto& t2 = decomp_subgraph.EmplaceTensor();
  t2.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));
  auto& t3 = decomp_subgraph.EmplaceTensor();
  t3.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));

  decomp_subgraph.Inputs().push_back(&t1);
  decomp_subgraph.Inputs().push_back(&t2);
  decomp_subgraph.Outputs().push_back(&t3);

  auto& mul_op = decomp_subgraph.EmplaceOp();
  mul_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&t1, mul_op);
  AttachInput(&t2, mul_op);
  AttachOutput(&t3, mul_op);

  // Create tensors for composite in main.
  auto& main_t1 = main_subgraph.EmplaceTensor();
  main_t1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));
  auto& main_t2 = main_subgraph.EmplaceTensor();
  main_t2.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));
  auto& main_t3 = main_subgraph.EmplaceTensor();
  main_t3.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));

  main_subgraph.Inputs().push_back(&main_t1);
  main_subgraph.Inputs().push_back(&main_t2);
  main_subgraph.Outputs().push_back(&main_t3);

  auto& cpu_call_op = main_subgraph.EmplaceOp();
  cpu_call_op.SetOpCode(kLiteRtOpCodeShloComposite);
  AttachInput(&main_t1, cpu_call_op);
  AttachInput(&main_t2, cpu_call_op);
  AttachOutput(&main_t3, cpu_call_op);

  // Set composite options.
  // Use index 1 as decomp_subgraph is the second subgraph created.
  tflite::StableHLOCompositeOptionsT options;
  options.name = std::string(CompositeOptions::kCpuCall);
  options.decomposition_subgraph_index = 1;

  internal::TflOptions2 tfl_options;
  tfl_options.type = ::tflite::BuiltinOptions2_StableHLOCompositeOptions;
  tfl_options.Set(std::move(options));
  litert::internal::SetTflOptions2(cpu_call_op, std::move(tfl_options));

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);

  // The cpu_call should be inlined, so the main subgraph should have the mul op
  // instead of a custom op.
  ASSERT_EQ(model.NumSubgraphs(), 1);
  auto& final_subgraph = *model.MainSubgraph();

  // The mul op should be there.
  ASSERT_EQ(final_subgraph.Ops().size(), 1);
  EXPECT_EQ(final_subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflMul);

  // No custom ops should be in the partition result.
  const auto& [ops, new_model] = *partition_result;
  EXPECT_TRUE(ops.empty());
  EXPECT_EQ(new_model.NumSubgraphs(), 0);
}

void BuildMulAddMulModel(LiteRtModelT& model) {
  auto& subgraph = model.EmplaceSubgraph();

  auto& t_in = subgraph.EmplaceTensor();
  t_in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));
  auto& t_tmp1 = subgraph.EmplaceTensor();
  t_tmp1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));
  auto& t_tmp2 = subgraph.EmplaceTensor();
  t_tmp2.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));
  auto& t_out = subgraph.EmplaceTensor();
  t_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1}));

  subgraph.Inputs().push_back(&t_in);
  subgraph.Outputs().push_back(&t_out);

  auto& mul1 = subgraph.EmplaceOp();
  mul1.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&t_in, mul1);
  AttachInput(&t_in, mul1);
  AttachOutput(&t_tmp1, mul1);

  auto& add = subgraph.EmplaceOp();
  add.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&t_tmp1, add);
  AttachInput(&t_tmp1, add);
  AttachOutput(&t_tmp2, add);

  auto& mul2 = subgraph.EmplaceOp();
  mul2.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&t_tmp2, mul2);
  AttachInput(&t_tmp2, mul2);
  AttachOutput(&t_out, mul2);
}

TEST(PartitionModelTest, MaxPartitionsSuccess) {
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  auto& model = *model_wrap.Get();

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  auto litert_options = Options::Create();
  ASSERT_TRUE(litert_options);
  auto compiler_options = CompilerOptions::Create();
  ASSERT_TRUE(compiler_options);
  LITERT_ASSERT_OK(compiler_options->SetMaxPartitions(1));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueCompilerOptionsData(
      compiler_options->Get(), &identifier, &payload, &payload_deleter));
  LiteRtOpaqueOptions opaque_opts = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(
      identifier, payload, payload_deleter, &opaque_opts));
  auto opaque_compiler_options =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
  litert_options->AddOpaqueOptions(std::move(opaque_compiler_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto litert_built_options,
                              litert::internal::LiteRtOptionsPtrBuilder::Build(
                                  *litert_options, env.GetHolder()));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin,
      CompilerPlugin::FindPlugin(kTestManufacturer,
                                 {GetLiteRtPath(kTestPluginSearchPath)},
                                 /*env=*/nullptr,
                                 litert_built_options.get()));

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 1);

  const auto& [ops, new_model] = *partition_result;
  EXPECT_EQ(ops.size(), 1);
  EXPECT_EQ(new_model.NumSubgraphs(), 1);
}

TEST(PartitionModelTest, MaxPartitionsFail) {
  LiteRtModelT model;
  BuildMulAddMulModel(model);

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  auto litert_options = Options::Create();
  ASSERT_TRUE(litert_options);
  auto compiler_options = CompilerOptions::Create();
  ASSERT_TRUE(compiler_options);
  LITERT_ASSERT_OK(compiler_options->SetMaxPartitions(1));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueCompilerOptionsData(
      compiler_options->Get(), &identifier, &payload, &payload_deleter));
  LiteRtOpaqueOptions opaque_opts = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(
      identifier, payload, payload_deleter, &opaque_opts));
  auto opaque_compiler_options =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
  litert_options->AddOpaqueOptions(std::move(opaque_compiler_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto litert_built_options,
                              litert::internal::LiteRtOptionsPtrBuilder::Build(
                                  *litert_options, env.GetHolder()));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin,
      CompilerPlugin::FindPlugin(kTestManufacturer,
                                 {GetLiteRtPath(kTestPluginSearchPath)},
                                 /*env=*/nullptr,
                                 litert_built_options.get()));

  auto partition_result = PartitionModel(plugin, model);
  EXPECT_FALSE(partition_result);
  EXPECT_EQ(partition_result.Error().Status(),
            kLiteRtStatusErrorRuntimeFailure);
}

TEST(PartitionModelTest, MaxPartitionsSuccessLimit2) {
  LiteRtModelT model;
  BuildMulAddMulModel(model);

  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  auto litert_options = Options::Create();
  ASSERT_TRUE(litert_options);
  auto compiler_options = CompilerOptions::Create();
  ASSERT_TRUE(compiler_options);
  LITERT_ASSERT_OK(compiler_options->SetMaxPartitions(2));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  LITERT_ASSERT_OK(LrtGetOpaqueCompilerOptionsData(
      compiler_options->Get(), &identifier, &payload, &payload_deleter));
  LiteRtOpaqueOptions opaque_opts = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(
      identifier, payload, payload_deleter, &opaque_opts));
  auto opaque_compiler_options =
      litert::OpaqueOptions::WrapCObject(opaque_opts, litert::OwnHandle::kYes);
  litert_options->AddOpaqueOptions(std::move(opaque_compiler_options));

  LITERT_ASSERT_OK_AND_ASSIGN(auto litert_built_options,
                              litert::internal::LiteRtOptionsPtrBuilder::Build(
                                  *litert_options, env.GetHolder()));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin,
      CompilerPlugin::FindPlugin(kTestManufacturer,
                                 {GetLiteRtPath(kTestPluginSearchPath)},
                                 /*env=*/nullptr,
                                 litert_built_options.get()));

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 1);

  const auto& [ops, new_model] = *partition_result;
  EXPECT_EQ(ops.size(), 2);
  EXPECT_EQ(new_model.NumSubgraphs(), 2);
}

TEST(PartitionModelTest, CstMultiSubgraph) {
  auto model_wrap = testing::LoadTestFileModel("multi_use_cst.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.MainSubgraph()->Ops().size(), 3);

  std::vector<LiteRtOpWithPartitionIndex> selected_ops = {
      {model.MainSubgraph()->Ops().front(), 0},
      {model.MainSubgraph()->Ops().back(), 0},
  };
  auto partition_result = PartitionModelDirect(std::move(selected_ops), model);
  ASSERT_TRUE(partition_result);

  const auto& [ops, new_model] = *partition_result;

  EXPECT_EQ(ops.size(), 2);
  EXPECT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_EQ(ops.back()->OpCode(), kLiteRtOpCodeTflCustom);

  EXPECT_EQ(new_model.NumSubgraphs(), 2);
  EXPECT_EQ(new_model.Subgraphs().front()->Ops().size(), 1);
  EXPECT_EQ(new_model.Subgraphs().back()->Ops().size(), 1);

  const auto& cst_1 =
      new_model.Subgraphs().front()->Ops().front()->Input(1).Weights();
  const auto& cst_2 =
      new_model.Subgraphs().back()->Ops().front()->Input(1).Weights();

  // Both weights should have the same object managed by the same buffer
  // manager.
  ASSERT_EQ(cst_1.GetBufferManager(), model.Buffers());
  ASSERT_EQ(cst_2.GetBufferManager(), model.Buffers());
  ASSERT_GT(cst_1.Buffer().Size(), 0);
  ASSERT_GT(cst_2.Buffer().Size(), 0);
  EXPECT_EQ(cst_1.GetBufferId(), cst_2.GetBufferId());
  ASSERT_EQ(cst_1.Buffer().Data(), cst_2.Buffer().Data());
}

TEST(ApplyTest, Simple) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  ApplyPluginsResult apply_result;
  ASSERT_TRUE(ApplyPlugin(plugins->front(), model, "", {}, apply_result));
  ASSERT_EQ(model.NumSubgraphs(), 1);

  auto& subgraph = *model.MainSubgraph();
  ASSERT_EQ(subgraph.Ops().size(), 1);

  auto* op = subgraph.Ops().front();

  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_TRUE(model.FindOpAsset(op));

  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));
  EXPECT_TRUE(apply_result.compiled_results.empty());
}

TEST(ApplyTest, RetainsCompiledResultForJitHandle) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  ApplyPluginsResult apply_result;
  ASSERT_TRUE(
      ApplyPlugin(plugins->front(), model, "JustInTime", {}, apply_result));

  EXPECT_EQ(apply_result.compiled_results.size(), 1);
  EXPECT_FALSE(apply_result.jit_executable_handles.empty());
}

TEST(ApplyTest, WithPartition) {
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  auto& model = *model_wrap.Get();

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 1);

  ApplyPluginsResult apply_result;
  ASSERT_TRUE(ApplyPluginWithPartition(
      plugins->front(), model, std::move(*partition_result), "", apply_result));

  auto& subgraph = model.Subgraph(0);
  ASSERT_EQ(subgraph.Ops().size(), 1);

  auto* op = subgraph.Ops().front();

  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_TRUE(model.FindOpAsset(op));
}

TEST(ApplyTest, MultiSubgraph) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto model_wrap = testing::LoadTestFileModel("multi_subgraph_mul.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  ApplyPluginsResult apply_result;
  ASSERT_TRUE(ApplyPlugin(plugins->front(), model, "", {}, apply_result));
  ASSERT_EQ(model.NumSubgraphs(), 2);

  {
    auto& subgraph = model.Subgraph(0);
    ASSERT_EQ(subgraph.Ops().size(), 1);

    auto* op = subgraph.Ops().front();

    EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
    EXPECT_TRUE(model.FindOpAsset(op));
  }

  {
    auto& subgraph = model.Subgraph(1);
    ASSERT_EQ(subgraph.Ops().size(), 1);

    auto* op = subgraph.Ops().front();

    EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
    EXPECT_TRUE(model.FindOpAsset(op));
  }

  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));
}

TEST(ApplyTest, ApplyPlugins) {
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  const std::string plugin_search_path = GetLiteRtPath(kTestPluginSearchPath);
  const std::array environment_options = {
      litert::EnvironmentOptions::Option{
          /*.tag=*/litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir,
          /*.value=*/plugin_search_path.c_str(),
      },
  };
  auto env = litert::Environment::Create(
      litert::EnvironmentOptions(environment_options));
  ASSERT_TRUE(env);

  LiteRtHwAccelerators compilation_options = static_cast<LiteRtHwAccelerators>(
      kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu |
      kLiteRtHwAcceleratorNpu);
  auto result = litert::internal::ApplyPlugins(env->GetHolder().handle,
                                               /*options=*/nullptr, &model,
                                               compilation_options);
  ASSERT_TRUE(result);

  ASSERT_EQ(model.NumSubgraphs(), 1);

  auto& subgraph = *model.MainSubgraph();
  ASSERT_EQ(subgraph.Ops().size(), 1);

  auto* op = subgraph.Ops().front();

  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_TRUE(model.FindOpAsset(op));

  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));
}

TEST(ApplyTest, ApplyLoadedPlugins) {
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  LiteRtHwAccelerators compilation_options = static_cast<LiteRtHwAccelerators>(
      kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu |
      kLiteRtHwAcceleratorNpu);
  auto result = litert::internal::ApplyPlugins(&model, compilation_options,
                                               *plugins, /*mutated=*/nullptr);
  ASSERT_TRUE(result);

  ASSERT_EQ(model.NumSubgraphs(), 1);

  auto& subgraph = *model.MainSubgraph();
  ASSERT_EQ(subgraph.Ops().size(), 1);

  auto* op = subgraph.Ops().front();

  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_TRUE(model.FindOpAsset(op));

  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));
}

TEST(ApplyTest, ExternalWeightsFullyCompiledAndStrippedOnSerialize) {
  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(
      testing::GetTestFilePath("add_cst.tflite"));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  // Change add -> mul so example_plugin compiles it, and move its constant
  // weight to an external buffer.
  tfl_model->operator_codes[0]->builtin_code = ::tflite::BuiltinOperator_MUL;
  const int32_t cst_tensor_idx =
      tfl_model->subgraphs[0]->operators[0]->inputs[1];
  auto& tfl_cst_tensor = *tfl_model->subgraphs[0]->tensors[cst_tensor_idx];
  tfl_model->buffers[tfl_cst_tensor.buffer]->data.clear();
  tfl_cst_tensor.buffer = 0;
  tfl_cst_tensor.external_buffer = 1;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kExternalWeights = {3.0f, 4.0f, 5.0f, 6.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kExternalWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto initial_serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);

  std::filesystem::path temp_path(std::string(temp_dir->Str()));
  std::filesystem::path model_path = temp_path / "ext_mul_cst.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(initial_serialized.StrData(), initial_serialized.Size());
  }
  std::filesystem::path weights_path = temp_path / "ext_weights.bin";
  {
    std::ofstream weights_file(weights_path,
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);
  auto& model = **loaded_model;
  EXPECT_TRUE(IsConstant(model.MainSubgraph()->Op(0).Input(1)));

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_TRUE(plugins);
  ASSERT_EQ(plugins->size(), 1);

  ApplyPluginsResult apply_result;
  ASSERT_TRUE(ApplyPlugin(plugins->front(), model, "", {}, apply_result));

  auto& subgraph = *model.MainSubgraph();
  ASSERT_EQ(subgraph.Ops().size(), 1);
  auto* dispatch_op = subgraph.Ops().front();
  EXPECT_EQ(dispatch_op->OpCode(), kLiteRtOpCodeTflCustom);
  // The external constant weight was recognized as IsConstant and absorbed into
  // the NPU partition subgraph, so the dispatch op only takes 1 runtime input.
  EXPECT_EQ(dispatch_op->Inputs().size(), 1u);

  auto op_asset = model.FindOpAsset(dispatch_op);
  ASSERT_TRUE(op_asset);
  auto asset_buf = model.Buffers()->GetBuffer(op_asset->first);
  ASSERT_TRUE(asset_buf);
  EXPECT_THAT(asset_buf->StrView(), ::testing::HasSubstr("DATA:3,4,5,6"));

  // Serializing the compiled model should omit external_buffers since no
  // uncompiled tensor references them anymore.
  auto compiled_serialized = SerializeModel(std::move(model));
  ASSERT_TRUE(compiled_serialized);
  auto compiled_fb = FlatbufferWrapper::CreateFromBuffer(*compiled_serialized);
  ASSERT_TRUE(compiled_fb);
  auto unpacked = compiled_fb->get()->Unpack();
  EXPECT_TRUE(unpacked->external_buffers.empty());
  EXPECT_TRUE(unpacked->external_buffer_groups.empty());
}

TEST(ApplyTest,
     ExternalWeightsPartiallyCompiledRetainsUncompiledExternalBuffers) {
  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(
      testing::GetTestFilePath("add_cst.tflite"));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  // Build a 2-op graph:
  //   op 0: tmp = mul(input0, ext_cst_1) [compiled by example_plugin]
  //   op 1: out = add(tmp, ext_cst_2)    [remains on CPU]
  auto mul_opcode = std::make_unique<TflOpCode>();
  mul_opcode->builtin_code = ::tflite::BuiltinOperator_MUL;
  tfl_model->operator_codes.push_back(std::move(mul_opcode));
  const uint32_t mul_opcode_idx = tfl_model->operator_codes.size() - 1;

  auto& sg = *tfl_model->subgraphs[0];
  const int32_t orig_input_idx = sg.operators[0]->inputs[0];
  const int32_t cst1_idx = sg.operators[0]->inputs[1];
  const int32_t orig_output_idx = sg.operators[0]->outputs[0];

  tfl_model->buffers[sg.tensors[cst1_idx]->buffer]->data.clear();
  sg.tensors[cst1_idx]->buffer = 0;
  sg.tensors[cst1_idx]->external_buffer = 1;

  auto tmp_tensor = std::make_unique<TflTensor>(*sg.tensors[orig_output_idx]);
  tmp_tensor->name = "mul_out";
  tmp_tensor->buffer = 0;
  tmp_tensor->external_buffer = 0;
  sg.tensors.push_back(std::move(tmp_tensor));
  const int32_t tmp_tensor_idx = sg.tensors.size() - 1;

  auto cst2_tensor = std::make_unique<TflTensor>(*sg.tensors[cst1_idx]);
  cst2_tensor->name = "ext_cst_2";
  cst2_tensor->buffer = 0;
  cst2_tensor->external_buffer = 2;
  sg.tensors.push_back(std::move(cst2_tensor));
  const int32_t cst2_idx = sg.tensors.size() - 1;

  // Op 0 becomes mul(input0, cst1) -> tmp
  sg.operators[0]->opcode_index = mul_opcode_idx;
  sg.operators[0]->inputs = {orig_input_idx, cst1_idx};
  sg.operators[0]->outputs = {tmp_tensor_idx};

  // Op 1 is add(tmp, cst2) -> orig_output
  auto add_op = std::make_unique<TflOp>();
  add_op->opcode_index = 0;  // ADD
  add_op->inputs = {tmp_tensor_idx, cst2_idx};
  add_op->outputs = {orig_output_idx};
  sg.operators.push_back(std::move(add_op));

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 8> kExternalWeights = {
      2.0f,  3.0f,  4.0f,  5.0f,  // ext_buffer 1 (for mul)
      10.0f, 20.0f, 30.0f, 40.0f  // ext_buffer 2 (for add)
  };
  auto ext_buf1 = std::make_unique<tflite::ExternalBufferT>();
  ext_buf1->id = 1;
  ext_buf1->group = 1;
  ext_buf1->offset = 0;
  ext_buf1->length = 4 * sizeof(float);
  auto ext_buf2 = std::make_unique<tflite::ExternalBufferT>();
  ext_buf2->id = 2;
  ext_buf2->group = 1;
  ext_buf2->offset = 4 * sizeof(float);
  ext_buf2->length = 4 * sizeof(float);
  tfl_model->external_buffers.push_back(std::move(ext_buf1));
  tfl_model->external_buffers.push_back(std::move(ext_buf2));

  auto initial_serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);

  std::filesystem::path temp_path(std::string(temp_dir->Str()));
  std::filesystem::path model_path = temp_path / "ext_mul_add.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(initial_serialized.StrData(), initial_serialized.Size());
  }
  std::filesystem::path weights_path = temp_path / "ext_weights.bin";
  {
    std::ofstream weights_file(weights_path,
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);
  auto& model = **loaded_model;

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_TRUE(plugins);
  ASSERT_EQ(plugins->size(), 1);

  ApplyPluginsResult apply_result;
  ASSERT_TRUE(ApplyPlugin(plugins->front(), model, "", {}, apply_result));

  auto& main_sg = *model.MainSubgraph();
  ASSERT_EQ(main_sg.Ops().size(), 2u);
  EXPECT_EQ(main_sg.Op(0).OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_EQ(main_sg.Op(1).OpCode(), kLiteRtOpCodeTflAdd);

  auto compiled_serialized = SerializeModel(std::move(model));
  ASSERT_TRUE(compiled_serialized);
  auto compiled_fb = FlatbufferWrapper::CreateFromBuffer(*compiled_serialized);
  ASSERT_TRUE(compiled_fb);
  auto unpacked = compiled_fb->get()->Unpack();

  // External buffer 1 (used by mul) was consumed by the NPU partition, while
  // external buffer 2 (used by CPU add) remains in external_buffers without
  // being inlined into buffers.
  ASSERT_EQ(unpacked->external_buffers.size(), 1u);
  EXPECT_EQ(unpacked->external_buffers[0]->id, 2u);
  EXPECT_EQ(unpacked->external_buffers[0]->offset, 4 * sizeof(float));
  EXPECT_EQ(unpacked->external_buffers[0]->length, 4 * sizeof(float));
  ASSERT_EQ(unpacked->external_buffer_groups.size(), 2u);
  EXPECT_EQ(unpacked->external_buffer_groups[1]->name, "ext_weights.bin");

  const int32_t add_cst_idx = unpacked->subgraphs[0]->operators[1]->inputs[1];
  EXPECT_EQ(unpacked->subgraphs[0]->tensors[add_cst_idx]->external_buffer, 2u);
  EXPECT_EQ(unpacked->subgraphs[0]->tensors[add_cst_idx]->buffer, 0u);
}

TEST(ApplyTest, SharedExternalWeightBetweenNpuAndCpuOpsRetainsExternalBuffer) {
  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(
      testing::GetTestFilePath("add_cst.tflite"));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  // Build a 2-op graph where both NPU mul and CPU add share the SAME external
  // constant tensor (external_buffer = 1):
  //   op 0: tmp = mul(input0, shared_ext_cst) [compiled onto NPU]
  //   op 1: out = add(tmp, shared_ext_cst)    [remains on CPU]
  auto mul_opcode = std::make_unique<TflOpCode>();
  mul_opcode->builtin_code = ::tflite::BuiltinOperator_MUL;
  tfl_model->operator_codes.push_back(std::move(mul_opcode));
  const uint32_t mul_opcode_idx = tfl_model->operator_codes.size() - 1;

  auto& sg = *tfl_model->subgraphs[0];
  const int32_t orig_input_idx = sg.operators[0]->inputs[0];
  const int32_t shared_cst_idx = sg.operators[0]->inputs[1];
  const int32_t orig_output_idx = sg.operators[0]->outputs[0];

  tfl_model->buffers[sg.tensors[shared_cst_idx]->buffer]->data.clear();
  sg.tensors[shared_cst_idx]->buffer = 0;
  sg.tensors[shared_cst_idx]->external_buffer = 1;

  auto tmp_tensor = std::make_unique<TflTensor>(*sg.tensors[orig_output_idx]);
  tmp_tensor->name = "mul_out";
  tmp_tensor->buffer = 0;
  tmp_tensor->external_buffer = 0;
  sg.tensors.push_back(std::move(tmp_tensor));
  const int32_t tmp_tensor_idx = sg.tensors.size() - 1;

  sg.operators[0]->opcode_index = mul_opcode_idx;
  sg.operators[0]->inputs = {orig_input_idx, shared_cst_idx};
  sg.operators[0]->outputs = {tmp_tensor_idx};

  auto add_op = std::make_unique<TflOp>();
  add_op->opcode_index = 0;  // ADD
  add_op->inputs = {tmp_tensor_idx, shared_cst_idx};
  add_op->outputs = {orig_output_idx};
  sg.operators.push_back(std::move(add_op));

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kSharedWeights = {2.0f, 3.0f, 4.0f, 5.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kSharedWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto initial_serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "shared_npu_cpu.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(initial_serialized.StrData(), initial_serialized.Size());
  }
  {
    std::ofstream weights_file(temp_path / "ext_weights.bin",
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kSharedWeights.data()),
                       sizeof(kSharedWeights));
  }

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);
  auto& model = **loaded_model;

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_TRUE(plugins);
  ASSERT_EQ(plugins->size(), 1);

  ApplyPluginsResult apply_result;
  ASSERT_TRUE(ApplyPlugin(plugins->front(), model, "", {}, apply_result));

  auto& main_sg = *model.MainSubgraph();
  ASSERT_EQ(main_sg.Ops().size(), 2u);
  auto* dispatch_op = &main_sg.Op(0);
  EXPECT_EQ(dispatch_op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_EQ(dispatch_op->Inputs().size(), 1u);

  auto op_asset = model.FindOpAsset(dispatch_op);
  ASSERT_TRUE(op_asset);
  auto asset_buf = model.Buffers()->GetBuffer(op_asset->first);
  ASSERT_TRUE(asset_buf);
  EXPECT_THAT(asset_buf->StrView(), ::testing::HasSubstr("DATA:2,3,4,5"));

  // Because CPU add still uses the shared external constant tensor,
  // SerializeModel must retain external_buffer = 1 in external_buffers.
  auto compiled_serialized = SerializeModel(std::move(model));
  ASSERT_TRUE(compiled_serialized);
  auto compiled_fb = FlatbufferWrapper::CreateFromBuffer(*compiled_serialized);
  ASSERT_TRUE(compiled_fb);
  auto unpacked = compiled_fb->get()->Unpack();

  ASSERT_EQ(unpacked->external_buffers.size(), 1u);
  EXPECT_EQ(unpacked->external_buffers[0]->id, 1u);
  ASSERT_EQ(unpacked->external_buffer_groups.size(), 2u);
  const int32_t add_cst_idx = unpacked->subgraphs[0]->operators[1]->inputs[1];
  EXPECT_EQ(unpacked->subgraphs[0]->tensors[add_cst_idx]->external_buffer, 1u);
  EXPECT_EQ(unpacked->subgraphs[0]->tensors[add_cst_idx]->buffer, 0u);
}

TEST(PartitionModelTest, ExternalWeightSharedAcrossMultipleNpuPartitions) {
  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(
      testing::GetTestFilePath("multi_use_cst.tflite"));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();

  // multi_use_cst.tflite has 3 ops sharing a constant tensor. Convert that
  // constant tensor to an external buffer and make ops 0 and 2 MUL (compiled by
  // example_plugin) and op 1 ADD (stays on CPU, splitting into 2 partitions).
  auto mul_opcode = std::make_unique<TflOpCode>();
  mul_opcode->builtin_code = ::tflite::BuiltinOperator_MUL;
  tfl_model->operator_codes.push_back(std::move(mul_opcode));
  const uint32_t mul_opcode_idx = tfl_model->operator_codes.size() - 1;

  auto& sg = *tfl_model->subgraphs[0];
  ASSERT_EQ(sg.operators.size(), 3u);
  sg.operators[0]->opcode_index = mul_opcode_idx;
  sg.operators[2]->opcode_index = mul_opcode_idx;

  const int32_t cst_idx = sg.operators[0]->inputs[1];
  tfl_model->buffers[sg.tensors[cst_idx]->buffer]->data.clear();
  sg.tensors[cst_idx]->buffer = 0;
  sg.tensors[cst_idx]->external_buffer = 1;

  // Give op 1 (ADD) a non-constant second input (its own first input) so the
  // shared external constant is ONLY used by the two NPU MUL partitions.
  sg.operators[1]->inputs[1] = sg.operators[1]->inputs[0];

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 4> kExternalWeights = {6.0f, 7.0f, 8.0f, 9.0f};
  auto ext_buf = std::make_unique<tflite::ExternalBufferT>();
  ext_buf->id = 1;
  ext_buf->group = 1;
  ext_buf->offset = 0;
  ext_buf->length = sizeof(kExternalWeights);
  tfl_model->external_buffers.push_back(std::move(ext_buf));

  auto initial_serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "multi_partition_ext.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(initial_serialized.StrData(), initial_serialized.Size());
  }
  {
    std::ofstream weights_file(temp_path / "ext_weights.bin",
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);
  auto& model = **loaded_model;

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_TRUE(plugins);
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);

  const auto& [ops, new_model] = *partition_result;
  ASSERT_EQ(ops.size(), 2u);
  ASSERT_EQ(new_model.NumSubgraphs(), 2u);

  // Both sliced subgraphs must have cloned the external constant tensor rather
  // than treating it as a subgraph input.
  EXPECT_EQ(new_model.Subgraphs()[0]->Inputs().size(), 1u);
  EXPECT_EQ(new_model.Subgraphs()[1]->Inputs().size(), 1u);

  const auto& cst_1 = new_model.Subgraphs()[0]->Op(0).Input(1).Weights();
  const auto& cst_2 = new_model.Subgraphs()[1]->Op(0).Input(1).Weights();
  EXPECT_EQ(cst_1.GetBufferManager(), model.Buffers());
  EXPECT_EQ(cst_2.GetBufferManager(), model.Buffers());
  EXPECT_EQ(cst_1.GetBufferId(), cst_2.GetBufferId());
  EXPECT_EQ(cst_1.Buffer().Data(), cst_2.Buffer().Data());
  EXPECT_EQ(cst_1.Buffer().Size(), sizeof(kExternalWeights));

  ApplyPluginsResult apply_result;
  ASSERT_TRUE(ApplyPluginWithPartition(
      plugin, model, std::move(*partition_result), "", apply_result));

  for (auto* dispatch_op : ops) {
    auto op_asset = model.FindOpAsset(dispatch_op);
    ASSERT_TRUE(op_asset);
    auto asset_buf = model.Buffers()->GetBuffer(op_asset->first);
    ASSERT_TRUE(asset_buf);
    EXPECT_THAT(asset_buf->StrView(), ::testing::HasSubstr("DATA:6,7,8,9"));
  }

  // Since both users of external_buffer 1 were absorbed into NPU partitions,
  // SerializeModel should strip external_buffers completely.
  auto compiled_serialized = SerializeModel(std::move(model));
  ASSERT_TRUE(compiled_serialized);
  auto compiled_fb = FlatbufferWrapper::CreateFromBuffer(*compiled_serialized);
  ASSERT_TRUE(compiled_fb);
  auto unpacked = compiled_fb->get()->Unpack();
  EXPECT_TRUE(unpacked->external_buffers.empty());
  EXPECT_TRUE(unpacked->external_buffer_groups.empty());
}

TEST(ApplyTest, MultiSubgraphExternalWeightsMixedCompilation) {
  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(
      testing::GetTestFilePath("cst_multi_subgraph.tflite"));
  ASSERT_TRUE(flatbuffer);
  auto tfl_model = flatbuffer->get()->Unpack();
  ASSERT_EQ(tfl_model->subgraphs.size(), 2u);

  // cst_multi_subgraph.tflite has MUL in both subgraphs. Keep subgraph 0's op
  // MUL (compiled onto NPU) with external_buffer 1, and change subgraph 1's op
  // to ADD (remains on CPU) with external_buffer 2.
  auto add_opcode = std::make_unique<TflOpCode>();
  add_opcode->builtin_code = ::tflite::BuiltinOperator_ADD;
  tfl_model->operator_codes.push_back(std::move(add_opcode));
  const uint32_t add_opcode_idx = tfl_model->operator_codes.size() - 1;

  tfl_model->subgraphs[1]->operators[0]->opcode_index = add_opcode_idx;
  const int32_t sg0_cst_idx = tfl_model->subgraphs[0]->operators[0]->inputs[1];
  const int32_t sg1_cst_idx = tfl_model->subgraphs[1]->operators[0]->inputs[1];
  tfl_model->buffers[tfl_model->subgraphs[0]->tensors[sg0_cst_idx]->buffer]
      ->data.clear();
  tfl_model->subgraphs[0]->tensors[sg0_cst_idx]->buffer = 0;
  tfl_model->subgraphs[0]->tensors[sg0_cst_idx]->external_buffer = 1;
  tfl_model->subgraphs[1]->tensors[sg1_cst_idx]->buffer = 0;
  tfl_model->subgraphs[1]->tensors[sg1_cst_idx]->external_buffer = 2;

  auto group0 = std::make_unique<tflite::ExternalBufferGroupT>();
  group0->name = "";
  auto group1 = std::make_unique<tflite::ExternalBufferGroupT>();
  group1->name = "ext_weights.bin";
  tfl_model->external_buffer_groups.push_back(std::move(group0));
  tfl_model->external_buffer_groups.push_back(std::move(group1));

  constexpr std::array<float, 8> kExternalWeights = {
      1.0f, 2.0f, 3.0f, 4.0f,  // ext_buf 1 (subgraph 0 MUL)
      5.0f, 6.0f, 7.0f, 8.0f   // ext_buf 2 (subgraph 1 ADD)
  };
  auto ext_buf1 = std::make_unique<tflite::ExternalBufferT>();
  ext_buf1->id = 1;
  ext_buf1->group = 1;
  ext_buf1->offset = 0;
  ext_buf1->length = 4 * sizeof(float);
  auto ext_buf2 = std::make_unique<tflite::ExternalBufferT>();
  ext_buf2->id = 2;
  ext_buf2->group = 1;
  ext_buf2->offset = 4 * sizeof(float);
  ext_buf2->length = 4 * sizeof(float);
  tfl_model->external_buffers.push_back(std::move(ext_buf1));
  tfl_model->external_buffers.push_back(std::move(ext_buf2));

  auto initial_serialized = SerializeFlatbuffer(*tfl_model);
  auto temp_dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(temp_dir);
  std::filesystem::path temp_path(std::string(temp_dir->Str()));

  std::filesystem::path model_path = temp_path / "multi_sg_ext.tflite";
  {
    std::ofstream model_file(model_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model_file.is_open());
    model_file.write(initial_serialized.StrData(), initial_serialized.Size());
  }
  {
    std::ofstream weights_file(temp_path / "ext_weights.bin",
                               std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(weights_file.is_open());
    weights_file.write(reinterpret_cast<const char*>(kExternalWeights.data()),
                       sizeof(kExternalWeights));
  }

  auto loaded_model = LoadModelFromFile(model_path.string());
  ASSERT_TRUE(loaded_model);
  auto& model = **loaded_model;

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_TRUE(plugins);
  ASSERT_EQ(plugins->size(), 1);

  ApplyPluginsResult apply_result;
  ASSERT_TRUE(ApplyPlugin(plugins->front(), model, "", {}, apply_result));
  ASSERT_EQ(model.NumSubgraphs(), 2u);
  EXPECT_EQ(model.Subgraph(0).Op(0).OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_EQ(model.Subgraph(1).Op(0).OpCode(), kLiteRtOpCodeTflAdd);

  auto compiled_serialized = SerializeModel(std::move(model));
  ASSERT_TRUE(compiled_serialized);
  auto compiled_fb = FlatbufferWrapper::CreateFromBuffer(*compiled_serialized);
  ASSERT_TRUE(compiled_fb);
  auto unpacked = compiled_fb->get()->Unpack();

  // Subgraph 0's ext_buf 1 is stripped; subgraph 1's ext_buf 2 is retained.
  ASSERT_EQ(unpacked->external_buffers.size(), 1u);
  EXPECT_EQ(unpacked->external_buffers[0]->id, 2u);
}

TEST(PartitionTest, MappedCompositeOp) {
  auto model_wrap = testing::LoadTestFileModel("rms_norm_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);
  // One new subgraph for the consumed composite op only, decomp not consumed.
  ASSERT_EQ(partition_result->second.NumSubgraphs(), 1);
  // Example plugin will select RMS norm composite op during partitioning. only
  // 1 subgraph should remain in the model.
  ASSERT_EQ(model.NumSubgraphs(), 1);
}

TEST(PartitionTest, InlineDecomposition) {
  auto model_wrap = testing::LoadTestFileModel("unsupported_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);
  // One new subgraph for the consumed composite op only, decomp not consumed.
  ASSERT_EQ(partition_result->second.NumSubgraphs(), 1);
  ASSERT_EQ(model.NumSubgraphs(), 1);
  auto main_subgraph = partition_result->second.MainSubgraph();
  ASSERT_EQ(main_subgraph->Ops().size(), 3);
  ASSERT_EQ(main_subgraph->Op(0).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(main_subgraph->Op(1).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(main_subgraph->Op(2).OpCode(), kLiteRtOpCodeTflMul);
}

TEST(PartitionTest,
     InlineDecompositionWithUnsupportedOpInDecompositionSubgraph) {
  auto model_wrap =
      testing::LoadTestFileModel("unsupported_composite_2.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);
  // One new subgraph for the consumed composite op only, decomp not consumed.
  ASSERT_EQ(partition_result->second.NumSubgraphs(), 1);
  ASSERT_EQ(model.NumSubgraphs(), 1);
  auto main_subgraph = partition_result->second.MainSubgraph();
  // There should be 2 ops selected partition
  ASSERT_EQ(main_subgraph->Ops().size(), 2);
  ASSERT_EQ(main_subgraph->Op(0).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(main_subgraph->Op(1).OpCode(), kLiteRtOpCodeTflMul);
  // There should be 2 ops in the main subgraph
  ASSERT_EQ(model.Subgraph(0).Ops().size(), 2);
  ASSERT_EQ(model.Subgraph(0).Op(0).OpCode(), kLiteRtOpCodeTflCustom);
  ASSERT_EQ(model.Subgraph(0).Op(1).OpCode(), kLiteRtOpCodeTflDiv);
}

TEST(PartitionTest, InlineDecompositionWithProducerConsumer) {
  auto model_wrap =
      testing::LoadTestFileModel("unsupported_composite_3.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);
  // One new subgraph for the consumed composite op only, decomp not consumed.
  ASSERT_EQ(partition_result->second.NumSubgraphs(), 1);
  ASSERT_EQ(model.NumSubgraphs(), 1);
  auto main_subgraph = partition_result->second.MainSubgraph();
  // There should be 2 ops selected partition
  ASSERT_EQ(main_subgraph->Ops().size(), 5);
  ASSERT_EQ(main_subgraph->Op(0).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(main_subgraph->Op(1).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(main_subgraph->Op(2).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(main_subgraph->Op(3).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(main_subgraph->Op(4).OpCode(), kLiteRtOpCodeTflMul);
  // There should be 1 op in the main subgraph
  ASSERT_EQ(model.Subgraph(0).Ops().size(), 1);
  ASSERT_EQ(model.Subgraph(0).Op(0).OpCode(), kLiteRtOpCodeTflCustom);
}

TEST(PartitionTest, InlineDecompositionWithUnsupportedProducerConsumer) {
  auto model_wrap =
      testing::LoadTestFileModel("unsupported_composite_4.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);
  // One new subgraph for the consumed composite op only, decomp not consumed.
  ASSERT_EQ(partition_result->second.NumSubgraphs(), 1);
  ASSERT_EQ(model.NumSubgraphs(), 1);
  auto main_subgraph = partition_result->second.MainSubgraph();
  // There should be 2 ops selected partition
  ASSERT_EQ(main_subgraph->Ops().size(), 3);
  ASSERT_EQ(main_subgraph->Op(0).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(main_subgraph->Op(1).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(main_subgraph->Op(2).OpCode(), kLiteRtOpCodeTflMul);
  // There should be 1 op in the main subgraph
  ASSERT_EQ(model.Subgraph(0).Ops().size(), 3);
  ASSERT_EQ(model.Subgraph(0).Op(0).OpCode(), kLiteRtOpCodeTflDiv);
  ASSERT_EQ(model.Subgraph(0).Op(1).OpCode(), kLiteRtOpCodeTflCustom);
  ASSERT_EQ(model.Subgraph(0).Op(2).OpCode(), kLiteRtOpCodeTflDiv);
}

TEST(PartitionTest, SimpleNpuCallComposite) {
  auto model_wrap = testing::LoadTestFileModel("simple_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  auto* decomp = model.Subgraphs()[1];

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);

  auto& ops = partition_result->first;
  ASSERT_EQ(ops.size(), 1);
  ASSERT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);

  const auto& sgs = partition_result->second.Subgraphs();
  ASSERT_EQ(sgs.size(), 1);
  ASSERT_EQ(sgs.front(), decomp);
}

TEST(PartitionTest, MultiNpuCallComposite) {
  auto model_wrap = testing::LoadTestFileModel("multi_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  ASSERT_EQ(model.NumSubgraphs(), 4);
  auto* decomp1 = model.Subgraphs()[1];
  auto* decomp2 = model.Subgraphs()[3];

  {
    // Before partitioning, the model has 4 subgraphs. 1-3 are decompositions,
    // and 0 is the main subgraph.
    auto npu_call_op_0_option =
        GetOptionsAs<CompositeOptions>(model.Subgraph(0).Ops()[0]);
    ASSERT_TRUE(npu_call_op_0_option);
    ASSERT_EQ(npu_call_op_0_option->subgraph, 1);

    auto non_npu_call_op_0_option =
        GetOptionsAs<CompositeOptions>(model.Subgraph(0).Ops()[1]);
    ASSERT_TRUE(non_npu_call_op_0_option);
    ASSERT_EQ(non_npu_call_op_0_option->subgraph, 2);

    auto npu_call_op_1_option =
        GetOptionsAs<CompositeOptions>(model.Subgraph(0).Ops()[2]);
    ASSERT_TRUE(npu_call_op_1_option);
    ASSERT_EQ(npu_call_op_1_option->subgraph, 3);
  }

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);

  {
    // Subgraphs to be compiled will be moved to the result from the model.
    // Non-npu-call decompositions will be reindexed.
    ASSERT_EQ(model.NumSubgraphs(), 1);
  }

  {
    // All npu call ops are now dispatch ops.
    auto& ops = partition_result->first;

    ASSERT_EQ(ops.size(), 2);
    auto* first_dispatch_op = ops.front();
    auto* second_dispatch_op = ops.back();

    ASSERT_EQ(first_dispatch_op->OpCode(), kLiteRtOpCodeTflCustom);
    ASSERT_EQ(first_dispatch_op, model.Subgraphs()[0]->Ops().front());

    ASSERT_EQ(second_dispatch_op->OpCode(), kLiteRtOpCodeTflCustom);
    ASSERT_EQ(second_dispatch_op, model.Subgraphs()[0]->Ops().back());
  }

  {
    // Bodies to compile are the decompositions of npu call ops.
    const auto& sgs = partition_result->second.Subgraphs();

    ASSERT_EQ(sgs.size(), 2);
    ASSERT_EQ(sgs.front(), decomp1);
    ASSERT_EQ(sgs.back(), decomp2);
  }
}

TEST(PartitionTest, NestedNpuCallComposite) {
  auto model_wrap = testing::LoadTestFileModel("nested_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  ASSERT_EQ(model.NumSubgraphs(), 3);

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);

  auto& ops = partition_result->first;
  ASSERT_EQ(ops.size(), 1);
  ASSERT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);

  const auto& sgs = partition_result->second.Subgraphs();
  ASSERT_EQ(sgs.size(), 1);
  ASSERT_EQ(sgs.front()->Op(0).OpCode(), kLiteRtOpCodeShloComposite);
}

TEST(PartitionModelTest, PartitionIsland) {
  auto model_wrap = testing::LoadTestFileModel("island_partial.tflite");
  auto& model = *model_wrap.Get();

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 1);

  const auto& [ops, new_model] = *partition_result;

  EXPECT_EQ(ops.size(), 2);

  EXPECT_EQ(new_model.NumSubgraphs(), 2);
  EXPECT_EQ(new_model.Subgraphs().at(0)->Ops().size(), 3);
  EXPECT_EQ(new_model.Subgraphs().at(1)->Ops().size(), 1);
}

TEST(CheckCompilerCompatibilityTest, Simple) {
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();
  ASSERT_TRUE(plugin.CheckCompilerCompatibility("ExampleSocModel"));
  ASSERT_FALSE(plugin.CheckCompilerCompatibility("UnsupportedSocModel"));
}

LiteRtStatus ReplaceAddWithMul(const LiteRtCompilerContext* context,
                               LiteRtBuilder builder, LiteRtOp op) {
  if (op->OpCode() != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusErrorNotFound;
  }
  LiteRtBuilderT* b = reinterpret_cast<LiteRtBuilderT*>(builder);
  auto& new_op = b->BuildOp(kLiteRtOpCodeTflMul, op->Inputs(), op->Outputs());
  (void)new_op;
  b->EraseOp(op);
  return kLiteRtStatusOk;
}

}  // namespace

class CompilerPluginFriend : public ::testing::Test {
 protected:
  CompilerPlugin CreatePlugin() { return CompilerPlugin(); }
  void AddTransformation(CompilerPlugin& plugin, LiteRtTransformation t) {
    plugin.transformations_.push_back(t);
  }
};

TEST_F(CompilerPluginFriend, GreedyPatternMatchAndRewrite) {
  CompilerPlugin plugin = CreatePlugin();
  LiteRtTransformation transformation;
  transformation.name = "ReplaceAddWithMul";
  transformation.pattern = ReplaceAddWithMul;
  AddTransformation(plugin, transformation);

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& tensor0 = subgraph.EmplaceTensor();
  auto& tensor1 = subgraph.EmplaceTensor();
  auto& tensor2 = subgraph.EmplaceTensor();

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&tensor0, op);
  AttachInput(&tensor1, op);
  AttachOutput(&tensor2, op);

  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflAdd);

  LITERT_ASSERT_OK(plugin.GreedyPatternMatchAndRewrite(model));

  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflMul);
}

TEST_F(CompilerPluginFriend, GreedyPatternMatchAndRewriteIterative) {
  CompilerPlugin plugin = CreatePlugin();
  // Transformation 1: Add -> Mul
  LiteRtTransformation transformation1;
  transformation1.name = "ReplaceAddWithMul";
  transformation1.pattern = ReplaceAddWithMul;
  AddTransformation(plugin, transformation1);

  // Transformation 2: Mul -> Sub
  LiteRtTransformation transformation2;
  transformation2.name = "ReplaceMulWithSub";
  transformation2.pattern = [](const LiteRtCompilerContext* context,
                               LiteRtBuilder builder,
                               LiteRtOp op) -> LiteRtStatus {
    if (op->OpCode() != kLiteRtOpCodeTflMul) {
      return kLiteRtStatusErrorNotFound;
    }
    LiteRtBuilderT* b = reinterpret_cast<LiteRtBuilderT*>(builder);
    auto& new_op = b->BuildOp(kLiteRtOpCodeTflSub, op->Inputs(), op->Outputs());
    (void)new_op;
    b->EraseOp(op);
    return kLiteRtStatusOk;
  };
  AddTransformation(plugin, transformation2);

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& tensor0 = subgraph.EmplaceTensor();
  auto& tensor1 = subgraph.EmplaceTensor();
  auto& tensor2 = subgraph.EmplaceTensor();

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&tensor0, op);
  AttachInput(&tensor1, op);
  AttachOutput(&tensor2, op);

  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflAdd);

  LITERT_ASSERT_OK(plugin.GreedyPatternMatchAndRewrite(model));

  ASSERT_EQ(subgraph.Ops().size(), 1);
  // Should undergo Add -> Mul -> Sub
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflSub);
}

TEST_F(CompilerPluginFriend, MaxTransformationIterations) {
  CompilerPlugin plugin = CreatePlugin();
  plugin.SetMaxTransformationIterations(5);

  // Define a helper template for transformation.
  auto add_transform = [&](const char* name, LiteRtPatternFn pattern) {
    LiteRtTransformation t;
    t.name = name;
    t.pattern = pattern;
    AddTransformation(plugin, t);
  };

  // Chain of transformations:
  // 1. Add -> Mul
  add_transform(
      "Add_to_Mul",
      [](const LiteRtCompilerContext* context, LiteRtBuilder builder,
         LiteRtOp op) -> LiteRtStatus {
        if (op->OpCode() != kLiteRtOpCodeTflAdd)
          return kLiteRtStatusErrorNotFound;
        LiteRtBuilderT* b = reinterpret_cast<LiteRtBuilderT*>(builder);
        auto& new_op =
            b->BuildOp(kLiteRtOpCodeTflMul, op->Inputs(), op->Outputs());
        (void)new_op;
        b->EraseOp(op);
        return kLiteRtStatusOk;
      });
  // 2. Mul -> Sub
  add_transform(
      "Mul_to_Sub",
      [](const LiteRtCompilerContext* context, LiteRtBuilder builder,
         LiteRtOp op) -> LiteRtStatus {
        if (op->OpCode() != kLiteRtOpCodeTflMul)
          return kLiteRtStatusErrorNotFound;
        LiteRtBuilderT* b = reinterpret_cast<LiteRtBuilderT*>(builder);
        auto& new_op =
            b->BuildOp(kLiteRtOpCodeTflSub, op->Inputs(), op->Outputs());
        (void)new_op;
        b->EraseOp(op);
        return kLiteRtStatusOk;
      });
  // 3. Sub -> Div
  add_transform(
      "Sub_to_Div",
      [](const LiteRtCompilerContext* context, LiteRtBuilder builder,
         LiteRtOp op) -> LiteRtStatus {
        if (op->OpCode() != kLiteRtOpCodeTflSub)
          return kLiteRtStatusErrorNotFound;
        LiteRtBuilderT* b = reinterpret_cast<LiteRtBuilderT*>(builder);
        auto& new_op =
            b->BuildOp(kLiteRtOpCodeTflDiv, op->Inputs(), op->Outputs());
        (void)new_op;
        b->EraseOp(op);
        return kLiteRtStatusOk;
      });
  // 4. Div -> Cos
  add_transform(
      "Div_to_Cos",
      [](const LiteRtCompilerContext* context, LiteRtBuilder builder,
         LiteRtOp op) -> LiteRtStatus {
        if (op->OpCode() != kLiteRtOpCodeTflDiv)
          return kLiteRtStatusErrorNotFound;
        LiteRtBuilderT* b = reinterpret_cast<LiteRtBuilderT*>(builder);
        auto& new_op =
            b->BuildOp(kLiteRtOpCodeTflCos, op->Inputs(), op->Outputs());
        (void)new_op;
        b->EraseOp(op);
        return kLiteRtStatusOk;
      });
  // 5. Cos -> Sin
  add_transform(
      "Cos_to_Sin",
      [](const LiteRtCompilerContext* context, LiteRtBuilder builder,
         LiteRtOp op) -> LiteRtStatus {
        if (op->OpCode() != kLiteRtOpCodeTflCos)
          return kLiteRtStatusErrorNotFound;
        LiteRtBuilderT* b = reinterpret_cast<LiteRtBuilderT*>(builder);
        auto& new_op =
            b->BuildOp(kLiteRtOpCodeTflSin, op->Inputs(), op->Outputs());
        (void)new_op;
        b->EraseOp(op);
        return kLiteRtStatusOk;
      });
  // 6. Sin -> Log (Should not happen if max iterations = 5)
  add_transform(
      "Sin_to_Log",
      [](const LiteRtCompilerContext* context, LiteRtBuilder builder,
         LiteRtOp op) -> LiteRtStatus {
        if (op->OpCode() != kLiteRtOpCodeTflSin)
          return kLiteRtStatusErrorNotFound;
        LiteRtBuilderT* b = reinterpret_cast<LiteRtBuilderT*>(builder);
        auto& new_op =
            b->BuildOp(kLiteRtOpCodeTflLog, op->Inputs(), op->Outputs());
        (void)new_op;
        b->EraseOp(op);
        return kLiteRtStatusOk;
      });

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& tensor0 = subgraph.EmplaceTensor();
  auto& tensor1 = subgraph.EmplaceTensor();
  auto& tensor2 = subgraph.EmplaceTensor();

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&tensor0, op);
  AttachInput(&tensor1, op);
  AttachOutput(&tensor2, op);

  LITERT_ASSERT_OK(plugin.GreedyPatternMatchAndRewrite(model));
  // Iteration 0: Add -> Mul
  // Iteration 1: Mul -> Sub
  // Iteration 2: Sub -> Div
  // Iteration 3: Div -> Cos
  // Iteration 4: Cos -> Sin
  // Iteration 5: Stop (limit reached before checking Sin -> Log)

  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops().front()->OpCode(), kLiteRtOpCodeTflSin);
}

TEST_F(CompilerPluginFriend, MultipleIndependentMatches) {
  CompilerPlugin plugin = CreatePlugin();
  LiteRtTransformation t;
  t.name = "Add_to_Mul";
  t.pattern = ReplaceAddWithMul;
  AddTransformation(plugin, t);

  LiteRtModelT model;
  auto& subgraph = model.EmplaceSubgraph();
  auto& t0 = subgraph.EmplaceTensor();
  auto& t1 = subgraph.EmplaceTensor();
  auto& t2 = subgraph.EmplaceTensor();
  auto& t3 = subgraph.EmplaceTensor();

  // Op 1: Add
  auto& op1 = subgraph.EmplaceOp();
  op1.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&t0, op1);
  AttachInput(&t0, op1);  // Dummy inputs
  AttachOutput(&t1, op1);

  // Op 2: Add
  auto& op2 = subgraph.EmplaceOp();
  op2.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&t2, op2);
  AttachInput(&t2, op2);
  AttachOutput(&t3, op2);

  LITERT_ASSERT_OK(plugin.GreedyPatternMatchAndRewrite(model));

  ASSERT_EQ(subgraph.Ops().size(), 2);
  EXPECT_EQ(subgraph.Ops()[0]->OpCode(), kLiteRtOpCodeTflMul);
  EXPECT_EQ(subgraph.Ops()[1]->OpCode(), kLiteRtOpCodeTflMul);
}

}  // namespace litert::internal
