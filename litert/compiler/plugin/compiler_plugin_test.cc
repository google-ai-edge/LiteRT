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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_builder.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/core/build_stamp.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/tools/dump.h"
#include "litert/vendors/c/litert_compiler_plugin.h"

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
  auto litert_options = Options::Create();
  auto compiler_options = CompilerOptions::Create();
  compiler_options->SetPartitionStrategy(
      kLiteRtCompilerOptionsPartitionStrategyDefault);
  litert_options->AddOpaqueOptions(std::move(*compiler_options));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin,
      CompilerPlugin::FindPlugin(kTestManufacturer,
                                 {GetLiteRtPath(kTestPluginSearchPath)},
                                 /*env=*/nullptr, litert_options->Get()));
  EXPECT_EQ(plugin.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, GetOptionsFromTestPluginOk) {
  auto litert_options = Options::Create();
  auto compiler_options = CompilerOptions::Create();
  compiler_options->SetPartitionStrategy(
      kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
  litert_options->AddOpaqueOptions(std::move(*compiler_options));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto plugin,
      CompilerPlugin::FindPlugin(kTestManufacturer,
                                 {GetLiteRtPath(kTestPluginSearchPath)},
                                 /*env=*/nullptr, litert_options->Get()));

  auto compiler_options_from_plugin = plugin.CompilerOptions();
  LiteRtCompilerOptionsPartitionStrategy strategy;
  auto status = LiteRtGetCompilerOptionsPartitionStrategy(
      *compiler_options_from_plugin, &strategy);
  EXPECT_EQ(status, kLiteRtStatusOk);
  EXPECT_EQ(strategy, kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
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

  auto result = plugins->front().Compile(&model);
  ASSERT_TRUE(result);

  auto byte_code = result->ByteCode();
  ASSERT_TRUE(byte_code && byte_code->Size() > 0);

  auto num_calls = result->NumCalls();
  ASSERT_TRUE(num_calls);
  ASSERT_EQ(*num_calls, 1);

  auto call_info = result->CallInfo(0);
  ASSERT_TRUE(call_info);
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

  ASSERT_TRUE(ApplyPlugin(plugins->front(), model));
  ASSERT_EQ(model.NumSubgraphs(), 1);

  auto& subgraph = *model.MainSubgraph();
  ASSERT_EQ(subgraph.Ops().size(), 1);

  auto* op = subgraph.Ops().front();

  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_TRUE(model.FindOpAsset(op));

  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));
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

  ASSERT_TRUE(ApplyPluginWithPartition(plugins->front(), model,
                                       std::move(*partition_result)));

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

  ASSERT_TRUE(ApplyPlugin(plugins->front(), model));
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
      litert::Environment::Option{
          /*.tag=*/litert::Environment::OptionTag::CompilerPluginLibraryDir,
          /*.value=*/plugin_search_path.c_str(),
      },
  };
  auto env = litert::Environment::Create(environment_options);
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

LiteRtStatus ReplaceAddWithMul(LiteRtBuilder builder, LiteRtOp op) {
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
  transformation2.pattern = [](LiteRtBuilder builder,
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
      "Add_to_Mul", [](LiteRtBuilder builder, LiteRtOp op) -> LiteRtStatus {
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
      "Mul_to_Sub", [](LiteRtBuilder builder, LiteRtOp op) -> LiteRtStatus {
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
      "Sub_to_Div", [](LiteRtBuilder builder, LiteRtOp op) -> LiteRtStatus {
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
      "Div_to_Cos", [](LiteRtBuilder builder, LiteRtOp op) -> LiteRtStatus {
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
      "Cos_to_Sin", [](LiteRtBuilder builder, LiteRtOp op) -> LiteRtStatus {
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
      "Sin_to_Log", [](LiteRtBuilder builder, LiteRtOp op) -> LiteRtStatus {
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
