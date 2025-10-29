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

#include "tflite/experimental/litert/compiler/plugin/compiler_plugin.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "testing/base/public/unique-test-directory.h"
#include "absl/strings/string_view.h"
#include "tflite/experimental/litert/c/litert_op_code.h"
#include "tflite/experimental/litert/core/filesystem.h"
#include "tflite/experimental/litert/test/common.h"
#include "tflite/experimental/litert/tools/dump.h"

namespace litert::internal {
namespace {

using ::testing::UniqueTestDirectory;

constexpr absl::string_view kTestPluginSearchPath =
    "tflite/experimental/litert/vendors/examples";

constexpr absl::string_view kTestManufacturer = "ExampleSocManufacturer";
constexpr absl::string_view kTestModels = "ExampleSocModel";

TEST(CompilerPluginTest, LoadTestPlugin) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);
  ASSERT_EQ(plugins->front().SocModels().size(), 1);
  EXPECT_EQ(plugins->front().SocModels().front(), kTestModels);
}

TEST(CompilerPluginTest, LoadTestPluginWithMalformed) {
  const auto dir = UniqueTestDirectory();
  Touch(Join({dir, "notLibLiteRt.so"}));

  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MultipleValidPlugins) {
  auto plugins = CompilerPlugin::LoadPlugins(
      {kTestPluginSearchPath, kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 2);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);
  EXPECT_EQ(plugins->back().SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MoveAssign) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  CompilerPlugin other = std::move(plugins->front());

  EXPECT_EQ(other.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MoveConstruct) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  CompilerPlugin other(std::move(plugins->front()));

  EXPECT_EQ(other.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, SocModels) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  EXPECT_THAT(plugins->front().SocModels(),
              ::testing::ElementsAreArray({kTestModels}));
}

TEST(CompilerPluginTest, Partition) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  auto model = testing::LoadTestFileModel("mul_simple.tflite");
  auto subgraph = model.MainSubgraph();
  auto ops = plugins->front().Partition(*subgraph);
  ASSERT_TRUE(ops);

  EXPECT_EQ(ops->size(), 2);
}

TEST(CompilerPluginTest, CompileModel) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  auto model = testing::LoadTestFileModel("mul_simple.tflite");
  auto subgraph = model.MainSubgraph();

  std::ostringstream byte_code_out;
  std::vector<std::string> call_info_out;
  LITERT_ASSERT_STATUS_OK(plugins->front().Compile(
      kTestModels, {subgraph->Get()}, byte_code_out, call_info_out));

  EXPECT_GT(byte_code_out.str().size(), 0);
  EXPECT_EQ(call_info_out.size(), 1);
}

TEST(CompilerPluginTest, Dump) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);

  std::stringstream dump;
  Dump(plugins->front(), dump);

  ASSERT_EQ(dump.view(),
            "SocManufacturer: ExampleSocManufacturer\nSocModels: { "
            "ExampleSocModel }\n");
}

TEST(ApplyPluginTest, ApplyPlugin) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  auto model = testing::LoadTestFileModel("mul_simple.tflite");
  ASSERT_TRUE(model);

  auto npu_code = ApplyPlugin(plugins->front(), model);
  ASSERT_TRUE(npu_code);
  EXPECT_GT(npu_code->Size(), 0);

  auto ops = model.MainSubgraph()->Ops();
  ASSERT_EQ(ops.size(), 1);
  EXPECT_EQ(ops.front().Code(), kLiteRtOpCodeTflCustom);
  EXPECT_EQ(ops.front().Get()->CustomOptions().StrView(), "Partition_0");
}

}  // namespace
}  // namespace litert::internal
