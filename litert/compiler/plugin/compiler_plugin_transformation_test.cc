// Copyright 2025 Google LLC.
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

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"

namespace litert::internal {
namespace {

constexpr absl::string_view kTestPluginSearchPath = "vendors/examples";

using testing::GetLiteRtPath;

TEST(TransformationTest, ApplyTransformation) {
  auto model_wrap = testing::LoadTestFileModel("sqrt_mean_mul_ops.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  ASSERT_EQ(plugins->size(), 1);

  ASSERT_EQ(model.MainSubgraph()->Op(0).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(model.MainSubgraph()->Op(1).OpCode(), kLiteRtOpCodeTflMean);
  ASSERT_EQ(model.MainSubgraph()->Op(2).OpCode(), kLiteRtOpCodeTflSqrt);

  auto transform_result = TransformModel(plugins->front(), model);
  ASSERT_TRUE(transform_result);

  ASSERT_EQ(model.MainSubgraph()->Ops().size(), 2);
  EXPECT_EQ(model.MainSubgraph()->Op(0).OpCode(), kLiteRtOpCodeTflAbs);
  EXPECT_EQ(model.MainSubgraph()->Op(1).OpCode(), kLiteRtOpCodeTflSqrt);

  auto serialized = SerializeModel(std::move(*model_wrap.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));
}

TEST(TransformationTest, PartiallyMatch) {
  auto model_wrap = testing::LoadTestFileModel("simple_mul_op.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  auto transform_result = TransformModel(plugins->front(), model);
  ASSERT_TRUE(transform_result);
  ASSERT_EQ(model.MainSubgraph()->Ops().size(), 1);
  EXPECT_EQ(model.MainSubgraph()->Op(0).OpCode(), kLiteRtOpCodeTflMul);

  auto serialized = SerializeModel(std::move(*model_wrap.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));
}

TEST(TransformationTest, NoMatch) {
  auto model_wrap = testing::LoadTestFileModel("simple_add_op.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});

  auto transform_result = TransformModel(plugins->front(), model);
  ASSERT_TRUE(transform_result);
  ASSERT_EQ(model.MainSubgraph()->Ops().size(), 1);
  EXPECT_EQ(model.MainSubgraph()->Op(0).OpCode(), kLiteRtOpCodeTflAdd);

  auto serialized = SerializeModel(std::move(*model_wrap.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));
}

TEST(TransformationTest, MatchesOnce) {
  auto model_wrap = testing::LoadTestFileModel("sqrt_mean_mul_multiple.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.MainSubgraph()->Ops().size(), 6);

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();
  plugin.SetMaxTransformationIterations(1);
  auto transform_result = TransformModel(plugin, model);
  ASSERT_EQ(model.MainSubgraph()->Ops().size(), 5);

  auto serialized = SerializeModel(std::move(*model_wrap.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));
}

TEST(TransformationTest, MatchesTwice) {
  auto model_wrap = testing::LoadTestFileModel("sqrt_mean_mul_multiple.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.MainSubgraph()->Ops().size(), 6);

  auto plugins =
      CompilerPlugin::LoadPlugins({GetLiteRtPath(kTestPluginSearchPath)});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();
  plugin.SetMaxTransformationIterations(2);
  auto transform_result = TransformModel(plugin, model);
  ASSERT_EQ(model.MainSubgraph()->Ops().size(), 4);

  auto serialized = SerializeModel(std::move(*model_wrap.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));
}

}  // namespace
}  // namespace litert::internal
