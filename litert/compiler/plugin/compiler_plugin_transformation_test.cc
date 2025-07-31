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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"

constexpr absl::string_view kTestPluginSearchPath =
    "third_party/odml/litert/litert/vendors/examples";

namespace litert::internal {
namespace {

TEST(TransformationTest, ApplyTransformation) {
  auto model_wrap = testing::LoadTestFileModel("sqrt_mean_mul.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 1);

  ASSERT_EQ(model.MainSubgraph()->Op(0).OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(model.MainSubgraph()->Op(1).OpCode(), kLiteRtOpCodeTflMean);
  ASSERT_EQ(model.MainSubgraph()->Op(2).OpCode(), kLiteRtOpCodeTflSqrt);

  auto transform_result = TransformModel(plugins->front(), model);
  ASSERT_TRUE(transform_result);

  ASSERT_EQ(model.MainSubgraph()->Ops().size(), 2);
  EXPECT_EQ(model.MainSubgraph()->Op(0).OpCode(), kLiteRtOpCodeTflAbs);
  EXPECT_EQ(model.MainSubgraph()->Op(1).OpCode(), kLiteRtOpCodeTflSqrt);
}

}  // namespace
}  // namespace litert::internal
