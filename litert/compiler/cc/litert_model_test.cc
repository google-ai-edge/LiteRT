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

#include "litert/compiler/cc/litert_model.h"

#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/test/common.h"

namespace litert {
namespace {

TEST(ModelTest, BasicModelInspection) {
  auto cc_model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  LiteRtModel c_model = cc_model.Get();

  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, c_model);

  auto num_subgraphs = model.NumSubgraphs();
  EXPECT_EQ(num_subgraphs, 1);

  auto subgraph_or = model.Subgraph(0);
  ASSERT_TRUE(subgraph_or.HasValue());
  auto subgraph = subgraph_or.Value();

  auto ops = subgraph.Ops();
  EXPECT_GT(ops.size(), 0);

  auto inputs = subgraph.Inputs();
  EXPECT_FALSE(inputs.empty());

  auto outputs = subgraph.Outputs();
  EXPECT_FALSE(outputs.empty());
}

}  // namespace
}  // namespace litert
