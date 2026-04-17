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

  auto num_subgraphs_or = model.NumSubgraphs();
  ASSERT_TRUE(num_subgraphs_or.HasValue());
  EXPECT_EQ(num_subgraphs_or.Value(), 1);

  auto subgraph_or = model.GetSubgraph(0);
  ASSERT_TRUE(subgraph_or.HasValue());
  auto subgraph = subgraph_or.Value();

  auto ops_or = subgraph.Ops();
  ASSERT_TRUE(ops_or.HasValue());
  EXPECT_GT(ops_or.Value().size(), 0);

  auto inputs_or = subgraph.Inputs();
  ASSERT_TRUE(inputs_or.HasValue());
  EXPECT_FALSE(inputs_or.Value().empty());

  auto outputs_or = subgraph.Outputs();
  ASSERT_TRUE(outputs_or.HasValue());
  EXPECT_FALSE(outputs_or.Value().empty());
}

}  // namespace
}  // namespace litert
