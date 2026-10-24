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

#include <cstddef>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"

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

TEST(ModelTest, NumEntrySubgraphs) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  ASSERT_TRUE(LITERT_ABI_IS_COMPATIBLE(ctx, 1, 6));
  ASSERT_TRUE(LITERT_ABI_HAS_API(ctx, 1, get_num_model_entry_subgraphs));

  LiteRtModelT c_model;
  c_model.EmplaceSubgraph();
  c_model.EmplaceSubgraph();
  litert::compiler::Model model(ctx, &c_model);
  EXPECT_EQ(model.NumEntrySubgraphs(), 2);

  // E.g. a partitions model with one entry and one private region subgraph.
  c_model.SetNumEntrySubgraphs(1);
  EXPECT_EQ(model.NumSubgraphs(), 2);
  EXPECT_EQ(model.NumEntrySubgraphs(), 1);

  LiteRtParamIndex num_entries = 0;
  EXPECT_EQ(ctx->get_num_model_entry_subgraphs(nullptr, &num_entries),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(ModelTest, NumEntrySubgraphsFallsBackOnOlderRuntime) {
  // A plugin built against LiteRtCompilerContext 1.6.0 headers running on a
  // 1.5.0 runtime must not read past the end of the function table. Older
  // runtimes never pass private region subgraphs to plugins.
  LiteRtCompilerContext old_ctx = *LrtGetCompilerContext();
  old_ctx.abi_header.minor_version = 5;
  old_ctx.abi_header.struct_size =
      offsetof(LiteRtCompilerContext, get_num_model_entry_subgraphs);
  ASSERT_FALSE(LITERT_ABI_HAS_API(&old_ctx, 1, get_num_model_entry_subgraphs));

  LiteRtModelT c_model;
  c_model.EmplaceSubgraph();
  c_model.EmplaceSubgraph();
  c_model.SetNumEntrySubgraphs(1);
  litert::compiler::Model model(&old_ctx, &c_model);
  EXPECT_EQ(model.NumEntrySubgraphs(), 2);
}

}  // namespace
}  // namespace litert
