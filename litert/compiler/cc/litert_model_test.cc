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
#include <cstdint>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
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

bool block_wise_quantization_called = false;

LiteRtStatus RecordBlockWiseQuantizationCall(
    LiteRtTensor tensor, LiteRtQuantizationBlockWise* quantization) {
  block_wise_quantization_called = true;
  return kLiteRtStatusOk;
}

TEST(ModelTest, BlockWiseQuantizationSkipsFunctionOutsideContext) {
  auto cc_model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");

  // Simulates an older compiler host whose context ends before
  // `get_block_wise_quantization` (added in 1.1.0). The slot is still
  // populated in memory and must not be called.
  LiteRtCompilerContext ctx = *LrtGetCompilerContext();
  ctx.get_block_wise_quantization = RecordBlockWiseQuantizationCall;
  ctx.abi_header.struct_size = static_cast<uint16_t>(
      offsetof(LiteRtCompilerContext, get_block_wise_quantization));

  litert::compiler::Model model(&ctx, cc_model.Get());
  auto subgraph = model.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());
  auto inputs = subgraph->Inputs();
  ASSERT_FALSE(inputs.empty());

  block_wise_quantization_called = false;
  LiteRtQuantizationBlockWise quantization = inputs[0].BlockWiseQuantization();
  EXPECT_FALSE(block_wise_quantization_called);
  EXPECT_EQ(quantization.scales, nullptr);
  EXPECT_EQ(quantization.zero_points, nullptr);
  EXPECT_EQ(quantization.block_size, 0);
}

// A partitions model: one entry subgraph followed by one helper.
TEST(ModelTest, NumEntrySubgraphs) {
  auto cc_model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  LiteRtModel c_model = cc_model.Get();
  c_model->EmplaceSubgraph();
  c_model->SetNumEntrySubgraphs(1);
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();

  EXPECT_EQ(litert::compiler::Model(ctx, c_model).NumEntrySubgraphs(), 1);

  // A context from before `get_num_entry_subgraphs` was added: every subgraph
  // is treated as an entry.
  LiteRtCompilerContext old_ctx = *ctx;
  old_ctx.abi_header.struct_size =
      offsetof(LiteRtCompilerContext, get_num_entry_subgraphs);
  EXPECT_EQ(litert::compiler::Model(&old_ctx, c_model).NumEntrySubgraphs(), 2);
}

}  // namespace
}  // namespace litert
