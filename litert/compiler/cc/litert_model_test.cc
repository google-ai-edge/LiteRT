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

  for (const auto& input : inputs) {
    auto parent_sg = input.GetSubgraph();
    ASSERT_TRUE(parent_sg.HasValue());
    EXPECT_EQ(*parent_sg, subgraph);
    EXPECT_FALSE(input.IsSubgraphOutput());
  }
  for (const auto& output : outputs) {
    auto parent_sg = output.GetSubgraph();
    ASSERT_TRUE(parent_sg.HasValue());
    EXPECT_EQ(*parent_sg, subgraph);
    EXPECT_TRUE(output.IsSubgraphOutput());
  }
  // Intermediate op outputs in `simple_multi_op.tflite` belong to `subgraph`
  // without being subgraph outputs.
  ASSERT_GT(ops.size(), 1);
  auto first_op_outputs = ops.front().Outputs();
  ASSERT_FALSE(first_op_outputs.empty());
  auto first_parent_sg = first_op_outputs.front().GetSubgraph();
  ASSERT_TRUE(first_parent_sg.HasValue());
  EXPECT_EQ(*first_parent_sg, subgraph);
  EXPECT_FALSE(first_op_outputs.front().IsSubgraphOutput());

  litert::compiler::Tensor null_tensor(ctx, nullptr);
  auto null_sg = null_tensor.GetSubgraph();
  ASSERT_FALSE(null_sg.HasValue());
  EXPECT_EQ(null_sg.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_FALSE(null_tensor.IsSubgraphOutput());
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

bool get_tensor_subgraph_called = false;

LiteRtStatus RecordGetTensorSubgraphCall(LiteRtTensor tensor,
                                         LiteRtSubgraph* subgraph) {
  get_tensor_subgraph_called = true;
  return kLiteRtStatusOk;
}

TEST(ModelTest, GetSubgraphSkipsFunctionOutsideContext) {
  auto cc_model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");

  // Simulates an older compiler host whose context ends before
  // `get_tensor_subgraph` (added in 1.7.0).
  LiteRtCompilerContext ctx = *LrtGetCompilerContext();
  ctx.get_tensor_subgraph = RecordGetTensorSubgraphCall;
  ctx.abi_header.struct_size = static_cast<uint16_t>(
      offsetof(LiteRtCompilerContext, get_tensor_subgraph));

  litert::compiler::Model model(&ctx, cc_model.Get());
  auto subgraph = model.Subgraph(0);
  ASSERT_TRUE(subgraph.HasValue());
  auto outputs = subgraph->Outputs();
  ASSERT_FALSE(outputs.empty());

  get_tensor_subgraph_called = false;
  auto parent_sg = outputs[0].GetSubgraph();
  EXPECT_FALSE(get_tensor_subgraph_called);
  ASSERT_FALSE(parent_sg.HasValue());
  EXPECT_EQ(parent_sg.Error().Status(), kLiteRtStatusErrorWrongVersion);
  EXPECT_TRUE(outputs[0].IsSubgraphOutput());
  EXPECT_FALSE(get_tensor_subgraph_called);
}

}  // namespace
}  // namespace litert
