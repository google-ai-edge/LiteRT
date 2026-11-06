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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/test/common.h"
#include "litert/test/load_test_model.h"
#include "litert/test/matchers.h"

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

TEST(ModelTest, SerializeSubgraphRoundTrip) {
  auto cc_model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());

  LITERT_ASSERT_OK_AND_ASSIGN(litert::compiler::Subgraph subgraph,
                              model.Subgraph(0));
  LITERT_ASSERT_OK_AND_ASSIGN(OwningBufferRef<uint8_t> serialized,
                              subgraph.Serialize());
  ASSERT_GT(serialized.Size(), 0);

  LiteRtModel reloaded_c_model = nullptr;
  LITERT_ASSERT_OK(LiteRtCreateModelFromBuffer(
      /*environment=*/nullptr, serialized.Data(), serialized.Size(),
      &reloaded_c_model));
  litert::compiler::Model reloaded_model(ctx, reloaded_c_model);
  EXPECT_EQ(reloaded_model.NumSubgraphs(), 1);
  LITERT_ASSERT_OK_AND_ASSIGN(litert::compiler::Subgraph reloaded_sg,
                              reloaded_model.Subgraph(0));
  EXPECT_EQ(reloaded_sg.Ops().size(), subgraph.Ops().size());
  EXPECT_EQ(reloaded_sg.Inputs().size(), subgraph.Inputs().size());
  EXPECT_EQ(reloaded_sg.Outputs().size(), subgraph.Outputs().size());
  LiteRtDestroyModel(reloaded_c_model);
}

TEST(ModelTest, SerializeSubgraphReturnsUnsupportedForOlderCompilerContext) {
  auto cc_model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  LiteRtCompilerContext legacy_ctx = *LrtGetCompilerContext();
  legacy_ctx.abi_header.minor_version = 5;
  legacy_ctx.abi_header.struct_size =
      offsetof(LiteRtCompilerContext, serialize_subgraph);

  litert::compiler::Model model(&legacy_ctx, cc_model.Get());
  LITERT_ASSERT_OK_AND_ASSIGN(litert::compiler::Subgraph subgraph,
                              model.Subgraph(0));
  EXPECT_THAT(subgraph.Serialize(),
              ::testing::litert::IsError(kLiteRtStatusErrorUnsupported));
}

TEST(ModelTest, SerializeSubgraphRejectsNullArguments) {
  auto cc_model = litert::testing::LoadTestFileModel("simple_multi_op.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraph subgraph = nullptr;
  ASSERT_EQ(ctx->get_model_subgraph(cc_model.Get(), 0, &subgraph),
            kLiteRtStatusOk);
  uint8_t* buf = nullptr;
  size_t size = 0;
  size_t offset = 0;
  const LiteRtModelSerializationOptions options = {/*bytecode_alignment=*/1};

  EXPECT_EQ(ctx->serialize_subgraph(nullptr, &buf, &size, &offset, options),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ctx->serialize_subgraph(subgraph, nullptr, &size, &offset, options),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ctx->serialize_subgraph(subgraph, &buf, nullptr, &offset, options),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ctx->serialize_subgraph(subgraph, &buf, &size, nullptr, options),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert
