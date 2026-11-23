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

#include "litert/c/internal/litert_compiler_context.h"

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/matchers.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

template <typename OptionsT>
void SetOptions(LiteRtOpT& op, tflite::BuiltinOptions type, OptionsT options) {
  TflOptions union_options;
  union_options.type = type;
  union_options.Set(std::move(options));
  SetTflOptions(op, std::move(union_options));
}

template <typename OptionsT>
void SetOptions2(LiteRtOpT& op, tflite::BuiltinOptions2 type,
                 OptionsT options) {
  TflOptions2 union_options;
  union_options.type = type;
  union_options.Set(std::move(options));
  SetTflOptions2(op, std::move(union_options));
}

std::vector<LiteRtParamIndex> ReadRefs(const LiteRtCompilerContext* ctx,
                                       LiteRtOpT& op) {
  LiteRtParamIndex num_refs = 0;
  EXPECT_EQ(ctx->get_num_op_subgraph_refs(&op, &num_refs), kLiteRtStatusOk);
  std::vector<LiteRtParamIndex> refs(num_refs);
  for (LiteRtParamIndex i = 0; i < num_refs; ++i) {
    EXPECT_EQ(ctx->get_op_subgraph_ref(&op, i, &refs[i]), kLiteRtStatusOk);
  }
  return refs;
}

TEST(CompilerContextTest, AbiVersionAndSubgraphRefGettersPresent) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(ctx->abi_header.major_version, 1);
  EXPECT_EQ(ctx->abi_header.minor_version, 6);
  EXPECT_EQ(ctx->abi_header.struct_size, sizeof(LiteRtCompilerContext));
  EXPECT_TRUE(
      LITERT_ABI_HAS_API(ctx, /*req_major=*/1, get_num_op_subgraph_refs));
  EXPECT_TRUE(LITERT_ABI_HAS_API(ctx, /*req_major=*/1, get_op_subgraph_ref));
}

TEST(CompilerContextTest, RepresentativeOpsReturnRefsInSlotOrder) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();

  // Fixed 2-slot op (TFL_WHILE: cond, body).
  LiteRtOpT while_op;
  while_op.SetOpCode(kLiteRtOpCodeTflWhile);
  tflite::WhileOptionsT while_opts;
  while_opts.cond_subgraph_index = 1;
  while_opts.body_subgraph_index = 2;
  SetOptions(while_op, tflite::BuiltinOptions_WhileOptions,
             std::move(while_opts));
  EXPECT_THAT(ReadRefs(ctx, while_op), ElementsAre(1, 2));

  // Variable-length op (STABLEHLO_CASE).
  LiteRtOpT case_op;
  case_op.SetOpCode(kLiteRtOpCodeShloCase);
  tflite::StablehloCaseOptionsT case_opts;
  case_opts.branch_subgraph_indices = {3, 1, 2};
  SetOptions2(case_op, tflite::BuiltinOptions2_StablehloCaseOptions,
              std::move(case_opts));
  EXPECT_THAT(ReadRefs(ctx, case_op), ElementsAre(3, 1, 2));

  // Nullable op (STABLEHLO_COMPOSITE): valid index vs -1 (absent).
  LiteRtOpT composite_op;
  composite_op.SetOpCode(kLiteRtOpCodeShloComposite);
  tflite::StableHLOCompositeOptionsT comp_opts;
  comp_opts.decomposition_subgraph_index = 4;
  SetOptions2(composite_op, tflite::BuiltinOptions2_StableHLOCompositeOptions,
              std::move(comp_opts));
  EXPECT_THAT(ReadRefs(ctx, composite_op), ElementsAre(4));

  tflite::StableHLOCompositeOptionsT empty_comp_opts;
  empty_comp_opts.decomposition_subgraph_index = -1;
  SetOptions2(composite_op, tflite::BuiltinOptions2_StableHLOCompositeOptions,
              std::move(empty_comp_opts));
  EXPECT_TRUE(ReadRefs(ctx, composite_op).empty());

  // Plain op has zero refs.
  LiteRtOpT add_op;
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  EXPECT_TRUE(ReadRefs(ctx, add_op).empty());
}

TEST(CompilerContextTest, ErrorCodesForOobNegativeAndNullArgs) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();

  // Out-of-range ref_index returns IndexOOB.
  LiteRtOpT add_op;
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  LiteRtParamIndex value = 999;
  EXPECT_EQ(ctx->get_op_subgraph_ref(&add_op, 0, &value),
            kLiteRtStatusErrorIndexOOB);

  // Malformed negative non-nullable ref preserves slot count and returns
  // InvalidGraphInvariant.
  LiteRtOpT call_once_op;
  call_once_op.SetOpCode(kLiteRtOpCodeTflCallOnce);
  tflite::CallOnceOptionsT call_once_opts;
  call_once_opts.init_subgraph_index = -1;
  SetOptions(call_once_op, tflite::BuiltinOptions_CallOnceOptions,
             std::move(call_once_opts));
  LiteRtParamIndex num_refs = 0;
  LITERT_ASSERT_OK(ctx->get_num_op_subgraph_refs(&call_once_op, &num_refs));
  EXPECT_EQ(num_refs, 1);
  EXPECT_EQ(ctx->get_op_subgraph_ref(&call_once_op, 0, &value),
            kLiteRtStatusErrorInvalidGraphInvariant);
  EXPECT_EQ(ctx->get_op_subgraph_ref(&call_once_op, 1, &value),
            kLiteRtStatusErrorIndexOOB);

  // Null pointer arguments return InvalidArgument.
  EXPECT_EQ(ctx->get_num_op_subgraph_refs(nullptr, &value),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ctx->get_num_op_subgraph_refs(&add_op, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ctx->get_op_subgraph_ref(nullptr, 0, &value),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(ctx->get_op_subgraph_ref(&add_op, 0, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
}  // namespace litert::internal
