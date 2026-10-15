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

#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_op_options.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"
#include "litert/vendors/qualcomm/compiler/transformations/attention_chunk_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/entry_embedding_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/gqa_attention_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/hadamard_transform_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/legalize_int32_ops_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/mask_simplification_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/mlp_quant_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/orphan_cleanup_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/rope_transformation.h"

namespace litert::compiler {
namespace {

using ::SetWeightsFromOwnedBuffer;
using ::litert::OwningBufferRef;
using ::litert::internal::AttachInput;
using ::litert::internal::AttachOutput;

//===----------------------------------------------------------------------===//
// LegalizeInt32SignTransformation Tests
//===----------------------------------------------------------------------===//

TEST(LegalizeInt32SignTest, PositiveMatchInt32) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in_tensor = subgraph.EmplaceTensor();
  in_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 10}));

  auto& out_tensor = subgraph.EmplaceTensor();
  out_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 10}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSign);
  AttachInput(&in_tensor, op);
  AttachOutput(&out_tensor, op);

  EXPECT_EQ(LegalizeInt32SignTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusOk);
}

TEST(LegalizeInt32SignTest, NegativeFloat32Rejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in_tensor = subgraph.EmplaceTensor();
  in_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 10}));

  auto& out_tensor = subgraph.EmplaceTensor();
  out_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 10}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSign);
  AttachInput(&in_tensor, op);
  AttachOutput(&out_tensor, op);

  EXPECT_EQ(LegalizeInt32SignTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

TEST(LegalizeInt32SignTest, NegativeWrongOpCodeRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in_tensor = subgraph.EmplaceTensor();
  in_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 10}));

  auto& out_tensor = subgraph.EmplaceTensor();
  out_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 10}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&in_tensor, op);
  AttachOutput(&out_tensor, op);

  EXPECT_EQ(LegalizeInt32SignTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

//===----------------------------------------------------------------------===//
// LegalizeInt32ReduceMaxTransformation Tests
//===----------------------------------------------------------------------===//

TEST(LegalizeInt32ReduceMaxTest, PositiveMatchInt32) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in_tensor = subgraph.EmplaceTensor();
  in_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 10, 20}));

  auto& axis_tensor = subgraph.EmplaceTensor();
  axis_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));

  auto& out_tensor = subgraph.EmplaceTensor();
  out_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 20}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflReduceMax);
  AttachInput(&in_tensor, op);
  AttachInput(&axis_tensor, op);
  AttachOutput(&out_tensor, op);

  EXPECT_EQ(LegalizeInt32ReduceMaxTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusOk);
}

TEST(LegalizeInt32ReduceMaxTest, NegativeFloat32Rejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in_tensor = subgraph.EmplaceTensor();
  in_tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 10, 20}));

  auto& axis_tensor = subgraph.EmplaceTensor();
  axis_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));

  auto& out_tensor = subgraph.EmplaceTensor();
  out_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 20}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflReduceMax);
  AttachInput(&in_tensor, op);
  AttachInput(&axis_tensor, op);
  AttachOutput(&out_tensor, op);

  EXPECT_EQ(LegalizeInt32ReduceMaxTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

TEST(LegalizeInt32ReduceMaxTest, NegativeInsufficientInputsRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in_tensor = subgraph.EmplaceTensor();
  in_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 10, 20}));

  auto& out_tensor = subgraph.EmplaceTensor();
  out_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 20}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflReduceMax);
  AttachInput(&in_tensor, op);
  AttachOutput(&out_tensor, op);

  EXPECT_EQ(LegalizeInt32ReduceMaxTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

//===----------------------------------------------------------------------===//
// MLPInt8QuantTransformation Tests
//===----------------------------------------------------------------------===//

TEST(MLPInt8QuantTest, PositiveMatchAndRewrite) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  // fc1_out (INT8) -> quant1 (INT16) -> gelu (INT16)
  auto& fc1_out = subgraph.EmplaceTensor();
  fc1_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));

  auto& quant1_out = subgraph.EmplaceTensor();
  quant1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant1_op = subgraph.EmplaceOp();
  quant1_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc1_out, quant1_op);
  AttachOutput(&quant1_out, quant1_op);

  auto& gelu_out = subgraph.EmplaceTensor();
  gelu_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& gelu_op = subgraph.EmplaceOp();
  gelu_op.SetOpCode(kLiteRtOpCodeTflGelu);
  AttachInput(&quant1_out, gelu_op);
  AttachOutput(&gelu_out, gelu_op);

  // fc2_out (INT8) -> quant2 (INT16)
  auto& fc2_out = subgraph.EmplaceTensor();
  fc2_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));

  auto& quant2_out = subgraph.EmplaceTensor();
  quant2_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant2_op = subgraph.EmplaceOp();
  quant2_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc2_out, quant2_op);
  AttachOutput(&quant2_out, quant2_op);

  // root mul: Mul(gelu_out, quant2_out) -> mul_out (INT16)
  auto& mul_out = subgraph.EmplaceTensor();
  mul_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& mul_op = subgraph.EmplaceOp();
  mul_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&gelu_out, mul_op);
  AttachInput(&quant2_out, mul_op);
  AttachOutput(&mul_out, mul_op);

  // quant3: Quantize(mul_out) -> final_out (INT8)
  auto& final_out = subgraph.EmplaceTensor();
  final_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant3_op = subgraph.EmplaceOp();
  quant3_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&mul_out, quant3_op);
  AttachOutput(&final_out, quant3_op);

  EXPECT_EQ(MLPInt8QuantTransformation(ctx, &builder_impl, &mul_op),
            kLiteRtStatusOk);
}

TEST(MLPInt8QuantTest, PositiveMatchSwappedCommutativeInputs) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  // fc1_out (INT8) -> quant1 (INT16) -> gelu (INT16)
  auto& fc1_out = subgraph.EmplaceTensor();
  fc1_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));

  auto& quant1_out = subgraph.EmplaceTensor();
  quant1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant1_op = subgraph.EmplaceOp();
  quant1_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc1_out, quant1_op);
  AttachOutput(&quant1_out, quant1_op);

  auto& gelu_out = subgraph.EmplaceTensor();
  gelu_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& gelu_op = subgraph.EmplaceOp();
  gelu_op.SetOpCode(kLiteRtOpCodeTflGelu);
  AttachInput(&quant1_out, gelu_op);
  AttachOutput(&gelu_out, gelu_op);

  // fc2_out (INT8) -> quant2 (INT16)
  auto& fc2_out = subgraph.EmplaceTensor();
  fc2_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));

  auto& quant2_out = subgraph.EmplaceTensor();
  quant2_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant2_op = subgraph.EmplaceOp();
  quant2_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc2_out, quant2_op);
  AttachOutput(&quant2_out, quant2_op);

  // root mul with inputs swapped: Mul(quant2_out, gelu_out) -> mul_out (INT16)
  auto& mul_out = subgraph.EmplaceTensor();
  mul_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& mul_op = subgraph.EmplaceOp();
  mul_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&quant2_out, mul_op);
  AttachInput(&gelu_out, mul_op);
  AttachOutput(&mul_out, mul_op);

  // quant3: Quantize(mul_out) -> final_out (INT8)
  auto& final_out = subgraph.EmplaceTensor();
  final_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant3_op = subgraph.EmplaceOp();
  quant3_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&mul_out, quant3_op);
  AttachOutput(&final_out, quant3_op);

  EXPECT_EQ(MLPInt8QuantTransformation(ctx, &builder_impl, &mul_op),
            kLiteRtStatusOk);
}

TEST(MLPInt8QuantTest, NegativeFloat32InputRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  // fc1_out is Float32 instead of INT8
  auto& fc1_out = subgraph.EmplaceTensor();
  fc1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 3072}));

  auto& quant1_out = subgraph.EmplaceTensor();
  quant1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant1_op = subgraph.EmplaceOp();
  quant1_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc1_out, quant1_op);
  AttachOutput(&quant1_out, quant1_op);

  auto& gelu_out = subgraph.EmplaceTensor();
  gelu_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& gelu_op = subgraph.EmplaceOp();
  gelu_op.SetOpCode(kLiteRtOpCodeTflGelu);
  AttachInput(&quant1_out, gelu_op);
  AttachOutput(&gelu_out, gelu_op);

  auto& fc2_out = subgraph.EmplaceTensor();
  fc2_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant2_out = subgraph.EmplaceTensor();
  quant2_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant2_op = subgraph.EmplaceOp();
  quant2_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc2_out, quant2_op);
  AttachOutput(&quant2_out, quant2_op);

  auto& mul_out = subgraph.EmplaceTensor();
  mul_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& mul_op = subgraph.EmplaceOp();
  mul_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&gelu_out, mul_op);
  AttachInput(&quant2_out, mul_op);
  AttachOutput(&mul_out, mul_op);

  auto& final_out = subgraph.EmplaceTensor();
  final_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant3_op = subgraph.EmplaceOp();
  quant3_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&mul_out, quant3_op);
  AttachOutput(&final_out, quant3_op);

  EXPECT_EQ(MLPInt8QuantTransformation(ctx, &builder_impl, &mul_op),
            kLiteRtStatusPatternNoMatch);
}

TEST(MLPInt8QuantTest, NegativeOutputNotQuantizeRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& fc1_out = subgraph.EmplaceTensor();
  fc1_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant1_out = subgraph.EmplaceTensor();
  quant1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant1_op = subgraph.EmplaceOp();
  quant1_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc1_out, quant1_op);
  AttachOutput(&quant1_out, quant1_op);

  auto& gelu_out = subgraph.EmplaceTensor();
  gelu_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& gelu_op = subgraph.EmplaceOp();
  gelu_op.SetOpCode(kLiteRtOpCodeTflGelu);
  AttachInput(&quant1_out, gelu_op);
  AttachOutput(&gelu_out, gelu_op);

  auto& fc2_out = subgraph.EmplaceTensor();
  fc2_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant2_out = subgraph.EmplaceTensor();
  quant2_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant2_op = subgraph.EmplaceOp();
  quant2_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc2_out, quant2_op);
  AttachOutput(&quant2_out, quant2_op);

  auto& mul_out = subgraph.EmplaceTensor();
  mul_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& mul_op = subgraph.EmplaceOp();
  mul_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&gelu_out, mul_op);
  AttachInput(&quant2_out, mul_op);
  AttachOutput(&mul_out, mul_op);

  // mul_out consumed by Add instead of Quantize
  auto& other_out = subgraph.EmplaceTensor();
  other_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& add_op = subgraph.EmplaceOp();
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&mul_out, add_op);
  AttachOutput(&other_out, add_op);

  EXPECT_EQ(MLPInt8QuantTransformation(ctx, &builder_impl, &mul_op),
            kLiteRtStatusPatternNoMatch);
}

TEST(MLPInt8QuantTest, NegativeMultipleUsesIntermediateRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& fc1_out = subgraph.EmplaceTensor();
  fc1_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant1_out = subgraph.EmplaceTensor();
  quant1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant1_op = subgraph.EmplaceOp();
  quant1_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc1_out, quant1_op);
  AttachOutput(&quant1_out, quant1_op);

  // Extra user of quant1_out
  auto& other_out = subgraph.EmplaceTensor();
  other_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& other_op = subgraph.EmplaceOp();
  other_op.SetOpCode(kLiteRtOpCodeTflRelu);
  AttachInput(&quant1_out, other_op);
  AttachOutput(&other_out, other_op);

  auto& gelu_out = subgraph.EmplaceTensor();
  gelu_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& gelu_op = subgraph.EmplaceOp();
  gelu_op.SetOpCode(kLiteRtOpCodeTflGelu);
  AttachInput(&quant1_out, gelu_op);
  AttachOutput(&gelu_out, gelu_op);

  auto& fc2_out = subgraph.EmplaceTensor();
  fc2_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant2_out = subgraph.EmplaceTensor();
  quant2_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant2_op = subgraph.EmplaceOp();
  quant2_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc2_out, quant2_op);
  AttachOutput(&quant2_out, quant2_op);

  auto& mul_out = subgraph.EmplaceTensor();
  mul_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& mul_op = subgraph.EmplaceOp();
  mul_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&gelu_out, mul_op);
  AttachInput(&quant2_out, mul_op);
  AttachOutput(&mul_out, mul_op);

  auto& final_out = subgraph.EmplaceTensor();
  final_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant3_op = subgraph.EmplaceOp();
  quant3_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&mul_out, quant3_op);
  AttachOutput(&final_out, quant3_op);

  EXPECT_EQ(MLPInt8QuantTransformation(ctx, &builder_impl, &mul_op),
            kLiteRtStatusPatternNoMatch);
}

TEST(MLPInt8QuantTest, NegativeMismatchedShapesRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  // fc1_out has shape [1, 630, 3072]
  auto& fc1_out = subgraph.EmplaceTensor();
  fc1_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant1_out = subgraph.EmplaceTensor();
  quant1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& quant1_op = subgraph.EmplaceOp();
  quant1_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc1_out, quant1_op);
  AttachOutput(&quant1_out, quant1_op);

  auto& gelu_out = subgraph.EmplaceTensor();
  gelu_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& gelu_op = subgraph.EmplaceOp();
  gelu_op.SetOpCode(kLiteRtOpCodeTflGelu);
  AttachInput(&quant1_out, gelu_op);
  AttachOutput(&gelu_out, gelu_op);

  // fc2_out has mismatched shape [1, 630, 1024]
  auto& fc2_out = subgraph.EmplaceTensor();
  fc2_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 1024}));
  auto& quant2_out = subgraph.EmplaceTensor();
  quant2_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 1024}));
  auto& quant2_op = subgraph.EmplaceOp();
  quant2_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc2_out, quant2_op);
  AttachOutput(&quant2_out, quant2_op);

  auto& mul_out = subgraph.EmplaceTensor();
  mul_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 630, 3072}));
  auto& mul_op = subgraph.EmplaceOp();
  mul_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&gelu_out, mul_op);
  AttachInput(&quant2_out, mul_op);
  AttachOutput(&mul_out, mul_op);

  auto& final_out = subgraph.EmplaceTensor();
  final_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 630, 3072}));
  auto& quant3_op = subgraph.EmplaceOp();
  quant3_op.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&mul_out, quant3_op);
  AttachOutput(&final_out, quant3_op);

  EXPECT_EQ(MLPInt8QuantTransformation(ctx, &builder_impl, &mul_op),
            kLiteRtStatusPatternNoMatch);
}

template <typename T>
void SetConstWeights(LiteRtTensorT& tensor, absl::Span<const T> values) {
  OwningBufferRef<uint8_t> buf(reinterpret_cast<const uint8_t*>(values.data()),
                               values.size() * sizeof(T));
  SetWeightsFromOwnedBuffer(tensor.Weights(), std::move(buf));
}

template <typename Opts>
LiteRtOpT& BuildOpWithOptions(const LiteRtCompilerContext* ctx,
                              LiteRtSubgraphT& subgraph, LiteRtOpCode code,
                              std::initializer_list<LiteRtTensor> ins,
                              std::initializer_list<LiteRtTensor> outs,
                              Opts opts) {
  auto& dest = subgraph.EmplaceOp();
  dest.SetOpCode(code);
  for (auto* t : ins) AttachInput(t, dest);
  for (auto* t : outs) AttachOutput(t, dest);

  LiteRtBuilderT setup_impl;
  Builder setup(ctx, &setup_impl);
  auto tmp_op = setup.BuildOp(code, {}, {});
  EXPECT_TRUE(tmp_op.HasValue());
  EXPECT_TRUE(setup.SetOpOptions(*tmp_op, std::move(opts)).HasValue());
  litert::internal::SetTflOptions(
      dest, litert::internal::GetTflOptions(*tmp_op->Get()));
  litert::internal::SetTflOptions2(
      dest, litert::internal::GetTflOptions2(*tmp_op->Get()));
  return dest;
}

//===----------------------------------------------------------------------===//
// EntryEmbeddingTransformation Tests
//===----------------------------------------------------------------------===//

TEST(EntryEmbeddingTest, PositiveMatchAndRewrite) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  // coord_x and coord_y [1, 630] (INT32)
  auto& coord_x = subgraph.EmplaceTensor();
  coord_x.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 630}));
  auto& coord_y = subgraph.EmplaceTensor();
  coord_y.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 630}));

  const int32_t kDepth[1] = {10240};
  const float kOn[1] = {1.0f};
  const float kOff[1] = {0.0f};

  // one_hot_x op: OneHot(coord_x, depth, on, off) -> one_hot_x [1, 630, 10240]
  auto& depth_x = subgraph.EmplaceTensor();
  depth_x.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));
  SetConstWeights<int32_t>(depth_x, kDepth);
  auto& on_x = subgraph.EmplaceTensor();
  on_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  SetConstWeights<float>(on_x, kOn);
  auto& off_x = subgraph.EmplaceTensor();
  off_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  SetConstWeights<float>(off_x, kOff);
  auto& one_hot_x = subgraph.EmplaceTensor();
  one_hot_x.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 10240}));
  OneHotOptions oh_opts;
  oh_opts.axis = -1;
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflOneHot,
                     {&coord_x, &depth_x, &on_x, &off_x}, {&one_hot_x},
                     oh_opts);

  // w_x [768, 10240] (FLOAT32)
  auto& w_x = subgraph.EmplaceTensor();
  w_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {768, 10240}));
  SetWeightsFromOwnedBuffer(
      w_x.Weights(), OwningBufferRef<uint8_t>(768 * 10240 * sizeof(float)));

  // fc_x op: FullyConnected(one_hot_x, w_x) -> fc_x_out [1, 630, 768]
  auto& fc_x_out = subgraph.EmplaceTensor();
  fc_x_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 768}));
  FullyConnectedOptions fc_opts;
  fc_opts.fused_activation_function = 0;
  fc_opts.weights_format = 0;
  fc_opts.keep_num_dims = false;
  fc_opts.quantized_bias_type = kLiteRtElementTypeNone;
  fc_opts.asymmetric_quantize_input = false;
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflFullyConnected,
                     {&one_hot_x, &w_x}, {&fc_x_out}, fc_opts);

  // reshape_x op: Reshape(fc_x_out, shape) -> reshape_x_out [1, 1, 630, 768]
  auto& shape_x = subgraph.EmplaceTensor();
  auto& reshape_x_out = subgraph.EmplaceTensor();
  reshape_x_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1, 630, 768}));
  auto& reshape_x_op = subgraph.EmplaceOp();
  reshape_x_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&fc_x_out, reshape_x_op);
  AttachInput(&shape_x, reshape_x_op);
  AttachOutput(&reshape_x_out, reshape_x_op);

  // one_hot_y op: OneHot(coord_y, depth, on, off) -> one_hot_y [1, 630, 10240]
  auto& depth_y = subgraph.EmplaceTensor();
  depth_y.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));
  SetConstWeights<int32_t>(depth_y, kDepth);
  auto& on_y = subgraph.EmplaceTensor();
  on_y.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  SetConstWeights<float>(on_y, kOn);
  auto& off_y = subgraph.EmplaceTensor();
  off_y.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  SetConstWeights<float>(off_y, kOff);
  auto& one_hot_y = subgraph.EmplaceTensor();
  one_hot_y.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 10240}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflOneHot,
                     {&coord_y, &depth_y, &on_y, &off_y}, {&one_hot_y},
                     oh_opts);

  // w_y [768, 10240] (FLOAT32)
  auto& w_y = subgraph.EmplaceTensor();
  w_y.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {768, 10240}));
  SetWeightsFromOwnedBuffer(
      w_y.Weights(), OwningBufferRef<uint8_t>(768 * 10240 * sizeof(float)));

  // fc_y op: FullyConnected(one_hot_y, w_y) -> fc_y_out [1, 630, 768]
  auto& fc_y_out = subgraph.EmplaceTensor();
  fc_y_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 768}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflFullyConnected,
                     {&one_hot_y, &w_y}, {&fc_y_out}, fc_opts);

  // reshape_y op: Reshape(fc_y_out, shape) -> reshape_y_out [1, 1, 630, 768]
  auto& shape_y = subgraph.EmplaceTensor();
  auto& reshape_y_out = subgraph.EmplaceTensor();
  reshape_y_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1, 630, 768}));
  auto& reshape_y_op = subgraph.EmplaceOp();
  reshape_y_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&fc_y_out, reshape_y_op);
  AttachInput(&shape_y, reshape_y_op);
  AttachOutput(&reshape_y_out, reshape_y_op);

  // concat op: Concatenation(reshape_x_out, reshape_y_out) -> concat_out
  // [2, 1, 630, 768]
  auto& concat_out = subgraph.EmplaceTensor();
  concat_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 1, 630, 768}));
  auto& concat_op = subgraph.EmplaceOp();
  concat_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&reshape_x_out, concat_op);
  AttachInput(&reshape_y_out, concat_op);
  AttachOutput(&concat_out, concat_op);

  // sum op: Sum(concat_out, axis) -> sum_out [1, 630, 768]
  auto& sum_axis = subgraph.EmplaceTensor();
  auto& sum_out = subgraph.EmplaceTensor();
  sum_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 768}));
  auto& sum_op = subgraph.EmplaceOp();
  sum_op.SetOpCode(kLiteRtOpCodeTflSum);
  AttachInput(&concat_out, sum_op);
  AttachInput(&sum_axis, sum_op);
  AttachOutput(&sum_out, sum_op);

  EXPECT_EQ(EntryEmbeddingTransformation(ctx, &builder_impl, &sum_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);
  EXPECT_EQ(OrphanCleanupTransformation(ctx, &builder_impl, &reshape_x_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);
  EXPECT_EQ(OrphanCleanupTransformation(ctx, &builder_impl, &reshape_y_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  int gather_count = 0;
  int add_count = 0;
  int one_hot_count = 0;
  for (const auto& op_item : subgraph.Ops()) {
    if (op_item->OpCode() == kLiteRtOpCodeTflGather) gather_count++;
    if (op_item->OpCode() == kLiteRtOpCodeTflAdd) add_count++;
    if (op_item->OpCode() == kLiteRtOpCodeTflOneHot) one_hot_count++;
  }
  EXPECT_EQ(gather_count, 2);
  EXPECT_GE(add_count, 1);
  EXPECT_EQ(one_hot_count, 0);
  ResetOrphanRegistry();
  ResetEntryEmbeddingTransformationState();
}

TEST(EntryEmbeddingTest, PositiveDynamicSeqLenWithSelectV2AndRewrite) {
  ResetOrphanRegistry();
  ResetEntryEmbeddingTransformationState();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  constexpr int32_t kDynamicSeqLen = 1260;
  const int32_t kDepth[1] = {10240};
  const float kOn[1] = {1.0f};
  const float kOff[1] = {0.0f};
  const float kFill[1] = {0.0f};

  OneHotOptions oh_opts;
  oh_opts.axis = -1;
  FullyConnectedOptions fc_opts;
  fc_opts.fused_activation_function = 0;
  fc_opts.weights_format = 0;
  fc_opts.keep_num_dims = false;
  fc_opts.quantized_bias_type = kLiteRtElementTypeNone;
  fc_opts.asymmetric_quantize_input = false;

  // coord_x and coord_y [1, 1260] (INT32)
  auto& coord_x = subgraph.EmplaceTensor();
  coord_x.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt32, {1, kDynamicSeqLen}));
  auto& coord_y = subgraph.EmplaceTensor();
  coord_y.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt32, {1, kDynamicSeqLen}));

  // one_hot_x -> select_x
  auto& depth_x = subgraph.EmplaceTensor();
  depth_x.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));
  SetConstWeights<int32_t>(depth_x, kDepth);
  auto& on_x = subgraph.EmplaceTensor();
  on_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  SetConstWeights<float>(on_x, kOn);
  auto& off_x = subgraph.EmplaceTensor();
  off_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  SetConstWeights<float>(off_x, kOff);
  auto& one_hot_x = subgraph.EmplaceTensor();
  one_hot_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                         {1, kDynamicSeqLen, 10240}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflOneHot,
                     {&coord_x, &depth_x, &on_x, &off_x}, {&one_hot_x},
                     oh_opts);

  auto& cond_x = subgraph.EmplaceTensor();
  cond_x.SetType(
      MakeRankedTensorType(kLiteRtElementTypeBool, {1, kDynamicSeqLen, 1}));
  auto& fill_x = subgraph.EmplaceTensor();
  fill_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  SetConstWeights<float>(fill_x, kFill);
  auto& select_x = subgraph.EmplaceTensor();
  select_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                        {1, kDynamicSeqLen, 10240}));
  auto& select_x_op = subgraph.EmplaceOp();
  select_x_op.SetOpCode(kLiteRtOpCodeTflSelectV2);
  AttachInput(&cond_x, select_x_op);
  AttachInput(&fill_x, select_x_op);
  AttachInput(&one_hot_x, select_x_op);
  AttachOutput(&select_x, select_x_op);

  auto& w_x = subgraph.EmplaceTensor();
  w_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {768, 10240}));
  SetWeightsFromOwnedBuffer(
      w_x.Weights(), OwningBufferRef<uint8_t>(768 * 10240 * sizeof(float)));

  auto& fc_x_out = subgraph.EmplaceTensor();
  fc_x_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                        {1, kDynamicSeqLen, 768}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflFullyConnected,
                     {&select_x, &w_x}, {&fc_x_out}, fc_opts);

  auto& shape_x = subgraph.EmplaceTensor();
  auto& reshape_x_out = subgraph.EmplaceTensor();
  reshape_x_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                             {1, 1, kDynamicSeqLen, 768}));
  auto& reshape_x_op = subgraph.EmplaceOp();
  reshape_x_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&fc_x_out, reshape_x_op);
  AttachInput(&shape_x, reshape_x_op);
  AttachOutput(&reshape_x_out, reshape_x_op);

  // one_hot_y -> select_y
  auto& depth_y = subgraph.EmplaceTensor();
  depth_y.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {}));
  SetConstWeights<int32_t>(depth_y, kDepth);
  auto& on_y = subgraph.EmplaceTensor();
  on_y.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  SetConstWeights<float>(on_y, kOn);
  auto& off_y = subgraph.EmplaceTensor();
  off_y.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  SetConstWeights<float>(off_y, kOff);
  auto& one_hot_y = subgraph.EmplaceTensor();
  one_hot_y.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                         {1, kDynamicSeqLen, 10240}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflOneHot,
                     {&coord_y, &depth_y, &on_y, &off_y}, {&one_hot_y},
                     oh_opts);

  auto& cond_y = subgraph.EmplaceTensor();
  cond_y.SetType(
      MakeRankedTensorType(kLiteRtElementTypeBool, {1, kDynamicSeqLen, 1}));
  auto& fill_y = subgraph.EmplaceTensor();
  fill_y.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  SetConstWeights<float>(fill_y, kFill);
  auto& select_y = subgraph.EmplaceTensor();
  select_y.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                        {1, kDynamicSeqLen, 10240}));
  auto& select_y_op = subgraph.EmplaceOp();
  select_y_op.SetOpCode(kLiteRtOpCodeTflSelectV2);
  AttachInput(&cond_y, select_y_op);
  AttachInput(&fill_y, select_y_op);
  AttachInput(&one_hot_y, select_y_op);
  AttachOutput(&select_y, select_y_op);

  auto& w_y = subgraph.EmplaceTensor();
  w_y.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {768, 10240}));
  SetWeightsFromOwnedBuffer(
      w_y.Weights(), OwningBufferRef<uint8_t>(768 * 10240 * sizeof(float)));

  auto& fc_y_out = subgraph.EmplaceTensor();
  fc_y_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                        {1, kDynamicSeqLen, 768}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflFullyConnected,
                     {&select_y, &w_y}, {&fc_y_out}, fc_opts);

  auto& shape_y = subgraph.EmplaceTensor();
  auto& reshape_y_out = subgraph.EmplaceTensor();
  reshape_y_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                             {1, 1, kDynamicSeqLen, 768}));
  auto& reshape_y_op = subgraph.EmplaceOp();
  reshape_y_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&fc_y_out, reshape_y_op);
  AttachInput(&shape_y, reshape_y_op);
  AttachOutput(&reshape_y_out, reshape_y_op);

  auto& concat_out = subgraph.EmplaceTensor();
  concat_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                          {2, 1, kDynamicSeqLen, 768}));
  auto& concat_op = subgraph.EmplaceOp();
  concat_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&reshape_x_out, concat_op);
  AttachInput(&reshape_y_out, concat_op);
  AttachOutput(&concat_out, concat_op);

  auto& sum_axis = subgraph.EmplaceTensor();
  auto& sum_out = subgraph.EmplaceTensor();
  sum_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                       {1, kDynamicSeqLen, 768}));
  auto& sum_op = subgraph.EmplaceOp();
  sum_op.SetOpCode(kLiteRtOpCodeTflSum);
  AttachInput(&concat_out, sum_op);
  AttachInput(&sum_axis, sum_op);
  AttachOutput(&sum_out, sum_op);

  EXPECT_EQ(EntryEmbeddingTransformation(ctx, &builder_impl, &sum_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);
  EXPECT_EQ(OrphanCleanupTransformation(ctx, &builder_impl, &reshape_x_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);
  EXPECT_EQ(OrphanCleanupTransformation(ctx, &builder_impl, &reshape_y_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  int gather_count = 0;
  int one_hot_count = 0;
  for (const auto& op_item : subgraph.Ops()) {
    if (op_item->OpCode() == kLiteRtOpCodeTflGather) gather_count++;
    if (op_item->OpCode() == kLiteRtOpCodeTflOneHot) one_hot_count++;
  }
  EXPECT_EQ(gather_count, 2);
  EXPECT_EQ(one_hot_count, 0);
  ResetOrphanRegistry();
  ResetEntryEmbeddingTransformationState();
}

TEST(EntryEmbeddingTest, NegativeNonSumOpRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in = subgraph.EmplaceTensor();
  in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 768}));
  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 768}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&in, op);
  AttachOutput(&out, op);

  EXPECT_EQ(EntryEmbeddingTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

TEST(EntryEmbeddingTest, NegativeWrongOutputShapeRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in = subgraph.EmplaceTensor();
  in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 500, 512}));
  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 500, 512}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSum);
  AttachInput(&in, op);
  AttachOutput(&out, op);

  EXPECT_EQ(EntryEmbeddingTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

TEST(EntryEmbeddingTest, NegativeWrongWeightShapeRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  auto& coord_x = subgraph.EmplaceTensor();
  coord_x.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 630}));
  auto& one_hot_x = subgraph.EmplaceTensor();
  one_hot_x.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 10240}));

  // Wrong weight shape: {512, 10240} instead of {768, 10240}
  auto& w_x = subgraph.EmplaceTensor();
  w_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {512, 10240}));
  SetWeightsFromOwnedBuffer(
      w_x.Weights(), OwningBufferRef<uint8_t>(512 * 10240 * sizeof(float)));

  auto& fc_x_out = subgraph.EmplaceTensor();
  fc_x_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 512}));
  auto& fc_x_op = subgraph.EmplaceOp();
  fc_x_op.SetOpCode(kLiteRtOpCodeTflFullyConnected);
  AttachInput(&one_hot_x, fc_x_op);
  AttachInput(&w_x, fc_x_op);
  AttachOutput(&fc_x_out, fc_x_op);

  auto& shape_x = subgraph.EmplaceTensor();
  auto& reshape_x_out = subgraph.EmplaceTensor();
  reshape_x_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1, 630, 512}));
  auto& reshape_x_op = subgraph.EmplaceOp();
  reshape_x_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&fc_x_out, reshape_x_op);
  AttachInput(&shape_x, reshape_x_op);
  AttachOutput(&reshape_x_out, reshape_x_op);

  auto& dummy_y = subgraph.EmplaceTensor();
  dummy_y.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1, 630, 512}));

  auto& concat_out = subgraph.EmplaceTensor();
  concat_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {2, 1, 630, 768}));
  auto& concat_op = subgraph.EmplaceOp();
  concat_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&reshape_x_out, concat_op);
  AttachInput(&dummy_y, concat_op);
  AttachOutput(&concat_out, concat_op);

  auto& sum_axis = subgraph.EmplaceTensor();
  auto& sum_out = subgraph.EmplaceTensor();
  sum_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 768}));
  auto& sum_op = subgraph.EmplaceOp();
  sum_op.SetOpCode(kLiteRtOpCodeTflSum);
  AttachInput(&concat_out, sum_op);
  AttachInput(&sum_axis, sum_op);
  AttachOutput(&sum_out, sum_op);

  EXPECT_EQ(EntryEmbeddingTransformation(ctx, &builder_impl, &sum_op),
            kLiteRtStatusPatternNoMatch);
}

//===----------------------------------------------------------------------===//
// RopeTransformation Tests
//===----------------------------------------------------------------------===//

TEST(RopeTransformationTest, PositiveMatchAndRewrite) {
  ResetRopeTransformationState();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  // in_tensor [1, 630, 12, 64] (FLOAT32)
  auto& in_tensor = subgraph.EmplaceTensor();
  in_tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 64}));

  auto make_slice = [&](LiteRtTensorT& src, int32_t offset,
                        int32_t len) -> LiteRtTensorT& {
    auto& begin = subgraph.EmplaceTensor();
    begin.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
    const int32_t begin_vals[4] = {0, 0, 0, offset};
    SetConstWeights<int32_t>(begin, begin_vals);
    auto& size = subgraph.EmplaceTensor();
    size.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
    const int32_t size_vals[4] = {1, 630, 12, len};
    SetConstWeights<int32_t>(size, size_vals);
    auto& out = subgraph.EmplaceTensor();
    out.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, len}));
    auto& op = subgraph.EmplaceOp();
    op.SetOpCode(kLiteRtOpCodeTflSlice);
    AttachInput(&src, op);
    AttachInput(&begin, op);
    AttachInput(&size, op);
    AttachOutput(&out, op);
    return out;
  };

  auto& half_lo = make_slice(in_tensor, 0, 32);
  auto& half_hi = make_slice(in_tensor, 32, 32);
  auto& s0 = make_slice(half_lo, 0, 16);
  auto& s1 = make_slice(half_lo, 16, 16);
  auto& s2 = make_slice(half_hi, 0, 16);
  auto& s3 = make_slice(half_hi, 16, 16);

  // cos2, sin2, cos3, sin3 [1, 630, 1, 16]
  auto& cos2 = subgraph.EmplaceTensor();
  cos2.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 1, 16}));
  auto& sin2 = subgraph.EmplaceTensor();
  sin2.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 1, 16}));
  auto& cos3 = subgraph.EmplaceTensor();
  cos3.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 1, 16}));
  auto& sin3 = subgraph.EmplaceTensor();
  sin3.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 1, 16}));

  auto make_mul = [&](LiteRtTensorT& a, LiteRtTensorT& b) -> LiteRtTensorT& {
    auto& out = subgraph.EmplaceTensor();
    out.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
    MulOptions mul_opts;
    mul_opts.fused_activation_function = 0;
    BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflMul, {&a, &b}, {&out},
                       mul_opts);
    return out;
  };

  ConcatenationOptions concat_opts;
  concat_opts.axis = 3;
  concat_opts.fused_activation_function = 0;
  SubOptions sub_opts;
  sub_opts.fused_activation_function = 0;
  AddOptions add_opts;
  add_opts.fused_activation_function = 0;

  // Branch 0: Concat(Sub(s0*cos2, s1*sin2), Add(s1*cos2, s0*sin2))
  auto& mul0_sub = make_mul(s0, cos2);
  auto& mul1_sub = make_mul(s1, sin2);
  auto& sub0 = subgraph.EmplaceTensor();
  sub0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflSub, {&mul0_sub, &mul1_sub},
                     {&sub0}, sub_opts);

  auto& mul0_add = make_mul(s1, cos2);
  auto& mul1_add = make_mul(s0, sin2);
  auto& add0 = subgraph.EmplaceTensor();
  add0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflAdd, {&mul0_add, &mul1_add},
                     {&add0}, add_opts);

  auto& concat0 = subgraph.EmplaceTensor();
  concat0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflConcatenation,
                     {&sub0, &add0}, {&concat0}, concat_opts);

  // Branch 1: Concat(Sub(s2*cos3, s3*sin3), Add(s3*cos3, s2*sin3))
  auto& mul2_sub = make_mul(s2, cos3);
  auto& mul3_sub = make_mul(s3, sin3);
  auto& sub1 = subgraph.EmplaceTensor();
  sub1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflSub, {&mul2_sub, &mul3_sub},
                     {&sub1}, sub_opts);

  auto& mul2_add = make_mul(s3, cos3);
  auto& mul3_add = make_mul(s2, sin3);
  auto& add1 = subgraph.EmplaceTensor();
  add1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflAdd, {&mul2_add, &mul3_add},
                     {&add1}, add_opts);

  auto& concat1 = subgraph.EmplaceTensor();
  concat1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflConcatenation,
                     {&sub1, &add1}, {&concat1}, concat_opts);

  // Root concat: Concat(concat0, concat1) -> root_out [1, 630, 12, 64]
  auto& root_out = subgraph.EmplaceTensor();
  root_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 64}));
  auto& root_op =
      BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflConcatenation,
                         {&concat0, &concat1}, {&root_out}, concat_opts);

  EXPECT_EQ(RopeAttentionLayerTransformation(ctx, &builder_impl, &root_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);
  ResetRopeTransformationState();
  ResetOrphanRegistry();
}

TEST(RopeTransformationTest, NegativeNonConcatOpRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in = subgraph.EmplaceTensor();
  in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 64}));
  auto& out = subgraph.EmplaceTensor();
  out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 64}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&in, op);
  AttachOutput(&out, op);

  EXPECT_EQ(RopeAttentionLayerTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

TEST(RopeTransformationTest, NegativeWrongOutputShapeRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in1 = subgraph.EmplaceTensor();
  in1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));
  auto& in2 = subgraph.EmplaceTensor();
  in2.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));
  auto& out = subgraph.EmplaceTensor();
  out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32,
                           {1, 630, 12, 32}));  // Wrong shape (expected 64)

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&in1, op);
  AttachInput(&in2, op);
  AttachOutput(&out, op);

  EXPECT_EQ(RopeAttentionLayerTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

TEST(RopeTransformationTest, NegativeWrongInnerOpRejected) {
  ResetRopeTransformationState();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in_tensor = subgraph.EmplaceTensor();
  in_tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 64}));
  auto& begin0 = subgraph.EmplaceTensor();
  auto& size0 = subgraph.EmplaceTensor();
  auto& slice_outer_out = subgraph.EmplaceTensor();
  slice_outer_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));
  auto& slice_outer_op = subgraph.EmplaceOp();
  slice_outer_op.SetOpCode(kLiteRtOpCodeTflSlice);
  AttachInput(&in_tensor, slice_outer_op);
  AttachInput(&begin0, slice_outer_op);
  AttachInput(&size0, slice_outer_op);
  AttachOutput(&slice_outer_out, slice_outer_op);

  auto& begin1 = subgraph.EmplaceTensor();
  auto& size1 = subgraph.EmplaceTensor();
  auto& s0 = subgraph.EmplaceTensor();
  s0.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& slice0_op = subgraph.EmplaceOp();
  slice0_op.SetOpCode(kLiteRtOpCodeTflSlice);
  AttachInput(&slice_outer_out, slice0_op);
  AttachInput(&begin1, slice0_op);
  AttachInput(&size1, slice0_op);
  AttachOutput(&s0, slice0_op);

  auto& s1 = subgraph.EmplaceTensor();
  s1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& cos2 = subgraph.EmplaceTensor();
  cos2.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 1, 16}));
  auto& sin2 = subgraph.EmplaceTensor();
  sin2.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 1, 16}));

  auto& mul0 = subgraph.EmplaceTensor();
  mul0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& mul0_op = subgraph.EmplaceOp();
  mul0_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&s0, mul0_op);
  AttachInput(&cos2, mul0_op);
  AttachOutput(&mul0, mul0_op);

  auto& mul1 = subgraph.EmplaceTensor();
  mul1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& mul1_op = subgraph.EmplaceOp();
  mul1_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&s1, mul1_op);
  AttachInput(&sin2, mul1_op);
  AttachOutput(&mul1, mul1_op);

  // Wrong op: Mul instead of Sub!
  auto& wrong_sub = subgraph.EmplaceTensor();
  wrong_sub.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& wrong_op = subgraph.EmplaceOp();
  wrong_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&mul0, wrong_op);
  AttachInput(&mul1, wrong_op);
  AttachOutput(&wrong_sub, wrong_op);

  auto& add0 = subgraph.EmplaceTensor();
  add0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& add0_op = subgraph.EmplaceOp();
  add0_op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&mul0, add0_op);
  AttachInput(&mul1, add0_op);
  AttachOutput(&add0, add0_op);

  auto& concat0 = subgraph.EmplaceTensor();
  concat0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));
  auto& concat0_op = subgraph.EmplaceOp();
  concat0_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&wrong_sub, concat0_op);
  AttachInput(&add0, concat0_op);
  AttachOutput(&concat0, concat0_op);

  auto& dummy_concat1 = subgraph.EmplaceTensor();
  dummy_concat1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));

  auto& root_out = subgraph.EmplaceTensor();
  root_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 64}));
  auto& root_op = subgraph.EmplaceOp();
  root_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&concat0, root_op);
  AttachInput(&dummy_concat1, root_op);
  AttachOutput(&root_out, root_op);

  EXPECT_EQ(RopeAttentionLayerTransformation(ctx, &builder_impl, &root_op),
            kLiteRtStatusPatternNoMatch);
  ResetRopeTransformationState();
}

//===----------------------------------------------------------------------===//
// RopeCleanupTransformation Tests
//===----------------------------------------------------------------------===//

TEST(RopeCleanupTest, PositiveCleanupDeadConcat) {
  ResetOrphanRegistry();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in_a = subgraph.EmplaceTensor();
  in_a.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& in_b = subgraph.EmplaceTensor();
  in_b.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& sub_out = subgraph.EmplaceTensor();
  sub_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& sub_op = subgraph.EmplaceOp();
  sub_op.SetOpCode(kLiteRtOpCodeTflSub);
  AttachInput(&in_a, sub_op);
  AttachInput(&in_b, sub_op);
  AttachOutput(&sub_out, sub_op);

  auto& in_c = subgraph.EmplaceTensor();
  in_c.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& in_d = subgraph.EmplaceTensor();
  in_d.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& add_out = subgraph.EmplaceTensor();
  add_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& add_op = subgraph.EmplaceOp();
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&in_c, add_op);
  AttachInput(&in_d, add_op);
  AttachOutput(&add_out, add_op);

  auto& dead_concat_out = subgraph.EmplaceTensor();
  dead_concat_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));
  auto& concat_op = subgraph.EmplaceOp();
  concat_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&sub_out, concat_op);
  AttachInput(&add_out, concat_op);
  AttachOutput(&dead_concat_out, concat_op);

  RegisterOrphanGroup({&concat_op, &sub_op, &add_op});
  EXPECT_EQ(RopeCleanupTransformation(ctx, &builder_impl, &concat_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 0);
  ResetOrphanRegistry();
}

TEST(RopeCleanupTest, NegativeActiveOutputRejected) {
  ResetOrphanRegistry();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  auto& in1 = subgraph.EmplaceTensor();
  in1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& in2 = subgraph.EmplaceTensor();
  in2.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& out = subgraph.EmplaceTensor();
  out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));

  auto& concat_op = subgraph.EmplaceOp();
  concat_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&in1, concat_op);
  AttachInput(&in2, concat_op);
  AttachOutput(&out, concat_op);

  // out is used by downstream add_op (so it is not dead)
  auto& final_out = subgraph.EmplaceTensor();
  final_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));
  auto& add_op = subgraph.EmplaceOp();
  add_op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&out, add_op);
  AttachOutput(&final_out, add_op);

  RegisterOrphanGroup({&concat_op});
  EXPECT_EQ(RopeCleanupTransformation(ctx, &builder_impl, &concat_op),
            kLiteRtStatusPatternNoMatch);
  ResetOrphanRegistry();
}

//===----------------------------------------------------------------------===//
// AttentionChunkTransformation Tests (Qualcomm MHA -> SHA)
//===----------------------------------------------------------------------===//

TEST(AttentionChunkTest, PositiveQualcommMhaToShaRewrite) {
  ResetOrphanRegistry();
  SetAttentionChunkConfig(/*head_chunks=*/2, /*query_chunks=*/1);
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  const int32_t kH = 2, kS = 4, kK = 4, kD = 8, kDv = 8;

  auto make_perm = [&](std::initializer_list<int32_t> vals) -> LiteRtTensorT& {
    auto& t = subgraph.EmplaceTensor();
    t.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
    std::vector<int32_t> v(vals);
    SetConstWeights<int32_t>(t, v);
    return t;
  };

  auto& perm_0213 = make_perm({0, 2, 1, 3});
  auto& perm_0231 = make_perm({0, 2, 3, 1});

  auto make_prologue =
      [&](LiteRtTensorT& perm, std::initializer_list<int32_t> t_dims,
          std::initializer_list<int32_t> r_dims) -> LiteRtTensorT& {
    auto& in_4d = subgraph.EmplaceTensor();
    in_4d.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, kS, kH, kD}));
    auto& t_out = subgraph.EmplaceTensor();
    t_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, t_dims));
    auto& t_op = subgraph.EmplaceOp();
    t_op.SetOpCode(kLiteRtOpCodeTflTranspose);
    AttachInput(&in_4d, t_op);
    AttachInput(&perm, t_op);
    AttachOutput(&t_out, t_op);

    auto& shape = subgraph.EmplaceTensor();
    auto& r_out = subgraph.EmplaceTensor();
    r_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, r_dims));
    auto& r_op = subgraph.EmplaceOp();
    r_op.SetOpCode(kLiteRtOpCodeTflReshape);
    AttachInput(&t_out, r_op);
    AttachInput(&shape, r_op);
    AttachOutput(&r_out, r_op);
    return r_out;
  };

  auto& q = make_prologue(perm_0213, {1, kH, kS, kD}, {kH, kS, kD});
  auto& kt = make_prologue(perm_0231, {1, kH, kD, kK}, {kH, kD, kK});
  auto& v = make_prologue(perm_0213, {1, kH, kK, kDv}, {kH, kK, kDv});

  BatchMatmulOptions bmm_opts;
  bmm_opts.adj_x = false;
  bmm_opts.adj_y = false;
  bmm_opts.asymmetric_quantize_input = false;

  auto& logits = subgraph.EmplaceTensor();
  logits.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {kH, kS, kK}));
  auto& bmm1 = BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflBatchMatmul,
                                  {&q, &kt}, {&logits}, bmm_opts);

  auto& r1_shape = subgraph.EmplaceTensor();
  auto& r1_out = subgraph.EmplaceTensor();
  r1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, kH, kS, kK}));
  auto& r1_op = subgraph.EmplaceOp();
  r1_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&logits, r1_op);
  AttachInput(&r1_shape, r1_op);
  AttachOutput(&r1_out, r1_op);

  auto& sm_out = subgraph.EmplaceTensor();
  sm_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, kH, kS, kK}));
  SoftmaxOptions sm_opts;
  sm_opts.beta = 1.0f;
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflSoftmax, {&r1_out},
                     {&sm_out}, sm_opts);

  auto& r2_shape = subgraph.EmplaceTensor();
  auto& r2_out = subgraph.EmplaceTensor();
  r2_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {kH, kS, kK}));
  auto& r2_op = subgraph.EmplaceOp();
  r2_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&sm_out, r2_op);
  AttachInput(&r2_shape, r2_op);
  AttachOutput(&r2_out, r2_op);

  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {kH, kS, kDv}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflBatchMatmul, {&r2_out, &v},
                     {&out}, bmm_opts);

  EXPECT_EQ(AttentionChunkTransformation(ctx, &builder_impl, &bmm1),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  int split_count = 0;
  int softmax_3d_count = 0;
  for (const auto& op_item : subgraph.Ops()) {
    if (op_item->OpCode() == kLiteRtOpCodeTflSplit) split_count++;
    if (op_item->OpCode() == kLiteRtOpCodeTflSoftmax) {
      const auto& d =
          op_item->Output(0).Type().second.ranked_tensor_type.layout;
      if (d.rank == 3) softmax_3d_count++;
    }
  }
  EXPECT_EQ(split_count, 3);
  EXPECT_EQ(softmax_3d_count, kH);

  ResetAttentionChunkTransformationState();
  ResetOrphanRegistry();
}

TEST(AttentionChunkTest, PositiveQualcomm4DMhaToShaRewriteForLongSequence) {
  ResetOrphanRegistry();
  SetAttentionChunkConfig(/*head_chunks=*/2, /*query_chunks=*/1);
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  const int32_t kH = 2, kS = 1260, kK = 1260, kD = 8, kDv = 8;

  auto make_perm = [&](std::initializer_list<int32_t> vals) -> LiteRtTensorT& {
    auto& t = subgraph.EmplaceTensor();
    t.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
    std::vector<int32_t> v(vals);
    SetConstWeights<int32_t>(t, v);
    return t;
  };

  auto& perm_0213 = make_perm({0, 2, 1, 3});
  auto& perm_0231 = make_perm({0, 2, 3, 1});

  auto make_prologue =
      [&](LiteRtTensorT& perm, std::initializer_list<int32_t> t_dims,
          std::initializer_list<int32_t> r_dims) -> LiteRtTensorT& {
    auto& in_4d = subgraph.EmplaceTensor();
    in_4d.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, kS, kH, kD}));
    auto& t_out = subgraph.EmplaceTensor();
    t_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, t_dims));
    auto& t_op = subgraph.EmplaceOp();
    t_op.SetOpCode(kLiteRtOpCodeTflTranspose);
    AttachInput(&in_4d, t_op);
    AttachInput(&perm, t_op);
    AttachOutput(&t_out, t_op);

    auto& shape = subgraph.EmplaceTensor();
    auto& r_out = subgraph.EmplaceTensor();
    r_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, r_dims));
    auto& r_op = subgraph.EmplaceOp();
    r_op.SetOpCode(kLiteRtOpCodeTflReshape);
    AttachInput(&t_out, r_op);
    AttachInput(&shape, r_op);
    AttachOutput(&r_out, r_op);
    return r_out;
  };

  auto& q = make_prologue(perm_0213, {1, kH, kS, kD}, {kH, kS, kD});
  auto& kt = make_prologue(perm_0231, {1, kH, kD, kK}, {kH, kD, kK});
  auto& v = make_prologue(perm_0213, {1, kH, kK, kDv}, {kH, kK, kDv});

  BatchMatmulOptions bmm_opts;
  bmm_opts.adj_x = false;
  bmm_opts.adj_y = false;
  bmm_opts.asymmetric_quantize_input = false;

  auto& logits = subgraph.EmplaceTensor();
  logits.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {kH, kS, kK}));
  auto& bmm1 = BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflBatchMatmul,
                                  {&q, &kt}, {&logits}, bmm_opts);

  auto& r1_shape = subgraph.EmplaceTensor();
  auto& r1_out = subgraph.EmplaceTensor();
  r1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, kH, kS, kK}));
  auto& r1_op = subgraph.EmplaceOp();
  r1_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&logits, r1_op);
  AttachInput(&r1_shape, r1_op);
  AttachOutput(&r1_out, r1_op);

  auto& sm_out = subgraph.EmplaceTensor();
  sm_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, kH, kS, kK}));
  SoftmaxOptions sm_opts;
  sm_opts.beta = 1.0f;
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflSoftmax, {&r1_out},
                     {&sm_out}, sm_opts);

  auto& r2_shape = subgraph.EmplaceTensor();
  auto& r2_out = subgraph.EmplaceTensor();
  r2_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {kH, kS, kK}));
  auto& r2_op = subgraph.EmplaceOp();
  r2_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&sm_out, r2_op);
  AttachInput(&r2_shape, r2_op);
  AttachOutput(&r2_out, r2_op);

  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {kH, kS, kDv}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflBatchMatmul, {&r2_out, &v},
                     {&out}, bmm_opts);

  EXPECT_EQ(AttentionChunkTransformation(ctx, &builder_impl, &bmm1),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  int split_count = 0;
  int softmax_4d_count = 0;
  for (const auto& op_item : subgraph.Ops()) {
    if (op_item->OpCode() == kLiteRtOpCodeTflSplit) split_count++;
    if (op_item->OpCode() == kLiteRtOpCodeTflSoftmax) {
      const auto& d =
          op_item->Output(0).Type().second.ranked_tensor_type.layout;
      if (d.rank == 4 && d.dimensions[1] == 15 && d.dimensions[2] == 84) {
        softmax_4d_count++;
      }
    }
  }
  EXPECT_EQ(split_count, 3);
  EXPECT_EQ(softmax_4d_count, kH);

  ResetAttentionChunkTransformationState();
  ResetOrphanRegistry();
}

TEST(MLPInt8QuantTest, PositiveResidualNorm4DFolding) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  const int32_t kS = 1260, kD = 768;

  auto& residual_in = subgraph.EmplaceTensor();
  residual_in.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, kS, kD}));

  auto& fc_out = subgraph.EmplaceTensor();
  fc_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {1, kS, kD}));

  auto& quant_post_out = subgraph.EmplaceTensor();
  quant_post_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, kS, kD}));
  auto& quant_post = subgraph.EmplaceOp();
  quant_post.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&fc_out, quant_post);
  AttachOutput(&quant_post_out, quant_post);

  auto& gamma_post = subgraph.EmplaceTensor();
  gamma_post.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {kD}));
  auto& norm_post_out = subgraph.EmplaceTensor();
  norm_post_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, kS, kD}));
  RmsNormOpts rms_opts;
  rms_opts.name = CompositeOptions::kRmsNorm;
  rms_opts.subgraph = 1;
  rms_opts.version = 1;
  rms_opts.epsilon = 1e-6f;
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeShloComposite,
                     {&quant_post_out, &gamma_post}, {&norm_post_out},
                     rms_opts);

  auto& add_out = subgraph.EmplaceTensor();
  add_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, kS, kD}));
  AddOptions add_opts;
  add_opts.fused_activation_function = 0;
  auto& add_op =
      BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflAdd,
                         {&residual_in, &norm_post_out}, {&add_out}, add_opts);

  auto& gamma_pre = subgraph.EmplaceTensor();
  gamma_pre.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {kD}));
  auto& norm_pre_out = subgraph.EmplaceTensor();
  norm_pre_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, kS, kD}));
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeShloComposite,
                     {&add_out, &gamma_pre}, {&norm_pre_out}, rms_opts);

  auto& quant_pre_out = subgraph.EmplaceTensor();
  quant_pre_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt8, {1, kS, kD}));
  auto& quant_pre = subgraph.EmplaceOp();
  quant_pre.SetOpCode(kLiteRtOpCodeTflQuantize);
  AttachInput(&norm_pre_out, quant_pre);
  AttachOutput(&quant_pre_out, quant_pre);

  EXPECT_EQ(MLPInt8QuantTransformation(ctx, &builder_impl, &add_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  int rms_4d_count = 0;
  int add_4d_count = 0;
  for (const auto& op_item : subgraph.Ops()) {
    if (op_item->OpCode() == kLiteRtOpCodeShloComposite) {
      const auto& d =
          op_item->Output(0).Type().second.ranked_tensor_type.layout;
      if (d.rank == 4 && d.dimensions[1] == 15 && d.dimensions[2] == 84) {
        rms_4d_count++;
      }
    }
    if (op_item->OpCode() == kLiteRtOpCodeTflAdd) {
      const auto& d =
          op_item->Output(0).Type().second.ranked_tensor_type.layout;
      if (d.rank == 4 && d.dimensions[1] == 15 && d.dimensions[2] == 84) {
        add_4d_count++;
      }
    }
  }
  EXPECT_EQ(rms_4d_count, 2);
  EXPECT_EQ(add_4d_count, 1);
}

//===----------------------------------------------------------------------===//
// HadamardTransform & ParallelizeHadamard Tests
//===----------------------------------------------------------------------===//

void SetSymQuant(LiteRtTensorT& t, float scale) {
  t.SetQarams(MakePerTensorQuantization(scale, 0));
}

TEST(HadamardTransformTest, PositiveMatchInt8SylvesterHadamard) {
  ResetOrphanRegistry();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  const int32_t kN = 4;
  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, kN}));
  SetSymQuant(input, 0.01f);

  auto& weights = subgraph.EmplaceTensor();
  weights.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {kN, kN}));
  SetSymQuant(weights, 1.0f / (127.0f * 2.0f));
  // 4x4 Sylvester Hadamard matrix with positive entry 127, negative -127
  std::vector<int8_t> h4 = {
      127, 127,  127,  127,   //
      127, -127, 127,  -127,  //
      127, 127,  -127, -127,  //
      127, -127, -127, 127,
  };
  SetConstWeights<int8_t>(weights, h4);

  auto& bias = subgraph.EmplaceTensor();
  bias.SetType(MakeRankedTensorType(kLiteRtElementTypeNone, {}));

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, kN}));
  SetSymQuant(output, 0.01f);

  FullyConnectedOptions fc_opts;
  fc_opts.fused_activation_function = 0;
  fc_opts.weights_format = 0;
  fc_opts.keep_num_dims = true;
  fc_opts.quantized_bias_type = kLiteRtElementTypeNone;
  fc_opts.asymmetric_quantize_input = false;
  auto& fc_op =
      BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflFullyConnected,
                         {&input, &weights, &bias}, {&output}, fc_opts);

  EXPECT_EQ(ConvertFcToHadamardTransform(ctx, &builder_impl, &fc_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops()[0]->OpCode(), kLiteRtOpCodeShloComposite);
  auto info = GetOptionsAs<CompositeOptions>(ctx, subgraph.Ops()[0]);
  ASSERT_TRUE(info.HasValue());
  EXPECT_EQ(info->name, "HadamardTransform");
  EXPECT_NEAR((*info->attributes_map)["scale"].AsFloat(), 1.0f, 1e-5f);
  ResetOrphanRegistry();
}

TEST(HadamardTransformTest, PositiveMatchInt4SylvesterHadamard) {
  ResetOrphanRegistry();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  const int32_t kN = 4;
  auto& input = subgraph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, kN}));
  SetSymQuant(input, 0.01f);

  auto& weights = subgraph.EmplaceTensor();
  weights.SetType(MakeRankedTensorType(kLiteRtElementTypeInt4, {kN, kN}));
  SetSymQuant(weights, 1.0f / (7.0f * 2.0f));
  // 4x4 Sylvester Hadamard packed into int4 nibbles (+7 = 0x7, -7 = 0x9)
  auto pack_nibbles = [](int8_t lo, int8_t hi) -> uint8_t {
    return static_cast<uint8_t>((lo & 0x0F) | ((hi & 0x0F) << 4));
  };
  std::vector<uint8_t> h4_packed = {
      pack_nibbles(7, 7),  pack_nibbles(7, 7),    // row 0: + + + +
      pack_nibbles(7, -7), pack_nibbles(7, -7),   // row 1: + - + -
      pack_nibbles(7, 7),  pack_nibbles(-7, -7),  // row 2: + + - -
      pack_nibbles(7, -7), pack_nibbles(-7, 7),   // row 3: + - - +
  };
  SetConstWeights<uint8_t>(weights, h4_packed);

  auto& output = subgraph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, kN}));
  SetSymQuant(output, 0.01f);

  FullyConnectedOptions fc_opts;
  fc_opts.fused_activation_function = 0;
  fc_opts.weights_format = 0;
  fc_opts.keep_num_dims = true;
  fc_opts.quantized_bias_type = kLiteRtElementTypeNone;
  fc_opts.asymmetric_quantize_input = false;
  auto& fc_op =
      BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflFullyConnected,
                         {&input, &weights}, {&output}, fc_opts);

  EXPECT_EQ(ConvertFcToHadamardTransform(ctx, &builder_impl, &fc_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Ops()[0]->OpCode(), kLiteRtOpCodeShloComposite);
  ResetOrphanRegistry();
}

TEST(ParallelizeHadamardTest, PositiveMatchAndRewrite) {
  ResetOrphanRegistry();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  auto& in_tensor = subgraph.EmplaceTensor();
  in_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 64}));
  SetSymQuant(in_tensor, 0.01f);

  auto& pre_shape = subgraph.EmplaceTensor();
  pre_shape.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
  const int32_t kPreShape[4] = {1, 2, 16, 4};
  SetConstWeights<int32_t>(pre_shape, kPreShape);

  auto& h_in = subgraph.EmplaceTensor();
  h_in.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 16, 4}));
  SetSymQuant(h_in, 0.01f);

  auto& pre_reshape = subgraph.EmplaceOp();
  pre_reshape.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&in_tensor, pre_reshape);
  AttachInput(&pre_shape, pre_reshape);
  AttachOutput(&h_in, pre_reshape);

  auto& w = subgraph.EmplaceTensor();
  w.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, {4, 4}));
  SetSymQuant(w, 1.0f / (127.0f * 2.0f));
  std::vector<int8_t> h4 = {
      127, 127, 127,  127,  127, -127, 127,  -127,
      127, 127, -127, -127, 127, -127, -127, 127,
  };
  SetConstWeights<int8_t>(w, h4);

  auto& h_out = subgraph.EmplaceTensor();
  h_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 16, 4}));
  SetSymQuant(h_out, 0.01f);

  FullyConnectedOptions fc_opts;
  fc_opts.fused_activation_function = 0;
  fc_opts.weights_format = 0;
  fc_opts.keep_num_dims = true;
  fc_opts.quantized_bias_type = kLiteRtElementTypeNone;
  fc_opts.asymmetric_quantize_input = false;
  auto& fc_op =
      BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflFullyConnected,
                         {&h_in, &w}, {&h_out}, fc_opts);

  LiteRtBuilderT b_fc;
  EXPECT_EQ(ConvertFcToHadamardTransform(ctx, &b_fc, &fc_op), kLiteRtStatusOk);
  b_fc.ApplyChanges(&subgraph);

  auto& post_shape = subgraph.EmplaceTensor();
  post_shape.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));
  const int32_t kPostShape[3] = {1, 2, 64};
  SetConstWeights<int32_t>(post_shape, kPostShape);

  auto& out_tensor = subgraph.EmplaceTensor();
  out_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 64}));
  SetSymQuant(out_tensor, 0.01f);

  auto& post_reshape = subgraph.EmplaceOp();
  post_reshape.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&h_out, post_reshape);
  AttachInput(&post_shape, post_reshape);
  AttachOutput(&out_tensor, post_reshape);

  EXPECT_EQ(ParallelizeHadamardTransform(ctx, &builder_impl, &post_reshape),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  int split_count = 0;
  int concat_count = 0;
  int hadamard_slice_count = 0;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflSplit) {
      split_count++;
      EXPECT_EQ(op->Outputs().size(), 16);
    } else if (op->OpCode() == kLiteRtOpCodeTflConcatenation) {
      concat_count++;
      EXPECT_EQ(op->Inputs().size(), 16);
    } else if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      const auto& d = op->Output(0).Type().second.ranked_tensor_type.layout;
      if (d.rank == 3 && d.dimensions[2] == 4) {
        hadamard_slice_count++;
      }
    }
  }
  EXPECT_EQ(split_count, 1);
  EXPECT_EQ(concat_count, 1);
  EXPECT_EQ(hadamard_slice_count, 16);
  ResetOrphanRegistry();
}

//===----------------------------------------------------------------------===//
// SimplifyMaskingAddTransformation Tests
//===----------------------------------------------------------------------===//

TEST(MaskSimplificationTest, PositiveMatchAndRewrite) {
  ResetOrphanRegistry();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  auto& b0 = subgraph.EmplaceTensor();
  b0.SetType(MakeRankedTensorType(kLiteRtElementTypeBool, {1, 1, 128, 1024}));
  auto& b1 = subgraph.EmplaceTensor();
  b1.SetType(MakeRankedTensorType(kLiteRtElementTypeBool, {1, 1, 128, 1024}));

  auto& c0_out = subgraph.EmplaceTensor();
  c0_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 1, 128, 1024}));
  auto& cast0 = subgraph.EmplaceOp();
  cast0.SetOpCode(kLiteRtOpCodeTflCast);
  AttachInput(&b0, cast0);
  AttachOutput(&c0_out, cast0);

  auto& c1_out = subgraph.EmplaceTensor();
  c1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 1, 128, 1024}));
  auto& cast1 = subgraph.EmplaceOp();
  cast1.SetOpCode(kLiteRtOpCodeTflCast);
  AttachInput(&b1, cast1);
  AttachOutput(&c1_out, cast1);

  auto& add_out = subgraph.EmplaceTensor();
  add_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 1, 128, 1024}));
  AddOptions add_opts{};
  add_opts.fused_activation_function = 0;
  auto& add_op = BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflAdd,
                                    {&c0_out, &c1_out}, {&add_out}, add_opts);

  EXPECT_EQ(SimplifyMaskingAddTransformation(ctx, &builder_impl, &add_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  int logical_or_count = 0;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflLogicalOr) {
      logical_or_count++;
    }
  }
  EXPECT_EQ(logical_or_count, 1);
  ResetOrphanRegistry();
}

TEST(MaskSimplificationTest, ReshapeAddReshapePositiveMatch) {
  ResetOrphanRegistry();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  auto& qk_in = subgraph.EmplaceTensor();
  qk_in.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 8, 16}));
  SetSymQuant(qk_in, 0.01f);

  auto& pre_shape = subgraph.EmplaceTensor();
  pre_shape.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {5}));
  const int32_t kPreShape[5] = {1, 2, 2, 4, 16};
  SetConstWeights<int32_t>(pre_shape, kPreShape);

  auto& pre_out = subgraph.EmplaceTensor();
  pre_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 4, 16}));
  SetSymQuant(pre_out, 0.01f);

  auto& pre_reshape = subgraph.EmplaceOp();
  pre_reshape.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&qk_in, pre_reshape);
  AttachInput(&pre_shape, pre_reshape);
  AttachOutput(&pre_out, pre_reshape);

  auto& mask = subgraph.EmplaceTensor();
  mask.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 1, 4, 16}));
  SetSymQuant(mask, 0.01f);

  auto& add_out = subgraph.EmplaceTensor();
  add_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 4, 16}));
  SetSymQuant(add_out, 0.01f);
  AddOptions add_opts{};
  add_opts.fused_activation_function = 0;
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflAdd, {&pre_out, &mask},
                     {&add_out}, add_opts);

  auto& post_shape = subgraph.EmplaceTensor();
  post_shape.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
  const int32_t kPostShape[4] = {1, 2, 8, 16};
  SetConstWeights<int32_t>(post_shape, kPostShape);

  auto& final_out = subgraph.EmplaceTensor();
  final_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 8, 16}));
  SetSymQuant(final_out, 0.01f);

  auto& post_reshape = subgraph.EmplaceOp();
  post_reshape.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&add_out, post_reshape);
  AttachInput(&post_shape, post_reshape);
  AttachOutput(&final_out, post_reshape);

  EXPECT_EQ(SimplifyMaskingAddTransformation(ctx, &builder_impl, &post_reshape),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  int concat_count = 0;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflConcatenation) {
      concat_count++;
      EXPECT_EQ(op->Inputs().size(), 2);
    }
  }
  EXPECT_EQ(concat_count, 1);
  ResetOrphanRegistry();
}

//===----------------------------------------------------------------------===//
// GqaAttentionTransformation Tests
//===----------------------------------------------------------------------===//

TEST(GqaAttentionTest, GqaDecodePositiveMatch) {
  ResetOrphanRegistry();
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::internal::BufferManager buffer_manager;
  LiteRtSubgraphT subgraph(&buffer_manager);
  LiteRtBuilderT builder_impl;

  // H_q = 4, H_kv = 2, G = 2, S = 1, D = 8, D_v = 8, L_cache = 15, L_curr = 1
  auto& q_in = subgraph.EmplaceTensor();
  q_in.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 4, 1, 8}));
  SetSymQuant(q_in, 0.01f);

  auto& q_shape = subgraph.EmplaceTensor();
  q_shape.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
  const int32_t kQGroupedShape[4] = {1, 2, 2, 8};
  SetConstWeights<int32_t>(q_shape, kQGroupedShape);

  auto& q_grouped = subgraph.EmplaceTensor();
  q_grouped.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 8}));
  SetSymQuant(q_grouped, 0.01f);

  auto& rq_op = subgraph.EmplaceOp();
  rq_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&q_in, rq_op);
  AttachInput(&q_shape, rq_op);
  AttachOutput(&q_grouped, rq_op);

  auto& k_cache = subgraph.EmplaceTensor();
  k_cache.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 8, 15}));
  SetSymQuant(k_cache, 0.01f);

  auto& k_curr = subgraph.EmplaceTensor();
  k_curr.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 8, 1}));
  SetSymQuant(k_curr, 0.01f);

  BatchMatmulOptions bmm_opts{};
  bmm_opts.adj_x = false;
  bmm_opts.adj_y = false;
  bmm_opts.asymmetric_quantize_input = false;

  auto& bmm_k0_out = subgraph.EmplaceTensor();
  bmm_k0_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 15}));
  SetSymQuant(bmm_k0_out, 0.01f);
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflBatchMatmul,
                     {&q_grouped, &k_cache}, {&bmm_k0_out}, bmm_opts);

  auto& bmm_k1_out = subgraph.EmplaceTensor();
  bmm_k1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 1}));
  SetSymQuant(bmm_k1_out, 0.01f);
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflBatchMatmul,
                     {&q_grouped, &k_curr}, {&bmm_k1_out}, bmm_opts);

  auto& concat_qk_out = subgraph.EmplaceTensor();
  concat_qk_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 16}));
  SetSymQuant(concat_qk_out, 0.01f);
  ConcatenationOptions concat_opts{};
  concat_opts.axis = 3;
  concat_opts.fused_activation_function = 0;
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflConcatenation,
                     {&bmm_k0_out, &bmm_k1_out}, {&concat_qk_out}, concat_opts);

  auto& base_mask = subgraph.EmplaceTensor();
  base_mask.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 1, 1, 16}));
  SetSymQuant(base_mask, 0.01f);

  auto& sm_in = subgraph.EmplaceTensor();
  sm_in.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 16}));
  SetSymQuant(sm_in, 0.01f);
  AddOptions add_opts{};
  add_opts.fused_activation_function = 0;
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflAdd,
                     {&concat_qk_out, &base_mask}, {&sm_in}, add_opts);

  auto& sm_out = subgraph.EmplaceTensor();
  sm_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 16}));
  SetSymQuant(sm_out, 0.01f);
  SoftmaxOptions sm_opts{};
  sm_opts.beta = 1.0f;
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflSoftmax, {&sm_in},
                     {&sm_out}, sm_opts);

  auto& b0 = subgraph.EmplaceTensor();
  b0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
  const int32_t kBegin0[4] = {0, 0, 0, 0};
  SetConstWeights<int32_t>(b0, kBegin0);
  auto& s0 = subgraph.EmplaceTensor();
  s0.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
  const int32_t kSize0[4] = {1, 2, 2, 15};
  SetConstWeights<int32_t>(s0, kSize0);
  auto& sl0_out = subgraph.EmplaceTensor();
  sl0_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 15}));
  SetSymQuant(sl0_out, 0.01f);
  auto& sl0_op = subgraph.EmplaceOp();
  sl0_op.SetOpCode(kLiteRtOpCodeTflSlice);
  AttachInput(&sm_out, sl0_op);
  AttachInput(&b0, sl0_op);
  AttachInput(&s0, sl0_op);
  AttachOutput(&sl0_out, sl0_op);

  auto& b1 = subgraph.EmplaceTensor();
  b1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
  const int32_t kBegin1[4] = {0, 0, 0, 15};
  SetConstWeights<int32_t>(b1, kBegin1);
  auto& s1 = subgraph.EmplaceTensor();
  s1.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
  const int32_t kSize1[4] = {1, 2, 2, 1};
  SetConstWeights<int32_t>(s1, kSize1);
  auto& sl1_out = subgraph.EmplaceTensor();
  sl1_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 1}));
  SetSymQuant(sl1_out, 0.01f);
  auto& sl1_op = subgraph.EmplaceOp();
  sl1_op.SetOpCode(kLiteRtOpCodeTflSlice);
  AttachInput(&sm_out, sl1_op);
  AttachInput(&b1, sl1_op);
  AttachInput(&s1, sl1_op);
  AttachOutput(&sl1_out, sl1_op);

  auto& v_cache = subgraph.EmplaceTensor();
  v_cache.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 15, 8}));
  SetSymQuant(v_cache, 0.01f);
  auto& v_curr = subgraph.EmplaceTensor();
  v_curr.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 1, 8}));
  SetSymQuant(v_curr, 0.01f);

  auto& bmm_v0_out = subgraph.EmplaceTensor();
  bmm_v0_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 8}));
  SetSymQuant(bmm_v0_out, 0.01f);
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflBatchMatmul,
                     {&sl0_out, &v_cache}, {&bmm_v0_out}, bmm_opts);

  auto& bmm_v1_out = subgraph.EmplaceTensor();
  bmm_v1_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 8}));
  SetSymQuant(bmm_v1_out, 0.01f);
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflBatchMatmul,
                     {&sl1_out, &v_curr}, {&bmm_v1_out}, bmm_opts);

  auto& attn_out = subgraph.EmplaceTensor();
  attn_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 2, 2, 8}));
  SetSymQuant(attn_out, 0.01f);
  BuildOpWithOptions(ctx, subgraph, kLiteRtOpCodeTflAdd,
                     {&bmm_v0_out, &bmm_v1_out}, {&attn_out}, add_opts);

  auto& final_shape = subgraph.EmplaceTensor();
  final_shape.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {3}));
  const int32_t kFinalShape[3] = {1, 1, 32};
  SetConstWeights<int32_t>(final_shape, kFinalShape);
  auto& final_out = subgraph.EmplaceTensor();
  final_out.SetType(MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 1, 32}));
  SetSymQuant(final_out, 0.01f);
  auto& final_reshape = subgraph.EmplaceOp();
  final_reshape.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&attn_out, final_reshape);
  AttachInput(&final_shape, final_reshape);
  AttachOutput(&final_out, final_reshape);

  EXPECT_EQ(GqaAttentionTransformation(ctx, &builder_impl, &final_reshape),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  // Single-token decode is decomposed per KV head: each slice stacks the G = 2
  // query heads that share a KV head, so there are H_kv = 2 softmaxes over
  // [1, 1, G, L_total] and 4 batch matmuls per KV head.
  int split_count = 0;
  int grouped_softmax_count = 0;
  int bmm_count = 0;
  for (const auto& op : subgraph.Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflSplit) {
      split_count++;
    } else if (op->OpCode() == kLiteRtOpCodeTflBatchMatmul) {
      // The original [1, H_kv, ...] matmuls remain as orphans until
      // OrphanCleanupTransformation runs; only count the per-KV-head ones.
      const auto& d = op->Output(0).Type().second.ranked_tensor_type.layout;
      if (d.rank == 4 && d.dimensions[1] == 1) bmm_count++;
    } else if (op->OpCode() == kLiteRtOpCodeTflSoftmax) {
      const auto& d = op->Output(0).Type().second.ranked_tensor_type.layout;
      if (d.rank == 4 && d.dimensions[1] == 1 && d.dimensions[2] == 2 &&
          d.dimensions[3] == 16) {
        grouped_softmax_count++;
      }
    }
  }
  EXPECT_EQ(split_count, 5);
  EXPECT_EQ(grouped_softmax_count, 2);
  EXPECT_EQ(bmm_count, 8);

  // The attention output is still produced by a Reshape into final_out.
  const LiteRtOp final_def = final_out.DefiningOp();
  ASSERT_NE(final_def, nullptr);
  EXPECT_EQ(final_def->OpCode(), kLiteRtOpCodeTflReshape);
  const auto& concat_dims =
      final_def->Input(0).Type().second.ranked_tensor_type.layout;
  ASSERT_EQ(concat_dims.rank, 4);
  EXPECT_EQ(concat_dims.dimensions[1], 2);
  EXPECT_EQ(concat_dims.dimensions[2], 2);
  EXPECT_EQ(concat_dims.dimensions[3], 8);
  ResetOrphanRegistry();
}

}  // namespace
}  // namespace litert::compiler
