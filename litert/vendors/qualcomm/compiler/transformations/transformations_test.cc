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

#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/buffer_manager.h"
#include "litert/core/model/model.h"
#include "litert/vendors/qualcomm/compiler/transformations/entry_embedding_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/legalize_int32_ops_transformation.h"
#include "litert/vendors/qualcomm/compiler/transformations/mlp_quant_transformation.h"
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

  // one_hot_x op: OneHot(coord_x, depth, on, off) -> one_hot_x [1, 630, 10240]
  auto& depth_x = subgraph.EmplaceTensor();
  auto& on_x = subgraph.EmplaceTensor();
  auto& off_x = subgraph.EmplaceTensor();
  auto& one_hot_x = subgraph.EmplaceTensor();
  one_hot_x.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 10240}));
  auto& one_hot_x_op = subgraph.EmplaceOp();
  one_hot_x_op.SetOpCode(kLiteRtOpCodeTflOneHot);
  AttachInput(&coord_x, one_hot_x_op);
  AttachInput(&depth_x, one_hot_x_op);
  AttachInput(&on_x, one_hot_x_op);
  AttachInput(&off_x, one_hot_x_op);
  AttachOutput(&one_hot_x, one_hot_x_op);

  // w_x [768, 10240] (FLOAT32)
  auto& w_x = subgraph.EmplaceTensor();
  w_x.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {768, 10240}));
  SetWeightsFromOwnedBuffer(
      w_x.Weights(), OwningBufferRef<uint8_t>(768 * 10240 * sizeof(float)));

  // fc_x op: FullyConnected(one_hot_x, w_x) -> fc_x_out [1, 630, 768]
  auto& fc_x_out = subgraph.EmplaceTensor();
  fc_x_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 768}));
  auto& fc_x_op = subgraph.EmplaceOp();
  fc_x_op.SetOpCode(kLiteRtOpCodeTflFullyConnected);
  AttachInput(&one_hot_x, fc_x_op);
  AttachInput(&w_x, fc_x_op);
  AttachOutput(&fc_x_out, fc_x_op);

  // reshape_x op: Reshape(fc_x_out, shape) -> reshape_x_out [1, 630, 768]
  auto& shape_x = subgraph.EmplaceTensor();
  auto& reshape_x_out = subgraph.EmplaceTensor();
  reshape_x_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 768}));
  auto& reshape_x_op = subgraph.EmplaceOp();
  reshape_x_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&fc_x_out, reshape_x_op);
  AttachInput(&shape_x, reshape_x_op);
  AttachOutput(&reshape_x_out, reshape_x_op);

  // one_hot_y op: OneHot(coord_y, depth, on, off) -> one_hot_y [1, 630, 10240]
  auto& depth_y = subgraph.EmplaceTensor();
  auto& on_y = subgraph.EmplaceTensor();
  auto& off_y = subgraph.EmplaceTensor();
  auto& one_hot_y = subgraph.EmplaceTensor();
  one_hot_y.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 10240}));
  auto& one_hot_y_op = subgraph.EmplaceOp();
  one_hot_y_op.SetOpCode(kLiteRtOpCodeTflOneHot);
  AttachInput(&coord_y, one_hot_y_op);
  AttachInput(&depth_y, one_hot_y_op);
  AttachInput(&on_y, one_hot_y_op);
  AttachInput(&off_y, one_hot_y_op);
  AttachOutput(&one_hot_y, one_hot_y_op);

  // w_y [768, 10240] (FLOAT32)
  auto& w_y = subgraph.EmplaceTensor();
  w_y.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {768, 10240}));
  SetWeightsFromOwnedBuffer(
      w_y.Weights(), OwningBufferRef<uint8_t>(768 * 10240 * sizeof(float)));

  // fc_y op: FullyConnected(one_hot_y, w_y) -> fc_y_out [1, 630, 768]
  auto& fc_y_out = subgraph.EmplaceTensor();
  fc_y_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 768}));
  auto& fc_y_op = subgraph.EmplaceOp();
  fc_y_op.SetOpCode(kLiteRtOpCodeTflFullyConnected);
  AttachInput(&one_hot_y, fc_y_op);
  AttachInput(&w_y, fc_y_op);
  AttachOutput(&fc_y_out, fc_y_op);

  // reshape_y op: Reshape(fc_y_out, shape) -> reshape_y_out [1, 630, 768]
  auto& shape_y = subgraph.EmplaceTensor();
  auto& reshape_y_out = subgraph.EmplaceTensor();
  reshape_y_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 768}));
  auto& reshape_y_op = subgraph.EmplaceOp();
  reshape_y_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&fc_y_out, reshape_y_op);
  AttachInput(&shape_y, reshape_y_op);
  AttachOutput(&reshape_y_out, reshape_y_op);

  // concat op: Concatenation(reshape_x_out, reshape_y_out) -> concat_out [1,
  // 630, 1536]
  auto& concat_out = subgraph.EmplaceTensor();
  concat_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 1536}));
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

  int gather_count = 0;
  int add_count = 0;
  for (const auto& op_item : subgraph.Ops()) {
    if (op_item->OpCode() == kLiteRtOpCodeTflGather) gather_count++;
    if (op_item->OpCode() == kLiteRtOpCodeTflAdd) add_count++;
  }
  EXPECT_EQ(gather_count, 2);
  EXPECT_EQ(add_count, 1);
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
  in.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 500, 768}));
  auto& out = subgraph.EmplaceTensor();
  out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 500, 768}));

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSum);
  AttachInput(&in, op);
  AttachOutput(&out, op);

  EXPECT_EQ(EntryEmbeddingTransformation(ctx, &builder_impl, &op),
            kLiteRtStatusPatternNoMatch);
}

TEST(EntryEmbeddingTest, NegativeWrongWeightShapeRejected) {
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  LiteRtSubgraphT subgraph;
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
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 512}));
  auto& reshape_x_op = subgraph.EmplaceOp();
  reshape_x_op.SetOpCode(kLiteRtOpCodeTflReshape);
  AttachInput(&fc_x_out, reshape_x_op);
  AttachInput(&shape_x, reshape_x_op);
  AttachOutput(&reshape_x_out, reshape_x_op);

  auto& dummy_y = subgraph.EmplaceTensor();
  dummy_y.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 512}));

  auto& concat_out = subgraph.EmplaceTensor();
  concat_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 1024}));
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
  LiteRtSubgraphT subgraph;
  LiteRtBuilderT builder_impl;

  // in_tensor [1, 630, 12, 64] (FLOAT32)
  auto& in_tensor = subgraph.EmplaceTensor();
  in_tensor.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 64}));

  // slice_outer: Slice(in_tensor, begin0, size0) -> slice_outer_out [1, 630,
  // 12, 32]
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

  // s0: Slice(slice_outer_out, begin1, size1) -> s0 [1, 630, 12, 16]
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

  // s1, s2, s3 [1, 630, 12, 16]
  auto& s1 = subgraph.EmplaceTensor();
  s1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& s2 = subgraph.EmplaceTensor();
  s2.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& s3 = subgraph.EmplaceTensor();
  s3.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));

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

  // Branch 0:
  // Mul(s0, cos2) -> mul0 [1, 630, 12, 16]
  auto& mul0 = subgraph.EmplaceTensor();
  mul0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& mul0_op = subgraph.EmplaceOp();
  mul0_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&s0, mul0_op);
  AttachInput(&cos2, mul0_op);
  AttachOutput(&mul0, mul0_op);

  // Mul(s1, sin2) -> mul1 [1, 630, 12, 16]
  auto& mul1 = subgraph.EmplaceTensor();
  mul1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& mul1_op = subgraph.EmplaceOp();
  mul1_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&s1, mul1_op);
  AttachInput(&sin2, mul1_op);
  AttachOutput(&mul1, mul1_op);

  // Sub(mul0, mul1) -> sub0 [1, 630, 12, 16]
  auto& sub0 = subgraph.EmplaceTensor();
  sub0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& sub0_op = subgraph.EmplaceOp();
  sub0_op.SetOpCode(kLiteRtOpCodeTflSub);
  AttachInput(&mul0, sub0_op);
  AttachInput(&mul1, sub0_op);
  AttachOutput(&sub0, sub0_op);

  // Add(mul0, mul1) -> add0 [1, 630, 12, 16]
  auto& add0 = subgraph.EmplaceTensor();
  add0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& add0_op = subgraph.EmplaceOp();
  add0_op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&mul0, add0_op);
  AttachInput(&mul1, add0_op);
  AttachOutput(&add0, add0_op);

  // Concat(sub0, add0) -> concat0 [1, 630, 12, 32]
  auto& concat0 = subgraph.EmplaceTensor();
  concat0.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));
  auto& concat0_op = subgraph.EmplaceOp();
  concat0_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&sub0, concat0_op);
  AttachInput(&add0, concat0_op);
  AttachOutput(&concat0, concat0_op);

  // Branch 1:
  // Mul(s2, cos3) -> mul2 [1, 630, 12, 16]
  auto& mul2 = subgraph.EmplaceTensor();
  mul2.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& mul2_op = subgraph.EmplaceOp();
  mul2_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&s2, mul2_op);
  AttachInput(&cos3, mul2_op);
  AttachOutput(&mul2, mul2_op);

  // Mul(s3, sin3) -> mul3 [1, 630, 12, 16]
  auto& mul3 = subgraph.EmplaceTensor();
  mul3.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& mul3_op = subgraph.EmplaceOp();
  mul3_op.SetOpCode(kLiteRtOpCodeTflMul);
  AttachInput(&s3, mul3_op);
  AttachInput(&sin3, mul3_op);
  AttachOutput(&mul3, mul3_op);

  // Sub(mul2, mul3) -> sub1 [1, 630, 12, 16]
  auto& sub1 = subgraph.EmplaceTensor();
  sub1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& sub1_op = subgraph.EmplaceOp();
  sub1_op.SetOpCode(kLiteRtOpCodeTflSub);
  AttachInput(&mul2, sub1_op);
  AttachInput(&mul3, sub1_op);
  AttachOutput(&sub1, sub1_op);

  // Add(mul2, mul3) -> add1 [1, 630, 12, 16]
  auto& add1 = subgraph.EmplaceTensor();
  add1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 16}));
  auto& add1_op = subgraph.EmplaceOp();
  add1_op.SetOpCode(kLiteRtOpCodeTflAdd);
  AttachInput(&mul2, add1_op);
  AttachInput(&mul3, add1_op);
  AttachOutput(&add1, add1_op);

  // Concat(sub1, add1) -> concat1 [1, 630, 12, 32]
  auto& concat1 = subgraph.EmplaceTensor();
  concat1.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 32}));
  auto& concat1_op = subgraph.EmplaceOp();
  concat1_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&sub1, concat1_op);
  AttachInput(&add1, concat1_op);
  AttachOutput(&concat1, concat1_op);

  // Root concat: Concat(concat0, concat1) -> root_out [1, 630, 12, 64]
  auto& root_out = subgraph.EmplaceTensor();
  root_out.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 630, 12, 64}));
  auto& root_op = subgraph.EmplaceOp();
  root_op.SetOpCode(kLiteRtOpCodeTflConcatenation);
  AttachInput(&concat0, root_op);
  AttachInput(&concat1, root_op);
  AttachOutput(&root_out, root_op);

  EXPECT_EQ(RopeAttentionLayerTransformation(ctx, &builder_impl, &root_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);
  ResetRopeTransformationState();
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

  EXPECT_EQ(RopeCleanupTransformation(ctx, &builder_impl, &concat_op),
            kLiteRtStatusOk);
  builder_impl.ApplyChanges(&subgraph);

  EXPECT_EQ(subgraph.Ops().size(), 0);
}

TEST(RopeCleanupTest, NegativeActiveOutputRejected) {
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

  EXPECT_EQ(RopeCleanupTransformation(ctx, &builder_impl, &concat_op),
            kLiteRtStatusPatternNoMatch);
}

}  // namespace
}  // namespace litert::compiler
