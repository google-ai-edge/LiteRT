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

#include "litert/vendors/examples/example_transformations.h"

#include <utility>
#include <vector>

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_matchers.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/cc/litert_op_options.h"

using litert::compiler::BatchMatmulOptions;
using litert::compiler::Builder;
using litert::compiler::Op;
using litert::compiler::OpInputs;
using litert::compiler::OpOutputs;
using litert::compiler::Tensor;

extern "C" {

LiteRtStatus SimpleAddOpToMulOpTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  // Convert to C++ objects.
  Builder builder = Builder(context, builder_ptr);
  Op root_op = Op(context, op);
  if (!litert::compiler::Match(
          root_op, litert::compiler::m_OpCode<kLiteRtOpCodeTflAdd>())) {
    return kLiteRtStatusPatternNoMatch;
  }
  OpInputs inputs = root_op.Inputs();
  std::vector<Tensor> inputs_vec(inputs.begin(), inputs.end());
  builder.ReplaceOp(root_op, kLiteRtOpCodeTflMul, inputs_vec);
  return kLiteRtStatusOk;
}

LiteRtStatus SqrtMeanSquareTransformation(const LiteRtCompilerContext* context,
                                          LiteRtBuilder builder_ptr,
                                          LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op root_op(context, op);
  Op mean_op;
  Op square_op;

  Tensor sq_in;

  // Match: Sqrt(Mean(Mul(x, x)))
  // Capture Mean and Mul ops, and the square input x.
  // Verify that Mean and Mul are only used once (safe to fuse).
  auto mul_op_matcher = litert::compiler::m_Op<kLiteRtOpCodeTflMul>(
      litert::compiler::m_CaptureOrSameAs(&sq_in, litert::compiler::m_Any()),
      litert::compiler::m_CaptureOrSameAs(&sq_in, litert::compiler::m_Any()));

  auto mean_input_matcher = litert::compiler::m_CaptureOrSameAs(
      &square_op, litert::compiler::m_AllOf(litert::compiler::m_HasOneUse(),
                                            mul_op_matcher));

  auto mean_op_matcher = litert::compiler::m_Op<kLiteRtOpCodeTflMean>(
      mean_input_matcher, litert::compiler::m_Any());

  auto sqrt_input_matcher = litert::compiler::m_CaptureOrSameAs(
      &mean_op, litert::compiler::m_AllOf(litert::compiler::m_HasOneUse(),
                                          mean_op_matcher));

  auto root_matcher =
      litert::compiler::m_Op<kLiteRtOpCodeTflSqrt>(sqrt_input_matcher);

  if (!litert::compiler::Match(root_op, root_matcher)) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Replace Mean with Abs(sq_in).
  // This implicitly reuses Mean's output tensor for Abs output.
  builder.ReplaceOp(mean_op, kLiteRtOpCodeTflAbs, {sq_in});

  // Clean up unused ops.
  builder.EraseOp(square_op);

  return kLiteRtStatusOk;
}

LiteRtStatus FuseMatMulRequantTransformation(
    const LiteRtCompilerContext* context, LiteRtBuilder builder_ptr,
    LiteRtOp op) {
  Builder builder(context, builder_ptr);
  Op root_op(context, op);
  Op matmul_op;

  // Match: Quantize(MatMul(...))
  if (!litert::compiler::Match(
          root_op,
          litert::compiler::m_Op<kLiteRtOpCodeTflQuantize>(
              litert::compiler::m_CaptureOrSameAs(
                  &matmul_op, litert::compiler::m_AllOf(
                                  litert::compiler::m_HasOneUse(),
                                  litert::compiler::m_OpCode<
                                      kLiteRtOpCodeTflBatchMatmul>()))))) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Check if it's a requantization: input/output element type must be the same.
  if (root_op.Inputs()[0].ElementType() != root_op.Outputs()[0].ElementType()) {
    return kLiteRtStatusPatternNoMatch;
  }

  OpInputs inputs = matmul_op.Inputs();
  std::vector<Tensor> inputs_vec(inputs.begin(), inputs.end());

  // Replace the Quant op with a new MatMul op that uses the same outputs as the
  // Quant op but takes inputs from the original MatMul op.
  Op new_matmul =
      builder.ReplaceOp(root_op, kLiteRtOpCodeTflBatchMatmul, inputs_vec);

  // The original Quant op is now replaced and will be erased by ApplyChanges.
  // We also need to explicitly erase the original MatMul op.
  builder.EraseOp(matmul_op);

  // Copy options from the original MatMul op to the new one.
  BatchMatmulOptions options;
  LITERT_RETURN_IF_ERROR(options.InitFromOp(context, matmul_op.Get()));
  auto res = builder.SetOpOptions(new_matmul, std::move(options));
  if (!res) {
    return res.Error().Status();
  }

  return kLiteRtStatusOk;
}

LiteRtStatus DummyTransformation(const LiteRtCompilerContext* context,
                                 LiteRtBuilder builder_ptr, LiteRtOp op) {
  return kLiteRtStatusPatternNoMatch;
}

}  // extern "C"
