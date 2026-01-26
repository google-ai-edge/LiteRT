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

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_builder.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_matchers.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_macros.h"

using litert::Builder;
using litert::Op;
using litert::OpInputs;
using litert::OpOutputs;

extern "C" {

LiteRtStatus SimpleAddOpToMulOpTransformation(LiteRtBuilder builder_ptr,
                                              LiteRtOp op) {
  // Convert to C++ objects.
  Builder builder = Builder(builder_ptr);
  Op root_op = Op(op);
  if (!litert::Match(root_op, litert::m_OpCode(kLiteRtOpCodeTflAdd))) {
    return kLiteRtStatusPatternNoMatch;
  }
  OpInputs inputs = root_op.Inputs();
  std::vector<litert::Tensor> inputs_vec(inputs.begin(), inputs.end());
  builder.ReplaceOp(root_op, kLiteRtOpCodeTflMul, inputs_vec);
  return kLiteRtStatusOk;
}

LiteRtStatus SqrtMeanSquareTransformation(LiteRtBuilder builder_ptr,
                                          LiteRtOp op) {
  Builder builder(builder_ptr);
  Op root_op(op);
  Op mean_op(nullptr);
  Op square_op(nullptr);

  litert::Tensor sq_in(nullptr);

  // Match: Sqrt(Mean(Mul(x, x)))
  // Capture Mean and Mul ops, and the square input x.
  // Verify that Mean and Mul are only used once (safe to fuse).
  if (!litert::Match(
          root_op,
          litert::m_Op(
              kLiteRtOpCodeTflSqrt,
              litert::m_Capture(
                  &mean_op,
                  litert::m_AllOf(
                      litert::m_HasOneUse(),
                      litert::m_Op(
                          kLiteRtOpCodeTflMean,
                          litert::m_Capture(
                              &square_op,
                              litert::m_AllOf(
                                  litert::m_HasOneUse(),
                                  litert::m_Op(kLiteRtOpCodeTflMul,
                                               litert::m_Capture(
                                                   &sq_in, litert::m_Any()),
                                               litert::m_SameAs(&sq_in)))),
                          litert::m_Any())))))) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Replace Mean with Abs(sq_in).
  // This implicitly reuses Mean's output tensor for Abs output.
  builder.ReplaceOp(mean_op, kLiteRtOpCodeTflAbs, {sq_in});

  // Clean up unused ops.
  builder.EraseOp(square_op);

  return kLiteRtStatusOk;
}

LiteRtStatus FuseMatMulRequantTransformation(LiteRtBuilder builder_ptr,
                                             LiteRtOp op) {
  Builder builder(builder_ptr);
  Op root_op(op);
  Op matmul_op(nullptr);

  // Match: Quantize(MatMul(...))
  if (!litert::Match(
          root_op,
          litert::m_Op(
              kLiteRtOpCodeTflQuantize,
              litert::m_Capture(
                  &matmul_op,
                  litert::m_AllOf(
                      litert::m_HasOneUse(),
                      litert::m_OpCode(kLiteRtOpCodeTflBatchMatmul)))))) {
    return kLiteRtStatusPatternNoMatch;
  }

  // Check if it's a requantization: input/output element type must be the same.
  if (root_op.Inputs()[0].ElementType() != root_op.Outputs()[0].ElementType()) {
    return kLiteRtStatusPatternNoMatch;
  }

  OpInputs inputs = matmul_op.Inputs();
  std::vector<litert::Tensor> inputs_vec(inputs.begin(), inputs.end());

  // Replace the Quant op with a new MatMul op that uses the same outputs as the
  // Quant op but takes inputs from the original MatMul op.
  Op new_matmul =
      builder.ReplaceOp(root_op, kLiteRtOpCodeTflBatchMatmul, inputs_vec);

  // The original Quant op is now replaced and will be erased by ApplyChanges.
  // We also need to explicitly erase the original MatMul op.
  builder.EraseOp(matmul_op);

  // Copy options from the original MatMul op to the new one.
  litert::BatchMatmulOptions options;
  LITERT_RETURN_IF_ERROR(options.InitFromOp(matmul_op.Get()));
  auto res = builder.SetOpOptions(new_matmul, std::move(options));
  if (!res) {
    return res.Error().Status();
  }

  return kLiteRtStatusOk;
}

LiteRtStatus DummyTransformation(LiteRtBuilder builder_ptr, LiteRtOp op) {
  return kLiteRtStatusPatternNoMatch;
}

}  // extern "C"
