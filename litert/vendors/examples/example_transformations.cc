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

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_rewriter.h"

using litert::Op;
using litert::OpInputs;
using litert::OpOutputs;
using litert::Rewriter;

extern "C" {

LiteRtStatus SimpleAddOpToMulOpTransformation(LiteRtOp op,
                                              LiteRtRewriter rewriter_ptr) {
  // Convert to C++ objects.
  Rewriter rewriter = Rewriter(rewriter_ptr);
  Op root_op = Op(op);
  if (root_op.Code() != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusPatternNoMatch;
  }
  OpInputs inputs = root_op.Inputs();
  OpOutputs outputs = root_op.Outputs();
  rewriter.BuildOp(kLiteRtOpCodeTflMul, inputs, outputs);
  rewriter.EraseOp(root_op);
  return kLiteRtStatusOk;
}
LiteRtStatus SqrtMeanSquareTransformation(LiteRtOp op,
                                          LiteRtRewriter rewriter_ptr) {
  Rewriter rewriter = Rewriter(rewriter_ptr);
  Op root_op = Op(op);

  // Pattern Match
  if (root_op.Code() != kLiteRtOpCodeTflSqrt) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op mean_op = Op(root_op.Inputs().front().DefiningOp().value().op);
  if (mean_op.Code() != kLiteRtOpCodeTflMean) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op square_op = Op(mean_op.Inputs().front().DefiningOp().value().op);
  if (square_op.Code() != kLiteRtOpCodeTflMul) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (square_op.Inputs().size() != 2) {
    return kLiteRtStatusPatternNoMatch;
  }
  if (square_op.Inputs().at(0).Get() != square_op.Inputs().at(1).Get()) {
    return kLiteRtStatusPatternNoMatch;
  }
  // Reuse the inputs of the mul(square op).
  OpInputs inputs = square_op.Inputs();
  // Reuse the outputs of the mean op.
  OpOutputs outputs = mean_op.Outputs();
  // Build the abs op.
  rewriter.BuildOp(kLiteRtOpCodeTflAbs, inputs, outputs);
  // Erase the original ops.
  rewriter.EraseOp(square_op);
  rewriter.EraseOp(mean_op);
  return kLiteRtStatusOk;
}

}  // extern "C"
