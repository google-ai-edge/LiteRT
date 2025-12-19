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
#include "litert/cc/internal/litert_builder.h"
#include "litert/cc/internal/litert_extended_model.h"

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
  if (root_op.Code() != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusPatternNoMatch;
  }
  OpInputs inputs = root_op.Inputs();
  OpOutputs outputs = root_op.Outputs();
  builder.BuildOp(kLiteRtOpCodeTflMul, inputs, outputs);
  builder.EraseOp(root_op);
  return kLiteRtStatusOk;
}

LiteRtStatus SqrtMeanSquareTransformation(LiteRtBuilder builder_ptr,
                                          LiteRtOp op) {
  Builder builder = Builder(builder_ptr);
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
  builder.BuildOp(kLiteRtOpCodeTflAbs, inputs, outputs);
  // Erase the original ops.
  builder.EraseOp(square_op);
  builder.EraseOp(mean_op);
  return kLiteRtStatusOk;
}

LiteRtStatus DummyTransformation(LiteRtBuilder builder_ptr, LiteRtOp op) {
  return kLiteRtStatusPatternNoMatch;
}

}  // extern "C"
