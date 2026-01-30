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

#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_builder.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_matchers.h"

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
  if (!litert::Match(root_op, litert::m_OpCode<kLiteRtOpCodeTflAdd>())) {
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
  Builder builder(builder_ptr);
  Op root_op(op);
  Op mean_op(nullptr);
  Op square_op(nullptr);

  litert::Tensor sq_in(nullptr);

  // Match: Sqrt(Mean(Mul(x, x)))
  // Capture Mean and Mul ops, and the square input x.
  // Verify that Mean and Mul are only used once (safe to fuse).
  auto mul_op_matcher = litert::m_Op<kLiteRtOpCodeTflMul>(
      litert::m_Capture(&sq_in, litert::m_Any()), litert::m_SameAs(&sq_in));

  auto mean_input_matcher = litert::m_Capture(
      &square_op, litert::m_AllOf(litert::m_HasOneUse(), mul_op_matcher));

  auto mean_op_matcher =
      litert::m_Op<kLiteRtOpCodeTflMean>(mean_input_matcher, litert::m_Any());

  auto sqrt_input_matcher = litert::m_Capture(
      &mean_op, litert::m_AllOf(litert::m_HasOneUse(), mean_op_matcher));

  auto root_matcher = litert::m_Op<kLiteRtOpCodeTflSqrt>(sqrt_input_matcher);

  if (!litert::Match(root_op, root_matcher)) {
    return kLiteRtStatusPatternNoMatch;
  }

  OpOutputs outputs = mean_op.Outputs();
  OpInputs abs_inputs;
  abs_inputs.push_back(std::move(sq_in));
  builder.BuildOp(kLiteRtOpCodeTflAbs, abs_inputs, outputs);
  builder.EraseOp(square_op);
  builder.EraseOp(mean_op);
  return kLiteRtStatusOk;
}

LiteRtStatus DummyTransformation(LiteRtBuilder builder_ptr, LiteRtOp op) {
  return kLiteRtStatusPatternNoMatch;
}

}  // extern "C"
