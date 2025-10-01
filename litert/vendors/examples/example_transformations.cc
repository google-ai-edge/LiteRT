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
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_rewriter.h"

extern "C" {

LiteRtStatus SimpleAddOpToMulOpTransformation(LiteRtOp op,
                                              LiteRtRewriter rewriter_ptr) {
  // Convert to C++ objects.
  litert::Rewriter rewriter = litert::Rewriter(rewriter_ptr);
  litert::Op root_op = litert::Op(op);
  if (root_op.Code() != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusPatternNoMatch;
  }
  litert::OpInputs inputs = root_op.Inputs();
  litert::OpOutputs outputs = root_op.Outputs();
  rewriter.BuildOp(kLiteRtOpCodeTflMul, inputs, outputs);
  rewriter.EraseOp(root_op);
  return kLiteRtStatusOk;
}

}  // extern "C"
