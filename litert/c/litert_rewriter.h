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

#ifndef ODML_LITERT_LITERT_C_LITERT_REWRITER_H_
#define ODML_LITERT_LITERT_C_LITERT_REWRITER_H_

#include <stdbool.h>  // NOLINT: To use bool type in C
#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//
// LiteRtRewriter (Expirementatal API)
//

// Creates a new tensor within the graph being rewritten.
// This function allows specifying all properties of the tensor, including its
// type, shape (ranked or unranked), weight data, quantization parameters, and
// name. The new tensor is created within the context of the provided
// rewriter.
LiteRtStatus LiteRtRewriterBuildTensor(
    LiteRtTensorTypeId tensor_type_id,
    LiteRtRankedTensorType ranked_tensor_type,
    LiteRtUnrankedTensorType unranked_tensor_type, LiteRtWeights weights,
    LiteRtQuantizationTypeId quantization_type_id,
    LiteRtQuantizationPerTensor per_tensor_quantization,
    LiteRtQuantizationPerChannel per_channel_quantization,
    LiteRtRewriter rewriter, const char* name, LiteRtTensor* new_tensor);

// Creates a new OP within the graph being rewritten.
// This function takes the OP code, input tensors, and output tensors
// to construct a new op in the graph.
// The new op is created within the context of the provided rewriter.
LiteRtStatus LiteRtRewriterBuildOp(LiteRtOpCode op_code,
                                   LiteRtParamIndex num_inputs,
                                   LiteRtTensor* inputs,
                                   LiteRtParamIndex num_outputs,
                                   LiteRtTensor* outputs,
                                   LiteRtRewriter rewriter, LiteRtOp* new_op);

// Removes an existing OP from the graph being rewritten.
// This function is performed transactionally within the rewriter context.
LiteRtStatus LiteRtRewriterEraseOp(LiteRtOp op_to_erase,
                                   LiteRtRewriter rewriter);

// Function pointer type for a rewrite pattern (also known as a transformation).
// A LiteRtPatternFn takes an LiteRtOp and an LiteRtRewriter
// instance. It can analyze the OP and use the rewriter to modify the graph,
// for example, by replacing the OP with a sequence of other operations.
typedef LiteRtStatus (*LiteRtPatternFn)(LiteRtOp op, LiteRtRewriter rewriter);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_REWRITER_H_
