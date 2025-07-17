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
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//
// LiteRtRewriter
//

// Builds a new tensor from its passed in specs.
LiteRtStatus LiteRtRewriterBuildTensor(
    LiteRtTensorTypeId tensor_type_id,
    LiteRtRankedTensorType ranked_tensor_type,
    LiteRtUnrankedTensorType unranked_tensor_type, LiteRtWeights weights,
    LiteRtQuantizationTypeId quantization_type_id,
    LiteRtQuantizationPerTensor per_tensor_quantization,
    LiteRtQuantizationPerChannel per_channel_quantization,
    LiteRtRewriter rewriter, const char* name, LiteRtParamIndex name_size,
    LiteRtTensor* new_tensor);

// Builds a new op from its passed in specs.
LiteRtStatus LiteRtRewriterBuildOp(LiteRtOpCode op_code,
                                   LiteRtParamIndex num_inputs,
                                   LiteRtTensor* inputs,
                                   LiteRtParamIndex num_outputs,
                                   LiteRtTensor* outputs,
                                   LiteRtRewriter rewriter, LiteRtOp* new_op);

// Transactionally erase an op.
LiteRtStatus LiteRtRewriterEraseOp(LiteRtOp op_to_erase,
                                   LiteRtRewriter rewriter);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_REWRITER_H_
