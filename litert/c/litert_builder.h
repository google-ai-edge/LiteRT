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

#ifndef ODML_LITERT_LITERT_C_LITERT_BUILDER_H_
#define ODML_LITERT_LITERT_C_LITERT_BUILDER_H_

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
// LiteRtBuilder (Expirementatal API)
//

// Creates a new tensor within the graph being rewritten.
// This function allows specifying all properties of the tensor, including its
// type, shape (ranked or unranked), weight data, quantization parameters, and
// name. The new tensor is created within the context of the provided
// builder.
LiteRtStatus LiteRtBuilderBuildTensor(
    LiteRtBuilder builder, LiteRtTensorTypeId tensor_type_id,
    LiteRtRankedTensorType ranked_tensor_type,
    LiteRtUnrankedTensorType unranked_tensor_type, LiteRtWeights weights,
    LiteRtQuantizationTypeId quantization_type_id,
    LiteRtQuantizationPerTensor per_tensor_quantization,
    LiteRtQuantizationPerChannel per_channel_quantization, const char* name,
    LiteRtTensor* new_tensor);

// Builds weights for a tensor. Builder will take the ownership of the data,
// the built weights will be owned by the tensor before calling ApplyChanges().
LiteRtStatus LiteRtBuilderBuildWeights(LiteRtBuilder builder,
                                       const uint8_t* data,
                                       LiteRtParamIndex size,
                                       LiteRtTensor tensor,
                                       LiteRtWeights* new_weights);

// Creates a new OP within the graph being rewritten.
// This function takes the OP code, input tensors, and output tensors
// to construct a new op in the graph.
// The new op is created within the context of the provided builder.
LiteRtStatus LiteRtBuilderBuildOp(LiteRtBuilder builder, LiteRtOpCode op_code,
                                  LiteRtParamIndex num_inputs,
                                  LiteRtTensor* inputs,
                                  LiteRtParamIndex num_outputs,
                                  LiteRtTensor* outputs, LiteRtOp* new_op);

// Removes an existing OP from the graph being rewritten.
// This function is performed transactionally within the builder context.
LiteRtStatus LiteRtBuilderEraseOp(LiteRtBuilder builder, LiteRtOp op_to_erase);

// Function pointer type for a rewrite pattern (also known as a transformation).
// A LiteRtPatternFn takes an LiteRtOp and an LiteRtBuilder
// instance. It can analyze the OP and use the builder to modify the graph,
// for example, by replacing the OP with a sequence of other operations.
typedef LiteRtStatus (*LiteRtPatternFn)(LiteRtBuilder builder, LiteRtOp op);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_BUILDER_H_
