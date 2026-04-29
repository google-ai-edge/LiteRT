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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_COMPILER_CONTEXT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_COMPILER_CONTEXT_H_

#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"

#ifdef __cplusplus
extern "C" {
#endif

/// A function table that contains LiteRT C APIs needed for Compiler Plugins.
typedef struct LiteRtCompilerContext {
  // Model inspection
  LiteRtStatus (*get_num_model_subgraphs)(LiteRtModel model,
                                          LiteRtParamIndex* num_subgraphs);
  LiteRtStatus (*get_model_subgraph)(LiteRtModel model,
                                     LiteRtParamIndex subgraph_index,
                                     LiteRtSubgraph* subgraph);

  // Subgraph inspection
  LiteRtStatus (*get_num_subgraph_ops)(LiteRtSubgraph subgraph,
                                       LiteRtParamIndex* num_ops);
  LiteRtStatus (*get_subgraph_op)(LiteRtSubgraph subgraph,
                                  LiteRtParamIndex op_index, LiteRtOp* op);
  LiteRtStatus (*get_num_subgraph_inputs)(LiteRtSubgraph subgraph,
                                          LiteRtParamIndex* num_inputs);
  LiteRtStatus (*get_subgraph_input)(LiteRtSubgraph subgraph,
                                     LiteRtParamIndex input_index,
                                     LiteRtTensor* input);
  LiteRtStatus (*get_num_subgraph_outputs)(LiteRtSubgraph subgraph,
                                           LiteRtParamIndex* num_outputs);
  LiteRtStatus (*get_subgraph_output)(LiteRtSubgraph subgraph,
                                      LiteRtParamIndex output_index,
                                      LiteRtTensor* output);

  // Op inspection
  LiteRtStatus (*get_op_code)(LiteRtOp op, LiteRtOpCode* code);
  LiteRtStatus (*get_custom_code)(LiteRtOp op, const char** code);
  LiteRtStatus (*get_num_op_inputs)(LiteRtOp op, LiteRtParamIndex* num_inputs);
  LiteRtStatus (*get_op_input)(LiteRtOp op, LiteRtParamIndex input_index,
                               LiteRtTensor* input);
  LiteRtStatus (*get_num_op_outputs)(LiteRtOp op,
                                     LiteRtParamIndex* num_outputs);
  LiteRtStatus (*get_op_output)(LiteRtOp op, LiteRtParamIndex output_index,
                                LiteRtTensor* output);

  // Tensor inspection
  LiteRtStatus (*get_tensor_name)(LiteRtTensor tensor, const char** name);
  LiteRtStatus (*get_tensor_index)(LiteRtTensor tensor, uint32_t* tensor_index);
  LiteRtStatus (*get_tensor_type_id)(LiteRtTensor tensor,
                                     LiteRtTensorTypeId* type_id);
  LiteRtStatus (*get_ranked_tensor_type)(
      LiteRtTensor tensor, LiteRtRankedTensorType* ranked_tensor_type);
  LiteRtStatus (*get_unranked_tensor_type)(
      LiteRtTensor tensor, LiteRtUnrankedTensorType* unranked_tensor_type);
  LiteRtStatus (*get_quantization_type_id)(LiteRtTensor tensor,
                                           LiteRtQuantizationTypeId* q_type_id);
  LiteRtStatus (*get_per_tensor_quantization)(
      LiteRtTensor tensor,
      LiteRtQuantizationPerTensor* per_tensor_quantization);
  LiteRtStatus (*get_per_channel_quantization)(
      LiteRtTensor tensor,
      LiteRtQuantizationPerChannel* per_channel_quantization);
  LiteRtStatus (*get_num_tensor_uses)(LiteRtTensor tensor,
                                      LiteRtParamIndex* num_uses);
  LiteRtStatus (*get_tensor_use)(LiteRtTensor tensor,
                                 LiteRtParamIndex use_index, LiteRtOp* user,
                                 LiteRtParamIndex* user_arg_index);
  LiteRtStatus (*get_tensor_defining_op)(LiteRtTensor tensor,
                                         bool* has_defining_op,
                                         LiteRtTensorDefiningOp* defining_op);

  // Weights
  LiteRtStatus (*get_tensor_weights)(LiteRtTensor tensor,
                                     LiteRtWeights* weights);
  LiteRtStatus (*get_weights_buffer_id)(LiteRtWeights weights,
                                        int32_t* buffer_id);
  LiteRtStatus (*get_weights_bytes)(LiteRtWeights weights, const void** addr,
                                    size_t* size);

  // Op options
  LiteRtStatus (*get_shlo_composite_op_name)(LiteRtOp op, const char** name);
  LiteRtStatus (*get_shlo_composite_op_decomposition_subgraph_index)(
      LiteRtOp op, int32_t* decomposition_subgraph_index);
  LiteRtStatus (*get_shlo_composite_op_attributes)(LiteRtOp op,
                                                   const uint8_t** attributes,
                                                   int32_t* attributes_size);
  LiteRtStatus (*get_shlo_composite_op_version)(LiteRtOp op, int32_t* version);

  // Utility
  LiteRtStatus (*push_op)(LiteRtOpList op_list, LiteRtOp op,
                          LiteRtParamIndex partition_index);

  // Options
  LiteRtStatus (*get_opaque_options)(LiteRtOptions options,
                                     LiteRtOpaqueOptions* opaque_options);
  LiteRtStatus (*find_opaque_options_data)(LiteRtOpaqueOptions options,
                                           const char* payload_identifier,
                                           void** payload_data);
  void (*destroy_options)(LiteRtOptions options);

  // Environment options
  LiteRtStatus (*get_environment_options_value)(
      LiteRtEnvironmentOptions options, LiteRtEnvOptionTag tag,
      LiteRtAny* value);
} LiteRtCompilerContext;

LiteRtCompilerContext* LrtGetCompilerContext();

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_COMPILER_CONTEXT_H_
