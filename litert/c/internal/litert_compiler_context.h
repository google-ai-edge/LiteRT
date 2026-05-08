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
///
/// @note This struct is shared with LiteRT runtime and Compiler Plugins. So it
/// must be ABI stable.
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

// ABI compatibility check for LiteRtCompilerContext.
//
// Note: Please get review from the LiteRT ABI compatibility team when you make
// changes to this struct.
#if defined(__cplusplus) && defined(__SIZEOF_POINTER__) && \
    __SIZEOF_POINTER__ == 8
static_assert(sizeof(LiteRtCompilerContext) == 296,
              "LiteRtCompilerContext size mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_model_subgraphs) == 0,
              "LiteRtCompilerContext get_num_model_subgraphs offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_model_subgraph) == 8,
              "LiteRtCompilerContext get_model_subgraph offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_subgraph_ops) == 16,
              "LiteRtCompilerContext get_num_subgraph_ops offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_subgraph_op) == 24,
              "LiteRtCompilerContext get_subgraph_op offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_subgraph_inputs) == 32,
              "LiteRtCompilerContext get_num_subgraph_inputs offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_subgraph_input) == 40,
              "LiteRtCompilerContext get_subgraph_input offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_subgraph_outputs) == 48,
              "LiteRtCompilerContext get_num_subgraph_outputs offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_subgraph_output) == 56,
              "LiteRtCompilerContext get_subgraph_output offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_op_code) == 64,
              "LiteRtCompilerContext get_op_code offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_custom_code) == 72,
              "LiteRtCompilerContext get_custom_code offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_op_inputs) == 80,
              "LiteRtCompilerContext get_num_op_inputs offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_op_input) == 88,
              "LiteRtCompilerContext get_op_input offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_op_outputs) == 96,
              "LiteRtCompilerContext get_num_op_outputs offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_op_output) == 104,
              "LiteRtCompilerContext get_op_output offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_name) == 112,
              "LiteRtCompilerContext get_tensor_name offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_index) == 120,
              "LiteRtCompilerContext get_tensor_index offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_type_id) == 128,
              "LiteRtCompilerContext get_tensor_type_id offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_ranked_tensor_type) == 136,
              "LiteRtCompilerContext get_ranked_tensor_type offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_unranked_tensor_type) == 144,
              "LiteRtCompilerContext get_unranked_tensor_type offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_quantization_type_id) == 152,
              "LiteRtCompilerContext get_quantization_type_id offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_per_tensor_quantization) == 160,
    "LiteRtCompilerContext get_per_tensor_quantization offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_per_channel_quantization) == 168,
    "LiteRtCompilerContext get_per_channel_quantization offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_num_tensor_uses) == 176,
              "LiteRtCompilerContext get_num_tensor_uses offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_use) == 184,
              "LiteRtCompilerContext get_tensor_use offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_defining_op) == 192,
              "LiteRtCompilerContext get_tensor_defining_op offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_tensor_weights) == 200,
              "LiteRtCompilerContext get_tensor_weights offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_weights_buffer_id) == 208,
              "LiteRtCompilerContext get_weights_buffer_id offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_weights_bytes) == 216,
              "LiteRtCompilerContext get_weights_bytes offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_shlo_composite_op_name) == 224,
    "LiteRtCompilerContext get_shlo_composite_op_name offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext,
             get_shlo_composite_op_decomposition_subgraph_index) == 232,
    "LiteRtCompilerContext get_shlo_composite_op_decomposition_subgraph_index "
    "offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_shlo_composite_op_attributes) == 240,
    "LiteRtCompilerContext get_shlo_composite_op_attributes offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_shlo_composite_op_version) == 248,
    "LiteRtCompilerContext get_shlo_composite_op_version offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, push_op) == 256,
              "LiteRtCompilerContext push_op offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, get_opaque_options) == 264,
              "LiteRtCompilerContext get_opaque_options offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, find_opaque_options_data) == 272,
              "LiteRtCompilerContext find_opaque_options_data offset mismatch");
static_assert(offsetof(LiteRtCompilerContext, destroy_options) == 280,
              "LiteRtCompilerContext destroy_options offset mismatch");
static_assert(
    offsetof(LiteRtCompilerContext, get_environment_options_value) == 288,
    "LiteRtCompilerContext get_environment_options_value offset mismatch");
#endif  // __cplusplus

LiteRtCompilerContext* LrtGetCompilerContext();

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_COMPILER_CONTEXT_H_
