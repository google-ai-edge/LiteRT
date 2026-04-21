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

#include "litert/c/internal/litert_compiler_context.h"

#include "litert/c/litert_model.h"
#include "litert/c/litert_op_options.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"

LiteRtCompilerContext* LrtGetCompilerContext() {
  static LiteRtCompilerContext ctx = {
      .get_num_model_subgraphs = LiteRtGetNumModelSubgraphs,
      .get_model_subgraph = LiteRtGetModelSubgraph,
      .get_num_subgraph_ops = LiteRtGetNumSubgraphOps,
      .get_subgraph_op = LiteRtGetSubgraphOp,
      .get_num_subgraph_inputs = LiteRtGetNumSubgraphInputs,
      .get_subgraph_input = LiteRtGetSubgraphInput,
      .get_num_subgraph_outputs = LiteRtGetNumSubgraphOutputs,
      .get_subgraph_output = LiteRtGetSubgraphOutput,
      .get_op_code = LiteRtGetOpCode,
      .get_num_op_inputs = LiteRtGetNumOpInputs,
      .get_op_input = LiteRtGetOpInput,
      .get_num_op_outputs = LiteRtGetNumOpOutputs,
      .get_op_output = LiteRtGetOpOutput,
      .get_tensor_name = LiteRtGetTensorName,
      .get_tensor_type_id = LiteRtGetTensorTypeId,
      .get_ranked_tensor_type = LiteRtGetRankedTensorType,
      .get_unranked_tensor_type = LiteRtGetUnrankedTensorType,
      .get_tensor_defining_op = LiteRtGetTensorDefiningOp,
      .get_tensor_weights = LiteRtGetTensorWeights,
      .get_weights_buffer_id = LiteRtGetWeightsBufferId,
      .get_weights_bytes = LiteRtGetWeightsBytes,
      .get_shlo_composite_op_name = LiteRtGetSHLOCompositeOpName,
      .get_shlo_composite_op_decomposition_subgraph_index =
          LiteRtGetSHLOCompositeOpDecompositionSubgraphIndex,
      .get_shlo_composite_op_attributes = LiteRtGetSHLOCompositeOpAttributes,
      .get_shlo_composite_op_version = LiteRtGetSHLOCompositeOpVersion,
      .push_op = LiteRtPushOp,
      .get_opaque_options = LiteRtGetOpaqueOptions,
      .find_opaque_options_data = LiteRtFindOpaqueOptionsData,
      .destroy_options = LiteRtDestroyOptions,
  };
  return &ctx;
}
