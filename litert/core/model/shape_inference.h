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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_SHAPE_INFERENCE_H_
#define ODML_LITERT_LITERT_CORE_MODEL_SHAPE_INFERENCE_H_

#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

// Engine to perform shape inference on a LiteRtModel or Subgraph.
class ShapeInferenceEngine {
 public:
  explicit ShapeInferenceEngine(LiteRtModelT* model);
  ShapeInferenceEngine();

  // Register a shape inferrer for a specific op code.
  void RegisterInferrer(LiteRtOpCode op_code, OpShapeInferrer inferrer);

  // Perform shape inference on the entire model (all subgraphs).
  // If `validation_only` is true, verifies that existing tensor shapes match
  // the inferred shapes without modifying them.
  // If validation fails, `failing_op` is updated to point to the problematic
  // op.
  LiteRtStatus InferShapes(bool validation_only = false,
                           LiteRtOp* failing_op = nullptr);

  // Perform shape inference on a specific subgraph.
  // If `validation_only` is true, verifies that existing tensor shapes match
  // the inferred shapes without modifying them.
  // If validation fails, `failing_op` is updated to point to the problematic
  // op.
  LiteRtStatus InferSubgraphShapes(LiteRtSubgraphT* subgraph,
                                   bool validation_only = false,
                                   LiteRtOp* failing_op = nullptr);

  // Helper to infer shapes for a single op.
  // If `validation_only` is true, verifies that existing output tensor shapes
  // match the inferred shapes.
  LiteRtStatus InferOpShapes(LiteRtOpT* op, bool validation_only = false);

  // Helper to infer shapes for a single op with provided inputs.
  // This function is stateless and pure calculation.
  LiteRtStatus InferOpShapes(const LiteRtOpT& op, absl::Span<Dims> input_shapes,
                             std::vector<Dims>& output_shapes);

  // Creates a new subgraph in the model which is a copy of `subgraph` but with
  // input shapes fixed to `input_shapes`. The shapes of all other tensors in
  // the new subgraph are then inferred and updated.
  // `new_subgraph` is an output parameter returning the newly created subgraph.
  // This allows creating a static shape version of a dynamic subgraph.
  LiteRtStatus SpecializeSubgraph(LiteRtSubgraphT* subgraph,
                                  absl::Span<Dims> input_shapes,
                                  LiteRtSubgraphT** new_subgraph);

 private:
  void RegisterStandardOps();

  LiteRtModelT* model_ = nullptr;
  absl::flat_hash_map<LiteRtOpCode, OpShapeInferrer> registry_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_SHAPE_INFERENCE_H_
