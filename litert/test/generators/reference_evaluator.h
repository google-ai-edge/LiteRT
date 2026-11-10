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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REFERENCE_EVALUATOR_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REFERENCE_EVALUATOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/simple_buffer.h"

namespace litert::testing {

// Evaluates a LiteRtModelT (starting at subgraph 0, recursively evaluating any
// composite operation decompositions) using reference operation kernels.
class ReferenceEvaluator {
 public:
  // Evaluates `model` from subgraph 0 using standard reference kernels,
  // recursively evaluating any kLiteRtOpCodeShloComposite decomposition
  // subgraphs.
  static Expected<void> Evaluate(const LiteRtModelT& model,
                                 const VarBuffers& inputs, VarBuffers& outputs);

  // Deprecated alias for Evaluate(); kept for backwards compatibility while
  // existing generators migrate to Evaluate().
  static Expected<void> EvaluateCompositeReference(const LiteRtModelT& model,
                                                   const VarBuffers& inputs,
                                                   VarBuffers& outputs) {
    return Evaluate(model, inputs, outputs);
  }

 private:
  friend class absl::NoDestructor<ReferenceEvaluator>;

  // Container representing intermediate tensor data and shape during reference
  // evaluation. Holds float32 (including converted float16) or int32 (including
  // converted bool) data.
  struct TensorData {
    // The data type of the tensor elements.
    LiteRtElementType element_type = kLiteRtElementTypeNone;

    // The shape dimensions of the tensor.
    std::vector<int32_t> dimensions;

    // Buffer for float32 and converted float16 tensor data.
    std::vector<float> f32_data;

    // Buffer for int32 and converted bool tensor data.
    std::vector<int32_t> i32_data;

    // Computes the total number of elements from `dimensions`. Returns an error
    // if any dimension is non-positive (<= 0).
    Expected<size_t> NumElements() const;

    // Ingests raw data from `data` based on `element_type` and converts if
    // necessary (e.g. float16 to float32).
    Expected<void> AssignData(const void* data, size_t num_elements);

    // Convenience overload that computes `num_elements` using `NumElements()`
    // and calls `AssignData(data, num_elements)`.
    Expected<void> AssignData(const void* data);

    // Copies the evaluated tensor data into the provided output SimpleBuffer,
    // handling type conversion and size validation.
    Expected<void> CopyTo(SimpleBuffer& out_buf) const;
  };

  // Environment mapping LiteRT tensors to their evaluated TensorData.
  using TensorEnv = absl::flat_hash_map<const LiteRtTensorT*, TensorData>;

  // Function signature for executing an operation kernel.
  // `out` is pre-allocated with the shape of `op.Outputs()[0]` (if ranked).
  using OpKernelHandler = std::function<Expected<void>(
      const LiteRtOpT& op, const TensorEnv& env, TensorData& out)>;

  ReferenceEvaluator();

  void RegisterOp(LiteRtOpCode op_code, OpKernelHandler handler);

  void RegisterStandardOps();

  Expected<void> EvaluateModel(const LiteRtModelT& model,
                               const VarBuffers& inputs,
                               VarBuffers& outputs) const;

  Expected<void> EvaluateSubgraphWithEnv(const LiteRtSubgraphT& subgraph,
                                         const LiteRtModelT& model,
                                         TensorEnv& tensor_env) const;

  Expected<void> EvaluateCompositeOp(const LiteRtOpT& composite_op,
                                     const LiteRtModelT& model,
                                     TensorEnv& tensor_env) const;

  Expected<void> ExecuteOp(const LiteRtOpT& op, TensorEnv& env) const;

  absl::flat_hash_map<LiteRtOpCode, OpKernelHandler> registry_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_REFERENCE_EVALUATOR_H_
