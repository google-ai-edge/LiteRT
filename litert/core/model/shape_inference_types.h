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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_SHAPE_INFERENCE_TYPES_H_
#define ODML_LITERT_LITERT_CORE_MODEL_SHAPE_INFERENCE_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"

namespace litert::internal {

// A vector of dimensions where -1 indicates a dynamic dimension.
using Dims = std::vector<int32_t>;

// Map of tensor pointers to their constant or propagated values.
using TensorDataMap = std::map<const LiteRtTensorT*, std::vector<uint8_t>>;

// Interface for operation-specific shape inference logic to access its inputs.
// Implementation-agnostic to support both graph-based and standalone inference.
class ShapeInferenceContext {
 public:
  virtual ~ShapeInferenceContext() = default;

  virtual Dims GetInputShape(size_t index) const = 0;

  // Returns empty span if input data is not statically available.
  virtual absl::Span<const uint8_t> GetInputData(size_t index) const = 0;

  virtual const TflOptions& GetOptions() const = 0;

  virtual LiteRtOpCode GetOpCode() const = 0;
};

// Captures the result of a stateless shape inference call.
struct InferenceResult {
  // Inferred shapes for each output of the operation.
  std::vector<Dims> output_shapes;

  // Optional: Constant data calculated during inference (e.g. for Shape op).
  // Key is the output index.
  std::map<size_t, std::vector<uint8_t>> propagated_data;
};

// Signature for the updated, stateless shape inference system.
using StatelessOpInferrer = std::function<LiteRtStatus(
    const ShapeInferenceContext& ctx, InferenceResult& result)>;

// Legacy signature for backward compatibility during transition.
using OpShapeInferrer = std::function<LiteRtStatus(
    const LiteRtOpT& op, absl::Span<Dims> input_shapes,
    std::vector<Dims>& output_shapes)>;

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_SHAPE_INFERENCE_TYPES_H_
