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

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"

namespace litert::internal {

// A vector of dimensions where -1 indicates a dynamic dimension.
using Dims = std::vector<int32_t>;

// Map of tensor pointers to their constant or propagated values.
using TensorDataMap =
    absl::flat_hash_map<const LiteRtTensorT*, std::vector<uint8_t>>;

// Interface for operation-specific shape inference logic to access its inputs.
// Implementation-agnostic to support both graph-based and standalone inference.
class ShapeInferenceContext {
 public:
  virtual ~ShapeInferenceContext() = default;

  virtual size_t GetNumInputs() const = 0;

  virtual size_t GetNumOutputs() const = 0;

  virtual Dims GetInputShape(size_t index) const = 0;

  // Returns empty span if input data is not statically available.
  virtual absl::Span<const uint8_t> GetInputData(size_t index) const = 0;

  virtual const TflOptions& GetOptions() const = 0;

  virtual LiteRtOpCode GetOpCode() const = 0;

  virtual LiteRtElementType GetInputElementType(size_t index) const {
    return kLiteRtElementTypeNone;
  }

  virtual const LiteRtOpT* GetOp() const { return nullptr; }
};

// Captures the result of a stateless shape inference call.
struct InferenceResult {
  // Inferred shapes for each output of the operation.
  std::vector<Dims> output_shapes;

  // Optional: Constant data calculated during inference (e.g. for Shape op).
  // Key is the output index.
  std::map<size_t, std::vector<uint8_t>> propagated_data;
};

// Concrete context for standalone shape inference given a LiteRtOpT and input
// shapes.
class StandaloneShapeInferenceContext : public ShapeInferenceContext {
 public:
  StandaloneShapeInferenceContext(const LiteRtOpT& op,
                                  absl::Span<const Dims> input_shapes)
      : op_(op), input_shapes_(input_shapes) {}

  size_t GetNumInputs() const override { return input_shapes_.size(); }

  size_t GetNumOutputs() const override { return op_.Outputs().size(); }

  Dims GetInputShape(size_t index) const override {
    if (index >= input_shapes_.size()) return {};
    return input_shapes_[index];
  }

  absl::Span<const uint8_t> GetInputData(size_t index) const override {
    if (index >= op_.Inputs().size() || op_.Inputs()[index] == nullptr) {
      return {};
    }
    const auto& tensor = *op_.Inputs()[index];
    if (tensor.Weights().Buffer().Size() > 0) {
      auto weights = tensor.Weights().Buffer();
      return absl::MakeConstSpan(weights.Data(), weights.Size());
    }
    return {};
  }

  const TflOptions& GetOptions() const override { return GetTflOptions(op_); }

  LiteRtOpCode GetOpCode() const override { return op_.OpCode(); }

  LiteRtElementType GetInputElementType(size_t index) const override {
    if (index >= op_.Inputs().size() || op_.Inputs()[index] == nullptr) {
      return kLiteRtElementTypeNone;
    }
    const auto& tensor = *op_.Inputs()[index];
    if (tensor.Type().first == kLiteRtRankedTensorType) {
      return tensor.Type().second.ranked_tensor_type.element_type;
    } else if (tensor.Type().first == kLiteRtUnrankedTensorType) {
      return tensor.Type().second.unranked_tensor_type.element_type;
    }
    return kLiteRtElementTypeNone;
  }

  const LiteRtOpT* GetOp() const override { return &op_; }

 private:
  const LiteRtOpT& op_;
  absl::Span<const Dims> input_shapes_;
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
