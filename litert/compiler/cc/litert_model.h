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

#ifndef ODML_LITERT_LITERT_COMPILER_CC_LITERT_MODEL_H_
#define ODML_LITERT_LITERT_COMPILER_CC_LITERT_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"

namespace litert::compiler {

class Op;
class Tensor;

class Weights {
 public:
  Weights(const LiteRtCompilerContext* ctx, LiteRtWeights weights)
      : ctx_(ctx), weights_(weights) {}

  Expected<int32_t> BufferId() const {
    int32_t buffer_id = -1;
    if (!ctx_ || !ctx_->get_weights_buffer_id) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_weights_buffer_id(weights_, &buffer_id));
    return buffer_id;
  }

  Expected<absl::Span<const uint8_t>> Bytes() const {
    size_t size = 0;
    const void* addr = nullptr;
    if (!ctx_ || !ctx_->get_weights_bytes) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_weights_bytes(weights_, &addr, &size));
    return absl::MakeSpan(static_cast<const uint8_t*>(addr), size);
  }

 private:
  const LiteRtCompilerContext* ctx_;
  LiteRtWeights weights_;
};

class Tensor {
 public:
  Tensor(const LiteRtCompilerContext* ctx, LiteRtTensor tensor)
      : ctx_(ctx), tensor_(tensor) {}

  LiteRtTensor Get() const { return tensor_; }

  Expected<absl::string_view> Name() const {
    const char* name = "";
    if (!ctx_ || !ctx_->get_tensor_name) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_tensor_name(tensor_, &name));
    return absl::string_view(name);
  }

  Expected<LiteRtTensorTypeId> TypeId() const {
    LiteRtTensorTypeId type_id = kLiteRtRankedTensorType;
    if (!ctx_ || !ctx_->get_tensor_type_id) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_tensor_type_id(tensor_, &type_id));
    return type_id;
  }

  Expected<litert::RankedTensorType> RankedTensorType() const {
    LiteRtRankedTensorType type;
    if (!ctx_ || !ctx_->get_ranked_tensor_type) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_ranked_tensor_type(tensor_, &type));
    return litert::RankedTensorType(type);
  }

  litert::ElementType ElementType() const {
    auto ranked_type = RankedTensorType();
    if (!ranked_type.HasValue()) {
      return litert::ElementType::None;
    }
    return ranked_type.Value().ElementType();
  }

  Expected<LiteRtUnrankedTensorType> UnrankedTensorType() const {
    LiteRtUnrankedTensorType type;
    if (!ctx_ || !ctx_->get_unranked_tensor_type) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_unranked_tensor_type(tensor_, &type));
    return type;
  }

  Expected<bool> HasWeights() const {
    LITERT_ASSIGN_OR_RETURN(auto weights, GetWeights());
    LITERT_ASSIGN_OR_RETURN(auto bytes, weights.Bytes());
    return !bytes.empty();
  }

  Expected<Weights> GetWeights() const {
    LiteRtWeights weights = nullptr;
    if (!ctx_ || !ctx_->get_tensor_weights) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_tensor_weights(tensor_, &weights));
    return Weights(ctx_, weights);
  }

  Expected<std::optional<LiteRtTensorDefiningOp>> DefiningOp() const {
    bool has_defining_op = false;
    LiteRtTensorDefiningOp defining_op;
    if (!ctx_ || !ctx_->get_tensor_defining_op) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(
        ctx_->get_tensor_defining_op(tensor_, &has_defining_op, &defining_op));
    if (has_defining_op) {
      return std::make_optional(defining_op);
    }
    return std::optional<LiteRtTensorDefiningOp>();
  }

  Expected<Op> GetDefiningOp() const;

  Expected<bool> IsConstant() const {
    LITERT_ASSIGN_OR_RETURN(auto has_weights, HasWeights());
    LITERT_ASSIGN_OR_RETURN(auto def_op, DefiningOp());
    return has_weights && !def_op.has_value();
  }

  bool operator==(const Tensor& other) const {
    return tensor_ == other.tensor_;
  }
  bool operator!=(const Tensor& other) const {
    return tensor_ != other.tensor_;
  }

 private:
  const LiteRtCompilerContext* ctx_;
  LiteRtTensor tensor_;
};

class Op {
 public:
  Op(const LiteRtCompilerContext* ctx, LiteRtOp op) : ctx_(ctx), op_(op) {}

  LiteRtOp Get() const { return op_; }

  Expected<LiteRtOpCode> Code() const {
    LiteRtOpCode opcode = kLiteRtOpCodeTflCustom;
    if (!ctx_ || !ctx_->get_op_code) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_op_code(op_, &opcode));
    return opcode;
  }

  Expected<std::vector<Tensor>> Inputs() const {
    LiteRtParamIndex num_inputs = 0;
    if (!ctx_ || !ctx_->get_num_op_inputs) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_num_op_inputs(op_, &num_inputs));
    std::vector<Tensor> inputs;
    inputs.reserve(num_inputs);
    for (LiteRtParamIndex i = 0; i < num_inputs; ++i) {
      LiteRtTensor input;
      LITERT_RETURN_IF_ERROR(ctx_->get_op_input(op_, i, &input));
      inputs.emplace_back(ctx_, input);
    }
    return inputs;
  }

  Expected<std::vector<Tensor>> Outputs() const {
    LiteRtParamIndex num_outputs = 0;
    if (!ctx_ || !ctx_->get_num_op_outputs) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_num_op_outputs(op_, &num_outputs));
    std::vector<Tensor> outputs;
    outputs.reserve(num_outputs);
    for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
      LiteRtTensor output;
      LITERT_RETURN_IF_ERROR(ctx_->get_op_output(op_, i, &output));
      outputs.emplace_back(ctx_, output);
    }
    return outputs;
  }

 private:
  const LiteRtCompilerContext* ctx_;
  LiteRtOp op_;
};

inline Expected<Op> Tensor::GetDefiningOp() const {
  LITERT_ASSIGN_OR_RETURN(auto def_op, DefiningOp());
  if (def_op.has_value()) {
    return Op(ctx_, def_op->op);
  }
  return Unexpected(kLiteRtStatusErrorNotFound);
}

class Subgraph {
 public:
  Subgraph(const LiteRtCompilerContext* ctx, LiteRtSubgraph subgraph)
      : ctx_(ctx), subgraph_(subgraph) {}

  LiteRtSubgraph Get() const { return subgraph_; }

  Expected<std::vector<Tensor>> Inputs() const {
    LiteRtParamIndex num_inputs = 0;
    if (!ctx_ || !ctx_->get_num_subgraph_inputs) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(
        ctx_->get_num_subgraph_inputs(subgraph_, &num_inputs));
    std::vector<Tensor> inputs;
    inputs.reserve(num_inputs);
    for (LiteRtParamIndex i = 0; i < num_inputs; ++i) {
      LiteRtTensor input;
      LITERT_RETURN_IF_ERROR(ctx_->get_subgraph_input(subgraph_, i, &input));
      inputs.emplace_back(ctx_, input);
    }
    return inputs;
  }

  Expected<std::vector<Tensor>> Outputs() const {
    LiteRtParamIndex num_outputs = 0;
    if (!ctx_ || !ctx_->get_num_subgraph_outputs) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(
        ctx_->get_num_subgraph_outputs(subgraph_, &num_outputs));
    std::vector<Tensor> outputs;
    outputs.reserve(num_outputs);
    for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
      LiteRtTensor output;
      LITERT_RETURN_IF_ERROR(ctx_->get_subgraph_output(subgraph_, i, &output));
      outputs.emplace_back(ctx_, output);
    }
    return outputs;
  }

  Expected<std::vector<Op>> Ops() const {
    LiteRtParamIndex num_ops = 0;
    if (!ctx_ || !ctx_->get_num_subgraph_ops) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_num_subgraph_ops(subgraph_, &num_ops));
    std::vector<Op> ops;
    ops.reserve(num_ops);
    for (LiteRtParamIndex i = 0; i < num_ops; ++i) {
      LiteRtOp op;
      LITERT_RETURN_IF_ERROR(ctx_->get_subgraph_op(subgraph_, i, &op));
      ops.emplace_back(ctx_, op);
    }
    return ops;
  }

 private:
  const LiteRtCompilerContext* ctx_;
  LiteRtSubgraph subgraph_;
};

class Model {
 public:
  Model(const LiteRtCompilerContext* ctx, LiteRtModel model)
      : ctx_(ctx), model_(model) {}

  LiteRtModel Get() const { return model_; }

  Expected<size_t> NumSubgraphs() const {
    LiteRtParamIndex num_subgraphs = 0;
    if (!ctx_ || !ctx_->get_num_model_subgraphs) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(
        ctx_->get_num_model_subgraphs(model_, &num_subgraphs));
    return num_subgraphs;
  }

  Expected<Subgraph> GetSubgraph(size_t index) const {
    LiteRtSubgraph subgraph;
    if (!ctx_ || !ctx_->get_model_subgraph) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_model_subgraph(model_, index, &subgraph));
    return Subgraph(ctx_, subgraph);
  }

  Expected<Subgraph> MainSubgraph() const { return GetSubgraph(0); }

 private:
  const LiteRtCompilerContext* ctx_;
  LiteRtModel model_;
};

}  // namespace litert::compiler

#endif  // ODML_LITERT_LITERT_COMPILER_CC_LITERT_MODEL_H_
