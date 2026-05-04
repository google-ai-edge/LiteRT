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
#include "litert/cc/internal/litert_detail.h"
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

  int32_t BufferId() const {
    int32_t buffer_id = -1;
    if (ctx_ && ctx_->get_weights_buffer_id) {
      internal::AssertOk(ctx_->get_weights_buffer_id, weights_, &buffer_id);
    }
    return buffer_id;
  }

  absl::Span<const uint8_t> Bytes() const {
    size_t size = 0;
    const void* addr = nullptr;
    if (ctx_ && ctx_->get_weights_bytes) {
      internal::AssertOk(ctx_->get_weights_bytes, weights_, &addr, &size);
    }
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

  absl::string_view Name() const {
    const char* name = "";
    if (ctx_ && ctx_->get_tensor_name) {
      internal::AssertOk(ctx_->get_tensor_name, tensor_, &name);
    }
    return absl::string_view(name);
  }

  LiteRtTensorTypeId TypeId() const {
    LiteRtTensorTypeId type_id = kLiteRtRankedTensorType;
    if (ctx_ && ctx_->get_tensor_type_id) {
      internal::AssertOk(ctx_->get_tensor_type_id, tensor_, &type_id);
    }
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

  bool HasWeights() const {
    auto weights = Weights();
    auto bytes = weights.Bytes();
    return !bytes.empty();
  }

  Weights Weights() const {
    LiteRtWeights weights = nullptr;
    if (ctx_ && ctx_->get_tensor_weights) {
      internal::AssertOk(ctx_->get_tensor_weights, tensor_, &weights);
    }
    return litert::compiler::Weights(ctx_, weights);
  }

  std::optional<LiteRtTensorDefiningOp> DefiningOp() const {
    bool has_defining_op = false;
    LiteRtTensorDefiningOp defining_op;
    if (ctx_ && ctx_->get_tensor_defining_op) {
      internal::AssertOk(ctx_->get_tensor_defining_op, tensor_,
                         &has_defining_op, &defining_op);
    }
    if (has_defining_op) {
      return defining_op;
    }
    return std::nullopt;
  }

  Expected<Op> GetDefiningOp() const;

  bool IsConstant() const { return HasWeights() && !DefiningOp().has_value(); }

  LiteRtQuantizationTypeId QTypeId() const {
    LiteRtQuantizationTypeId q_type_id = kLiteRtQuantizationNone;
    if (ctx_ && ctx_->get_quantization_type_id) {
      internal::AssertOk(ctx_->get_quantization_type_id, Get(), &q_type_id);
    }
    return q_type_id;
  }

  bool HasQuantization() const { return QTypeId() != kLiteRtQuantizationNone; }

  LiteRtQuantizationPerTensor PerTensorQuantization() const {
    LiteRtQuantizationPerTensor per_tensor_quantization;
    if (ctx_ && ctx_->get_per_tensor_quantization) {
      internal::AssertOk(ctx_->get_per_tensor_quantization, Get(),
                         &per_tensor_quantization);
    }
    return per_tensor_quantization;
  }

  LiteRtQuantizationPerChannel PerChannelQuantization() const {
    LiteRtQuantizationPerChannel per_channel_quantization;
    if (ctx_ && ctx_->get_per_channel_quantization) {
      internal::AssertOk(ctx_->get_per_channel_quantization, Get(),
                         &per_channel_quantization);
    }
    return per_channel_quantization;
  }

  struct TensorUse;
  std::vector<TensorUse> Uses() const;

  bool IsSubgraphInput() const {
    auto def_op = DefiningOp();
    return !HasWeights() && !def_op.has_value();
  }

  uint32_t TensorIndex() const {
    uint32_t index = 0;
    if (ctx_ && ctx_->get_tensor_index) {
      internal::AssertOk(ctx_->get_tensor_index, Get(), &index);
    }
    return index;
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

  LiteRtOpCode Code() const {
    LiteRtOpCode opcode = kLiteRtOpCodeTflCustom;
    if (ctx_ && ctx_->get_op_code) {
      internal::AssertOk(ctx_->get_op_code, op_, &opcode);
    }
    return opcode;
  }

  std::vector<Tensor> Inputs() const {
    LiteRtParamIndex num_inputs = 0;
    if (ctx_ && ctx_->get_num_op_inputs) {
      internal::AssertOk(ctx_->get_num_op_inputs, op_, &num_inputs);
    }
    std::vector<Tensor> inputs;
    inputs.reserve(num_inputs);
    for (LiteRtParamIndex i = 0; i < num_inputs; ++i) {
      LiteRtTensor input;
      if (ctx_ && ctx_->get_op_input) {
        internal::AssertOk(ctx_->get_op_input, op_, i, &input);
      }
      inputs.emplace_back(ctx_, input);
    }
    return inputs;
  }

  std::vector<Tensor> Outputs() const {
    LiteRtParamIndex num_outputs = 0;
    if (ctx_ && ctx_->get_num_op_outputs) {
      internal::AssertOk(ctx_->get_num_op_outputs, op_, &num_outputs);
    }
    std::vector<Tensor> outputs;
    outputs.reserve(num_outputs);
    for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
      LiteRtTensor output;
      if (ctx_ && ctx_->get_op_output) {
        internal::AssertOk(ctx_->get_op_output, op_, i, &output);
      }
      outputs.emplace_back(ctx_, output);
    }
    return outputs;
  }

  Expected<absl::string_view> CustomCode() const {
    const char* custom_code;
    if (!ctx_ || !ctx_->get_custom_code) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    auto status = ctx_->get_custom_code(Get(), &custom_code);
    if (status != kLiteRtStatusOk) {
      return Error(status, "Failed to get custom code");
    }
    return absl::string_view(custom_code);
  }

  bool Is(LiteRtOpCode code) const { return Code() == code; }

 private:
  const LiteRtCompilerContext* ctx_;
  LiteRtOp op_;
};

inline Expected<Op> Tensor::GetDefiningOp() const {
  auto def_op = DefiningOp();
  if (def_op.has_value()) {
    return Op(ctx_, def_op->op);
  }
  return Unexpected(kLiteRtStatusErrorNotFound);
}

struct Tensor::TensorUse {
  Op user;
  LiteRtParamIndex user_arg_ind;
};

inline std::vector<Tensor::TensorUse> Tensor::Uses() const {
  LiteRtParamIndex num_uses = 0;
  if (ctx_ && ctx_->get_num_tensor_uses) {
    internal::AssertOk(ctx_->get_num_tensor_uses, Get(), &num_uses);
  }
  std::vector<TensorUse> uses;
  uses.reserve(num_uses);
  for (LiteRtParamIndex i = 0; i < num_uses; ++i) {
    LiteRtOp user = nullptr;
    LiteRtParamIndex user_arg_index = 0;
    if (ctx_ && ctx_->get_tensor_use) {
      internal::AssertOk(ctx_->get_tensor_use, Get(), i, &user,
                         &user_arg_index);
    }
    uses.emplace_back(TensorUse{Op(ctx_, user), user_arg_index});
  }
  return uses;
}

class Subgraph {
 public:
  Subgraph(const LiteRtCompilerContext* ctx, LiteRtSubgraph subgraph)
      : ctx_(ctx), subgraph_(subgraph) {}

  LiteRtSubgraph Get() const { return subgraph_; }

  std::vector<Tensor> Inputs() const {
    LiteRtParamIndex num_inputs = 0;
    if (ctx_ && ctx_->get_num_subgraph_inputs) {
      internal::AssertOk(ctx_->get_num_subgraph_inputs, subgraph_, &num_inputs);
    }
    std::vector<Tensor> inputs;
    inputs.reserve(num_inputs);
    for (LiteRtParamIndex i = 0; i < num_inputs; ++i) {
      LiteRtTensor input;
      if (ctx_ && ctx_->get_subgraph_input) {
        internal::AssertOk(ctx_->get_subgraph_input, subgraph_, i, &input);
      }
      inputs.emplace_back(ctx_, input);
    }
    return inputs;
  }

  std::vector<Tensor> Outputs() const {
    LiteRtParamIndex num_outputs = 0;
    if (ctx_ && ctx_->get_num_subgraph_outputs) {
      internal::AssertOk(ctx_->get_num_subgraph_outputs, subgraph_,
                         &num_outputs);
    }
    std::vector<Tensor> outputs;
    outputs.reserve(num_outputs);
    for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
      LiteRtTensor output;
      if (ctx_ && ctx_->get_subgraph_output) {
        internal::AssertOk(ctx_->get_subgraph_output, subgraph_, i, &output);
      }
      outputs.emplace_back(ctx_, output);
    }
    return outputs;
  }

  std::vector<Op> Ops() const {
    LiteRtParamIndex num_ops = 0;
    if (ctx_ && ctx_->get_num_subgraph_ops) {
      internal::AssertOk(ctx_->get_num_subgraph_ops, subgraph_, &num_ops);
    }
    std::vector<Op> ops;
    ops.reserve(num_ops);
    for (LiteRtParamIndex i = 0; i < num_ops; ++i) {
      LiteRtOp op;
      if (ctx_ && ctx_->get_subgraph_op) {
        internal::AssertOk(ctx_->get_subgraph_op, subgraph_, i, &op);
      }
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

  size_t NumSubgraphs() const {
    LiteRtParamIndex num_subgraphs = 0;
    if (ctx_ && ctx_->get_num_model_subgraphs) {
      internal::AssertOk(ctx_->get_num_model_subgraphs, model_, &num_subgraphs);
    }
    return num_subgraphs;
  }

  Expected<class Subgraph> Subgraph(size_t index) const {
    LiteRtSubgraph subgraph;
    if (!ctx_ || !ctx_->get_model_subgraph) {
      return Unexpected(kLiteRtStatusErrorNotFound);
    }
    LITERT_RETURN_IF_ERROR(ctx_->get_model_subgraph(model_, index, &subgraph));
    return litert::compiler::Subgraph(ctx_, subgraph);
  }

  Expected<class Subgraph> MainSubgraph() const { return this->Subgraph(0); }

 private:
  const LiteRtCompilerContext* ctx_;
  LiteRtModel model_;
};

}  // namespace litert::compiler

#endif  // ODML_LITERT_LITERT_COMPILER_CC_LITERT_MODEL_H_
