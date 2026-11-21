// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CC_INTERNAL_LITERT_MODEL_H_
#define ODML_LITERT_LITERT_CC_INTERNAL_LITERT_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_ranked_tensor_type.h"
// copybara:uncomment_begin(google_only)
// #include "tflite/converter/allocation.h"  // IWYU pragma: keep
// copybara:uncomment_end

/// @file
/// @brief Defines internal C++ wrappers for the LiteRT model, subgraph, op,
/// tensor, and signature types.

namespace litert {

// Forward declaration of Op class.
class Op;

/// @brief A C++ wrapper for `LiteRtWeights`, representing tensor weights.
class Weights : public internal::NonOwnedHandle<LiteRtWeights> {
 public:
  explicit Weights(LiteRtWeights weights)
      : internal::NonOwnedHandle<LiteRtWeights>(weights) {}

  int32_t BufferId() const {
    int32_t buffer_id;
    internal::AssertOk(LiteRtGetWeightsBufferId, Get(), &buffer_id);
    return buffer_id;
  }

  absl::Span<const uint8_t> Bytes() const {
    size_t size;
    const void* addr;
    internal::AssertOk(LiteRtGetWeightsBytes, Get(), &addr, &size);
    return absl::MakeSpan(static_cast<const uint8_t*>(addr), size);
  }
};

namespace internal::model_detail {

inline absl::string_view FetchTensorName(LiteRtTensor tensor) {
  if (tensor == nullptr) {
    return "";
  }
  const char* name;
  internal::AssertOk(LiteRtGetTensorName, tensor, &name);
  return name;
}

inline std::uint32_t FetchTensorIndex(LiteRtTensor tensor) {
  if (tensor == nullptr) {
    return 0;
  }
  std::uint32_t index;
  internal::AssertOk(LiteRtGetTensorIndex, tensor, &index);
  return index;
}

inline LiteRtTensorTypeId FetchTensorTypeId(LiteRtTensor tensor) {
  if (tensor == nullptr) {
    return kLiteRtRankedTensorType;
  }
  LiteRtTensorTypeId type_id;
  internal::AssertOk(LiteRtGetTensorTypeId, tensor, &type_id);
  return type_id;
}

inline std::variant<LiteRtUnrankedTensorType, litert::RankedTensorType>
FetchTensorType(LiteRtTensor tensor, LiteRtTensorTypeId type_id) {
  if (tensor == nullptr) {
    return {};
  }
  if (type_id == kLiteRtRankedTensorType) {
    LiteRtRankedTensorType ranked_tensor_type;
    internal::AssertOk(LiteRtGetRankedTensorType, tensor, &ranked_tensor_type);
    return litert::RankedTensorType(ranked_tensor_type);
  } else {
    LiteRtUnrankedTensorType unranked_tensor_type;
    internal::AssertOk(LiteRtGetUnrankedTensorType, tensor,
                       &unranked_tensor_type);
    return unranked_tensor_type;
  }
}

inline QuantizationTypeId FetchTensorQuantizationTypeId(LiteRtTensor tensor) {
  if (tensor == nullptr) {
    return QuantizationTypeId::None;
  }
  LiteRtQuantizationTypeId quantization_type_id;
  internal::AssertOk(LiteRtGetQuantizationTypeId, tensor,
                     &quantization_type_id);
  return static_cast<QuantizationTypeId>(quantization_type_id);
}

inline QuantizationPerTensor FetchTensorQuantizationPerTensor(
    LiteRtTensor tensor) {
  if (FetchTensorQuantizationTypeId(tensor) != QuantizationTypeId::PerTensor) {
    return {};
  }
  QuantizationPerTensor per_tensor_quantization;
  internal::AssertOk(LiteRtGetPerTensorQuantization, tensor,
                     &per_tensor_quantization);
  return per_tensor_quantization;
}

inline QuantizationPerChannel FetchTensorQuantizationPerChannel(
    LiteRtTensor tensor) {
  if (FetchTensorQuantizationTypeId(tensor) != QuantizationTypeId::PerChannel) {
    return {};
  }
  QuantizationPerChannel per_channel_quantization;
  internal::AssertOk(LiteRtGetPerChannelQuantization, tensor,
                     &per_channel_quantization);
  return per_channel_quantization;
}

inline QuantizationBlockWise FetchTensorQuantizationBlockWise(
    LiteRtTensor tensor) {
  if (FetchTensorQuantizationTypeId(tensor) != QuantizationTypeId::BlockWise) {
    return {};
  }
  QuantizationBlockWise block_wise_quantization;
  internal::AssertOk(LiteRtGetBlockWiseQuantization, tensor,
                     &block_wise_quantization);
  return block_wise_quantization;
}

}  // namespace internal::model_detail

/// @brief A C++ wrapper for `LiteRtTensor`, representing a tensor in the model.
class Tensor : public internal::NonOwnedHandle<LiteRtTensor>,
               public litert::SimpleTensor {
 public:
  explicit Tensor(LiteRtTensor tensor)
      : internal::NonOwnedHandle<LiteRtTensor>(tensor),
        litert::SimpleTensor(
            internal::model_detail::FetchTensorIndex(tensor),
            internal::model_detail::FetchTensorName(tensor),
            internal::model_detail::FetchTensorTypeId(tensor),
            internal::model_detail::FetchTensorType(
                tensor, internal::model_detail::FetchTensorTypeId(tensor)),
            internal::model_detail::FetchTensorQuantizationTypeId(tensor),
            internal::model_detail::FetchTensorQuantizationPerTensor(tensor),
            internal::model_detail::FetchTensorQuantizationPerChannel(tensor),
            internal::model_detail::FetchTensorQuantizationBlockWise(tensor)) {}

  // Allow copying Tensors.
  Tensor(const Tensor& other)
      : internal::NonOwnedHandle<LiteRtTensor>(other.Get()),
        litert::SimpleTensor(other) {}
  Tensor(Tensor&&) = default;
  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      *this = Tensor(other);
    }
    return *this;
  }
  Tensor& operator=(Tensor&&) = default;

  // We need to keep these functions in sync with the SimpleTensor class. These
  // functions are used when Tensor is used as a live handle, reflecting the
  // most up-to-date state of the underlying LiteRtTensor.
  QuantizationTypeId QTypeId() const {
    LiteRtQuantizationTypeId q_type_id;
    internal::AssertOk(LiteRtGetQuantizationTypeId, Get(), &q_type_id);
    return static_cast<QuantizationTypeId>(q_type_id);
  }

  bool HasQuantization() const { return QTypeId() != QuantizationTypeId::None; }

  QuantizationPerTensor PerTensorQuantization() const {
    QuantizationPerTensor per_tensor_quantization;
    internal::AssertOk(LiteRtGetPerTensorQuantization, Get(),
                       &per_tensor_quantization);
    return per_tensor_quantization;
  }

  QuantizationPerChannel PerChannelQuantization() const {
    QuantizationPerChannel per_channel_quantization;
    internal::AssertOk(LiteRtGetPerChannelQuantization, Get(),
                       &per_channel_quantization);
    return per_channel_quantization;
  }

  QuantizationBlockWise BlockWiseQuantization() const {
    QuantizationBlockWise block_wise_quantization;
    internal::AssertOk(LiteRtGetBlockWiseQuantization, Get(),
                       &block_wise_quantization);
    return block_wise_quantization;
  }

  bool HasWeights() const {
    auto weights = Weights();
    return !weights.Bytes().empty();
  }

  Weights Weights() const {
    LiteRtWeights weights;
    internal::AssertOk(LiteRtGetTensorWeights, Get(), &weights);
    return litert::Weights(weights);
  }

  struct TensorUse;
  using TensorUses =
      absl::InlinedVector<TensorUse, kExpectedMaxNumOfTensorUses>;

  TensorUses Uses() const;

  template <typename T>
  Expected<absl::Span<const T>> WeightsData() const {
    auto ranked_tensor_type = RankedTensorType();
    if (!ranked_tensor_type) {
      return ranked_tensor_type.Error();
    }

    const enum ElementType ty = ranked_tensor_type->ElementType();
    if (ty != GetElementType<T>()) {
      return Unexpected(Status::kErrorInvalidArgument);
    }

    if (!HasWeights()) {
      return Unexpected(Status::kErrorInvalidArgument);
    }
    const absl::Span<const uint8_t> weights = Weights().Bytes();

    auto num_elements = ranked_tensor_type->Layout().NumElements();
    if (!num_elements) {
      return num_elements.Error();
    }
    auto byte_width = GetByteWidth(ty);
    if (!byte_width.has_value()) {
      return Unexpected(Status::kErrorInvalidArgument);
    }

    if (byte_width.value() * *num_elements != weights.size()) {
      return Unexpected(Status::kErrorInvalidArgument);
    }

    return absl::MakeConstSpan(reinterpret_cast<const T*>(weights.data()),
                               *num_elements);
  }

  std::optional<LiteRtTensorDefiningOp> DefiningOp() const {
    bool has_defining_op;
    LiteRtTensorDefiningOp defining_op;
    internal::AssertOk(LiteRtGetTensorDefiningOp, Get(), &has_defining_op,
                       &defining_op);
    if (has_defining_op) {
      return defining_op;
    }
    return std::nullopt;
  }

  /// @brief Gets the defining op of the tensor.
  /// @return The defining op of the tensor if it exists, otherwise an error.
  Expected<Op> GetDefiningOp() const;

  bool IsSubgraphInput() const {
    LITERT_ASSIGN_OR_ABORT(auto ranked_tensor_type, RankedTensorType());
    if (ranked_tensor_type.Layout().Rank() == 1 &&
        ranked_tensor_type.Layout().Dimensions()[0] == 0) {
      return false;
    }
    return !HasWeights() && !DefiningOp().has_value();
  }
  bool IsConstant() const { return HasWeights() && !DefiningOp().has_value(); }

  /// @brief Compares two tensors for equality.
  /// @param other The other tensor to compare with.
  /// @return True if the tensors are the same, false otherwise.
  bool operator==(const Tensor& other) const { return Get() == other.Get(); }

  /// @brief Compares two tensors for inequality.
  /// @param other The other tensor to compare with.
  /// @return True if the tensors are different, false otherwise.
  bool operator!=(const Tensor& other) const { return Get() != other.Get(); }
};

namespace internal::model_detail {

inline absl::string_view FetchSignatureKey(LiteRtSignature signature) {
  const char* key;
  internal::AssertOk(LiteRtGetSignatureKey, signature, &key);
  return key;
}

inline std::vector<absl::string_view> FetchSignatureInputNames(
    LiteRtSignature signature) {
  LiteRtParamIndex num_inputs;
  internal::AssertOk(LiteRtGetNumSignatureInputs, signature, &num_inputs);
  std::vector<absl::string_view> input_names;
  input_names.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const char* name;
    internal::AssertOk(LiteRtGetSignatureInputName, signature, i, &name);
    input_names.push_back(name);
  }
  return input_names;
}

inline std::vector<absl::string_view> FetchSignatureOutputNames(
    LiteRtSignature signature) {
  LiteRtParamIndex num_outputs;
  internal::AssertOk(LiteRtGetNumSignatureOutputs, signature, &num_outputs);
  std::vector<absl::string_view> output_names;
  output_names.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    const char* name;
    internal::AssertOk(LiteRtGetSignatureOutputName, signature, i, &name);
    output_names.push_back(name);
  }
  return output_names;
}

inline std::vector<std::unique_ptr<SimpleTensor>> FetchSignatureInputTensors(
    LiteRtSignature signature) {
  LiteRtParamIndex num_inputs;
  internal::AssertOk(LiteRtGetNumSignatureInputs, signature, &num_inputs);
  std::vector<std::unique_ptr<SimpleTensor>> input_tensors;
  input_tensors.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    LiteRtTensor tensor;
    internal::AssertOk(LiteRtGetSignatureInputTensorByIndex, signature, i,
                       &tensor);
    input_tensors.push_back(std::make_unique<Tensor>(tensor));
  }
  return input_tensors;
}

inline std::vector<std::unique_ptr<SimpleTensor>> FetchSignatureOutputTensors(
    LiteRtSignature signature) {
  LiteRtParamIndex num_outputs;
  internal::AssertOk(LiteRtGetNumSignatureOutputs, signature, &num_outputs);
  std::vector<std::unique_ptr<SimpleTensor>> output_tensors;
  output_tensors.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    LiteRtTensor tensor;
    internal::AssertOk(LiteRtGetSignatureOutputTensorByIndex, signature, i,
                       &tensor);
    output_tensors.push_back(std::make_unique<Tensor>(tensor));
  }
  return output_tensors;
}

inline std::unique_ptr<SimpleTensor> FetchSignatureOutputTensorByIndex(
    LiteRtSignature signature, LiteRtParamIndex output_index) {
  LiteRtTensor tensor;
  internal::AssertOk(LiteRtGetSignatureOutputTensorByIndex, signature,
                     output_index, &tensor);
  return std::make_unique<Tensor>(tensor);
}

}  // namespace internal::model_detail

using OpInputs = absl::InlinedVector<Tensor, kExpectedMaxNumOfOpInputs>;
using OpOutputs = absl::InlinedVector<Tensor, kExpectedMaxNumOfOpOutputs>;

/// @brief A C++ wrapper for `LiteRtOp`, representing an operator in the model.
class Op : public internal::NonOwnedHandle<LiteRtOp> {
 public:
  explicit Op(LiteRtOp op) : NonOwnedHandle<LiteRtOp>(op) {}

  // Allow copying Ops (they are just handles).
  Op(const Op& other) : NonOwnedHandle<LiteRtOp>(other.Get()) {}
  Op(Op&&) = default;
  Op& operator=(Op&&) = default;

  LiteRtOpCode Code() const {
    LiteRtOpCode opcode;
    internal::AssertOk(LiteRtGetOpCode, Get(), &opcode);
    return opcode;
  }

  /// @brief Gets the custom code.
  /// @return The custom code if the op is a custom op, otherwise an error.
  Expected<absl::string_view> CustomCode() const {
    const char* custom_code;
    auto stat = LiteRtGetCustomCode(Get(), &custom_code);
    if (stat != kLiteRtStatusOk) {
      return Error(stat, "Failed to get custom code");
    }
    return absl::string_view(custom_code);
  }

  /// @brief Gets the custom options.
  /// @return The custom options if present, otherwise an error.
  Expected<absl::Span<const uint8_t>> CustomOptions() const {
    const uint8_t* options = nullptr;
    int32_t options_bytes = 0;
    auto stat = LiteRtGetCustomOptions(Get(), &options, &options_bytes);
    if (stat != kLiteRtStatusOk) {
      return Error(stat, "Failed to get custom options");
    }
    return absl::MakeConstSpan(options, options_bytes);
  }

  OpInputs Inputs() const {
    LiteRtParamIndex num_inputs;
    internal::AssertOk(LiteRtGetNumOpInputs, Get(), &num_inputs);

    OpInputs inputs;
    for (auto i = 0; i < num_inputs; ++i) {
      LiteRtTensor input;
      internal::AssertOk(LiteRtGetOpInput, Get(), i, &input);
      inputs.emplace_back(Tensor(input));
    }
    return inputs;
  }
  OpOutputs Outputs() const {
    LiteRtParamIndex num_outputs;
    internal::AssertOk(LiteRtGetNumOpOutputs, Get(), &num_outputs);

    OpOutputs outputs;
    for (auto i = 0; i < num_outputs; ++i) {
      LiteRtTensor output;
      internal::AssertOk(LiteRtGetOpOutput, Get(), i, &output);
      outputs.emplace_back(Tensor(output));
    }
    return outputs;
  }

  /// @brief Checks if the op has the given opcode.
  /// @param code The opcode to check.
  /// @return True if the op has the given opcode, false otherwise.
  bool Is(LiteRtOpCode code) const { return Code() == code; }

  /// @brief Gets the input tensor at the given index.
  /// @param index The index of the input tensor.
  /// @return The input tensor at the given index if it exists, otherwise an
  /// error.
  Expected<Tensor> Input(size_t index) const {
    LiteRtParamIndex num_inputs;
    LITERT_RETURN_IF_ERROR(LiteRtGetNumOpInputs(Get(), &num_inputs));
    if (index >= num_inputs) {
      return Unexpected(Status::kErrorIndexOOB);
    }
    LiteRtTensor input;
    LITERT_RETURN_IF_ERROR(LiteRtGetOpInput(Get(), index, &input));
    return Tensor(input);
  }

  /// @brief Gets the output tensor at the given index.
  /// @param index The index of the output tensor.
  /// @return The output tensor at the given index if it exists, otherwise an
  /// error.
  Expected<Tensor> Output(size_t index) const {
    LiteRtParamIndex num_outputs;
    LITERT_RETURN_IF_ERROR(LiteRtGetNumOpOutputs(Get(), &num_outputs));
    if (index >= num_outputs) {
      return Unexpected(Status::kErrorIndexOOB);
    }
    LiteRtTensor output;
    LITERT_RETURN_IF_ERROR(LiteRtGetOpOutput(Get(), index, &output));
    return Tensor(output);
  }

  /// @brief Gets the defining op of the input tensor at the given index.
  /// @param index The index of the input tensor.
  /// @return The defining op of the input tensor if it exists, otherwise an
  /// error.
  Expected<Op> InputDefiningOp(size_t index) const;
};

struct Tensor::TensorUse {
  Op user;
  LiteRtParamIndex user_arg_ind;
};

inline Tensor::TensorUses Tensor::Uses() const {
  LiteRtParamIndex num_uses;
  internal::AssertOk(LiteRtGetNumTensorUses, Get(), &num_uses);

  TensorUses uses;
  for (auto i = 0; i < num_uses; ++i) {
    LiteRtOp user;
    LiteRtParamIndex user_arg_index;
    internal::AssertOk(LiteRtGetTensorUse, Get(), i, &user, &user_arg_index);
    uses.emplace_back(TensorUse{Op(user), user_arg_index});
  }
  return uses;
}

inline Expected<Op> Tensor::GetDefiningOp() const {
  bool has_defining_op;
  LiteRtTensorDefiningOp defining_op;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorDefiningOp(Get(), &has_defining_op, &defining_op));
  if (!has_defining_op) {
    return Unexpected(Status::kErrorNotFound);
  }
  return Op(defining_op.op);
}

using SubgraphInputs =
    absl::InlinedVector<Tensor, kExpectedMaxNumOfSubgraphInputs>;
using SubgraphOutputs =
    absl::InlinedVector<Tensor, kExpectedMaxNumOfSubgraphOutputs>;

/// @brief A C++ wrapper for `LiteRtSubgraph`, representing a subgraph in the
/// model.
class Subgraph : public internal::NonOwnedHandle<LiteRtSubgraph> {
 public:
  explicit Subgraph(LiteRtSubgraph subgraph)
      : internal::NonOwnedHandle<LiteRtSubgraph>(subgraph) {}

  SubgraphInputs Inputs() const {
    LiteRtParamIndex num_inputs;
    internal::AssertOk(LiteRtGetNumSubgraphInputs, Get(), &num_inputs);

    SubgraphInputs inputs;
    for (auto i = 0; i < num_inputs; ++i) {
      LiteRtTensor input;
      internal::AssertOk(LiteRtGetSubgraphInput, Get(), i, &input);
      inputs.emplace_back(Tensor(input));
    }
    return inputs;
  }
  SubgraphOutputs Outputs() const {
    LiteRtParamIndex num_outputs;
    internal::AssertOk(LiteRtGetNumSubgraphOutputs, Get(), &num_outputs);

    SubgraphOutputs outputs;
    for (auto i = 0; i < num_outputs; ++i) {
      LiteRtTensor output;
      internal::AssertOk(LiteRtGetSubgraphOutput, Get(), i, &output);
      outputs.emplace_back(Tensor(output));
    }
    return outputs;
  }
  std::vector<Op> Ops() const {
    LiteRtParamIndex num_ops;
    internal::AssertOk(LiteRtGetNumSubgraphOps, Get(), &num_ops);

    std::vector<Op> ops;
    for (auto i = 0; i < num_ops; ++i) {
      LiteRtOp op;
      internal::AssertOk(LiteRtGetSubgraphOp, Get(), i, &op);
      ops.emplace_back(Op(op));
    }
    return ops;
  }

  /// @brief Returns the input tensor with the given input signature name.
  Expected<Tensor> Input(absl::string_view name) const {
    LiteRtParamIndex num_inputs;
    internal::AssertOk(LiteRtGetNumSubgraphInputs, Get(), &num_inputs);

    for (auto i = 0; i < num_inputs; ++i) {
      LiteRtTensor input;
      internal::AssertOk(LiteRtGetSubgraphInput, Get(), i, &input);
      const char* input_name;
      internal::AssertOk(LiteRtGetTensorName, input, &input_name);
      if (name == input_name) {
        return Tensor(input);
      }
    }
    return Unexpected(Status::kErrorNotFound,
                      "Failed to find input: " + std::string(name));
  }

  /// @brief Returns the output tensor with the given output signature name.
  Expected<Tensor> Output(absl::string_view name) const {
    LiteRtParamIndex num_outputs;
    internal::AssertOk(LiteRtGetNumSubgraphOutputs, Get(), &num_outputs);

    for (auto i = 0; i < num_outputs; ++i) {
      LiteRtTensor output;
      internal::AssertOk(LiteRtGetSubgraphOutput, Get(), i, &output);
      const char* output_name;
      internal::AssertOk(LiteRtGetTensorName, output, &output_name);
      if (name == output_name) {
        return Tensor(output);
      }
    }
    return Unexpected(Status::kErrorNotFound,
                      "Failed to find output: " + std::string(name));
  }
};

/// @brief A C++ wrapper for `LiteRtSignature`, representing a model signature.
class Signature : public internal::NonOwnedHandle<LiteRtSignature>,
                  public litert::SimpleSignature {
 public:
  explicit Signature(LiteRtSignature signature)
      : internal::NonOwnedHandle<LiteRtSignature>(signature),
        litert::SimpleSignature(
            internal::model_detail::FetchSignatureKey(signature),
            internal::model_detail::FetchSignatureInputNames(signature),
            internal::model_detail::FetchSignatureInputTensors(signature),
            internal::model_detail::FetchSignatureOutputNames(signature),
            internal::model_detail::FetchSignatureOutputTensors(signature)) {}

  Signature(Signature&&) = default;
  Signature& operator=(Signature&&) = default;

  LiteRtSubgraph Subgraph() const {
    LiteRtSubgraph subgraph;
    internal::AssertOk(LiteRtGetSignatureSubgraph, Get(), &subgraph);
    return subgraph;
  }
};

/// @brief An internal C++ wrapper for `LiteRtModel`, representing a LiteRT
/// model with full graph introspection and manipulation capabilities.
class Model : public internal::BaseHandle<LiteRtModel> {
 public:
  Model() = default;

  static Model CreateFromOwnedHandle(LiteRtModel model) {
    return Model(model, OwnHandle::kYes);
  }

  static Model CreateFromNonOwnedHandle(LiteRtModel model) {
    return Model(model, OwnHandle::kNo);
  }

  /// @brief Creates a model from a file.
  static Expected<Model> CreateFromFile(Environment& env,
                                        const std::string& filename) {
    LiteRtModel model;
    if (auto status =
            LiteRtCreateModelFromFile(env.Get(), filename.c_str(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from file");
    }
    return CreateFromOwnedHandle(model);
  }

  /// @brief Creates a model from a file.
  /// @deprecated Use API with explicit Environment instead.
  static Expected<Model> CreateFromFile(const std::string& filename) {
    const auto& env = GetDefaultEnvironment();
    LiteRtModel model;
    if (auto status =
            LiteRtCreateModelFromFile(env->Get(), filename.c_str(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from file");
    }
    return CreateFromOwnedHandle(model);
  }

  /// @brief Creates a model from a buffer.
  ///
  /// The caller must ensure that the buffer remains valid for the lifetime of
  /// the model.
  static Expected<Model> CreateFromBuffer(Environment& env,
                                          BufferRef<uint8_t> buffer) {
    LiteRtModel model;
    if (auto status = LiteRtCreateModelFromBuffer(env.Get(), buffer.Data(),
                                                  buffer.Size(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from buffer");
    }
    return CreateFromOwnedHandle(model);
  }

  /// @brief Creates a model from a buffer.
  ///
  /// The caller must ensure that the buffer remains valid for the lifetime of
  /// the model.
  /// @deprecated Use API with explicit Environment instead.
  static Expected<Model> CreateFromBuffer(BufferRef<uint8_t> buffer) {
    const auto& env = GetDefaultEnvironment();
    LiteRtModel model;
    if (auto status = LiteRtCreateModelFromBuffer(env->Get(), buffer.Data(),
                                                  buffer.Size(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from buffer");
    }
    return CreateFromOwnedHandle(model);
  }

  /// @brief Creates a model from a file descriptor region.
  ///
  /// LiteRT duplicates the file descriptor internally; the caller retains
  /// ownership of `fd`.
  static Expected<Model> CreateFromFd(Environment& env, int fd, size_t offset,
                                      size_t size) {
    LiteRtModel model;
    if (auto status =
            LiteRtCreateModelFromFd(env.Get(), fd, offset, size, &model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from fd");
    }
    return CreateFromOwnedHandle(model);
  }

  /// @brief Creates a model from a file descriptor region.
  ///
  /// LiteRT duplicates the file descriptor internally; the caller retains
  /// ownership of `fd`.
  /// @deprecated Use API with explicit Environment instead.
  static Expected<Model> CreateFromFd(int fd, size_t offset, size_t size) {
    const auto& env = GetDefaultEnvironment();
    LiteRtModel model;
    if (auto status =
            LiteRtCreateModelFromFd(env->Get(), fd, offset, size, &model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from fd");
    }
    return CreateFromOwnedHandle(model);
  }

#if !defined(LITERT_DYNAMIC_RUNTIME)
  // copybara:uncomment_begin(google_only)
  // /// @internal
  // /// @brief Creates a model from an owned TFLite allocation.
  // ///
  // /// LiteRT takes ownership of the allocation wrapper.
  // static Expected<Model> CreateFromAllocation(
      // Environment& env, std::unique_ptr<tflite::Allocation> allocation) {
    // LiteRtModel model;
    // LiteRtAllocation c_allocation = allocation.release();
    // if (auto status =
            // LiteRtCreateModelFromAllocation(env.Get(), c_allocation, &model);
        // status != kLiteRtStatusOk) {
      // return Unexpected(ToStatus(status),
                        // "Failed to load model from allocation");
    // }
    // return CreateFromOwnedHandle(model);
  // }
// 
  // /// @internal
  // /// @brief Creates a model from an owned TFLite allocation.
  // ///
  // /// LiteRT takes ownership of the allocation wrapper.
  // /// @deprecated Use API with explicit Environment instead.
  // static Expected<Model> CreateFromAllocation(
      // std::unique_ptr<tflite::Allocation> allocation) {
    // LiteRtModel model;
    // LiteRtAllocation c_allocation = allocation.release();
    // if (auto status = LiteRtCreateModelFromAllocation(
            // /*environment=*/nullptr, c_allocation, &model);
        // status != kLiteRtStatusOk) {
      // return Unexpected(ToStatus(status),
                        // "Failed to load model from allocation");
    // }
    // return CreateFromOwnedHandle(model);
  // }
  // copybara:uncomment_end
#endif  // !defined(LITERT_DYNAMIC_RUNTIME)

  Expected<absl::Span<const uint8_t>> Metadata(
      const std::string& metadata_key) const {
    const void* buffer;
    size_t buffer_size;
    if (LiteRtGetModelMetadata(Get(), metadata_key.data(), &buffer,
                               &buffer_size) != kLiteRtStatusOk) {
      return Unexpected(Status::kErrorNotFound, "Metadata key not found");
    }
    return absl::MakeSpan(static_cast<const uint8_t*>(buffer), buffer_size);
  }

  Expected<void> AddMetadata(const std::string& metadata_key,
                             const std::string& metadata_data) {
    LiteRtStatus status = LiteRtAddModelMetadata(
        Get(), metadata_key.data(), metadata_data.data(), metadata_data.size());
    if (status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to add metadata");
    }
    return {};
  }

  Expected<Subgraph> MainSubgraph() const {
    LiteRtParamIndex main_subgraph_index;
    internal::AssertOk(LiteRtGetMainModelSubgraphIndex, Get(),
                       &main_subgraph_index);
    return this->Subgraph(main_subgraph_index);
  }

  size_t NumSubgraphs() const {
    LiteRtParamIndex num_subgraphs;
    internal::AssertOk(LiteRtGetNumModelSubgraphs, Get(), &num_subgraphs);
    return num_subgraphs;
  }

  Expected<class Subgraph> Subgraph(size_t subgraph_index) const {
    LiteRtSubgraph subgraph;
    if (LiteRtGetModelSubgraph(Get(), subgraph_index, &subgraph) !=
        kLiteRtStatusOk) {
      return Unexpected(Status::kErrorNotFound, "Subgraph not found");
    }
    return litert::Subgraph(subgraph);
  }

  Expected<class Subgraph> Subgraph(absl::string_view signature_key) const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      auto key = internal::model_detail::FetchSignatureKey(lite_rt_signature);
      if (key == signature_key) {
        LiteRtSubgraph subgraph;
        if (LiteRtGetSignatureSubgraph(lite_rt_signature, &subgraph) !=
            kLiteRtStatusOk) {
          return Unexpected(Status::kErrorNotFound, "Subgraph not found");
        }
        return litert::Subgraph(subgraph);
      }
    }
    return Unexpected(Status::kErrorNotFound,
                      "Signature not found: " + std::string(signature_key));
  }

  size_t GetNumSignatures() const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    return num_signatures;
  }

  /// @brief Returns the list of signatures defined in the model.
  // TODO: Return std::vector<Signature> once LiteRT-LM migrates to
  // cc/internal/litert_model.h (see ExtendedModel).
  Expected<std::vector<SimpleSignature>> GetSignatures() const {
    size_t num_signatures = GetNumSignatures();
    std::vector<SimpleSignature> signatures;
    signatures.reserve(num_signatures);
    for (size_t i = 0; i < num_signatures; ++i) {
      LITERT_ASSIGN_OR_RETURN(auto signature, GetSignature(i));
      signatures.push_back(std::move(signature));
    }
    return std::move(signatures);
  }

  /// @brief Returns the list of signature key names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureKeys() const {
    size_t num_signatures = GetNumSignatures();
    std::vector<absl::string_view> signature_keys;
    signature_keys.reserve(num_signatures);
    for (size_t i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      signature_keys.push_back(
          internal::model_detail::FetchSignatureKey(lite_rt_signature));
    }
    return signature_keys;
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    return internal::model_detail::FetchSignatureInputNames(lite_rt_signature);
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames() const {
    return GetSignatureInputNames(/*signature_index=*/0);
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      absl::string_view signature_key) const {
    LITERT_ASSIGN_OR_RETURN(auto signature, FindSignature(signature_key));
    return signature.InputNames();
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    return internal::model_detail::FetchSignatureOutputNames(lite_rt_signature);
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames() const {
    return GetSignatureOutputNames(/*signature_index=*/0);
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      absl::string_view signature_key) const {
    LITERT_ASSIGN_OR_RETURN(auto signature, FindSignature(signature_key));
    return signature.OutputNames();
  }

  /// @brief Returns the signature at the given index.
  Expected<Signature> GetSignature(size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    if (auto status =
            LiteRtGetModelSignature(Get(), signature_index, &lite_rt_signature);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to get signature");
    }
    return Signature(lite_rt_signature);
  }

  /// @brief Returns the signature index for the given signature key.
  ///
  /// Returns 0 if the signature key is empty.
  Expected<size_t> GetSignatureIndex(absl::string_view signature_key) const {
    if (signature_key.empty()) {
      return 0;
    }
    size_t num_signatures = GetNumSignatures();
    for (size_t i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      auto key = internal::model_detail::FetchSignatureKey(lite_rt_signature);
      if (key == signature_key) {
        return i;
      }
    }
    return Unexpected(Status::kErrorNotFound,
                      "Signature not found: " + std::string(signature_key));
  }

  /// @brief Returns the `Signature` object for the given signature key.
  ///
  /// Returns the default signature if the signature key is empty.
  Expected<Signature> FindSignature(absl::string_view signature_key) const {
    LITERT_ASSIGN_OR_RETURN(auto signature_index,
                            GetSignatureIndex(signature_key));
    return GetSignature(signature_index);
  }

  static absl::string_view DefaultSignatureKey() {
    return kDefaultSignatureKey;
  }

  /// @brief Returns the tensor type for the n-th input tensor.
  Expected<RankedTensorType> GetInputTensorType(size_t signature_index,
                                                size_t input_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    auto input_tensors =
        internal::model_detail::FetchSignatureInputTensors(lite_rt_signature);
    if (input_index >= input_tensors.size()) {
      return Error(Status::kErrorInvalidArgument, "Input index out of bounds");
    }
    return input_tensors[input_index]->RankedTensorType();
  }

  /// @brief Returns the tensor type for the given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      size_t signature_index, absl::string_view input_name) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    auto input_tensors =
        internal::model_detail::FetchSignatureInputTensors(lite_rt_signature);
    LiteRtParamIndex num_inputs;
    internal::AssertOk(LiteRtGetNumSignatureInputs, lite_rt_signature,
                       &num_inputs);
    for (LiteRtParamIndex i = 0; i < num_inputs; ++i) {
      const char* name;
      internal::AssertOk(LiteRtGetSignatureInputName, lite_rt_signature, i,
                         &name);
      if (absl::string_view(name) == input_name) {
        return input_tensors[i]->RankedTensorType();
      }
    }
    // Keep accepting graph tensor names for callers that used that behavior.
    for (const auto& input_tensor : input_tensors) {
      if (input_tensor->Name() == input_name) {
        return input_tensor->RankedTensorType();
      }
    }
    return Error(Status::kErrorNotFound,
                 "Input tensor not found: " + std::string(input_name));
  }

  /// @brief Returns the tensor type for the given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      absl::string_view signature_key, absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(auto index, GetSignatureIndex(signature_key));
    return GetInputTensorType(index, input_name);
  }

  /// @brief Gets the input tensor type of the default signature for a given
  /// input name.
  Expected<RankedTensorType> GetInputTensorType(
      absl::string_view input_name) const {
    return GetInputTensorType(/*signature_index=*/0, input_name);
  }

  /// @brief Returns the tensor type for the n-th output tensor.
  Expected<RankedTensorType> GetOutputTensorType(size_t signature_index,
                                                 size_t output_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    LiteRtParamIndex num_outputs;
    internal::AssertOk(LiteRtGetNumSignatureOutputs, lite_rt_signature,
                       &num_outputs);
    if (output_index >= static_cast<size_t>(num_outputs)) {
      return Error(Status::kErrorInvalidArgument, "Output index out of bounds");
    }
    return internal::model_detail::FetchSignatureOutputTensorByIndex(
               lite_rt_signature, static_cast<LiteRtParamIndex>(output_index))
        ->RankedTensorType();
  }

  /// @brief Returns the tensor type for the given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      size_t signature_index, absl::string_view output_name) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    LiteRtParamIndex num_outputs;
    internal::AssertOk(LiteRtGetNumSignatureOutputs, lite_rt_signature,
                       &num_outputs);
    for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
      const char* name;
      internal::AssertOk(LiteRtGetSignatureOutputName, lite_rt_signature, i,
                         &name);
      if (absl::string_view(name) == output_name) {
        return internal::model_detail::FetchSignatureOutputTensorByIndex(
                   lite_rt_signature, i)
            ->RankedTensorType();
      }
    }
    return Error(Status::kErrorNotFound,
                 "Output tensor not found: " + std::string(output_name));
  }

  /// @brief Returns the tensor type for the given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view signature_key, absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(auto index, GetSignatureIndex(signature_key));
    return GetOutputTensorType(index, output_name);
  }

  /// @brief Gets the output tensor type of the default signature for a given
  /// output name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view output_name) const {
    return GetOutputTensorType(/*signature_index=*/0, output_name);
  }

  /// @brief Serializes a model to a buffer.
  ///
  /// The model is released after serialization.
  static Expected<OwningBufferRef<uint8_t>> Serialize(
      Model&& model, const LiteRtModelSerializationOptions& options) {
    OwningBufferRef<uint8_t> buf;
    auto [data, size, offset] = buf.GetWeak();

    LITERT_RETURN_IF_ERROR(LiteRtSerializeModel(
        std::move(model.Release()), &data, &size, &offset, true, options));
    return std::move(buf);
  }

 protected:
  /// @param owned Indicates if the created `Model` object should take
  /// ownership of the provided `model` handle.
  Model(LiteRtModel model, OwnHandle owned)
      : internal::BaseHandle<LiteRtModel>(model, LiteRtDestroyModel, owned) {}

  // This is only used for managing the lifetime of ad-hoc environment for
  // legacy cases.
  /// @deprecated Do not use this field.
  static const Expected<Environment>& GetDefaultEnvironment() {
    static const Expected<Environment>* kDefaultEnvironment =
        new Expected<Environment>(Environment::Create({}));
    return *kDefaultEnvironment;
  }
};

/// @brief Transitional subclass for `Model` preserving `GetSignatures()` ->
/// `Expected<std::vector<Signature>>` while downstream repositories migrate.
/// @deprecated Use `litert::Model` instead.
class [[deprecated("Use litert::Model instead.")]] ExtendedModel
    : public Model {
 public:
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  ExtendedModel() = default;
  // NOLINTNEXTLINE(*-explicit-constructor)
  ExtendedModel(Model&& model) noexcept : Model(std::move(model)) {}

  static ExtendedModel CreateFromOwnedHandle(LiteRtModel model) {
    return ExtendedModel(model, OwnHandle::kYes);
  }

  static ExtendedModel CreateFromNonOwnedHandle(LiteRtModel model) {
    return ExtendedModel(model, OwnHandle::kNo);
  }

  static Expected<ExtendedModel> CreateFromFile(Environment& env,
                                                const std::string& filename) {
    LITERT_ASSIGN_OR_RETURN(Model model, Model::CreateFromFile(env, filename));
    return ExtendedModel(std::move(model));
  }

  /// @deprecated Use API with explicit Environment instead.
  static Expected<ExtendedModel> CreateFromFile(const std::string& filename) {
    LITERT_ASSIGN_OR_RETURN(Model model, Model::CreateFromFile(filename));
    return ExtendedModel(std::move(model));
  }

  static Expected<ExtendedModel> CreateFromBuffer(Environment& env,
                                                  BufferRef<uint8_t> buffer) {
    LITERT_ASSIGN_OR_RETURN(Model model, Model::CreateFromBuffer(env, buffer));
    return ExtendedModel(std::move(model));
  }

  /// @deprecated Use API with explicit Environment instead.
  static Expected<ExtendedModel> CreateFromBuffer(BufferRef<uint8_t> buffer) {
    LITERT_ASSIGN_OR_RETURN(Model model, Model::CreateFromBuffer(buffer));
    return ExtendedModel(std::move(model));
  }

  static Expected<ExtendedModel> CreateFromFd(Environment& env, int fd,
                                              size_t offset, size_t size) {
    LITERT_ASSIGN_OR_RETURN(Model model,
                            Model::CreateFromFd(env, fd, offset, size));
    return ExtendedModel(std::move(model));
  }

  /// @deprecated Use API with explicit Environment instead.
  static Expected<ExtendedModel> CreateFromFd(int fd, size_t offset,
                                              size_t size) {
    LITERT_ASSIGN_OR_RETURN(Model model, Model::CreateFromFd(fd, offset, size));
    return ExtendedModel(std::move(model));
  }

#if !defined(LITERT_DYNAMIC_RUNTIME)
  // copybara:uncomment_begin(google_only)
  // static Expected<ExtendedModel> CreateFromAllocation(
      // Environment& env, std::unique_ptr<tflite::Allocation> allocation) {
    // LITERT_ASSIGN_OR_RETURN(
        // Model model, Model::CreateFromAllocation(env, std::move(allocation)));
    // return ExtendedModel(std::move(model));
  // }
// 
  // /// @deprecated Use API with explicit Environment instead.
  // static Expected<ExtendedModel> CreateFromAllocation(
      // std::unique_ptr<tflite::Allocation> allocation) {
    // LITERT_ASSIGN_OR_RETURN(Model model,
                            // Model::CreateFromAllocation(std::move(allocation)));
    // return ExtendedModel(std::move(model));
  // }
  // copybara:uncomment_end
#endif  // !defined(LITERT_DYNAMIC_RUNTIME)

  Expected<std::vector<Signature>> GetSignatures() const {
    size_t num_signatures = GetNumSignatures();
    std::vector<Signature> signatures;
    signatures.reserve(num_signatures);
    for (size_t i = 0; i < num_signatures; ++i) {
      LITERT_ASSIGN_OR_RETURN(auto signature, GetSignature(i));
      signatures.push_back(std::move(signature));
    }
    return std::move(signatures);
  }

 private:
  ExtendedModel(LiteRtModel model, OwnHandle owned) : Model(model, owned) {}
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
};

struct SerializationOptions {
  static LiteRtModelSerializationOptions Defaults() {
    LiteRtModelSerializationOptions opts{};
    opts.bytecode_alignment = 1;
    return opts;
  }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_LITERT_MODEL_H_
