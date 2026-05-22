// Copyright 2025 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CC_INTERNAL_LITERT_EXTENDED_MODEL_H_
#define ODML_LITERT_LITERT_CC_INTERNAL_LITERT_EXTENDED_MODEL_H_

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
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_ranked_tensor_type.h"

/// @file
/// @brief Defines extended C++ wrappers for the LiteRT model components,
/// providing more detailed introspection and manipulation capabilities.

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

namespace internal::extended_model_detail {

inline absl::string_view FetchExtendedTensorName(LiteRtTensor tensor) {
  if (tensor == nullptr) {
    return "";
  }
  const char* name;
  internal::AssertOk(LiteRtGetTensorName, tensor, &name);
  return name;
}

inline std::uint32_t FetchExtendedTensorIndex(LiteRtTensor tensor) {
  if (tensor == nullptr) {
    return 0;
  }
  std::uint32_t index;
  internal::AssertOk(LiteRtGetTensorIndex, tensor, &index);
  return index;
}

inline LiteRtTensorTypeId FetchExtendedTensorTypeId(LiteRtTensor tensor) {
  if (tensor == nullptr) {
    return kLiteRtRankedTensorType;
  }
  LiteRtTensorTypeId type_id;
  internal::AssertOk(LiteRtGetTensorTypeId, tensor, &type_id);
  return type_id;
}

inline std::variant<LiteRtUnrankedTensorType, litert::RankedTensorType>
FetchExtendedTensorType(LiteRtTensor tensor, LiteRtTensorTypeId type_id) {
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

inline LiteRtQuantizationTypeId FetchExtendedTensorQuantizationTypeId(
    LiteRtTensor tensor) {
  if (tensor == nullptr) {
    return kLiteRtQuantizationNone;
  }
  LiteRtQuantizationTypeId quantization_type_id;
  internal::AssertOk(LiteRtGetQuantizationTypeId, tensor,
                     &quantization_type_id);
  return quantization_type_id;
}

inline LiteRtQuantizationPerTensor FetchExtendedTensorQuantizationPerTensor(
    LiteRtTensor tensor) {
  if (FetchExtendedTensorQuantizationTypeId(tensor) !=
      kLiteRtQuantizationPerTensor) {
    return {};
  }
  LiteRtQuantizationPerTensor per_tensor_quantization;
  internal::AssertOk(LiteRtGetPerTensorQuantization, tensor,
                     &per_tensor_quantization);
  return per_tensor_quantization;
}

inline LiteRtQuantizationPerChannel FetchExtendedTensorQuantizationPerChannel(
    LiteRtTensor tensor) {
  if (FetchExtendedTensorQuantizationTypeId(tensor) !=
      kLiteRtQuantizationPerChannel) {
    return {};
  }
  LiteRtQuantizationPerChannel per_channel_quantization;
  internal::AssertOk(LiteRtGetPerChannelQuantization, tensor,
                     &per_channel_quantization);
  return per_channel_quantization;
}

}  // namespace internal::extended_model_detail

/// @brief A C++ wrapper for `LiteRtTensor`, representing a tensor in the model.
class Tensor : public internal::NonOwnedHandle<LiteRtTensor>,
               public litert::SimpleTensor {
 public:
  explicit Tensor(LiteRtTensor tensor)
      : internal::NonOwnedHandle<LiteRtTensor>(tensor),
        litert::SimpleTensor(
            internal::extended_model_detail::FetchExtendedTensorIndex(tensor),
            internal::extended_model_detail::FetchExtendedTensorName(tensor),
            internal::extended_model_detail::FetchExtendedTensorTypeId(tensor),
            internal::extended_model_detail::FetchExtendedTensorType(
                tensor,
                internal::extended_model_detail::FetchExtendedTensorTypeId(
                    tensor)),
            internal::extended_model_detail::
                FetchExtendedTensorQuantizationTypeId(tensor),
            internal::extended_model_detail::
                FetchExtendedTensorQuantizationPerTensor(tensor),
            internal::extended_model_detail::
                FetchExtendedTensorQuantizationPerChannel(tensor)) {}

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
  // function are used when Tensor is used as a live handle, reflects the most
  // up-to-date state of the underlying LiteRtTensor.
  LiteRtQuantizationTypeId QTypeId() const {
    LiteRtQuantizationTypeId q_type_id;
    internal::AssertOk(LiteRtGetQuantizationTypeId, Get(), &q_type_id);
    return q_type_id;
  }

  bool HasQuantization() const { return QTypeId() != kLiteRtQuantizationNone; }

  LiteRtQuantizationPerTensor PerTensorQuantization() const {
    LiteRtQuantizationPerTensor per_tensor_quantization;
    internal::AssertOk(LiteRtGetPerTensorQuantization, Get(),
                       &per_tensor_quantization);
    return per_tensor_quantization;
  }

  LiteRtQuantizationPerChannel PerChannelQuantization() const {
    LiteRtQuantizationPerChannel per_channel_quantization;
    internal::AssertOk(LiteRtGetPerChannelQuantization, Get(),
                       &per_channel_quantization);
    return per_channel_quantization;
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

namespace internal::extended_model_detail {

inline absl::string_view FetchExtendedSignatureKey(LiteRtSignature signature) {
  const char* key;
  internal::AssertOk(LiteRtGetSignatureKey, signature, &key);
  return key;
}

inline std::vector<absl::string_view> FetchExtendedSignatureInputNames(
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

inline std::vector<absl::string_view> FetchExtendedSignatureOutputNames(
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

inline std::vector<std::unique_ptr<SimpleTensor>>
FetchExtendedSignatureInputTensors(LiteRtSignature signature) {
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

inline std::vector<std::unique_ptr<SimpleTensor>>
FetchExtendedSignatureOutputTensors(LiteRtSignature signature) {
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

}  // namespace internal::extended_model_detail

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
    return Unexpected(Status::kErrorNotFound, "Failed to find input");
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
    return Unexpected(Status::kErrorNotFound, "Failed to find output");
  }
};

/// @brief A C++ wrapper for `LiteRtSignature`, representing a model signature.
class Signature : public internal::NonOwnedHandle<LiteRtSignature>,
                  public litert::SimpleSignature {
 public:
  explicit Signature(LiteRtSignature signature)
      : internal::NonOwnedHandle<LiteRtSignature>(signature),
        litert::SimpleSignature(
            internal::extended_model_detail::FetchExtendedSignatureKey(
                signature),
            internal::extended_model_detail::FetchExtendedSignatureInputNames(
                signature),
            internal::extended_model_detail::FetchExtendedSignatureInputTensors(
                signature),
            internal::extended_model_detail::FetchExtendedSignatureOutputNames(
                signature),
            internal::extended_model_detail::
                FetchExtendedSignatureOutputTensors(signature)) {}

  LiteRtSubgraph Subgraph() const {
    LiteRtSubgraph subgraph;
    internal::AssertOk(LiteRtGetSignatureSubgraph, Get(), &subgraph);
    return subgraph;
  }
};

/// @brief An extended C++ wrapper for `LiteRtModel`, providing additional
/// model introspection and manipulation capabilities.
class ExtendedModel : public litert::Model {
 public:
  ExtendedModel() = default;

  static ExtendedModel CreateFromOwnedHandle(LiteRtModel model) {
    return ExtendedModel(model, OwnHandle::kYes);
  }

  static ExtendedModel CreateFromNonOwnedHandle(LiteRtModel model) {
    return ExtendedModel(model, OwnHandle::kNo);
  }

  static Expected<ExtendedModel> CreateFromFile(const std::string& filename) {
    LiteRtModel model;
    if (auto status = LiteRtCreateModelFromFile(filename.c_str(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from file");
    }
    return CreateFromOwnedHandle(model);
  }

  /// @brief Creates a model from a buffer.
  ///
  /// The caller must ensure that the buffer remains valid for the lifetime of
  /// the model.
  static Expected<ExtendedModel> CreateFromBuffer(BufferRef<uint8_t> buffer) {
    LiteRtModel model;
    if (auto status =
            LiteRtCreateModelFromBuffer(buffer.Data(), buffer.Size(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from buffer");
    }
    return CreateFromOwnedHandle(model);
  }

#if !defined(LITERT_DYNAMIC_RUNTIME)
// copybara:uncomment_begin(google_only)
//   /// @brief Creates a model from an owned TFLite allocation.
//   ///
//   /// LiteRT takes ownership of the allocation wrapper.
//   static Expected<ExtendedModel> CreateFromAllocation(
//       std::unique_ptr<tflite::Allocation> allocation) {
//     LITERT_ASSIGN_OR_RETURN(
//         auto model, litert::Model::CreateFromAllocation(std::move(allocation)));
//     return CreateFromOwnedHandle(model.Release());
//   }
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
      auto key = internal::extended_model_detail::FetchExtendedSignatureKey(
          lite_rt_signature);
      if (key == signature_key) {
        LiteRtSubgraph subgraph;
        if (LiteRtGetSignatureSubgraph(lite_rt_signature, &subgraph) !=
            kLiteRtStatusOk) {
          return Unexpected(Status::kErrorNotFound, "Subgraph not found");
        }
        return litert::Subgraph(subgraph);
      }
    }
    return Unexpected(Status::kErrorNotFound, "Signature not found");
  }

  /// @brief Returns the list of signatures defined in the model.
  Expected<std::vector<Signature>> GetSignatures() const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    std::vector<Signature> signatures;
    signatures.reserve(num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      Signature signature(lite_rt_signature);
      signatures.push_back(std::move(signature));
    }
    return std::move(signatures);
  }

  /// @brief Returns the list of signature key names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureKeys() const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    std::vector<absl::string_view> signature_keys;
    signature_keys.reserve(num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      const char* key_cstr;
      internal::AssertOk(LiteRtGetSignatureKey, lite_rt_signature, &key_cstr);
      signature_keys.push_back(key_cstr);
    }
    return signature_keys;
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    auto signature = Signature(lite_rt_signature);
    return signature.InputNames();
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames() const {
    return GetSignatureInputNames(/*signature_index=*/0);
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      absl::string_view signature_key) const {
    auto signature = FindSignature(signature_key);
    if (!signature) {
      return Unexpected(Status::kErrorNotFound, "Signature not found");
    }
    return signature->InputNames();
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    auto signature = Signature(lite_rt_signature);
    return signature.OutputNames();
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames() const {
    return GetSignatureOutputNames(/*signature_index=*/0);
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      absl::string_view signature_key) const {
    auto signature = FindSignature(signature_key);
    if (!signature) {
      return Unexpected(Status::kErrorNotFound, "Signature not found");
    }
    return signature->OutputNames();
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

 private:
  /// @param owned Indicates if the created `TensorBuffer` object should take
  /// ownership of the provided `tensor_buffer` handle.
  ExtendedModel(LiteRtModel model, OwnHandle owned)
      : litert::Model(model, owned) {}
};

struct SerializationOptions {
  static LiteRtModelSerializationOptions Defaults() {
    LiteRtModelSerializationOptions opts{};
    opts.bytecode_alignment = 1;
    return opts;
  }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_LITERT_EXTENDED_MODEL_H_
