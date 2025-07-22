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

#ifndef ODML_LITERT_LITERT_CC_LITERT_MODEL_H_
#define ODML_LITERT_LITERT_CC_LITERT_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_rewriter.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_consts.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"

namespace litert {

// Type for tensors with known dimensions. C++ equivalent to
// LiteRtRankedTensorType.
class RankedTensorType {
 public:
  RankedTensorType(ElementType element_type, Layout&& layout)
      : element_type_(element_type), layout_(std::move(layout)) {}
  explicit RankedTensorType(const LiteRtRankedTensorType& type)
      : element_type_(static_cast<enum ElementType>(type.element_type)),
        layout_(type.layout) {}

  explicit operator LiteRtRankedTensorType() const {
    return LiteRtRankedTensorType{
        /*.element_type=*/static_cast<LiteRtElementType>(element_type_),
        /*layout=*/static_cast<LiteRtLayout>(layout_),
    };
  }

  bool operator==(const RankedTensorType& other) const {
    return ElementType() == other.ElementType() && Layout() == other.Layout();
  }

  bool operator!=(const RankedTensorType& other) const {
    return !(*this == other);
  }

  ElementType ElementType() const { return element_type_; }

  const Layout& Layout() const { return layout_; }

  Expected<size_t> Bytes() const {
    LITERT_ASSIGN_OR_RETURN(const size_t num_elements, layout_.NumElements());
    auto byte_width = GetByteWidth(element_type_);
    if (!byte_width) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument);
    }
    return num_elements * *byte_width;
  }

 private:
  enum ElementType element_type_;
  class Layout layout_;
};

// Construct a ranked tensor type from c++ type.
template <typename T>
RankedTensorType MakeRankedTensorType(
    std::initializer_list<Layout::Dim> shape) {
  return RankedTensorType(GetElementType<T>(), Layout(std::move(shape)));
}
template <typename T, typename Shape>
RankedTensorType MakeRankedTensorType(const Shape& shape) {
  return RankedTensorType(
      GetElementType<T>(),
      Layout(Dimensions(std::cbegin(shape), std::cend(shape))));
}

// Tensor weights. C++ equivalent of LiteRtWeights.
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

// Tensor. C++ equivalent of LiteRtTensor.
class Tensor : public internal::NonOwnedHandle<LiteRtTensor> {
 public:
  explicit Tensor(LiteRtTensor tensor) : NonOwnedHandle(tensor) {}

  ElementType ElementType() const {
    if (TypeId() == kLiteRtUnrankedTensorType) {
      LITERT_ASSIGN_OR_ABORT(auto tensor_type, UnrankedTensorType());
      return static_cast<enum ElementType>(tensor_type.element_type);
    } else {
      LITERT_ASSIGN_OR_ABORT(auto tensor_type, RankedTensorType());
      return tensor_type.ElementType();
    }
  }

  bool HasType(const RankedTensorType& type) const {
    auto t = RankedTensorType();
    return t && *t == type;
  }

  bool HasType(const LiteRtRankedTensorType& type) const {
    auto t = RankedTensorType();
    return t && *t == ::litert::RankedTensorType(type);
  }

  LiteRtTensorTypeId TypeId() const {
    LiteRtTensorTypeId type_id;
    internal::AssertOk(LiteRtGetTensorTypeId, Get(), &type_id);
    return type_id;
  }

  Expected<LiteRtUnrankedTensorType> UnrankedTensorType() const {
    if (TypeId() != kLiteRtUnrankedTensorType) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Not an unranked invalid tensor");
    }
    LiteRtUnrankedTensorType unranked_tensor_type;
    internal::AssertOk(LiteRtGetUnrankedTensorType, Get(),
                       &unranked_tensor_type);
    return unranked_tensor_type;
  }

  Expected<RankedTensorType> RankedTensorType() const {
    if (TypeId() != kLiteRtRankedTensorType) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Not a ranked tensor type");
    }
    LiteRtRankedTensorType ranked_tensor_type;
    internal::AssertOk(LiteRtGetRankedTensorType, Get(), &ranked_tensor_type);
    return litert::RankedTensorType(ranked_tensor_type);
  }

  LiteRtQuantizationTypeId QTypeId() const {
    LiteRtQuantizationTypeId q_type_id;
    internal::AssertOk(LiteRtGetQuantizationTypeId, Get(), &q_type_id);
    return q_type_id;
  }

  bool HasQuantization() const { return QTypeId() != kLiteRtQuantizationNone; }

  LiteRtQuantizationPerTensor PerTensorQuantization() const {
    internal::AssertEq([&]() { return QTypeId(); },
                       kLiteRtQuantizationPerTensor);
    LiteRtQuantizationPerTensor per_tensor_quantization;
    internal::AssertOk(LiteRtGetPerTensorQuantization, Get(),
                       &per_tensor_quantization);
    return per_tensor_quantization;
  }

  LiteRtQuantizationPerChannel PerChannelQuantization() const {
    internal::AssertEq([&]() { return QTypeId(); },
                       kLiteRtQuantizationPerChannel);
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

  absl::string_view Name() const {
    const char* name;
    internal::AssertOk(LiteRtGetTensorName, Get(), &name);
    return absl::string_view(name);
  }

  std::uint32_t TensorIndex() const {
    std::uint32_t tensor_index;
    internal::AssertOk(LiteRtGetTensorIndex, Get(), &tensor_index);
    return tensor_index;
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
      return Unexpected(kLiteRtStatusErrorInvalidArgument);
    }

    if (!HasWeights()) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument);
    }
    const absl::Span<const uint8_t> weights = Weights().Bytes();

    auto num_elements = ranked_tensor_type->Layout().NumElements();
    if (!num_elements) {
      return num_elements.Error();
    }
    auto byte_width = GetByteWidth(ty);
    if (!byte_width.has_value()) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument);
    }

    if (byte_width.value() * *num_elements != weights.size()) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument);
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

  bool IsSubgraphOutput() const;
  bool IsSubgraphInput() const;
  bool IsConstant() const;
};

using OpInputs = absl::InlinedVector<Tensor, kExpectedMaxNumOfOpInputs>;
using OpOutputs = absl::InlinedVector<Tensor, kExpectedMaxNumOfOpOutputs>;

// Operator. C++ equivalent of LiteRtOp.
class Op : public internal::NonOwnedHandle<LiteRtOp> {
 public:
  explicit Op(LiteRtOp op) : NonOwnedHandle<LiteRtOp>(op) {}

  LiteRtOpCode Code() const {
    LiteRtOpCode opcode;
    internal::AssertOk(LiteRtGetOpCode, Get(), &opcode);
    return opcode;
  }

  OpInputs Inputs() const;
  OpOutputs Outputs() const;
};

struct Tensor::TensorUse {
  Op user;
  LiteRtParamIndex user_arg_ind;
};

using SubgraphInputs =
    absl::InlinedVector<Tensor, kExpectedMaxNumOfSubgraphInputs>;
using SubgraphOutputs =
    absl::InlinedVector<Tensor, kExpectedMaxNumOfSubgraphOutputs>;

// Model subgraph. C++ equivalent of LiteRtSubgraph.
class Subgraph : public internal::NonOwnedHandle<LiteRtSubgraph> {
 public:
  explicit Subgraph(LiteRtSubgraph subgraph)
      : internal::NonOwnedHandle<LiteRtSubgraph>(subgraph) {}

  SubgraphInputs Inputs() const;
  SubgraphOutputs Outputs() const;
  std::vector<Op> Ops() const;

  // Returns the input tensor with the given input signature name.
  Expected<Tensor> Input(absl::string_view name) const;

  // Returns the output tensor with the given output signature name.
  Expected<Tensor> Output(absl::string_view name) const;
};

// Model Rewriter. C++ equivalent of LiteRtRewriter.
struct RankedTensorSpec {
  RankedTensorType ranked_tensor_type;
  std::optional<Weights> weights = std::nullopt;
  std::optional<LiteRtQuantizationPerTensor> per_tensor_quantization =
      std::nullopt;
  std::optional<LiteRtQuantizationPerChannel> per_channel_quantization =
      std::nullopt;
  std::optional<std::string> tensor_name = std::nullopt;
};

class RankedTensorSpecBuilder {
 public:
  RankedTensorSpecBuilder&& with_ranked_tensor_type(RankedTensorType type) && {
    ranked_tensor_type_ = std::move(type);
    return std::move(*this);
  }

  RankedTensorSpecBuilder&& with_weights(Weights w) && {
    weights_ = std::move(w);
    return std::move(*this);
  }

  RankedTensorSpecBuilder&& with_per_tensor_quantization(
      LiteRtQuantizationPerTensor q) && {
    per_tensor_quantization_ = std::move(q);
    return std::move(*this);
  }

  RankedTensorSpecBuilder&& with_per_channel_quantization(
      LiteRtQuantizationPerChannel q) && {
    per_channel_quantization_ = std::move(q);
    return std::move(*this);
  }

  RankedTensorSpecBuilder&& with_tensor_name(std::string name) && {
    tensor_name_ = std::move(name);
    return std::move(*this);
  }

  std::optional<RankedTensorSpec> build() && {
    if (!ranked_tensor_type_.has_value()) {
      return std::nullopt;
    }

    return RankedTensorSpec{
        *std::move(ranked_tensor_type_), std::move(weights_),
        std::move(per_tensor_quantization_),
        std::move(per_channel_quantization_), std::move(tensor_name_)};
  }

 private:
  std::optional<RankedTensorType> ranked_tensor_type_;
  std::optional<Weights> weights_;
  std::optional<LiteRtQuantizationPerTensor> per_tensor_quantization_;
  std::optional<LiteRtQuantizationPerChannel> per_channel_quantization_;
  std::optional<std::string> tensor_name_;
};

class Rewriter : public internal::NonOwnedHandle<LiteRtRewriter> {
 public:
  explicit Rewriter(LiteRtRewriter rewriter)
      : internal::NonOwnedHandle<LiteRtRewriter>(rewriter) {}
  // For ranked tensors.
  Expected<Tensor> BuildTensor(const RankedTensorSpec& spec) const {
    // tensor holds the newly created tensor.
    LiteRtTensor tensor;
    LiteRtRankedTensorType ranked_tensor_type_litert =
        static_cast<LiteRtRankedTensorType>(spec.ranked_tensor_type);

    LiteRtWeights litert_weights;
    if (spec.weights.has_value()) {
      litert_weights = spec.weights.value().Get();
    } else {
      litert_weights = nullptr;
    }

    LiteRtQuantizationTypeId quantization_type_id = kLiteRtQuantizationNone;
    LiteRtQuantizationPerTensor litert_per_tensor_quantization;
    if (spec.per_tensor_quantization.has_value()) {
      litert_per_tensor_quantization = spec.per_tensor_quantization.value();
      quantization_type_id = kLiteRtQuantizationPerTensor;
    }
    LiteRtQuantizationPerChannel litert_per_channel_quantization;
    if (spec.per_channel_quantization.has_value()) {
      litert_per_channel_quantization = spec.per_channel_quantization.value();
      quantization_type_id = kLiteRtQuantizationPerChannel;
    }
    internal::AssertOk(LiteRtRewriterBuildTensor, kLiteRtRankedTensorType,
                       ranked_tensor_type_litert, LiteRtUnrankedTensorType(),
                       litert_weights, quantization_type_id,
                       litert_per_tensor_quantization,
                       litert_per_channel_quantization, this->Get(),
                       spec.tensor_name.value_or("").c_str(),
                       spec.tensor_name.value_or("").size(), &tensor);
    return Tensor(tensor);
  }

  // Trait for building scalars.
  Expected<Tensor> BuildScalar(
      LiteRtElementType element_type,
      std::optional<std::string> name = std::nullopt) const {
    LiteRtTensor tensor;
    LiteRtUnrankedTensorType unranked_tensor_type;
    unranked_tensor_type.element_type = element_type;
    internal::AssertOk(
        LiteRtRewriterBuildTensor, kLiteRtUnrankedTensorType,
        LiteRtRankedTensorType(), unranked_tensor_type, LiteRtWeights(),
        kLiteRtQuantizationNone, LiteRtQuantizationPerTensor(),
        LiteRtQuantizationPerChannel(), this->Get(), name.value_or("").c_str(),
        name.value_or("").size(), &tensor);
    return Tensor(tensor);
  }

  Op BuildOp(LiteRtOpCode op_code, OpInputs& inputs, OpOutputs& outputs,
             std::optional<std::string> name = std::nullopt) const {
    LiteRtOp litert_op;
    std::vector<LiteRtTensor> input_tensors;
    input_tensors.reserve(inputs.size());
    for (const auto& input : inputs) {
      input_tensors.push_back(input.Get());
    }
    std::vector<LiteRtTensor> output_tensors;
    output_tensors.reserve(outputs.size());
    for (const auto& output : outputs) {
      output_tensors.push_back(output.Get());
    }
    internal::AssertOk(LiteRtRewriterBuildOp, op_code, input_tensors.size(),
                       input_tensors.data(), output_tensors.size(),
                       output_tensors.data(), this->Get(), &litert_op);
    return Op(litert_op);
  }
  // Clone the given op.
  Op BuildOp(Op& src, OpInputs& inputs, OpOutputs& outputs) {
    return BuildOp(src.Code(), inputs, outputs);
  };

  // Record the op to be erased.
  void EraseOp(Op& op) const {
    internal::AssertOk(LiteRtRewriterEraseOp, op.Get(), this->Get());
  }
};

// Model signature. C++ equivalent of LiteRtSignature.
class Signature : public internal::NonOwnedHandle<LiteRtSignature> {
 public:
  explicit Signature(LiteRtSignature signature)
      : internal::NonOwnedHandle<LiteRtSignature>(signature) {}

  absl::string_view Key() const {
    const char* key;
    internal::AssertOk(LiteRtGetSignatureKey, Get(), &key);
    return key;
  }

  LiteRtSubgraph Subgraph() const {
    LiteRtSubgraph subgraph;
    internal::AssertOk(LiteRtGetSignatureSubgraph, Get(), &subgraph);
    return subgraph;
  }

  std::vector<absl::string_view> InputNames() const {
    LiteRtParamIndex num_inputs;
    internal::AssertOk(LiteRtGetNumSignatureInputs, Get(), &num_inputs);
    std::vector<absl::string_view> input_names;
    input_names.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      const char* input_name;
      internal::AssertOk(LiteRtGetSignatureInputName, Get(), i, &input_name);
      input_names.push_back(input_name);
    }
    return input_names;
  }

  std::vector<absl::string_view> OutputNames() const {
    LiteRtParamIndex num_outputs;
    internal::AssertOk(LiteRtGetNumSignatureOutputs, Get(), &num_outputs);
    std::vector<absl::string_view> output_names;
    output_names.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      const char* output_name;
      internal::AssertOk(LiteRtGetSignatureOutputName, Get(), i, &output_name);
      output_names.push_back(output_name);
    }
    return output_names;
  }
};

// Model. C++ equivalent of LiteRtModel.
class Model : public internal::Handle<LiteRtModel, LiteRtDestroyModel> {
 public:
  Model() = default;

  static Model CreateFromOwnedHandle(LiteRtModel model) {
    return Model(model, OwnHandle::kYes);
  }

  static Model CreateFromNonOwnedHandle(LiteRtModel model) {
    return Model(model, OwnHandle::kNo);
  }

  static Expected<Model> CreateFromFile(const std::string& filename) {
    LiteRtModel model;
    if (auto status = LiteRtCreateModelFromFile(filename.c_str(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to load model from file");
    }
    return CreateFromOwnedHandle(model);
  }

  // The caller must ensure that the buffer remains valid for the lifetime of
  // the model.
  static Expected<Model> CreateFromBuffer(BufferRef<uint8_t> buffer) {
    LiteRtModel model;
    if (auto status =
            LiteRtCreateModelFromBuffer(buffer.Data(), buffer.Size(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to load model from buffer");
    }
    return CreateFromOwnedHandle(model);
  }

  Expected<absl::Span<const uint8_t>> Metadata(
      const std::string& metadata_key) const {
    const void* buffer;
    size_t buffer_size;
    if (LiteRtGetModelMetadata(Get(), metadata_key.data(), &buffer,
                               &buffer_size) != kLiteRtStatusOk) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Metadata key not found");
    }
    return absl::MakeSpan(static_cast<const uint8_t*>(buffer), buffer_size);
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

  Expected<Subgraph> Subgraph(size_t subgraph_index) const {
    LiteRtSubgraph subgraph;
    if (LiteRtGetModelSubgraph(Get(), subgraph_index, &subgraph) !=
        kLiteRtStatusOk) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Subgraph not found");
    }
    return litert::Subgraph(subgraph);
  }

  Expected<class Subgraph> Subgraph(absl::string_view signature_key) const {
    auto signature = FindSignature(signature_key);
    if (!signature) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
    }
    return litert::Subgraph(signature->Subgraph());
  }

  size_t GetNumSignatures() const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    return num_signatures;
  }

  // Returns the list of signatures defined in the model.
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

  // Returns the signature at the given index.
  Expected<Signature> GetSignature(size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    return Signature(lite_rt_signature);
  }

  // Returns the signature index for the given signature key.
  Expected<size_t> GetSignatureIndex(absl::string_view signature_key) const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      const char* key_cstr;
      internal::AssertOk(LiteRtGetSignatureKey, lite_rt_signature, &key_cstr);
      if (absl::string_view(key_cstr) == signature_key) {
        return i;
      }
    }
    return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
  }

  // Returns the Signature object for the given signature key.
  Expected<Signature> FindSignature(absl::string_view signature_key) const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      const char* key_cstr;
      internal::AssertOk(LiteRtGetSignatureKey, lite_rt_signature, &key_cstr);
      if (absl::string_view(key_cstr) == signature_key) {
        return Signature(lite_rt_signature);
      }
    }
    return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
  }

  static absl::string_view DefaultSignatureKey() {
    const char* key;
    internal::AssertOk(LiteRtGetDefaultSignatureKey, &key);
    return key;
  }

  // Returns the tensor type for the given n-th input tensor.
  Expected<RankedTensorType> GetInputTensorType(size_t signature_index,
                                                size_t input_index) const {
    LITERT_ASSIGN_OR_RETURN(auto subgraph, Subgraph(signature_index));
    return subgraph.Inputs()[input_index].RankedTensorType();
  }

  // Returns the tensor type for the given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      size_t signature_index, absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(auto subgraph, Subgraph(signature_index));
    LITERT_ASSIGN_OR_RETURN(auto tensor, subgraph.Input(input_name));
    return tensor.RankedTensorType();
  }

  // Get input tensor type of the default signature for input name.
  Expected<RankedTensorType> GetInputTensorType(
      absl::string_view input_name) const {
    return GetInputTensorType(/*signature_index=*/0, input_name);
  }

  // Returns the tensor type for the given n-th output tensor.
  Expected<RankedTensorType> GetOutputTensorType(size_t signature_index,
                                                 size_t output_index) const {
    auto subgraph = Subgraph(signature_index);
    return subgraph->Outputs()[output_index].RankedTensorType();
  }

  // Returns the tensor type for the given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      size_t signature_index, absl::string_view output_name) const {
    auto subgraph = Subgraph(signature_index);
    LITERT_ASSIGN_OR_RETURN(auto tensor, subgraph->Output(output_name));
    return tensor.RankedTensorType();
  }

  // Get output tensor type of the default signature for output name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view output_name) const {
    return GetOutputTensorType(/*signature_index=*/0, output_name);
  }

 private:
  // Parameter `owned` indicates if the created TensorBuffer object should
  // take ownership of the provided `tensor_buffer` handle.
  Model(LiteRtModel model, OwnHandle owned) : Handle(model, owned) {}
};

struct SerializationOptions {
  static LiteRtModelSerializationOptions Defaults() {
    return LiteRtModelSerializationOptions{};
  }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_MODEL_H_
