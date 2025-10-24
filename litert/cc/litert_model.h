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
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"

namespace litert {

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

  // Get the custom code. Returns value if and only if the op is a custom op.
  Expected<absl::string_view> CustomCode() const {
    const char* custom_code;
    auto stat = LiteRtGetCustomCode(Get(), &custom_code);
    if (stat != kLiteRtStatusOk) {
      return Error(stat, "Failed to get custom code");
    }
    return absl::string_view(custom_code);
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

  // Returns the input tensor with the given input signature name in the
  // signature entry.
  Expected<Tensor> InputTensor(absl::string_view name) const;

  // Returns the input tensor at the given index in the signature entry.
  Expected<Tensor> InputTensor(size_t index) const;

  // Returns the output tensor with the given output signature name in the
  // signature entry.
  Expected<Tensor> OutputTensor(absl::string_view name) const;

  // Returns the output tensor at the given index in the signature entry.
  Expected<Tensor> OutputTensor(size_t index) const;
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

  // Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    auto signature = Signature(lite_rt_signature);
    return signature.InputNames();
  }

  // Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames() const {
    return GetSignatureInputNames(/*signature_index=*/0);
  }

  // Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      absl::string_view signature_key) const {
    auto signature = FindSignature(signature_key);
    if (!signature) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
    }
    return signature->InputNames();
  }

  // Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    auto signature = Signature(lite_rt_signature);
    return signature.OutputNames();
  }

  // Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames() const {
    return GetSignatureOutputNames(/*signature_index=*/0);
  }

  // Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      absl::string_view signature_key) const {
    auto signature = FindSignature(signature_key);
    if (!signature) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
    }
    return signature->OutputNames();
  }

  // Returns the signature at the given index.
  Expected<Signature> GetSignature(size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    return Signature(lite_rt_signature);
  }

  // Returns the signature index for the given signature key.
  // Returns 0 if the signature key is empty.
  Expected<size_t> GetSignatureIndex(absl::string_view signature_key) const {
    if (signature_key.empty()) {
      return 0;
    }
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
  // Returns the default signature if the signature key is empty.
  Expected<Signature> FindSignature(absl::string_view signature_key) const {
    if (signature_key.empty()) {
      return GetSignature(/*signature_index=*/0);
    }
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
    LITERT_ASSIGN_OR_RETURN(const Signature& signature,
                            GetSignature(signature_index));
    LITERT_ASSIGN_OR_RETURN(const Tensor& tensor,
                            signature.InputTensor(input_index));
    return tensor.RankedTensorType();
  }

  // Returns the tensor type for the given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      size_t signature_index, absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(const Signature& signature,
                            GetSignature(signature_index));
    LITERT_ASSIGN_OR_RETURN(auto tensor, signature.InputTensor(input_name));
    return tensor.RankedTensorType();
  }

  // Returns the tensor type for the given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      absl::string_view signature_key, absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(const Signature& signature,
                            FindSignature(signature_key));
    LITERT_ASSIGN_OR_RETURN(auto tensor, signature.InputTensor(input_name));
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
    LITERT_ASSIGN_OR_RETURN(const Signature& signature,
                            GetSignature(signature_index));
    LITERT_ASSIGN_OR_RETURN(const Tensor& tensor,
                            signature.OutputTensor(output_index));
    return tensor.RankedTensorType();
  }

  // Returns the tensor type for the given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      size_t signature_index, absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(const Signature& signature,
                            GetSignature(signature_index));
    LITERT_ASSIGN_OR_RETURN(auto tensor, signature.OutputTensor(output_name));
    return tensor.RankedTensorType();
  }

  // Returns the tensor type for the given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view signature_key, absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(const Signature& signature,
                            FindSignature(signature_key));
    LITERT_ASSIGN_OR_RETURN(auto tensor, signature.OutputTensor(output_name));
    return tensor.RankedTensorType();
  }

  // Get output tensor type of the default signature for output name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view output_name) const {
    return GetOutputTensorType(/*signature_index=*/0, output_name);
  }

 private:
  // Parameter `owned` indicates if the created TensorBuffer object should take
  // ownership of the provided `tensor_buffer` handle.
  Model(LiteRtModel model, OwnHandle owned) : Handle(model, owned) {}
};

struct SerializationOptions {
  static LiteRtModelSerializationOptions Defaults() {
    return LiteRtModelSerializationOptions{};
  }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_MODEL_H_
