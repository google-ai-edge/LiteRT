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
#include <optional>
#include <string>
#include <utility>
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
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"

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
class Tensor : public litert::SimpleTensor {
 public:
  explicit Tensor(LiteRtTensor tensor) : litert::SimpleTensor(tensor) {}

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

// Signature. C++ equivalent of LiteRtSignature.
class Signature : public litert::SimpleSignature {
 public:
  explicit Signature(LiteRtSignature signature)
      : litert::SimpleSignature(signature) {}

  LiteRtSubgraph Subgraph() const {
    LiteRtSubgraph subgraph;
    internal::AssertOk(LiteRtGetSignatureSubgraph, Get(), &subgraph);
    return subgraph;
  }
};

// ExtendedModel. C++ equivalent of LiteRtModel.
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
      return Unexpected(status, "Failed to load model from file");
    }
    return CreateFromOwnedHandle(model);
  }

  // The caller must ensure that the buffer remains valid for the lifetime of
  // the model.
  static Expected<ExtendedModel> CreateFromBuffer(BufferRef<uint8_t> buffer) {
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

  Expected<void> AddMetadata(const std::string& metadata_key,
                             const std::string& metadata_data) {
    LiteRtStatus status = LiteRtAddModelMetadata(
        Get(), metadata_key.data(), metadata_data.data(), metadata_data.size());
    if (status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to add metadata");
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
    LiteRtSubgraph subgraph;
    if (LiteRtGetSignatureSubgraph(signature->Get(), &subgraph) !=
        kLiteRtStatusOk) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Subgraph not found");
    }

    return litert::Subgraph(subgraph);
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

  // Returns the list of signature key names defined in the signature.
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

  // Serializes a model to a buffer. Model would be released after
  // serialization.
  static Expected<OwningBufferRef<uint8_t>> Serialize(
      Model&& model, const LiteRtModelSerializationOptions& options) {
    OwningBufferRef<uint8_t> buf;
    auto [data, size, offset] = buf.GetWeak();

    LITERT_RETURN_IF_ERROR(LiteRtSerializeModel(
        std::move(model.Release()), &data, &size, &offset, true, options));
    return std::move(buf);
  }

 private:
  // Parameter `owned` indicates if the created TensorBuffer object should take
  // ownership of the provided `tensor_buffer` handle.
  ExtendedModel(LiteRtModel model, OwnHandle owned)
      : litert::Model(model, owned) {}
};

struct SerializationOptions {
  static LiteRtModelSerializationOptions Defaults() {
    return LiteRtModelSerializationOptions{};
  }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_LITERT_EXTENDED_MODEL_H_
