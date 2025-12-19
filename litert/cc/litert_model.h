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
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"

/// @file
/// @brief Defines C++ wrappers for the LiteRT model, signature, and tensor
/// types.

namespace litert {

/// @brief A C++ wrapper for `LiteRtTensor` with limited functionality.
class SimpleTensor : public internal::NonOwnedHandle<LiteRtTensor> {
 public:
  explicit SimpleTensor(LiteRtTensor tensor) : NonOwnedHandle(tensor) {}

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
};

/// @brief A simplified C++ wrapper for `LiteRtSignature`, representing a model
/// signature.
class SimpleSignature : public internal::NonOwnedHandle<LiteRtSignature> {
 public:
  explicit SimpleSignature(LiteRtSignature signature)
      : internal::NonOwnedHandle<LiteRtSignature>(signature) {}

  absl::string_view Key() const {
    const char* key;
    internal::AssertOk(LiteRtGetSignatureKey, Get(), &key);
    return key;
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

  /// @brief Returns the input tensor type for the given input signature name.
  Expected<RankedTensorType> InputTensorType(absl::string_view name) const {
    LITERT_ASSIGN_OR_RETURN(auto tensor, InputTensor(name));
    return tensor.RankedTensorType();
  }

  /// @brief Returns the input tensor type at the given index.
  Expected<RankedTensorType> InputTensorType(size_t index) const {
    LITERT_ASSIGN_OR_RETURN(auto tensor, InputTensor(index));
    return tensor.RankedTensorType();
  }

  /// @brief Returns the output tensor type for the given output signature
  /// name.
  Expected<RankedTensorType> OutputTensorType(absl::string_view name) const {
    LITERT_ASSIGN_OR_RETURN(auto tensor, OutputTensor(name));
    return tensor.RankedTensorType();
  }

  /// @brief Returns the output tensor type at the given index.
  Expected<RankedTensorType> OutputTensorType(size_t index) const {
    LITERT_ASSIGN_OR_RETURN(auto tensor, OutputTensor(index));
    return tensor.RankedTensorType();
  }

  /// @brief Returns the input tensor for the given input signature name.
  Expected<SimpleTensor> InputTensor(absl::string_view name) const;

  /// @brief Returns the input tensor at the given index.
  Expected<SimpleTensor> InputTensor(size_t index) const;

  /// @brief Returns the output tensor for the given output signature name.
  Expected<SimpleTensor> OutputTensor(absl::string_view name) const;

  /// @brief Returns the output tensor at the given index.
  Expected<SimpleTensor> OutputTensor(size_t index) const;
};

/// @brief A C++ wrapper for `LiteRtModel`, representing a LiteRT model.
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

  /// @brief Creates a model from a buffer.
  ///
  /// The caller must ensure that the buffer remains valid for the lifetime of
  /// the model.
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

  size_t GetNumSignatures() const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    return num_signatures;
  }

  /// @brief Returns the list of signatures defined in the model.
  Expected<std::vector<SimpleSignature>> GetSignatures() const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    std::vector<SimpleSignature> signatures;
    signatures.reserve(num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      SimpleSignature signature(lite_rt_signature);
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
    auto signature = SimpleSignature(lite_rt_signature);
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
      return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
    }
    return signature->InputNames();
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames(
      size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    auto signature = SimpleSignature(lite_rt_signature);
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
      return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
    }
    return signature->OutputNames();
  }

  /// @brief Returns the signature at the given index.
  Expected<SimpleSignature> GetSignature(size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    return SimpleSignature(lite_rt_signature);
  }

  /// @brief Returns the signature index for the given signature key.
  ///
  /// Returns 0 if the signature key is empty.
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

  /// @brief Returns the `SimpleSignature` object for the given signature key.
  ///
  /// Returns the default signature if the signature key is empty.
  Expected<SimpleSignature> FindSignature(
      absl::string_view signature_key) const {
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
        return SimpleSignature(lite_rt_signature);
      }
    }
    return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
  }

  static absl::string_view DefaultSignatureKey() {
    const char* key;
    internal::AssertOk(LiteRtGetDefaultSignatureKey, &key);
    return key;
  }

  /// @brief Returns the tensor type for the n-th input tensor.
  Expected<RankedTensorType> GetInputTensorType(size_t signature_index,
                                                size_t input_index) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            GetSignature(signature_index));
    return signature.InputTensorType(input_index);
  }

  /// @brief Returns the tensor type for the given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      size_t signature_index, absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            GetSignature(signature_index));
    return signature.InputTensorType(input_name);
  }

  /// @brief Returns the tensor type for the given input tensor name.
  Expected<RankedTensorType> GetInputTensorType(
      absl::string_view signature_key, absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            FindSignature(signature_key));
    return signature.InputTensorType(input_name);
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
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            GetSignature(signature_index));
    return signature.OutputTensorType(output_index);
  }

  /// @brief Returns the tensor type for the given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      size_t signature_index, absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            GetSignature(signature_index));
    return signature.OutputTensorType(output_name);
  }

  /// @brief Returns the tensor type for the given output tensor name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view signature_key, absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature& signature,
                            FindSignature(signature_key));
    return signature.OutputTensorType(output_name);
  }

  /// @brief Gets the output tensor type of the default signature for a given
  /// output name.
  Expected<RankedTensorType> GetOutputTensorType(
      absl::string_view output_name) const {
    return GetOutputTensorType(/*signature_index=*/0, output_name);
  }

 protected:
  /// @param owned Indicates if the created `TensorBuffer` object should take
  /// ownership of the provided `tensor_buffer` handle.
  Model(LiteRtModel model, OwnHandle owned) : Handle(model, owned) {}
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_MODEL_H_
