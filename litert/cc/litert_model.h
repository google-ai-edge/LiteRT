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
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_consts.h"
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
class SimpleTensor {
 public:
  virtual ~SimpleTensor() = default;

  explicit SimpleTensor(
      LiteRtParamIndex index, absl::string_view name,
      LiteRtTensorTypeId type_id,
      std::variant<LiteRtUnrankedTensorType, litert::RankedTensorType>&& type)
      : index_(index), name_(name), type_id_(type_id), type_(std::move(type)) {}

  // Allow copying SimpleTensors.
  SimpleTensor(const SimpleTensor& other) = default;
  SimpleTensor(SimpleTensor&&) = default;
  SimpleTensor& operator=(const SimpleTensor& other) = default;
  SimpleTensor& operator=(SimpleTensor&&) = default;

  ElementType ElementType() const {
    if (type_id_ == kLiteRtUnrankedTensorType) {
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

  LiteRtTensorTypeId TypeId() const { return type_id_; }

  Expected<LiteRtUnrankedTensorType> UnrankedTensorType() const {
    if (type_id_ != kLiteRtUnrankedTensorType) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Not an unranked invalid tensor");
    }
    return std::get<LiteRtUnrankedTensorType>(type_);
  }

  Expected<RankedTensorType> RankedTensorType() const {
    if (type_id_ != kLiteRtRankedTensorType) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Not a ranked tensor type");
    }
    return std::get<litert::RankedTensorType>(type_);
  }

  absl::string_view Name() const { return name_; }

  std::uint32_t TensorIndex() const { return index_; }

 private:
  std::uint32_t index_;
  std::string_view name_;
  LiteRtTensorTypeId type_id_;
  std::variant<LiteRtUnrankedTensorType, litert::RankedTensorType> type_;
};

/// @brief A simplified C++ wrapper for `LiteRtSignature`, representing a model
/// signature.
class SimpleSignature {
 public:
  virtual ~SimpleSignature() = default;

  explicit SimpleSignature(
      absl::string_view key, std::vector<absl::string_view> input_names,
      std::vector<std::unique_ptr<SimpleTensor>> input_tensors,
      std::vector<absl::string_view> output_names,
      std::vector<std::unique_ptr<SimpleTensor>> output_tensors)
      : key_(key),
        input_names_(std::move(input_names)),
        input_tensors_(std::move(input_tensors)),
        output_names_(std::move(output_names)),
        output_tensors_(std::move(output_tensors)) {}

  SimpleSignature(SimpleSignature&&) = default;
  SimpleSignature& operator=(SimpleSignature&&) = default;

  absl::string_view Key() const { return key_; }

  std::vector<absl::string_view> InputNames() const {
    std::vector<absl::string_view> input_names;
    input_names.reserve(input_names_.size());
    for (const auto& input_name : input_names_) {
      input_names.push_back(input_name);
    }
    return input_names;
  }

  std::vector<absl::string_view> OutputNames() const {
    std::vector<absl::string_view> output_names;
    output_names.reserve(output_names_.size());
    for (const auto& output_name : output_names_) {
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
  Expected<const SimpleTensor&> InputTensor(absl::string_view name) const {
    for (int i = 0; i < input_names_.size(); ++i) {
      if (input_names_[i] == name) {
        return *input_tensors_[i];
      }
    }
    return Error(kLiteRtStatusErrorNotFound, "Input tensor not found");
  }

  /// @brief Returns the input tensor at the given index.
  Expected<const SimpleTensor&> InputTensor(size_t index) const {
    if (index >= input_names_.size()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Input index out of bounds");
    }
    return InputTensor(input_names_[index]);
  }

  /// @brief Returns the output tensor for the given output signature name.
  Expected<const SimpleTensor&> OutputTensor(absl::string_view name) const {
    for (int i = 0; i < output_names_.size(); ++i) {
      if (output_names_[i] == name) {
        return *output_tensors_[i];
      }
    }
    return Error(kLiteRtStatusErrorNotFound, "Output tensor not found");
  }

  /// @brief Returns the output tensor at the given index.
  Expected<const SimpleTensor&> OutputTensor(size_t index) const {
    if (index >= output_names_.size()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Output index out of bounds");
    }
    return OutputTensor(output_names_[index]);
  }

 private:
  std::string_view key_;
  std::vector<absl::string_view> input_names_;
  std::vector<std::unique_ptr<SimpleTensor>> input_tensors_;
  std::vector<absl::string_view> output_names_;
  std::vector<std::unique_ptr<SimpleTensor>> output_tensors_;
};

/// @brief A C++ wrapper for `LiteRtModel`, representing a LiteRT model.
///
/// \internal
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
  Expected<std::vector<SimpleSignature>> GetSignatures() const;

  /// @brief Returns the list of signature key names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureKeys() const;

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      size_t signature_index) const;

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
      size_t signature_index) const;

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
  Expected<SimpleSignature> GetSignature(size_t signature_index) const;

  /// @brief Returns the signature index for the given signature key.
  ///
  /// Returns 0 if the signature key is empty.
  Expected<size_t> GetSignatureIndex(absl::string_view signature_key) const;

  /// @brief Returns the `SimpleSignature` object for the given signature key.
  ///
  /// Returns the default signature if the signature key is empty.
  Expected<SimpleSignature> FindSignature(
      absl::string_view signature_key) const;

  static absl::string_view DefaultSignatureKey() {
    return kDefaultSignatureKey;
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
