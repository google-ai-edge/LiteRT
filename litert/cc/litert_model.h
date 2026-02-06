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
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
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
  std::string name_;
  LiteRtTensorTypeId type_id_;
  std::variant<LiteRtUnrankedTensorType, litert::RankedTensorType> type_;
};

/// @brief A simplified C++ wrapper for `LiteRtSignature`, representing a model
/// signature.
class SimpleSignature {
 public:
  virtual ~SimpleSignature() = default;

  explicit SimpleSignature(
      absl::string_view key, std::vector<std::string> input_names,
      std::vector<std::unique_ptr<SimpleTensor>> input_tensors,
      std::vector<std::string> output_names,
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
  std::string key_;
  std::vector<std::string> input_names_;
  std::vector<std::unique_ptr<SimpleTensor>> input_tensors_;
  std::vector<std::string> output_names_;
  std::vector<std::unique_ptr<SimpleTensor>> output_tensors_;
};

/// @brief A C++ wrapper for `LiteRtModel`, representing a LiteRT model.
class Model : public internal::BaseHandle<LiteRtModel> {
 public:
  virtual ~Model() = default;

  Model() = default;

  Model(Model&&) = default;
  Model& operator=(Model&&) = default;

  static Model CreateFromOwnedHandle(const Environment& env,
                                     LiteRtModel model) {
    return Model(env.GetHolder(), model, OwnHandle::kYes);
  }

  static Model CreateFromNonOwnedHandle(const Environment& env,
                                        LiteRtModel model) {
    return Model(env.GetHolder(), model, OwnHandle::kNo);
  }

  [[deprecated(
      "Use CreateFromFile(const Environment& env, "
      "const std::string& filename) instead.")]]
  static Expected<Model> CreateFromFile(const std::string& filename) {
    auto& env = Environment::GetDefault();
    if (!env) {
      return env.Error();
    }
    return CreateFromFile(*env, filename);
  }

  static Expected<Model> CreateFromFile(const Environment& env,
                                        const std::string& filename) {
    LiteRtModel model;
    if (auto status = env.GetHolder().runtime->CreateModelFromFile(
            filename.c_str(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to load model from file");
    }
    return CreateFromOwnedHandle(env, model);
  }

  /// @brief Creates a model from a buffer.
  ///
  /// The caller must ensure that the buffer remains valid for the lifetime of
  /// the model.
  [[deprecated(
      "Use CreateFromBuffer(const Environment& env, "
      "BufferRef<uint8_t> buffer) instead.")]]
  static Expected<Model> CreateFromBuffer(BufferRef<uint8_t> buffer) {
    auto& env = Environment::GetDefault();
    if (!env) {
      return env.Error();
    }
    return CreateFromBuffer(*env, buffer);
  }

  /// @brief Creates a model from a buffer.
  ///
  /// The caller must ensure that the buffer remains valid for the lifetime of
  /// the model.
  static Expected<Model> CreateFromBuffer(const Environment& env,
                                          BufferRef<uint8_t> buffer) {
    LiteRtModel model;
    if (auto status = env.GetHolder().runtime->CreateModelFromBuffer(
            buffer.Data(), buffer.Size(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to load model from buffer");
    }
    return CreateFromOwnedHandle(env, model);
  }

  Expected<absl::Span<const uint8_t>> Metadata(
      const std::string& metadata_key) const {
    const void* buffer;
    size_t buffer_size;
    if (env_.runtime->GetModelMetadata(Get(), metadata_key.data(), &buffer,
                                       &buffer_size) != kLiteRtStatusOk) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Metadata key not found");
    }
    return absl::MakeSpan(static_cast<const uint8_t*>(buffer), buffer_size);
  }

  size_t GetNumSignatures() const {
    LiteRtParamIndex num_signatures;
    LITERT_CHECK_STATUS_OK(
        env_.runtime->GetNumModelSignatures(Get(), &num_signatures));
    return num_signatures;
  }

  /// @brief Returns the list of signatures defined in the model.
  Expected<const std::vector<std::unique_ptr<SimpleSignature>>&> GetSignatures()
      const {
    if (!signatures_) {
      LiteRtParamIndex num_signatures;
      LITERT_CHECK_STATUS_OK(
          env_.runtime->GetNumModelSignatures(Get(), &num_signatures));
      std::vector<std::unique_ptr<SimpleSignature>> signatures;
      signatures.reserve(num_signatures);
      for (int i = 0; i < num_signatures; ++i) {
        LiteRtSignature lite_rt_signature;
        LITERT_CHECK_STATUS_OK(
            env_.runtime->GetModelSignature(Get(), i, &lite_rt_signature));
        LITERT_ASSIGN_OR_RETURN(auto signature,
                                ToSimpleSignature(lite_rt_signature));
        signatures.push_back(std::move(signature));
      }
      signatures_ = std::move(signatures);
    }
    return *signatures_;
  }

  /// @brief Returns the list of signature key names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureKeys() const {
    LITERT_ASSIGN_OR_RETURN(const auto& signatures, GetSignatures());
    std::vector<absl::string_view> signature_keys;
    signature_keys.reserve(signatures.size());
    for (const auto& signature : signatures) {
      signature_keys.push_back(signature->Key());
    }
    return signature_keys;
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames(
      size_t signature_index) const {
    LITERT_ASSIGN_OR_RETURN(const auto& signature,
                            GetSignature(signature_index));
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
    LITERT_ASSIGN_OR_RETURN(const auto& signature,
                            GetSignature(signature_index));
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
  Expected<const SimpleSignature&> GetSignature(size_t signature_index) const {
    LITERT_ASSIGN_OR_RETURN(const auto& signatures, GetSignatures());
    if (signature_index >= signatures.size()) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Signature index out of bounds");
    }
    return *signatures[signature_index];
  }

  /// @brief Returns the signature index for the given signature key.
  ///
  /// Returns 0 if the signature key is empty.
  Expected<size_t> GetSignatureIndex(absl::string_view signature_key) const {
    if (signature_key.empty()) {
      return 0;
    }
    LITERT_ASSIGN_OR_RETURN(const auto& signatures, GetSignatures());
    auto signature = absl::c_find_if(
        signatures,
        [signature_key](const std::unique_ptr<SimpleSignature>& signature) {
          return signature->Key() == signature_key;
        });
    if (signature != signatures.end()) {
      return std::distance(signatures.begin(), signature);
    }
    return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
  }

  /// @brief Returns the `SimpleSignature` object for the given signature key.
  ///
  /// Returns the default signature if the signature key is empty.
  Expected<const SimpleSignature&> FindSignature(
      absl::string_view signature_key) const {
    LITERT_ASSIGN_OR_RETURN(const auto& signatures, GetSignatures());
    auto signature = absl::c_find_if(
        signatures,
        [signature_key](const std::unique_ptr<SimpleSignature>& signature) {
          return signature->Key() == signature_key;
        });
    if (signature != signatures.end()) {
      return **signature;
    }
    return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
  }

  static absl::string_view DefaultSignatureKey() {
    auto& env = Environment::GetDefault();
    if (!env) {
      LITERT_LOG(LITERT_ERROR, "Failed to get default environment");
      return "";
    }
    const char* key;
    LITERT_CHECK_STATUS_OK(
        env->GetHolder().runtime->GetDefaultSignatureKey(&key));
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
  Model(const internal::EnvironmentHolder& env, LiteRtModel model,
        OwnHandle owned)
      : BaseHandle<LiteRtModel>(
            model,
            [runtime = env.runtime](LiteRtModel model) {
              runtime->DestroyModel(model);
            },
            owned),
        env_(env) {}

  /// @brief Creates a `SimpleTensor` object from a `LiteRtTensor`.
  ///
  /// @param index The index of the tensor in the signature.
  /// @param lite_rt_tensor The `LiteRtTensor` object to convert.
  /// @return A `SimpleTensor` object representing the tensor.
  virtual Expected<std::unique_ptr<SimpleTensor>> ToSimpleTensor(
      LiteRtTensor lite_rt_tensor) const {
    const char* name;
    LITERT_CHECK_STATUS_OK(env_.runtime->GetTensorName(lite_rt_tensor, &name));
    std::uint32_t index;
    LITERT_CHECK_STATUS_OK(
        env_.runtime->GetTensorIndex(lite_rt_tensor, &index));
    LiteRtTensorTypeId type_id;
    LITERT_CHECK_STATUS_OK(
        env_.runtime->GetTensorTypeId(lite_rt_tensor, &type_id));
    std::variant<LiteRtUnrankedTensorType, RankedTensorType> tensor_type;
    if (type_id == kLiteRtRankedTensorType) {
      LiteRtRankedTensorType ranked_tensor_type;
      LITERT_CHECK_STATUS_OK(env_.runtime->GetRankedTensorType(
          lite_rt_tensor, &ranked_tensor_type));
      tensor_type = RankedTensorType(ranked_tensor_type);
    } else {
      LiteRtUnrankedTensorType unranked_tensor_type;
      LITERT_CHECK_STATUS_OK(env_.runtime->GetUnrankedTensorType(
          lite_rt_tensor, &unranked_tensor_type));
      tensor_type = unranked_tensor_type;
    }
    return std::make_unique<SimpleTensor>(index, name, type_id,
                                          std::move(tensor_type));
  }

  virtual Expected<std::unique_ptr<SimpleSignature>> ToSimpleSignature(
      LiteRtSignature lite_rt_signature) const {
    // key
    const char* key_cstr;
    LITERT_CHECK_STATUS_OK(
        env_.runtime->GetSignatureKey(lite_rt_signature, &key_cstr));
    // input names
    LiteRtParamIndex num_inputs;
    LITERT_CHECK_STATUS_OK(
        env_.runtime->GetNumSignatureInputs(lite_rt_signature, &num_inputs));
    std::vector<std::string> input_names;
    input_names.reserve(num_inputs);
    std::vector<std::unique_ptr<SimpleTensor>> input_tensors;
    input_tensors.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      const char* input_name;
      LITERT_CHECK_STATUS_OK(env_.runtime->GetSignatureInputName(
          lite_rt_signature, i, &input_name));
      input_names.push_back(input_name);

      LiteRtTensor lite_rt_tensor;
      LITERT_CHECK_STATUS_OK(env_.runtime->GetSignatureInputTensorByIndex(
          lite_rt_signature, i, &lite_rt_tensor));
      LITERT_ASSIGN_OR_RETURN(auto input_tensor,
                              ToSimpleTensor(lite_rt_tensor));
      input_tensors.push_back(std::move(input_tensor));
    }
    // output names
    LiteRtParamIndex num_outputs;
    LITERT_CHECK_STATUS_OK(
        env_.runtime->GetNumSignatureOutputs(lite_rt_signature, &num_outputs));
    std::vector<std::string> output_names;
    output_names.reserve(num_outputs);
    std::vector<std::unique_ptr<SimpleTensor>> output_tensors;
    output_tensors.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      const char* output_name;
      LITERT_CHECK_STATUS_OK(env_.runtime->GetSignatureOutputName(
          lite_rt_signature, i, &output_name));
      output_names.push_back(output_name);
      LiteRtTensor lite_rt_tensor;
      LITERT_CHECK_STATUS_OK(env_.runtime->GetSignatureOutputTensorByIndex(
          lite_rt_signature, i, &lite_rt_tensor));
      LITERT_ASSIGN_OR_RETURN(auto output_tensor,
                              ToSimpleTensor(lite_rt_tensor));
      output_tensors.push_back(std::move(output_tensor));
    }
    return std::make_unique<SimpleSignature>(
        key_cstr, input_names, std::move(input_tensors), output_names,
        std::move(output_tensors));
  }

  internal::EnvironmentHolder env_;
  mutable std::optional<std::vector<std::unique_ptr<SimpleSignature>>>
      signatures_;

  friend class CompiledModel;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_MODEL_H_
