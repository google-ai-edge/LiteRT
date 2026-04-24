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
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/internal/litert_consts.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model_types.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "tflite/converter/allocation.h"
// copybara:uncomment_begin(google_only)
// #include "litert/core/model/model_load.h"
// copybara:uncomment_end

/// @file
/// @brief Defines C++ wrappers for the LiteRT model, signature, and tensor
/// types.

// copybara:comment_begin(google_only)
namespace tflite {
class Allocation;
}  // namespace tflite
// copybara:comment_end

namespace litert {

namespace {

absl::string_view FetchTensorName(LiteRtTensor tensor) {
  const char *name;
  internal::AssertOk(LiteRtGetTensorName, tensor, &name);
  return name;
}

std::uint32_t FetchTensorIndex(LiteRtTensor tensor) {
  std::uint32_t index;
  internal::AssertOk(LiteRtGetTensorIndex, tensor, &index);
  return index;
}

LiteRtTensorTypeId FetchTensorTypeId(LiteRtTensor tensor) {
  LiteRtTensorTypeId type_id;
  internal::AssertOk(LiteRtGetTensorTypeId, tensor, &type_id);
  return type_id;
}

std::variant<LiteRtUnrankedTensorType, litert::RankedTensorType>
FetchTensorType(LiteRtTensor tensor, LiteRtTensorTypeId type_id) {
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

LiteRtQuantizationTypeId FetchTensorQuantizationTypeId(LiteRtTensor tensor) {
  LiteRtQuantizationTypeId quantization_type_id;
  internal::AssertOk(LiteRtGetQuantizationTypeId, tensor,
                     &quantization_type_id);
  return quantization_type_id;
}

LiteRtQuantizationPerTensor
FetchTensorQuantizationPerTensor(LiteRtTensor tensor) {
  if (FetchTensorQuantizationTypeId(tensor) != kLiteRtQuantizationPerTensor) {
    return {};
  }
  LiteRtQuantizationPerTensor per_tensor_quantization;
  internal::AssertOk(LiteRtGetPerTensorQuantization, tensor,
                     &per_tensor_quantization);
  return per_tensor_quantization;
}

LiteRtQuantizationPerChannel
FetchTensorQuantizationPerChannel(LiteRtTensor tensor) {
  if (FetchTensorQuantizationTypeId(tensor) != kLiteRtQuantizationPerChannel) {
    return {};
  }
  LiteRtQuantizationPerChannel per_channel_quantization;
  internal::AssertOk(LiteRtGetPerChannelQuantization, tensor,
                     &per_channel_quantization);
  return per_channel_quantization;
}

absl::string_view FetchSignatureKey(LiteRtSignature signature) {
  const char *key;
  internal::AssertOk(LiteRtGetSignatureKey, signature, &key);
  return key;
}

std::vector<absl::string_view>
FetchSignatureInputNames(LiteRtSignature signature) {
  LiteRtParamIndex num_inputs;
  internal::AssertOk(LiteRtGetNumSignatureInputs, signature, &num_inputs);
  std::vector<absl::string_view> input_names;
  input_names.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const char *name;
    internal::AssertOk(LiteRtGetSignatureInputName, signature, i, &name);
    input_names.push_back(name);
  }
  return input_names;
}

std::vector<absl::string_view>
FetchSignatureOutputNames(LiteRtSignature signature) {
  LiteRtParamIndex num_outputs;
  internal::AssertOk(LiteRtGetNumSignatureOutputs, signature, &num_outputs);
  std::vector<absl::string_view> output_names;
  output_names.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    const char *name;
    internal::AssertOk(LiteRtGetSignatureOutputName, signature, i, &name);
    output_names.push_back(name);
  }
  return output_names;
}

std::vector<std::unique_ptr<SimpleTensor>>
FetchSignatureInputTensors(LiteRtSignature signature) {
  LiteRtParamIndex num_inputs;
  internal::AssertOk(LiteRtGetNumSignatureInputs, signature, &num_inputs);
  std::vector<std::unique_ptr<SimpleTensor>> input_tensors;
  input_tensors.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    LiteRtTensor tensor;
    internal::AssertOk(LiteRtGetSignatureInputTensorByIndex, signature, i,
                       &tensor);
    input_tensors.push_back(std::make_unique<SimpleTensor>(
        FetchTensorIndex(tensor), FetchTensorName(tensor),
        FetchTensorTypeId(tensor),
        FetchTensorType(tensor, FetchTensorTypeId(tensor)),
        FetchTensorQuantizationTypeId(tensor),
        FetchTensorQuantizationPerTensor(tensor),
        FetchTensorQuantizationPerChannel(tensor)));
  }
  return input_tensors;
}

std::vector<std::unique_ptr<SimpleTensor>>
FetchSignatureOutputTensors(LiteRtSignature signature) {
  LiteRtParamIndex num_outputs;
  internal::AssertOk(LiteRtGetNumSignatureOutputs, signature, &num_outputs);
  std::vector<std::unique_ptr<SimpleTensor>> output_tensors;
  output_tensors.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    LiteRtTensor tensor;
    internal::AssertOk(LiteRtGetSignatureOutputTensorByIndex, signature, i,
                       &tensor);
    output_tensors.push_back(std::make_unique<SimpleTensor>(
        FetchTensorIndex(tensor), FetchTensorName(tensor),
        FetchTensorTypeId(tensor),
        FetchTensorType(tensor, FetchTensorTypeId(tensor)),
        FetchTensorQuantizationTypeId(tensor),
        FetchTensorQuantizationPerTensor(tensor),
        FetchTensorQuantizationPerChannel(tensor)));
  }
  return output_tensors;
}

} // namespace

/// @brief A C++ wrapper for `LiteRtModel`, representing a LiteRT model.
///
/// \internal
class Model : public internal::BaseHandle<LiteRtModel> {
public:
  Model() = default;

  static Model CreateFromOwnedHandle(LiteRtModel model) {
    return Model(model, OwnHandle::kYes);
  }

  static Model CreateFromNonOwnedHandle(LiteRtModel model) {
    return Model(model, OwnHandle::kNo);
  }

  static Expected<Model> CreateFromFile(const std::string &filename) {
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
  static Expected<Model> CreateFromBuffer(BufferRef<uint8_t> buffer) {
    LiteRtModel model;
    if (auto status =
            LiteRtCreateModelFromBuffer(buffer.Data(), buffer.Size(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status), "Failed to load model from buffer");
    }
    return CreateFromOwnedHandle(model);
  }

#if !defined(LITERT_DYNAMIC_RUNTIME)
  /// @internal
  /// @brief Creates a model from an owned TFLite allocation.
  /// @note This is an internal experimetal API which is not available through
  /// libLiteRt.so. It's not part of the official LiteRT public C++ API.
  static Expected<Model>
  CreateFromAllocation(std::unique_ptr<tflite::Allocation> allocation) {
    LiteRtModel model;
    if (auto status =
            LiteRtCreateModelFromAllocation(std::move(allocation), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(ToStatus(status),
                        "Failed to load model from allocation");
    }
    return CreateFromOwnedHandle(model);
  }
#endif // !defined(LITERT_DYNAMIC_RUNTIME)

  Expected<absl::Span<const uint8_t>>
  Metadata(const std::string &metadata_key) const {
    const void *buffer;
    size_t buffer_size;
    if (LiteRtGetModelMetadata(Get(), metadata_key.data(), &buffer,
                               &buffer_size) != kLiteRtStatusOk) {
      return Unexpected(Status::kErrorNotFound, "Metadata key not found");
    }
    return absl::MakeSpan(static_cast<const uint8_t *>(buffer), buffer_size);
  }

#ifdef LITERT_NO_ABSL
  Expected<std::span<const uint8_t>>
  Metadata(std::string_view metadata_key) const {
    return internal::ToStdSpan(Metadata(std::string(metadata_key)));
  }
#endif // LITERT_NO_ABSL

  size_t GetNumSignatures() const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    return num_signatures;
  }

  /// @brief Returns the list of signatures defined in the model.
  Expected<std::vector<SimpleSignature>> GetSignatures() const {
    size_t num_signatures = GetNumSignatures();
    std::vector<SimpleSignature> signatures;
    signatures.reserve(num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LITERT_ASSIGN_OR_RETURN(auto signature, GetSignature(i));
      signatures.push_back(std::move(signature));
    }
    return std::move(signatures);
  }

  /// @brief Returns the list of signature key names defined in the signature.
#ifdef LITERT_NO_ABSL
  Expected<std::vector<std::string_view>> GetSignatureKeys() const {
    return internal::ToStdStringViews(GetSignatureKeysImpl());
  }
#else
  Expected<std::vector<absl::string_view>> GetSignatureKeys() const {
    return GetSignatureKeysImpl();
  }
#endif

  /// @brief Returns the list of input names defined in the signature.
#ifdef LITERT_NO_ABSL
  Expected<std::vector<std::string_view>>
  GetSignatureInputNames(size_t signature_index) const {
    return internal::ToStdStringViews(
        GetSignatureInputNamesImpl(signature_index));
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<std::string_view>> GetSignatureInputNames() const {
    return GetSignatureInputNames(/*signature_index=*/0);
  }
#else
  Expected<std::vector<absl::string_view>>
  GetSignatureInputNames(size_t signature_index) const {
    return GetSignatureInputNamesImpl(signature_index);
  }

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureInputNames() const {
    return GetSignatureInputNames(/*signature_index=*/0);
  }
#endif

  /// @brief Returns the list of input names defined in the signature.
  Expected<std::vector<absl::string_view>>
  GetSignatureInputNames(absl::string_view signature_key) const {
    auto signature = FindSignature(signature_key);
    if (!signature) {
      return Unexpected(Status::kErrorNotFound, "Signature not found");
    }
    return signature->InputNames();
  }

#ifdef LITERT_NO_ABSL
  Expected<std::vector<std::string_view>>
  GetSignatureInputNames(std::string_view signature_key) const {
    return internal::ToStdStringViews(
        GetSignatureInputNames(internal::ToAbslStringView(signature_key)));
  }
#endif // LITERT_NO_ABSL

  /// @brief Returns the list of output names defined in the signature.
#ifdef LITERT_NO_ABSL
  Expected<std::vector<std::string_view>>
  GetSignatureOutputNames(size_t signature_index) const {
    return internal::ToStdStringViews(
        GetSignatureOutputNamesImpl(signature_index));
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<std::string_view>> GetSignatureOutputNames() const {
    return GetSignatureOutputNames(/*signature_index=*/0);
  }
#else
  Expected<std::vector<absl::string_view>>
  GetSignatureOutputNames(size_t signature_index) const {
    return GetSignatureOutputNamesImpl(signature_index);
  }

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>> GetSignatureOutputNames() const {
    return GetSignatureOutputNames(/*signature_index=*/0);
  }
#endif

  /// @brief Returns the list of output names defined in the signature.
  Expected<std::vector<absl::string_view>>
  GetSignatureOutputNames(absl::string_view signature_key) const {
    auto signature = FindSignature(signature_key);
    if (!signature) {
      return Unexpected(Status::kErrorNotFound, "Signature not found");
    }
    return signature->OutputNames();
  }

#ifdef LITERT_NO_ABSL
  Expected<std::vector<std::string_view>>
  GetSignatureOutputNames(std::string_view signature_key) const {
    return internal::ToStdStringViews(
        GetSignatureOutputNames(internal::ToAbslStringView(signature_key)));
  }
#endif // LITERT_NO_ABSL

  /// @brief Returns the signature at the given index.
  Expected<SimpleSignature> GetSignature(size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    return SimpleSignature(FetchSignatureKey(lite_rt_signature),
                           FetchSignatureInputNames(lite_rt_signature),
                           FetchSignatureInputTensors(lite_rt_signature),
                           FetchSignatureOutputNames(lite_rt_signature),
                           FetchSignatureOutputTensors(lite_rt_signature));
  }

  /// @brief Returns the signature index for the given signature key.
  ///
  /// Returns 0 if the signature key is empty.
  Expected<size_t> GetSignatureIndex(absl::string_view signature_key) const {
    if (signature_key.empty()) {
      return 0;
    }
    size_t num_signatures = GetNumSignatures();
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      auto key = FetchSignatureKey(lite_rt_signature);
      if (key == signature_key) {
        return i;
      }
    }
    return Unexpected(Status::kErrorNotFound, "Signature not found");
  }

#ifdef LITERT_NO_ABSL
  Expected<size_t> GetSignatureIndex(std::string_view signature_key) const {
    return GetSignatureIndex(internal::ToAbslStringView(signature_key));
  }
#endif // LITERT_NO_ABSL

  /// @brief Returns the `SimpleSignature` object for the given signature key.
  ///
  /// Returns the default signature if the signature key is empty.
  Expected<SimpleSignature>
  FindSignature(absl::string_view signature_key) const {
    LITERT_ASSIGN_OR_RETURN(auto signature_index,
                            GetSignatureIndex(signature_key));
    return GetSignature(signature_index);
  }

#ifdef LITERT_NO_ABSL
  Expected<SimpleSignature>
  FindSignature(std::string_view signature_key) const {
    return FindSignature(internal::ToAbslStringView(signature_key));
  }
#endif // LITERT_NO_ABSL

#ifdef LITERT_NO_ABSL
  static std::string_view DefaultSignatureKey() { return kDefaultSignatureKey; }
#else
  static absl::string_view DefaultSignatureKey() {
    return kDefaultSignatureKey;
  }
#endif

  /// @brief Returns the tensor type for the n-th input tensor.
  Expected<RankedTensorType> GetInputTensorType(size_t signature_index,
                                                size_t input_index) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature &signature,
                            GetSignature(signature_index));
    return signature.InputTensorType(input_index);
  }

  /// @brief Returns the tensor type for the given input tensor name.
  Expected<RankedTensorType>
  GetInputTensorType(size_t signature_index,
                     absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature &signature,
                            GetSignature(signature_index));
    return signature.InputTensorType(input_name);
  }

  /// @brief Returns the tensor type for the given input tensor name.
  Expected<RankedTensorType>
  GetInputTensorType(absl::string_view signature_key,
                     absl::string_view input_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature &signature,
                            FindSignature(signature_key));
    return signature.InputTensorType(input_name);
  }

  /// @brief Gets the input tensor type of the default signature for a given
  /// input name.
  Expected<RankedTensorType>
  GetInputTensorType(absl::string_view input_name) const {
    return GetInputTensorType(/*signature_index=*/0, input_name);
  }

#ifdef LITERT_NO_ABSL
  Expected<RankedTensorType>
  GetInputTensorType(size_t signature_index,
                     std::string_view input_name) const {
    return GetInputTensorType(signature_index,
                              internal::ToAbslStringView(input_name));
  }

  Expected<RankedTensorType>
  GetInputTensorType(std::string_view signature_key,
                     std::string_view input_name) const {
    return GetInputTensorType(internal::ToAbslStringView(signature_key),
                              internal::ToAbslStringView(input_name));
  }

  Expected<RankedTensorType>
  GetInputTensorType(std::string_view input_name) const {
    return GetInputTensorType(/*signature_index=*/0, input_name);
  }
#endif // LITERT_NO_ABSL

  /// @brief Returns the tensor type for the n-th output tensor.
  Expected<RankedTensorType> GetOutputTensorType(size_t signature_index,
                                                 size_t output_index) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature &signature,
                            GetSignature(signature_index));
    return signature.OutputTensorType(output_index);
  }

  /// @brief Returns the tensor type for the given output tensor name.
  Expected<RankedTensorType>
  GetOutputTensorType(size_t signature_index,
                      absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature &signature,
                            GetSignature(signature_index));
    return signature.OutputTensorType(output_name);
  }

  /// @brief Returns the tensor type for the given output tensor name.
  Expected<RankedTensorType>
  GetOutputTensorType(absl::string_view signature_key,
                      absl::string_view output_name) const {
    LITERT_ASSIGN_OR_RETURN(const SimpleSignature &signature,
                            FindSignature(signature_key));
    return signature.OutputTensorType(output_name);
  }

  /// @brief Gets the output tensor type of the default signature for a given
  /// output name.
  Expected<RankedTensorType>
  GetOutputTensorType(absl::string_view output_name) const {
    return GetOutputTensorType(/*signature_index=*/0, output_name);
  }

#ifdef LITERT_NO_ABSL
  Expected<RankedTensorType>
  GetOutputTensorType(size_t signature_index,
                      std::string_view output_name) const {
    return GetOutputTensorType(signature_index,
                               internal::ToAbslStringView(output_name));
  }

  Expected<RankedTensorType>
  GetOutputTensorType(std::string_view signature_key,
                      std::string_view output_name) const {
    return GetOutputTensorType(internal::ToAbslStringView(signature_key),
                               internal::ToAbslStringView(output_name));
  }

  Expected<RankedTensorType>
  GetOutputTensorType(std::string_view output_name) const {
    return GetOutputTensorType(/*signature_index=*/0, output_name);
  }
#endif // LITERT_NO_ABSL

protected:
  /// @param owned Indicates if the created `TensorBuffer` object should take
  /// ownership of the provided `tensor_buffer` handle.
  Model(LiteRtModel model, OwnHandle owned)
      : internal::BaseHandle<LiteRtModel>(model, LiteRtDestroyModel, owned) {}

private:
  Expected<std::vector<absl::string_view>> GetSignatureKeysImpl() const {
    size_t num_signatures = GetNumSignatures();
    std::vector<absl::string_view> signature_keys;
    signature_keys.reserve(num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      signature_keys.push_back(FetchSignatureKey(lite_rt_signature));
    }
    return signature_keys;
  }
  Expected<std::vector<absl::string_view>>
  GetSignatureInputNamesImpl(size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    return FetchSignatureInputNames(lite_rt_signature);
  }
  Expected<std::vector<absl::string_view>>
  GetSignatureOutputNamesImpl(size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    return FetchSignatureOutputNames(lite_rt_signature);
  }
};

} // namespace litert

#endif // ODML_LITERT_LITERT_CC_LITERT_MODEL_H_
