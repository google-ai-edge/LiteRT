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

#include "litert/cc/litert_model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"

namespace litert {

namespace {

absl::string_view FetchTensorName(LiteRtTensor tensor) {
  const char* name;
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

absl::string_view FetchSignatureKey(LiteRtSignature signature) {
  const char* key;
  internal::AssertOk(LiteRtGetSignatureKey, signature, &key);
  return key;
}

std::vector<absl::string_view> FetchSignatureInputNames(
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

std::vector<absl::string_view> FetchSignatureOutputNames(
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

std::vector<std::unique_ptr<SimpleTensor>> FetchSignatureInputTensors(
    LiteRtSignature signature) {
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
        FetchTensorType(tensor, FetchTensorTypeId(tensor))));
  }
  return input_tensors;
}

std::vector<std::unique_ptr<SimpleTensor>> FetchSignatureOutputTensors(
    LiteRtSignature signature) {
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
        FetchTensorType(tensor, FetchTensorTypeId(tensor))));
  }
  return output_tensors;
}

}  // namespace

Expected<std::vector<SimpleSignature>> Model::GetSignatures() const {
  size_t num_signatures = GetNumSignatures();
  std::vector<SimpleSignature> signatures;
  signatures.reserve(num_signatures);
  for (int i = 0; i < num_signatures; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto signature, GetSignature(i));
    signatures.push_back(std::move(signature));
  }
  return std::move(signatures);
}

Expected<std::vector<absl::string_view>> Model::GetSignatureKeys() const {
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

Expected<std::vector<absl::string_view>> Model::GetSignatureInputNames(
    size_t signature_index) const {
  LiteRtSignature lite_rt_signature;
  internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                     &lite_rt_signature);
  return FetchSignatureInputNames(lite_rt_signature);
}

Expected<std::vector<absl::string_view>> Model::GetSignatureOutputNames(
    size_t signature_index) const {
  LiteRtSignature lite_rt_signature;
  internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                     &lite_rt_signature);
  return FetchSignatureOutputNames(lite_rt_signature);
}

Expected<SimpleSignature> Model::GetSignature(size_t signature_index) const {
  LiteRtSignature lite_rt_signature;
  internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                     &lite_rt_signature);
  return SimpleSignature(FetchSignatureKey(lite_rt_signature),
                         FetchSignatureInputNames(lite_rt_signature),
                         FetchSignatureInputTensors(lite_rt_signature),
                         FetchSignatureOutputNames(lite_rt_signature),
                         FetchSignatureOutputTensors(lite_rt_signature));
}

Expected<size_t> Model::GetSignatureIndex(
    absl::string_view signature_key) const {
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
  return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
}

Expected<SimpleSignature> Model::FindSignature(
    absl::string_view signature_key) const {
  LITERT_ASSIGN_OR_RETURN(auto signature_index,
                          GetSignatureIndex(signature_key));
  return GetSignature(signature_index);
}

}  // namespace litert
