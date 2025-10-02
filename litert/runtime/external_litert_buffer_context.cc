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

#include "litert/runtime/external_litert_buffer_context.h"

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/tensor_buffer.h"
#include "litert/runtime/tensor_buffer_requirements.h"
#include "litert/runtime/tensor_identifier.h"
#include "litert/runtime/tfl_utils.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/common.h"

using litert::internal::TfLiteTensorIdentifier;

LiteRtStatus LiteRtExternalLiteRtBufferContextT::RegisterBufferRequirements(
    const TfLiteOpaqueTensor* tensor,
    LiteRtTensorBufferRequirementsPtr buffer_requirements) {
  TfLiteTensorIdentifier tensor_id = get_tensor_identifier_fn_(tensor);
  auto [iter, inserted] = buffer_requirements_.try_emplace(
      tensor_id, std::move(buffer_requirements));
  if (!inserted) {
    LiteRtTensorBufferRequirementsT* joined_requirements;
    LITERT_RETURN_IF_ERROR(LiteRtJoinTensorBufferRequirements(
        iter->second.get(), buffer_requirements.get(), &joined_requirements));
    LiteRtTensorBufferRequirementsPtr joined_requirements_ptr(
        joined_requirements);
    iter->second = std::move(joined_requirements_ptr);
  }
  return kLiteRtStatusOk;
}

litert::Expected<LiteRtTensorBufferRequirementsConst>
LiteRtExternalLiteRtBufferContextT::GetBufferRequirements(
    const TfLiteOpaqueTensor* tensor) {
  TfLiteTensorIdentifier tensor_id = get_tensor_identifier_fn_(tensor);
  auto it = buffer_requirements_.find(tensor_id);
  if (it == buffer_requirements_.end()) {
    return litert::Unexpected(
        kLiteRtStatusErrorNotFound,
        absl::StrFormat("Buffer requirements not found for tensor %p", tensor));
  }
  return it->second.get();
}

LiteRtStatus LiteRtExternalLiteRtBufferContextT::RegisterTensorBuffer(
    const TfLiteOpaqueTensor* tensor, LiteRtTensorBufferPtr tensor_buffer) {
  TfLiteTensorIdentifier tensor_id = get_tensor_identifier_fn_(tensor);
  tensor_buffers_[tensor_id] = std::move(tensor_buffer);
  return kLiteRtStatusOk;
}

litert::Expected<LiteRtTensorBufferPtr>
LiteRtExternalLiteRtBufferContextT::GetTensorBuffer(
    const TfLiteOpaqueTensor* tensor) {
  TfLiteTensorIdentifier tensor_id = get_tensor_identifier_fn_(tensor);
  auto it = tensor_buffers_.find(tensor_id);
  if (it == tensor_buffers_.end()) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Tensor buffer not found");
  }
  it->second->Duplicate();
  return LiteRtTensorBufferPtr(it->second.get());
}

litert::Expected<LiteRtTensorBufferPtr>
LiteRtExternalLiteRtBufferContextT::CreateBufferForTensor(
    const TfLiteOpaqueTensor* tensor) {
  LITERT_ASSIGN_OR_RETURN(
      LiteRtTensorBufferRequirementsConst tensor_buffer_requirements,
      GetBufferRequirements(tensor));

  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                          litert::internal::ConvertTensorType(tensor));

  const auto& supported_tensor_buffer_types =
      tensor_buffer_requirements->SupportedBufferTypes();
  if (supported_tensor_buffer_types.empty()) {
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "Insufficient number of supported tensor buffer types");
  }

  // For now we simply pick the first buffer type that's supported.
  LiteRtTensorBufferType tensor_buffer_type = supported_tensor_buffer_types[0];

  size_t tensor_buffer_size = tensor_buffer_requirements->BufferSize();
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(tensor_type);

  LiteRtTensorBufferT* litert_tensor_buffer;
  LITERT_RETURN_IF_ERROR(LiteRtCreateManagedTensorBuffer(
      env_, tensor_buffer_type, &litert_tensor_type, tensor_buffer_size,
      &litert_tensor_buffer));

  return LiteRtTensorBufferPtr(litert_tensor_buffer);
}
