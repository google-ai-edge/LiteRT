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
#include <utility>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/runtime/tfl_utils.h"
#include "tflite/c/c_api_opaque.h"  // from @org_tensorflow
#include "tflite/c/c_api_types.h"  // from @org_tensorflow

namespace litert {
namespace internal {

LiteRtStatus ExternalLiteRtBufferContext::RegisterBufferRequirements(
    const TfLiteOpaqueTensor* tensor,
    TensorBufferRequirements&& buffer_requirements) {
  assert(buffer_requirements.IsOwned());
  auto iter = buffer_requirements_.find(tensor);
  if (iter == buffer_requirements_.end()) {
    buffer_requirements_.insert(
        iter, std::make_pair(tensor, std::move(buffer_requirements)));
  } else {
    auto joined_tensor = Join(iter->second, buffer_requirements);
    if (!joined_tensor) {
      return joined_tensor.Error().Status();
    }
    buffer_requirements_[tensor] = std::move(*joined_tensor);
  }
  return kLiteRtStatusOk;
}

litert::Expected<const TensorBufferRequirements*>
ExternalLiteRtBufferContext::GetBufferRequirements(
    const TfLiteOpaqueTensor* tensor) {
  auto it = buffer_requirements_.find(tensor);
  if (it == buffer_requirements_.end()) {
    return litert::Unexpected(
        kLiteRtStatusErrorNotFound,
        absl::StrFormat("Buffer requirements not found for tensor %p", tensor));
  }
  return &(it->second);
}

LiteRtStatus ExternalLiteRtBufferContext::RegisterTensorBuffer(
    const TfLiteOpaqueTensor* tensor, TensorBuffer&& tensor_buffer) {
  tensor_buffers_[tensor] = std::move(tensor_buffer);
  return kLiteRtStatusOk;
}

litert::Expected<TensorBuffer> ExternalLiteRtBufferContext::GetTensorBuffer(
    const TfLiteOpaqueTensor* tensor) {
  auto it = tensor_buffers_.find(tensor);
  if (it == tensor_buffers_.end()) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Tensor buffer not found");
  }

  auto duplicate_tensor_buffer = it->second.Duplicate();
  if (!duplicate_tensor_buffer) {
    return litert::Unexpected(duplicate_tensor_buffer.Error());
  }
  return std::move(duplicate_tensor_buffer.Value());
}

litert::Expected<TensorBuffer>
ExternalLiteRtBufferContext::CreateBufferForTensor(
    const TfLiteOpaqueTensor* tensor) {
  auto tensor_buffer_requirements = GetBufferRequirements(tensor);
  if (!tensor_buffer_requirements) {
    return litert::Unexpected(tensor_buffer_requirements.Error());
  }

  auto tensor_type = litert::internal::ConvertTensorType(tensor);
  if (!tensor_type) {
    return litert::Unexpected(tensor_type.Error());
  }

  auto supported_tensor_buffer_types =
      (*tensor_buffer_requirements)->SupportedTypes();
  if (!supported_tensor_buffer_types) {
    return litert::Unexpected(supported_tensor_buffer_types.Error());
  }
  if (supported_tensor_buffer_types->empty()) {
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "Insufficient number of supported tensor buffer types");
  }

  // For now we simply pick the first buffer type that's supported.
  LiteRtTensorBufferType tensor_buffer_type =
      (*supported_tensor_buffer_types)[0];

  auto tensor_buffer_size = (*tensor_buffer_requirements)->BufferSize();
  if (!tensor_buffer_size) {
    return litert::Unexpected(tensor_buffer_size.Error());
  }
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(*tensor_type);

  LiteRtTensorBuffer litert_tensor_buffer;
  if (auto status = LiteRtCreateManagedTensorBuffer(
          tensor_buffer_type, &litert_tensor_type, *tensor_buffer_size,
          &litert_tensor_buffer);
      status != kLiteRtStatusOk) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create managed tensor buffer");
  }

  return TensorBuffer(litert_tensor_buffer, OwnHandle::kYes);
}

}  // namespace internal
}  // namespace litert
