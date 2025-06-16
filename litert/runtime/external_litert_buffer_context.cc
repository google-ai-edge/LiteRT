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
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/runtime/tensor_buffer_requirements.h"
#include "litert/runtime/tfl_utils.h"
#include "tflite/c/c_api_opaque.h"
#include "tflite/c/c_api_types.h"

namespace litert {
namespace internal {

ExternalLiteRtBufferContext::~ExternalLiteRtBufferContext() {
  // Clean up owned tensor buffers
  for (auto& [tensor, buffer] : tensor_buffers_) {
    if (buffer != nullptr) {
      LiteRtDestroyTensorBuffer(buffer);
    }
  }
}

LiteRtStatus ExternalLiteRtBufferContext::RegisterBufferRequirements(
    const TfLiteOpaqueTensor* tensor,
    std::unique_ptr<LiteRtTensorBufferRequirementsT> buffer_requirements) {
  auto iter = buffer_requirements_.find(tensor);
  if (iter == buffer_requirements_.end()) {
    buffer_requirements_.insert(
        iter, std::make_pair(tensor, std::move(buffer_requirements)));
  } else {
    // Join the requirements
    LITERT_ASSIGN_OR_RETURN(auto joined_requirements,
                            internal::Join(*iter->second, *buffer_requirements));
    buffer_requirements_[tensor] = std::move(joined_requirements);
  }
  return kLiteRtStatusOk;
}

litert::Expected<const LiteRtTensorBufferRequirementsT*>
ExternalLiteRtBufferContext::GetBufferRequirements(
    const TfLiteOpaqueTensor* tensor) {
  auto it = buffer_requirements_.find(tensor);
  if (it == buffer_requirements_.end()) {
    return litert::Unexpected(
        kLiteRtStatusErrorNotFound,
        absl::StrFormat("Buffer requirements not found for tensor %p", tensor));
  }
  return it->second.get();
}

LiteRtStatus ExternalLiteRtBufferContext::RegisterTensorBuffer(
    const TfLiteOpaqueTensor* tensor, LiteRtTensorBuffer tensor_buffer) {
  // Duplicate the buffer to maintain ownership (increments ref count)
  LiteRtDuplicateTensorBuffer(tensor_buffer);
  
  // If we already have a buffer for this tensor, release the old one
  auto it = tensor_buffers_.find(tensor);
  if (it != tensor_buffers_.end() && it->second != nullptr) {
    LiteRtDestroyTensorBuffer(it->second);
  }
  
  tensor_buffers_[tensor] = tensor_buffer;
  return kLiteRtStatusOk;
}

litert::Expected<LiteRtTensorBuffer> ExternalLiteRtBufferContext::GetTensorBuffer(
    const TfLiteOpaqueTensor* tensor) {
  auto it = tensor_buffers_.find(tensor);
  if (it == tensor_buffers_.end()) {
    return litert::Unexpected(kLiteRtStatusErrorNotFound,
                              "Tensor buffer not found");
  }

  // Duplicate the buffer (increments ref count)
  // Caller is responsible for calling LiteRtDestroyTensorBuffer
  LiteRtDuplicateTensorBuffer(it->second);
  return it->second;
}

litert::Expected<LiteRtTensorBuffer>
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

  const auto& supported_types = (*tensor_buffer_requirements)->SupportedBufferTypes();
  if (supported_types.empty()) {
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "Insufficient number of supported tensor buffer types");
  }

  // For now we simply pick the first buffer type that's supported.
  LiteRtTensorBufferType tensor_buffer_type = supported_types[0];

  size_t tensor_buffer_size = (*tensor_buffer_requirements)->BufferSize();
  auto litert_tensor_type = static_cast<LiteRtRankedTensorType>(*tensor_type);

  LiteRtTensorBuffer litert_tensor_buffer;
  if (auto status = LiteRtCreateManagedTensorBuffer(
          env_, tensor_buffer_type, &litert_tensor_type, tensor_buffer_size,
          &litert_tensor_buffer);
      status != kLiteRtStatusOk) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create managed tensor buffer");
  }

  return litert_tensor_buffer;
}

}  // namespace internal
}  // namespace litert
