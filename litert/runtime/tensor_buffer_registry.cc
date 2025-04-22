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

#include "litert/runtime/tensor_buffer_registry.h"

#include <cstddef>

#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace litert {
namespace internal {

TensorBufferRegistry* TensorBufferRegistry::instance_ = nullptr;

LiteRtStatus TensorBufferRegistry::RegisterAccessors(
    LiteRtTensorBufferType buffer_type, CustomTensorBufferCreate create_func,
    CustomTensorBufferUpload upload_func,
    CustomTensorBufferDownload download_func) {
  create_funcs_[buffer_type] = create_func;
  upload_funcs_[buffer_type] = upload_func;
  download_funcs_[buffer_type] = download_func;
  return kLiteRtStatusOk;
}

LiteRtStatus TensorBufferRegistry::CreateCustomTensorBuffer(
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, void** hw_memory) {
  auto it = create_funcs_.find(buffer_type);
  if (it == create_funcs_.end()) {
    LITERT_LOG(LITERT_ERROR,
               "No custom tensor buffer create function registered for "
               "buffer type %d",
               buffer_type);
    return kLiteRtStatusErrorInvalidArgument;
  }
  return it->second(tensor_type, buffer_type, bytes, hw_memory);
}

LiteRtStatus TensorBufferRegistry::UploadCustomTensorBuffer(
    void* hw_memory, size_t bytes, const void* ptr,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type) {
  auto it = upload_funcs_.find(buffer_type);
  if (it == upload_funcs_.end()) {
    LITERT_LOG(LITERT_ERROR,
               "No custom tensor buffer upload function registered for "
               "buffer type %d",
               buffer_type);
    return kLiteRtStatusErrorInvalidArgument;
  }
  return it->second(hw_memory, bytes, ptr, tensor_type, buffer_type);
}

LiteRtStatus TensorBufferRegistry::DownloadCustomTensorBuffer(
    void* hw_memory, size_t bytes, void* ptr,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type) {
  auto it = download_funcs_.find(buffer_type);
  if (it == download_funcs_.end()) {
    LITERT_LOG(LITERT_ERROR,
               "No custom tensor buffer download function registered for "
               "buffer type %d",
               buffer_type);
    return kLiteRtStatusErrorInvalidArgument;
  }
  return it->second(hw_memory, bytes, ptr, tensor_type, buffer_type);
}

}  // namespace internal
}  // namespace litert
