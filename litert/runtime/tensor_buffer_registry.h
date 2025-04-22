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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_TENSOR_BUFFER_REGISTRY_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_TENSOR_BUFFER_REGISTRY_H_

#include <cstddef>
#include <unordered_map>

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer_types.h"

extern "C" {

typedef LiteRtStatus (*CustomTensorBufferCreate)(
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type, size_t bytes, void** hw_memory);

typedef LiteRtStatus (*CustomTensorBufferUpload)(
    void* hw_memory, size_t bytes, const void* ptr,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type);

typedef LiteRtStatus (*CustomTensorBufferDownload)(
    void* hw_memory, size_t bytes, void* ptr,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferType buffer_type);
}

namespace litert {
namespace internal {

class TensorBufferRegistry {
 public:
  TensorBufferRegistry(const TensorBufferRegistry&) = delete;
  TensorBufferRegistry& operator=(const TensorBufferRegistry&) = delete;
  ~TensorBufferRegistry() = default;

  static TensorBufferRegistry& GetInstance() {
    if (instance_ == nullptr) {
      instance_ = new TensorBufferRegistry();
    }
    return *instance_;
  }

  // Registers custom tensor buffer accessors for the given buffer type.
  LiteRtStatus RegisterAccessors(LiteRtTensorBufferType buffer_type,
                                 CustomTensorBufferCreate create_func,
                                 CustomTensorBufferUpload upload_func,
                                 CustomTensorBufferDownload download_func);

  // Create a custom tensor buffer and return the created hardware memory handle
  // in `hw_memory`.
  LiteRtStatus CreateCustomTensorBuffer(
      const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferType buffer_type, size_t bytes, void** hw_memory);

  // Upload the data from the CPU memory to the custom tensor buffer in
  // `hw_memory`.
  LiteRtStatus UploadCustomTensorBuffer(
      void* hw_memory, size_t bytes, const void* ptr,
      const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferType buffer_type);

  // Download the data from the custom tensor buffer in `hw_memory` to the CPU
  // memory.
  LiteRtStatus DownloadCustomTensorBuffer(
      void* hw_memory, size_t bytes, void* ptr,
      const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferType buffer_type);

 private:
  explicit TensorBufferRegistry() = default;

  static TensorBufferRegistry* instance_;

  std::unordered_map<LiteRtTensorBufferType, CustomTensorBufferCreate>
      create_funcs_;
  std::unordered_map<LiteRtTensorBufferType, CustomTensorBufferUpload>
      upload_funcs_;
  std::unordered_map<LiteRtTensorBufferType, CustomTensorBufferDownload>
      download_funcs_;
};

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_TENSOR_BUFFER_REGISTRY_H_
