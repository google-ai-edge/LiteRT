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

#include <unordered_map>

#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"

namespace litert {
namespace internal {

struct CustomTensorBufferHandlers {
  ::CreateCustomTensorBuffer create_func;
  ::DestroyCustomTensorBuffer destroy_func;
  ::LockCustomTensorBuffer lock_func;
  ::UnlockCustomTensorBuffer unlock_func;
  // Optional function to import an existing buffer.
  // TODO(b/446717438): Merge this with the create function.
  ::ImportCustomTensorBuffer import_func;
};

class TensorBufferRegistry {
 public:
  explicit TensorBufferRegistry() = default;
  TensorBufferRegistry(const TensorBufferRegistry&) = delete;
  TensorBufferRegistry& operator=(const TensorBufferRegistry&) = delete;
  ~TensorBufferRegistry() = default;

  // Registers custom tensor buffer handlers for the given buffer type.
  litert::Expected<void> RegisterHandlers(
      LiteRtTensorBufferType buffer_type,
      const CustomTensorBufferHandlers& handlers);

  // Returns the custom tensor buffer handlers for the given buffer type.
  litert::Expected<CustomTensorBufferHandlers> GetCustomHandlers(
      const LiteRtTensorBufferType buffer_type);

 private:
  std::unordered_map<LiteRtTensorBufferType, CustomTensorBufferHandlers>
      handlers_;
};

}  // namespace internal
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_TENSOR_BUFFER_REGISTRY_H_
