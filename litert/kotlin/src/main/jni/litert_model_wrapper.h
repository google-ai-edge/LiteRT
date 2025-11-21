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

#ifndef LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_MODEL_WRAPPER_H_
#define LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_MODEL_WRAPPER_H_

#include <cstdint>
#include <utility>

#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_compiled_model.h"

namespace litert {
namespace jni {

// Wrapper to keep compiled model and its buffer alive together
struct CompiledModelWrapper {
  litert::CompiledModel compiled_model;
  litert::OwningBufferRef<uint8_t> buffer;  // For models loaded from assets

  explicit CompiledModelWrapper(litert::CompiledModel&& m)
      : compiled_model(std::move(m)) {}
  CompiledModelWrapper(litert::CompiledModel&& m,
                       litert::OwningBufferRef<uint8_t>&& b)
      : compiled_model(std::move(m)), buffer(std::move(b)) {}

  ~CompiledModelWrapper() = default;

  // Disable copy to prevent double-free
  CompiledModelWrapper(const CompiledModelWrapper&) = delete;
  CompiledModelWrapper& operator=(const CompiledModelWrapper&) = delete;
};

}  // namespace jni
}  // namespace litert

#endif  // LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_MODEL_WRAPPER_H_
