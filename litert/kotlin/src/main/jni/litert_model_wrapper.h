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

#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_buffer_ref.h"

namespace litert {
namespace jni {

// Wrapper to keep model and its buffer alive together
struct ModelWrapper {
  LiteRtModel model = nullptr;
  litert::OwningBufferRef<uint8_t> buffer;  // For models loaded from assets

  explicit ModelWrapper(LiteRtModel m) : model(m) {}
  ModelWrapper(LiteRtModel m, litert::OwningBufferRef<uint8_t>&& b)
      : model(m), buffer(std::move(b)) {}

  ~ModelWrapper() {
    if (model) {
      LiteRtDestroyModel(model);
      model = nullptr;
    }
  }

  // Disable copy to prevent double-free
  ModelWrapper(const ModelWrapper&) = delete;
  ModelWrapper& operator=(const ModelWrapper&) = delete;

  // Allow move
  ModelWrapper(ModelWrapper&& other) noexcept
      : model(other.model), buffer(std::move(other.buffer)) {
    other.model = nullptr;
  }

  ModelWrapper& operator=(ModelWrapper&& other) noexcept {
    if (this != &other) {
      if (model) {
        LiteRtDestroyModel(model);
      }
      model = other.model;
      buffer = std::move(other.buffer);
      other.model = nullptr;
    }
    return *this;
  }
};

}  // namespace jni
}  // namespace litert

#endif  // LITERT_KOTLIN_SRC_MAIN_JNI_LITERT_MODEL_WRAPPER_H_
