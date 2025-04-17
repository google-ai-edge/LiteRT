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

#ifndef ODML_LITERT_LITERT_RUNTIME_ACCELERATOR_MODEL_COMPILATION_DATA_H_
#define ODML_LITERT_LITERT_RUNTIME_ACCELERATOR_MODEL_COMPILATION_DATA_H_

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::internal {

// Holds environment data that accelerators may need to prepare their
// delegates.
//
// These options are automatically added to the compilation options list
// during the creation of the compiled model.
struct ModelCompilationData {
  static constexpr auto kIdentifier = "environment-compilation-options";

  static Expected<OpaqueOptions> CreateOptions() {
    auto* payload_data = new ModelCompilationData;
    auto payload_destructor = [](void* payload_data) {
      delete reinterpret_cast<ModelCompilationData*>(payload_data);
    };
    return OpaqueOptions::Create(kIdentifier, payload_data, payload_destructor);
  }

  // Pointer to the start of the model file memory allocation.
  const char* allocation_base;
  // File descriptor of the model file memory allocation. If there is no such
  // file descriptor, this must be set to -1.
  int allocation_fd;

 private:
  ModelCompilationData() = default;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_ACCELERATOR_MODEL_COMPILATION_DATA_H_
