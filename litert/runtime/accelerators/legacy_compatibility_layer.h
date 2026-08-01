// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_LEGACY_COMPATIBILITY_LAYER_H_
#define ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_LEGACY_COMPATIBILITY_LAYER_H_

#include <memory>

#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/litert_common.h"

namespace litert::internal {

typedef void (*WrapperDeleter)(void*);

// Abstract base class for adapting legacy accelerator definitions to the
// current LITERT_ACCELERATOR_DEF_CURRENT_VERSION.
class AcceleratorDefAdapter {
 public:
  virtual ~AcceleratorDefAdapter() = default;

  // Adapts the legacy accelerator definition to the current version structure.
  // Populates `current_def` with the adapted function pointers.
  // Returns wrapper data and deleter if heap allocation is required for thunks.
  virtual LiteRtStatus Adapt(const LiteRtAcceleratorDef* legacy_def,
                             LiteRtAcceleratorDef* current_def,
                             void** wrapper_data,
                             WrapperDeleter* wrapper_deleter) = 0;
};

// Factory for creating version-specific accelerator definition adapters.
class AcceleratorDefAdapterFactory {
 public:
  // Creates an adapter for the given legacy version.
  // Returns nullptr if the version is unsupported.
  static std::unique_ptr<AcceleratorDefAdapter> Create(int version);
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_RUNTIME_ACCELERATORS_LEGACY_COMPATIBILITY_LAYER_H_
