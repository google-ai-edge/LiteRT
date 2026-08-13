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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_CONCRETE_OPTIONS_BASE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_CONCRETE_OPTIONS_BASE_H_

#include "litert/c/litert_common.h"

namespace litert {

/// @brief Base class for all LiteRT concrete C++ option wrappers.
///
/// All option classes configurable through `litert::Options::GetOptions<T>()`
/// must inherit from `ConcreteOptionsBase` and implement `GetOpaqueOptionsData`
/// to provide C-API serialization and opaque option integration.
///
/// Note: Relationship to `litert::OpaqueOptions`:
/// - This class, `ConcreteOptionsBase`, is the polymorphic C++ base class
///   implemented by concrete options (e.g. `GpuOptions`, `CpuOptions`)
///   to allow them to be stored and configured in `litert::Options`.
/// - `litert::OpaqueOptions` (in `litert/cc/litert_opaque_options.h`) is an
///   RAII wrapper for the C-API object `LiteRtOpaqueOptions` which is a
///   linked-list chain of opaque options.
///
/// - Interaction: During `litert::Options::Build()`, each
///   `ConcreteOptionsBase` instance is queried via `GetOpaqueOptionsData()` to
///   extract its identifier and payload, which are then used to create
///   `LiteRtOpaqueOptions` C handles and attach them to `LiteRtOptions`.
///   At the plugin and runtime dispatch level, backends traverse and inspect
///   these attached options using `litert::OpaqueOptions` (e.g., via
///   `FindOpaqueOptions`).
class ConcreteOptionsBase {
 public:
  virtual ~ConcreteOptionsBase() = default;

  /// @brief Retrieves the opaque options metadata (identifier, payload data
  /// pointer, and payload deleter) required to serialize and attach this
  /// option to LiteRtOptions.
  virtual LiteRtStatus GetOpaqueOptionsData(
      const char** identifier, void** payload,
      void (**payload_deleter)(void*)) const = 0;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_CONCRETE_OPTIONS_BASE_H_
