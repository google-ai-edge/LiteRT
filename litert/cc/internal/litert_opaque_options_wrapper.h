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

#ifndef ODML_LITERT_LITERT_CC_INTERNAL_LITERT_OPAQUE_OPTIONS_WRAPPER_H_
#define ODML_LITERT_LITERT_CC_INTERNAL_LITERT_OPAQUE_OPTIONS_WRAPPER_H_

#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_context_wrapper.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::internal {

/// A wrapper for LiteRtOpaqueOptions that provides convenient access to
/// opaque options data. This class does not own the underlying handle.
///
/// @note: This class is supposed to be used only by Dispatch API and Compiler
/// Plugin API.
class OpaqueOptionsWrapper : public NonOwnedHandle<LiteRtOpaqueOptions> {
 public:
  /// Constructs an `OpaqueOptionsWrapper`.
  /// @param ctx The context wrapper.
  /// @param options The C handle to the LiteRT opaque options.
  OpaqueOptionsWrapper(ContextWrapper ctx, LiteRtOpaqueOptions options)
      : NonOwnedHandle<LiteRtOpaqueOptions>(options), ctx_(ctx) {}

  /// Finds opaque options data by payload identifier.
  /// @param payload_identifier The identifier for the opaque options.
  /// @return A pointer to the payload data if found, or an error.
  Expected<void*> FindOpaqueOptions(const char* payload_identifier) const {
    void* payload_data = nullptr;
    LITERT_RETURN_IF_ERROR(
        ctx_.FindOpaqueOptionsData(Get(), payload_identifier, &payload_data));
    return payload_data;
  }

 private:
  ContextWrapper ctx_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_LITERT_OPAQUE_OPTIONS_WRAPPER_H_
