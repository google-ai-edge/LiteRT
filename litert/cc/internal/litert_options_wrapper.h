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

#ifndef ODML_LITERT_LITERT_CC_INTERNAL_LITERT_OPTIONS_WRAPPER_H_
#define ODML_LITERT_LITERT_CC_INTERNAL_LITERT_OPTIONS_WRAPPER_H_

#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_context_wrapper.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_opaque_options_wrapper.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::internal {

/// A wrapper for LiteRtOptions that provides convenient access to options and
/// associated opaque options. This class can optionally take ownership of the
/// underlying handle.
///
/// @note: This class is supposed to be used only by Dispatch API and Compiler
/// Plugin API.
class OptionsWrapper : public BaseHandle<LiteRtOptions> {
 public:
  /// Constructs an `OptionsWrapper` object from a C handle.
  /// @param ctx The context wrapper.
  /// @param options The C handle to the LiteRT options.
  /// @param own Indicates whether this object should take ownership of the
  /// provided handle.
  explicit OptionsWrapper(ContextWrapper ctx, LiteRtOptions options,
                          OwnHandle own = OwnHandle::kNo)
      : BaseHandle<LiteRtOptions>(
            options, [ctx](LiteRtOptions opt) { ctx.DestroyOptions(opt); },
            own),
        ctx_(ctx) {}

  /// Retrieves the opaque options associated with these options.
  /// @return An `OpaqueOptionsWrapper` if found, or an error.
  Expected<OpaqueOptionsWrapper> GetOpaqueOptions() const {
    LiteRtOpaqueOptions opaque_options = nullptr;
    LITERT_RETURN_IF_ERROR(ctx_.GetOpaqueOptions(Get(), &opaque_options));
    return OpaqueOptionsWrapper(ctx_, opaque_options);
  }

  /// Finds opaque options data by payload identifier directly from these
  /// options.
  /// @param payload_identifier The identifier for the opaque options.
  /// @return A pointer to the payload data if found, or an error.
  Expected<void*> FindOpaqueOptionsData(const char* payload_identifier) const {
    LITERT_ASSIGN_OR_RETURN(auto opaque_options, GetOpaqueOptions());
    return opaque_options.FindOpaqueOptions(payload_identifier);
  }

 private:
  ContextWrapper ctx_;
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_LITERT_OPTIONS_WRAPPER_H_
