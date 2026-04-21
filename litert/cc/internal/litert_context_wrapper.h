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

#ifndef ODML_LITERT_LITERT_CC_INTERNAL_LITERT_CONTEXT_WRAPPER_H_
#define ODML_LITERT_LITERT_CC_INTERNAL_LITERT_CONTEXT_WRAPPER_H_

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"

namespace litert::internal {

/// A C API context wrapper for internal::OptionsWrapper and
/// internal::OpaqueOptionsWrapper classes.
///
/// @note: This class is supposed to be used only by Dispatch API and Compiler
/// Plugin API.
class ContextWrapper {
 public:
  explicit ContextWrapper(const LiteRtCompilerContext* ctx)
      : get_opaque_options_(ctx->get_opaque_options),
        find_opaque_options_data_(ctx->find_opaque_options_data),
        destroy_options_(ctx->destroy_options) {}

  explicit ContextWrapper(const LiteRtRuntimeContext* ctx)
      : get_opaque_options_(ctx->get_opaque_options),
        find_opaque_options_data_(ctx->find_opaque_options_data),
        destroy_options_(ctx->destroy_options) {}

  LiteRtStatus GetOpaqueOptions(LiteRtOptions options,
                                LiteRtOpaqueOptions* opaque_options) const {
    if (!get_opaque_options_) {
      return kLiteRtStatusErrorNotFound;
    }
    return get_opaque_options_(options, opaque_options);
  }

  LiteRtStatus FindOpaqueOptionsData(LiteRtOpaqueOptions options,
                                     const char* payload_identifier,
                                     void** payload_data) const {
    if (!find_opaque_options_data_) {
      return kLiteRtStatusErrorNotFound;
    }
    return find_opaque_options_data_(options, payload_identifier, payload_data);
  }

  void DestroyOptions(LiteRtOptions options) const {
    if (destroy_options_) {
      destroy_options_(options);
    }
  }

 private:
  LiteRtStatus (*get_opaque_options_)(LiteRtOptions, LiteRtOpaqueOptions*);
  LiteRtStatus (*find_opaque_options_data_)(LiteRtOpaqueOptions, const char*,
                                            void**);
  void (*destroy_options_)(LiteRtOptions);
};

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_LITERT_CONTEXT_WRAPPER_H_
